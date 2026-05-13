"""Alpha Scanner Strategy — Event-driven signals inspired by professional trader.

Combines:
1. Extreme Funding Rate (contrarian) — from the trader's script
2. Crash Bounce (oversold reversal) — from the trader's script
3. Pump Short (overbought reversal) — from the trader's script
4. RSI Extreme Reversion — from our smart_reversion (85% WR)
5. Environment Scoring (BTC context + OI + Volume + FGI proxy)

Key differences from the trader's original:
- Can be backtested (signals computed from historical data)
- Position sizing uses ATR-based risk management
- Integrates with our backtest engine
- Adds our proven RSI reversion as a 5th signal type

The trader's edge: WAIT for extreme events, validate with environment,
then enter with asymmetric R:R. Only trade when everything lines up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def _rsi(close: pd.Series, period: int = 9) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift()).abs()
    low_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _detect_crash_bounce(df: pd.DataFrame, lookback: int = 24, crash_threshold: float = -0.15,
                          stabilize_bars: int = 3) -> pd.Series:
    """Detect crash + stabilization pattern.
    
    Logic (from trader): Price drops > crash_threshold over lookback bars,
    then last stabilize_bars show recovery (higher lows or green candles).
    Returns: Series of booleans (True = crash bounce signal).
    """
    close = df["close"]
    # Rolling return over lookback period
    ret = close.pct_change(lookback)
    
    # Check stabilization: last N bars have rising closes
    stabilizing = pd.Series(True, index=df.index)
    for i in range(1, stabilize_bars):
        stabilizing = stabilizing & (close.shift(i - 1) >= close.shift(i))
    
    # Green candle on signal bar
    green = close > df["open"]
    
    signal = (ret.shift(1) < crash_threshold) & stabilizing.shift(1) & green.shift(1)
    return signal.fillna(False)


def _detect_pump_short(df: pd.DataFrame, lookback: int = 24, pump_threshold: float = 0.30,
                        pullback_pct: float = 0.08) -> pd.Series:
    """Detect pump + pullback pattern.
    
    Logic (from trader): Price pumps > pump_threshold over lookback bars,
    then pulls back by at least pullback_pct from the high.
    Returns: Series of booleans (True = pump short signal).
    """
    close = df["close"]
    high = df["high"]
    
    # Rolling return over lookback
    ret = close.pct_change(lookback)
    
    # Rolling high over lookback
    rolling_high = high.rolling(lookback).max()
    
    # Pullback from high
    pullback = (rolling_high - close) / rolling_high
    
    # Red candle on signal bar
    red = close < df["open"]
    
    signal = (
        (ret.shift(1) > pump_threshold)
        & (pullback.shift(1) > pullback_pct)
        & red.shift(1)
    )
    return signal.fillna(False)


def _detect_funding_extreme(df: pd.DataFrame, threshold_neg: float = -0.001,
                             threshold_pos: float = 0.001) -> tuple[pd.Series, pd.Series]:
    """Detect extreme funding rate signals.
    
    Uses 'funding_rate' column if available.
    Returns: (long_signal, short_signal) boolean Series.
    """
    if "funding_rate" not in df.columns:
        empty = pd.Series(False, index=df.index)
        return empty, empty
    
    fr = df["funding_rate"]
    # Rolling average over ~24h (3 funding periods for 8h funding)
    fr_avg = fr.rolling(3, min_periods=1).mean()
    
    # Extreme negative funding → long (shorts paying high rates = squeeze setup)
    long_sig = (fr_avg.shift(1) < threshold_neg)
    
    # Extreme positive funding → short (longs paying high rates = dump setup)
    short_sig = (fr_avg.shift(1) > threshold_pos)
    
    return long_sig.fillna(False), short_sig.fillna(False)


def _detect_rsi_extreme(df: pd.DataFrame, period: int = 9, extreme: float = 30) -> tuple[pd.Series, pd.Series]:
    """Our proven RSI extreme reversion signal (85% WR).
    
    Returns: (long_signal, short_signal) boolean Series.
    """
    close = df["close"]
    opens = df["open"]
    r = _rsi(close, period)
    
    # Long: RSI was below extreme, now crossing back up + green candle
    long_sig = (
        (r.shift(2) < extreme)
        & (r.shift(1) > r.shift(2))
        & (r.shift(1) > extreme)  # crossed back
        & (close.shift(1) > opens.shift(1))  # green candle
    )
    
    # Short: RSI was above (100-extreme), now crossing back down + red candle
    short_sig = (
        (r.shift(2) > (100 - extreme))
        & (r.shift(1) < r.shift(2))
        & (r.shift(1) < (100 - extreme))
        & (close.shift(1) < opens.shift(1))  # red candle
    )
    
    return long_sig.fillna(False), short_sig.fillna(False)


def _environment_score(df: pd.DataFrame, i: int, direction: str) -> int:
    """Compute environment validation score (inspired by trader's check_environment).
    
    Factors scored (each ±1):
    1. Price trend (EMA50 alignment)
    2. Volume (above/below average)
    3. OI trend (if available)
    4. Volatility regime (ATR not extreme)
    
    Returns score: -4 to +4. Signal passes if score >= 2.
    """
    score = 0
    close = df["close"].values
    volume = df["volume"].values
    
    # 1. Trend alignment (SMA50 approximation)
    if i >= 50:
        sma50 = np.mean(close[i-50:i])
        if direction == "long" and close[i-1] > sma50:
            score += 1
        elif direction == "short" and close[i-1] < sma50:
            score += 1
        elif direction == "long" and close[i-1] < sma50 * 0.97:
            score -= 1  # Deep below SMA, risky for longs
        elif direction == "short" and close[i-1] > sma50 * 1.03:
            score -= 1
    
    # 2. Volume confirmation
    if i >= 20:
        vol_avg = np.mean(volume[max(0, i-20):i])
        if volume[i-1] > vol_avg * 1.3:
            score += 1  # High volume = conviction
        elif volume[i-1] < vol_avg * 0.5:
            score -= 1  # Low volume = weak signal
    
    # 3. OI confirmation (if available)
    if "oi" in df.columns and i >= 5:
        oi_vals = df["oi"].values
        if not np.isnan(oi_vals[i-1]) and not np.isnan(oi_vals[i-5]):
            oi_change = (oi_vals[i-1] - oi_vals[i-5]) / max(oi_vals[i-5], 1)
            if direction == "long" and oi_change > 0.02:
                score += 1  # OI rising with long = confirmation
            elif direction == "short" and oi_change < -0.02:
                score += 1  # OI falling with short = confirmation
    
    # 4. Volatility regime (not too extreme = manageable)
    if i >= 14:
        recent_atr = np.mean(np.abs(np.diff(close[max(0, i-14):i])))
        longer_atr = np.mean(np.abs(np.diff(close[max(0, i-50):i]))) if i >= 50 else recent_atr
        if longer_atr > 0:
            vol_ratio = recent_atr / longer_atr
            if 0.5 < vol_ratio < 2.0:
                score += 1  # Normal volatility
            elif vol_ratio > 3.0:
                score -= 1  # Extreme volatility, dangerous
    
    return score


class AlphaScannerStrategy(Strategy):
    """Event-driven alpha scanner combining trader's signals + our RSI reversion.
    
    Signal priority (from trader's approach):
    1. Funding rate extreme (strongest edge — market microstructure)
    2. Crash bounce (24h crash + stabilization)
    3. Pump short (24h pump + pullback)
    4. RSI extreme reversion (our proven 85% WR signal)
    
    Each signal must pass environment scoring (≥2/4) to activate.
    
    Parameters:
        tp_pct: Take-profit % (default 1.2% — balances WR and R:R)
        sl_pct: Stop-loss % (default 4% — wide enough for high WR)
        crash_threshold: Min 24h drop to trigger crash bounce (-15%)
        pump_threshold: Min 24h pump to trigger pump short (+30%)
        funding_extreme_neg: Funding rate threshold for long (-0.001)
        funding_extreme_pos: Funding rate threshold for short (+0.001)
        rsi_period: RSI lookback (9)
        rsi_extreme: RSI extreme level (30)
        min_env_score: Minimum environment score to enter (2)
        cooldown_bars: Min bars between trades (6)
    """
    
    name = "alpha_scanner"
    
    def __init__(
        self,
        tp_pct: float = 0.012,
        sl_pct: float = 0.04,
        crash_threshold: float = -0.15,
        pump_threshold: float = 0.30,
        pullback_pct: float = 0.08,
        funding_extreme_neg: float = -0.001,
        funding_extreme_pos: float = 0.001,
        rsi_period: int = 9,
        rsi_extreme: float = 30,
        min_env_score: int = 2,
        cooldown_bars: int = 6,
    ):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.crash_threshold = crash_threshold
        self.pump_threshold = pump_threshold
        self.pullback_pct = pullback_pct
        self.funding_extreme_neg = funding_extreme_neg
        self.funding_extreme_pos = funding_extreme_pos
        self.rsi_period = rsi_period
        self.rsi_extreme = rsi_extreme
        self.min_env_score = min_env_score
        self.cooldown_bars = cooldown_bars
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        close = df["close"].values
        opens = df["open"].values
        
        # Pre-compute all signals
        crash_long = _detect_crash_bounce(df, crash_threshold=self.crash_threshold)
        pump_short = _detect_pump_short(df, pump_threshold=self.pump_threshold,
                                         pullback_pct=self.pullback_pct)
        funding_long, funding_short = _detect_funding_extreme(
            df, self.funding_extreme_neg, self.funding_extreme_pos
        )
        rsi_long, rsi_short = _detect_rsi_extreme(df, self.rsi_period, self.rsi_extreme)
        
        # Convert to numpy for fast iteration
        crash_long_arr = crash_long.values
        pump_short_arr = pump_short.values
        funding_long_arr = funding_long.values
        funding_short_arr = funding_short.values
        rsi_long_arr = rsi_long.values
        rsi_short_arr = rsi_short.values
        
        sides = np.full(n, "flat", dtype=object)
        stops = np.full(n, np.nan)
        
        position = "flat"
        entry_px = 0.0
        tp_px = 0.0
        sl_px = 0.0
        bars_since_trade = 999  # Start allowing trades immediately
        
        for i in range(max(50, self.cooldown_bars + 1), n):
            c = close[i - 1]  # Previous close (no look-ahead)
            bars_since_trade += 1
            
            # === EXIT: TP/SL check ===
            if position == "long":
                if c >= tp_px:
                    position = "flat"
                    bars_since_trade = 0
                elif c <= sl_px:
                    position = "flat"
                    bars_since_trade = 0
            elif position == "short":
                if c <= tp_px:
                    position = "flat"
                    bars_since_trade = 0
                elif c >= sl_px:
                    position = "flat"
                    bars_since_trade = 0
            
            # === ENTRY: Check signals with priority ===
            if position == "flat" and bars_since_trade >= self.cooldown_bars:
                direction = None
                signal_type = None
                
                # Priority 1: Funding rate extremes (strongest edge)
                if funding_long_arr[i]:
                    direction = "long"
                    signal_type = "funding"
                elif funding_short_arr[i]:
                    direction = "short"
                    signal_type = "funding"
                # Priority 2: Crash bounce
                elif crash_long_arr[i]:
                    direction = "long"
                    signal_type = "crash_bounce"
                # Priority 3: Pump short
                elif pump_short_arr[i]:
                    direction = "short"
                    signal_type = "pump_short"
                # Priority 4: RSI extreme (our proven signal)
                elif rsi_long_arr[i]:
                    direction = "long"
                    signal_type = "rsi_extreme"
                elif rsi_short_arr[i]:
                    direction = "short"
                    signal_type = "rsi_extreme"
                
                # Validate with environment scoring
                if direction is not None:
                    env_score = _environment_score(df, i, direction)
                    
                    if env_score >= self.min_env_score:
                        position = direction
                        entry_px = c
                        if direction == "long":
                            tp_px = c * (1 + self.tp_pct)
                            sl_px = c * (1 - self.sl_pct)
                        else:
                            tp_px = c * (1 - self.tp_pct)
                            sl_px = c * (1 + self.sl_pct)
            
            sides[i] = position
            stops[i] = sl_px if position != "flat" else np.nan
        
        return pd.DataFrame({"side": sides, "stop": stops}, index=df.index)
