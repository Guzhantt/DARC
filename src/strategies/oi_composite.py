"""Composite strategy: Price Action + Open Interest + On-chain Signals.

This strategy combines:
1. TREND (price-based): EMA cross for direction
2. OI MOMENTUM: Confirms trend via OI acceleration
3. WHALE SCORE: Detects smart-money accumulation/distribution
4. FUNDING RATE: Contrarian signal at extremes
5. LONG/SHORT RATIO: Crowd positioning (contrarian)

Signal logic:
- Each factor produces a score in [-1, +1] (positive=bullish, negative=bearish)
- Weighted sum → composite score
- Enter long if composite > threshold, short if < -threshold
- Exit to flat if composite flips through zero or stop is hit

This is designed to be used with the enhanced DataFrame that includes
columns: close, open, high, low, volume, oi, oi_momentum, whale_score,
funding_signal, flow_proxy, ls_ratio, taker_ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .ma_cross import _atr


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _price_trend_score(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """EMA cross → trend score in [-1, 1]."""
    close = df["close"]
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    diff = fast_ema - slow_ema
    # Normalize by ATR to make it comparable across assets
    atr = _atr(df, 14).replace(0, np.nan)
    score = (diff / atr).clip(-2, 2) / 2
    return score.fillna(0).rename("trend_score")


def _oi_divergence_score(df: pd.DataFrame) -> pd.Series:
    """OI divergence: OI going up + price going up = strong trend.
    OI going up + price going down = potential reversal (but we use it as accumulation).
    Returns score in [-1, 1].
    """
    if "oi" not in df.columns:
        return pd.Series(0.0, index=df.index, name="oi_div_score")

    oi_pct = df["oi"].pct_change(5)
    price_pct = df["close"].pct_change(5)

    # Concordance: both up = bullish, both down = bearish
    concordance = np.sign(oi_pct) * np.sign(price_pct)
    # Magnitude weighted by OI change
    score = concordance * oi_pct.abs().clip(upper=0.1) * 10  # scale up
    return score.clip(-1, 1).fillna(0).rename("oi_div_score")


def _ls_contrarian_score(df: pd.DataFrame) -> pd.Series:
    """Long/Short ratio contrarian: very high LS ratio = crowded long = bearish.
    Uses z-score of LS ratio.
    """
    if "ls_ratio" not in df.columns:
        return pd.Series(0.0, index=df.index, name="ls_score")

    ls = df["ls_ratio"]
    # Z-score over rolling window
    mean = ls.rolling(50, min_periods=10).mean()
    std = ls.rolling(50, min_periods=10).std().replace(0, np.nan)
    z = ((ls - mean) / std).clip(-2, 2)
    # CONTRARIAN: high LS (crowded long) → bearish signal (negative)
    score = -z / 2
    return score.fillna(0).rename("ls_score")


def _taker_momentum_score(df: pd.DataFrame) -> pd.Series:
    """Taker buy/sell ratio momentum.
    Rising taker ratio (aggressive buyers) = bullish.
    """
    if "taker_ratio" not in df.columns:
        return pd.Series(0.0, index=df.index, name="taker_score")

    tr = df["taker_ratio"]
    # Use change in taker ratio as momentum signal
    tr_change = tr.rolling(5, min_periods=1).mean() - tr.rolling(20, min_periods=5).mean()
    mean = tr_change.rolling(50, min_periods=10).mean()
    std = tr_change.rolling(50, min_periods=10).std().replace(0, np.nan)
    z = ((tr_change - mean) / std).clip(-2, 2) / 2
    return z.fillna(0).rename("taker_score")


class OICompositeStrategy(Strategy):
    """Composite strategy using OI + on-chain + price action.
    
    Weights (configurable):
        trend:    0.25  (price action EMA cross)
        oi_div:   0.20  (OI-price divergence/concordance)
        oi_mom:   0.15  (OI momentum if pre-computed)
        whale:    0.15  (whale accumulation score if pre-computed)
        funding:  0.10  (funding rate contrarian)
        ls_ratio: 0.10  (long/short contrarian)
        taker:    0.05  (taker momentum)
    
    Entry threshold: |composite| > entry_threshold
    Exit: composite crosses zero OR stop hit
    """

    name = "oi_composite"

    def __init__(
        self,
        # EMA periods for trend
        fast_ema: int = 10,
        slow_ema: int = 30,
        # ATR stop
        atr_period: int = 14,
        atr_mult: float = 2.5,
        # Composite thresholds
        entry_threshold: float = 0.20,
        exit_threshold: float = 0.0,
        # Factor weights (must sum to ~1)
        w_trend: float = 0.25,
        w_oi_div: float = 0.20,
        w_oi_mom: float = 0.15,
        w_whale: float = 0.15,
        w_funding: float = 0.10,
        w_ls: float = 0.10,
        w_taker: float = 0.05,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.weights = {
            "trend": w_trend,
            "oi_div": w_oi_div,
            "oi_mom": w_oi_mom,
            "whale": w_whale,
            "funding": w_funding,
            "ls": w_ls,
            "taker": w_taker,
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Compute each factor score
        trend = _price_trend_score(df, self.fast_ema, self.slow_ema)
        oi_div = _oi_divergence_score(df)
        taker_s = _taker_momentum_score(df)
        ls_s = _ls_contrarian_score(df)

        # Pre-computed columns from onchain_data module (may or may not exist)
        oi_mom = df["oi_momentum"] if "oi_momentum" in df.columns else pd.Series(0.0, index=df.index)
        whale = df["whale_score"] if "whale_score" in df.columns else pd.Series(0.0, index=df.index)
        funding = df["funding_signal"] if "funding_signal" in df.columns else pd.Series(0.0, index=df.index)

        # Weighted composite
        composite = (
            self.weights["trend"] * trend
            + self.weights["oi_div"] * oi_div
            + self.weights["oi_mom"] * oi_mom
            + self.weights["whale"] * whale
            + self.weights["funding"] * funding
            + self.weights["ls"] * ls_s
            + self.weights["taker"] * taker_s
        )

        # Shift by 1 so signal at bar t is based on bar t-1 data (no look-ahead)
        composite = composite.shift(1)
        atr = _atr(df, self.atr_period).shift(1)
        close_s = df["close"].shift(1)

        # Generate sides with hysteresis (stay in position until exit_threshold crossed)
        sides = np.full(len(df), "flat", dtype=object)
        stops = np.full(len(df), np.nan)
        position = "flat"

        for i in range(len(df)):
            c = composite.iloc[i]
            if np.isnan(c):
                sides[i] = position
                continue

            if position == "flat":
                if c > self.entry_threshold:
                    position = "long"
                elif c < -self.entry_threshold:
                    position = "short"
            elif position == "long":
                if c < self.exit_threshold:
                    position = "flat"
                    # Check if should reverse
                    if c < -self.entry_threshold:
                        position = "short"
            elif position == "short":
                if c > -self.exit_threshold:
                    position = "flat"
                    if c > self.entry_threshold:
                        position = "long"

            sides[i] = position
            if position == "long" and not np.isnan(close_s.iloc[i]) and not np.isnan(atr.iloc[i]):
                stops[i] = close_s.iloc[i] - self.atr_mult * atr.iloc[i]
            elif position == "short" and not np.isnan(close_s.iloc[i]) and not np.isnan(atr.iloc[i]):
                stops[i] = close_s.iloc[i] + self.atr_mult * atr.iloc[i]

        return pd.DataFrame({"side": sides, "stop": stops}, index=df.index)
