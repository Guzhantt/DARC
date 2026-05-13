"""On-chain metrics fetcher.

Data sources:
1. Exchange net flow estimation — derived from large transfers (whale movements).
   We use Binance's own large-transfer tracking via their public API where available,
   otherwise we estimate from OI changes + price to infer accumulation/distribution.

2. Whale accumulation score — derived from:
   - OI increasing while price flat/down = smart money accumulating
   - OI decreasing while price flat/up = smart money distributing
   - Taker buy dominant + OI rising = aggressive whale buying

3. Funding rate extremes — from Binance funding rate API.
   Very positive funding = crowded long (bearish contrarian signal)
   Very negative funding = crowded short (bullish contrarian signal)

NOTE: True on-chain data (UTXO movements, exchange wallet balance) requires premium
APIs like Glassnode, CryptoQuant, or Nansen. This module provides FREE alternatives
derived from publicly available Binance data that capture similar dynamics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.oi_data import (
    fetch_all_derivatives_data,
    fetch_funding_rate_history,
)


def compute_whale_score(df: pd.DataFrame) -> pd.Series:
    """Compute a whale accumulation/distribution score from OI + price + taker data.
    
    Logic:
    - OI rising + price flat/down → whales accumulating (score > 0)
    - OI falling + price flat/up → whales distributing (score < 0)
    - Taker ratio > 1 amplifies bullish, < 1 amplifies bearish
    
    Returns: Series with values in [-1, 1] range (z-score clipped).
    """
    if "oi" not in df.columns or "close" not in df.columns:
        return pd.Series(0.0, index=df.index, name="whale_score")

    oi_pct = df["oi"].pct_change(5)  # 5-bar OI change
    price_pct = df["close"].pct_change(5)

    # Divergence: OI going up while price goes down = accumulation
    # OI going down while price goes up = distribution
    divergence = oi_pct - price_pct

    # Amplify by taker ratio if available
    if "taker_ratio" in df.columns:
        taker_bias = (df["taker_ratio"] - 1.0).clip(-0.5, 0.5)
        divergence = divergence + taker_bias * 0.3

    # Normalize to [-1, 1]
    mean = divergence.rolling(50, min_periods=10).mean()
    std = divergence.rolling(50, min_periods=10).std().replace(0, np.nan)
    z = ((divergence - mean) / std).clip(-2, 2) / 2
    return z.fillna(0).rename("whale_score")


def compute_funding_signal(funding_df: pd.DataFrame, extreme_threshold: float = 0.001) -> pd.Series:
    """Compute contrarian signal from funding rate.
    
    - funding > +threshold → crowded long → bearish signal (-1)
    - funding < -threshold → crowded short → bullish signal (+1)
    - In between → neutral (0)
    
    Returns: Series with values in [-1, 1].
    """
    if funding_df.empty or "funding_rate" not in funding_df.columns:
        return pd.Series(dtype=float, name="funding_signal")

    fr = funding_df["funding_rate"]
    # Use rolling average (last 3 funding periods = 24h)
    fr_avg = fr.rolling(3, min_periods=1).mean()

    signal = np.where(
        fr_avg > extreme_threshold, -fr_avg / extreme_threshold,  # bearish (short signal)
        np.where(fr_avg < -extreme_threshold, -fr_avg / extreme_threshold, 0.0)  # bullish (long signal)
    )
    signal = np.clip(signal, -1, 1)
    return pd.Series(signal, index=funding_df.index, name="funding_signal")


def compute_exchange_flow_proxy(df: pd.DataFrame) -> pd.Series:
    """Estimate exchange net-flow from OI and volume patterns.
    
    Rationale: When large OI increases happen with relatively low volume,
    it suggests limit-order accumulation (like exchange inflow for selling 
    or OTC for buying). This is a rough proxy, not real on-chain data.
    
    - High OI growth + low volume growth = accumulation (positive score)
    - Low OI growth + high volume growth = distribution/retail FOMO (negative score)
    """
    if "oi" not in df.columns:
        return pd.Series(0.0, index=df.index, name="flow_proxy")

    oi_growth = df["oi"].pct_change(3)

    if "buy_vol" in df.columns and "sell_vol" in df.columns:
        total_vol = df["buy_vol"] + df["sell_vol"]
        vol_growth = total_vol.pct_change(3)
    elif "volume" in df.columns:
        vol_growth = df["volume"].pct_change(3)
    else:
        return pd.Series(0.0, index=df.index, name="flow_proxy")

    # Divergence: OI growing faster than volume = smart money
    flow = oi_growth - vol_growth
    mean = flow.rolling(30, min_periods=5).mean()
    std = flow.rolling(30, min_periods=5).std().replace(0, np.nan)
    z = ((flow - mean) / std).clip(-2, 2) / 2
    return z.fillna(0).rename("flow_proxy")


def compute_oi_momentum(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """OI momentum: fast MA of OI change vs slow MA.
    
    Positive = OI acceleration (new positions entering, trend strengthening)
    Negative = OI deceleration (positions closing, trend weakening)
    """
    if "oi" not in df.columns:
        return pd.Series(0.0, index=df.index, name="oi_momentum")

    oi_change = df["oi"].pct_change()
    fast_ma = oi_change.rolling(fast, min_periods=1).mean()
    slow_ma = oi_change.rolling(slow, min_periods=1).mean()
    momentum = fast_ma - slow_ma
    # Normalize
    std = momentum.rolling(50, min_periods=10).std().replace(0, np.nan)
    z = (momentum / std).clip(-2, 2) / 2
    return z.fillna(0).rename("oi_momentum")


def build_onchain_features(
    symbol_raw: str,
    period: str = "1h",
    limit: int = 500,
    ohlcv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Main entry: fetch derivatives data and compute all on-chain-like features.
    
    Args:
        symbol_raw: Symbol without slashes, e.g. 'BTCUSDT'
        period: Data period (1h, 4h, etc.)
        limit: Number of data points
        ohlcv_df: Optional OHLCV DataFrame to merge with (must have 'ts' and 'close' columns)
    
    Returns:
        DataFrame with columns: ts, oi, oi_value, ls_ratio, taker_ratio,
        whale_score, funding_signal, flow_proxy, oi_momentum
    """
    # Fetch derivatives market data
    deriv = fetch_all_derivatives_data(symbol_raw, period, limit)
    if deriv.empty:
        return pd.DataFrame()

    # Merge with OHLCV if provided (to get close price for whale score)
    if ohlcv_df is not None and "close" in ohlcv_df.columns:
        if "ts" in ohlcv_df.columns:
            ohlcv_aligned = ohlcv_df[["ts", "close", "volume"]].copy() if "volume" in ohlcv_df.columns else ohlcv_df[["ts", "close"]].copy()
            deriv = pd.merge_asof(
                deriv.sort_values("ts"),
                ohlcv_aligned.sort_values("ts"),
                on="ts",
                direction="backward",
            )

    # Compute features
    deriv["whale_score"] = compute_whale_score(deriv).values
    deriv["flow_proxy"] = compute_exchange_flow_proxy(deriv).values
    deriv["oi_momentum"] = compute_oi_momentum(deriv).values

    # Funding rate (separate timeline, resample to match)
    try:
        funding = fetch_funding_rate_history(symbol_raw, limit=100)
        if not funding.empty:
            funding["funding_signal"] = compute_funding_signal(funding).values
            deriv = pd.merge_asof(
                deriv.sort_values("ts"),
                funding[["ts", "funding_rate", "funding_signal"]].sort_values("ts"),
                on="ts",
                direction="backward",
            )
    except Exception:
        deriv["funding_rate"] = 0.0
        deriv["funding_signal"] = 0.0

    if "funding_signal" not in deriv.columns:
        deriv["funding_signal"] = 0.0
    if "funding_rate" not in deriv.columns:
        deriv["funding_rate"] = 0.0

    return deriv.sort_values("ts").reset_index(drop=True)
