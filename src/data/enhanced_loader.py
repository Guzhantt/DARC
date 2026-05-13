"""Enhanced data loader: OHLCV + OI + derivatives data merged into one DataFrame.

Used by the oi_composite strategy for backtesting and live execution.
Merges OHLCV candles with Open Interest, Long/Short ratio, Taker ratio,
and on-chain-like computed features (whale_score, funding_signal, etc.).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import TF_MS, load_ohlcv
from src.data.onchain_data import (
    compute_exchange_flow_proxy,
    compute_funding_signal,
    compute_oi_momentum,
    compute_whale_score,
)
from src.data.oi_data import (
    fetch_all_derivatives_data,
    fetch_funding_rate_history,
)
from src.exchange.binance_client import BinanceFutures


# Map ccxt-style symbols to raw Binance symbols
def _to_raw_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT:USDT' or 'BTC/USDT' → 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":USD", "")


# Map timeframes to valid OI periods
_TF_TO_OI_PERIOD = {
    "1m": "5m",
    "3m": "5m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
}


def load_enhanced_ohlcv(
    client: BinanceFutures,
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV + derivatives data merged into a single DataFrame.

    Returns a DataFrame with columns:
        ts, open, high, low, close, volume,     (from OHLCV)
        oi, oi_value,                            (from OI history)
        ls_ratio, long_ratio, short_ratio,       (from global LS ratio)
        top_ls_ratio,                            (from top trader LS ratio)
        taker_ratio, buy_vol, sell_vol,          (from taker data)
        whale_score, flow_proxy, oi_momentum,    (computed on-chain features)
        funding_rate, funding_signal             (from funding rate)

    NOTE: Binance OI history is limited to ~30 days. For longer backtests,
    only the last 30 days will have OI data; earlier rows will have NaN
    for OI-derived columns. The strategy handles this gracefully.
    """
    # 1. Load OHLCV (can go back months/years)
    ohlcv = load_ohlcv(client, symbol, timeframe, days, data_dir, use_cache=use_cache)
    if ohlcv.empty:
        return ohlcv

    # 2. Determine raw symbol and OI period
    raw_sym = _to_raw_symbol(symbol)
    oi_period = _TF_TO_OI_PERIOD.get(timeframe, "1h")

    # 3. Fetch derivatives data (max ~30 days from Binance)
    try:
        deriv = fetch_all_derivatives_data(raw_sym, oi_period, limit=500)
    except Exception:
        deriv = pd.DataFrame()

    # 4. Fetch funding rate
    try:
        funding = fetch_funding_rate_history(raw_sym, limit=500)
    except Exception:
        funding = pd.DataFrame()

    # 5. Merge derivatives data into OHLCV by nearest timestamp
    if not deriv.empty and "ts" in deriv.columns:
        # Ensure ts columns are timezone-aware
        if ohlcv["ts"].dt.tz is None:
            ohlcv["ts"] = ohlcv["ts"].dt.tz_localize("UTC")
        if deriv["ts"].dt.tz is None:
            deriv["ts"] = deriv["ts"].dt.tz_localize("UTC")

        # Select only the columns we need from deriv
        deriv_cols = ["ts"]
        for col in ["oi", "oi_value", "ls_ratio", "long_ratio", "short_ratio",
                    "top_ls_ratio", "top_long_ratio", "top_short_ratio",
                    "taker_ratio", "buy_vol", "sell_vol"]:
            if col in deriv.columns:
                deriv_cols.append(col)

        merged = pd.merge_asof(
            ohlcv.sort_values("ts"),
            deriv[deriv_cols].sort_values("ts"),
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta(TF_MS.get(timeframe, 3600000) * 2, "ms"),
        )
    else:
        merged = ohlcv.copy()

    # 6. Merge funding rate
    if not funding.empty and "ts" in funding.columns:
        if funding["ts"].dt.tz is None:
            funding["ts"] = funding["ts"].dt.tz_localize("UTC")
        funding["funding_signal"] = compute_funding_signal(funding).values
        merged = pd.merge_asof(
            merged.sort_values("ts"),
            funding[["ts", "funding_rate", "funding_signal"]].sort_values("ts"),
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("24h"),  # funding comes every 8h
        )

    # 7. Compute derived on-chain features (these work with NaN gracefully)
    merged["whale_score"] = compute_whale_score(merged).values
    merged["flow_proxy"] = compute_exchange_flow_proxy(merged).values
    merged["oi_momentum"] = compute_oi_momentum(merged).values

    # 8. Fill missing columns with defaults for strategies
    for col in ["oi", "oi_value", "ls_ratio", "taker_ratio", "buy_vol", "sell_vol",
                "funding_rate", "funding_signal", "whale_score", "flow_proxy", "oi_momentum"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged.sort_values("ts").reset_index(drop=True)


def load_enhanced_ohlcv_synthetic(
    ohlcv: pd.DataFrame,
    oi: pd.Series | None = None,
    ls_ratio: pd.Series | None = None,
    taker_ratio: pd.Series | None = None,
    funding_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """For backtesting with synthetic/preloaded data.
    
    Adds OI and derivatives columns to an existing OHLCV DataFrame.
    Used for unit testing and offline analysis.
    """
    df = ohlcv.copy()
    if oi is not None:
        df["oi"] = oi.values if hasattr(oi, "values") else oi
    if ls_ratio is not None:
        df["ls_ratio"] = ls_ratio.values if hasattr(ls_ratio, "values") else ls_ratio
    if taker_ratio is not None:
        df["taker_ratio"] = taker_ratio.values if hasattr(taker_ratio, "values") else taker_ratio

    # Compute derived features
    df["whale_score"] = compute_whale_score(df).values
    df["flow_proxy"] = compute_exchange_flow_proxy(df).values
    df["oi_momentum"] = compute_oi_momentum(df).values

    if funding_rate is not None:
        df["funding_rate"] = funding_rate.values if hasattr(funding_rate, "values") else funding_rate
        fr_df = pd.DataFrame({"ts": df["ts"], "funding_rate": df["funding_rate"]})
        df["funding_signal"] = compute_funding_signal(fr_df).values
    else:
        df["funding_rate"] = 0.0
        df["funding_signal"] = 0.0

    return df
