"""Open Interest and derivatives market data from Binance Futures API.

Endpoints used (no API key needed for market data):
- /futures/data/openInterestHist       — OI history
- /futures/data/globalLongShortAccountRatio — global LS ratio
- /futures/data/topLongShortAccountRatio    — top trader LS ratio
- /futures/data/takerlongshortRatio         — taker buy/sell volume ratio
- /fapi/v1/fundingRate                      — funding rate history
- /fapi/v1/openInterest                     — current snapshot OI
"""
from __future__ import annotations

import time
from typing import Literal

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
DATA_BASE = "https://fapi.binance.com"

Period = Literal["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]


def _get(url: str, params: dict) -> list[dict]:
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------- Open Interest History ----------

def fetch_oi_history(
    symbol: str,
    period: Period = "1h",
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Fetch OI history. Symbol should be e.g. 'BTCUSDT' (no slash).
    Returns DataFrame with columns: ts, oi (sumOpenInterest), oi_value (sumOpenInterestValue).
    Max 30 days of data available.
    """
    params: dict = {"symbol": symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    data = _get(f"{DATA_BASE}/futures/data/openInterestHist", params)
    if not data:
        return pd.DataFrame(columns=["ts", "oi", "oi_value"])
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    df["oi_value"] = df["sumOpenInterestValue"].astype(float)
    return df[["ts", "oi", "oi_value"]].sort_values("ts").reset_index(drop=True)


def fetch_current_oi(symbol: str) -> float:
    """Current open interest in contracts."""
    data = _get(f"{BASE}/fapi/v1/openInterest", {"symbol": symbol})
    return float(data.get("openInterest", 0))


# ---------- Long/Short Ratio ----------

def fetch_global_long_short_ratio(
    symbol: str,
    period: Period = "1h",
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Global accounts long/short ratio.
    Returns: ts, long_ratio, short_ratio, ls_ratio (long/short).
    """
    params: dict = {"symbol": symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    data = _get(f"{DATA_BASE}/futures/data/globalLongShortAccountRatio", params)
    if not data:
        return pd.DataFrame(columns=["ts", "long_ratio", "short_ratio", "ls_ratio"])
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_ratio"] = df["longAccount"].astype(float)
    df["short_ratio"] = df["shortAccount"].astype(float)
    df["ls_ratio"] = df["longShortRatio"].astype(float)
    return df[["ts", "long_ratio", "short_ratio", "ls_ratio"]].sort_values("ts").reset_index(drop=True)


def fetch_top_trader_long_short_ratio(
    symbol: str,
    period: Period = "1h",
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Top trader (top 20% by margin balance) long/short account ratio."""
    params: dict = {"symbol": symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    data = _get(f"{DATA_BASE}/futures/data/topLongShortAccountRatio", params)
    if not data:
        return pd.DataFrame(columns=["ts", "top_long_ratio", "top_short_ratio", "top_ls_ratio"])
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["top_long_ratio"] = df["longAccount"].astype(float)
    df["top_short_ratio"] = df["shortAccount"].astype(float)
    df["top_ls_ratio"] = df["longShortRatio"].astype(float)
    return df[["ts", "top_long_ratio", "top_short_ratio", "top_ls_ratio"]].sort_values("ts").reset_index(drop=True)


# ---------- Taker Buy/Sell Volume ----------

def fetch_taker_long_short_ratio(
    symbol: str,
    period: Period = "1h",
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Taker buy/sell volume ratio (aggressors).
    High ratio = taker buy dominant (bullish aggression).
    """
    params: dict = {"symbol": symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    data = _get(f"{DATA_BASE}/futures/data/takerlongshortRatio", params)
    if not data:
        return pd.DataFrame(columns=["ts", "buy_vol", "sell_vol", "taker_ratio"])
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["buy_vol"] = df["buyVol"].astype(float)
    df["sell_vol"] = df["sellVol"].astype(float)
    df["taker_ratio"] = df["buySellRatio"].astype(float)
    return df[["ts", "buy_vol", "sell_vol", "taker_ratio"]].sort_values("ts").reset_index(drop=True)


# ---------- Funding Rate ----------

def fetch_funding_rate_history(
    symbol: str,
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Funding rate history (every 8h for most pairs)."""
    params: dict = {"symbol": symbol, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    data = _get(f"{BASE}/fapi/v1/fundingRate", params)
    if not data:
        return pd.DataFrame(columns=["ts", "funding_rate"])
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)


# ---------- Composite fetch ----------

def fetch_all_derivatives_data(
    symbol: str,
    period: Period = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OI + Long/Short + Taker Ratio + merge into single DataFrame aligned by timestamp.
    This is the main entry point for strategies that need derivatives market data.
    """
    oi = fetch_oi_history(symbol, period, limit)
    ls = fetch_global_long_short_ratio(symbol, period, limit)
    top_ls = fetch_top_trader_long_short_ratio(symbol, period, limit)
    taker = fetch_taker_long_short_ratio(symbol, period, limit)

    if oi.empty:
        return pd.DataFrame()

    merged = oi.copy()
    for other in [ls, top_ls, taker]:
        if not other.empty:
            merged = pd.merge_asof(
                merged.sort_values("ts"),
                other.sort_values("ts"),
                on="ts",
                direction="backward",
            )

    return merged.sort_values("ts").reset_index(drop=True)
