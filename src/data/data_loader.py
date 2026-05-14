"""Historical kline downloader with CSV cache."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.exchange.binance_client import BinanceFutures

# timeframe -> milliseconds
TF_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _cache_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return data_dir / f"{safe}_{timeframe}.csv"


def load_ohlcv(
    client: BinanceFutures,
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download (with cache) the last ``days`` of OHLCV for ``symbol``."""
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    path = _cache_path(data_dir, symbol, timeframe)
    end_ms = client.exchange.milliseconds()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    cached: pd.DataFrame | None = None
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["ts"])
        if not cached.empty:
            last_ts = int(cached["ts"].iloc[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ts + TF_MS[timeframe])

    frames: list[pd.DataFrame] = []
    cur = start_ms
    tf_ms = TF_MS[timeframe]
    while cur < end_ms:
        df = client.fetch_ohlcv(symbol, timeframe, since_ms=cur, limit=1500)
        if df.empty:
            break
        frames.append(df)
        last = int(df["ts"].iloc[-1].timestamp() * 1000)
        if last <= cur:
            break
        cur = last + tf_ms
        time.sleep(client.exchange.rateLimit / 1000)

    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cached is not None and not cached.empty:
        combined = pd.concat([cached, new_df], ignore_index=True)
    else:
        combined = new_df
    if combined.empty:
        return combined
    combined = (
        combined.drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    combined.to_csv(path, index=False)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    return combined[combined["ts"] >= cutoff].reset_index(drop=True)
