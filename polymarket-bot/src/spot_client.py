"""Spot momentum signal from Binance public klines.

Returns a multi-timeframe reading so the strategy can require agreement
across 1-2m (fast) and 5m (slow) windows, plus a 14-period RSI and a
volume-vs-baseline ratio. This is intentionally simple — no ML, no
complex indicators — but having multiple signals reduces false positives
from a single noisy 1m candle.

Falls back from Binance.com to Binance.US automatically (the .com host is
blocked from US residential IPs and from datacenters; .us works for both
geos for the major USDT pairs we care about).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal

import requests

DEFAULT_BASES: tuple[str, ...] = (
    "https://api.binance.com",
    "https://api.binance.us",
)

log = logging.getLogger(__name__)

Side = Literal["YES", "NO"]


class SpotError(RuntimeError):
    pass


@dataclass
class MomentumReading:
    """All spot signals computed from one klines fetch."""
    last_price: float
    short_return: float        # signed return over short_lookback_minutes
    long_return: float         # signed return over long_lookback_minutes
    rsi: float                 # 14-period RSI on closes (0..100)
    volume_ratio: float        # latest_volume / mean_volume_over_lookback (1.0 = average)
    candle_count: int

    @property
    def short_direction(self) -> Literal["UP", "DOWN", "FLAT"]:
        if self.short_return > 0:
            return "UP"
        if self.short_return < 0:
            return "DOWN"
        return "FLAT"


class SpotClient:
    def __init__(
        self,
        symbol: str,
        short_lookback_minutes: int,
        long_lookback_minutes: int,
        alignment_threshold: float,
        rsi_period: int = 14,
        volume_lookback_minutes: int = 20,
        timeout_sec: float = 5.0,
        bases: tuple[str, ...] = DEFAULT_BASES,
    ):
        self.symbol = symbol.upper()
        self.short_lookback = max(1, int(short_lookback_minutes))
        self.long_lookback = max(self.short_lookback, int(long_lookback_minutes))
        self.alignment_threshold = abs(float(alignment_threshold))
        self.rsi_period = max(2, int(rsi_period))
        self.volume_lookback = max(2, int(volume_lookback_minutes))
        self._timeout = timeout_sec
        self._bases: List[str] = list(bases)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/0.1"})

    def fetch(self) -> MomentumReading:
        """Pull recent 1-minute klines and compute all signals."""
        # Need enough history for the longest indicator. RSI traditionally needs
        # ~3x the period to stabilize. +1 candle for the in-progress one.
        limit = max(self.long_lookback, self.rsi_period * 3, self.volume_lookback) + 2
        last_err: Exception | None = None
        for idx, base in enumerate(list(self._bases)):
            try:
                resp = self._session.get(
                    f"{base}/api/v3/klines",
                    params={"symbol": self.symbol, "interval": "1m", "limit": limit},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                rows = resp.json()
            except requests.RequestException as e:
                last_err = e
                log.debug("Spot base %s failed: %s", base, e)
                continue

            if not isinstance(rows, list) or len(rows) < self.short_lookback + 2:
                last_err = SpotError(f"{base} returned too few candles: got {len(rows) if isinstance(rows, list) else '?'}")
                continue

            try:
                closes = [float(r[4]) for r in rows]
                volumes = [float(r[5]) for r in rows]
            except (IndexError, TypeError, ValueError) as e:
                last_err = SpotError(f"Malformed kline row from {base}: {rows[0] if rows else '?'}")
                continue

            new_close = closes[-1]
            if new_close <= 0:
                last_err = SpotError(f"Non-positive last close from {base}: {new_close}")
                continue

            short_return = _signed_return(closes, self.short_lookback)
            long_return = _signed_return(closes, self.long_lookback)
            rsi = _rsi(closes, self.rsi_period)
            volume_ratio = _volume_ratio(volumes, self.volume_lookback)

            # Promote working base to front for next call.
            if idx > 0:
                self._bases.insert(0, self._bases.pop(idx))

            return MomentumReading(
                last_price=new_close,
                short_return=short_return,
                long_return=long_return,
                rsi=rsi,
                volume_ratio=volume_ratio,
                candle_count=len(rows),
            )

        raise SpotError(f"All spot bases failed for {self.symbol}: {last_err}")


def momentum_aligns(reading: MomentumReading, side: Side, threshold: float, require_multi_tf: bool = True) -> bool:
    """Direction confirmation across timeframes.

    Strict mode (require_multi_tf=True): both short and long returns must
    agree with the side, and the short return magnitude must exceed
    threshold. This kills most single-bar noise spikes.

    Loose mode: only short return is checked. Use when callers don't have
    enough historical data (e.g. just-started bot).
    """
    short_r = reading.short_return
    long_r = reading.long_return
    if side == "YES":
        ok_short = short_r >= threshold
        ok_long = long_r >= 0  # long can be smaller, just direction-consistent
    elif side == "NO":
        ok_short = short_r <= -threshold
        ok_long = long_r <= 0
    else:
        return False
    return ok_short and (ok_long or not require_multi_tf)


def momentum_score(reading: MomentumReading, total: float) -> float:
    """User-spec metric: (short return) × (1 - total).

    Larger absolute value = better entry candidate. Sign indicates direction
    (positive = supports YES, negative = supports NO). Logged for visibility;
    the strategy still uses the discrete momentum_aligns gate above.
    """
    return reading.short_return * max(0.0, 1.0 - total)


# ---------- internal helpers ----------

def _signed_return(closes: List[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    old = closes[-1 - lookback]
    new = closes[-1]
    if old <= 0:
        return 0.0
    return (new - old) / old


def _rsi(closes: List[float], period: int) -> float:
    """Standard Wilder RSI on closes. Returns 50.0 on insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    # Initial average over the first `period` deltas.
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    # Wilder smoothing for remaining points.
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _volume_ratio(volumes: List[float], lookback: int) -> float:
    """Latest volume divided by mean over the previous `lookback` candles.

    Returns 1.0 on insufficient data so callers don't apply a filter that
    can't be evaluated.
    """
    if len(volumes) < 2:
        return 1.0
    n = min(lookback, len(volumes) - 1)
    if n <= 0:
        return 1.0
    baseline = sum(volumes[-1 - n:-1]) / n
    if baseline <= 0:
        return 1.0
    return volumes[-1] / baseline
