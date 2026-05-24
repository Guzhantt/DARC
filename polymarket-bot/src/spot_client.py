"""Spot momentum signal from Binance public klines.

We use the anonymous /api/v3/klines endpoint — no API key required. The
client tries multiple base URLs in order, which handles two common cases:
  - Binance.com is geo-blocked from the user's network (e.g. US residential
    or any datacenter IP) → automatically falls back to Binance.US.
  - Either host is temporarily 5xx → the next host is tried on the same tick.

The signal is a signed return over the last N 1-minute candles, normalized
so that a positive value means "spot is trending up over the lookback".

If you want fancier signals (EMA cross, RSI, realized vol) extend
`compute_momentum` — the strategy only consumes the final scalar via
`momentum_aligns()`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal

import requests

# Tried in order. Binance.US uses the same endpoint shape and symbol naming
# for major pairs like BTCUSDT/ETHUSDT.
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
    signed_return: float       # e.g. +0.0012 = +12 bps over the lookback
    last_price: float
    candle_count: int

    @property
    def direction(self) -> Literal["UP", "DOWN", "FLAT"]:
        if self.signed_return > 0:
            return "UP"
        if self.signed_return < 0:
            return "DOWN"
        return "FLAT"


class SpotClient:
    def __init__(
        self,
        symbol: str,
        lookback_minutes: int,
        alignment_threshold: float,
        timeout_sec: float = 5.0,
        bases: tuple[str, ...] = DEFAULT_BASES,
    ):
        self.symbol = symbol.upper()
        self.lookback_minutes = max(1, int(lookback_minutes))
        self.alignment_threshold = abs(float(alignment_threshold))
        self._timeout = timeout_sec
        self._bases: List[str] = list(bases)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/0.1"})

    def fetch(self) -> MomentumReading:
        """Pull recent 1-minute klines and compute signed return.

        Tries each base URL in order; the first that responds with usable
        data wins, and that base is moved to the front for the next call so
        we don't re-pay the geo-block timeout every tick.
        """
        # Ask for one extra candle so we always have a full window even if the
        # most recent one is still forming.
        limit = self.lookback_minutes + 1
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

            if not isinstance(rows, list) or len(rows) < 2:
                last_err = SpotError(f"{base} returned too few candles: {rows!r}")
                continue

            try:
                old_close = float(rows[-1 - self.lookback_minutes][4]) if len(rows) > self.lookback_minutes else float(rows[0][4])
                new_close = float(rows[-1][4])
            except (IndexError, TypeError, ValueError) as e:
                last_err = SpotError(f"Malformed kline row from {base}: {rows[:2]!r}")
                continue

            if old_close <= 0:
                last_err = SpotError(f"Non-positive reference close from {base}: {old_close}")
                continue

            # Promote the working base to the front so subsequent calls hit it first.
            if idx > 0:
                self._bases.insert(0, self._bases.pop(idx))

            signed_return = (new_close - old_close) / old_close
            return MomentumReading(
                signed_return=signed_return,
                last_price=new_close,
                candle_count=len(rows),
            )

        raise SpotError(f"All spot bases failed for {self.symbol}: {last_err}")


def momentum_aligns(reading: MomentumReading, side: Side, threshold: float) -> bool:
    """True if momentum supports buying `side`.

    For YES = "price will go up", we want positive momentum above threshold.
    For NO  = "price will go down", we want negative momentum below -threshold.

    Markets where YES/NO don't map to up/down (e.g. event outcomes) should
    disable this filter — the strategy degrades gracefully if you set
    spot.alignment_threshold to a huge value, making this always False.
    """
    if side == "YES":
        return reading.signed_return >= threshold
    if side == "NO":
        return reading.signed_return <= -threshold
    return False
