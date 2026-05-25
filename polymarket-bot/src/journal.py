"""Trade journal — append-only CSV of every fill and every tick decision."""
from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_TRADES_HEADER = [
    "timestamp_utc", "mode", "market_slug", "side", "action",
    "shares", "fill_price", "fee_usd", "notional_usd",
    "cash_after", "yes_shares_after", "no_shares_after", "reason",
]
_TICKS_HEADER = [
    "timestamp_utc", "market_slug", "yes_price", "no_price", "total",
    "remaining_sec", "spot_last", "spot_short_return", "spot_long_return",
    "spot_rsi", "spot_volume_ratio", "momentum_score", "decision",
    "yes_shares", "no_shares", "cash_usd", "equity_usd",
]


class Journal:
    """Thread-safe-ish CSV writer (single thread but the lock is cheap)."""

    def __init__(self, log_dir: str | Path = "logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trades_path = self.dir / "trades.csv"
        self.ticks_path = self.dir / "ticks.csv"
        self._lock = threading.Lock()
        self._ensure_header(self.trades_path, _TRADES_HEADER)
        self._ensure_header(self.ticks_path, _TICKS_HEADER)

    def _ensure_header(self, path: Path, header: list[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    def record_trade(
        self, *,
        mode: str, market_slug: str, side: str, action: str,
        shares: float, fill_price: float, fee_usd: float, notional_usd: float,
        cash_after: float, yes_shares_after: float, no_shares_after: float,
        reason: str,
    ) -> None:
        row = [
            _utc_now(), mode, market_slug, side, action,
            f"{shares:.6f}", f"{fill_price:.6f}", f"{fee_usd:.6f}", f"{notional_usd:.6f}",
            f"{cash_after:.4f}", f"{yes_shares_after:.6f}", f"{no_shares_after:.6f}",
            reason,
        ]
        self._append(self.trades_path, row)

    def record_tick(
        self, *,
        market_slug: str, yes_price: float, no_price: float, remaining_sec: float,
        spot_last: float, spot_short_return: float, spot_long_return: float,
        spot_rsi: float, spot_volume_ratio: float, momentum_score: float,
        decision: str, yes_shares: float, no_shares: float, cash_usd: float, equity_usd: float,
    ) -> None:
        row = [
            _utc_now(), market_slug,
            f"{yes_price:.4f}", f"{no_price:.4f}", f"{yes_price + no_price:.4f}",
            f"{remaining_sec:.0f}", f"{spot_last:.2f}",
            f"{spot_short_return:+.6f}", f"{spot_long_return:+.6f}",
            f"{spot_rsi:.1f}", f"{spot_volume_ratio:.2f}", f"{momentum_score:+.6f}",
            decision,
            f"{yes_shares:.4f}", f"{no_shares:.4f}",
            f"{cash_usd:.4f}", f"{equity_usd:.4f}",
        ]
        self._append(self.ticks_path, row)

    def _append(self, path: Path, row: list) -> None:
        try:
            with self._lock, path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            log.warning("Journal write failed (%s): %s", path, e)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
