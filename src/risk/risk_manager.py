"""Risk manager for live execution: position sizing + circuit breaker."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RiskManager:
    risk_per_trade: float
    max_leverage: int
    max_daily_drawdown: float
    max_open_positions: int

    # Mutable state
    day_start_equity: float = 0.0
    day_key: str = ""
    trading_halted: bool = False

    def on_new_day(self, equity: float, now: datetime | None = None) -> None:
        now = now or datetime.now(tz=timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day != self.day_key:
            self.day_key = day
            self.day_start_equity = equity
            self.trading_halted = False

    def check_daily_drawdown(self, equity: float) -> bool:
        """Returns True if trading should be halted for the day."""
        if self.day_start_equity <= 0:
            return False
        dd = (self.day_start_equity - equity) / self.day_start_equity
        if dd >= self.max_daily_drawdown:
            self.trading_halted = True
        return self.trading_halted

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> float:
        """Return quantity in base units. 0 if invalid setup."""
        if entry_price <= 0 or stop_price <= 0:
            return 0.0
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        qty = (equity * self.risk_per_trade) / risk_per_unit
        max_qty_by_lev = (equity * self.max_leverage) / entry_price
        return max(0.0, min(qty, max_qty_by_lev))
