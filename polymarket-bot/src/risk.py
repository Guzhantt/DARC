"""Risk guardrails — the most important file for a beginner.

Two things this module enforces:

1. Daily loss kill switch. Once realized + unrealized P/L drops below
   `daily_max_loss_usd`, the bot refuses to take any new entries until the
   UTC day rolls over. Existing positions are still managed (endgame hedge
   may still fire) — we just stop opening new risk.

2. Per-tick sanity caps. Refuse to enter if cash is below a floor, refuse
   to size a single trade above a fraction of starting equity. These prevent
   single-bug runaway losses.

The bot can still lose money. This just makes catastrophic losses much
harder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    starting_equity_usd: float
    daily_max_loss_usd: float
    min_cash_floor_usd: float
    max_single_trade_usd: float
    # Day boundary used for resetting the kill switch.
    _last_check_day: Optional[date] = None
    _kill_switch_engaged: bool = False
    _peak_equity_today: float = field(default=0.0)

    def __post_init__(self) -> None:
        self._peak_equity_today = self.starting_equity_usd

    def update(self, current_equity_usd: float) -> None:
        """Call once per tick with mark-to-market equity (cash + open shares).

        Tracks intraday peak so the kill switch can also fire on
        peak-to-trough drawdown if you want. Right now we only fire on
        absolute loss vs. starting equity (simpler for a beginner to reason
        about).
        """
        today = datetime.now(timezone.utc).date()
        if self._last_check_day != today:
            if self._last_check_day is not None:
                log.info("New UTC day — resetting kill switch")
            self._last_check_day = today
            self._kill_switch_engaged = False
            self._peak_equity_today = current_equity_usd

        self._peak_equity_today = max(self._peak_equity_today, current_equity_usd)
        loss = self.starting_equity_usd - current_equity_usd
        if loss >= self.daily_max_loss_usd and not self._kill_switch_engaged:
            self._kill_switch_engaged = True
            log.error(
                "KILL SWITCH ENGAGED: equity=$%.2f, loss=$%.2f >= limit $%.2f. "
                "No new entries until UTC day rollover.",
                current_equity_usd, loss, self.daily_max_loss_usd,
            )

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_engaged

    def can_open_new_position(self, cash_usd: float) -> tuple[bool, str]:
        """Return (allowed, reason). Reason is empty if allowed."""
        if self._kill_switch_engaged:
            return False, "daily kill switch engaged"
        if cash_usd < self.min_cash_floor_usd:
            return False, f"cash ${cash_usd:.2f} below floor ${self.min_cash_floor_usd:.2f}"
        return True, ""

    def cap_trade_size(self, requested_usd: float) -> float:
        """Hard cap per-trade notional. Silently downsizes — never raises."""
        if requested_usd <= self.max_single_trade_usd:
            return requested_usd
        log.warning(
            "Trade size $%.2f capped to $%.2f (max_single_trade_usd)",
            requested_usd, self.max_single_trade_usd,
        )
        return self.max_single_trade_usd
