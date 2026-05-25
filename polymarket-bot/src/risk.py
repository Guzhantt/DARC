"""Risk guardrails — the most important file for a beginner.

What this enforces:
  1. Daily loss kill switch. Disables NEW entries when intraday equity falls
     below `starting_equity - daily_max_loss`. Existing positions (and their
     stop-loss / endgame hedges) are still managed — those reduce risk.
     Resets at UTC day boundary.
  2. Cash floor. Refuse to open new positions when free cash drops below
     `min_cash_floor_usd`.
  3. Per-trade notional cap. Silently downsize any single trade above
     `max_single_trade_usd`.

Loss limit accepts either an absolute USD amount, a % of starting equity,
or both (whichever is stricter wins). At least one must be set.
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
    daily_max_loss_pct: float       # e.g. 0.025 = 2.5%. 0 disables this lever.
    daily_max_loss_usd: float       # 0 disables this lever.
    min_cash_floor_usd: float
    max_single_trade_usd: float
    _last_check_day: Optional[date] = None
    _kill_switch_engaged: bool = False
    _peak_equity_today: float = field(default=0.0)

    def __post_init__(self) -> None:
        self._peak_equity_today = self.starting_equity_usd
        if self.effective_loss_limit_usd() <= 0:
            raise ValueError(
                "Risk config: either daily_max_loss_pct or daily_max_loss_usd must be > 0"
            )

    def effective_loss_limit_usd(self) -> float:
        """Stricter of the two configured limits. Stricter = smaller positive."""
        candidates = []
        if self.daily_max_loss_pct > 0:
            candidates.append(self.starting_equity_usd * self.daily_max_loss_pct)
        if self.daily_max_loss_usd > 0:
            candidates.append(self.daily_max_loss_usd)
        return min(candidates) if candidates else 0.0

    def update(self, current_equity_usd: float) -> None:
        today = datetime.now(timezone.utc).date()
        if self._last_check_day != today:
            if self._last_check_day is not None:
                log.info("New UTC day — resetting kill switch")
            self._last_check_day = today
            self._kill_switch_engaged = False
            self._peak_equity_today = current_equity_usd

        self._peak_equity_today = max(self._peak_equity_today, current_equity_usd)
        loss = self.starting_equity_usd - current_equity_usd
        limit = self.effective_loss_limit_usd()
        if loss >= limit and not self._kill_switch_engaged:
            self._kill_switch_engaged = True
            log.error(
                "KILL SWITCH ENGAGED: equity=$%.2f, loss=$%.2f >= limit $%.2f. "
                "No new opening trades until UTC day rollover.",
                current_equity_usd, loss, limit,
            )

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_engaged

    def can_open_new_position(self, cash_usd: float) -> tuple[bool, str]:
        if self._kill_switch_engaged:
            return False, "daily kill switch engaged"
        if cash_usd < self.min_cash_floor_usd:
            return False, f"cash ${cash_usd:.2f} below floor ${self.min_cash_floor_usd:.2f}"
        return True, ""

    def cap_trade_size(self, requested_usd: float) -> float:
        if requested_usd <= self.max_single_trade_usd:
            return requested_usd
        log.warning(
            "Trade size $%.2f capped to $%.2f (max_single_trade_usd)",
            requested_usd, self.max_single_trade_usd,
        )
        return self.max_single_trade_usd
