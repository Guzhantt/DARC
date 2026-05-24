"""In-memory position + cash tracking.

Polymarket trades binary outcome shares: each YES share pays $1 if YES wins,
$0 otherwise (and vice versa for NO). Share price is always in [0, 1].

We track shares per side, weighted-avg fill price, and free cash. Restart =
fresh state — there's no persistence layer (intentional, this is a paper
strategy iteration tool).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

log = logging.getLogger(__name__)

Side = Literal["YES", "NO"]


def opposite(side: Side) -> Side:
    return "NO" if side == "YES" else "YES"


@dataclass
class Position:
    """Aggregate position on one side of a market."""
    side: Side
    shares: float = 0.0
    avg_price: float = 0.0      # weighted-avg fill price across all buys
    cost_usd: float = 0.0       # cumulative USD spent on this side (incl. fees)

    def add_fill(self, shares: float, price: float, fee_usd: float) -> None:
        if shares <= 0:
            return
        new_shares = self.shares + shares
        # Weighted-avg fill price (excluding fees, so it stays in [0,1]).
        self.avg_price = (self.avg_price * self.shares + price * shares) / new_shares
        self.shares = new_shares
        self.cost_usd += shares * price + fee_usd

    @property
    def is_open(self) -> bool:
        return self.shares > 1e-9


@dataclass
class PortfolioState:
    cash_usd: float
    yes: Position = field(default_factory=lambda: Position(side="YES"))
    no: Position = field(default_factory=lambda: Position(side="NO"))

    # ---------- queries used by the strategy ----------

    def get(self, side: Side) -> Position:
        return self.yes if side == "YES" else self.no

    def has_position(self, side: Optional[Side] = None) -> bool:
        if side is None:
            return self.yes.is_open or self.no.is_open
        return self.get(side).is_open

    def has_low_side_position(self, low_side: Side) -> bool:
        """Did we already scout the cheap side this cycle?"""
        return self.get(low_side).is_open

    def net_share_exposure(self) -> float:
        """yes_shares - no_shares. Positive = net long YES."""
        return self.yes.shares - self.no.shares

    def is_unbalanced(self, threshold_usd: float) -> bool:
        """Treat |net shares| as USD because each share has max $1 payout.

        If we hold equal shares of both sides we're delta-neutral on the
        binary outcome (worst case payout = total shares of the loser × $1
        offset by total shares of the winner × $1 ≈ break-even minus cost).
        """
        return abs(self.net_share_exposure()) > threshold_usd

    def mark_to_market_equity(self, yes_price: float, no_price: float) -> float:
        """Cash + open positions valued at current midpoint.

        For binary outcomes share value is bounded in [0, 1] and unrealized
        P/L is share_count * (current_midpoint - avg_buy_price). Using
        midpoint understates risk slightly (you'd sell into the bid in
        reality) but it's the right conservative read for risk decisions.
        """
        yes_value = self.yes.shares * yes_price
        no_value = self.no.shares * no_price
        return self.cash_usd + yes_value + no_value

    # ---------- mutation ----------

    def apply_buy(self, side: Side, shares: float, price: float, fee_usd: float) -> None:
        cash_out = shares * price + fee_usd
        self.cash_usd -= cash_out
        self.get(side).add_fill(shares, price, fee_usd)

    def summary(self) -> str:
        return (
            f"cash=${self.cash_usd:.2f} "
            f"YES={self.yes.shares:.2f}@{self.yes.avg_price:.3f} "
            f"NO={self.no.shares:.2f}@{self.no.avg_price:.3f} "
            f"net_shares={self.net_share_exposure():+.2f}"
        )
