"""In-memory position + cash tracking.

Polymarket trades binary outcome shares: each YES share pays $1 if YES wins,
$0 otherwise (and vice versa for NO). Share price is always in [0, 1].

We track shares per side, weighted-avg fill price, free cash, and per-side
entry context (entry midpoint, entry spot price) used by stop-loss and
inverted-book early-exit logic. Restart = fresh state — there's no
persistence layer (intentional, this is a paper strategy iteration tool).
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
    avg_price: float = 0.0       # weighted-avg fill price across all buys
    cost_usd: float = 0.0        # cumulative USD spent on this side (incl. fees)
    entry_spot_price: float = 0.0  # spot price at first entry — used for reversal detection

    def add_fill(self, shares: float, price: float, fee_usd: float, spot_price: float = 0.0) -> None:
        if shares <= 0:
            return
        new_shares = self.shares + shares
        # Weighted-avg fill price (excluding fees, so it stays in [0,1]).
        self.avg_price = (self.avg_price * self.shares + price * shares) / new_shares
        if self.shares <= 1e-9 and spot_price > 0:
            # Record entry context only on the first fill of a new position.
            self.entry_spot_price = spot_price
        self.shares = new_shares
        self.cost_usd += shares * price + fee_usd

    def reduce(self, shares: float, fill_price: float) -> float:
        """Reduce position by `shares` at `fill_price`. Returns USD received.

        Realized P/L is left to the caller to log if needed; here we just
        track shares and reset entry context if the position is fully closed.
        """
        if shares <= 0 or self.shares <= 0:
            return 0.0
        shares = min(shares, self.shares)
        proceeds = shares * fill_price
        self.shares -= shares
        if self.shares <= 1e-9:
            self.shares = 0.0
            self.avg_price = 0.0
            self.cost_usd = 0.0
            self.entry_spot_price = 0.0
        return proceeds

    @property
    def is_open(self) -> bool:
        return self.shares > 1e-9

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """(current - avg) / avg. Returns 0 when there's no position."""
        if not self.is_open or self.avg_price <= 0:
            return 0.0
        return (current_price - self.avg_price) / self.avg_price


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
        return self.get(low_side).is_open

    def net_share_exposure(self) -> float:
        """yes_shares - no_shares. Positive = net long YES."""
        return self.yes.shares - self.no.shares

    def is_unbalanced(self, threshold_usd: float) -> bool:
        return abs(self.net_share_exposure()) > threshold_usd

    def mark_to_market_equity(self, yes_price: float, no_price: float) -> float:
        yes_value = self.yes.shares * yes_price
        no_value = self.no.shares * no_price
        return self.cash_usd + yes_value + no_value

    # ---------- mutation ----------

    def apply_buy(self, side: Side, shares: float, price: float, fee_usd: float, spot_price: float = 0.0) -> None:
        cash_out = shares * price + fee_usd
        self.cash_usd -= cash_out
        self.get(side).add_fill(shares, price, fee_usd, spot_price=spot_price)

    def apply_sell(self, side: Side, shares: float, fill_price: float, fee_usd: float = 0.0) -> float:
        proceeds = self.get(side).reduce(shares, fill_price)
        net_in = proceeds - fee_usd
        self.cash_usd += net_in
        return net_in

    def summary(self) -> str:
        return (
            f"cash=${self.cash_usd:.2f} "
            f"YES={self.yes.shares:.2f}@{self.yes.avg_price:.3f} "
            f"NO={self.no.shares:.2f}@{self.no.avg_price:.3f} "
            f"net_shares={self.net_share_exposure():+.2f}"
        )
