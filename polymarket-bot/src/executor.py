"""Order execution adapters.

PaperExecutor simulates fills against (midpoint + slippage) using config-driven
fees, mutating the in-memory PortfolioState. It's the default and what every
strategy iteration should be tested against first.

LiveExecutor is a stub. To wire it up:
  pip install py-clob-client
  Implement _place_order using ClobClient.create_and_post_order.
  Set execution.mode: "live" in config.yaml.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .journal import Journal
from .state import PortfolioState, Side

log = logging.getLogger(__name__)


@dataclass
class FillResult:
    side: Side
    shares: float
    fill_price: float
    fee_usd: float
    notional_usd: float
    ok: bool
    reason: str = ""


class BaseExecutor(ABC):
    @abstractmethod
    def buy(self, side: Side, target_usd: float, midpoint: float, market_slug: str = "", reason: str = "") -> FillResult:
        ...


class PaperExecutor(BaseExecutor):
    """Fills at midpoint + slippage. No real network calls."""

    def __init__(
        self,
        state: PortfolioState,
        fee_rate: float,
        slippage: float,
        journal: Optional[Journal] = None,
        mode_label: str = "paper",
    ):
        self.state = state
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)
        self.journal = journal
        self.mode_label = mode_label

    def buy(self, side: Side, target_usd: float, midpoint: float, market_slug: str = "", reason: str = "") -> FillResult:
        if target_usd <= 0:
            return FillResult(side, 0.0, 0.0, 0.0, 0.0, ok=False, reason="non-positive target_usd")
        # Buys cross the book → pay above midpoint. Cap at 0.999 — share price
        # can never exceed 1 in a binary market.
        fill_price = min(midpoint + self.slippage, 0.999)
        if fill_price <= 0:
            return FillResult(side, 0.0, 0.0, 0.0, 0.0, ok=False, reason=f"invalid fill_price={fill_price}")

        # Solve for shares such that shares*fill_price + shares*fill_price*fee_rate <= target_usd.
        cost_per_share = fill_price * (1 + self.fee_rate)
        shares = target_usd / cost_per_share
        notional = shares * fill_price
        fee = notional * self.fee_rate
        total_cost = notional + fee

        if total_cost > self.state.cash_usd + 1e-6:
            if self.state.cash_usd <= 0.01:
                return FillResult(side, 0.0, fill_price, 0.0, 0.0, ok=False, reason="insufficient cash")
            shares = self.state.cash_usd / cost_per_share
            notional = shares * fill_price
            fee = notional * self.fee_rate

        self.state.apply_buy(side, shares, fill_price, fee)
        log.info(
            "[%s] BUY %s %.3f shares @ %.4f (mid=%.4f, fee=$%.4f, notional=$%.2f)",
            self.mode_label.upper(), side, shares, fill_price, midpoint, fee, notional,
        )
        if self.journal is not None:
            self.journal.record_trade(
                mode=self.mode_label,
                market_slug=market_slug,
                side=side,
                shares=shares,
                fill_price=fill_price,
                fee_usd=fee,
                notional_usd=notional,
                cash_after=self.state.cash_usd,
                yes_shares_after=self.state.yes.shares,
                no_shares_after=self.state.no.shares,
                reason=reason,
            )
        return FillResult(side, shares, fill_price, fee, notional, ok=True)


class LiveExecutor(BaseExecutor):
    """Real Polygon execution. Not implemented yet — see README 'Going Live'."""

    def __init__(self, state: PortfolioState, fee_rate: float, slippage: float, journal: Optional[Journal] = None):
        self.state = state
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.journal = journal

    def buy(self, side: Side, target_usd: float, midpoint: float, market_slug: str = "", reason: str = "") -> FillResult:
        raise NotImplementedError(
            "LiveExecutor is a stub. Install py-clob-client, set up POLYGON_PRIVATE_KEY "
            "+ CLOB_API_KEY/SECRET/PASSPHRASE in .env, and implement order placement here."
        )
