"""Order execution adapters.

PaperExecutor simulates fills against the real orderbook (best ask for buys,
best bid for sells, or midpoint +/- slippage as fallback). It mutates the
in-memory PortfolioState and journals each fill.

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
    action: str              # "BUY" | "SELL"
    shares: float
    fill_price: float
    fee_usd: float
    notional_usd: float
    ok: bool
    reason: str = ""


class BaseExecutor(ABC):
    @abstractmethod
    def buy(
        self, side: Side, target_usd: float, midpoint: float,
        market_slug: str = "", reason: str = "", spot_price: float = 0.0,
        best_ask: Optional[float] = None,
    ) -> FillResult: ...

    @abstractmethod
    def sell(
        self, side: Side, target_shares: float, midpoint: float,
        market_slug: str = "", reason: str = "",
        best_bid: Optional[float] = None,
    ) -> FillResult: ...


class PaperExecutor(BaseExecutor):
    """Fills at best-ask/bid when known, else midpoint +/- slippage."""

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

    def buy(
        self, side: Side, target_usd: float, midpoint: float,
        market_slug: str = "", reason: str = "", spot_price: float = 0.0,
        best_ask: Optional[float] = None,
    ) -> FillResult:
        if target_usd <= 0:
            return FillResult(side, "BUY", 0.0, 0.0, 0.0, 0.0, ok=False, reason="non-positive target_usd")
        # Cross the book: pay best_ask if available, else midpoint + slippage.
        # Cap at 0.999 — share price can never exceed $1 in a binary market.
        raw_fill = best_ask if best_ask and best_ask > 0 else midpoint + self.slippage
        fill_price = min(raw_fill, 0.999)
        if fill_price <= 0:
            return FillResult(side, "BUY", 0.0, 0.0, 0.0, 0.0, ok=False, reason=f"invalid fill_price={fill_price}")

        cost_per_share = fill_price * (1 + self.fee_rate)
        shares = target_usd / cost_per_share
        notional = shares * fill_price
        fee = notional * self.fee_rate
        total_cost = notional + fee

        if total_cost > self.state.cash_usd + 1e-6:
            if self.state.cash_usd <= 0.01:
                return FillResult(side, "BUY", 0.0, fill_price, 0.0, 0.0, ok=False, reason="insufficient cash")
            shares = self.state.cash_usd / cost_per_share
            notional = shares * fill_price
            fee = notional * self.fee_rate

        self.state.apply_buy(side, shares, fill_price, fee, spot_price=spot_price)
        log.info(
            "[%s] BUY  %s %.3f sh @ %.4f (mid=%.4f, fee=$%.4f, notional=$%.2f)",
            self.mode_label.upper(), side, shares, fill_price, midpoint, fee, notional,
        )
        if self.journal is not None:
            self.journal.record_trade(
                mode=self.mode_label, market_slug=market_slug,
                side=side, action="BUY",
                shares=shares, fill_price=fill_price, fee_usd=fee, notional_usd=notional,
                cash_after=self.state.cash_usd,
                yes_shares_after=self.state.yes.shares,
                no_shares_after=self.state.no.shares,
                reason=reason,
            )
        return FillResult(side, "BUY", shares, fill_price, fee, notional, ok=True)

    def sell(
        self, side: Side, target_shares: float, midpoint: float,
        market_slug: str = "", reason: str = "",
        best_bid: Optional[float] = None,
    ) -> FillResult:
        position = self.state.get(side)
        if not position.is_open:
            return FillResult(side, "SELL", 0.0, 0.0, 0.0, 0.0, ok=False, reason="no position")
        target_shares = min(target_shares, position.shares)
        if target_shares <= 1e-6:
            return FillResult(side, "SELL", 0.0, 0.0, 0.0, 0.0, ok=False, reason="non-positive shares")

        # Hit the bid (or midpoint - slippage). Floor at 0.001 so we don't divide
        # by zero anywhere downstream.
        raw_fill = best_bid if best_bid and best_bid > 0 else midpoint - self.slippage
        fill_price = max(raw_fill, 0.001)
        notional = target_shares * fill_price
        fee = notional * self.fee_rate

        net_in = self.state.apply_sell(side, target_shares, fill_price, fee_usd=fee)
        log.info(
            "[%s] SELL %s %.3f sh @ %.4f (mid=%.4f, fee=$%.4f, net=$%.2f)",
            self.mode_label.upper(), side, target_shares, fill_price, midpoint, fee, net_in,
        )
        if self.journal is not None:
            self.journal.record_trade(
                mode=self.mode_label, market_slug=market_slug,
                side=side, action="SELL",
                shares=target_shares, fill_price=fill_price, fee_usd=fee, notional_usd=notional,
                cash_after=self.state.cash_usd,
                yes_shares_after=self.state.yes.shares,
                no_shares_after=self.state.no.shares,
                reason=reason,
            )
        return FillResult(side, "SELL", target_shares, fill_price, fee, notional, ok=True)


class LiveExecutor(BaseExecutor):
    """Real Polygon execution. Not implemented yet — see README 'Going Live'."""

    def __init__(self, state: PortfolioState, fee_rate: float, slippage: float, journal: Optional[Journal] = None):
        self.state = state
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.journal = journal

    def buy(self, *args, **kwargs) -> FillResult:
        raise NotImplementedError(
            "LiveExecutor is a stub. Install py-clob-client, set up POLYGON_PRIVATE_KEY "
            "+ CLOB_API_KEY/SECRET/PASSPHRASE in .env, and implement order placement here."
        )

    def sell(self, *args, **kwargs) -> FillResult:
        raise NotImplementedError("LiveExecutor.sell not implemented")
