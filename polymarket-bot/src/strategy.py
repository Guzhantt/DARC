"""Three-rule decision logic, mapped 1:1 from the user pseudocode.

The strategy is pure: it consumes a snapshot + state and returns an
ordered list of intended actions. The main loop runs them through the
executor. This separation makes the rules easy to unit-test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .polymarket_client import MarketSnapshot
from .spot_client import MomentumReading, momentum_aligns
from .state import PortfolioState, Side, opposite

log = logging.getLogger(__name__)


@dataclass
class Intent:
    """A buy decision the strategy wants the executor to perform."""
    side: Side
    target_usd: float
    midpoint: float
    reason: str


def cheap_side(snap: MarketSnapshot) -> Side:
    """Whichever side currently has the lower midpoint."""
    return "YES" if snap.yes_price <= snap.no_price else "NO"


def midpoint_for(snap: MarketSnapshot, side: Side) -> float:
    return snap.yes_price if side == "YES" else snap.no_price


def decide(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
) -> List[Intent]:
    """Return the intents to act on this tick. Empty list = do nothing.

    Three rules in order — at most one buy intent per tick to keep behavior
    obvious. The endgame branch may emit a hedge or stay silent.
    """
    s = cfg.strategy
    intents: List[Intent] = []
    total = snap.yes_price + snap.no_price
    low = cheap_side(snap)
    low_price = midpoint_for(snap, low)

    log.debug(
        "tick: yes=%.4f no=%.4f total=%.4f remaining=%.0fs cheap=%s mom=%+.5f %s",
        snap.yes_price, snap.no_price, total, snap.remaining_sec,
        low, momentum.signed_return, momentum.direction,
    )

    # ---------- Rule 3 (endgame) takes precedence ----------
    if snap.remaining_sec < s.endgame_remaining_sec:
        hedge = _endgame_hedge(cfg, snap, momentum, state)
        if hedge is not None:
            intents.append(hedge)
        return intents

    # ---------- Rule 1: scout entry ----------
    scout = _maybe_scout(cfg, snap, momentum, state, low, low_price)
    if scout is not None:
        intents.append(scout)
        # Don't also try to lock spread on the same tick — wait for next tick to
        # observe the post-fill book.
        return intents

    # ---------- Rule 2: lock spread / complement ----------
    arb = _maybe_lock_spread(cfg, snap, state, total)
    if arb is not None:
        intents.append(arb)

    return intents


def _maybe_scout(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
    low: Side,
    low_price: float,
) -> Optional[Intent]:
    s = cfg.strategy
    if snap.remaining_sec <= s.scout_min_remaining_sec:
        return None
    if low_price > s.scout_max_price:
        return None
    if state.has_position(low):
        # Already scouted this side — don't pyramid.
        return None
    if not momentum_aligns(momentum, low, cfg.spot.alignment_threshold):
        return None
    return Intent(
        side=low,
        target_usd=s.scout_size_usd,
        midpoint=low_price,
        reason=f"scout: cheap={low}@{low_price:.3f} mom={momentum.signed_return:+.5f}",
    )


def _maybe_lock_spread(
    cfg: Config,
    snap: MarketSnapshot,
    state: PortfolioState,
    total: float,
) -> Optional[Intent]:
    s = cfg.strategy
    if total > s.arb_total_threshold:
        return None
    if snap.remaining_sec <= s.arb_min_remaining_sec:
        return None

    # Need an existing low-side position to "lock" against.
    held_side: Optional[Side] = None
    for side in ("YES", "NO"):
        if state.has_position(side):  # type: ignore[arg-type]
            held_side = side  # type: ignore[assignment]
            break
    if held_side is None:
        return None

    complement: Side = opposite(held_side)
    complement_price = midpoint_for(snap, complement)
    held_shares = state.get(held_side).shares

    # Edge math:
    #   For each share-pair (1 YES + 1 NO) we paid (held_avg + complement_price).
    #   The pair guarantees $1 payout at settlement.
    #   Edge per pair = 1 - (held_avg + complement_fill_price + fees).
    held_avg = state.get(held_side).avg_price
    complement_fill = min(complement_price + cfg.fees.slippage_estimate, 0.999)
    fees_per_share = (held_avg + complement_fill) * cfg.fees.per_side_fee_rate
    edge_per_pair = 1.0 - (held_avg + complement_fill + fees_per_share)

    if edge_per_pair < s.min_profit_after_fees:
        log.debug(
            "spread fires (total=%.4f) but edge=%.4f < min=%.4f — skip",
            total, edge_per_pair, s.min_profit_after_fees,
        )
        return None

    # Buy enough complement shares to fully balance.
    needed_shares = max(0.0, held_shares - state.get(complement).shares)
    if needed_shares <= 1e-6:
        return None
    target_usd = needed_shares * complement_fill

    return Intent(
        side=complement,
        target_usd=target_usd,
        midpoint=complement_price,
        reason=(
            f"lock spread: total={total:.4f} edge={edge_per_pair:.4f} "
            f"buy {needed_shares:.2f} {complement} to balance"
        ),
    )


def _endgame_hedge(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
) -> Optional[Intent]:
    """In the last 2 minutes, only act if we're unbalanced AND spot disagrees."""
    s = cfg.strategy
    if not state.is_unbalanced(s.unbalanced_threshold_usd):
        return None

    net = state.net_share_exposure()
    long_side: Side = "YES" if net > 0 else "NO"
    # "Spot disagrees" = momentum points opposite to the side we're net-long.
    spot_confirms = momentum_aligns(momentum, long_side, cfg.spot.alignment_threshold)
    if spot_confirms:
        log.info(
            "endgame: net %s exposure %.2f, spot confirms (mom=%+.5f) — HOLD",
            long_side, abs(net), momentum.signed_return,
        )
        return None

    hedge_side: Side = opposite(long_side)
    hedge_shares = abs(net) * s.endgame_hedge_ratio
    hedge_price = midpoint_for(snap, hedge_side)
    hedge_fill = min(hedge_price + cfg.fees.slippage_estimate, 0.999)
    target_usd = hedge_shares * hedge_fill
    return Intent(
        side=hedge_side,
        target_usd=target_usd,
        midpoint=hedge_price,
        reason=(
            f"endgame hedge: net {long_side} {abs(net):.2f} shares, "
            f"spot disagrees (mom={momentum.signed_return:+.5f}), "
            f"hedge {hedge_shares:.2f} {hedge_side}"
        ),
    )
