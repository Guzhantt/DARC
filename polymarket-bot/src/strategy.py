"""Trading rules.

The strategy is pure: takes (config, snapshot, momentum, state) → returns
(intents, decision_label). At most one intent per tick to keep behaviour
obvious. The label is what gets logged to ticks.csv — that label is the
single most useful signal for tuning by hand.

Rule order (highest priority first, first match wins):
  1. STOP_LOSS  — close a leg that's down more than stop_loss_pct
  2. INVERTED_BOOK_EXIT — total > 1.02 AND spot reversed → close
  3. ENDGAME_HEDGE — last 2-4 minutes, hedge if unbalanced and spot disagrees
  4. SCOUT — early window: buy cheap side if momentum confirms
  5. LOCK_SPREAD — middle window: buy complement when YES+NO < 0.97 and EV >= cutoff
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from .config import Config
from .polymarket_client import BookSide, MarketSnapshot
from .spot_client import MomentumReading, momentum_aligns
from .state import PortfolioState, Side, opposite

log = logging.getLogger(__name__)

Action = Literal["BUY", "SELL"]


@dataclass
class Intent:
    """What the strategy wants the executor to do this tick."""
    action: Action
    side: Side
    # For BUY: USD notional. For SELL: number of shares.
    size: float
    midpoint: float
    reason: str


def cheap_side(snap: MarketSnapshot) -> Side:
    return "YES" if snap.yes_price <= snap.no_price else "NO"


def midpoint_for(snap: MarketSnapshot, side: Side) -> float:
    return snap.yes_price if side == "YES" else snap.no_price


def book_for(snap: MarketSnapshot, side: Side) -> Optional[BookSide]:
    return snap.yes_book if side == "YES" else snap.no_book


def decide(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
) -> Tuple[List[Intent], str]:
    s = cfg.strategy
    total = snap.yes_price + snap.no_price

    # 1. STOP_LOSS — fires anytime, even off-window. Free up capital.
    sl_intent, sl_label = _maybe_stop_loss(cfg, snap, state)
    if sl_intent is not None:
        return [sl_intent], sl_label

    # 2. INVERTED_BOOK_EXIT — total > 1.02 and spot reversal against our side.
    inv_intent, inv_label = _maybe_inverted_exit(cfg, snap, momentum, state, total)
    if inv_intent is not None:
        return [inv_intent], inv_label

    # 3. ENDGAME_HEDGE
    if snap.remaining_sec < s.endgame_remaining_sec:
        eg_intent, eg_label = _endgame_hedge(cfg, snap, momentum, state)
        if eg_intent is not None:
            return [eg_intent], eg_label
        return [], eg_label

    # 4. SCOUT
    low = cheap_side(snap)
    low_price = midpoint_for(snap, low)
    sc_intent, sc_label = _maybe_scout(cfg, snap, momentum, state, low, low_price)
    if sc_intent is not None:
        return [sc_intent], sc_label

    # 5. LOCK_SPREAD
    arb_intent, arb_label = _maybe_lock_spread(cfg, snap, state, total)
    if arb_intent is not None:
        return [arb_intent], arb_label

    return [], sc_label or arb_label or "HOLD"


# ---------- Rule helpers ----------

def _maybe_stop_loss(cfg: Config, snap: MarketSnapshot, state: PortfolioState) -> Tuple[Optional[Intent], str]:
    s = cfg.strategy
    if s.stop_loss_pct <= 0:
        return None, ""
    for side in ("YES", "NO"):
        pos = state.get(side)  # type: ignore[arg-type]
        if not pos.is_open:
            continue
        mid = midpoint_for(snap, side)  # type: ignore[arg-type]
        # Sell would actually realize at the bid; use midpoint as the conservative
        # "current value" for the stop-loss trigger so we react before a thin
        # bid eats even more.
        loss_pct = pos.unrealized_pnl_pct(mid)
        if loss_pct <= -s.stop_loss_pct:
            book = book_for(snap, side)  # type: ignore[arg-type]
            best_bid = book.best_price - book.spread if (book and book.has_quotes and book.spread > 0) else None
            return Intent(
                action="SELL", side=side, size=pos.shares,  # type: ignore[arg-type]
                midpoint=mid,
                reason=f"stop-loss: {side} avg={pos.avg_price:.4f} mid={mid:.4f} ({loss_pct*100:+.2f}%)",
            ), f"STOP_LOSS_{side}"
    return None, ""


def _maybe_inverted_exit(
    cfg: Config, snap: MarketSnapshot, momentum: MomentumReading,
    state: PortfolioState, total: float,
) -> Tuple[Optional[Intent], str]:
    """If YES+NO > 1.02 (degraded book) AND spot has reversed since we entered,
    bail out of the side we're longer of. Better to take a small loss now than
    wait for endgame slippage.
    """
    s = cfg.strategy
    if s.inverted_total_threshold <= 0 or total <= s.inverted_total_threshold:
        return None, ""
    net = state.net_share_exposure()
    if abs(net) < 1e-6:
        return None, ""
    long_side: Side = "YES" if net > 0 else "NO"
    pos = state.get(long_side)
    if pos.entry_spot_price <= 0:
        return None, ""
    spot_change = (momentum.last_price - pos.entry_spot_price) / pos.entry_spot_price
    # Reversal: spot moved against the side we're long.
    reversed_against_us = (long_side == "YES" and spot_change < -s.inverted_spot_reversal_pct) or \
                          (long_side == "NO" and spot_change > s.inverted_spot_reversal_pct)
    if not reversed_against_us:
        return None, ""
    return Intent(
        action="SELL", side=long_side, size=pos.shares,
        midpoint=midpoint_for(snap, long_side),
        reason=f"inverted exit: total={total:.3f}, spot {spot_change*100:+.2f}% vs entry, side={long_side}",
    ), "INVERTED_EXIT"


def _maybe_scout(
    cfg: Config, snap: MarketSnapshot, momentum: MomentumReading,
    state: PortfolioState, low: Side, low_price: float,
) -> Tuple[Optional[Intent], str]:
    """Buy the cheap (small-probability) side as the first leg of the
    'underdog scout + lock spread' play.

    Momentum gate is configurable via strategy.scout_momentum_mode:
      - aligned: cheap side must agree with momentum (original; near-impossible
                 since underdog is cheap precisely because momentum is against it)
      - counter: enter only when momentum is AGAINST cheap side (overreaction)
      - none:    enter purely on price + time + book quality (matches the user's
                 'buy underdog whenever it's cheap, wait for rebound' strategy)
    """
    s = cfg.strategy
    if snap.remaining_sec <= s.scout_min_remaining_sec:
        return None, f"HOLD_scout_too_late({snap.remaining_sec:.0f}s)"
    if low_price > s.scout_max_price:
        return None, f"HOLD_scout_price_high({low}={low_price:.3f})"
    if state.has_position(low):
        return None, f"HOLD_scout_already_in({low})"

    mode = s.scout_momentum_mode
    if mode == "aligned":
        if not momentum_aligns(momentum, low, cfg.spot.alignment_threshold, require_multi_tf=True):
            return None, (
                f"HOLD_scout_momentum_against(short={momentum.short_return:+.4f},"
                f"long={momentum.long_return:+.4f})"
            )
    elif mode == "counter":
        opp = opposite(low)
        if not momentum_aligns(momentum, opp, cfg.spot.alignment_threshold, require_multi_tf=False):
            return None, (
                f"HOLD_scout_no_overreaction(short={momentum.short_return:+.4f})"
            )
    # mode == "none": skip momentum filter entirely

    book = book_for(snap, low)
    quality_reason = _book_quality_block(book, low_price, cfg)
    if quality_reason:
        return None, f"HOLD_scout_book_{quality_reason}"

    return Intent(
        action="BUY", side=low, size=s.scout_size_usd, midpoint=low_price,
        reason=(
            f"scout({mode}): cheap={low}@{low_price:.3f} "
            f"short={momentum.short_return:+.4f} long={momentum.long_return:+.4f} "
            f"rsi={momentum.rsi:.0f} vol={momentum.volume_ratio:.2f}x"
        ),
    ), "SCOUT"


def _maybe_lock_spread(
    cfg: Config, snap: MarketSnapshot, state: PortfolioState, total: float,
) -> Tuple[Optional[Intent], str]:
    """Lock the spread when COST BASIS (held_avg + current complement price)
    makes a +EV pair.

    The previous `current_total > arb_total_threshold` gate was wrong: it
    blocked exactly the cases the user wants.

    Concrete example from user spec:
      held YES @ 0.04, current YES=0.10 NO=0.85, current_total = 0.95
      cost basis = 0.04 + 0.85 = 0.89 → edge = 0.11 → +EV
      old code: would also have fired (total < 0.97)

      held YES @ 0.04, current YES=0.15 NO=0.85, current_total = 1.00
      cost basis = 0.04 + 0.85 = 0.89 → edge = 0.11 → +EV
      old code: WOULD HAVE BLOCKED (total > 0.97)  ← bug

    What matters is what we already paid for the first leg, not the current
    midpoint sum.
    """
    s = cfg.strategy
    if snap.remaining_sec <= s.arb_min_remaining_sec:
        return None, f"HOLD_arb_too_late({snap.remaining_sec:.0f}s)"

    held_side: Optional[Side] = None
    for side in ("YES", "NO"):
        if state.has_position(side):  # type: ignore[arg-type]
            held_side = side  # type: ignore[assignment]
            break
    if held_side is None:
        return None, "HOLD_arb_no_position"

    complement: Side = opposite(held_side)
    complement_price = midpoint_for(snap, complement)
    held_shares = state.get(held_side).shares
    held_avg = state.get(held_side).avg_price

    book = book_for(snap, complement)
    quality_reason = _book_quality_block(book, complement_price, cfg)
    if quality_reason:
        return None, f"HOLD_arb_book_{quality_reason}"

    complement_fill = (
        book.best_price if book and book.has_quotes
        else min(complement_price + cfg.fees.slippage_estimate, 0.999)
    )
    fees_per_share = (held_avg + complement_fill) * cfg.fees.per_side_fee_rate
    edge_per_pair = 1.0 - (held_avg + complement_fill + fees_per_share)
    if edge_per_pair < s.min_profit_after_fees:
        return None, (
            f"HOLD_arb_edge_thin(edge={edge_per_pair:.4f},"
            f"avg={held_avg:.3f},comp={complement_fill:.3f})"
        )

    needed_shares = max(0.0, held_shares - state.get(complement).shares)
    if needed_shares <= 1e-6:
        return None, "HOLD_arb_already_balanced"
    target_usd = needed_shares * complement_fill

    return Intent(
        action="BUY", side=complement, size=target_usd, midpoint=complement_price,
        reason=(
            f"lock spread: held_avg={held_avg:.3f} comp={complement_fill:.3f} "
            f"edge={edge_per_pair:.4f} buy {needed_shares:.2f} {complement}"
        ),
    ), "LOCK_SPREAD"


def _endgame_hedge(
    cfg: Config, snap: MarketSnapshot, momentum: MomentumReading, state: PortfolioState,
) -> Tuple[Optional[Intent], str]:
    """Last 3-4 minutes: if we only have ONE side (scout succeeded but
    lock_spread never fired), and spot confirms the market is going against
    us (single-sided trend), SELL the losing position to cut losses.

    This is the ONLY scenario where we sell. If we have BOTH sides locked,
    we're already guaranteed profit → just hold to settlement.
    """
    s = cfg.strategy

    # If we have both sides (locked pair), hold to settlement — guaranteed profit.
    if state.yes.is_open and state.no.is_open:
        return None, "HOLD_endgame_locked_pair"

    # If no position at all, nothing to do.
    if not state.has_position():
        return None, "HOLD_endgame_balanced"

    # We have only ONE side — check if we should cut losses.
    long_side: Side = "YES" if state.yes.is_open else "NO"
    pos = state.get(long_side)

    # Only sell if spot momentum DISAGREES with our side (confirms single-sided trend against us).
    if momentum_aligns(momentum, long_side, cfg.spot.alignment_threshold, require_multi_tf=False):
        return None, f"HOLD_endgame_spot_confirms({long_side})"

    # Sell the losing leg to cut losses.
    sell_price = midpoint_for(snap, long_side)
    return Intent(
        action="SELL", side=long_side, size=pos.shares,
        midpoint=sell_price,
        reason=(
            f"endgame stop: {long_side} unhedged, spot against us "
            f"(short={momentum.short_return:+.4f}), sell {pos.shares:.2f} sh @ {sell_price:.3f}"
        ),
    ), "ENDGAME_HEDGE"


def _book_quality_block(book: Optional[BookSide], midpoint: float, cfg: Config) -> str:
    """Empty string = OK, otherwise short reason for blocking."""
    if book is None:
        return ""  # endpoint flaky; don't block on missing data
    if not book.has_quotes:
        return "no_quotes"
    if cfg.execution_quality.max_spread > 0 and book.spread > cfg.execution_quality.max_spread:
        return f"spread_wide({book.spread:.3f})"
    if cfg.execution_quality.min_depth_usd > 0 and book.depth_within_1pct_usd < cfg.execution_quality.min_depth_usd:
        return f"depth_thin(${book.depth_within_1pct_usd:.0f})"
    # Sanity: best_price drifted too far from midpoint = stale/corrupted feed.
    # Use ABSOLUTE delta (not percentage) because at extreme prices like 4¢,
    # a normal 1¢ ask-vs-mid gap looks like 25% in relative terms.
    if abs(book.best_price - midpoint) > 0.05:
        return f"price_drift({book.best_price:.3f}_vs_mid{midpoint:.3f})"
    return ""
