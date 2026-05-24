"""Three-rule decision logic, mapped 1:1 from the user pseudocode, plus
execution-quality guards (spread + depth) so we don't enter against thin
or wide books.

Pure function: consumes (config, snapshot, momentum, state) and returns
(intents, decision_label). The label is what gets logged to ticks.csv —
it explains *why* we did or did not act, which is the single most useful
signal for tuning the strategy by hand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Config
from .polymarket_client import BookSide, MarketSnapshot
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


def book_for(snap: MarketSnapshot, side: Side) -> Optional[BookSide]:
    return snap.yes_book if side == "YES" else snap.no_book


def decide(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
) -> Tuple[List[Intent], str]:
    """Return (intents, label). At most one buy per tick to keep behaviour obvious.

    Labels are short tags suitable for CSV: SCOUT, LOCK_SPREAD, ENDGAME_HEDGE,
    HOLD_*. Strategy never emits multiple intents on one tick.
    """
    intents: List[Intent] = []
    s = cfg.strategy
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
        hedge, label = _endgame_hedge(cfg, snap, momentum, state)
        if hedge is not None:
            intents.append(hedge)
        return intents, label

    # ---------- Rule 1: scout entry ----------
    scout, scout_label = _maybe_scout(cfg, snap, momentum, state, low, low_price)
    if scout is not None:
        intents.append(scout)
        return intents, scout_label

    # ---------- Rule 2: lock spread / complement ----------
    arb, arb_label = _maybe_lock_spread(cfg, snap, state, total)
    if arb is not None:
        intents.append(arb)
        return intents, arb_label

    # No action — return the most informative reason from whichever rule was closest.
    return intents, scout_label or arb_label or "HOLD"


# ---------- Rule helpers ----------

def _maybe_scout(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
    low: Side,
    low_price: float,
) -> Tuple[Optional[Intent], str]:
    s = cfg.strategy
    if snap.remaining_sec <= s.scout_min_remaining_sec:
        return None, f"HOLD_scout_too_late({snap.remaining_sec:.0f}s)"
    if low_price > s.scout_max_price:
        return None, f"HOLD_scout_price_high({low}={low_price:.3f})"
    if state.has_position(low):
        return None, f"HOLD_scout_already_in({low})"
    if not momentum_aligns(momentum, low, cfg.spot.alignment_threshold):
        return None, f"HOLD_scout_momentum_against({momentum.signed_return:+.5f})"

    # Execution-quality guard: skip if book is unusable.
    book = book_for(snap, low)
    quality_reason = _book_quality_block(book, low_price, cfg)
    if quality_reason:
        return None, f"HOLD_scout_book_{quality_reason}"

    return Intent(
        side=low,
        target_usd=s.scout_size_usd,
        midpoint=low_price,
        reason=f"scout: cheap={low}@{low_price:.3f} mom={momentum.signed_return:+.5f}",
    ), "SCOUT"


def _maybe_lock_spread(
    cfg: Config,
    snap: MarketSnapshot,
    state: PortfolioState,
    total: float,
) -> Tuple[Optional[Intent], str]:
    s = cfg.strategy
    if total > s.arb_total_threshold:
        return None, f"HOLD_arb_no_spread(total={total:.4f})"
    if snap.remaining_sec <= s.arb_min_remaining_sec:
        return None, f"HOLD_arb_too_late({snap.remaining_sec:.0f}s)"

    # Need an existing low-side position to "lock" against.
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

    # Use book best ask if we have one — that's the realistic fill, not midpoint.
    book = book_for(snap, complement)
    quality_reason = _book_quality_block(book, complement_price, cfg)
    if quality_reason:
        return None, f"HOLD_arb_book_{quality_reason}"
    complement_fill = book.best_price if book and book.has_quotes else min(
        complement_price + cfg.fees.slippage_estimate, 0.999
    )

    # Edge math: each share-pair (1 YES + 1 NO) pays $1, so:
    #   edge_per_pair = 1 - (held_avg + complement_fill + per-share fees)
    fees_per_share = (held_avg + complement_fill) * cfg.fees.per_side_fee_rate
    edge_per_pair = 1.0 - (held_avg + complement_fill + fees_per_share)
    if edge_per_pair < s.min_profit_after_fees:
        return None, f"HOLD_arb_edge_thin({edge_per_pair:.4f})"

    needed_shares = max(0.0, held_shares - state.get(complement).shares)
    if needed_shares <= 1e-6:
        return None, "HOLD_arb_already_balanced"
    target_usd = needed_shares * complement_fill

    return Intent(
        side=complement,
        target_usd=target_usd,
        midpoint=complement_price,
        reason=(
            f"lock spread: total={total:.4f} edge={edge_per_pair:.4f} "
            f"buy {needed_shares:.2f} {complement} to balance"
        ),
    ), "LOCK_SPREAD"


def _endgame_hedge(
    cfg: Config,
    snap: MarketSnapshot,
    momentum: MomentumReading,
    state: PortfolioState,
) -> Tuple[Optional[Intent], str]:
    """In the last 2 minutes, only act if we're unbalanced AND spot disagrees."""
    s = cfg.strategy
    if not state.is_unbalanced(s.unbalanced_threshold_usd):
        return None, "HOLD_endgame_balanced"

    net = state.net_share_exposure()
    long_side: Side = "YES" if net > 0 else "NO"
    if momentum_aligns(momentum, long_side, cfg.spot.alignment_threshold):
        return None, f"HOLD_endgame_spot_confirms({long_side})"

    hedge_side: Side = opposite(long_side)
    hedge_shares = abs(net) * s.endgame_hedge_ratio
    hedge_price = midpoint_for(snap, hedge_side)

    book = book_for(snap, hedge_side)
    quality_reason = _book_quality_block(book, hedge_price, cfg)
    if quality_reason:
        # Endgame is critical: prefer to hedge even on a thin book over carrying
        # full directional risk into settlement. We log the warning but proceed.
        log.warning("Endgame hedge proceeding despite book quality issue: %s", quality_reason)

    hedge_fill = book.best_price if book and book.has_quotes else min(
        hedge_price + cfg.fees.slippage_estimate, 0.999
    )
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
    ), "ENDGAME_HEDGE"


def _book_quality_block(book: Optional[BookSide], midpoint: float, cfg: Config) -> str:
    """Empty string = book is OK, otherwise short reason for blocking.

    Prefers safety: if the book endpoint failed (book is None), we proceed
    rather than block — otherwise a flaky API endpoint would freeze the bot.
    """
    if book is None:
        return ""
    if not book.has_quotes:
        return "no_quotes"
    if cfg.execution_quality.max_spread > 0 and book.spread > cfg.execution_quality.max_spread:
        return f"spread_wide({book.spread:.3f})"
    # Only check depth if we'd actually trade at meaningful size.
    min_depth = cfg.execution_quality.min_depth_usd
    if min_depth > 0 and book.depth_within_1pct_usd < min_depth:
        return f"depth_thin(${book.depth_within_1pct_usd:.0f})"
    # Sanity: if best ask is wildly off midpoint, don't trust it.
    if midpoint > 0 and abs(book.best_price - midpoint) / midpoint > 0.10:
        return f"price_drift({book.best_price:.3f}_vs_mid{midpoint:.3f})"
    return ""
