"""Startup self-check. Runs once before the main loop.

Catches the 90% of "why isn't it working" cases up front:
  - Bad config values
  - Polymarket API unreachable / market not found
  - Binance (and US fallback) both blocked
  - Live mode requested but missing credentials

Prints clear OK / FAIL lines so a beginner can read the output.
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

from .config import Config
from .polymarket_client import PolymarketClient, PolymarketError
from .spot_client import SpotClient, SpotError

log = logging.getLogger(__name__)


def run_preflight(cfg: Config) -> Tuple[bool, List[str]]:
    """Return (all_ok, lines). Lines are printable summary regardless of outcome."""
    lines: List[str] = []
    all_ok = True

    # 1. Config sanity warnings (already validated, but warn on suspicious values).
    if cfg.strategy.scout_size_usd > cfg.execution.paper_starting_balance_usd / 5:
        lines.append(
            f"  WARN  scout_size_usd ${cfg.strategy.scout_size_usd:.2f} is more than 20% of "
            f"starting equity ${cfg.execution.paper_starting_balance_usd:.2f}. "
            "Consider reducing for safety."
        )
    if cfg.strategy.scout_max_price >= 0.5:
        lines.append(
            f"  WARN  scout_max_price={cfg.strategy.scout_max_price:.2f} is at or above 0.5 — "
            "you'll be entering at fair-coin prices, edge becomes thin."
        )
    if cfg.fees.slippage_estimate < 0.003:
        lines.append(
            f"  WARN  slippage_estimate={cfg.fees.slippage_estimate:.4f} looks optimistic "
            "for thin endgame books. 0.005-0.01 is more realistic."
        )

    # 2. Polymarket reachability + market resolution.
    poly = PolymarketClient()
    try:
        if cfg.market.recurring.enabled:
            m = poly.get_current_recurring_market(
                cfg.market.recurring.slug_template, cfg.market.recurring.period_sec
            )
        elif cfg.market.yes_token_id and cfg.market.no_token_id:
            from .polymarket_client import MarketInfo
            m = MarketInfo(
                slug=cfg.market.slug or "(pinned)", question="(pinned)",
                yes_token_id=cfg.market.yes_token_id,
                no_token_id=cfg.market.no_token_id,
                end_date_iso=None, closed=False, active=True,
            )
        else:
            m = poly.get_market_by_slug(cfg.market.slug)
        snap = poly.get_snapshot(m)
        lines.append(
            f"  OK    Polymarket: {m.slug} (YES={snap.yes_price:.3f}, NO={snap.no_price:.3f}, "
            f"{snap.remaining_sec/60:.1f} min remaining)"
        )
    except PolymarketError as e:
        all_ok = False
        lines.append(f"  FAIL  Polymarket: {e}")

    # 3. Spot momentum reachability.
    spot = SpotClient(cfg.spot.symbol, cfg.spot.lookback_minutes, cfg.spot.alignment_threshold)
    try:
        m_reading = spot.fetch()
        lines.append(
            f"  OK    Spot ({cfg.spot.symbol}): last={m_reading.last_price:.2f}, "
            f"{cfg.spot.lookback_minutes}m return = {m_reading.signed_return*100:+.3f}%"
        )
    except SpotError as e:
        all_ok = False
        lines.append(
            f"  FAIL  Spot: {e}\n"
            "        If you're in mainland China, you may need a VPN for binance.com.\n"
            "        If you're in the US, the binance.us fallback should normally work."
        )

    # 4. Live mode credential check.
    if cfg.execution.mode == "live":
        missing = [
            k for k in ("POLYGON_PRIVATE_KEY", "CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASSPHRASE")
            if not os.environ.get(k)
        ]
        if missing:
            all_ok = False
            lines.append(
                f"  FAIL  Live mode requires env vars, missing: {', '.join(missing)}\n"
                "        Copy .env.example to .env and fill in the values."
            )
        else:
            lines.append("  OK    Live credentials present")
        # Live executor itself is still a stub — warn loudly.
        lines.append(
            "  WARN  LiveExecutor is currently a stub (raises NotImplementedError on first buy)."
        )

    return all_ok, lines
