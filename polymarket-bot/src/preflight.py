"""Startup self-check. Runs once before the main loop."""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

from .config import Config
from .polymarket_client import PolymarketClient, PolymarketError
from .session import from_config as session_from_config
from .spot_client import SpotClient, SpotError

log = logging.getLogger(__name__)


def run_preflight(cfg: Config) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    all_ok = True

    # 1. Session window summary.
    try:
        win = session_from_config(
            cfg.session.enabled_hours_utc_start,
            cfg.session.enabled_hours_utc_end,
            cfg.session.enabled_weekdays,
        )
        status = "open" if win.is_open() else "closed"
        lines.append(f"  OK    Session: {win.describe()} (currently {status})")
    except ValueError as e:
        all_ok = False
        lines.append(f"  FAIL  Session config: {e}")

    # 2. Config sanity warnings.
    if cfg.strategy.scout_size_usd > cfg.execution.paper_starting_balance_usd / 5:
        lines.append(
            f"  WARN  scout_size_usd ${cfg.strategy.scout_size_usd:.2f} is more than 20% of "
            f"starting equity ${cfg.execution.paper_starting_balance_usd:.2f}."
        )
    if cfg.strategy.scout_max_price >= 0.50:
        lines.append(
            f"  WARN  scout_max_price={cfg.strategy.scout_max_price:.2f} is at fair-coin level — "
            "edge becomes thin near 0.5."
        )

    # 3. Polymarket reachability + market resolution.
    poly = PolymarketClient()
    try:
        if cfg.market.recurring.enabled:
            m = poly.get_current_recurring_market(
                cfg.market.recurring.slug_templates, cfg.market.recurring.period_sec
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

    # 4. Spot momentum reachability.
    spot = SpotClient(
        cfg.spot.symbol,
        short_lookback_minutes=cfg.spot.short_lookback_minutes,
        long_lookback_minutes=cfg.spot.long_lookback_minutes,
        alignment_threshold=cfg.spot.alignment_threshold,
        rsi_period=cfg.spot.rsi_period,
        volume_lookback_minutes=cfg.spot.volume_lookback_minutes,
    )
    try:
        m_reading = spot.fetch()
        lines.append(
            f"  OK    Spot ({cfg.spot.symbol}): last={m_reading.last_price:.2f}, "
            f"{cfg.spot.short_lookback_minutes}m={m_reading.short_return*100:+.3f}%, "
            f"{cfg.spot.long_lookback_minutes}m={m_reading.long_return*100:+.3f}%, "
            f"RSI={m_reading.rsi:.0f}, vol={m_reading.volume_ratio:.2f}x"
        )
    except SpotError as e:
        all_ok = False
        lines.append(
            f"  FAIL  Spot: {e}\n"
            "        binance.com is geo-blocked from US/datacenter IPs; binance.us fallback should work."
        )

    # 5. Risk summary.
    pct = cfg.risk.daily_max_loss_pct
    usd = cfg.risk.daily_max_loss_usd
    parts = []
    if pct > 0:
        parts.append(f"{pct*100:.1f}% of starting equity")
    if usd > 0:
        parts.append(f"${usd:.2f}")
    lines.append(f"  OK    Risk: daily loss limit = stricter of {' and '.join(parts) if parts else '(none)'}; "
                 f"max single trade = ${cfg.risk.max_single_trade_usd:.2f}")

    # 6. Live mode credential check.
    if cfg.execution.mode == "live":
        missing = [
            k for k in ("POLYGON_PRIVATE_KEY", "CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASSPHRASE")
            if not os.environ.get(k)
        ]
        if missing:
            all_ok = False
            lines.append(f"  FAIL  Live mode requires env vars, missing: {', '.join(missing)}")
        else:
            lines.append("  OK    Live credentials present")
        lines.append("  WARN  LiveExecutor is currently a stub.")

    return all_ok, lines
