"""Bot entry point.

Adaptive polling loop:
  - Default interval = strategy.tick_interval_sec.
  - Inside the endgame window (remaining < strategy.endgame_remaining_sec)
    we drop to strategy.endgame_tick_interval_sec for faster reaction.
  - When a recurring market settles, we resolve the next one and continue,
    keeping the same in-memory portfolio state.

Run with:  python -m src.main
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional

from .config import Config, load_config
from .executor import BaseExecutor, LiveExecutor, PaperExecutor
from .polymarket_client import MarketInfo, PolymarketClient, PolymarketError
from .spot_client import SpotClient, SpotError
from .state import PortfolioState
from .strategy import decide

log = logging.getLogger("polymarket_bot")


_should_stop = False


def _install_signal_handlers() -> None:
    def _handle(signum, _frame):
        global _should_stop
        log.info("Caught signal %s — shutting down after this tick", signum)
        _should_stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_executor(cfg: Config, state: PortfolioState) -> BaseExecutor:
    if cfg.execution.mode == "live":
        return LiveExecutor(state, cfg.fees.per_side_fee_rate, cfg.fees.slippage_estimate)
    return PaperExecutor(state, cfg.fees.per_side_fee_rate, cfg.fees.slippage_estimate)


def _resolve_market(cfg: Config, client: PolymarketClient) -> MarketInfo:
    """Resolve current market according to config. Pinned token IDs win if set."""
    m = cfg.market
    if m.yes_token_id and m.no_token_id:
        return MarketInfo(
            slug=m.slug or "(pinned)",
            question="(pinned token IDs)",
            yes_token_id=m.yes_token_id,
            no_token_id=m.no_token_id,
            end_date_iso=None,
            closed=False,
            active=True,
        )
    if m.recurring.enabled:
        return client.get_current_recurring_market(m.recurring.slug_template, m.recurring.period_sec)
    return client.get_market_by_slug(m.slug)


def run() -> int:
    cfg = load_config("config.yaml")
    _setup_logging(cfg.logging.level)
    _install_signal_handlers()

    log.info("Starting polymarket-bot in %s mode", cfg.execution.mode.upper())
    log.info("Recurring: %s | template=%r period=%ds",
             cfg.market.recurring.enabled, cfg.market.recurring.slug_template, cfg.market.recurring.period_sec)

    poly = PolymarketClient()
    spot = SpotClient(cfg.spot.symbol, cfg.spot.lookback_minutes, cfg.spot.alignment_threshold)
    state = PortfolioState(cash_usd=cfg.execution.paper_starting_balance_usd)
    executor = _build_executor(cfg, state)

    market: Optional[MarketInfo] = None

    while not _should_stop:
        # 1. Resolve / rotate the market.
        try:
            if market is None or market.closed:
                market = _resolve_market(cfg, poly)
                log.info("Active market: slug=%s | question=%r", market.slug, market.question)
        except PolymarketError as e:
            log.error("Cannot resolve market: %s — retrying in %ds", e, cfg.strategy.tick_interval_sec)
            time.sleep(cfg.strategy.tick_interval_sec)
            continue

        # 2. Snapshot prices + spot momentum.
        try:
            snap = poly.get_snapshot(market)
        except PolymarketError as e:
            log.warning("Price snapshot failed: %s", e)
            time.sleep(cfg.strategy.tick_interval_sec)
            continue

        try:
            mom = spot.fetch()
        except SpotError as e:
            log.warning("Spot fetch failed: %s", e)
            time.sleep(cfg.strategy.tick_interval_sec)
            continue

        # 3. If the market settled while we slept, log P/L stub and rotate next loop.
        if snap.remaining_sec <= 0:
            log.info("Market %s ended (remaining=%.1fs). %s", market.slug, snap.remaining_sec, state.summary())
            market = None
            # Brief pause to let Gamma list the next slug.
            time.sleep(2)
            continue

        # 4. Run strategy → executor.
        intents = decide(cfg, snap, mom, state)
        for intent in intents:
            log.info("INTENT %s $%.2f @ ~%.4f | %s",
                     intent.side, intent.target_usd, intent.midpoint, intent.reason)
            try:
                executor.buy(intent.side, intent.target_usd, intent.midpoint)
            except NotImplementedError as e:
                log.error("Execution not wired: %s", e)
                return 2
            except Exception as e:
                log.exception("Executor error: %s", e)

        log.info(
            "tick: yes=%.4f no=%.4f total=%.4f remaining=%.0fs | %s",
            snap.yes_price, snap.no_price, snap.yes_price + snap.no_price,
            snap.remaining_sec, state.summary(),
        )

        # 5. Adaptive sleep.
        interval = (
            cfg.strategy.endgame_tick_interval_sec
            if snap.remaining_sec < cfg.strategy.endgame_remaining_sec
            else cfg.strategy.tick_interval_sec
        )
        # Don't oversleep past market end.
        interval = min(interval, max(1, int(snap.remaining_sec)))
        time.sleep(interval)

    log.info("Stopped. Final state: %s", state.summary())
    return 0


if __name__ == "__main__":
    sys.exit(run())
