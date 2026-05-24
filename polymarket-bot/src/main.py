"""Bot entry point.

Adaptive polling loop:
  - Default interval = strategy.tick_interval_sec.
  - Inside the endgame window (remaining < strategy.endgame_remaining_sec)
    we drop to strategy.endgame_tick_interval_sec for faster reaction.
  - When a recurring market settles, we resolve the next one and continue,
    keeping the same in-memory portfolio state.

Run with:  python -m src.main
       or: ./run.sh         (recommended, handles venv + deps)

Flags:
  --preflight-only   Run health checks and exit. Useful before going live.
  --once             Trade exactly one market window then exit.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from .config import Config, load_config
from .executor import BaseExecutor, LiveExecutor, PaperExecutor
from .journal import Journal
from .polymarket_client import MarketInfo, PolymarketClient, PolymarketError
from .preflight import run_preflight
from .risk import RiskState
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


def _build_executor(cfg: Config, state: PortfolioState, journal: Journal) -> BaseExecutor:
    if cfg.execution.mode == "live":
        return LiveExecutor(state, cfg.fees.per_side_fee_rate, cfg.fees.slippage_estimate, journal)
    return PaperExecutor(
        state, cfg.fees.per_side_fee_rate, cfg.fees.slippage_estimate,
        journal=journal, mode_label="paper",
    )


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


def _print_banner(cfg: Config) -> None:
    log.info("=" * 64)
    log.info("polymarket-bot starting")
    log.info("  mode             : %s", cfg.execution.mode.upper())
    log.info("  starting equity  : $%.2f", cfg.execution.paper_starting_balance_usd)
    log.info("  daily loss limit : $%.2f (kill switch)", cfg.risk.daily_max_loss_usd)
    log.info("  max per trade    : $%.2f", cfg.risk.max_single_trade_usd)
    log.info("  recurring market : %s | template=%s",
             cfg.market.recurring.enabled, cfg.market.recurring.slug_template)
    log.info("=" * 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket short-cycle trading bot")
    p.add_argument("--preflight-only", action="store_true",
                   help="Run health checks and exit without trading")
    p.add_argument("--once", action="store_true",
                   help="Trade one market window and exit")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def run() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    _setup_logging(cfg.logging.level)
    _install_signal_handlers()
    _print_banner(cfg)

    log.info("Running preflight checks...")
    ok, lines = run_preflight(cfg)
    for ln in lines:
        log.info("%s", ln)
    if not ok:
        log.error("Preflight failed — fix the issues above and re-run.")
        return 2
    log.info("Preflight OK.")
    if args.preflight_only:
        return 0

    poly = PolymarketClient()
    spot = SpotClient(cfg.spot.symbol, cfg.spot.lookback_minutes, cfg.spot.alignment_threshold)
    state = PortfolioState(cash_usd=cfg.execution.paper_starting_balance_usd)
    journal = Journal()
    executor = _build_executor(cfg, state, journal)
    risk = RiskState(
        starting_equity_usd=cfg.execution.paper_starting_balance_usd,
        daily_max_loss_usd=cfg.risk.daily_max_loss_usd,
        min_cash_floor_usd=cfg.risk.min_cash_floor_usd,
        max_single_trade_usd=cfg.risk.max_single_trade_usd,
    )

    market: Optional[MarketInfo] = None
    markets_traded = 0
    last_summary_ts = 0.0

    while not _should_stop:
        # 1. Resolve / rotate the market.
        try:
            if market is None or market.closed:
                market = _resolve_market(cfg, poly)
                log.info("Active market: %s | %r", market.slug, market.question)
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

        # 3. Mark-to-market and update kill switch.
        equity = state.mark_to_market_equity(snap.yes_price, snap.no_price)
        risk.update(equity)

        # 4. Settled? Roll to next market.
        if snap.remaining_sec <= 0:
            log.info("Market %s ended. equity=$%.2f %s", market.slug, equity, state.summary())
            markets_traded += 1
            if args.once:
                log.info("--once: completed one market, exiting.")
                break
            market = None
            time.sleep(2)  # let Gamma list the next slug
            continue

        # 5. Run strategy → executor.
        intents, decision = decide(cfg, snap, mom, state)

        # Risk veto: block any new opening trade if kill switch / cash floor is hit.
        # Endgame hedges are always allowed (they reduce risk, not increase it).
        if intents and decision != "ENDGAME_HEDGE":
            allowed, why = risk.can_open_new_position(state.cash_usd)
            if not allowed:
                decision = f"BLOCKED_{why}"
                intents = []

        for intent in intents:
            capped_usd = risk.cap_trade_size(intent.target_usd)
            log.info("INTENT %s $%.2f @ ~%.4f | %s",
                     intent.side, capped_usd, intent.midpoint, intent.reason)
            try:
                executor.buy(
                    intent.side, capped_usd, intent.midpoint,
                    market_slug=market.slug, reason=intent.reason,
                )
            except NotImplementedError as e:
                log.error("Execution not wired: %s", e)
                return 2
            except Exception as e:
                log.exception("Executor error: %s", e)

        # 6. Journal this tick.
        journal.record_tick(
            market_slug=market.slug,
            yes_price=snap.yes_price,
            no_price=snap.no_price,
            remaining_sec=snap.remaining_sec,
            spot_last=mom.last_price,
            spot_signed_return=mom.signed_return,
            decision=decision,
            yes_shares=state.yes.shares,
            no_shares=state.no.shares,
            cash_usd=state.cash_usd,
        )

        # 7. Periodic summary (always, even if no trade).
        now = time.time()
        if now - last_summary_ts >= cfg.logging.summary_interval_sec:
            last_summary_ts = now
            log.info(
                "[summary] yes=%.4f no=%.4f total=%.4f remaining=%.0fs | equity=$%.2f | %s | last=%s",
                snap.yes_price, snap.no_price, snap.yes_price + snap.no_price,
                snap.remaining_sec, equity, state.summary(), decision,
            )

        # 8. Adaptive sleep.
        interval = (
            cfg.strategy.endgame_tick_interval_sec
            if snap.remaining_sec < cfg.strategy.endgame_remaining_sec
            else cfg.strategy.tick_interval_sec
        )
        interval = min(interval, max(1, int(snap.remaining_sec)))
        time.sleep(interval)

    log.info(
        "Stopped. markets_traded=%d | final equity=$%.2f | %s",
        markets_traded,
        state.mark_to_market_equity(0.5, 0.5),  # midpoint estimate when shut down
        state.summary(),
    )
    log.info("Trade log: logs/trades.csv | Tick log: logs/ticks.csv")
    return 0


if __name__ == "__main__":
    sys.exit(run())
