"""Bot entry point.

Adaptive polling:
  - Outside session window: sleep `session.off_hours_poll_sec` and re-check.
  - In session, normal market: `strategy.tick_interval_sec` (default 30s).
  - In session, endgame:       `strategy.endgame_tick_interval_sec` (default 10s).

Run with:  ./run.sh
       or: python -m src.main

Flags:
  --preflight-only  Run health checks and exit.
  --once            Trade exactly one market window then exit.
  --config PATH     Use a different config file.
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
from .session import SessionWindow, from_config as session_from_config
from .spot_client import SpotClient, SpotError, momentum_score
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
    m = cfg.market
    if m.yes_token_id and m.no_token_id:
        return MarketInfo(
            slug=m.slug or "(pinned)", question="(pinned token IDs)",
            yes_token_id=m.yes_token_id, no_token_id=m.no_token_id,
            end_date_iso=None, closed=False, active=True,
        )
    if m.recurring.enabled:
        return client.get_current_recurring_market(m.recurring.slug_templates, m.recurring.period_sec)
    return client.get_market_by_slug(m.slug)


def _print_banner(cfg: Config) -> None:
    log.info("=" * 64)
    log.info("polymarket-bot starting")
    log.info("  mode             : %s", cfg.execution.mode.upper())
    log.info("  starting equity  : $%.2f", cfg.execution.paper_starting_balance_usd)
    if cfg.risk.daily_max_loss_pct > 0:
        log.info("  daily loss limit : %.1f%% of equity (= $%.2f) — kill switch",
                 cfg.risk.daily_max_loss_pct * 100,
                 cfg.execution.paper_starting_balance_usd * cfg.risk.daily_max_loss_pct)
    if cfg.risk.daily_max_loss_usd > 0:
        log.info("                     also capped at $%.2f", cfg.risk.daily_max_loss_usd)
    log.info("  max per trade    : $%.2f", cfg.risk.max_single_trade_usd)
    log.info("  stop-loss        : %.1f%% per side", cfg.strategy.stop_loss_pct * 100)
    log.info("  recurring market : %s", cfg.market.recurring.slug_templates)
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
    spot = SpotClient(
        cfg.spot.symbol,
        short_lookback_minutes=cfg.spot.short_lookback_minutes,
        long_lookback_minutes=cfg.spot.long_lookback_minutes,
        alignment_threshold=cfg.spot.alignment_threshold,
        rsi_period=cfg.spot.rsi_period,
        volume_lookback_minutes=cfg.spot.volume_lookback_minutes,
    )
    state = PortfolioState(cash_usd=cfg.execution.paper_starting_balance_usd)
    journal = Journal()
    executor = _build_executor(cfg, state, journal)
    risk = RiskState(
        starting_equity_usd=cfg.execution.paper_starting_balance_usd,
        daily_max_loss_pct=cfg.risk.daily_max_loss_pct,
        daily_max_loss_usd=cfg.risk.daily_max_loss_usd,
        min_cash_floor_usd=cfg.risk.min_cash_floor_usd,
        max_single_trade_usd=cfg.risk.max_single_trade_usd,
    )
    session_window: SessionWindow = session_from_config(
        cfg.session.enabled_hours_utc_start,
        cfg.session.enabled_hours_utc_end,
        cfg.session.enabled_weekdays,
    )

    market: Optional[MarketInfo] = None
    markets_traded = 0
    last_summary_ts = 0.0

    while not _should_stop:
        # 0. Session gate — sleep idly outside hours unless we have an open
        #    position that still needs management (stop-loss / endgame).
        if not session_window.is_open() and not state.has_position():
            log.info("[off-hours] no positions, sleeping %ds", cfg.session.off_hours_poll_sec)
            time.sleep(cfg.session.off_hours_poll_sec)
            continue

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

        # 3. Mark-to-market & risk update.
        equity = state.mark_to_market_equity(snap.yes_price, snap.no_price)
        risk.update(equity)

        # 4. Settled?
        if snap.remaining_sec <= 0:
            log.info("Market %s ended. equity=$%.2f %s", market.slug, equity, state.summary())
            markets_traded += 1
            # Reset position for next market. Each 15m market is independent —
            # settlement pays out automatically, so we start fresh.
            state.yes.shares = 0.0
            state.yes.avg_price = 0.0
            state.yes.cost_usd = 0.0
            state.yes.entry_spot_price = 0.0
            state.no.shares = 0.0
            state.no.avg_price = 0.0
            state.no.cost_usd = 0.0
            state.no.entry_spot_price = 0.0
            # Simulate settlement payout: winning side pays $1/share.
            # In paper mode we approximate by resetting cash to current equity
            # (which already includes the mark-to-market value of shares).
            state.cash_usd = equity
            log.info("Position reset for next market. cash=$%.2f", state.cash_usd)
            if args.once:
                log.info("--once: completed one market, exiting.")
                break
            market = None
            # Wait for Polymarket to list the next 15m market slug.
            # Too short (2s) caused skipping a cycle; 10s is safe.
            time.sleep(10)
            continue

        # 5. Decide.
        intents, decision = decide(cfg, snap, mom, state)

        # 6. Risk veto: block opening trades if kill switch / cash floor is hit.
        #    Closing trades (SELL or ENDGAME_HEDGE) are always allowed — they reduce risk.
        if intents:
            opening = [i for i in intents if i.action == "BUY" and decision not in ("ENDGAME_HEDGE",)]
            if opening:
                allowed, why = risk.can_open_new_position(state.cash_usd)
                if not allowed:
                    decision = f"BLOCKED_{why}"
                    intents = [i for i in intents if i not in opening]

        # 7. Off-session: refuse to OPEN new positions; SELL / ENDGAME still allowed.
        if not session_window.is_open():
            opening = [i for i in intents if i.action == "BUY" and decision != "ENDGAME_HEDGE"]
            if opening:
                decision = "BLOCKED_off_hours"
                intents = [i for i in intents if i not in opening]

        # 8. Execute.
        for intent in intents:
            try:
                if intent.action == "BUY":
                    # Risk cap applies only to OPENING trades (SCOUT). LOCK_SPREAD
                    # and ENDGAME_HEDGE are balancing trades that REDUCE directional
                    # risk by completing a near-arbitrage pair — capping them at
                    # max_single_trade_usd would leave us with a partially hedged
                    # position that's actually riskier than a full hedge.
                    if decision in ("LOCK_SPREAD", "ENDGAME_HEDGE"):
                        capped_usd = intent.size
                    else:
                        capped_usd = risk.cap_trade_size(intent.size)
                    log.info("INTENT BUY  %s $%.2f @ ~%.4f | %s",
                             intent.side, capped_usd, intent.midpoint, intent.reason)
                    book = snap.yes_book if intent.side == "YES" else snap.no_book
                    best_ask = book.best_price if book and book.has_quotes else None
                    executor.buy(
                        intent.side, capped_usd, intent.midpoint,
                        market_slug=market.slug, reason=intent.reason,
                        spot_price=mom.last_price, best_ask=best_ask,
                    )
                else:  # SELL
                    log.info("INTENT SELL %s %.3f sh @ ~%.4f | %s",
                             intent.side, intent.size, intent.midpoint, intent.reason)
                    book = snap.yes_book if intent.side == "YES" else snap.no_book
                    best_bid = (book.best_price - book.spread) if (book and book.has_quotes and book.spread > 0) else None
                    executor.sell(
                        intent.side, intent.size, intent.midpoint,
                        market_slug=market.slug, reason=intent.reason,
                        best_bid=best_bid,
                    )
            except NotImplementedError as e:
                log.error("Execution not wired: %s", e)
                return 2
            except Exception as e:
                log.exception("Executor error: %s", e)

        # 9. Journal this tick.
        total = snap.yes_price + snap.no_price
        journal.record_tick(
            market_slug=market.slug,
            yes_price=snap.yes_price, no_price=snap.no_price,
            remaining_sec=snap.remaining_sec,
            spot_last=mom.last_price,
            spot_short_return=mom.short_return,
            spot_long_return=mom.long_return,
            spot_rsi=mom.rsi,
            spot_volume_ratio=mom.volume_ratio,
            momentum_score=momentum_score(mom, total),
            decision=decision,
            yes_shares=state.yes.shares, no_shares=state.no.shares,
            cash_usd=state.cash_usd, equity_usd=equity,
        )

        # 10. Periodic summary.
        now = time.time()
        if now - last_summary_ts >= cfg.logging.summary_interval_sec:
            last_summary_ts = now
            log.info(
                "[summary] %s | yes=%.4f no=%.4f total=%.4f rem=%.0fs | "
                "spot=%.0f short=%+.3f%% long=%+.3f%% rsi=%.0f vol=%.2fx | equity=$%.2f | %s | %s",
                market.slug, snap.yes_price, snap.no_price, total, snap.remaining_sec,
                mom.last_price, mom.short_return * 100, mom.long_return * 100,
                mom.rsi, mom.volume_ratio, equity, state.summary(), decision,
            )

        # 11. Adaptive sleep.
        interval = (
            cfg.strategy.endgame_tick_interval_sec
            if snap.remaining_sec < cfg.strategy.endgame_remaining_sec
            else cfg.strategy.tick_interval_sec
        )
        interval = min(interval, max(1, int(snap.remaining_sec)))
        time.sleep(interval)

    log.info(
        "Stopped. markets_traded=%d | %s",
        markets_traded, state.summary(),
    )
    log.info("Trade log: logs/trades.csv | Tick log: logs/ticks.csv")
    return 0


if __name__ == "__main__":
    sys.exit(run())
