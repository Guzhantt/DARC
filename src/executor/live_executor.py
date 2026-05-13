"""Live executor: polls one symbol + timeframe, acts on signal at bar close.

By default uses BINANCE FUTURES TESTNET. Set USE_TESTNET=false to trade real.
"""
from __future__ import annotations

import time

import pandas as pd

from src.config import Config
from src.data.data_loader import TF_MS
from src.exchange.binance_client import BinanceFutures
from src.risk.risk_manager import RiskManager
from src.strategies.base import Strategy
from src.utils.logger import get_logger


class LiveExecutor:
    def __init__(self, cfg: Config, client: BinanceFutures, strategy: Strategy, symbol: str, timeframe: str):
        self.cfg = cfg
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.log = get_logger(f"live.{symbol}", cfg.logs_dir / "live.log")
        self.risk = RiskManager(
            risk_per_trade=cfg.risk_per_trade,
            max_leverage=cfg.max_leverage,
            max_daily_drawdown=cfg.max_daily_drawdown,
            max_open_positions=cfg.max_open_positions,
        )
        self._last_processed_ts: pd.Timestamp | None = None

    def _fetch_recent(self, limit: int = 500) -> pd.DataFrame:
        return self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)

    def _current_side(self) -> str:
        pos = self.client.fetch_position(self.symbol)
        if not pos:
            return "flat"
        contracts = float(pos.get("contracts") or 0)
        if contracts == 0:
            return "flat"
        return "long" if (pos.get("side") or "").lower() == "long" else "short"

    def _close_position(self) -> None:
        pos = self.client.fetch_position(self.symbol)
        if not pos:
            return
        contracts = float(pos.get("contracts") or 0)
        if contracts == 0:
            return
        side = "sell" if (pos.get("side") or "").lower() == "long" else "buy"
        self.log.info("Closing position: %s %s", side, contracts)
        self.client.cancel_all(self.symbol)
        self.client.create_market_order(self.symbol, side, contracts, reduce_only=True)

    def _open_position(self, side: str, qty: float, stop_price: float) -> None:
        order_side = "buy" if side == "long" else "sell"
        self.log.info("Opening %s %s qty=%s stop=%s", side, self.symbol, qty, stop_price)
        self.client.create_market_order(self.symbol, order_side, qty)
        stop_side = "sell" if side == "long" else "buy"
        self.client.create_stop_market(self.symbol, stop_side, qty, stop_price)

    def step(self) -> None:
        df = self._fetch_recent()
        if df.empty:
            return
        # Use only CLOSED bars (drop the last, which may be forming)
        df = df.iloc[:-1].reset_index(drop=True)
        if len(df) < 60:
            return
        last_ts = df["ts"].iloc[-1]
        if self._last_processed_ts is not None and last_ts <= self._last_processed_ts:
            return
        self._last_processed_ts = last_ts

        sig = self.strategy.generate(df).iloc[-1]
        target, stop_price = sig["side"], sig["stop"]

        equity = self.client.fetch_balance_usdt()
        self.risk.on_new_day(equity)
        if self.risk.check_daily_drawdown(equity):
            self.log.warning("Daily drawdown exceeded. Halting until tomorrow.")
            return

        current = self._current_side()
        last_close = float(df["close"].iloc[-1])
        self.log.info(
            "ts=%s close=%.4f current=%s target=%s stop=%.4f equity=%.2f",
            last_ts, last_close, current, target, stop_price if stop_price == stop_price else -1, equity,
        )

        if target == current:
            return

        if current != "flat":
            self._close_position()
            time.sleep(1)

        if target in ("long", "short") and stop_price == stop_price:  # not NaN
            qty = self.risk.position_size(equity, last_close, stop_price)
            qty = float(self.client.exchange.amount_to_precision(self.symbol, qty))
            if qty <= 0:
                self.log.warning("Position size rounded to 0, skipping entry.")
                return
            self.client.set_leverage(self.symbol, self.cfg.max_leverage)
            self._open_position(target, qty, stop_price)

    def run_forever(self, poll_sec: int = 30) -> None:
        tf_ms = TF_MS[self.timeframe]
        self.log.info(
            "Starting live loop: symbol=%s tf=%s testnet=%s",
            self.symbol, self.timeframe, self.cfg.use_testnet,
        )
        while True:
            try:
                self.step()
            except Exception as e:
                self.log.exception("step error: %s", e)
            # Sleep until approximately next bar close + small buffer
            now_ms = self.client.exchange.milliseconds()
            next_close = ((now_ms // tf_ms) + 1) * tf_ms
            sleep_s = max(poll_sec, (next_close - now_ms) // 1000 + 5)
            time.sleep(sleep_s)
