"""Live executor: polls one symbol + timeframe, acts on signal at bar close.

By default uses BINANCE FUTURES TESTNET. Set USE_TESTNET=false to trade real.
Supports both price-only and enhanced (OI + on-chain) strategies.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import Config
from src.data.data_loader import TF_MS
from src.data.enhanced_loader import _to_raw_symbol, _TF_TO_OI_PERIOD
from src.data.oi_data import fetch_all_derivatives_data, fetch_funding_rate_history
from src.data.onchain_data import (
    compute_exchange_flow_proxy,
    compute_funding_signal,
    compute_oi_momentum,
    compute_whale_score,
)
from src.exchange.binance_client import BinanceFutures
from src.risk.risk_manager import RiskManager
from src.strategies.base import Strategy
from src.utils.logger import get_logger

# Strategies that need enhanced data
_ENHANCED_STRATEGIES = {"oi_composite"}


class LiveExecutor:
    def __init__(self, cfg: Config, client: BinanceFutures, strategy: Strategy, symbol: str, timeframe: str):
        self.cfg = cfg
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_enhanced = strategy.name in _ENHANCED_STRATEGIES
        self.log = get_logger(f"live.{symbol}", cfg.logs_dir / "live.log")
        self.risk = RiskManager(
            risk_per_trade=cfg.risk_per_trade,
            max_leverage=cfg.max_leverage,
            max_daily_drawdown=cfg.max_daily_drawdown,
            max_open_positions=cfg.max_open_positions,
        )
        self._last_processed_ts: pd.Timestamp | None = None

    def _fetch_recent(self, limit: int = 500) -> pd.DataFrame:
        df = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        if not self.use_enhanced or df.empty:
            return df
        # Enhance with OI + derivatives data
        return self._enrich_with_derivatives(df)

    def _enrich_with_derivatives(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Fetch live OI/derivatives data and merge into OHLCV."""
        raw_sym = _to_raw_symbol(self.symbol)
        oi_period = _TF_TO_OI_PERIOD.get(self.timeframe, "1h")
        try:
            deriv = fetch_all_derivatives_data(raw_sym, oi_period, limit=200)
        except Exception as e:
            self.log.warning("Failed to fetch derivatives data: %s", e)
            deriv = pd.DataFrame()
        try:
            funding = fetch_funding_rate_history(raw_sym, limit=100)
        except Exception as e:
            self.log.warning("Failed to fetch funding data: %s", e)
            funding = pd.DataFrame()

        df = ohlcv.copy()
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")

        if not deriv.empty and "ts" in deriv.columns:
            if deriv["ts"].dt.tz is None:
                deriv["ts"] = deriv["ts"].dt.tz_localize("UTC")
            deriv_cols = [c for c in deriv.columns if c != "ts" or c == "ts"]
            df = pd.merge_asof(
                df.sort_values("ts"),
                deriv.sort_values("ts"),
                on="ts",
                direction="backward",
            )

        if not funding.empty and "ts" in funding.columns:
            if funding["ts"].dt.tz is None:
                funding["ts"] = funding["ts"].dt.tz_localize("UTC")
            funding["funding_signal"] = compute_funding_signal(funding).values
            df = pd.merge_asof(
                df.sort_values("ts"),
                funding[["ts", "funding_rate", "funding_signal"]].sort_values("ts"),
                on="ts",
                direction="backward",
            )

        df["whale_score"] = compute_whale_score(df).values
        df["flow_proxy"] = compute_exchange_flow_proxy(df).values
        df["oi_momentum"] = compute_oi_momentum(df).values

        for col in ["oi", "ls_ratio", "taker_ratio", "funding_rate", "funding_signal",
                    "whale_score", "flow_proxy", "oi_momentum"]:
            if col not in df.columns:
                df[col] = np.nan

        return df.sort_values("ts").reset_index(drop=True)

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
