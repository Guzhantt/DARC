"""Event-driven backtester for single-symbol futures strategies.

Rules:
- Signal computed on bar t close -> entered at bar t+1 open.
- Intra-bar stop-loss: if bar high/low touches ``stop``, exit at stop price.
- Fees applied on every entry/exit.
- Position sizing: ``risk_per_trade`` fraction of equity divided by per-unit risk
  (distance from entry to stop). Capped by ``max_leverage``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

from .metrics import Metrics, compute_metrics


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    fee_rate: float = 0.0004       # taker fee ~0.04%
    slippage: float = 0.0002       # 2 bps one-way
    risk_per_trade: float = 0.01   # risk 1% equity per trade
    max_leverage: float = 3.0


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: pd.Series
    trades: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    if len(df) < 50:
        raise ValueError("Not enough bars for a meaningful backtest.")

    signals = strategy.generate(df)
    sides = signals["side"].to_numpy()
    stops = signals["stop"].to_numpy()

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ts = df["ts"].to_numpy()

    equity = cfg.initial_equity
    equity_series = np.empty(len(df))
    pos_side = "flat"        # 'long' | 'short' | 'flat'
    pos_qty = 0.0
    pos_entry = 0.0
    pos_stop = np.nan

    trades: list[dict] = []
    trade_pnls: list[float] = []

    def close_position(exit_price: float, reason: str, bar_idx: int) -> None:
        nonlocal equity, pos_side, pos_qty, pos_entry, pos_stop
        if pos_side == "flat" or pos_qty <= 0:
            return
        sign = 1 if pos_side == "long" else -1
        gross = sign * (exit_price - pos_entry) * pos_qty
        fee = (pos_entry + exit_price) * pos_qty * cfg.fee_rate
        pnl = gross - fee
        equity += pnl
        trades.append(
            {
                "entry_ts": pos_entry_ts,
                "exit_ts": ts[bar_idx],
                "side": pos_side,
                "entry": pos_entry,
                "exit": exit_price,
                "qty": pos_qty,
                "pnl": pnl,
                "reason": reason,
            }
        )
        trade_pnls.append(pnl)
        pos_side = "flat"
        pos_qty = 0.0
        pos_entry = 0.0
        pos_stop = np.nan

    pos_entry_ts = None
    for i in range(len(df)):
        # 1) Check stop-loss first using current bar's high/low
        if pos_side == "long" and not np.isnan(pos_stop) and lows[i] <= pos_stop:
            close_position(pos_stop, "stop", i)
        elif pos_side == "short" and not np.isnan(pos_stop) and highs[i] >= pos_stop:
            close_position(pos_stop, "stop", i)

        # 2) Act on signal generated at previous bar close (sides[i] is the TARGET for bar i)
        target = sides[i]
        target_stop = stops[i]
        if target != pos_side:
            # Close existing first
            if pos_side != "flat":
                exit_px = opens[i] * (1 - cfg.slippage) if pos_side == "long" else opens[i] * (1 + cfg.slippage)
                close_position(exit_px, "signal", i)
            # Open new
            if target in ("long", "short") and not np.isnan(target_stop):
                entry_px = opens[i] * (1 + cfg.slippage) if target == "long" else opens[i] * (1 - cfg.slippage)
                risk_per_unit = abs(entry_px - target_stop)
                if risk_per_unit > 0:
                    qty = (equity * cfg.risk_per_trade) / risk_per_unit
                    # Cap by max leverage
                    max_qty = (equity * cfg.max_leverage) / entry_px
                    qty = min(qty, max_qty)
                    if qty > 0:
                        pos_side = target
                        pos_qty = qty
                        pos_entry = entry_px
                        pos_stop = target_stop
                        pos_entry_ts = ts[i]

        # 3) Mark-to-market equity
        if pos_side == "long":
            unreal = (closes[i] - pos_entry) * pos_qty
        elif pos_side == "short":
            unreal = (pos_entry - closes[i]) * pos_qty
        else:
            unreal = 0.0
        equity_series[i] = equity + unreal

    # Close any open position at last close
    if pos_side != "flat":
        close_position(closes[-1], "end", len(df) - 1)
        equity_series[-1] = equity

    equity_curve = pd.Series(equity_series, index=df["ts"], name="equity")
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity_curve, trade_pnls, cfg.initial_equity)
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trades=trades_df)
