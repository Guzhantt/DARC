"""Performance metrics for a backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    sharpe: float
    max_drawdown: float
    final_equity: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(equity_curve: pd.Series, trade_pnls: list[float], initial_equity: float) -> Metrics:
    trades = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p < 0)
    win_rate = wins / trades if trades else 0.0
    avg_win = float(np.mean([p for p in trade_pnls if p > 0])) if wins else 0.0
    avg_loss = float(np.mean([p for p in trade_pnls if p < 0])) if losses else 0.0
    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = -sum(p for p in trade_pnls if p < 0)
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else initial_equity
    total_return = final_equity / initial_equity - 1.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Assume input is per-bar; annualization factor is approximate and left to caller's timeframe.
        sharpe = float(returns.mean() / returns.std() * np.sqrt(365 * 24))  # rough: hourly-ish
    else:
        sharpe = 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return Metrics(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        final_equity=final_equity,
    )
