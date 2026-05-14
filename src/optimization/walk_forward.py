"""Walk-Forward Parameter Optimization.

防止过拟合的标准方法:
- 把历史数据分成 N 个窗口
- 每个窗口: 用前 train_pct 比例数据找最优参数, 用剩余数据验证
- 最终输出: 训练集表现 vs 测试集表现的对比 (差距大 = 过拟合)

我们用最简单的 1 折分割: 前 12 个月训练, 后 6 个月测试.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.strategies.base import Strategy


@dataclass
class WalkForwardResult:
    """Walk-forward 优化结果."""
    symbol: str
    best_params: dict
    train_metrics: dict
    test_metrics: dict
    overfitting_score: float  # 训练-测试 Sharpe 差距; 越小越稳健

    def is_robust(self, max_overfit: float = 2.0) -> bool:
        """训练/测试 Sharpe 差距 < max_overfit 视为稳健."""
        return abs(self.overfitting_score) < max_overfit


def grid_search(
    df: pd.DataFrame,
    strategy_factory: Callable[[dict], Strategy],
    param_grid: dict[str, list],
    cfg: BacktestConfig | None = None,
    objective: str = "sharpe",
    min_trades: int = 3,
) -> tuple[dict, dict]:
    """网格搜索最优参数.

    Args:
        df: OHLCV 数据
        strategy_factory: 接受 dict 参数返回 Strategy 实例的函数
        param_grid: {param_name: [values_to_try]}
        cfg: 回测配置
        objective: 优化目标 ('sharpe', 'total_return', 'profit_factor', 'sortino')
        min_trades: 最少交易数 (避免样本不足)

    Returns:
        (best_params, best_metrics_dict)
    """
    cfg = cfg or BacktestConfig(initial_equity=10000, risk_per_trade=0.05, max_leverage=1)

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_score = -np.inf
    best_params: dict = {}
    best_metrics: dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            s = strategy_factory(params)
            r = run_backtest(df, s, cfg)
            m = r.metrics
            if m.trades < min_trades:
                continue

            if objective == "sharpe":
                score = m.sharpe
            elif objective == "total_return":
                score = m.total_return
            elif objective == "profit_factor":
                score = m.profit_factor if m.profit_factor != float("inf") else 999
            elif objective == "calmar":
                score = m.total_return / abs(m.max_drawdown) if m.max_drawdown < 0 else 0
            else:
                score = m.sharpe

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = m.as_dict()
        except Exception:
            continue

    return best_params, best_metrics


def walk_forward_optimize(
    df: pd.DataFrame,
    symbol: str,
    strategy_factory: Callable[[dict], Strategy],
    param_grid: dict[str, list],
    train_ratio: float = 0.66,
    objective: str = "sharpe",
    cfg: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Walk-Forward 单折优化.

    Args:
        df: 完整 OHLCV 数据 (按时间排序)
        symbol: 标识用
        strategy_factory: dict -> Strategy
        param_grid: 参数搜索空间
        train_ratio: 训练集比例 (默认 0.66 ≈ 12个月/18个月)
        objective: 优化目标
        cfg: 回测配置

    Returns:
        WalkForwardResult
    """
    cfg = cfg or BacktestConfig(initial_equity=10000, risk_per_trade=0.05, max_leverage=1)
    n = len(df)
    split = int(n * train_ratio)

    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    # 训练: 在训练集上找最优参数
    best_params, train_metrics = grid_search(
        train_df, strategy_factory, param_grid, cfg, objective
    )

    if not best_params:
        return WalkForwardResult(
            symbol=symbol,
            best_params={},
            train_metrics={},
            test_metrics={},
            overfitting_score=float("inf"),
        )

    # 测试: 用最优参数跑测试集 (out-of-sample)
    s = strategy_factory(best_params)
    r = run_backtest(test_df, s, cfg)
    test_metrics = r.metrics.as_dict()

    # 过拟合评估: 训练 Sharpe vs 测试 Sharpe 差距
    train_sharpe = train_metrics.get("sharpe", 0)
    test_sharpe = test_metrics.get("sharpe", 0)
    overfitting_score = train_sharpe - test_sharpe

    return WalkForwardResult(
        symbol=symbol,
        best_params=best_params,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        overfitting_score=overfitting_score,
    )


def format_wf_report(result: WalkForwardResult) -> str:
    """格式化 walk-forward 结果为可读报告."""
    lines = []
    lines.append(f"=== {result.symbol} Walk-Forward 报告 ===")
    if not result.best_params:
        lines.append("  数据不足或无有效参数")
        return "\n".join(lines)

    lines.append(f"  最优参数: {result.best_params}")
    lines.append(f"")
    lines.append(f"  {'指标':<15} {'训练集 (前12月)':>18} {'测试集 (后6月)':>18}")
    lines.append(f"  {'-' * 55}")
    for key, label in [
        ("trades", "交易次数"),
        ("win_rate", "胜率"),
        ("profit_factor", "盈亏比"),
        ("total_return", "总收益"),
        ("max_drawdown", "最大回撤"),
        ("sharpe", "Sharpe"),
    ]:
        tr = result.train_metrics.get(key, 0)
        te = result.test_metrics.get(key, 0)
        if key in ("win_rate", "total_return", "max_drawdown"):
            lines.append(f"  {label:<15} {tr*100:>16.1f}% {te*100:>16.1f}%")
        elif key in ("trades",):
            lines.append(f"  {label:<15} {int(tr):>17d} {int(te):>17d}")
        else:
            lines.append(f"  {label:<15} {tr:>17.2f} {te:>17.2f}")

    lines.append(f"")
    if result.is_robust():
        lines.append(f"  ✅ 稳健 (Sharpe 差距 {result.overfitting_score:+.2f})")
    else:
        lines.append(f"  ⚠️  可能过拟合 (Sharpe 差距 {result.overfitting_score:+.2f})")

    return "\n".join(lines)
