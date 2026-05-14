"""多币种组合回测器.

核心逻辑:
- 同时跑 N 个币的策略, 共享一个总资金池
- 每个币独立判断信号, 但仓位大小受组合层级风控约束
- 组合层级风控:
  - 最大同时持仓数 (max_positions)
  - 单币最大仓位占比 (max_pct_per_symbol)
  - 总杠杆上限 (max_total_leverage)
  - 每日组合最大回撤熔断 (daily_dd_halt)

仓位分配模式:
- equal: 等额分配 (每个开仓的币用 1/N 资金)
- volatility: 反波动率分配 (波动小的多分, 波动大的少分)
- equity: 按胜率/Sharpe动态分配 (训练期表现好的币多分)

输出:
- 组合权益曲线
- 每个币的贡献分解
- 组合 Sharpe / MaxDD / 总收益
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig
from src.backtest.metrics import compute_metrics, Metrics
from src.strategies.base import Strategy


@dataclass
class PortfolioConfig:
    initial_equity: float = 10000.0
    fee_rate: float = 0.0004
    slippage: float = 0.0002
    max_positions: int = 5  # 同时持仓上限
    max_pct_per_symbol: float = 0.25  # 单币最大仓位 (占总资金)
    risk_per_trade: float = 0.02  # 单笔风险占总资金的比例
    max_total_leverage: float = 3.0  # 所有持仓的总杠杆
    daily_dd_halt: float = 0.10  # 当日跌 10% 停止开新仓
    allocation_mode: str = "equal"  # 'equal' | 'volatility' | 'equity'


@dataclass
class CoinResult:
    symbol: str
    trades: int
    pnl: float
    win_rate: float
    contribution: float  # 占组合总收益的比例


@dataclass
class PortfolioResult:
    metrics: Metrics
    equity_curve: pd.Series
    trades_by_symbol: dict[str, pd.DataFrame]
    coin_results: list[CoinResult]
    daily_returns: pd.Series

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f" 组合回测结果")
        lines.append("=" * 70)
        m = self.metrics
        lines.append(f"  初始资金:   ${10000:,.2f}")
        lines.append(f"  最终资金:   ${m.final_equity:,.2f}")
        lines.append(f"  总收益:     {m.total_return:+.2%}")
        lines.append(f"  Sharpe:     {m.sharpe:.2f}")
        lines.append(f"  最大回撤:   {m.max_drawdown:.2%}")
        lines.append(f"  总交易数:   {m.trades}")
        if m.trades > 0:
            lines.append(f"  胜率:       {m.win_rate:.1%}")
            lines.append(f"  盈亏比:     {m.profit_factor:.2f}")
        lines.append("")
        lines.append(f"  --- 各币种贡献 ---")
        for c in sorted(self.coin_results, key=lambda x: -x.pnl):
            lines.append(
                f"    {c.symbol:<8} {c.trades:>3}笔 胜率{c.win_rate*100:>5.1f}% "
                f"PnL ${c.pnl:>+10,.2f} 贡献 {c.contribution:>+6.1%}"
            )
        return "\n".join(lines)


def run_portfolio_backtest(
    coin_data: dict[str, pd.DataFrame],
    strategy_factory: Callable[[str], Strategy],
    cfg: PortfolioConfig | None = None,
) -> PortfolioResult:
    """运行多币种组合回测.

    Args:
        coin_data: {symbol: ohlcv_df} - 每个币的数据 (必须时间对齐)
        strategy_factory: 接受 symbol, 返回该币的 Strategy
        cfg: 组合配置

    Returns:
        PortfolioResult
    """
    cfg = cfg or PortfolioConfig()

    # 1. 时间轴对齐: 取所有币共同的时间戳交集
    all_timestamps = None
    for sym, df in coin_data.items():
        ts = pd.to_datetime(df["ts"])
        if all_timestamps is None:
            all_timestamps = set(ts)
        else:
            all_timestamps &= set(ts)
    common_ts = sorted(all_timestamps)
    if len(common_ts) < 50:
        raise ValueError(f"共同时间戳太少: {len(common_ts)}")

    # 重新索引每个币的数据到共同时间轴
    aligned: dict[str, pd.DataFrame] = {}
    signals_by_coin: dict[str, pd.DataFrame] = {}
    for sym, df in coin_data.items():
        d = df[df["ts"].isin(common_ts)].sort_values("ts").reset_index(drop=True)
        aligned[sym] = d
        s = strategy_factory(sym)
        signals_by_coin[sym] = s.generate(d)

    n_bars = len(common_ts)
    symbols = list(aligned.keys())

    # 2. 状态: 每个币的持仓
    @dataclass
    class Position:
        side: str = "flat"
        entry: float = 0.0
        qty: float = 0.0
        stop: float = 0.0
        entry_idx: int = -1
        peak_pnl: float = 0.0

    positions: dict[str, Position] = {s: Position() for s in symbols}
    cash = cfg.initial_equity
    trades_log: dict[str, list] = {s: [] for s in symbols}
    equity_series = np.zeros(n_bars)

    daily_high = cfg.initial_equity
    halted_today = False
    last_day = None

    def total_equity(bar_idx: int) -> float:
        eq = cash
        for sym in symbols:
            p = positions[sym]
            if p.side != "flat":
                close_px = aligned[sym]["close"].iloc[bar_idx]
                sign = 1 if p.side == "long" else -1
                eq += sign * (close_px - p.entry) * p.qty
        return eq

    # 3. 逐根K线遍历
    for i in range(n_bars):
        cur_ts = pd.Timestamp(common_ts[i])
        cur_day = cur_ts.normalize()
        if last_day != cur_day:
            last_day = cur_day
            daily_high = total_equity(i)
            halted_today = False

        # 检查日内回撤
        cur_eq = total_equity(i)
        if not halted_today and daily_high > 0:
            dd = (daily_high - cur_eq) / daily_high
            if dd > cfg.daily_dd_halt:
                halted_today = True

        # 第1步: 检查每个币的止损 (基于当前K线 high/low)
        for sym in symbols:
            p = positions[sym]
            if p.side == "flat":
                continue
            df_s = aligned[sym]
            high_i = df_s["high"].iloc[i]
            low_i = df_s["low"].iloc[i]
            if p.side == "long" and not np.isnan(p.stop) and low_i <= p.stop:
                # 止损出场
                exit_px = p.stop
                pnl = (exit_px - p.entry) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
                cash += pnl + p.entry * p.qty  # 退回保证金 + 收益
                trades_log[sym].append({
                    "entry_ts": pd.Timestamp(common_ts[p.entry_idx]),
                    "exit_ts": cur_ts,
                    "side": "long",
                    "entry": p.entry,
                    "exit": exit_px,
                    "qty": p.qty,
                    "pnl": pnl,
                    "reason": "stop",
                })
                positions[sym] = Position()
            elif p.side == "short" and not np.isnan(p.stop) and high_i >= p.stop:
                exit_px = p.stop
                pnl = (p.entry - exit_px) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
                cash += pnl + p.entry * p.qty
                trades_log[sym].append({
                    "entry_ts": pd.Timestamp(common_ts[p.entry_idx]),
                    "exit_ts": cur_ts,
                    "side": "short",
                    "entry": p.entry,
                    "exit": exit_px,
                    "qty": p.qty,
                    "pnl": pnl,
                    "reason": "stop",
                })
                positions[sym] = Position()

        # 第2步: 处理每个币的目标信号 (基于上一根K线的 close)
        for sym in symbols:
            sig = signals_by_coin[sym]
            target_side = sig["side"].iloc[i] if i < len(sig) else "flat"
            target_stop = sig["stop"].iloc[i] if i < len(sig) else np.nan
            p = positions[sym]
            df_s = aligned[sym]
            open_i = df_s["open"].iloc[i]

            if target_side == p.side:
                continue

            # 平掉旧仓 (信号反转)
            if p.side != "flat":
                exit_px = open_i * (1 - cfg.slippage) if p.side == "long" else open_i * (1 + cfg.slippage)
                if p.side == "long":
                    pnl = (exit_px - p.entry) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
                else:
                    pnl = (p.entry - exit_px) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
                cash += pnl + p.entry * p.qty
                trades_log[sym].append({
                    "entry_ts": pd.Timestamp(common_ts[p.entry_idx]),
                    "exit_ts": cur_ts,
                    "side": p.side,
                    "entry": p.entry,
                    "exit": exit_px,
                    "qty": p.qty,
                    "pnl": pnl,
                    "reason": "signal",
                })
                positions[sym] = Position()

            # 开新仓 (如果有目标且未停止 & 仓位数未满)
            if target_side in ("long", "short") and not halted_today:
                open_count = sum(1 for s in symbols if positions[s].side != "flat")
                if open_count >= cfg.max_positions:
                    continue
                if np.isnan(target_stop):
                    continue

                entry_px = open_i * (1 + cfg.slippage) if target_side == "long" else open_i * (1 - cfg.slippage)
                risk_per_unit = abs(entry_px - target_stop)
                if risk_per_unit <= 0:
                    continue

                # 仓位大小: 风险预算 (单笔最大风险)
                eq = total_equity(i)
                target_risk = eq * cfg.risk_per_trade
                qty_by_risk = target_risk / risk_per_unit

                # 不超过单币最大仓位
                qty_by_pct = (eq * cfg.max_pct_per_symbol) / entry_px

                # 不超过总杠杆 (考虑现有持仓)
                used_notional = sum(positions[s].entry * positions[s].qty for s in symbols if positions[s].side != "flat")
                remaining_notional = max(0, eq * cfg.max_total_leverage - used_notional)
                qty_by_lev = remaining_notional / entry_px

                qty = min(qty_by_risk, qty_by_pct, qty_by_lev)

                # 现金检查 (不能超过可用现金)
                margin_required = entry_px * qty
                if margin_required > cash:
                    qty = cash / entry_px

                if qty > 0:
                    cash -= entry_px * qty  # 占用保证金
                    positions[sym] = Position(
                        side=target_side,
                        entry=entry_px,
                        qty=qty,
                        stop=target_stop,
                        entry_idx=i,
                    )

        equity_series[i] = total_equity(i)

    # 4. 收尾: 按最后一根 K 线 close 平所有仓
    last_idx = n_bars - 1
    for sym in symbols:
        p = positions[sym]
        if p.side != "flat":
            exit_px = aligned[sym]["close"].iloc[last_idx]
            if p.side == "long":
                pnl = (exit_px - p.entry) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
            else:
                pnl = (p.entry - exit_px) * p.qty - (p.entry + exit_px) * p.qty * cfg.fee_rate
            cash += pnl + p.entry * p.qty
            trades_log[sym].append({
                "entry_ts": pd.Timestamp(common_ts[p.entry_idx]),
                "exit_ts": pd.Timestamp(common_ts[last_idx]),
                "side": p.side,
                "entry": p.entry,
                "exit": exit_px,
                "qty": p.qty,
                "pnl": pnl,
                "reason": "end",
            })
            positions[sym] = Position()
            equity_series[last_idx] = cash

    # 5. 整理结果
    equity_curve = pd.Series(equity_series, index=pd.to_datetime(common_ts), name="portfolio")
    all_pnls = [t["pnl"] for sym in symbols for t in trades_log[sym]]
    trades_dfs = {sym: pd.DataFrame(trades_log[sym]) for sym in symbols}
    metrics = compute_metrics(equity_curve, all_pnls, cfg.initial_equity)

    total_pnl = equity_curve.iloc[-1] - cfg.initial_equity
    coin_results = []
    for sym in symbols:
        coin_pnl = sum(t["pnl"] for t in trades_log[sym])
        coin_trades = len(trades_log[sym])
        coin_wins = sum(1 for t in trades_log[sym] if t["pnl"] > 0)
        wr = coin_wins / coin_trades if coin_trades > 0 else 0
        contrib = coin_pnl / total_pnl if abs(total_pnl) > 0 else 0
        coin_results.append(CoinResult(
            symbol=sym, trades=coin_trades, pnl=coin_pnl,
            win_rate=wr, contribution=contrib,
        ))

    daily_returns = equity_curve.pct_change().dropna()

    return PortfolioResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades_by_symbol=trades_dfs,
        coin_results=coin_results,
        daily_returns=daily_returns,
    )
