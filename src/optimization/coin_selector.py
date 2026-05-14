"""币种自适应选择器 v2 — 多维评分 + 子时段一致性 + 相关性去重 + Top-K.

设计哲学的演变:
v1 (旧版): 二元通过/不通过. 问题:
  - Sharpe > 0.5 这种硬阈值很脆弱 (0.49 就被拒, 0.51 就通过, 没有梯度)
  - 单时段评估容易因为一两笔幸运交易"通过"
  - 没有币种间相关性去重 (BTC + ETH + SOL 高度相关, 同时持仓不是真分散)

v2 (本版): 综合评分排序. 改进:
  1. 多维评分: Sharpe + Calmar + Profit Factor + 一致性 + 流动性 → 0-100 综合分
  2. 子时段一致性: 把训练期切 3 段, 每段都跑回测, 统计 "几段盈利"
     → 3/3 段盈利的币比偶尔大赚的币更可信
  3. 相关性聚类去重: 高度相关的币 (>0.85) 只保留分数最高的一个
  4. Top-K 排序: 不是 "有几个过了" 而是 "选 K 个最好的"
  5. 软门槛 + 硬门槛: 流动性是硬门槛 (流动性差就直接拒), 其他是软评分

输出: 推荐币列表 + 每币参数 + 评分明细
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.strategies import get_strategy


@dataclass
class CoinProfile:
    symbol: str
    # === 市场特征 ===
    avg_atr_pct: float          # ATR / price
    avg_adx: float              # 趋势强度
    avg_volume_usd: float       # 美元成交量
    # === 训练期表现 ===
    train_sharpe: float
    train_return: float
    train_trades: int
    train_pf: float             # profit factor
    train_calmar: float         # return / max_dd
    train_max_dd: float
    # === 一致性 (核心创新点) ===
    sub_period_wins: int        # 几个子时段盈利
    sub_period_count: int       # 总子时段数
    consistency_score: float    # 0-1, 一致性指标
    # === 综合评分 ===
    composite_score: float      # 0-100
    suggested_params: dict
    # === 决策 ===
    is_recommended: bool
    rank: int = 0               # Top-K 排名 (1-indexed, 0 = 未入选)
    reject_reason: str = ""
    correlated_with: list[str] = field(default_factory=list)


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift()).abs()
    low_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up, dn = high.diff(), -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def adaptive_params(avg_atr_pct: float, avg_adx: float) -> dict:
    """根据波动率推荐参数 (粗粒度, 3 档)."""
    if avg_atr_pct > 0.05:  # 高波动 (SOL, DOGE)
        return {
            "ma_period": 60, "atr_mult_zone": 2.0, "vol_mult": 1.8,
            "use_weekly_macd": True, "require_macd_filter": True,
        }
    if avg_atr_pct < 0.025:  # 低波动 (BTC)
        return {
            "ma_period": 50, "atr_mult_zone": 1.0, "vol_mult": 1.2,
            "use_weekly_macd": True, "require_macd_filter": True,
        }
    return {  # 中波动 (ETH, BNB, ...)
        "ma_period": 60, "atr_mult_zone": 1.5, "vol_mult": 1.5,
        "use_weekly_macd": True, "require_macd_filter": True,
    }


def _evaluate_sub_periods(
    df: pd.DataFrame,
    params: dict,
    n_periods: int = 3,
    cfg: Optional[BacktestConfig] = None,
) -> tuple[int, float]:
    """子时段一致性测试.
    
    把训练期切成 n_periods 段, 每段独立跑回测.
    返回 (盈利段数, 一致性分数 0-1).
    
    一致性分数 = 盈利段数 / 总段数 + 平均段收益的稳定性奖励.
    """
    cfg = cfg or BacktestConfig(initial_equity=10000, risk_per_trade=0.05, max_leverage=1)
    n = len(df)
    seg_size = n // n_periods
    if seg_size < 80:  # 段太短没意义
        return 0, 0.0

    returns = []
    wins = 0
    for i in range(n_periods):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_periods - 1 else n
        seg = df.iloc[start:end].reset_index(drop=True)
        try:
            s = get_strategy("trend_filter", **params)
            r = run_backtest(seg, s, cfg)
            ret = r.metrics.total_return
            returns.append(ret)
            if ret > 0:
                wins += 1
        except Exception:
            returns.append(-0.05)  # 失败视为小亏

    # 一致性分数: 胜率 + 稳定性奖励 (回报标准差越小奖励越高)
    win_rate = wins / n_periods
    if len(returns) > 1 and np.std(returns) > 0:
        # 标准差归一化: 越小越好, 给稳定的币加分
        stability = 1.0 / (1.0 + abs(np.std(returns)) * 10)
    else:
        stability = 0.5
    consistency = win_rate * 0.7 + stability * 0.3
    return wins, consistency


def _compute_composite_score(p: CoinProfile) -> float:
    """计算综合评分 0-100.
    
    权重设计:
    - Sharpe (30%)        : 风险调整收益, 最重要
    - 一致性 (25%)        : 多段稳定盈利, 防止靠运气
    - Profit Factor (15%) : 盈亏比
    - Calmar (15%)        : 回撤效率
    - 收益率 (10%)        : 绝对收益
    - 交易频率 (5%)       : 样本量
    """
    # Sharpe: 0-3 映射到 0-100
    sharpe_score = min(max(p.train_sharpe / 3.0, 0), 1) * 100

    # 一致性: 0-1 映射到 0-100
    consistency_score = p.consistency_score * 100

    # Profit Factor: 1.0 = 50, 2.0 = 75, 3+ = 100, <1 = 线性递减
    if p.train_pf == float("inf"):
        pf_score = 100
    elif p.train_pf >= 1:
        pf_score = min(50 + (p.train_pf - 1) * 25, 100)
    else:
        pf_score = max(p.train_pf * 50, 0)

    # Calmar: 1.0 = 50, 3.0 = 100
    calmar_score = min(max(p.train_calmar / 3.0, 0), 1) * 100

    # 总收益: 0% = 50, 50%+ = 100, 负值递减
    return_score = min(max(50 + p.train_return * 100, 0), 100)

    # 交易频率: 5+ = 100, <2 = 0 (样本量惩罚)
    if p.train_trades < 2:
        trade_score = 0
    elif p.train_trades >= 5:
        trade_score = 100
    else:
        trade_score = (p.train_trades - 2) * 33

    composite = (
        sharpe_score * 0.30 +
        consistency_score * 0.25 +
        pf_score * 0.15 +
        calmar_score * 0.15 +
        return_score * 0.10 +
        trade_score * 0.05
    )
    return composite


def profile_coin(
    symbol: str,
    df: pd.DataFrame,
    train_ratio: float = 0.66,
    min_volume_usd: float = 100_000,
    min_trades: int = 2,
) -> CoinProfile:
    """对单个币做完整画像 (无 pass/fail, 只算分)."""
    n = len(df)
    train_end = int(n * train_ratio)
    train_df = df.iloc[:train_end].reset_index(drop=True)

    # 1. 市场特征
    atr = _atr_series(train_df).dropna()
    avg_atr_pct = atr.mean() / train_df["close"].mean() if len(atr) > 0 else 0.03
    adx = _adx_series(train_df).dropna()
    avg_adx = adx.mean() if len(adx) > 0 else 0
    avg_vol_usd = (train_df["close"] * train_df["volume"]).mean()

    # 2. 自适应参数
    params = adaptive_params(avg_atr_pct, avg_adx)

    # 3. 训练期完整回测
    cfg = BacktestConfig(initial_equity=10000, risk_per_trade=0.05, max_leverage=1)
    try:
        s = get_strategy("trend_filter", **params)
        r = run_backtest(train_df, s, cfg)
        m = r.metrics
        train_sharpe = m.sharpe
        train_return = m.total_return
        train_trades = m.trades
        train_pf = m.profit_factor
        train_max_dd = m.max_drawdown
        train_calmar = abs(train_return / train_max_dd) if train_max_dd < 0 else 0
    except Exception:
        train_sharpe, train_return, train_trades = -99, -1, 0
        train_pf, train_max_dd, train_calmar = 0, -1, 0

    # 4. 子时段一致性测试 (核心创新)
    sub_wins, consistency = _evaluate_sub_periods(train_df, params, n_periods=3, cfg=cfg)

    # 5. 硬门槛: 流动性
    reject_reasons = []
    if avg_vol_usd < min_volume_usd:
        reject_reasons.append(f"流动性不足({avg_vol_usd/1e6:.2f}M)")
    if train_trades < min_trades:
        reject_reasons.append(f"交易过少({train_trades})")

    profile = CoinProfile(
        symbol=symbol,
        avg_atr_pct=avg_atr_pct,
        avg_adx=avg_adx,
        avg_volume_usd=avg_vol_usd,
        train_sharpe=train_sharpe,
        train_return=train_return,
        train_trades=train_trades,
        train_pf=train_pf,
        train_calmar=train_calmar,
        train_max_dd=train_max_dd,
        sub_period_wins=sub_wins,
        sub_period_count=3,
        consistency_score=consistency,
        composite_score=0,
        suggested_params=params,
        is_recommended=False,
        reject_reason=", ".join(reject_reasons) if reject_reasons else "",
    )
    profile.composite_score = _compute_composite_score(profile)
    return profile


def _compute_correlation_matrix(coin_data: dict[str, pd.DataFrame], train_ratio: float) -> pd.DataFrame:
    """计算训练期的币种价格相关性矩阵."""
    returns = {}
    for sym, df in coin_data.items():
        train_end = int(len(df) * train_ratio)
        train_df = df.iloc[:train_end]
        ret = train_df["close"].pct_change().dropna()
        returns[sym] = ret.reset_index(drop=True)
    
    aligned = pd.DataFrame(returns).dropna()
    if len(aligned) < 30:
        return pd.DataFrame()
    return aligned.corr()


def _deduplicate_correlated(
    profiles: list[CoinProfile],
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> list[CoinProfile]:
    """对相关性高的币种聚类, 每个簇保留分数最高的.
    
    贪心算法: 按分数从高到低遍历, 跳过与已选币高度相关的.
    """
    if corr_matrix.empty:
        return profiles

    # 按 composite_score 降序排
    sorted_profiles = sorted(profiles, key=lambda p: -p.composite_score)
    selected: list[CoinProfile] = []

    for p in sorted_profiles:
        if p.symbol not in corr_matrix.columns:
            selected.append(p)
            continue
        # 检查与已选币的相关性
        too_correlated = False
        for s in selected:
            if s.symbol not in corr_matrix.columns:
                continue
            corr = corr_matrix.loc[p.symbol, s.symbol]
            if abs(corr) >= threshold:
                # 标记被它代表
                p.correlated_with.append(s.symbol)
                too_correlated = True
                break
        if not too_correlated:
            selected.append(p)

    return selected


def select_coins(
    coin_data: dict[str, pd.DataFrame],
    train_ratio: float = 0.66,
    top_k: int = 5,
    min_composite_score: float = 50.0,
    correlation_threshold: float = 0.85,
    min_volume_usd: float = 100_000,
    min_trades: int = 2,
) -> tuple[list[CoinProfile], dict[str, dict]]:
    """v2 选币器主入口.
    
    Args:
        coin_data: {symbol: ohlcv_df}
        train_ratio: 训练集比例
        top_k: 最终选 K 个币 (按综合分排序)
        min_composite_score: 综合分最低线 (0-100)
        correlation_threshold: 相关性聚类阈值
        min_volume_usd: 流动性硬门槛
        min_trades: 交易数硬门槛
    
    Returns:
        (profiles, params_map) - profiles 包含所有币(已排名), params_map 仅包含推荐币
    """
    # 1. 对每个币做画像
    profiles = []
    for sym, df in coin_data.items():
        p = profile_coin(sym, df, train_ratio=train_ratio,
                         min_volume_usd=min_volume_usd, min_trades=min_trades)
        profiles.append(p)

    # 2. 计算相关性矩阵 + 聚类去重
    corr_matrix = _compute_correlation_matrix(coin_data, train_ratio)

    # 3. 先过滤掉硬门槛失败的
    eligible = [p for p in profiles if not p.reject_reason]

    # 4. 相关性去重 (在合格池中)
    deduped = _deduplicate_correlated(eligible, corr_matrix, threshold=correlation_threshold)

    # 5. 综合分门槛 + Top-K
    above_threshold = [p for p in deduped if p.composite_score >= min_composite_score]
    above_threshold.sort(key=lambda p: -p.composite_score)
    selected = above_threshold[:top_k]

    # 6. 标记入选 + 排名
    selected_syms = set()
    for rank, p in enumerate(selected, 1):
        p.is_recommended = True
        p.rank = rank
        selected_syms.add(p.symbol)

    # 给被相关性淘汰的标记理由
    for p in profiles:
        if not p.is_recommended and not p.reject_reason:
            if p.correlated_with:
                p.reject_reason = f"与 {','.join(p.correlated_with)} 高度相关 (聚类去重)"
            elif p.composite_score < min_composite_score:
                p.reject_reason = f"综合分不足 ({p.composite_score:.1f}<{min_composite_score})"
            else:
                p.reject_reason = f"未进入 Top-{top_k}"

    params_map = {p.symbol: p.suggested_params for p in selected}
    return profiles, params_map


def format_profile_report(profiles: list[CoinProfile]) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(" 币种综合评分报告 (v2)")
    lines.append("=" * 90)
    lines.append(f"  {'排名':<4} {'币种':<6} {'综合分':>6} {'Sharpe':>7} {'PF':>5} {'Calmar':>6} "
                 f"{'一致性':>7} {'收益':>7} {'交易':>4} {'状态':>4}")
    lines.append("  " + "-" * 80)

    sorted_profiles = sorted(profiles, key=lambda p: (-p.is_recommended, -p.composite_score))
    for p in sorted_profiles:
        flag = f"#{p.rank}" if p.is_recommended else "❌"
        consistency_str = f"{p.sub_period_wins}/{p.sub_period_count}"
        pf_str = f"{p.train_pf:.2f}" if p.train_pf != float("inf") else "inf"
        lines.append(
            f"  {flag:<4} {p.symbol:<6} {p.composite_score:>6.1f} "
            f"{p.train_sharpe:>7.2f} {pf_str:>5} {p.train_calmar:>6.2f} "
            f"{consistency_str:>7} {p.train_return*100:>+6.1f}% "
            f"{p.train_trades:>4} {flag:>4}"
        )
        if not p.is_recommended and p.reject_reason:
            lines.append(f"         拒绝原因: {p.reject_reason}")

    lines.append("")
    lines.append("  评分维度: Sharpe 30% + 一致性 25% + PF 15% + Calmar 15% + 收益 10% + 交易数 5%")
    lines.append("  一致性: 训练期切3段, 几段盈利 (3/3 = 全部时段稳定)")
    return "\n".join(lines)
