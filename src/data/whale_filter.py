"""链上鲸鱼数据过滤器.

数据源 (优先级排序):
1. 币安 Top Trader Long/Short Ratio (前 20% 大户持仓多空比)
   - 这是币安公开API最接近"鲸鱼数据"的指标
   - 链接: /futures/data/topLongShortPositionRatio

2. Open Interest 变化 (大户加仓/减仓信号)
   - OI上升 + 价格上升 = 大户加仓做多 (强势)
   - OI上升 + 价格下跌 = 大户加仓做空 (危险)
   - OI下降 + 价格上升 = 散户买单, 大户离场 (假突破风险)

3. Taker Buy/Sell Ratio (主动买卖压力)
   - taker_buy/taker_sell > 1.2 = 主动买盘强势
   - 配合 MA60 回踩 = 鲸鱼吸筹

4. 资金费率 (跨市场情绪)
   - 极端正值 = 多头拥挤, 鲸鱼可能反向
   - 极端负值 = 空头拥挤, 鲸鱼可能逼空

回退策略:
- 如果无法获取实盘衍生品数据, 用价量结构推导:
  - "假鲸鱼信号" = 量缩价涨 (没有大单跟进)
  - "真鲸鱼信号" = 放量突破 + 收盘强势

实际使用:
- 这是个"否决器" (veto): 默认信号通过, 鲸鱼明显反向才否决
- 不是"激活器" (gate): 不要求鲸鱼正向才入场 (会错过早期信号)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WhaleSignal:
    """鲸鱼数据信号."""
    score: float  # -1 (强烈卖出) 到 +1 (强烈买入)
    reason: str
    has_data: bool  # False = 无数据, 默认放行


def compute_oi_divergence(
    df: pd.DataFrame,
    lookback: int = 5,
) -> pd.Series:
    """OI 与价格的背离度.

    返回每根K线的 OI 散度评分 (-1 到 +1):
    - +1: OI 增长 + 价格增长 (健康趋势)
    - +0.5: OI 增长 + 价格震荡 (鲸鱼吸筹)
    - 0: OI 与价格无显著相关
    - -0.5: OI 下降 + 价格上涨 (散户接盘)
    - -1: OI 下降 + 价格下跌 (恐慌出局)
    """
    if "oi" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    oi = df["oi"]
    close = df["close"]

    oi_chg = oi.pct_change(lookback)
    price_chg = close.pct_change(lookback)

    score = pd.Series(0.0, index=df.index)
    # 同向移动: 健康
    same_dir_up = (oi_chg > 0.01) & (price_chg > 0.01)
    same_dir_dn = (oi_chg < -0.01) & (price_chg < -0.01)
    # 反向移动: 警告
    div_oi_up_px_dn = (oi_chg > 0.02) & (price_chg < -0.01)  # OI涨价跌 = 重仓做空
    div_oi_dn_px_up = (oi_chg < -0.02) & (price_chg > 0.01)  # OI跌价涨 = 大户卖给散户
    # 鲸鱼吸筹: OI涨价稳
    accumulation = (oi_chg > 0.03) & (price_chg.abs() < 0.01)

    score[same_dir_up] = 1.0
    score[same_dir_dn] = -1.0
    score[div_oi_up_px_dn] = -1.0
    score[div_oi_dn_px_up] = -0.5
    score[accumulation] = 0.5

    return score


def compute_top_trader_signal(df: pd.DataFrame) -> pd.Series:
    """Top trader 多空比信号 (币安前 20% 大户).

    返回每根K线的 Top Trader 评分 (-1 到 +1):
    - top_ls_ratio > 1.5 且上升 = 大户加仓做多 (+1)
    - top_ls_ratio < 0.7 且下降 = 大户加仓做空 (-1)
    - top_ls_ratio 接近 1 = 中性 (0)
    """
    if "top_ls_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    ratio = df["top_ls_ratio"]
    ratio_chg = ratio.diff(3)

    score = pd.Series(0.0, index=df.index)
    score[(ratio > 1.5) & (ratio_chg > 0)] = 1.0
    score[(ratio > 1.3) & (ratio_chg > 0)] = 0.5
    score[(ratio < 0.7) & (ratio_chg < 0)] = -1.0
    score[(ratio < 0.85) & (ratio_chg < 0)] = -0.5

    return score


def compute_taker_pressure(df: pd.DataFrame) -> pd.Series:
    """Taker 主动买卖压力.

    返回每根K线的主动盘评分 (-1 到 +1):
    - taker_ratio > 1.3 持续 3 根 = 持续主动买盘 (+1)
    - taker_ratio < 0.7 持续 3 根 = 持续主动卖盘 (-1)
    """
    if "taker_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    tr = df["taker_ratio"]
    tr_ma = tr.rolling(3, min_periods=1).mean()

    score = pd.Series(0.0, index=df.index)
    score[tr_ma > 1.3] = 1.0
    score[(tr_ma > 1.1) & (tr_ma <= 1.3)] = 0.5
    score[tr_ma < 0.7] = -1.0
    score[(tr_ma < 0.9) & (tr_ma >= 0.7)] = -0.5

    return score


def compute_volume_quality(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """价量结构判断 (无衍生品数据时的回退方案).

    "鲸鱼吸筹" 的价量特征:
    - 量比 > 1.5 (放量)
    - 收盘价高于开盘价 (阳线)
    - 上影线 < 实体 (没有抛压)

    返回每根K线的价量质量评分 (-1 到 +1):
    - +1: 量价齐升, 健康
    - +0.5: 缩量阳线 (健康但弱)
    - 0: 中性
    - -0.5: 放量阴线 (危险)
    - -1: 量价齐跌或长上影 (出货)
    """
    close = df["close"]
    opens = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    vol_avg = volume.rolling(period, min_periods=5).mean()
    vol_ratio = volume / vol_avg

    body = close - opens
    upper_wick = high - close.where(close > opens, opens)
    body_abs = body.abs()

    is_bullish = body > 0
    is_bearish = body < 0
    high_volume = vol_ratio > 1.5
    has_upper_pressure = (upper_wick > body_abs * 1.5) & body_abs.gt(0)

    score = pd.Series(0.0, index=df.index)
    # 量价齐升: 强势
    score[is_bullish & high_volume & ~has_upper_pressure] = 1.0
    # 阳线但缩量: 弱
    score[is_bullish & ~high_volume] = 0.3
    # 长上影: 出货
    score[has_upper_pressure & high_volume] = -1.0
    score[has_upper_pressure & ~high_volume] = -0.5
    # 放量阴线: 抛压
    score[is_bearish & high_volume] = -1.0
    score[is_bearish & ~high_volume] = -0.3

    return score


def whale_check(
    df: pd.DataFrame,
    i: int,
    direction: str = "long",
    veto_threshold: float = -0.5,
) -> WhaleSignal:
    """在第 i 根K线上检查鲸鱼数据是否反向.

    Args:
        df: 包含 OHLCV + (可选) oi/top_ls_ratio/taker_ratio 的数据
        i: 当前索引
        direction: 'long' 或 'short' — 我们想做的方向
        veto_threshold: 鲸鱼综合评分低于此值则否决 (默认 -0.5)

    Returns:
        WhaleSignal: 如果 score < veto_threshold (做多) 或 > -veto_threshold (做空), 拒绝信号
    """
    if i < 5:
        return WhaleSignal(score=0.0, reason="数据不足", has_data=False)

    components = {}
    weights = {}

    # 1. OI 散度 (权重 0.4)
    oi_score = compute_oi_divergence(df.iloc[:i+1]).iloc[-1] if "oi" in df.columns else None
    if oi_score is not None and not np.isnan(oi_score):
        components["oi"] = oi_score
        weights["oi"] = 0.4

    # 2. Top Trader (权重 0.3)
    tt_score = compute_top_trader_signal(df.iloc[:i+1]).iloc[-1] if "top_ls_ratio" in df.columns else None
    if tt_score is not None and not np.isnan(tt_score):
        components["top_trader"] = tt_score
        weights["top_trader"] = 0.3

    # 3. Taker 压力 (权重 0.2)
    tp_score = compute_taker_pressure(df.iloc[:i+1]).iloc[-1] if "taker_ratio" in df.columns else None
    if tp_score is not None and not np.isnan(tp_score):
        components["taker"] = tp_score
        weights["taker"] = 0.2

    # 4. 价量结构 (权重 0.1, 永远可用)
    vq_score = compute_volume_quality(df.iloc[:i+1]).iloc[-1]
    if not np.isnan(vq_score):
        components["volume_quality"] = vq_score
        weights["volume_quality"] = 0.1

    if not components:
        return WhaleSignal(score=0.0, reason="无可用数据", has_data=False)

    # 加权平均
    total_weight = sum(weights.values())
    composite = sum(components[k] * weights[k] for k in components) / total_weight

    # 做多: 反向阈值是低分 (大户在卖)
    # 做空: 反向阈值是高分 (大户在买)
    if direction == "long":
        passed = composite > veto_threshold
    else:
        passed = composite < -veto_threshold

    has_real_data = "oi" in components or "top_trader" in components

    parts = [f"{k}:{v:+.2f}" for k, v in components.items()]
    reason = f"鲸鱼分:{composite:+.2f} ({', '.join(parts)})"
    if not passed:
        reason = f"❌ {reason} - 鲸鱼反向, 否决"
    return WhaleSignal(score=composite, reason=reason, has_data=has_real_data)


def add_whale_filter_to_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    veto_threshold: float = -0.5,
) -> pd.DataFrame:
    """将鲸鱼过滤器应用到现有的策略信号上.

    在每根 'flat' -> 'long'/'short' 的转换点检查鲸鱼数据,
    如果鲸鱼反向则保持 flat.

    Args:
        df: OHLCV (+衍生品数据)
        signals: 原始策略信号 (side, stop)
        veto_threshold: 否决阈值

    Returns:
        过滤后的信号 DataFrame
    """
    out = signals.copy()
    sides = out["side"].to_numpy()
    stops = out["stop"].to_numpy()

    n = len(df)
    prev_side = "flat"
    veto_active = False

    for i in range(n):
        cur_side = sides[i]
        # 检测转换点 (flat -> long/short)
        if prev_side == "flat" and cur_side in ("long", "short"):
            wh = whale_check(df, i, direction=cur_side, veto_threshold=veto_threshold)
            if wh.has_data and wh.score < veto_threshold and cur_side == "long":
                veto_active = True
            elif wh.has_data and wh.score > -veto_threshold and cur_side == "short":
                veto_active = True
            else:
                veto_active = False

        if veto_active:
            sides[i] = "flat"
            stops[i] = np.nan
            # 一旦原信号变 flat, 取消 veto
            if cur_side == "flat":
                veto_active = False
        else:
            prev_side = cur_side

    out["side"] = sides
    out["stop"] = stops
    return out
