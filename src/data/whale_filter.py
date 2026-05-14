"""鲸鱼数据过滤器 v2 — 多模式集成 + 仓位调整.

设计哲学的演变:
v1 (旧版): 只有"否决器"模式. 问题:
  - 在日线趋势策略上几乎不触发 (因为 trend_filter 已经有量能过滤)
  - 错过了"确认器"价值: 鲸鱼正向应该让我们更激进
  - 没有"仓位调整"模式: 鲸鱼数据应该影响仓位大小

v2 (本版): 三种模式可组合使用:
  1. VETO 模式 (否决器): 鲸鱼明显反向 → 取消信号
     用于: 高频策略、有假信号风险的策略
  
  2. CONFIRM 模式 (确认器): 鲸鱼必须正向才入场
     用于: 保守策略、追求高胜率
  
  3. SIZE 模式 (仓位调整): 根据 whale score 动态调整仓位
     - score > +0.5: 加仓 (1.3x)
     - score in [-0.3, +0.5]: 标准仓位
     - score < -0.3: 减仓 (0.5x)
     - score < -0.5: 跳过
     用于: 风险预算管理

数据源:
1. OI (Open Interest) - 最重要, 反映总持仓
2. Top Trader Long/Short Ratio - 大户持仓方向
3. Taker Buy/Sell Ratio - 主动买卖压力
4. 价量结构 - 无衍生品数据时的回退
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class FilterMode(str, Enum):
    """过滤器模式."""
    VETO = "veto"          # 鲸鱼反向就取消信号
    CONFIRM = "confirm"    # 必须鲸鱼正向才入场
    SIZE = "size"          # 根据鲸鱼分数调整仓位
    DISABLED = "disabled"


@dataclass
class WhaleSignal:
    """鲸鱼综合信号."""
    score: float              # -1 到 +1
    reason: str
    has_data: bool            # False = 无衍生品数据 (只用了价量回退)
    components: dict          # 各维度分数
    size_multiplier: float = 1.0  # SIZE 模式下的仓位倍数
    action: str = "pass"      # "pass" | "veto" | "boost" | "reduce"


# ═════════════════════ 各维度信号计算 ═════════════════════

def compute_oi_divergence(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """OI 与价格的散度.
    
    +1: OI 增长 + 价格增长 (健康趋势)
    +0.5: OI 增长 + 价格震荡 (鲸鱼吸筹)
    -0.5: OI 下降 + 价格上涨 (散户接盘)
    -1: OI 涨价跌 (机构做空) 或 量价齐跌
    """
    if "oi" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    oi_chg = df["oi"].pct_change(lookback)
    price_chg = df["close"].pct_change(lookback)
    score = pd.Series(0.0, index=df.index)
    score[(oi_chg > 0.01) & (price_chg > 0.01)] = 1.0
    score[(oi_chg < -0.01) & (price_chg < -0.01)] = -1.0
    score[(oi_chg > 0.02) & (price_chg < -0.01)] = -1.0
    score[(oi_chg < -0.02) & (price_chg > 0.01)] = -0.5
    score[(oi_chg > 0.03) & (price_chg.abs() < 0.01)] = 0.5
    return score


def compute_top_trader_signal(df: pd.DataFrame) -> pd.Series:
    """大户多空比 (币安前 20%).
    
    > 1.5 + 上升: 大户加仓做多 +1
    > 1.3 + 上升: 偏多 +0.5
    < 0.7 + 下降: 大户加仓做空 -1
    < 0.85 + 下降: 偏空 -0.5
    """
    if "top_ls_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    ratio = df["top_ls_ratio"]
    chg = ratio.diff(3)
    score = pd.Series(0.0, index=df.index)
    score[(ratio > 1.5) & (chg > 0)] = 1.0
    score[(ratio > 1.3) & (ratio <= 1.5) & (chg > 0)] = 0.5
    score[(ratio < 0.7) & (chg < 0)] = -1.0
    score[(ratio < 0.85) & (ratio >= 0.7) & (chg < 0)] = -0.5
    return score


def compute_taker_pressure(df: pd.DataFrame) -> pd.Series:
    """主动买卖压力."""
    if "taker_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tr = df["taker_ratio"].rolling(3, min_periods=1).mean()
    score = pd.Series(0.0, index=df.index)
    score[tr > 1.3] = 1.0
    score[(tr > 1.1) & (tr <= 1.3)] = 0.5
    score[(tr < 0.9) & (tr >= 0.7)] = -0.5
    score[tr < 0.7] = -1.0
    return score


def compute_volume_quality(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """价量结构 (回退方案, 永远可用)."""
    close, opens, high, low, volume = df["close"], df["open"], df["high"], df["low"], df["volume"]
    vol_avg = volume.rolling(period, min_periods=5).mean()
    vol_ratio = volume / vol_avg
    body = close - opens
    upper_wick = high - close.where(close > opens, opens)
    body_abs = body.abs()
    is_bull = body > 0
    is_bear = body < 0
    high_vol = vol_ratio > 1.5
    upper_pressure = (upper_wick > body_abs * 1.5) & body_abs.gt(0)
    score = pd.Series(0.0, index=df.index)
    score[is_bull & high_vol & ~upper_pressure] = 1.0
    score[is_bull & ~high_vol] = 0.3
    score[upper_pressure & high_vol] = -1.0
    score[upper_pressure & ~high_vol] = -0.5
    score[is_bear & high_vol] = -1.0
    score[is_bear & ~high_vol] = -0.3
    return score


# ═════════════════════ v2 核心: 综合评分 + 多模式 ═════════════════════

def compute_whale_score(df: pd.DataFrame, i: int) -> WhaleSignal:
    """单点 whale score 计算 (v2 加入更多组件 + 加权调整)."""
    if i < 5:
        return WhaleSignal(0.0, "数据不足", False, {})

    components = {}
    weights = {}

    # 1. OI 散度 (核心, 权重 0.4)
    if "oi" in df.columns:
        oi_score = compute_oi_divergence(df.iloc[:i+1]).iloc[-1]
        if not np.isnan(oi_score):
            components["oi"] = oi_score
            weights["oi"] = 0.40

    # 2. Top Trader (权重 0.30)
    if "top_ls_ratio" in df.columns:
        tt_score = compute_top_trader_signal(df.iloc[:i+1]).iloc[-1]
        if not np.isnan(tt_score):
            components["top_trader"] = tt_score
            weights["top_trader"] = 0.30

    # 3. Taker (权重 0.20)
    if "taker_ratio" in df.columns:
        tp_score = compute_taker_pressure(df.iloc[:i+1]).iloc[-1]
        if not np.isnan(tp_score):
            components["taker"] = tp_score
            weights["taker"] = 0.20

    # 4. 价量 (永远可用, 权重 0.10)
    vq_score = compute_volume_quality(df.iloc[:i+1]).iloc[-1]
    if not np.isnan(vq_score):
        components["volume_quality"] = vq_score
        weights["volume_quality"] = 0.10

    if not components:
        return WhaleSignal(0.0, "无数据", False, {})

    total_w = sum(weights.values())
    composite = sum(components[k] * weights[k] for k in components) / total_w
    has_real = "oi" in components or "top_trader" in components

    parts = [f"{k}:{v:+.2f}" for k, v in components.items()]
    reason = f"鲸鱼分:{composite:+.2f} ({', '.join(parts)})"

    return WhaleSignal(
        score=composite, reason=reason, has_data=has_real,
        components=components, size_multiplier=1.0, action="pass",
    )


def evaluate_whale_for_entry(
    df: pd.DataFrame,
    i: int,
    direction: str,
    mode: FilterMode = FilterMode.VETO,
    veto_threshold: float = -0.3,
    confirm_threshold: float = 0.3,
) -> WhaleSignal:
    """对入场点做鲸鱼评估, 返回包含决策的 signal.
    
    Args:
        df: 数据
        i: K 线索引
        direction: 'long' or 'short'
        mode: VETO / CONFIRM / SIZE / DISABLED
        veto_threshold: VETO 模式下, 鲸鱼分低于此值就否决 (做多场景)
        confirm_threshold: CONFIRM 模式下, 鲸鱼分高于此值才放行 (做多场景)
    
    Returns: WhaleSignal with action ∈ {pass, veto, boost, reduce}
    """
    sig = compute_whale_score(df, i)
    
    if mode == FilterMode.DISABLED:
        sig.action = "pass"
        return sig
    
    # 做空场景: 鲸鱼分含义反过来
    score = sig.score if direction == "long" else -sig.score
    
    if mode == FilterMode.VETO:
        # 鲸鱼明显反向 -> 否决
        if sig.has_data and score < veto_threshold:
            sig.action = "veto"
            sig.reason = f"❌ {sig.reason} - VETO (反向, 取消信号)"
        else:
            sig.action = "pass"
    
    elif mode == FilterMode.CONFIRM:
        # 必须正向才放行
        if score >= confirm_threshold:
            sig.action = "pass"
            sig.reason = f"✓ {sig.reason} - CONFIRM (鲸鱼支持)"
        else:
            sig.action = "veto"
            sig.reason = f"❌ {sig.reason} - CONFIRM (鲸鱼不支持, 跳过)"
    
    elif mode == FilterMode.SIZE:
        # 根据 score 调整仓位倍数
        if score >= 0.5:
            sig.size_multiplier = 1.3
            sig.action = "boost"
            sig.reason = f"⬆ {sig.reason} - 加仓 1.3x"
        elif score >= -0.3:
            sig.size_multiplier = 1.0
            sig.action = "pass"
        elif score >= -0.5:
            sig.size_multiplier = 0.5
            sig.action = "reduce"
            sig.reason = f"⬇ {sig.reason} - 减仓 0.5x"
        else:
            sig.size_multiplier = 0.0
            sig.action = "veto"
            sig.reason = f"❌ {sig.reason} - 鲸鱼强烈反向, 跳过"
    
    return sig


def add_whale_filter_to_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    mode: FilterMode = FilterMode.VETO,
    veto_threshold: float = -0.3,
    confirm_threshold: float = 0.3,
) -> pd.DataFrame:
    """把鲸鱼过滤应用到现有信号 (v2 多模式).
    
    返回 DataFrame:
    - side: 调整后的 side
    - stop: 调整后的 stop
    - whale_size_mult: 仓位倍数 (SIZE 模式) — 可被回测引擎读取
    - whale_action: pass/veto/boost/reduce
    - whale_score: 原始分数
    """
    out = signals.copy()
    sides = out["side"].to_numpy().copy()
    stops = out["stop"].to_numpy().copy()
    size_mults = np.ones(len(df))
    actions = np.full(len(df), "pass", dtype=object)
    scores = np.zeros(len(df))

    n = len(df)
    veto_active = False
    veto_size_mult = 1.0
    prev_side = "flat"

    for i in range(n):
        cur_side = sides[i]

        # 检测转换点
        if prev_side == "flat" and cur_side in ("long", "short"):
            sig = evaluate_whale_for_entry(df, i, cur_side, mode,
                                           veto_threshold, confirm_threshold)
            scores[i] = sig.score
            actions[i] = sig.action
            
            if sig.action == "veto":
                veto_active = True
                veto_size_mult = 0.0
            elif sig.action in ("boost", "reduce"):
                veto_active = False
                veto_size_mult = sig.size_multiplier
            else:
                veto_active = False
                veto_size_mult = 1.0

        if veto_active:
            sides[i] = "flat"
            stops[i] = np.nan
            size_mults[i] = 0.0
            if cur_side == "flat":
                veto_active = False
                veto_size_mult = 1.0
        else:
            size_mults[i] = veto_size_mult
            # 当持仓回到 flat 时, 重置倍数
            if cur_side == "flat":
                veto_size_mult = 1.0

        prev_side = cur_side

    out["side"] = sides
    out["stop"] = stops
    out["whale_size_mult"] = size_mults
    out["whale_action"] = actions
    out["whale_score"] = scores
    return out


# ═════════════════════ 拉真实数据辅助 ═════════════════════

def fetch_whale_data_for_period(
    symbol_raw: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    period: str = "1h",
) -> Optional[pd.DataFrame]:
    """从币安拉取指定时间段的衍生品数据 (OI + LS + Taker).
    
    注意: 币安 OI 历史只有 30 天.
    
    Returns: DataFrame (ts, oi, top_ls_ratio, taker_ratio) 或 None.
    """
    try:
        from src.data.oi_data import (
            fetch_oi_history, fetch_top_trader_long_short_ratio,
            fetch_taker_long_short_ratio,
        )
    except Exception:
        return None

    try:
        oi = fetch_oi_history(symbol_raw, period=period, limit=500)
        tt = fetch_top_trader_long_short_ratio(symbol_raw, period=period, limit=500)
        tk = fetch_taker_long_short_ratio(symbol_raw, period=period, limit=500)

        if oi.empty:
            return None

        merged = oi[["ts", "oi"]].copy()
        if not tt.empty:
            merged = pd.merge_asof(
                merged.sort_values("ts"),
                tt[["ts", "top_ls_ratio"]].sort_values("ts"),
                on="ts", direction="backward",
            )
        if not tk.empty:
            merged = pd.merge_asof(
                merged.sort_values("ts"),
                tk[["ts", "taker_ratio"]].sort_values("ts"),
                on="ts", direction="backward",
            )
        return merged
    except Exception:
        return None
