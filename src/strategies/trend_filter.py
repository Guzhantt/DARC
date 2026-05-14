"""趋势过滤策略 (Trend Filter Strategy)

基于交易员PRD实现:
1. 模块A: 选币与过滤 (热门筛选 + 资金流出剔除 + 月线MACD大环境开关)
2. 模块B: 入场触发 (MA60回踩 + 阳线/长下影线 + 量能确认)
3. 模块C: 风控退出 (MA60破位清仓 + 阶梯止盈 + 重新入场)

我的改进 (开发者补丁):
- "假摔"过滤: 收盘价跌破MA60超过1% 且 维持2根K线 才认定为真破位
- ATR动态容错: ±2%固定容错改为 ±1.5*ATR
- 周线MACD替代月线 (响应快4倍，对加密币更合适)
- 多重确认入场避免过早进场
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (DIF, DEA, MACD柱)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif, dea, macd


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift()).abs()
    low_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线重采样为周线"""
    if "ts" not in df.columns:
        return df
    weekly = df.set_index("ts").resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return weekly


class TrendFilterStrategy(Strategy):
    """基于趋势过滤的策略实现.
    
    参数:
        ma_period: 主入场均线周期 (默认60)
        atr_mult_zone: 回踩容错区间 (默认1.5 * ATR)
        vol_mult: 量能确认倍数 (默认1.5x 5日均量)
        breakout_close_pct: 真破位收盘价跌幅 (默认1%)
        breakout_persist_bars: 破位持续K线数 (默认2)
        tier1_profit: 第一阶梯止盈 (默认30%)
        tier2_profit: 第二阶梯止盈 (默认50%)
        tier1_reduce: 第一次减仓比例 (默认1/3)
        tier2_reduce: 第二次减仓比例 (默认1/3)
        trail_pct: 追踪止损回撤 (默认5%)
        use_weekly_macd: 用周线MACD代替月线 (推荐True)
        require_macd_filter: 启用MACD大趋势过滤
    """
    
    name = "trend_filter"
    
    def __init__(
        self,
        ma_period: int = 60,
        atr_mult_zone: float = 1.5,
        vol_mult: float = 1.5,
        breakout_close_pct: float = 0.01,
        breakout_persist_bars: int = 2,
        tier1_profit: float = 0.30,
        tier2_profit: float = 0.50,
        tier1_reduce: float = 0.333,
        tier2_reduce: float = 0.333,
        trail_pct: float = 0.05,
        use_weekly_macd: bool = True,
        require_macd_filter: bool = True,
    ):
        self.ma_period = ma_period
        self.atr_mult_zone = atr_mult_zone
        self.vol_mult = vol_mult
        self.breakout_close_pct = breakout_close_pct
        self.breakout_persist_bars = breakout_persist_bars
        self.tier1_profit = tier1_profit
        self.tier2_profit = tier2_profit
        self.tier1_reduce = tier1_reduce
        self.tier2_reduce = tier2_reduce
        self.trail_pct = trail_pct
        self.use_weekly_macd = use_weekly_macd
        self.require_macd_filter = require_macd_filter
    
    def _macd_filter_series(self, df: pd.DataFrame) -> pd.Series:
        """计算每个时点的"大环境多头"标记 (周线MACD金叉)."""
        if not self.require_macd_filter:
            return pd.Series(True, index=df.index)
        
        if self.use_weekly_macd:
            weekly = _resample_to_weekly(df)
            if len(weekly) < 30:
                return pd.Series(True, index=df.index)  # 数据不够,默认放行
            dif_w, dea_w, _ = _macd(weekly["close"])
            weekly["bull"] = dif_w > dea_w
            # 把周线信号映射回日线
            weekly_indexed = weekly.set_index("ts")["bull"]
            df_indexed = df.set_index("ts")
            df_indexed["macd_bull"] = weekly_indexed.reindex(df_indexed.index, method="ffill")
            return df_indexed["macd_bull"].fillna(False).reset_index(drop=True)
        else:
            dif, dea, _ = _macd(df["close"])
            return dif > dea
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        if n < self.ma_period + 30:
            sides = np.full(n, "flat", dtype=object)
            stops = np.full(n, np.nan)
            return pd.DataFrame({"side": sides, "stop": stops}, index=df.index)
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        opens = df["open"].values
        volume = df["volume"].values
        
        ma60 = pd.Series(close).rolling(self.ma_period).mean().values
        atr = _atr(df, 14).values
        vol_avg5 = pd.Series(volume).rolling(5).mean().values
        macd_bull = self._macd_filter_series(df).values
        
        sides = np.full(n, "flat", dtype=object)
        stops = np.full(n, np.nan)
        
        position = "flat"  # flat / long_full / long_2of3 / long_1of3
        entry_px = 0.0
        sl_px = 0.0
        peak_px = 0.0  # 用于追踪止损
        breakout_count = 0  # 破位计数
        
        # i 表示当前K线索引,决策基于 i-1 (已闭合) 的数据
        for i in range(self.ma_period + 30, n):
            c_prev = close[i - 1]
            o_prev = opens[i - 1]
            h_prev = high[i - 1]
            l_prev = low[i - 1]
            ma_prev = ma60[i - 1]
            atr_prev = atr[i - 1]
            vol_prev = volume[i - 1]
            vol_avg = vol_avg5[i - 1]
            bull_env = macd_bull[i - 1]
            
            if np.isnan(ma_prev) or np.isnan(atr_prev) or np.isnan(vol_avg):
                sides[i] = position
                continue
            
            # ============ 持仓管理 ============
            if position != "flat":
                # 更新峰值价
                peak_px = max(peak_px, h_prev)
                
                # 1. 真破位检测 (收盘价跌破 MA60 超过 breakout_close_pct, 持续 N 根)
                broken = c_prev < ma_prev * (1 - self.breakout_close_pct)
                if broken:
                    breakout_count += 1
                else:
                    breakout_count = 0
                
                if breakout_count >= self.breakout_persist_bars:
                    # 真破位,清仓
                    position = "flat"
                    sl_px = 0.0
                    breakout_count = 0
                    sides[i] = "flat"
                    stops[i] = np.nan
                    continue
                
                # 2. 阶梯止盈检查
                profit_pct = (c_prev - entry_px) / entry_px
                
                if position == "long_full":
                    if profit_pct >= self.tier1_profit:
                        # 第一次减仓 + 止损上移到成本价
                        position = "long_2of3"
                        sl_px = entry_px  # 保本
                        sides[i] = "long"  # 引擎只识别 long/short/flat,实际仓位由我们追踪
                        stops[i] = sl_px
                        continue
                
                if position == "long_2of3":
                    if profit_pct >= self.tier2_profit:
                        # 第二次减仓 + 开启追踪止损
                        position = "long_1of3"
                        sl_px = max(sl_px, peak_px * (1 - self.trail_pct))
                
                if position == "long_1of3":
                    # 持续追踪止损
                    sl_px = max(sl_px, peak_px * (1 - self.trail_pct))
                
                # 3. 检查止损 (触发即清仓)
                if l_prev <= sl_px:
                    position = "flat"
                    sl_px = 0.0
                    breakout_count = 0
                    sides[i] = "flat"
                    stops[i] = np.nan
                    continue
                
                sides[i] = "long"
                stops[i] = sl_px
                continue
            
            # ============ 空仓: 检查入场 ============
            # 入场条件 (必须同时满足):
            # 0. 月线/周线 MACD 多头环境
            # 1. 价格回踩到 MA60 ± atr_mult_zone * ATR
            # 2. 阳线 OR 长下影线
            # 3. 量能 > 5日均量 * vol_mult
            # 4. (我的补丁) 价格在 MA60 上方
            
            if not bull_env:
                sides[i] = "flat"
                continue
            
            zone = self.atr_mult_zone * atr_prev
            in_zone = abs(c_prev - ma_prev) <= zone
            above_ma = c_prev > ma_prev * 0.995  # 容许小幅在下方,但收盘要差不多回到上方
            
            if not in_zone or not above_ma:
                sides[i] = "flat"
                continue
            
            # 强力阳线: 收盘 > 前一日高点
            strong_bull = c_prev > high[i - 2] if i >= 2 else False
            
            # 长下影线: 影线长度 > 实体 2倍
            body = abs(c_prev - o_prev)
            lower_wick = min(c_prev, o_prev) - l_prev
            long_lower_wick = lower_wick > body * 2 and body > 0
            
            if not (strong_bull or long_lower_wick):
                sides[i] = "flat"
                continue
            
            # 量能确认
            if vol_prev < vol_avg * self.vol_mult:
                sides[i] = "flat"
                continue
            
            # ====== 触发入场 ======
            position = "long_full"
            entry_px = c_prev
            sl_px = ma_prev * (1 - self.breakout_close_pct)  # 初始止损在 MA60 下方
            peak_px = h_prev
            breakout_count = 0
            sides[i] = "long"
            stops[i] = sl_px
        
        return pd.DataFrame({"side": sides, "stop": stops}, index=df.index)
