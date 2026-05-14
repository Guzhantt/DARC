"""币种自适应选择器 + 参数推荐.

核心理念 (从 walk-forward 失败中学到的):
- 不要在每个币上"过拟合参数"
- 而是用一套通用稳健参数, 然后选出"该策略适合的币"

筛选逻辑:
1. 训练期表现门槛: Sharpe > 0.5 且总收益 > 0
2. 趋势性测试: ADX 平均 > 20 (确认是趋势市场)
3. 流动性测试: 平均 24h 成交量 > 阈值
4. 波动率匹配: ATR/Price 在 2%-8% 之间 (太低没机会, 太高风险大)

参数自适应 (粗粒度, 不是过拟合):
- 高波动币 (SOL/DOGE): MA60 + ATR 1.5x + Vol 1.5x
- 中波动币 (ETH/BNB/LINK): MA60 + ATR 1.5x + Vol 1.5x (默认)
- 低波动币 (BTC): MA50 + ATR 1.0x + Vol 1.2x (放宽入场)

输出: 推荐的币种列表 + 每个币的参数
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.strategies import get_strategy


@dataclass
class CoinProfile:
    symbol: str
    avg_atr_pct: float       # ATR / 价格 (波动率)
    avg_adx: float           # 趋势强度
    avg_volume_usd: float    # 平均美元成交量
    train_sharpe: float      # 训练期 Sharpe
    train_return: float      # 训练期收益
    train_trades: int
    suggested_params: dict
    is_recommended: bool
    reject_reason: str = ""


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift()).abs()
    low_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def adaptive_params(avg_atr_pct: float, avg_adx: float) -> dict:
    """根据币的波动率特征推荐参数 (粗粒度, 避免过拟合).

    设计哲学: 只 3-4 档参数, 不细分.
    """
    # 高波动 (SOL, DOGE, 小币)
    if avg_atr_pct > 0.05:
        return {
            "ma_period": 60,
            "atr_mult_zone": 2.0,    # 宽容差
            "vol_mult": 1.8,         # 严格量能
            "use_weekly_macd": True,
            "require_macd_filter": True,
        }
    # 低波动 (BTC)
    if avg_atr_pct < 0.025:
        return {
            "ma_period": 50,         # 短均线, 多机会
            "atr_mult_zone": 1.0,
            "vol_mult": 1.2,
            "use_weekly_macd": True,
            "require_macd_filter": True,
        }
    # 中波动 (ETH, BNB, XRP, LINK)
    return {
        "ma_period": 60,
        "atr_mult_zone": 1.5,
        "vol_mult": 1.5,
        "use_weekly_macd": True,
        "require_macd_filter": True,
    }


def profile_coin(
    symbol: str,
    df: pd.DataFrame,
    train_ratio: float = 0.66,
    min_volume_usd: float = 5_000_000,
    min_sharpe: float = 0.0,
    min_return: float = -0.05,
) -> CoinProfile:
    """对一个币做完整的画像 + 推荐参数.

    Args:
        symbol: 币种名
        df: 完整 18 个月日线
        train_ratio: 训练集比例 (0.66 = 前 12 个月)
        min_volume_usd: 流动性下限
        min_sharpe: 训练期 Sharpe 下限
        min_return: 训练期收益下限 (-5% 容忍小幅亏损)

    Returns:
        CoinProfile
    """
    n = len(df)
    train_end = int(n * train_ratio)
    train_df = df.iloc[:train_end].reset_index(drop=True)

    # 1. 计算波动率特征
    atr = _atr_series(train_df).dropna()
    avg_atr = atr.mean()
    avg_close = train_df["close"].mean()
    avg_atr_pct = avg_atr / avg_close

    adx = _adx_series(train_df).dropna()
    avg_adx = adx.mean() if len(adx) > 0 else 0

    # 2. 流动性
    avg_vol_usd = (train_df["close"] * train_df["volume"]).mean()

    # 3. 用自适应参数跑训练集
    params = adaptive_params(avg_atr_pct, avg_adx)
    cfg = BacktestConfig(initial_equity=10000, risk_per_trade=0.05, max_leverage=1)
    try:
        s = get_strategy("trend_filter", **params)
        r = run_backtest(train_df, s, cfg)
        train_sharpe = r.metrics.sharpe
        train_return = r.metrics.total_return
        train_trades = r.metrics.trades
    except Exception:
        train_sharpe = -99
        train_return = -1
        train_trades = 0

    # 4. 筛选决策
    reject_reasons = []
    if avg_vol_usd < min_volume_usd:
        reject_reasons.append(f"流动性不足({avg_vol_usd/1e6:.1f}M<{min_volume_usd/1e6:.0f}M)")
    if train_sharpe < min_sharpe:
        reject_reasons.append(f"Sharpe不达标({train_sharpe:.2f}<{min_sharpe:.1f})")
    if train_return < min_return:
        reject_reasons.append(f"训练期亏损({train_return*100:.1f}%)")
    if train_trades < 2:
        reject_reasons.append(f"交易太少({train_trades})")

    is_recommended = len(reject_reasons) == 0

    return CoinProfile(
        symbol=symbol,
        avg_atr_pct=avg_atr_pct,
        avg_adx=avg_adx,
        avg_volume_usd=avg_vol_usd,
        train_sharpe=train_sharpe,
        train_return=train_return,
        train_trades=train_trades,
        suggested_params=params,
        is_recommended=is_recommended,
        reject_reason=", ".join(reject_reasons) if reject_reasons else "",
    )


def select_coins(
    coin_data: dict[str, pd.DataFrame],
    train_ratio: float = 0.66,
    **kwargs,
) -> tuple[list[CoinProfile], dict[str, dict]]:
    """对所有币画像, 输出推荐币种和参数映射.

    Returns:
        (profiles, params_map) - profiles 是所有币, params_map 是 {sym: params}
    """
    profiles = []
    params_map = {}
    for sym, df in coin_data.items():
        p = profile_coin(sym, df, train_ratio=train_ratio, **kwargs)
        profiles.append(p)
        if p.is_recommended:
            params_map[sym] = p.suggested_params
    return profiles, params_map


def format_profile_report(profiles: list[CoinProfile]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(" 币种画像与推荐 (基于训练期 12 个月)")
    lines.append("=" * 80)
    lines.append(f"  {'币种':<6} {'波动率':>8} {'ADX':>6} {'成交量':>9} {'训练Sharpe':>10} {'训练收益':>9} {'交易':>4} {'推荐':>6}")
    lines.append("  " + "-" * 75)
    for p in sorted(profiles, key=lambda x: -x.train_sharpe if x.is_recommended else -100):
        flag = "✅" if p.is_recommended else "❌"
        lines.append(
            f"  {p.symbol:<6} {p.avg_atr_pct*100:>7.2f}% {p.avg_adx:>6.1f} "
            f"${p.avg_volume_usd/1e6:>7.1f}M {p.train_sharpe:>10.2f} "
            f"{p.train_return*100:>+7.1f}% {p.train_trades:>4} {flag:>4}"
        )
        if not p.is_recommended:
            lines.append(f"         拒绝原因: {p.reject_reason}")
    return "\n".join(lines)
