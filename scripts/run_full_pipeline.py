"""完整 Pipeline 演示脚本.

四步流程:
1. 拉取多币种历史日线数据
2. Walk-forward 优化 (检查参数稳健性)
3. 自适应币种筛选 (前12月画像 → 推荐币种 + 参数)
4. 多币种组合回测 (后6月 out-of-sample 验证)

用法:
  python -m scripts.run_full_pipeline
  python -m scripts.run_full_pipeline --coins BTC,ETH,SOL --days 540
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import ccxt
import pandas as pd

warnings.filterwarnings("ignore")

from src.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from src.optimization.coin_selector import select_coins, format_profile_report
from src.optimization.walk_forward import format_wf_report, walk_forward_optimize
from src.strategies import get_strategy


def fetch_data(symbols: list[str], days: int, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """拉取多币种历史数据 (带缓存)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    coin_data = {}
    exchange = ccxt.binanceus({"enableRateLimit": True})
    exchange.load_markets()

    for sym in symbols:
        cache_file = cache_dir / f"{sym.replace('/', '_')}_1d.pkl"
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            if len(df) >= days * 0.9:
                coin_data[sym.split("/")[0]] = df
                continue

        print(f"  拉取 {sym}...", end=" ", flush=True)
        try:
            all_ohlcv = []
            since = exchange.milliseconds() - days * 86400000
            while True:
                batch = exchange.fetch_ohlcv(sym, "1d", since=since, limit=1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                if len(batch) < 1000:
                    break
                since = batch[-1][0] + 86400000
                time.sleep(0.3)
            df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            df.to_pickle(cache_file)
            coin_data[sym.split("/")[0]] = df
            print(f"{len(df)}天")
        except Exception as e:
            print(f"失败: {e}")
    return coin_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,LINK/USDT")
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--train-ratio", type=float, default=0.66)
    parser.add_argument("--cache-dir", default="/tmp")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.coins.split(",")]

    print("=" * 70)
    print(" 趋势过滤策略 — 完整 Pipeline 测试")
    print("=" * 70)

    # === 步骤1: 拉数据 ===
    print(f"\n[1/4] 拉取 {len(symbols)} 个币种 {args.days} 天数据...")
    coin_data = fetch_data(symbols, args.days, Path(args.cache_dir))
    print(f"  完成: {len(coin_data)} 个币种")

    if not coin_data:
        return

    # === 步骤2: Walk-forward 验证 (诚实对比训练/测试) ===
    print(f"\n[2/4] Walk-Forward 参数优化 (训练比例 {args.train_ratio*100:.0f}%)...")
    print("  说明: 检查参数稳健性. 训练-测试 Sharpe 差距大 = 过拟合.")

    param_grid = {
        "ma_period": [50, 60, 80],
        "atr_mult_zone": [1.0, 1.5, 2.0],
        "vol_mult": [1.2, 1.5, 1.8],
        "use_weekly_macd": [True],
        "require_macd_filter": [True],
    }

    def factory_with_params(params):
        return get_strategy("trend_filter", **params)

    for sym, df in coin_data.items():
        try:
            res = walk_forward_optimize(
                df, sym, factory_with_params, param_grid,
                train_ratio=args.train_ratio, objective="sharpe",
            )
            print()
            print(format_wf_report(res))
        except Exception as e:
            print(f"  {sym}: walk-forward 失败 ({e})")

    # === 步骤3: 币种筛选 + 自适应参数 ===
    print(f"\n\n[3/4] 币种画像与筛选 (用通用稳健参数)...")
    profiles, params_map = select_coins(
        coin_data, train_ratio=args.train_ratio,
        min_sharpe=0.5, min_return=0.0, min_volume_usd=100_000,
    )
    print(format_profile_report(profiles))
    print()

    if not params_map:
        print("\n  ⚠️  没有币种通过训练期筛选. 该策略不适合当前市场.")
        return

    # === 步骤4: 测试期组合回测 ===
    print(f"\n[4/4] 测试期组合回测 (后 {(1-args.train_ratio)*100:.0f}%, out-of-sample)...")
    n = len(next(iter(coin_data.values())))
    test_start = int(n * args.train_ratio)
    recommended = list(params_map.keys())
    test_data = {sym: coin_data[sym].iloc[test_start:].reset_index(drop=True) for sym in recommended}

    print(f"  推荐币种 ({len(recommended)}个): {recommended}")
    print()

    def factory_per_coin(sym):
        return get_strategy("trend_filter", **params_map[sym])

    cfg = PortfolioConfig(
        initial_equity=10000,
        max_positions=min(5, len(recommended)),
        max_pct_per_symbol=0.30,
        risk_per_trade=0.02,
        max_total_leverage=2.0,
        daily_dd_halt=0.10,
    )
    result = run_portfolio_backtest(test_data, factory_per_coin, cfg)
    print(result.summary())

    # 对比基准
    print("\n  --- 基准对比 ---")
    test_data_all = {
        sym: coin_data[sym].iloc[test_start:].reset_index(drop=True)
        for sym in coin_data
    }
    result_all = run_portfolio_backtest(test_data_all, lambda s: get_strategy("trend_filter"), cfg)
    print(f"    全币种默认参数: 收益 {result_all.metrics.total_return*100:+.2f}%, MaxDD {result_all.metrics.max_drawdown*100:.2f}%")
    print(f"    筛选后+自适应:  收益 {result.metrics.total_return*100:+.2f}%, MaxDD {result.metrics.max_drawdown*100:.2f}%")
    diff_ret = (result.metrics.total_return - result_all.metrics.total_return) * 100
    diff_dd = (result.metrics.max_drawdown - result_all.metrics.max_drawdown) * 100
    print(f"    改善: 收益 {diff_ret:+.2f}%, 回撤 {diff_dd:+.2f}%")


if __name__ == "__main__":
    main()
