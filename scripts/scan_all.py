"""Backtest a strategy across many top-volume USDT perpetuals, rank results.

Example:
    python -m scripts.scan_all --timeframe 1h --days 30 --strategy oi_composite --top 30
    python -m scripts.scan_all --timeframe 1h --days 90 --strategy donchian --top 30
"""
from __future__ import annotations

import argparse

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.backtest.engine import BacktestConfig, run_backtest
from src.config import load_config
from src.data.data_loader import load_ohlcv
from src.data.enhanced_loader import load_enhanced_ohlcv
from src.exchange.binance_client import BinanceFutures
from src.strategies import get_strategy

# Strategies that require enhanced (OI + on-chain) data
ENHANCED_STRATEGIES = {"oi_composite"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--strategy", default="ma_cross")
    p.add_argument("--top", type=int, default=30, help="how many top-volume symbols to test")
    p.add_argument("--min-trades", type=int, default=5)
    p.add_argument("--sort", default="sharpe", choices=["sharpe", "total_return", "profit_factor", "win_rate"])
    args = p.parse_args()

    cfg = load_config()
    client = BinanceFutures(cfg)

    use_enhanced = args.strategy in ENHANCED_STRATEGIES
    if use_enhanced and args.days > 30:
        print(f"WARNING: OI data limited to ~30 days. Using days={args.days} but OI "
              f"columns will be NaN for older bars.")

    print(f"Fetching top {args.top} USDT perpetuals by 24h volume...")
    symbols = client.fetch_top_volume_symbols(n=args.top)
    print(f"Got {len(symbols)} symbols. Strategy: {args.strategy} "
          f"({'enhanced OI+chain data' if use_enhanced else 'price-only'})")

    rows: list[dict] = []
    for sym in tqdm(symbols, desc="Backtesting"):
        try:
            if use_enhanced:
                df = load_enhanced_ohlcv(client, sym, args.timeframe, args.days, cfg.data_dir)
            else:
                df = load_ohlcv(client, sym, args.timeframe, args.days, cfg.data_dir)
            if len(df) < 100:
                continue
            strat = get_strategy(args.strategy)
            res = run_backtest(df, strat, BacktestConfig())
            m = res.metrics
            if m.trades < args.min_trades:
                continue
            rows.append({"symbol": sym, **m.as_dict()})
        except Exception as e:
            tqdm.write(f"skip {sym}: {e}")

    if not rows:
        print("No results.")
        return

    table = pd.DataFrame(rows).sort_values(args.sort, ascending=False)
    print("\n=== Ranking ===")
    print(tabulate(table, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    out = cfg.logs_dir / f"scan_{args.strategy}_{args.timeframe}_{args.days}d.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
