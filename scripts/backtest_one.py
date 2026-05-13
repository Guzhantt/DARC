"""Backtest a single symbol with a chosen strategy.

Example:
    python -m scripts.backtest_one --symbol BTC/USDT:USDT --timeframe 1h --days 180 --strategy ma_cross
"""
from __future__ import annotations

import argparse
import json

from tabulate import tabulate

from src.backtest.engine import BacktestConfig, run_backtest
from src.config import load_config
from src.data.data_loader import load_ohlcv
from src.exchange.binance_client import BinanceFutures
from src.strategies import get_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="e.g. BTC/USDT:USDT or BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--strategy", default="ma_cross")
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--leverage", type=float, default=3.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    client = BinanceFutures(cfg)

    print(f"Loading {args.days}d of {args.timeframe} data for {args.symbol}...")
    df = load_ohlcv(
        client, args.symbol, args.timeframe, args.days, cfg.data_dir, use_cache=not args.no_cache
    )
    if df.empty:
        raise SystemExit("No data returned.")

    strat = get_strategy(args.strategy)
    result = run_backtest(
        df,
        strat,
        BacktestConfig(
            initial_equity=args.initial_equity,
            risk_per_trade=args.risk,
            max_leverage=args.leverage,
        ),
    )
    print("\n=== Metrics ===")
    print(json.dumps(result.metrics.as_dict(), indent=2, default=str))

    if not result.trades.empty:
        print("\n=== Last 10 Trades ===")
        print(tabulate(result.trades.tail(10), headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
