"""Run a single-symbol strategy live (defaults to Testnet).

Example:
    python -m scripts.run_live --symbol BTC/USDT:USDT --timeframe 15m --strategy ma_cross
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.exchange.binance_client import BinanceFutures
from src.executor.live_executor import LiveExecutor
from src.strategies import get_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--strategy", default="ma_cross")
    args = p.parse_args()

    cfg = load_config()
    if not cfg.api_key or not cfg.api_secret:
        raise SystemExit(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set. "
            "Copy .env.example to .env and fill in TESTNET keys from "
            "https://testnet.binancefuture.com"
        )

    client = BinanceFutures(cfg)
    strat = get_strategy(args.strategy)
    mode = "TESTNET" if cfg.use_testnet else "*** LIVE (REAL MONEY) ***"
    print(f"Running {args.strategy} on {args.symbol} {args.timeframe} — {mode}")
    if not cfg.use_testnet:
        confirm = input("Type 'I UNDERSTAND' to continue with REAL money: ")
        if confirm.strip() != "I UNDERSTAND":
            raise SystemExit("Aborted.")

    LiveExecutor(cfg, client, strat, args.symbol, args.timeframe).run_forever()


if __name__ == "__main__":
    main()
