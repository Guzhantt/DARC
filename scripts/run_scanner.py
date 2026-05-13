"""Run the autonomous full-market scanner.

This combines the trader's event-driven approach with our risk management:
- Scans all USDT perpetuals every 5 minutes
- Detects: funding extremes, crash bounces, pump shorts
- Validates with environment scoring (BTC + FGI + OI + Volume)
- Telegram notifications for all trades
- Network resilience with exponential backoff

Example:
    python -m scripts.run_scanner
    python -m scripts.run_scanner --max-positions 5 --interval 180
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.exchange.binance_client import BinanceFutures
from src.executor.autonomous_scanner import AutonomousScanner


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous market scanner")
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--position-pct", type=float, default=0.30, help="fraction of equity per trade")
    p.add_argument("--cooldown", type=int, default=4, help="hours between trades on same symbol")
    p.add_argument("--min-volume", type=float, default=10, help="min 24h volume in millions USD")
    p.add_argument("--interval", type=int, default=300, help="scan interval in seconds")
    p.add_argument("--min-score", type=int, default=3, help="min environment score to open (0-6)")
    args = p.parse_args()

    cfg = load_config()
    if not cfg.api_key or not cfg.api_secret:
        raise SystemExit(
            "BINANCE_API_KEY / BINANCE_API_SECRET not set.\n"
            "Copy .env.example to .env and fill in your keys.\n"
            "Use TESTNET keys first: https://testnet.binancefuture.com"
        )

    client = BinanceFutures(cfg)
    mode = "TESTNET" if cfg.use_testnet else "*** LIVE (REAL MONEY) ***"
    print(f"Starting autonomous scanner — {mode}")

    if not cfg.use_testnet:
        confirm = input("Type 'I UNDERSTAND' to continue with REAL money: ")
        if confirm.strip() != "I UNDERSTAND":
            raise SystemExit("Aborted.")

    scanner = AutonomousScanner(
        cfg=cfg,
        client=client,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
        cooldown_hours=args.cooldown,
        min_volume_m=args.min_volume,
        scan_interval=args.interval,
        min_env_score=args.min_score,
    )
    scanner.run_forever()


if __name__ == "__main__":
    main()
