# Binance Futures Trading Bot (Educational)

A modular Python framework for **backtesting** and **testnet-trading** crypto strategies on Binance USDT-M Futures.

> ## ⚠️ Disclaimer
> - This is **educational code**. No strategy in this repo has any proven edge.
> - Trading crypto futures is **extremely risky**. You can lose all your money.
> - **Always run on Testnet first.** Default config points to Binance Futures Testnet.
> - The author is not responsible for any losses. Use at your own risk.

## Features

- Binance USDT-M Futures client (via `ccxt`) with Testnet support
- Historical kline downloader with CSV cache
- Pluggable strategy interface (3 example strategies included)
- Event-driven backtester with fees, PnL, Sharpe, max drawdown, win rate
- Risk manager: position sizing, stop-loss, daily drawdown circuit breaker
- CLI scripts for single-pair backtest, multi-pair scan, and live testnet execution

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your testnet API keys (https://testnet.binancefuture.com)
```

## Usage

### 1. Single-pair backtest

```bash
python -m scripts.backtest_one --symbol BTC/USDT --timeframe 1h --days 180 --strategy ma_cross
```

### 2. Scan all top-volume pairs

```bash
python -m scripts.scan_all --timeframe 1h --days 90 --strategy rsi --top 30
```

### 3. Live trading on Testnet

```bash
python -m scripts.run_live --symbol BTC/USDT --timeframe 15m --strategy ma_cross
```

## Available strategies

| Name         | Description                                      |
|--------------|--------------------------------------------------|
| `ma_cross`   | Fast/slow SMA crossover (long + short)           |
| `rsi`        | RSI mean-reversion                               |
| `donchian`   | Donchian channel breakout (trend-following)      |

Add your own by subclassing `strategies.base.Strategy`.

## Project layout

See `README.md` comments. Everything lives under `src/` and `scripts/`.

## Important: how to actually find a "profitable" strategy

There is **no magic signal** that guarantees profit. Realistic workflow:

1. Pick a strategy template from `src/strategies/`.
2. Backtest it across many pairs and timeframes.
3. Look at the metrics: **Sharpe > 1**, **max drawdown reasonable**, **profit factor > 1.3**, and win rate + payoff ratio that combine to positive expectancy.
4. **Walk-forward validate** (split data into in-sample / out-of-sample).
5. If (and only if) it still looks good: paper trade on **Testnet** for weeks.
6. Only then, with money you can lose, consider small real-money deployment.

Do **not** skip these steps. Backtest profitability does not imply live profitability.
