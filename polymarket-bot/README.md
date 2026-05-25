# polymarket-bot

A short-cycle YES/NO trading bot for [Polymarket](https://polymarket.com/) recurring 15-minute up/down markets, driven by orderbook spread + spot momentum confirmation, with execution-quality guards, risk guardrails, and full CSV journaling.

> **Status:** paper-trading only by default. Live execution is a stub — wiring it up requires a Polygon wallet, USDC, and the `py-clob-client` SDK. See "Going Live" below.

## Beginner quick start (macOS)

```bash
cd polymarket-bot
./run.sh                 # creates venv, installs deps, runs preflight + bot
```

If you only want to verify everything works without trading:

```bash
./run.sh --check         # preflight only, exit
```

That's it. Stop with `Ctrl+C`. After a session, open `logs/trades.csv` and `logs/ticks.csv` in Numbers/Excel to see exactly what happened.

## Strategy summary (5 rules, in priority order)

Every tick the bot pulls YES/NO midpoints, the L2 orderbook, and 1m spot klines (multi-timeframe + RSI + volume). Then it evaluates rules in this order — first match wins:

| # | Rule | Trigger | Action |
|---|---|---|---|
| 1 | **STOP_LOSS** | A held side is down more than `stop_loss_pct` (default 1.5%) | Sell that leg |
| 2 | **INVERTED_EXIT** | `total > 1.02` AND spot has reversed `>0.3%` against our larger side | Sell the losing leg |
| 3 | **ENDGAME_HEDGE** | `remaining < 4min` AND unbalanced AND spot disagrees with the bigger side | Buy `endgame_hedge_ratio` of the smaller side |
| 4 | **SCOUT** | `remaining > 11min` AND cheap side <= `scout_max_price` AND multi-tf momentum confirms | Buy small on cheap side |
| 5 | **LOCK_SPREAD** | `YES + NO <= 0.97` AND `remaining > 4min` AND have scout AND `edge >= 0.025` | Buy complement to balance |

All thresholds in `config.yaml`. The strategy emits at most **one** intent per tick.

## Safety features

- **Trading session gate** (`session.*`): default UTC 13:00-01:00 weekdays only. Outside the window the bot sleeps. **However it will still run STOP_LOSS / INVERTED_EXIT / ENDGAME_HEDGE if you have an open position** — those reduce risk.
- **Daily kill switch** (`risk.daily_max_loss_pct/usd`): default 2.5% of starting equity. Once hit, no new opening trades until UTC day rollover. Closing trades still fire.
- **Cash floor** (`risk.min_cash_floor_usd`): refuses opening trades when free cash drops below this.
- **Per-trade cap** (`risk.max_single_trade_usd`): silently downsizes any trade above this notional.
- **Stop-loss** (`strategy.stop_loss_pct`): closes a losing leg before it gets worse.
- **Inverted-book early exit** (`strategy.inverted_total_threshold`): bails out when `total > 1.02` AND spot has reversed against us — book is degraded, take the small loss now instead of letting it grow.
- **Spread / depth guard** (`execution_quality.*`): refuses entries on wide bid-ask or thin top-of-book.
- **Pre-flight check**: validates Polymarket, Binance, session config, and risk config before starting. Fails fast with a clear message.

The bot can still lose money. These features just make catastrophic losses much harder.

## Observability

Two CSV files are auto-written to `logs/`:

**`logs/trades.csv`** — every fill (BUY or SELL):

```
timestamp_utc, mode, market_slug, side, action, shares, fill_price,
fee_usd, notional_usd, cash_after, yes_shares_after, no_shares_after, reason
```

**`logs/ticks.csv`** — every tick decision (this is the most useful file):

```
timestamp_utc, market_slug, yes_price, no_price, total, remaining_sec,
spot_last, spot_short_return, spot_long_return, spot_rsi, spot_volume_ratio,
momentum_score, decision, yes_shares, no_shares, cash_usd, equity_usd
```

The `decision` column tells you *why* the bot did or didn't act every 30 seconds:

- `SCOUT` / `LOCK_SPREAD` / `ENDGAME_HEDGE` / `STOP_LOSS_YES` / `INVERTED_EXIT` — actually traded
- `HOLD_scout_too_late(204s)` — not enough time left for a scout
- `HOLD_scout_price_high(NO=0.490)` — cheap side above `scout_max_price`
- `HOLD_scout_momentum(short=+0.001,long=-0.0008)` — multi-tf momentum doesn't confirm
- `HOLD_scout_book_spread_wide(0.05)` — bid-ask too wide
- `HOLD_scout_book_depth_thin($30)` — top-of-book too thin
- `HOLD_arb_no_spread(total=1.0000)` — no arbitrage edge
- `HOLD_arb_edge_thin(0.018)` — edge after fees below `min_profit_after_fees`
- `BLOCKED_daily kill switch engaged` — risk veto
- `BLOCKED_off_hours` — outside session

A periodic console summary (`logging.summary_interval_sec`) prints the same info every 60s so you don't need to tail the CSV.

## Multi-asset cascade

Default config trades BTC 15m first, falls back to ETH 15m or SOL 15m if BTC isn't listed:

```yaml
market:
  recurring:
    slug_templates:
      - "btc-updown-15m-{start_unix}"
      - "eth-updown-15m-{start_unix}"
      - "sol-updown-15m-{start_unix}"
    period_sec: 900
```

Reorder, remove, or add templates as you like. Note: BTC/ETH usually have the deepest books; SOL is thinner so the depth guard may block more often.

## Tuning the defaults

After running for a session, count which `HOLD_*` reasons appear most in `ticks.csv`. Then:

| If you see mostly | Try |
|---|---|
| `HOLD_scout_price_high` | Bump `scout_max_price` slightly (0.47 -> 0.48). Edge drops fast above 0.5. |
| `HOLD_scout_momentum` | Lower `spot.alignment_threshold` (0.0035 -> 0.0025) — but expect more whipsaws. |
| `HOLD_arb_no_spread` | Normal. The book is fair-priced. Don't loosen — that's the whole point. |
| `HOLD_*_book_depth_thin` | Market too thin for your size. Reduce `scout_size_usd` or stick with BTC. |
| `STOP_LOSS_*` firing a lot | Either size is too big or you're entering at fair coin. Tighten `scout_max_price`. |

**Change one thing at a time and let it run for several markets before re-tuning.**

## Going live

**Read this before flipping the switch.**

Polymarket settles on Polygon. Live execution requires:

1. A Polygon wallet (EOA) funded with USDC + a tiny bit of MATIC for gas
2. API credentials from Polymarket CLOB (`POLYGON_PRIVATE_KEY`, `CLOB_API_KEY`, `CLOB_SECRET`, `CLOB_PASSPHRASE`) — copy `.env.example` to `.env`
3. `pip install py-clob-client`
4. Implement `LiveExecutor.buy` and `LiveExecutor.sell` in `src/executor.py`
5. Set `execution.mode: "live"` in `config.yaml`

Risks the strategy does **not** mitigate, that you must understand before going live:

- **Hedge leg may not fill.** The depth guard helps but doesn't eliminate this.
- **Endgame slippage.** Books thin out in the last minutes. `slippage_estimate=0.008` is conservative for normal play but optimistic for endgame.
- **Resolution risk.** Polymarket markets are oracle-resolved. Read each market's resolution criteria.
- **Polygon RPC issues.** Tx may stick or revert. Retries are not implemented.
- **Strategy edge is unproven.** Paper-trade for at least a week and look at `trades.csv` real P/L before live size.

Start with $10-20 of paper-equivalent size before scaling.

## Project layout

```
polymarket-bot/
|- run.sh                    one-command launcher
|- config.yaml               all knobs, edit this
|- requirements.txt
|- .env.example              live mode credentials template
|- logs/                     auto-created - trades.csv + ticks.csv
`- src/
   |- main.py                entry point + adaptive loop + session gate
   |- config.py              YAML loader
   |- session.py             UTC hours / weekday gate
   |- polymarket_client.py   Gamma + CLOB (cascade slug resolver, /book)
   |- spot_client.py         Binance public klines (multi-tf + RSI + volume)
   |- state.py               positions + cash + entry context for stop-loss
   |- strategy.py            5 rules: stop-loss / inverted-exit / endgame / scout / lock
   |- executor.py            PaperExecutor (buy + sell) + LiveExecutor stub
   |- journal.py             CSV writer for trades + ticks
   |- risk.py                daily kill switch (% or $) + per-trade caps
   `- preflight.py           startup health checks
```

## Running directly (without run.sh)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main                     # normal run
python -m src.main --preflight-only    # health check, exit
python -m src.main --once              # trade one 15m window then exit
python -m src.main --config foo.yaml   # different config file
```

## Reality check

This strategy buys cheap directional bets and tries to lock in spread when the market becomes briefly mispriced. It is **not** a guaranteed-profit machine:

- The "lock spread" leg only +EVs **if** the complement leg fills. In thin books or the last 2 minutes, that's not guaranteed.
- Spot momentum on a 15m horizon is noisy. Multi-timeframe + RSI helps but does not eliminate false signals.
- Polymarket recurring markets attract bots; the easy spreads close fast.
- Without backtesting against historical fills, all parameters are educated guesses.

The honest path: paper-trade for at least a week, look at `trades.csv` to compute your realized win rate and average per-trade P/L, then decide whether to go live with small size. If the paper P/L is negative or noisy, no amount of tweaking will save it.
