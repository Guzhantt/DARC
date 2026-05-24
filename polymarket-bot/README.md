# polymarket-bot

A short-cycle YES/NO trading bot for [Polymarket](https://polymarket.com/), driven by orderbook spread + spot momentum confirmation.

Built around the recurring 15-minute BTC/ETH up-down markets (e.g. `btc-updown-15m-<unix_ts>`), with auto-rotation between consecutive 15m windows. Works with any binary market by switching off the `recurring` flag.

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

That's it. Stop with `Ctrl+C`. After a session, open `logs/trades.csv` and `logs/ticks.csv` in Numbers/Excel/Google Sheets to see exactly what happened.

## Strategy summary

Every `tick_interval_sec` (default 30s, dropped to 10s in the last 2 minutes), the bot:

1. Pulls YES / NO midpoints, **L2 orderbook**, and `remaining_time` from Polymarket
2. Pulls 1-minute spot momentum from Binance (default `BTCUSDT`, with Binance.US fallback)
3. Runs three rules (mapped 1:1 from the original pseudocode):

| Rule | Trigger | Action |
|---|---|---|
| Scout | `remaining > 10min` AND cheap side `<= 0.45` AND momentum aligned | Buy small on cheap side |
| Lock spread | `YES + NO <= 0.97` AND `remaining > 3min` AND have scout position | Buy complement if `edge - fees >= 0.02` |
| Endgame | `remaining < 2min` AND unbalanced | Hedge half if spot disagrees, else hold |

Every entry is also gated by **execution-quality guards** and **risk guardrails** (see below).

## Safety features

These are what stop a buggy strategy from blowing up your account:

- **Daily kill switch** (`risk.daily_max_loss_usd`, default $10): once equity drops more than this from the day's start, the bot refuses new entries until the UTC day rolls over. Existing positions still finish (endgame hedge can still fire — that *reduces* risk).
- **Cash floor** (`risk.min_cash_floor_usd`, default $5): refuse to open new entries when free cash drops below this.
- **Per-trade cap** (`risk.max_single_trade_usd`, default $20): silently downsizes any requested trade above this notional.
- **Spread / depth guard** (`execution_quality.*`): refuses to enter on a wide bid-ask spread or a thin top-of-book. Prevents "looks great at midpoint, fills 5% worse" surprises.
- **Preflight check**: validates Polymarket, Binance, config sanity, and (in live mode) credentials before the loop starts. Fails fast with a clear message.

The bot can still lose money. These features just make catastrophic losses much harder.

## Observability

Two CSV files are written to `logs/` (auto-created):

- **`logs/trades.csv`** — one row per fill: timestamp, side, shares, fill price, fees, cash after, position after, reason
- **`logs/ticks.csv`** — one row per tick: prices, momentum, decision label (e.g. `SCOUT`, `HOLD_scout_momentum_against`, `LOCK_SPREAD`, `ENDGAME_HEDGE`, `BLOCKED_daily kill switch engaged`)

The decision label is the most useful debugging tool: it tells you *why* the bot did or did not act on every tick.

A periodic summary (every `logging.summary_interval_sec`, default 60s) also prints to console so you don't need to tail the CSV to see what's happening.

## Picking a market

### Recurring 15-minute up/down markets (default)

Polymarket runs a continuous series of 15-minute BTC and ETH up/down markets. Their slugs follow a fixed pattern with a Unix timestamp aligned to 15-minute boundaries:

```
https://polymarket.com/event/btc-updown-15m-1779634800
                              └────────┬─────────┘
                              slug_template + start_unix
                              (1779634800 / 900 == 0 → :00/:15/:30/:45 UTC start)
```

`config.yaml` ships with this enabled by default. To trade ETH instead, change `slug_template` to `eth-updown-15m-{start_unix}`.

### Pinning a specific market

Set `market.recurring.enabled: false` and either:

- Put a full slug into `market.slug`, or
- Pre-resolve token IDs and paste them into `market.yes_token_id` / `market.no_token_id` to skip the Gamma lookup entirely.

## Tuning the defaults

Defaults are conservative on purpose. After running for a session, look at `logs/ticks.csv` and count which `HOLD_*` reasons appear most often. Then:

| If the bot rarely enters | Try |
|---|---|
| Mostly `HOLD_scout_price_high` | Raise `scout_max_price` slightly (e.g. 0.45 → 0.48). Be careful: edge thins quickly above 0.5. |
| Mostly `HOLD_scout_momentum_against` | Lower `spot.alignment_threshold` (e.g. 0.0005 → 0.0003) to accept weaker signals — but expect more noise. |
| Mostly `HOLD_arb_no_spread` | The book is fair-priced; this is normal. Don't loosen — the whole point is to wait for spread. |
| Mostly `HOLD_*_book_depth_thin` | The market is too illiquid for your size. Reduce `scout_size_usd`, or move to a more active market. |

| If the bot enters too often / loses | Try |
|---|---|
| Frequent endgame hedges firing | Lower `endgame_hedge_ratio` (0.5 → 0.3) so you hedge less. |
| Daily kill switch keeps firing | Either reduce `scout_size_usd` or accept that this strategy isn't a fit for current market conditions. |

**Always change one thing at a time and let it run for several markets before re-tuning.**

## Going live

**Read this before flipping the switch.**

Polymarket settles on Polygon. Live execution requires:

1. A Polygon wallet (EOA) funded with USDC + a tiny bit of MATIC for gas
2. API credentials from Polymarket CLOB (`POLYGON_PRIVATE_KEY`, `CLOB_API_KEY`, `CLOB_SECRET`, `CLOB_PASSPHRASE`) — copy `.env.example` to `.env`
3. `pip install py-clob-client`
4. Implement `LiveExecutor` in `src/executor.py` (currently raises `NotImplementedError`)
5. Set `execution.mode: "live"` in `config.yaml`

Risks the strategy does **not** mitigate, that you must understand before going live:

- **Hedge leg may not fill.** If `total <= 0.97` looks like free spread but the complement side has no liquidity, you're stuck holding directional risk. The depth guard helps but doesn't eliminate this.
- **Endgame slippage.** Books thin out in the last minutes. The 0.5% slippage estimate in config is optimistic.
- **Resolution risk.** Polymarket markets are oracle-resolved. Read each market's resolution criteria; weird outcomes happen.
- **Polygon RPC issues.** Tx may stick or revert. Retries are not implemented.
- **Strategy edge is unproven.** Paper-trade for at least a week and look at `trades.csv` before considering live size.

Start with $10–20 of paper-mode-equivalent size before scaling.

## Project layout

```
polymarket-bot/
├── run.sh                    one-command launcher (recommended entry point)
├── config.yaml               all knobs, edit this
├── requirements.txt
├── .env.example              live mode credentials template
├── logs/                     auto-created — trades.csv + ticks.csv
└── src/
    ├── main.py               entry point + adaptive loop + preflight + summary
    ├── config.py             YAML loader
    ├── polymarket_client.py  Gamma + CLOB REST adapters (slug rotation, /book)
    ├── spot_client.py        Binance public klines (.com → .us fallback)
    ├── state.py              position + cash + mark-to-market
    ├── strategy.py           three-rule decision logic + execution-quality guards
    ├── executor.py           PaperExecutor + LiveExecutor stub (journals fills)
    ├── journal.py            CSV writer for trades + ticks
    ├── risk.py               daily kill switch + per-trade caps
    └── preflight.py          startup health checks
```

## Running directly (without run.sh)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main                     # normal run
python -m src.main --preflight-only    # health check, exit
python -m src.main --once              # trade one 15m window then exit
```
