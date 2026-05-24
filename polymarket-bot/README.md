# polymarket-bot

A short-cycle YES/NO trading bot for [Polymarket](https://polymarket.com/), driven by orderbook spread + spot momentum confirmation.

Built around the recurring 15-minute BTC/ETH up-down markets (e.g. `btc-updown-15m-<unix_ts>`), with auto-rotation between consecutive 15m windows. Works with any binary market by switching off the `recurring` flag.

> **Status:** paper-trading only by default. Live execution is a stub — wiring it up requires a Polygon wallet, USDC, and the `py-clob-client` SDK. See "Going Live" below.

## Strategy summary

Every `tick_interval_sec` (default 30s, dropped to 10s in the last 2 minutes), the bot:

1. Pulls YES / NO midpoints and `remaining_time` from Polymarket
2. Pulls 1-minute spot momentum from Binance (default `BTCUSDT`)
3. Runs three rules (mapped 1:1 from the user pseudocode):

| Rule | Trigger | Action |
|---|---|---|
| Scout | `remaining > 10min` AND cheap side `<= 0.45` AND momentum aligned | Buy small on cheap side |
| Lock spread | `YES + NO <= 0.97` AND `remaining > 3min` AND have scout position | Buy complement if `edge - fees >= 0.02` |
| Endgame | `remaining < 2min` AND unbalanced | Hedge half if spot disagrees, else hold |

All thresholds live in `config.yaml`.

## Setup (macOS)

```bash
# 1. Python 3.11+ (use Homebrew if you don't have it)
brew install python@3.11

# 2. Clone / unzip into a folder, then:
cd polymarket-bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env       # only needed for live mode
# edit config.yaml — at minimum set market.slug to a real Polymarket slug
```

## Picking a market

### Recurring 15-minute up/down markets (default)

Polymarket runs a continuous series of 15-minute BTC and ETH up/down markets. Their slugs follow a fixed pattern with a Unix timestamp aligned to 15-minute boundaries:

```
https://polymarket.com/event/btc-updown-15m-1779634800
                              └────────┬─────────┘
                              slug_template + end_unix
                              (1779634800 / 900 == 0 → :00/:15/:30/:45 UTC)
```

`config.yaml` ships with this enabled by default:

```yaml
market:
  recurring:
    enabled: true
    slug_template: "btc-updown-15m-{start_unix}"
    period_sec: 900       # 15 minutes
```

The bot computes the current aligned timestamp, plugs it into the template, and rotates automatically when each market settles. To trade the ETH series instead, change the template to `eth-updown-15m-{start_unix}`. For other periods, adjust `period_sec`.

### Pinning a specific market

Set `market.recurring.enabled: false` and either:

- Put a full slug into `market.slug` (e.g. `btc-updown-15m-1779634800` for one specific window), or
- Pre-resolve token IDs and paste them into `market.yes_token_id` / `market.no_token_id` to skip the Gamma lookup entirely.

## Running (paper mode)

```bash
source .venv/bin/activate
python -m src.main
```

You'll see structured log lines per tick. Position state is held in memory only — restart = fresh state.

Stop with `Ctrl+C` (graceful shutdown).

## Going live

**Read this before flipping the switch.**

Polymarket settles on Polygon. Live execution requires:

1. A Polygon wallet (EOA) funded with USDC + a tiny bit of MATIC for gas
2. API credentials from Polymarket CLOB (`POLYGON_PRIVATE_KEY`, `CLOB_API_KEY`, `CLOB_SECRET`, `CLOB_PASSPHRASE`)
3. `pip install py-clob-client`
4. Implement `LiveExecutor` in `src/executor.py` (currently raises `NotImplementedError`)
5. Set `execution.mode: "live"` in `config.yaml`

Risks the strategy does **not** mitigate, that you must understand before going live:

- **Hedge leg may not fill.** If `total <= 0.97` looks like free spread but the complement side has no liquidity, you're stuck holding directional risk.
- **Endgame slippage.** Books thin out in the last minutes. The 0.5% slippage estimate in config is optimistic.
- **Resolution risk.** Polymarket markets are oracle-resolved. Read each market's resolution criteria; weird outcomes happen.
- **Polygon RPC issues.** Tx may stick or revert. Retries are not implemented.

Start with $10–20 of paper-mode-equivalent size before scaling.

## Project layout

```
polymarket-bot/
├── config.yaml               # all knobs, edit this
├── requirements.txt
├── .env.example
└── src/
    ├── main.py               # entry point + adaptive loop
    ├── config.py             # YAML loader
    ├── polymarket_client.py  # Gamma + CLOB REST adapters
    ├── spot_client.py        # Binance public klines
    ├── state.py              # position tracking
    ├── executor.py           # PaperExecutor + LiveExecutor stub
    └── strategy.py           # three-rule decision logic
```
