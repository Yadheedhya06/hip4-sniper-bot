# HIP-4 BTC PriceBinary Sniper Bot

A real-time sniper bot for Hyperliquid HIP-4 BTC priceBinary outcome contracts. It monitors active binary contracts and places small buy orders on the winning side in the final minutes when there's a clear statistical edge.

## How It Works

HIP-4 priceBinary contracts settle to **$1** (Yes) or **$0** (No) based on whether BTC mark price is above or below a target price at expiry. In the final minutes before settlement, the outcome becomes increasingly predictable but market prices often lag — this bot exploits that edge.

### Statistical Model

Uses a normal approximation for the probability that BTC mark price ≥ strike at expiry:

```
d = (S - K) / (S * σ * √T)
P(Yes) = Φ(d)   # standard normal CDF
```

Where:
- `S` = current BTC mark price
- `K` = target/strike price from contract metadata
- `T` = time remaining as fraction of a year
- `σ` = annualized realized volatility (conservative estimate from recent 1-min returns)

### Entry Conditions (ALL must be true)

- Time remaining ≤ 5 minutes (ideally < 3 min)
- Model probability exceeds market mid by `edge_threshold` (default 0.10)
- |d| ≥ 1.8 (~96%+ model probability)
- Sufficient orderbook depth with < 3% slippage

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env
```

## Configuration

All knobs are env vars (see `.env.example`). The only required one is the signing key:

```
HYPERLIQUID_PRIVATE_KEY=0xYOUR_AGENT_WALLET_KEY
```

The bot uses **HIP-4 outcomeMeta** for discovery and addresses each side of every binary as `#<10*outcome+side>` (Yes = side 0, No = side 1). Settlement collateral is **USDH**.

### Sizing (env-driven)

| Var | Default | Purpose |
|---|---|---|
| `SIZE_PCT_OF_EQUITY` | 0.05 | Baseline % of equity per trade |
| `SIZE_MAX_PCT_OF_EQUITY` | 0.10 | Hard ceiling as % of equity |
| `SIZE_MIN_USD` | 10 | Floor per trade |
| `SIZE_MAX_USD` | 200 | Ceiling per trade |
| `TRAINING_WHEELS_MAX_USD` | 25 | Extra hard cap. Empty disables. |
| `EQUITY_FLOOR_USD` | 50 | Halt trading below this balance |
| `DAILY_LOSS_LIMIT_PCT` | 0.10 | Kill-switch when day P&L ≤ -10% |

### Operational controls

| Var | Default | Purpose |
|---|---|---|
| `DRY_RUN` | false | Log decisions, never sign |
| `PORT` | 8080 | Healthcheck HTTP port (Railway sets this) |
| `LOG_TO_STDOUT` / `LOG_TO_FILE` | true / false | |
| `HEARTBEAT_SEC` | 60 | Periodic status log cadence |

### Healthcheck

`GET /health` (or `/`) returns JSON with equity, BTC price, contracts tracked, kill-switch state, and day P&L anchor. Useful for Railway healthchecks and local debugging.

## Running locally

```bash
python sniper.py
```

The bot will:
1. Discover all active BTC priceBinary contracts every 30s
2. Stream real-time BTC price and orderbook data
3. Calculate rolling volatility from 1-minute candles
4. Evaluate edge on each contract as expiry approaches
5. Place limit orders when all entry conditions are met
6. Log every decision with full model parameters

## Logging

Stdout-first (Railway-friendly). UTC timestamps. Set `LOG_TO_FILE=true` to also write to `sniper.log`.

## Deploy to Railway

1. Connect this repo to a Railway project.
2. Set env vars in Railway:
   - `HYPERLIQUID_PRIVATE_KEY` (required, agent wallet preferred)
   - any sizing overrides from `.env.example`
   - leave `PORT` alone — Railway injects it.
3. Procfile is `worker: python -u sniper.py`. The `-u` flag is important so logs flush immediately.
4. **First deploy: set `DRY_RUN=true`.** Watch logs for one full daily cycle (24h around 06:00 UTC) to confirm the bot discovers the new outcome, sizes correctly, and would have placed sane orders. Then unset `DRY_RUN` to go live.
5. The `daily_state.json` kill-switch file lives on local disk. Railway containers are ephemeral, so on redeploy the budget resets. Mount a Railway volume at the repo root if you want persistence across deploys.

## Safety

- **Equity-based sizing** with hard $ cap and `TRAINING_WHEELS_MAX_USD` short-circuit
- **Daily loss kill-switch** that halts further trading until next UTC day
- **Equity floor** below which the bot refuses to trade
- **`DRY_RUN`** to validate the full decision path without signing
- **Strict entry conditions**: edge ≥ 0.10, |d| ≥ 1.8, ≤ 5 min to expiry, depth check
- **Single trade per contract** within a session
- **Graceful SIGTERM** for Railway redeploys

## Disclaimer

**USE AT YOUR OWN RISK.** This is experimental software for educational purposes. Trading binary outcomes involves significant risk of loss. Start with the smallest possible size. The authors are not responsible for any financial losses.

## License

MIT
