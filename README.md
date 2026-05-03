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
```

## Configuration

Set your private key:
```bash
export HYPERLIQUID_PRIVATE_KEY="0xYOUR_PRIVATE_KEY_HERE"
```

Edit the `CONFIG` dict at the top of `sniper.py` to adjust:
- `position_size_usd` — max trade size (default $25)
- `edge_threshold` — minimum edge to trade (default 0.10)
- `max_time_left_minutes` — latest entry window (default 5 min)
- `min_abs_d` — minimum |d| for entry (default 1.8)
- `vol_windows` — lookback windows for volatility calc
- `vol_multiplier` — conservatism factor on vol (default 1.3)

## Running

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

Logs go to both console and `sniper.log` file with timestamps, prices, model params, and trade decisions.

## Safety

- **Small size**: Default $25 position, configurable
- **Strict conditions**: Won't trade unless ALL conditions pass
- **Single trade per contract**: Stops monitoring after fill
- **Graceful shutdown**: Ctrl+C cleanly exits

## Disclaimer

**USE AT YOUR OWN RISK.** This is experimental software for educational purposes. Trading binary outcomes involves significant risk of loss. Start with the smallest possible size. The authors are not responsible for any financial losses.

## License

MIT
