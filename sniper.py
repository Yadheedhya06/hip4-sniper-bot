#!/usr/bin/env python3
"""
HIP-4 BTC PriceBinary Sniper Bot
=================================
Monitors Hyperliquid HIP-4 BTC binary outcome contracts and places small buy
orders on the winning side in the final minutes before settlement when there
is a clear statistical edge between the model probability and market price.

Usage:
    export HYPERLIQUID_PRIVATE_KEY="0x..."
    python sniper.py
"""

import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import websockets
import asyncio
from scipy.stats import norm

# =============================================================================
# CONFIGURATION — adjust these to your risk tolerance
# =============================================================================

CONFIG = {
    # ── Position sizing ──
    "position_size_usd": 25,           # max USD per trade (start small!)
    "max_open_positions": 3,            # max concurrent positions across contracts

    # ── Edge & entry thresholds ──
    "edge_threshold": 0.10,             # min edge (model_prob - market_mid) to trade
    "min_abs_d": 1.8,                   # min |d| for entry (~96%+ model prob)
    "max_time_left_minutes": 5.0,       # only trade when ≤ this many minutes remain
    "ideal_time_left_minutes": 3.0,     # preferred entry window

    # ── Volatility estimation ──
    "vol_windows": [15, 30, 60],        # lookback windows in minutes for vol calc
    "vol_multiplier": 1.3,             # conservatism multiplier on realized vol
    "min_vol_annualized": 0.30,        # floor on annualized vol (30%)

    # ── Orderbook & execution ──
    "max_slippage_pct": 0.03,          # max 3% slippage on entry
    "edge_preservation_min": 0.80,     # order price must preserve ≥80% of edge
    "use_limit_orders": True,          # True=limit, False=IOC market

    # ── Discovery & polling ──
    "discovery_interval_sec": 30,      # how often to scan for new contracts
    "price_poll_interval_sec": 2,      # how often to refresh BTC price (fallback)
    "ws_reconnect_delay_sec": 5,       # delay before WS reconnect

    # ── Logging ──
    "log_to_file": True,
    "log_file": "sniper.log",
    "log_level": "INFO",

    # ── API endpoints ──
    "rest_url": "https://api.hyperliquid.xyz/info",
    "ws_url": "wss://api.hyperliquid.xyz/ws",
    "exchange_url": "https://api.hyperliquid.xyz/exchange",
}

# Minutes in a year for T conversion
MINUTES_PER_YEAR = 525_600

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("hip4_sniper")
    logger.setLevel(getattr(logging, CONFIG["log_level"]))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # Optional file handler
    if CONFIG["log_to_file"]:
        fh = logging.FileHandler(CONFIG["log_file"])
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

log = setup_logging()

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BinaryContract:
    """Represents a single BTC priceBinary outcome contract."""
    asset_name: str               # e.g. "BTC-BINARY-20260503-78213"
    asset_index: int              # index in the universe array
    underlying: str               # "BTC"
    target_price: float           # strike price
    expiry_utc: datetime          # expiry as UTC datetime
    period: str                   # "1d", "1h", etc.
    raw_description: str          # full metadata string
    traded: bool = False          # whether we already traded this
    last_yes_mid: Optional[float] = None
    last_no_mid: Optional[float] = None


@dataclass
class VolEstimate:
    """Rolling volatility estimate."""
    annualized: float
    window_minutes: int
    num_returns: int
    timestamp: datetime


@dataclass
class TradeDecision:
    """Captures every evaluation for logging."""
    timestamp: datetime
    contract: str
    btc_price: float
    strike: float
    time_left_min: float
    sigma: float
    d_value: float
    p_model_yes: float
    market_mid_yes: float
    edge: float
    action: str             # "BUY_YES", "BUY_NO", "SKIP"
    reason: str


# =============================================================================
# HYPERLIQUID API CLIENT (REST + WebSocket)
# =============================================================================

class HyperliquidClient:
    """Handles all communication with Hyperliquid REST and WS APIs."""

    def __init__(self, private_key: str):
        self.private_key = private_key
        self.rest_url = CONFIG["rest_url"]
        self.ws_url = CONFIG["ws_url"]
        self.exchange_url = CONFIG["exchange_url"]
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Try to import and use the SDK for order placement
        self._sdk_exchange = None
        self._sdk_info = None
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
            self._sdk_info = Info(constants.MAINNET_API_URL, skip_ws=True)
            self._sdk_exchange = Exchange(
                wallet=None,  # We'll handle signing manually or pass key
                base_url=constants.MAINNET_API_URL,
            )
            log.info("SDK loaded (Info for queries). Order placement uses REST.")
        except ImportError:
            log.warning("hyperliquid SDK not found — using raw REST/WS only")
        except Exception as e:
            log.warning(f"SDK init issue (non-fatal): {e}")

    # ── REST helpers ──

    def _post_info(self, payload: dict) -> dict:
        """POST to /info endpoint."""
        resp = self.session.post(self.rest_url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_meta_and_asset_ctxs(self) -> dict:
        """Fetch full metadata + asset contexts."""
        return self._post_info({"type": "metaAndAssetCtxs"})

    def get_all_mids(self) -> dict:
        """Fetch current mid prices for all assets."""
        return self._post_info({"type": "allMids"})

    def get_l2_book(self, coin: str) -> dict:
        """Fetch L2 orderbook for a specific coin."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def get_candles(self, coin: str, interval: str = "1m", lookback: int = 60) -> list:
        """
        Fetch recent candle data.
        interval: "1m", "5m", "15m", "1h", etc.
        """
        # Calculate start time (lookback minutes ago)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (lookback * 60 * 1000)
        try:
            return self._post_info({
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": now_ms,
                }
            })
        except Exception as e:
            log.warning(f"Failed to fetch candles for {coin}: {e}")
            return []

    # ── Order placement ──
    # NOTE: Full order placement requires signing with the private key.
    # The SDK's Exchange class handles this. Below is the structure for
    # placing orders — you must ensure the SDK is properly initialized
    # with your wallet/private key.

    def place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> dict:
        """
        Place a limit order on a binary outcome token.

        For HIP-4 binary tokens:
        - Buy Yes token = go long on the Yes coin
        - Buy No token = go long on the No coin (or short Yes, depending on structure)

        Returns the order response dict.
        """
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants
            import eth_account

            account = eth_account.Account.from_key(self.private_key)
            exchange = Exchange(
                wallet=account,
                base_url=constants.MAINNET_API_URL,
            )

            order_result = exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=reduce_only,
            )
            log.info(f"Order placed: {coin} {'BUY' if is_buy else 'SELL'} "
                     f"size={size} price={price} -> {order_result}")
            return order_result

        except ImportError:
            log.error("eth_account or hyperliquid SDK not available for signing")
            return self._place_order_raw(coin, is_buy, size, price)
        except Exception as e:
            log.error(f"Order placement failed: {e}")
            return {"error": str(e)}

    def _place_order_raw(self, coin: str, is_buy: bool, size: float, price: float) -> dict:
        """
        Fallback: place order via raw REST POST to /exchange.
        Requires manual EIP-712 signing — left as a TODO since the SDK handles this.
        """
        # TODO: Implement raw EIP-712 signed order if SDK is unavailable.
        # This requires constructing the order action, hashing it per EIP-712,
        # signing with the private key, and POSTing to /exchange.
        log.error("Raw order placement not implemented — install hyperliquid SDK")
        return {"error": "raw_not_implemented"}


# =============================================================================
# CONTRACT DISCOVERY & PARSING
# =============================================================================

def parse_binary_description(desc: str) -> Optional[dict]:
    """
    Parse a priceBinary outcome description string.

    Example format:
        class:priceBinary|underlying:BTC|expiry:20260503-0600|targetPrice:78213|period:1d

    Returns dict with parsed fields or None if not a BTC priceBinary.
    """
    if "priceBinary" not in desc or "BTC" not in desc:
        return None

    fields = {}
    for part in desc.split("|"):
        if ":" in part:
            key, val = part.split(":", 1)
            fields[key.strip()] = val.strip()

    if fields.get("class") != "priceBinary" or fields.get("underlying") != "BTC":
        return None

    try:
        # Parse expiry: "20260503-0600" -> datetime
        expiry_str = fields["expiry"]
        expiry_dt = datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)

        return {
            "underlying": fields["underlying"],
            "target_price": float(fields["targetPrice"]),
            "expiry_utc": expiry_dt,
            "period": fields.get("period", "1d"),
        }
    except (KeyError, ValueError) as e:
        log.debug(f"Failed to parse binary description '{desc}': {e}")
        return None


def discover_btc_binaries(client: HyperliquidClient) -> list[BinaryContract]:
    """
    Scan the Hyperliquid universe for active BTC priceBinary contracts.
    Returns a list of BinaryContract objects for contracts not yet expired.
    """
    contracts = []
    now = datetime.now(timezone.utc)

    try:
        data = client.get_meta_and_asset_ctxs()
    except Exception as e:
        log.error(f"Failed to fetch metaAndAssetCtxs: {e}")
        return contracts

    # data is typically [meta, assetCtxs] — meta contains universe array
    if not isinstance(data, list) or len(data) < 2:
        log.warning(f"Unexpected metaAndAssetCtxs format: {type(data)}")
        return contracts

    meta = data[0]
    universe = meta.get("universe", [])

    for idx, asset in enumerate(universe):
        name = asset.get("name", "")
        # The description may be in the asset dict or in a separate field
        # Try multiple possible locations for the description
        desc = (
            asset.get("description", "")
            or asset.get("szDecimals", "")  # sometimes embedded
            or name
        )

        # Also check if the name itself contains priceBinary info
        # or if there's an 'extra' or 'info' field
        for field_name in ["description", "extra", "info", "className"]:
            if field_name in asset:
                candidate = str(asset[field_name])
                if "priceBinary" in candidate and "BTC" in candidate:
                    desc = candidate
                    break

        # Try parsing the description
        parsed = parse_binary_description(desc)
        if parsed is None:
            # Also try the asset name — some formats encode info there
            parsed = parse_binary_description(name)
        if parsed is None:
            continue

        # Skip expired contracts
        if parsed["expiry_utc"] <= now:
            continue

        contract = BinaryContract(
            asset_name=name,
            asset_index=idx,
            underlying=parsed["underlying"],
            target_price=parsed["target_price"],
            expiry_utc=parsed["expiry_utc"],
            period=parsed["period"],
            raw_description=desc,
        )
        contracts.append(contract)

    log.info(f"Discovered {len(contracts)} active BTC priceBinary contracts")
    for c in contracts:
        time_left = (c.expiry_utc - now).total_seconds() / 60
        log.info(f"  {c.asset_name} | strike={c.target_price} | "
                 f"expiry={c.expiry_utc.isoformat()} | {time_left:.1f}min left")

    return contracts


# =============================================================================
# VOLATILITY CALCULATOR
# =============================================================================

class VolatilityCalculator:
    """
    Calculates annualized realized volatility from 1-minute BTC log returns.
    Uses multiple lookback windows and takes the conservative (max) estimate.
    """

    def __init__(self, windows: list[int] = None, multiplier: float = 1.3):
        self.windows = windows or CONFIG["vol_windows"]
        self.multiplier = multiplier
        self.min_vol = CONFIG["min_vol_annualized"]
        # Store recent 1-minute close prices
        self.prices: list[tuple[float, float]] = []  # (timestamp, price)

    def add_price(self, timestamp: float, price: float):
        """Add a 1-minute price observation."""
        self.prices.append((timestamp, price))
        # Keep last 120 minutes of data
        cutoff = timestamp - 120 * 60
        self.prices = [(t, p) for t, p in self.prices if t >= cutoff]

    def load_from_candles(self, candles: list):
        """Load prices from candle data (REST response)."""
        self.prices.clear()
        for candle in candles:
            try:
                # Candle format: {"t": timestamp_ms, "c": close, ...}
                ts = candle.get("t", candle.get("T", 0))
                if ts > 1e12:  # milliseconds
                    ts = ts / 1000
                close = float(candle.get("c", candle.get("close", 0)))
                if close > 0:
                    self.prices.append((ts, close))
            except (ValueError, TypeError):
                continue
        self.prices.sort(key=lambda x: x[0])
        log.debug(f"Loaded {len(self.prices)} candle prices for vol calc")

    def calculate(self) -> Optional[VolEstimate]:
        """
        Calculate annualized realized volatility.
        Uses multiple windows, takes the max, and multiplies by conservatism factor.
        """
        if len(self.prices) < 5:
            log.warning(f"Not enough price data for vol calc ({len(self.prices)} points)")
            return None

        all_prices = np.array([p for _, p in self.prices])
        all_log_returns = np.diff(np.log(all_prices))

        best_vol = 0.0
        best_window = 0
        best_n = 0

        for window in self.windows:
            # Use the last `window` returns
            returns = all_log_returns[-window:] if len(all_log_returns) >= window else all_log_returns
            if len(returns) < 3:
                continue

            # Standard deviation of 1-minute log returns
            std_1m = np.std(returns, ddof=1)

            # Annualize: multiply by sqrt(minutes per year)
            annualized = std_1m * math.sqrt(MINUTES_PER_YEAR)

            if annualized > best_vol:
                best_vol = annualized
                best_window = window
                best_n = len(returns)

        # Apply conservatism multiplier
        final_vol = max(best_vol * self.multiplier, self.min_vol)

        estimate = VolEstimate(
            annualized=final_vol,
            window_minutes=best_window,
            num_returns=best_n,
            timestamp=datetime.now(timezone.utc),
        )
        log.debug(f"Vol estimate: {final_vol:.4f} (window={best_window}m, "
                  f"n={best_n}, raw_max={best_vol:.4f})")
        return estimate


# =============================================================================
# EDGE MODEL
# =============================================================================

def compute_model_probability(
    btc_price: float,
    strike: float,
    time_left_minutes: float,
    sigma_annual: float,
) -> tuple[float, float]:
    """
    Compute the model probability of Yes (BTC ≥ strike at expiry).

    Uses normal approximation:
        T = time_left_minutes / 525600
        d = (S - K) / (S * σ * √T)
        P(Yes) = Φ(d)

    Returns: (p_yes, d_value)
    """
    if time_left_minutes <= 0:
        # Already expired — outcome is deterministic
        return (1.0 if btc_price >= strike else 0.0, float('inf') if btc_price >= strike else float('-inf'))

    T = time_left_minutes / MINUTES_PER_YEAR

    # Denominator: S * σ * √T
    denom = btc_price * sigma_annual * math.sqrt(T)
    if denom <= 0:
        return (1.0 if btc_price >= strike else 0.0, 0.0)

    d = (btc_price - strike) / denom
    p_yes = norm.cdf(d)

    return (p_yes, d)


def evaluate_edge(
    contract: BinaryContract,
    btc_price: float,
    yes_mid: float,
    vol_estimate: VolEstimate,
) -> TradeDecision:
    """
    Evaluate whether there's a tradable edge on this contract.

    Returns a TradeDecision with action = BUY_YES, BUY_NO, or SKIP.
    """
    now = datetime.now(timezone.utc)
    time_left_min = (contract.expiry_utc - now).total_seconds() / 60

    p_yes, d_val = compute_model_probability(
        btc_price=btc_price,
        strike=contract.target_price,
        time_left_minutes=time_left_min,
        sigma_annual=vol_estimate.annualized,
    )
    p_no = 1.0 - p_yes

    # Edge for buying Yes
    edge_yes = p_yes - yes_mid
    # Edge for buying No (No mid = 1 - Yes mid)
    no_mid = 1.0 - yes_mid
    edge_no = p_no - no_mid

    # Default: skip
    action = "SKIP"
    reason_parts = []

    # ── Check time constraint ──
    if time_left_min > CONFIG["max_time_left_minutes"]:
        reason_parts.append(f"time_left={time_left_min:.1f}m > max={CONFIG['max_time_left_minutes']}m")
    elif time_left_min <= 0:
        reason_parts.append("expired")
    else:
        # ── Check for Buy Yes ──
        if (edge_yes >= CONFIG["edge_threshold"]
                and d_val >= CONFIG["min_abs_d"]
                and time_left_min <= CONFIG["max_time_left_minutes"]):
            action = "BUY_YES"
            reason_parts.append(
                f"YES edge={edge_yes:.4f} >= {CONFIG['edge_threshold']}, "
                f"d={d_val:.3f} >= {CONFIG['min_abs_d']}"
            )

        # ── Check for Buy No ──
        elif (edge_no >= CONFIG["edge_threshold"]
              and d_val <= -CONFIG["min_abs_d"]
              and time_left_min <= CONFIG["max_time_left_minutes"]):
            action = "BUY_NO"
            reason_parts.append(
                f"NO edge={edge_no:.4f} >= {CONFIG['edge_threshold']}, "
                f"d={d_val:.3f} <= -{CONFIG['min_abs_d']}"
            )

        else:
            # Not enough edge
            if abs(d_val) < CONFIG["min_abs_d"]:
                reason_parts.append(f"|d|={abs(d_val):.3f} < {CONFIG['min_abs_d']}")
            if edge_yes < CONFIG["edge_threshold"] and edge_no < CONFIG["edge_threshold"]:
                reason_parts.append(
                    f"edge_yes={edge_yes:.4f}, edge_no={edge_no:.4f} "
                    f"< threshold={CONFIG['edge_threshold']}"
                )

    reason = "; ".join(reason_parts) if reason_parts else "no conditions met"

    decision = TradeDecision(
        timestamp=now,
        contract=contract.asset_name,
        btc_price=btc_price,
        strike=contract.target_price,
        time_left_min=time_left_min,
        sigma=vol_estimate.annualized,
        d_value=d_val,
        p_model_yes=p_yes,
        market_mid_yes=yes_mid,
        edge=max(edge_yes, edge_no),
        action=action,
        reason=reason,
    )
    return decision


# =============================================================================
# ORDERBOOK ANALYSIS
# =============================================================================

def check_orderbook_depth(
    client: HyperliquidClient,
    coin: str,
    is_buy: bool,
    size_usd: float,
) -> tuple[bool, float]:
    """
    Check if the orderbook has enough depth for our size with acceptable slippage.

    Returns: (has_depth, estimated_fill_price)
    """
    try:
        book = client.get_l2_book(coin)
    except Exception as e:
        log.warning(f"Failed to fetch orderbook for {coin}: {e}")
        return False, 0.0

    # Parse the book — format: {"levels": [[bids], [asks]]}
    # or {"coin": ..., "levels": ...}
    levels = book.get("levels", [[], []])
    if len(levels) < 2:
        return False, 0.0

    # For buying, we hit the asks; for selling, we hit the bids
    side = levels[1] if is_buy else levels[0]  # asks for buy, bids for sell

    if not side:
        log.debug(f"No {'asks' if is_buy else 'bids'} in book for {coin}")
        return False, 0.0

    # Walk the book and calculate fill price
    remaining_usd = size_usd
    total_cost = 0.0
    total_size = 0.0

    for level in side:
        try:
            price = float(level.get("px", level[0]) if isinstance(level, dict) else level[0])
            qty = float(level.get("sz", level[1]) if isinstance(level, dict) else level[1])
        except (IndexError, ValueError, TypeError):
            continue

        level_usd = price * qty
        fill_amt = min(remaining_usd, level_usd)
        fill_qty = fill_amt / price if price > 0 else 0

        total_cost += fill_amt
        total_size += fill_qty
        remaining_usd -= fill_amt

        if remaining_usd <= 0:
            break

    if total_size <= 0 or remaining_usd > size_usd * 0.5:
        log.debug(f"Insufficient depth for {coin}: filled only ${size_usd - remaining_usd:.2f} of ${size_usd:.2f}")
        return False, 0.0

    avg_fill = total_cost / total_size
    best_price = float(side[0].get("px", side[0][0]) if isinstance(side[0], dict) else side[0][0])

    slippage = abs(avg_fill - best_price) / best_price if best_price > 0 else 1.0

    has_depth = slippage <= CONFIG["max_slippage_pct"]
    if not has_depth:
        log.debug(f"Slippage too high for {coin}: {slippage:.4f} > {CONFIG['max_slippage_pct']}")

    return has_depth, avg_fill


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class HyperliquidSniperBot:
    """
    Main bot that orchestrates discovery, monitoring, edge evaluation, and execution.
    """

    def __init__(self, private_key: str):
        self.client = HyperliquidClient(private_key)
        self.vol_calc = VolatilityCalculator()
        self.contracts: dict[str, BinaryContract] = {}  # name -> contract
        self.open_positions: int = 0
        self._shutdown = threading.Event()
        self._btc_price: float = 0.0
        self._mids: dict[str, float] = {}
        self._last_discovery = 0.0
        self._last_vol_update = 0.0

    # ── Lifecycle ──

    def start(self):
        """Start the bot — runs until Ctrl+C."""
        log.info("=" * 60)
        log.info("HIP-4 BTC PriceBinary Sniper Bot — Starting")
        log.info(f"Position size: ${CONFIG['position_size_usd']}")
        log.info(f"Edge threshold: {CONFIG['edge_threshold']}")
        log.info(f"Max time window: {CONFIG['max_time_left_minutes']}min")
        log.info(f"Min |d|: {CONFIG['min_abs_d']}")
        log.info("=" * 60)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        try:
            # Try async WebSocket approach first
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down...")
        except Exception as e:
            log.error(f"Fatal error in async loop: {e}", exc_info=True)
            log.info("Falling back to polling mode...")
            self._run_polling()

    def _handle_signal(self, signum, frame):
        log.info(f"Received signal {signum} — initiating graceful shutdown...")
        self._shutdown.set()

    # ── Async main loop (WebSocket-based) ──

    async def _run_async(self):
        """Main async loop using WebSocket for real-time price updates."""
        # Initial setup
        self._refresh_btc_price()
        self._refresh_volatility()
        self._discover_contracts()

        # Start WebSocket listener in background
        ws_task = asyncio.create_task(self._ws_listener())

        try:
            while not self._shutdown.is_set():
                now = time.time()

                # Periodic discovery of new contracts
                if now - self._last_discovery >= CONFIG["discovery_interval_sec"]:
                    self._discover_contracts()
                    self._last_discovery = now

                # Periodic volatility refresh
                if now - self._last_vol_update >= 60:  # every minute
                    self._refresh_volatility()
                    self._last_vol_update = now

                # Refresh prices (fallback if WS is lagging)
                self._refresh_btc_price()

                # Evaluate all active contracts
                self._evaluate_all_contracts()

                # Clean up expired contracts
                self._cleanup_expired()

                await asyncio.sleep(CONFIG["price_poll_interval_sec"])
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            log.info("Bot shutdown complete.")

    async def _ws_listener(self):
        """WebSocket listener for real-time allMids updates."""
        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    CONFIG["ws_url"],
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    # Subscribe to allMids
                    sub_msg = {
                        "method": "subscribe",
                        "subscription": {"type": "allMids"}
                    }
                    await ws.send(json.dumps(sub_msg))
                    log.info("WebSocket connected — subscribed to allMids")

                    async for raw_msg in ws:
                        if self._shutdown.is_set():
                            break
                        try:
                            msg = json.loads(raw_msg)
                            self._handle_ws_message(msg)
                        except json.JSONDecodeError:
                            continue

            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"WebSocket closed: {e} — reconnecting in {CONFIG['ws_reconnect_delay_sec']}s")
            except Exception as e:
                log.warning(f"WebSocket error: {e} — reconnecting in {CONFIG['ws_reconnect_delay_sec']}s")

            if not self._shutdown.is_set():
                await asyncio.sleep(CONFIG["ws_reconnect_delay_sec"])

    def _handle_ws_message(self, msg: dict):
        """Process incoming WebSocket messages."""
        channel = msg.get("channel", "")
        data = msg.get("data", {})

        if channel == "allMids":
            mids = data.get("mids", data)
            if isinstance(mids, dict):
                self._mids.update(mids)
                # Update BTC price
                for key in ["BTC", "BTC-PERP", "@1"]:  # common BTC identifiers
                    if key in mids:
                        try:
                            self._btc_price = float(mids[key])
                        except (ValueError, TypeError):
                            pass
                        break

    # ── Polling fallback ──

    def _run_polling(self):
        """Fallback polling loop if WebSocket fails."""
        while not self._shutdown.is_set():
            try:
                now = time.time()

                if now - self._last_discovery >= CONFIG["discovery_interval_sec"]:
                    self._discover_contracts()
                    self._last_discovery = now

                if now - self._last_vol_update >= 60:
                    self._refresh_volatility()
                    self._last_vol_update = now

                self._refresh_btc_price()
                self._evaluate_all_contracts()
                self._cleanup_expired()

                time.sleep(CONFIG["price_poll_interval_sec"])

            except Exception as e:
                log.error(f"Error in polling loop: {e}", exc_info=True)
                time.sleep(5)

        log.info("Polling loop shutdown complete.")

    # ── Core operations ──

    def _refresh_btc_price(self):
        """Fetch current BTC price and all mids via REST."""
        try:
            mids = self.client.get_all_mids()
            if isinstance(mids, dict):
                self._mids.update(mids)
                # Try common BTC identifiers
                for key in ["BTC", "BTC-PERP", "@1"]:
                    if key in mids:
                        self._btc_price = float(mids[key])
                        break
                # Also update contract mids
                for name, contract in self.contracts.items():
                    if name in mids:
                        try:
                            contract.last_yes_mid = float(mids[name])
                            contract.last_no_mid = 1.0 - contract.last_yes_mid
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            log.debug(f"Price refresh failed: {e}")

    def _refresh_volatility(self):
        """Update BTC volatility estimate from recent candles."""
        candles = self.client.get_candles("BTC", interval="1m", lookback=120)
        if candles:
            self.vol_calc.load_from_candles(candles)
        self._last_vol_update = time.time()

    def _discover_contracts(self):
        """Discover and register active BTC priceBinary contracts."""
        new_contracts = discover_btc_binaries(self.client)
        for c in new_contracts:
            if c.asset_name not in self.contracts:
                self.contracts[c.asset_name] = c
                log.info(f"Tracking new contract: {c.asset_name}")
            else:
                # Update expiry and strike in case metadata changed
                existing = self.contracts[c.asset_name]
                existing.expiry_utc = c.expiry_utc
                existing.target_price = c.target_price
        self._last_discovery = time.time()

    def _evaluate_all_contracts(self):
        """Evaluate edge on all tracked contracts."""
        if self._btc_price <= 0:
            log.debug("No BTC price available yet — skipping evaluation")
            return

        vol = self.vol_calc.calculate()
        if vol is None:
            log.debug("No vol estimate available — skipping evaluation")
            return

        now = datetime.now(timezone.utc)

        for name, contract in list(self.contracts.items()):
            # Skip already traded contracts
            if contract.traded:
                continue

            # Skip if no mid price for this contract
            if contract.last_yes_mid is None:
                # Try to get from stored mids
                if name in self._mids:
                    try:
                        contract.last_yes_mid = float(self._mids[name])
                        contract.last_no_mid = 1.0 - contract.last_yes_mid
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

            # Skip if yes_mid is invalid
            if contract.last_yes_mid <= 0 or contract.last_yes_mid >= 1:
                continue

            time_left = (contract.expiry_utc - now).total_seconds() / 60

            # Only log detailed evaluations when within 2x the max window
            if time_left > CONFIG["max_time_left_minutes"] * 2:
                continue

            # Evaluate edge
            decision = evaluate_edge(
                contract=contract,
                btc_price=self._btc_price,
                yes_mid=contract.last_yes_mid,
                vol_estimate=vol,
            )

            # Log the decision
            self._log_decision(decision)

            # Execute if we have an actionable signal
            if decision.action in ("BUY_YES", "BUY_NO"):
                self._execute_trade(contract, decision)

    def _execute_trade(self, contract: BinaryContract, decision: TradeDecision):
        """Execute a trade based on the decision."""
        if self.open_positions >= CONFIG["max_open_positions"]:
            log.warning(f"Max positions ({CONFIG['max_open_positions']}) reached — skipping {contract.asset_name}")
            return

        is_buy_yes = decision.action == "BUY_YES"
        coin = contract.asset_name  # The Yes token coin name

        # Check orderbook depth
        has_depth, est_fill = check_orderbook_depth(
            self.client,
            coin=coin,
            is_buy=True,
            size_usd=CONFIG["position_size_usd"],
        )

        if not has_depth:
            log.info(f"Insufficient depth for {contract.asset_name} — skipping")
            return

        # Calculate order price that preserves edge
        if is_buy_yes:
            # We want to buy Yes — max price = model_prob - (1 - edge_preservation) * edge
            max_price = decision.p_model_yes - (1 - CONFIG["edge_preservation_min"]) * decision.edge
            price = min(est_fill, max_price)
            size = CONFIG["position_size_usd"] / price if price > 0 else 0
        else:
            # We want to buy No — buying No is equivalent to selling Yes or buying on the No side
            # No token price = 1 - Yes_mid
            no_price = 1.0 - decision.market_mid_yes
            max_no_price = (1.0 - decision.p_model_yes) + (1 - CONFIG["edge_preservation_min"]) * decision.edge
            price = min(no_price, max_no_price)
            size = CONFIG["position_size_usd"] / price if price > 0 else 0
            # For No, we might need to use a different coin or short Yes
            # This depends on the HIP-4 contract structure

        if size <= 0:
            log.warning(f"Calculated size <= 0 for {contract.asset_name} — skipping")
            return

        log.info("=" * 50)
        log.info(f"EXECUTING TRADE: {decision.action} on {contract.asset_name}")
        log.info(f"  BTC={decision.btc_price:.2f} Strike={decision.strike:.2f}")
        log.info(f"  P_model={decision.p_model_yes:.4f} Mid={decision.market_mid_yes:.4f}")
        log.info(f"  Edge={decision.edge:.4f} d={decision.d_value:.3f}")
        log.info(f"  Price={price:.4f} Size={size:.2f} ({CONFIG['position_size_usd']}$)")
        log.info("=" * 50)

        # Place the order
        result = self.client.place_limit_order(
            coin=coin,
            is_buy=is_buy_yes,
            size=round(size, 2),
            price=round(price, 4),
        )

        if "error" not in result:
            contract.traded = True
            self.open_positions += 1
            log.info(f"Order submitted for {contract.asset_name}: {result}")
        else:
            log.error(f"Order failed for {contract.asset_name}: {result}")

    def _cleanup_expired(self):
        """Remove expired contracts from tracking."""
        now = datetime.now(timezone.utc)
        expired = [
            name for name, c in self.contracts.items()
            if c.expiry_utc <= now
        ]
        for name in expired:
            c = self.contracts.pop(name)
            if c.traded:
                self.open_positions = max(0, self.open_positions - 1)
            log.info(f"Removed expired contract: {name}")

    def _log_decision(self, d: TradeDecision):
        """Log a trade decision with full parameters."""
        level = logging.INFO if d.action != "SKIP" else logging.DEBUG
        # Always log at INFO when close to expiry
        if d.time_left_min <= CONFIG["max_time_left_minutes"]:
            level = logging.INFO

        log.log(
            level,
            f"[{d.contract}] BTC={d.btc_price:.2f} K={d.strike:.2f} "
            f"t={d.time_left_min:.1f}m σ={d.sigma:.4f} d={d.d_value:.3f} "
            f"P={d.p_model_yes:.4f} mid={d.market_mid_yes:.4f} "
            f"edge={d.edge:.4f} -> {d.action} ({d.reason})"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    # Load private key from environment
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "")

    if not private_key:
        log.error(
            "HYPERLIQUID_PRIVATE_KEY not set!\n"
            "  export HYPERLIQUID_PRIVATE_KEY='0xYOUR_KEY_HERE'\n"
            "  Then run: python sniper.py"
        )
        sys.exit(1)

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    log.info(f"Private key loaded (ends with ...{private_key[-4:]})")

    bot = HyperliquidSniperBot(private_key)
    bot.start()


if __name__ == "__main__":
    main()


# =============================================================================
# TODO: Optional improvements
# =============================================================================
# TODO: Add backtesting framework — replay historical binary outcomes and
#       evaluate model accuracy + PnL.
# TODO: Add risk limits — max daily loss, max total exposure, cool-down
#       after consecutive losses.
# TODO: Support multiple outcomes in parallel (e.g., ETH binaries, hourly).
# TODO: Add Telegram/Discord notifications for trades and daily PnL.
# TODO: Implement Kelly criterion for dynamic position sizing based on edge.
# TODO: Add a simple web dashboard (Flask/FastAPI) for monitoring.
# TODO: Track and log actual settlement outcomes for model calibration.
# TODO: Use candle WebSocket subscription instead of REST polling for vol.
# TODO: Add circuit breaker if BTC vol spikes dramatically (>3x normal).
# TODO: Implement partial fill handling and order amendment logic.
