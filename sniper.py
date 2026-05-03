#!/usr/bin/env python3
"""
HIP-4 BTC PriceBinary Sniper Bot
=================================
Monitors Hyperliquid HIP-4 BTC binary outcome contracts and places small buy
orders on the winning side in the final minutes before settlement when there
is a clear statistical edge between the model probability and market price.

Entry conditions are a hybrid of three gates (all must pass):
    1. Time gate           — within MAX_TIME_LEFT_MINUTES of expiry
    2. Statistical gate    — |d| ≥ dynamic min_d AND edge ≥ threshold.
                             min_d decays linearly with time_left so we accept
                             weaker statistical signals as expiry approaches.
    3. Conviction gate     — taker-buy volume ratio AND same-side bid depth
                             imbalance both clear thresholds that *tighten* as
                             statistical confidence weakens.
Pure-flow override: in the final PURE_FLOW_TIME_LEFT_MIN seconds the
statistical floor relaxes to MIN_D_FLOOR while conviction thresholds are
forced to maximum strictness — flow dominates near settlement.

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
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np
import requests
import websockets
import asyncio
from scipy.stats import norm

# Load .env if present (no-op in prod where env vars come from Railway)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_opt_float(key: str) -> Optional[float]:
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None

# =============================================================================
# CONFIGURATION
# =============================================================================
# Tuned for ~$800 equity with $100 hard per-trade cap. All values are
# env-overridable. Sections:
#   • Risk model         — fractional sizing with edge scaling, capped at $100
#   • Statistical gate   — dynamic |d| floor that decays with time_left
#   • Conviction gate    — taker-buy volume + same-side bid depth, with
#                          thresholds that tighten as |d| weakens
#   • Pure-flow window   — final 60-90s where flow dominates
# =============================================================================

CONFIG = {
    # ── Risk model (env-overridable) ──
    # At equity=$800: base risk = $80 (10%); clamped to ≤ $100 by max_risk_pct + cap.
    "base_risk_pct":       _env_float("SIZE_PCT_OF_EQUITY", 0.10),
    "max_risk_pct":        _env_float("SIZE_MAX_PCT_OF_EQUITY", 0.125),
    "min_trade_usd":       _env_float("SIZE_MIN_USD", 10),
    "max_trade_usd":       _env_float("SIZE_MAX_USD", 100),
    "edge_reference":      _env_float("EDGE_REFERENCE", 0.10),
    "max_open_positions":  _env_int("MAX_OPEN_POSITIONS", 3),
    "equity_refresh_sec":  _env_int("EQUITY_REFRESH_SEC", 300),

    # Training-wheels: hard cap per-trade USD regardless of equity.
    # Empty/unset = disabled (rely on SIZE_MAX_USD instead).
    "training_wheels_max_usd": _env_opt_float("TRAINING_WHEELS_MAX_USD"),
    "equity_floor_usd":    _env_float("EQUITY_FLOOR_USD", 50),

    # ── Daily loss kill-switch ──
    "daily_loss_limit_pct": _env_float("DAILY_LOSS_LIMIT_PCT", 0.10),  # 10% of day-start
    "daily_state_file":    os.environ.get("DAILY_STATE_FILE", "daily_state.json"),

    # ── Statistical gate (dynamic |d| floor) ──
    # min_d_required = max(MIN_D_FLOOR, MIN_D_BASE * (time_left / MAX_TIME_LEFT_MIN))
    # MIN_ABS_D is honored as a fallback for backwards-compat with old env files.
    "min_d_base":          _env_float("MIN_D_BASE", _env_float("MIN_ABS_D", 1.8)),
    "min_d_floor":         _env_float("MIN_D_FLOOR", 0.3),
    "edge_threshold":      _env_float("EDGE_THRESHOLD", 0.10),
    "max_time_left_minutes":   _env_float("MAX_TIME_LEFT_MIN", 5.0),
    "ideal_time_left_minutes": _env_float("IDEAL_TIME_LEFT_MIN", 3.0),

    # ── Conviction gate (taker-buy volume + same-side bid depth) ──
    # Required vol_ratio and depth_imb scale linearly with statistical
    # confidence: conf = min(1, |d| / MIN_D_BASE).
    #   required_vol_ratio = MAX - conf*(MAX - MIN)
    #   required_depth_imb = MAX - conf*(MAX - MIN)
    # At |d|=MIN_D_BASE → MIN thresholds. At |d|→0 → MAX thresholds.
    "volume_ratio_min":      _env_float("VOLUME_RATIO_MIN", 1.5),
    "volume_ratio_max":      _env_float("VOLUME_RATIO_MAX", 2.5),
    "depth_imb_min":         _env_float("DEPTH_IMB_MIN", 0.60),
    "depth_imb_max":         _env_float("DEPTH_IMB_MAX", 0.70),
    "volume_lookback_min":   _env_float("VOLUME_LOOKBACK_MIN", 10.0),
    "depth_price_band_pct":  _env_float("DEPTH_PRICE_BAND_PCT", 0.05),
    "depth_fallback_levels": _env_int("DEPTH_FALLBACK_LEVELS", 10),
    "depth_min_qualifying_levels": _env_int("DEPTH_MIN_QUALIFYING_LEVELS", 3),

    # ── Pure-flow window (final 60-90s) ──
    # Statistical floor drops to MIN_D_FLOOR; edge threshold relaxes; conviction
    # thresholds clamp to MAX (strictest). Refuses to trade on stale tape.
    "pure_flow_time_left_min":  _env_float("PURE_FLOW_TIME_LEFT_MIN", 1.5),
    "pure_flow_edge_threshold": _env_float("PURE_FLOW_EDGE_THRESHOLD", 0.05),
    "pure_flow_freshness_sec":  _env_float("PURE_FLOW_FRESHNESS_SEC", 60.0),

    # ── Volatility estimation ──
    "vol_windows": [15, 30, 60],

    # ── Orderbook & execution ──
    "max_slippage_pct":     _env_float("MAX_SLIPPAGE_PCT", 0.03),
    "edge_preservation_min": _env_float("EDGE_PRESERVATION_MIN", 0.80),
    "use_limit_orders":     _env_bool("USE_LIMIT_ORDERS", True),

    # ── Discovery & polling ──
    "discovery_interval_sec":  _env_int("DISCOVERY_INTERVAL_SEC", 300),
    "price_poll_interval_sec": _env_int("PRICE_POLL_INTERVAL_SEC", 15),
    "active_poll_interval_sec": _env_int("ACTIVE_POLL_INTERVAL_SEC", 2),
    "vol_refresh_sec":         _env_int("VOL_REFRESH_SEC", 60),
    "ws_reconnect_delay_sec":  _env_int("WS_RECONNECT_DELAY_SEC", 5),
    "ws_ping_interval_sec":    _env_int("WS_PING_INTERVAL_SEC", 30),
    "ws_sub_reconcile_sec":    _env_int("WS_SUB_RECONCILE_SEC", 5),
    "heartbeat_sec":           _env_int("HEARTBEAT_SEC", 300),

    # ── Logging ──
    "log_to_file":   _env_bool("LOG_TO_FILE", False),   # default off in prod
    "log_to_stdout": _env_bool("LOG_TO_STDOUT", True),
    "log_file":      os.environ.get("LOG_FILE", "sniper.log"),
    "log_level":     os.environ.get("LOG_LEVEL", "INFO"),

    # ── HTTP healthcheck (Railway expects $PORT) ──
    "health_port": _env_int("PORT", 8080),

    # ── Trading toggle (set DRY_RUN=true to log decisions but never sign) ──
    "dry_run": _env_bool("DRY_RUN", False),

    # ── API endpoints ──
    "rest_url":     os.environ.get("HL_REST_URL", "https://api.hyperliquid.xyz/info"),
    "ws_url":       os.environ.get("HL_WS_URL",   "wss://api.hyperliquid.xyz/ws"),
    "exchange_url": os.environ.get("HL_EXCHANGE_URL", "https://api.hyperliquid.xyz/exchange"),
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
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    fmt.converter = time.gmtime  # always log in UTC
    if CONFIG["log_to_stdout"]:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
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
    """A single BTC priceBinary outcome (HIP-4). Carries both Yes and No coins."""
    asset_name: str               # synthetic id, e.g. "BTC-BINARY-20260504-78443"
    outcome_id: int               # the `outcome` field from outcomeMeta
    yes_coin: str                 # "#<10*outcome+0>"  e.g. "#0"
    no_coin: str                  # "#<10*outcome+1>"  e.g. "#1"
    yes_asset_id: int             # 100_000_000 + 10*outcome + 0
    no_asset_id: int              # 100_000_000 + 10*outcome + 1
    underlying: str               # "BTC"
    target_price: float           # strike price
    expiry_utc: datetime          # expiry as UTC datetime
    period: str                   # "1d", "1h", etc.
    raw_description: str
    traded: bool = False
    last_yes_mid: Optional[float] = None
    last_no_mid: Optional[float] = None
    pure_flow_logged: bool = False  # one-shot log when entering pure-flow window


@dataclass
class VolEstimate:
    """Rolling volatility estimate."""
    annualized: float
    window_minutes: int
    num_returns: int
    timestamp: datetime


@dataclass
class TradeDecision:
    """Captures every evaluation for logging — statistical + conviction signals."""
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
    mode: str = "STAT"      # "STAT" | "HYBRID" | "PURE_FLOW"
    min_d_required: float = 0.0
    vol_ratio: Optional[float] = None
    vol_ratio_required: float = 0.0
    depth_imbalance: Optional[float] = None
    depth_imb_required: float = 0.0


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
        self._exchange = None  # lazily built by _ensure_exchange

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

    def get_outcome_meta(self) -> dict:
        """Fetch HIP-4 outcome metadata. Shape: {outcomes:[{outcome,name,description,sideSpecs:[…]}], questions:[…]}."""
        return self._post_info({"type": "outcomeMeta"})

    def get_all_mids(self) -> dict:
        """Fetch current mid prices for all assets."""
        return self._post_info({"type": "allMids"})

    def get_l2_book(self, coin: str) -> dict:
        """Fetch L2 orderbook for a specific coin."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def get_user_state(self, address: str) -> dict:
        """Fetch perp clearinghouse state — includes marginSummary.accountValue (equity)."""
        return self._post_info({"type": "clearinghouseState", "user": address})

    def get_spot_state(self, address: str) -> dict:
        """Fetch spot clearinghouse state. Contains balances[] keyed by coin (USDH, USDC, …)."""
        return self._post_info({"type": "spotClearinghouseState", "user": address})

    def get_usdh_balance(self, address: str) -> float:
        """Return spot USDH balance for the given address (HIP-4 collateral)."""
        state = self.get_spot_state(address)
        balances = state.get("balances", []) if isinstance(state, dict) else []
        for b in balances:
            if b.get("coin") == "USDH":
                try:
                    return float(b.get("total", 0) or 0)
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

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
    # The SDK's Exchange class handles signing. Two HIP-4-specific quirks:
    #   1. The SDK shipped name_to_asset only knows spot+perp, so we inject
    #      outcome coins ("#<10*outcome+side>") into its lookup tables.
    #   2. Newer SDKs (>=0.10) use positional args; `coin=` kwarg is gone.

    def _ensure_exchange(self):
        """Lazily build and cache the SDK Exchange instance with agent-wallet routing."""
        if getattr(self, "_exchange", None) is not None:
            return self._exchange
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils import constants
        import eth_account

        account = eth_account.Account.from_key(self.private_key)
        master = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip() or None
        self._exchange = Exchange(
            wallet=account,
            base_url=constants.MAINNET_API_URL,
            account_address=master,
        )
        return self._exchange

    @staticmethod
    def _maybe_inject_outcome(exchange, coin: str) -> None:
        """If coin is a HIP-4 outcome ("#<encoding>"), register asset id in SDK lookup tables."""
        if not (isinstance(coin, str) and coin.startswith("#")):
            return
        try:
            encoding = int(coin[1:])
        except ValueError:
            return
        asset_id = 100_000_000 + encoding
        exchange.info.name_to_coin[coin] = coin
        exchange.info.coin_to_asset[coin] = asset_id

    @staticmethod
    def _parse_order_response(coin: str, result: dict) -> dict:
        """Convert HL's nested order response into a flat dict; surface embedded errors."""
        if not isinstance(result, dict):
            return {"error": f"unexpected_response_type:{type(result).__name__}"}
        try:
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        except Exception:
            statuses = []
        for s in statuses:
            if isinstance(s, dict):
                if "error" in s:
                    return {"error": f"hl_rejected:{s['error']}"}
                if "resting" in s:
                    return {"resting_oid": s["resting"].get("oid"), "raw": result}
                if "filled" in s:
                    return {
                        "filled_oid": s["filled"].get("oid"),
                        "fill_sz": s["filled"].get("totalSz"),
                        "fill_px": s["filled"].get("avgPx"),
                        "raw": result,
                    }
        return {"raw": result}

    def place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> dict:
        """Place a HIP-4 outcome limit order. For HIP-4: coin like "#10" or "#11"."""
        try:
            exchange = self._ensure_exchange()
            self._maybe_inject_outcome(exchange, coin)
            order_result = exchange.order(
                coin,                            # name (positional)
                bool(is_buy),
                float(size),
                float(price),
                {"limit": {"tif": "Gtc"}},
                bool(reduce_only),
            )
            parsed = self._parse_order_response(coin, order_result)
            log.info(
                f"Order {'placed' if 'error' not in parsed else 'REJECTED'}: "
                f"{coin} {'BUY' if is_buy else 'SELL'} size={size} px={price} -> {parsed}"
            )
            return parsed

        except ImportError:
            log.error("hyperliquid SDK not available for signing")
            return {"error": "sdk_unavailable"}
        except KeyError as e:
            log.error(
                f"SDK could not resolve coin '{coin}' (KeyError: {e}). "
                f"asset_id should be 100_000_000 + 10*outcome + side"
            )
            return {"error": f"sdk_unknown_coin:{coin}"}
        except Exception as e:
            log.error(f"Order placement failed for coin={coin}: {e}", exc_info=True)
            return {"error": str(e)}

    def cancel_order(self, coin: str, oid: int) -> dict:
        """Cancel an existing HIP-4 outcome order by oid."""
        try:
            exchange = self._ensure_exchange()
            self._maybe_inject_outcome(exchange, coin)
            return exchange.cancel(coin, oid)
        except Exception as e:
            log.error(f"Cancel failed for {coin} oid={oid}: {e}", exc_info=True)
            return {"error": str(e)}

    def _place_order_raw(self, coin: str, is_buy: bool, size: float, price: float) -> dict:
        """
        Fallback: place order via raw REST POST to /exchange.
        Requires manual EIP-712 signing — left as a TODO since the SDK handles this.
        """
        # TODO: Implement raw EIP-712 signed order if SDK is unavailable.
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


OUTCOME_ASSET_BASE = 100_000_000  # exchange asset id base for outcome assets


def _outcome_coin(outcome_id: int, side: int) -> str:
    """HIP-4 outcome coin string used by /info?l2Book and allMids."""
    return f"#{10 * outcome_id + side}"


def _outcome_asset_id(outcome_id: int, side: int) -> int:
    """Numeric asset id for /exchange order action: 100_000_000 + 10*outcome + side."""
    return OUTCOME_ASSET_BASE + 10 * outcome_id + side


def discover_btc_binaries(client: "HyperliquidClient") -> list[BinaryContract]:
    """Scan HIP-4 outcomeMeta for active BTC priceBinary contracts not yet expired."""
    contracts: list[BinaryContract] = []
    now = datetime.now(timezone.utc)

    try:
        data = client.get_outcome_meta()
    except Exception as e:
        log.error(f"Failed to fetch outcomeMeta: {e}")
        return contracts

    outcomes = data.get("outcomes", []) if isinstance(data, dict) else []

    for entry in outcomes:
        try:
            outcome_id = int(entry.get("outcome"))
        except (TypeError, ValueError):
            continue
        desc = entry.get("description", "") or ""
        parsed = parse_binary_description(desc)
        if parsed is None:
            continue
        if parsed["expiry_utc"] <= now:
            continue

        sides = entry.get("sideSpecs", [])
        if len(sides) != 2:
            log.warning(f"outcome={outcome_id} has {len(sides)} sides, expected 2 — skipping")
            continue

        expiry_tag = parsed["expiry_utc"].strftime("%Y%m%d-%H%M")
        synthetic_name = (
            f"BTC-BINARY-{expiry_tag}-{int(parsed['target_price'])}"
        )

        contracts.append(BinaryContract(
            asset_name=synthetic_name,
            outcome_id=outcome_id,
            yes_coin=_outcome_coin(outcome_id, 0),
            no_coin=_outcome_coin(outcome_id, 1),
            yes_asset_id=_outcome_asset_id(outcome_id, 0),
            no_asset_id=_outcome_asset_id(outcome_id, 1),
            underlying=parsed["underlying"],
            target_price=parsed["target_price"],
            expiry_utc=parsed["expiry_utc"],
            period=parsed["period"],
            raw_description=desc,
        ))

    log.info(f"Discovered {len(contracts)} active BTC priceBinary contracts")
    for c in contracts:
        time_left = (c.expiry_utc - now).total_seconds() / 60
        log.info(
            f"  {c.asset_name} outcome={c.outcome_id} yes={c.yes_coin} no={c.no_coin} "
            f"strike={c.target_price} expiry={c.expiry_utc.isoformat()} "
            f"t_left={time_left:.1f}min"
        )

    return contracts


# =============================================================================
# VOLATILITY CALCULATOR
# =============================================================================

class VolatilityCalculator:
    """
    Annualized realized volatility from 1-minute BTC log returns.

    Spec: σ = MAX of realized annualized vol across {15, 30, 60}-minute windows.
    Taking the max across windows is itself the conservatism — no multiplier.
    """

    def __init__(self, windows: list[int] = None):
        self.windows = windows or CONFIG["vol_windows"]
        # Store recent 1-minute close prices
        self.prices: list[tuple[float, float]] = []  # (timestamp, price)

    def add_price(self, timestamp: float, price: float):
        """Add a 1-minute price observation."""
        self.prices.append((timestamp, price))
        # Keep last 120 minutes of data
        cutoff = timestamp - 120 * 60
        self.prices = [(t, p) for t, p in self.prices if t >= cutoff]

    def set_price(self, timestamp: float, price: float):
        """
        Streaming-candle ingestion: a 1m candle gets multiple WS messages as
        it forms. Dedupe by timestamp so we update the in-progress candle
        rather than appending duplicates.
        """
        if not (price > 0):
            return
        if self.prices and self.prices[-1][0] == timestamp:
            self.prices[-1] = (timestamp, price)
            return
        self.prices.append((timestamp, price))
        cutoff = timestamp - 120 * 60
        self.prices = [(t, p) for t, p in self.prices if t >= cutoff]

    def load_from_candles(self, candles: list):
        """Load prices from candle data (REST response)."""
        self.prices.clear()
        for candle in candles:
            try:
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
        """σ = max over windows of stdev(1m log returns) * √(minutes per year)."""
        if len(self.prices) < 5:
            log.warning(f"Not enough price data for vol calc ({len(self.prices)} points)")
            return None

        all_prices = np.array([p for _, p in self.prices])
        all_log_returns = np.diff(np.log(all_prices))

        best_vol = 0.0
        best_window = 0
        best_n = 0

        for window in self.windows:
            returns = all_log_returns[-window:] if len(all_log_returns) >= window else all_log_returns
            if len(returns) < 3:
                continue

            std_1m = np.std(returns, ddof=1)
            annualized = std_1m * math.sqrt(MINUTES_PER_YEAR)

            if annualized > best_vol:
                best_vol = annualized
                best_window = window
                best_n = len(returns)

        if best_vol <= 0:
            return None

        estimate = VolEstimate(
            annualized=best_vol,
            window_minutes=best_window,
            num_returns=best_n,
            timestamp=datetime.now(timezone.utc),
        )
        log.debug(f"Vol estimate: σ={best_vol:.4f} (window={best_window}m, n={best_n})")
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
    Normal approximation for P(BTC ≥ strike at expiry):
        T = time_left_minutes / 525_600
        d = (S - K) / (S * σ * √T)
        P(Yes) = Φ(d)
    """
    if time_left_minutes <= 0:
        return (1.0 if btc_price >= strike else 0.0,
                float('inf') if btc_price >= strike else float('-inf'))

    T = time_left_minutes / MINUTES_PER_YEAR
    denom = btc_price * sigma_annual * math.sqrt(T)
    if denom <= 0:
        return (1.0 if btc_price >= strike else 0.0, 0.0)

    d = (btc_price - strike) / denom
    p_yes = norm.cdf(d)
    return (p_yes, d)


# =============================================================================
# TRADES TAPE TRACKER (per-coin taker-buy volume from WS `trades` feed)
# =============================================================================
#
# Conviction signal (Option B): for each outcome coin, maintain a rolling
# window of trades and sum the size of those whose taker side was a buyer
# ("aggressive buying"). Captures whale stacking on one side without the
# noise of total-volume comparisons.
#
# Side normalization handles both schemas Hyperliquid has used:
#   • "B"/"S" or "B"/"A"  (legacy single-letter)
#   • "Buy"/"Sell"        (current)

class TradesTracker:
    """Per-coin rolling tape of (timestamp, side, size). Single-threaded."""

    def __init__(self, lookback_minutes: float):
        self.lookback_sec = lookback_minutes * 60
        self.trades: dict[str, list[tuple[float, str, float]]] = {}
        self._last_trade_ts: dict[str, float] = {}

    @staticmethod
    def _normalize_side(raw: str) -> str:
        s = (raw or "").strip().upper()
        if s in ("B", "BUY", "BID"):
            return "Buy"
        if s in ("S", "A", "SELL", "ASK"):
            return "Sell"
        return "Unknown"

    def add_trade(self, coin: str, ts_sec: float, side_raw: str, size: float):
        side = self._normalize_side(side_raw)
        bucket = self.trades.setdefault(coin, [])
        bucket.append((ts_sec, side, size))
        self._last_trade_ts[coin] = max(self._last_trade_ts.get(coin, 0.0), ts_sec)
        # Keep up to 2x lookback for safety margin; trim only when oldest entry is stale.
        cutoff = ts_sec - self.lookback_sec * 2
        if bucket and bucket[0][0] < cutoff:
            self.trades[coin] = [t for t in bucket if t[0] >= cutoff]

    def taker_buy_volume(self, coin: str, now_ts: Optional[float] = None) -> float:
        if coin not in self.trades:
            return 0.0
        if now_ts is None:
            now_ts = time.time()
        cutoff = now_ts - self.lookback_sec
        return sum(sz for ts, side, sz in self.trades[coin]
                   if side == "Buy" and ts >= cutoff)

    def is_fresh(self, coin: str, max_age_sec: float, now_ts: Optional[float] = None) -> bool:
        if now_ts is None:
            now_ts = time.time()
        last = self._last_trade_ts.get(coin, 0.0)
        return last > 0 and (now_ts - last) <= max_age_sec

    def drop_coin(self, coin: str):
        self.trades.pop(coin, None)
        self._last_trade_ts.pop(coin, None)


# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================
#
#   base_risk   = equity · BASE_RISK_PCT
#   edge_scale  = clamp(edge / EDGE_REFERENCE, 0.6, 2.0)
#   risk_amount = base_risk · edge_scale
#   risk_amount = clamp(risk_amount, MIN_TRADE_USD,
#                       equity·MAX_RISK_PCT, MAX_TRADE_USD, TRAINING_WHEELS)
#   shares      = floor(risk_amount / entry_price)
#
# Defaults tuned for $800 equity → $80 baseline / $100 hard cap.

def compute_position_size(
    equity: float,
    edge: float,
    entry_price: float,
) -> tuple[float, int]:
    """Returns (risk_amount_usd, shares)."""
    if equity <= 0 or entry_price <= 0:
        return 0.0, 0

    base = CONFIG["base_risk_pct"]
    max_pct = CONFIG["max_risk_pct"]
    min_usd = CONFIG["min_trade_usd"]
    max_usd = CONFIG["max_trade_usd"]
    edge_ref = CONFIG["edge_reference"]

    base_risk = equity * base
    edge_scale = edge / edge_ref if edge_ref > 0 else 1.0
    risk_amount = base_risk * min(2.0, max(0.6, edge_scale))
    risk_amount = max(min_usd, min(risk_amount, equity * max_pct, max_usd))

    # Training-wheels: hard cap regardless of equity. None/empty disables.
    tw = CONFIG.get("training_wheels_max_usd")
    if tw is not None and tw > 0:
        risk_amount = min(risk_amount, tw)

    shares = math.floor(risk_amount / entry_price)
    return risk_amount, shares


# =============================================================================
# ORDERBOOK ANALYSIS (slippage check before crossing)
# =============================================================================

def check_orderbook_depth(
    client: HyperliquidClient,
    coin: str,
    is_buy: bool,
    size_usd: float,
) -> tuple[bool, float, float]:
    """
    Walk the book on the side we're crossing; return (has_depth, avg_fill, best_price).
    """
    try:
        book = client.get_l2_book(coin)
    except Exception as e:
        log.warning(f"Failed to fetch orderbook for {coin}: {e}")
        return False, 0.0, 0.0

    levels = book.get("levels", [[], []])
    if len(levels) < 2:
        return False, 0.0, 0.0

    side = levels[1] if is_buy else levels[0]
    if not side:
        log.debug(f"No {'asks' if is_buy else 'bids'} in book for {coin}")
        return False, 0.0, 0.0

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

    best_price = float(side[0].get("px", side[0][0]) if isinstance(side[0], dict) else side[0][0])

    if total_size <= 0 or remaining_usd > size_usd * 0.5:
        log.debug(f"Insufficient depth for {coin}: filled only ${size_usd - remaining_usd:.2f} of ${size_usd:.2f}")
        return False, 0.0, best_price

    avg_fill = total_cost / total_size
    slippage = abs(avg_fill - best_price) / best_price if best_price > 0 else 1.0

    has_depth = slippage <= CONFIG["max_slippage_pct"]
    if not has_depth:
        log.debug(f"Slippage too high for {coin}: {slippage:.4f} > {CONFIG['max_slippage_pct']}")

    return has_depth, avg_fill, best_price


# =============================================================================
# DEPTH IMBALANCE (Option A: same-side bid depth across both coins)
# =============================================================================

def _sum_bids_in_band(
    book: dict,
    mid: float,
    band_pct: float,
    fallback_levels: int,
    min_qualifying: int,
) -> float:
    """
    Sum bid sizes within ±band_pct of mid for the favorable coin's bid side.
    Falls back to top-N bids if fewer than min_qualifying levels are in band —
    typical for sparse outcome books with one big bid then nothing.
    """
    levels = book.get("levels", [[], []])
    if not levels:
        return 0.0
    bids = levels[0] if len(levels) >= 1 else []
    if not bids:
        return 0.0

    parsed: list[tuple[float, float]] = []
    for lvl in bids:
        try:
            px = float(lvl.get("px", lvl[0]) if isinstance(lvl, dict) else lvl[0])
            sz = float(lvl.get("sz", lvl[1]) if isinstance(lvl, dict) else lvl[1])
        except (IndexError, ValueError, TypeError):
            continue
        if px > 0 and sz > 0:
            parsed.append((px, sz))

    if not parsed:
        return 0.0

    if mid > 0:
        low = mid * (1.0 - band_pct)
        in_band = [(px, sz) for px, sz in parsed if low <= px <= mid]
    else:
        in_band = []

    if len(in_band) >= min_qualifying:
        return sum(sz for _, sz in in_band)

    # Fallback: top N bids (HL returns sorted, best first)
    return sum(sz for _, sz in parsed[:fallback_levels])


def compute_depth_imbalance(
    client: "HyperliquidClient",
    favorable_coin: str,
    favorable_mid: float,
    unfavorable_coin: str,
    unfavorable_mid: float,
) -> tuple[float, float, Optional[float]]:
    """
    Returns (favorable_bid_depth, unfavorable_bid_depth, imbalance ∈ [0,1]).
    imbalance = favorable / (favorable + unfavorable). None if both are zero.
    """
    band_pct = CONFIG["depth_price_band_pct"]
    fallback = CONFIG["depth_fallback_levels"]
    min_qual = CONFIG["depth_min_qualifying_levels"]

    try:
        fav_book = client.get_l2_book(favorable_coin)
        unfav_book = client.get_l2_book(unfavorable_coin)
    except Exception as e:
        log.warning(f"Depth fetch failed for {favorable_coin}/{unfavorable_coin}: {e}")
        return 0.0, 0.0, None

    fav = _sum_bids_in_band(fav_book, favorable_mid, band_pct, fallback, min_qual)
    unfav = _sum_bids_in_band(unfav_book, unfavorable_mid, band_pct, fallback, min_qual)
    total = fav + unfav
    if total <= 0:
        return fav, unfav, None
    return fav, unfav, fav / total


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class HyperliquidSniperBot:
    """Orchestrates discovery, monitoring, edge evaluation, conviction, execution."""

    def __init__(self, private_key: str):
        self.client = HyperliquidClient(private_key)
        self.vol_calc = VolatilityCalculator()
        self.trades_tracker = TradesTracker(CONFIG["volume_lookback_min"])
        self.contracts: dict[str, BinaryContract] = {}  # name -> contract
        self.open_positions: int = 0
        self._shutdown = threading.Event()
        self._btc_price: float = 0.0
        self._mids: dict[str, float] = {}
        self._equity: float = 0.0
        self._wallet_address: Optional[str] = self._derive_address(private_key)
        self._last_discovery = 0.0
        self._last_vol_update = 0.0
        self._last_equity_update = 0.0
        self._last_heartbeat = 0.0
        self._active_mode: bool = False

        # WS trades subscription state
        self._desired_trade_coins: set[str] = set()
        self._actual_trade_subs: set[str] = set()
        self._side_field_validated: bool = False

        # Daily loss kill-switch state (persisted to disk so restarts honor budget)
        self._day_utc: str = ""
        self._day_start_equity: float = 0.0
        self._kill_switch_engaged: bool = False
        self._load_daily_state()

        # HTTP healthcheck server (Railway expects $PORT to bind)
        self._health_server: Optional[HTTPServer] = None

    @staticmethod
    def _derive_address(private_key: str) -> Optional[str]:
        try:
            import eth_account
            return eth_account.Account.from_key(private_key).address
        except Exception as e:
            log.error(f"Could not derive wallet address from private key: {e}")
            return None

    # ── Daily loss kill-switch ──

    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_daily_state(self):
        path = CONFIG["daily_state_file"]
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self._day_utc = state.get("day_utc", "")
            self._day_start_equity = float(state.get("day_start_equity", 0.0) or 0.0)
            self._kill_switch_engaged = bool(state.get("kill_switch_engaged", False))
            log.info(
                f"Loaded daily state from {path}: day={self._day_utc} "
                f"start_equity=${self._day_start_equity:.2f} kill={self._kill_switch_engaged}"
            )
        except FileNotFoundError:
            log.info(f"No daily state file at {path} — fresh start")
        except Exception as e:
            log.warning(f"Failed to load daily state {path}: {e} — ignoring")

    def _save_daily_state(self):
        path = CONFIG["daily_state_file"]
        try:
            with open(path, "w") as f:
                json.dump({
                    "day_utc": self._day_utc,
                    "day_start_equity": self._day_start_equity,
                    "kill_switch_engaged": self._kill_switch_engaged,
                }, f)
        except Exception as e:
            log.warning(f"Failed to persist daily state {path}: {e}")

    def _kill_switch_can_trade(self) -> bool:
        """Refresh day boundary, recompute kill-switch from latest equity."""
        today = self._today_utc()
        if today != self._day_utc:
            self._day_utc = today
            self._day_start_equity = self._equity
            self._kill_switch_engaged = False
            log.info(
                f"UTC day rollover -> {today}, day_start_equity=${self._day_start_equity:.2f}"
            )
            self._save_daily_state()
            return True

        if self._day_start_equity <= 0:
            if self._equity > 0:
                self._day_start_equity = self._equity
                self._save_daily_state()
            return True

        limit_pct = CONFIG["daily_loss_limit_pct"]
        loss = self._day_start_equity - self._equity
        loss_pct = loss / self._day_start_equity if self._day_start_equity > 0 else 0.0
        if loss_pct >= limit_pct and not self._kill_switch_engaged:
            self._kill_switch_engaged = True
            log.error(
                f"KILL-SWITCH ENGAGED: day loss ${loss:.2f} ({loss_pct*100:.2f}%) "
                f">= {limit_pct*100:.2f}% of day-start ${self._day_start_equity:.2f}"
            )
            self._save_daily_state()
        return not self._kill_switch_engaged

    # ── Healthcheck ──

    def _start_health_server(self):
        bot = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):  # silence default access logs
                return
            def do_GET(self):
                if self.path in ("/", "/health"):
                    payload = {
                        "status": "ok",
                        "wallet": bot._wallet_address,
                        "equity_usd": round(bot._equity, 2),
                        "btc_price": bot._btc_price,
                        "contracts_tracked": len(bot.contracts),
                        "open_positions": bot.open_positions,
                        "kill_switch_engaged": bot._kill_switch_engaged,
                        "day_utc": bot._day_utc,
                        "day_start_equity": bot._day_start_equity,
                        "dry_run": CONFIG["dry_run"],
                        "trade_subs_desired": len(bot._desired_trade_coins),
                        "trade_subs_actual": len(bot._actual_trade_subs),
                        "active_mode": bot._active_mode,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()
        port = CONFIG["health_port"]
        try:
            self._health_server = HTTPServer(("0.0.0.0", port), Handler)
            t = threading.Thread(target=self._health_server.serve_forever, daemon=True)
            t.start()
            log.info(f"Healthcheck listening on :{port}/health")
        except Exception as e:
            log.warning(f"Could not start healthcheck on port {port}: {e}")

    def _heartbeat(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_heartbeat) < CONFIG["heartbeat_sec"]:
            return
        self._last_heartbeat = now
        log.info(
            f"HB equity=${self._equity:.2f} btc={self._btc_price:.2f} "
            f"contracts={len(self.contracts)} open={self.open_positions} "
            f"trade_subs={len(self._actual_trade_subs)}/{len(self._desired_trade_coins)} "
            f"kill={self._kill_switch_engaged} day={self._day_utc}"
        )

    # ── Lifecycle ──

    def start(self):
        """Start the bot — runs until Ctrl+C."""
        log.info("=" * 60)
        log.info("HIP-4 BTC PriceBinary Sniper Bot — Starting")
        log.info(
            f"Risk: base={CONFIG['base_risk_pct']*100:.2f}%  "
            f"max={CONFIG['max_risk_pct']*100:.2f}%  "
            f"trade=${CONFIG['min_trade_usd']}-{CONFIG['max_trade_usd']}  "
            f"tw_cap={('$' + str(CONFIG['training_wheels_max_usd'])) if CONFIG['training_wheels_max_usd'] else 'off'}  "
            f"edge_ref={CONFIG['edge_reference']}"
        )
        log.info(
            f"Stat gate: edge≥{CONFIG['edge_threshold']}  "
            f"min_d_base={CONFIG['min_d_base']}  min_d_floor={CONFIG['min_d_floor']}  "
            f"t≤{CONFIG['max_time_left_minutes']}min "
            f"(ideal <{CONFIG['ideal_time_left_minutes']}min)"
        )
        log.info(
            f"Conviction gate: vol_ratio={CONFIG['volume_ratio_min']}-{CONFIG['volume_ratio_max']}  "
            f"depth_imb={CONFIG['depth_imb_min']}-{CONFIG['depth_imb_max']}  "
            f"vol_lookback={CONFIG['volume_lookback_min']}min  "
            f"depth_band=±{CONFIG['depth_price_band_pct']*100:.1f}%"
        )
        log.info(
            f"Pure-flow: t<{CONFIG['pure_flow_time_left_min']}min  "
            f"edge≥{CONFIG['pure_flow_edge_threshold']}  "
            f"freshness≤{CONFIG['pure_flow_freshness_sec']:.0f}s"
        )
        log.info(f"Wallet: {self._wallet_address or '(unknown)'}")
        log.info("=" * 60)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._start_health_server()
        self._heartbeat(force=True)

        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down...")
        except Exception as e:
            log.error(f"Fatal error in async loop: {e}", exc_info=True)
            log.info("Falling back to polling mode (conviction gate will be flow-blind)...")
            self._run_polling()

    def _handle_signal(self, signum, frame):
        log.info(f"Received signal {signum} — initiating graceful shutdown...")
        self._shutdown.set()

    def _nearest_t_left_min(self) -> float:
        """Minutes until soonest non-traded, non-expired contract expiry. inf if none."""
        now = datetime.now(timezone.utc)
        candidates = [
            (c.expiry_utc - now).total_seconds() / 60
            for c in self.contracts.values()
            if not c.traded and c.expiry_utc > now
        ]
        return min(candidates) if candidates else float("inf")

    def _loop_sleep_sec(self) -> float:
        """Tight cadence inside the entry window; slow cadence when nothing is close."""
        nearest = self._nearest_t_left_min()
        active_window = CONFIG["max_time_left_minutes"]
        if nearest <= active_window:
            if not self._active_mode:
                self._active_mode = True
                log.info(
                    f"ACTIVE MODE: t_left={nearest:.1f}min ≤ {active_window}min — "
                    f"snipe cadence {CONFIG['active_poll_interval_sec']}s"
                )
            return CONFIG["active_poll_interval_sec"]
        if self._active_mode:
            self._active_mode = False
            log.info(f"Exiting active mode — idle cadence {CONFIG['price_poll_interval_sec']}s")
        return CONFIG["price_poll_interval_sec"]

    # ── Async main loop (WebSocket-based) ──

    async def _run_async(self):
        """Main async loop using WebSocket for real-time price updates."""
        # Initial setup — bootstrap vol from REST (with retry); WS candle feed takes over.
        self._refresh_btc_price()
        await self._bootstrap_volatility()
        self._refresh_equity()
        self._discover_contracts()

        # Start WebSocket listener in background
        ws_task = asyncio.create_task(self._ws_listener())

        try:
            while not self._shutdown.is_set():
                now = time.time()

                if now - self._last_discovery >= CONFIG["discovery_interval_sec"]:
                    self._discover_contracts()
                    self._last_discovery = now

                # Vol fallback: only fire REST if WS candle feed has gone stale (>5min).
                vol_stale = (
                    not self.vol_calc.prices
                    or time.time() - self.vol_calc.prices[-1][0] > 300
                )
                if vol_stale and now - self._last_vol_update >= CONFIG["vol_refresh_sec"]:
                    self._refresh_volatility()
                    self._last_vol_update = now

                if now - self._last_equity_update >= CONFIG["equity_refresh_sec"]:
                    self._refresh_equity()

                # Refresh prices (fallback if WS is lagging)
                self._refresh_btc_price()

                self._kill_switch_can_trade()
                self._heartbeat()
                self._evaluate_all_contracts()
                self._cleanup_expired()

                await asyncio.sleep(self._loop_sleep_sec())
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            log.info("Bot shutdown complete.")

    async def _ws_listener(self):
        """Streams allMids + BTC 1m candles + per-coin trades; keepalive ping; sub manager."""
        while not self._shutdown.is_set():
            ping_task = None
            sub_mgr_task = None
            try:
                async with websockets.connect(
                    CONFIG["ws_url"],
                    ping_interval=20,
                    ping_timeout=None,  # rely on HL app-level ping; don't kill conn on missing protocol pong
                    close_timeout=5,
                ) as ws:
                    # Initial subs: allMids + candle@BTC@1m + every currently desired trade coin.
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "allMids"},
                    }))
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "candle", "coin": "BTC", "interval": "1m"},
                    }))
                    self._actual_trade_subs.clear()
                    for coin in list(self._desired_trade_coins):
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": coin},
                        }))
                        self._actual_trade_subs.add(coin)
                    log.info(
                        f"WebSocket connected — allMids + candle@BTC@1m + "
                        f"{len(self._actual_trade_subs)} trade tape(s)"
                    )

                    # HL closes idle connections at 60s; send {"method":"ping"} every 30s.
                    ping_task = asyncio.create_task(self._ws_ping_loop(ws))
                    sub_mgr_task = asyncio.create_task(self._ws_sub_manager(ws))

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
            finally:
                for t in (ping_task, sub_mgr_task):
                    if t is not None and not t.done():
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

            if not self._shutdown.is_set():
                await asyncio.sleep(CONFIG["ws_reconnect_delay_sec"])

    async def _ws_ping_loop(self, ws):
        """Send {'method':'ping'} every WS_PING_INTERVAL_SEC. HL idle timeout is 60s."""
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(CONFIG["ws_ping_interval_sec"])
                if self._shutdown.is_set():
                    return
                await ws.send(json.dumps({"method": "ping"}))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug(f"Ping loop ended: {e}")

    async def _ws_sub_manager(self, ws):
        """Reconciles desired vs actual trade subscriptions every N seconds."""
        interval = CONFIG["ws_sub_reconcile_sec"]
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(interval)
                # Subscribe to newly-desired coins
                to_add = self._desired_trade_coins - self._actual_trade_subs
                for coin in list(to_add):
                    try:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": coin},
                        }))
                        self._actual_trade_subs.add(coin)
                        log.info(f"WS: subscribed to trades:{coin}")
                    except Exception as e:
                        log.warning(f"WS sub failed for {coin}: {e}")
                        return  # let outer loop reconnect

                # Unsubscribe from no-longer-desired coins
                to_remove = self._actual_trade_subs - self._desired_trade_coins
                for coin in list(to_remove):
                    try:
                        await ws.send(json.dumps({
                            "method": "unsubscribe",
                            "subscription": {"type": "trades", "coin": coin},
                        }))
                        self._actual_trade_subs.discard(coin)
                        self.trades_tracker.drop_coin(coin)
                        log.info(f"WS: unsubscribed from trades:{coin}")
                    except Exception as e:
                        log.warning(f"WS unsub failed for {coin}: {e}")
                        return
        except asyncio.CancelledError:
            raise

    def _handle_ws_message(self, msg: dict):
        """Process incoming WebSocket messages."""
        channel = msg.get("channel", "")
        data = msg.get("data", {})

        if channel == "allMids":
            mids = data.get("mids", data)
            if isinstance(mids, dict):
                self._mids.update(mids)
                # Update BTC price
                for key in ("BTC", "BTC-PERP", "@1"):
                    if key in mids:
                        try:
                            self._btc_price = float(mids[key])
                        except (ValueError, TypeError):
                            pass
                        break

        elif channel == "candle":
            # Server sends Candle | Candle[]; handle both shapes.
            items = data if isinstance(data, list) else [data]
            for c in items:
                if not isinstance(c, dict):
                    continue
                if c.get("s") != "BTC" or c.get("i") != "1m":
                    continue
                try:
                    t_ms = c.get("T")
                    close = c.get("c")
                    if t_ms is None or close is None:
                        continue
                    self.vol_calc.set_price(float(t_ms) / 1000.0, float(close))
                except (ValueError, TypeError):
                    continue

        elif channel == "trades":
            # `data` is a list of trade dicts. One-shot validate the schema.
            if isinstance(data, list):
                for trade in data:
                    if not isinstance(trade, dict):
                        continue
                    coin = trade.get("coin")
                    side_raw = trade.get("side", "")
                    try:
                        sz = float(trade.get("sz", 0))
                        ts_ms = trade.get("time", trade.get("t", 0))
                        ts = ts_ms / 1000.0 if ts_ms > 1e12 else float(ts_ms)
                    except (ValueError, TypeError):
                        continue
                    if not coin or sz <= 0 or ts <= 0:
                        continue

                    if not self._side_field_validated:
                        if str(side_raw).strip().upper() not in ("B", "S", "A", "BUY", "SELL"):
                            log.warning(
                                f"trades.side has unexpected value {side_raw!r} for {coin}; "
                                "conviction signal may be unreliable"
                            )
                        else:
                            log.info(f"trades schema validated (side={side_raw!r} on {coin})")
                        self._side_field_validated = True

                    self.trades_tracker.add_trade(coin, ts, side_raw, sz)

        elif channel == "pong":
            return  # heartbeat ack
        elif channel == "subscriptionResponse":
            return  # subscribe ack

    # ── Polling fallback ──

    def _run_polling(self):
        """Fallback polling loop if WebSocket fails. Conviction gate will block all trades."""
        log.warning(
            "POLLING MODE: no WS trades data; conviction gate will block all trades. "
            "Investigate WS failure to restore normal operation."
        )
        while not self._shutdown.is_set():
            try:
                now = time.time()

                if now - self._last_discovery >= CONFIG["discovery_interval_sec"]:
                    self._discover_contracts()
                    self._last_discovery = now

                if now - self._last_vol_update >= CONFIG["vol_refresh_sec"]:
                    self._refresh_volatility()
                    self._last_vol_update = now

                if now - self._last_equity_update >= CONFIG["equity_refresh_sec"]:
                    self._refresh_equity()

                self._refresh_btc_price()
                self._kill_switch_can_trade()
                self._heartbeat()
                self._evaluate_all_contracts()
                self._cleanup_expired()

                time.sleep(self._loop_sleep_sec())

            except Exception as e:
                log.error(f"Error in polling loop: {e}", exc_info=True)
                time.sleep(5)

        log.info("Polling loop shutdown complete.")

    # ── Core operations ──

    def _refresh_btc_price(self):
        """Fetch current BTC price and outcome mids via REST."""
        try:
            mids = self.client.get_all_mids()
            if isinstance(mids, dict):
                self._mids.update(mids)
                for key in ("BTC", "BTC-PERP", "@1"):
                    if key in mids:
                        try:
                            self._btc_price = float(mids[key])
                        except (ValueError, TypeError):
                            pass
                        break
                # Update each contract's Yes/No mid via its outcome coin keys.
                for contract in self.contracts.values():
                    yes_raw = mids.get(contract.yes_coin)
                    no_raw = mids.get(contract.no_coin)
                    if yes_raw is not None:
                        try:
                            contract.last_yes_mid = float(yes_raw)
                        except (ValueError, TypeError):
                            pass
                    if no_raw is not None:
                        try:
                            contract.last_no_mid = float(no_raw)
                        except (ValueError, TypeError):
                            pass
                    if contract.last_yes_mid is not None and contract.last_no_mid is None:
                        contract.last_no_mid = 1.0 - contract.last_yes_mid
                    elif contract.last_no_mid is not None and contract.last_yes_mid is None:
                        contract.last_yes_mid = 1.0 - contract.last_no_mid
        except Exception as e:
            log.debug(f"Price refresh failed: {e}")

    def _refresh_volatility(self):
        """Update BTC volatility estimate from recent candles (REST fallback path)."""
        candles = self.client.get_candles("BTC", interval="1m", lookback=120)
        if candles:
            self.vol_calc.load_from_candles(candles)
        self._last_vol_update = time.time()

    async def _bootstrap_volatility(self, max_attempts: int = 3):
        """
        One-shot REST candle fetch at startup with backoff. WS candle feed
        takes over from here, so we only need ~5 closed candles before the
        feed warms up; even a partial bootstrap is fine.
        """
        for attempt in range(1, max_attempts + 1):
            candles = self.client.get_candles("BTC", interval="1m", lookback=120)
            if candles:
                self.vol_calc.load_from_candles(candles)
                self._last_vol_update = time.time()
                log.info(f"Vol bootstrap: loaded {len(self.vol_calc.prices)} candles from REST")
                return
            delay = 5 * attempt
            log.warning(
                f"Vol bootstrap attempt {attempt}/{max_attempts} got no candles "
                f"(likely 429) — retry in {delay}s"
            )
            await asyncio.sleep(delay)
        log.warning("Vol bootstrap failed all attempts — relying on WS candle feed to fill")

    def _refresh_equity(self):
        """
        Equity = spot USDH balance of the *trading account* (master wallet if
        we're using an agent, otherwise the signer itself). HIP-4 outcomes are
        USDH-collateralized.
        """
        master = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip() or None
        addr = master or self._wallet_address
        if not addr:
            return
        try:
            equity = self.client.get_usdh_balance(addr)
            if equity > 0:
                if abs(equity - self._equity) > 0.01:
                    log.info(f"Equity refreshed (USDH @ {addr[:10]}…): ${equity:.2f}")
                self._equity = equity
            else:
                log.warning(f"USDH spot balance is 0 for {addr} — bot will not trade")
        except Exception as e:
            log.debug(f"Equity refresh failed: {e}")
        self._last_equity_update = time.time()

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
        # Recompute desired trade subs (Yes + No coins of every active contract)
        desired = set()
        for c in self.contracts.values():
            desired.add(c.yes_coin)
            desired.add(c.no_coin)
        self._desired_trade_coins = desired
        self._last_discovery = time.time()

    # ── Decision pipeline ──

    def _make_decision(self, contract: BinaryContract, vol_estimate: VolEstimate) -> TradeDecision:
        """
        Three gates in cheap-to-expensive order:
          1. Time + statistical (free)
          2. Volume conviction (local tape data)
          3. Depth conviction (2 L2 fetches)
        Each gate's failure short-circuits with a TradeDecision(action=SKIP).
        """
        now = datetime.now(timezone.utc)
        time_left_min = (contract.expiry_utc - now).total_seconds() / 60
        btc_price = self._btc_price
        yes_mid = contract.last_yes_mid

        def _skip(reason: str, mode: str = "STAT", **extra) -> TradeDecision:
            return TradeDecision(
                timestamp=now, contract=contract.asset_name, btc_price=btc_price,
                strike=contract.target_price, time_left_min=time_left_min,
                sigma=vol_estimate.annualized,
                d_value=extra.get("d_value", 0.0),
                p_model_yes=extra.get("p_yes", 0.0),
                market_mid_yes=yes_mid if yes_mid is not None else 0.0,
                edge=extra.get("edge", 0.0),
                action="SKIP", reason=reason, mode=mode,
                min_d_required=extra.get("min_d_required", 0.0),
                vol_ratio=extra.get("vol_ratio"),
                vol_ratio_required=extra.get("vol_ratio_required", 0.0),
                depth_imbalance=extra.get("depth_imbalance"),
                depth_imb_required=extra.get("depth_imb_required", 0.0),
            )

        # ── Gate 1a: time window ──
        if time_left_min <= 0:
            return _skip("expired")
        if time_left_min > CONFIG["max_time_left_minutes"]:
            return _skip(f"time_left={time_left_min:.1f}m > max={CONFIG['max_time_left_minutes']}m")
        if yes_mid is None or yes_mid <= 0 or yes_mid >= 1:
            return _skip(f"invalid yes_mid={yes_mid}")

        # ── Gate 1b: statistical ──
        p_yes, d_val = compute_model_probability(
            btc_price=btc_price, strike=contract.target_price,
            time_left_minutes=time_left_min, sigma_annual=vol_estimate.annualized,
        )
        p_no = 1.0 - p_yes
        no_mid = 1.0 - yes_mid
        edge_yes = p_yes - yes_mid
        edge_no = p_no - no_mid

        is_pure_flow = time_left_min < CONFIG["pure_flow_time_left_min"]
        edge_threshold = (
            CONFIG["pure_flow_edge_threshold"] if is_pure_flow else CONFIG["edge_threshold"]
        )

        if is_pure_flow:
            min_d_required = CONFIG["min_d_floor"]
            # Log pure-flow entry once per contract
            if not contract.pure_flow_logged:
                log.info(
                    f"[{contract.asset_name}] PURE_FLOW_MODE active "
                    f"(t={time_left_min*60:.0f}s, |d|≥{min_d_required}, edge≥{edge_threshold})"
                )
                contract.pure_flow_logged = True
        else:
            min_d_required = max(
                CONFIG["min_d_floor"],
                CONFIG["min_d_base"] * (time_left_min / CONFIG["max_time_left_minutes"]),
            )

        intended_side: Optional[str] = None
        edge: float = 0.0
        action_pending: str = ""
        if edge_yes >= edge_threshold and d_val >= min_d_required:
            intended_side, edge, action_pending = "YES", edge_yes, "BUY_YES"
        elif edge_no >= edge_threshold and d_val <= -min_d_required:
            intended_side, edge, action_pending = "NO", edge_no, "BUY_NO"
        else:
            reasons = []
            if abs(d_val) < min_d_required:
                reasons.append(f"|d|={abs(d_val):.3f} < min_d={min_d_required:.3f}")
            max_edge = max(edge_yes, edge_no)
            if max_edge < edge_threshold:
                reasons.append(f"max_edge={max_edge:.4f} < {edge_threshold}")
            return _skip(
                f"STAT_FAIL: {'; '.join(reasons) if reasons else 'no edge'}",
                mode="PURE_FLOW" if is_pure_flow else "STAT",
                d_value=d_val, p_yes=p_yes, edge=max_edge, min_d_required=min_d_required,
            )

        mode = (
            "PURE_FLOW" if is_pure_flow
            else ("HYBRID" if abs(d_val) < CONFIG["min_d_base"] else "STAT")
        )

        # ── Gate 2: conviction thresholds ──
        if is_pure_flow:
            vol_ratio_required = CONFIG["volume_ratio_max"]
            depth_imb_required = CONFIG["depth_imb_max"]
        else:
            conf = min(1.0, abs(d_val) / max(1e-6, CONFIG["min_d_base"]))
            vol_ratio_required = (
                CONFIG["volume_ratio_max"]
                - conf * (CONFIG["volume_ratio_max"] - CONFIG["volume_ratio_min"])
            )
            depth_imb_required = (
                CONFIG["depth_imb_max"]
                - conf * (CONFIG["depth_imb_max"] - CONFIG["depth_imb_min"])
            )

        favorable_coin = contract.yes_coin if intended_side == "YES" else contract.no_coin
        unfavorable_coin = contract.no_coin if intended_side == "YES" else contract.yes_coin
        favorable_mid = yes_mid if intended_side == "YES" else no_mid
        unfavorable_mid = no_mid if intended_side == "YES" else yes_mid

        # Pure-flow demands fresh tape — refuse to fire on stale data
        if is_pure_flow and not self.trades_tracker.is_fresh(
            favorable_coin, max_age_sec=CONFIG["pure_flow_freshness_sec"]
        ):
            return _skip(
                f"PURE_FLOW_NO_FRESH_DATA: no trades on {favorable_coin} "
                f"in last {CONFIG['pure_flow_freshness_sec']:.0f}s",
                mode=mode, d_value=d_val, p_yes=p_yes, edge=edge,
                min_d_required=min_d_required,
            )

        fav_vol = self.trades_tracker.taker_buy_volume(favorable_coin)
        unfav_vol = self.trades_tracker.taker_buy_volume(unfavorable_coin)
        vol_ratio = fav_vol / max(1e-6, unfav_vol)

        if vol_ratio < vol_ratio_required:
            return _skip(
                f"STAT_PASS_FLOW_FAIL: vol_ratio={vol_ratio:.2f} < req={vol_ratio_required:.2f} "
                f"(fav_taker_buy={fav_vol:.2f} unfav_taker_buy={unfav_vol:.2f})",
                mode=mode, d_value=d_val, p_yes=p_yes, edge=edge,
                min_d_required=min_d_required,
                vol_ratio=vol_ratio, vol_ratio_required=vol_ratio_required,
            )

        # ── Gate 3: depth imbalance (costs 2 L2 fetches) ──
        fav_depth, unfav_depth, depth_imb = compute_depth_imbalance(
            self.client, favorable_coin, favorable_mid,
            unfavorable_coin, unfavorable_mid,
        )

        if depth_imb is None or depth_imb < depth_imb_required:
            depth_str = f"{depth_imb:.3f}" if depth_imb is not None else "n/a"
            return _skip(
                f"STAT_PASS_FLOW_FAIL: depth_imb={depth_str} < req={depth_imb_required:.3f} "
                f"(fav_bids={fav_depth:.2f} unfav_bids={unfav_depth:.2f})",
                mode=mode, d_value=d_val, p_yes=p_yes, edge=edge,
                min_d_required=min_d_required,
                vol_ratio=vol_ratio, vol_ratio_required=vol_ratio_required,
                depth_imbalance=depth_imb, depth_imb_required=depth_imb_required,
            )

        # ── All gates passed ──
        return TradeDecision(
            timestamp=now, contract=contract.asset_name, btc_price=btc_price,
            strike=contract.target_price, time_left_min=time_left_min,
            sigma=vol_estimate.annualized,
            d_value=d_val, p_model_yes=p_yes, market_mid_yes=yes_mid,
            edge=edge, action=action_pending,
            reason=f"ALL_GATES_PASS mode={mode} min_d={min_d_required:.3f}",
            mode=mode, min_d_required=min_d_required,
            vol_ratio=vol_ratio, vol_ratio_required=vol_ratio_required,
            depth_imbalance=depth_imb, depth_imb_required=depth_imb_required,
        )

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
            if contract.traded:
                continue

            # Always pull WS-fresh mids at eval time — don't latch to a stale REST snapshot.
            yes_raw = self._mids.get(contract.yes_coin)
            no_raw = self._mids.get(contract.no_coin)
            try:
                if yes_raw is not None:
                    contract.last_yes_mid = float(yes_raw)
                if no_raw is not None:
                    contract.last_no_mid = float(no_raw)
            except (ValueError, TypeError):
                pass
            if contract.last_yes_mid is None:
                continue
            if contract.last_no_mid is None:
                contract.last_no_mid = 1.0 - contract.last_yes_mid

            if contract.last_yes_mid <= 0 or contract.last_yes_mid >= 1:
                continue

            time_left = (contract.expiry_utc - now).total_seconds() / 60
            # Only evaluate when within 2x the max window (saves L2 calls)
            if time_left > CONFIG["max_time_left_minutes"] * 2 or time_left <= 0:
                continue

            decision = self._make_decision(contract, vol)
            self._log_decision(decision)

            if decision.action in ("BUY_YES", "BUY_NO"):
                self._execute_trade(contract, decision)

    def _execute_trade(self, contract: BinaryContract, decision: TradeDecision):
        """Execute a trade with dynamic position sizing per the risk model."""
        if self.open_positions >= CONFIG["max_open_positions"]:
            log.warning(f"Max positions ({CONFIG['max_open_positions']}) reached — skipping {contract.asset_name}")
            return

        if self._equity <= 0:
            log.warning(f"Equity unavailable — cannot size trade on {contract.asset_name}")
            return

        if self._equity < CONFIG["equity_floor_usd"]:
            log.warning(
                f"Equity ${self._equity:.2f} below floor ${CONFIG['equity_floor_usd']:.2f} "
                f"— halting trades on {contract.asset_name}"
            )
            return

        if not self._kill_switch_can_trade():
            log.warning(f"Kill-switch active — skipping {contract.asset_name}")
            return

        is_buy_yes = decision.action == "BUY_YES"
        coin = contract.yes_coin if is_buy_yes else contract.no_coin
        current_mid = decision.market_mid_yes if is_buy_yes else (1.0 - decision.market_mid_yes)

        # Initial sizing at the mid to choose a depth probe size
        probe_risk, _ = compute_position_size(
            equity=self._equity,
            edge=decision.edge,
            entry_price=current_mid,
        )

        has_depth, _, best_price = check_orderbook_depth(
            self.client,
            coin=coin,
            is_buy=True,
            size_usd=probe_risk,
        )

        if not has_depth:
            log.info(f"Insufficient depth for {contract.asset_name} — skipping")
            return

        entry_price = max(current_mid, best_price) if best_price > 0 else current_mid

        # Edge-preservation guard: don't pay above (model_prob - (1-pres)*edge) on the side we're buying
        model_prob_side = decision.p_model_yes if is_buy_yes else (1.0 - decision.p_model_yes)
        max_price = model_prob_side - (1.0 - CONFIG["edge_preservation_min"]) * decision.edge
        if entry_price > max_price:
            log.info(
                f"Best price {entry_price:.4f} > edge-preservation cap {max_price:.4f} "
                f"for {contract.asset_name} — skipping"
            )
            return

        # Final sizing at the actual entry price
        risk_amount, shares = compute_position_size(
            equity=self._equity,
            edge=decision.edge,
            entry_price=entry_price,
        )

        if shares <= 0:
            log.warning(
                f"Sizing produced 0 shares for {contract.asset_name} "
                f"(equity=${self._equity:.2f}, risk=${risk_amount:.2f}, "
                f"price={entry_price:.4f}) — skipping"
            )
            return

        log.info("=" * 50)
        log.info(f"EXECUTING TRADE: {decision.action} on {contract.asset_name} (coin={coin})")
        log.info(f"  Mode={decision.mode}  d={decision.d_value:.3f} (min={decision.min_d_required:.3f})")
        log.info(f"  Equity=${self._equity:.2f}  Risk=${risk_amount:.2f}  "
                 f"(edge_scale={decision.edge / CONFIG['edge_reference']:.2f}x)")
        log.info(f"  BTC={decision.btc_price:.2f}  K={decision.strike:.2f}")
        log.info(f"  P_model={decision.p_model_yes:.4f}  YesMid={decision.market_mid_yes:.4f}")
        log.info(f"  Edge={decision.edge:.4f}")
        if decision.vol_ratio is not None:
            log.info(f"  VolRatio={decision.vol_ratio:.2f} (req {decision.vol_ratio_required:.2f})")
        if decision.depth_imbalance is not None:
            log.info(f"  DepthImb={decision.depth_imbalance:.3f} (req {decision.depth_imb_required:.3f})")
        log.info(f"  Entry=${entry_price:.4f}  Shares={shares}")
        log.info("=" * 50)

        if CONFIG["dry_run"]:
            log.warning(f"DRY_RUN=true — would have placed: {coin} BUY size={shares} px={entry_price:.4f}")
            contract.traded = True
            return

        result = self.client.place_limit_order(
            coin=coin,
            is_buy=is_buy_yes,
            size=shares,
            price=round(entry_price, 4),
        )

        if "error" not in result:
            contract.traded = True
            self.open_positions += 1
            oid = result.get("resting_oid") or result.get("filled_oid")
            log.info(
                f"Order accepted for {contract.asset_name}: oid={oid} "
                f"{'(resting)' if 'resting_oid' in result else '(filled)'}"
            )
        else:
            log.error(f"Order failed for {contract.asset_name}: {result['error']}")

    def _cleanup_expired(self):
        """Remove expired contracts from tracking."""
        now = datetime.now(timezone.utc)
        expired = [name for name, c in self.contracts.items() if c.expiry_utc <= now]
        for name in expired:
            c = self.contracts.pop(name)
            if c.traded:
                self.open_positions = max(0, self.open_positions - 1)
            log.info(f"Removed expired contract: {name}")
        if expired:
            # Recompute desired subs so the WS sub-manager unsubscribes
            desired = set()
            for c in self.contracts.values():
                desired.add(c.yes_coin)
                desired.add(c.no_coin)
            self._desired_trade_coins = desired

    def _log_decision(self, d: TradeDecision):
        """Log a trade decision with full parameters."""
        level = logging.INFO if d.action != "SKIP" else logging.DEBUG
        if d.time_left_min <= CONFIG["max_time_left_minutes"]:
            level = logging.INFO

        parts = [
            f"[{d.contract}]", f"mode={d.mode}",
            f"BTC={d.btc_price:.2f}", f"K={d.strike:.2f}",
            f"t={d.time_left_min:.2f}m", f"σ={d.sigma:.4f}",
            f"d={d.d_value:.3f}/min={d.min_d_required:.3f}",
            f"P={d.p_model_yes:.4f}", f"mid={d.market_mid_yes:.4f}",
            f"edge={d.edge:.4f}",
        ]
        if d.vol_ratio is not None:
            parts.append(f"vr={d.vol_ratio:.2f}/{d.vol_ratio_required:.2f}")
        if d.depth_imbalance is not None:
            parts.append(f"di={d.depth_imbalance:.3f}/{d.depth_imb_required:.3f}")
        parts.append(f"-> {d.action} ({d.reason})")
        log.log(level, " ".join(parts))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
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
#       evaluate model + conviction-gate accuracy + PnL.
# TODO: REST-based recent-trades polling so polling-fallback mode can still
#       compute taker-buy volume (currently flow-blind in fallback).
# TODO: Telegram/Discord notifications for trades and daily PnL.
# TODO: Track and log actual settlement outcomes for model + conviction calibration.
# TODO: Implement partial fill handling and order amendment logic.
# TODO: Add circuit breaker if BTC vol spikes dramatically (>3x normal).
