#!/usr/bin/env python3
"""
Smoke test: exercise every code path the production bot uses against live
Hyperliquid mainnet, WITHOUT spending USDH. Steps:

  1. Read USDH spot balance (the new equity path)
  2. Discover the live BTC priceBinary outcome
  3. Pull the L2 book for the YES coin
  4. SIGN AND PLACE a buy limit at $0.05 (far below ~$0.57 mid → unfillable)
  5. Read openOrders to verify the order landed
  6. Cancel it
  7. Read openOrders again to verify the cancel

If step 4 fails with a coin-resolution error, the installed SDK doesn't yet
recognise HIP-4 outcome coins (`#<encoding>`) and we need a manual-signing path.

Usage:
    cp .env.example .env       # set HYPERLIQUID_PRIVATE_KEY
    pip install -r requirements.txt
    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KEY = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "").strip()
if not KEY:
    print("ERROR: HYPERLIQUID_PRIVATE_KEY not set (check .env)")
    sys.exit(1)
if not KEY.startswith("0x"):
    KEY = "0x" + KEY

import requests
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

REST_URL = "https://api.hyperliquid.xyz/info"
TEST_LIMIT_PRICE = 0.05    # well below any plausible YES mid; will not fill
TEST_SIZE = 250.0          # 250 shares × $0.05 = $12.50 notional (above HL's $10 min)
                           # Locks $12.50 of master's USDH as bid margin until cancelled.


def post_info(payload: dict) -> dict:
    r = requests.post(REST_URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def banner(msg: str) -> None:
    print()
    print("=" * 70)
    print(msg)
    print("=" * 70)


def main() -> int:
    account = eth_account.Account.from_key(KEY)
    addr = account.address
    print(f"Wallet address: {addr}")

    # ── 1. USDH spot balance ──
    banner("[1/7] Spot USDH balance (sniper.py _refresh_equity path)")
    state = post_info({"type": "spotClearinghouseState", "user": addr})
    usdh = 0.0
    for b in state.get("balances", []):
        if b.get("coin") == "USDH":
            usdh = float(b.get("total") or 0)
    print(f"  USDH balance: ${usdh:,.4f}")
    if usdh < 1:
        print("  WARNING: USDH < $1. Even the unfillable test order may reject for insufficient margin.")
        print("  Continuing anyway so we can observe the SDK error message.")

    # ── 2. outcomeMeta discovery ──
    banner("[2/7] outcomeMeta — find live BTC priceBinary")
    om = post_info({"type": "outcomeMeta"})
    now = datetime.now(timezone.utc)
    target = None
    for entry in om.get("outcomes", []):
        desc = entry.get("description", "") or ""
        if "priceBinary" not in desc or "BTC" not in desc:
            continue
        m_exp = re.search(r"expiry:(\d{8}-\d{4})", desc)
        m_strike = re.search(r"targetPrice:(\d+(?:\.\d+)?)", desc)
        if not (m_exp and m_strike):
            continue
        exp = datetime.strptime(m_exp.group(1), "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
        if exp <= now:
            continue
        target = {
            "outcome": int(entry["outcome"]),
            "expiry": exp,
            "strike": float(m_strike.group(1)),
            "desc": desc,
        }
        break
    if target is None:
        print("  ERROR: no active BTC priceBinary outcome found")
        return 2
    o = target["outcome"]
    yes_coin = f"#{10 * o + 0}"
    no_coin = f"#{10 * o + 1}"
    print(f"  outcome={o}  strike=${target['strike']}  expires={target['expiry'].isoformat()}")
    print(f"  yes_coin={yes_coin}   no_coin={no_coin}")
    mins_left = (target["expiry"] - now).total_seconds() / 60
    print(f"  time_to_expiry: {mins_left:.1f} min")

    # ── 3. L2 book ──
    banner(f"[3/7] L2 orderbook for {yes_coin}")
    book = post_info({"type": "l2Book", "coin": yes_coin})
    levels = book.get("levels", [[], []])
    if len(levels) < 2 or not levels[0] or not levels[1]:
        print("  ERROR: empty book")
        return 3
    best_bid = float(levels[0][0]["px"])
    best_ask = float(levels[1][0]["px"])
    print(f"  best_bid={best_bid:.4f}  best_ask={best_ask:.4f}  spread={best_ask - best_bid:.4f}")
    print(f"  test limit ${TEST_LIMIT_PRICE} is {(best_bid - TEST_LIMIT_PRICE):.4f} below best_bid → will NOT fill")

    # ── 4. SIGNED unfillable order ──
    banner(f"[4/7] Place SIGNED limit: BUY {TEST_SIZE} {yes_coin} @ ${TEST_LIMIT_PRICE} (GTC, unfillable)")
    # Optional: trade on behalf of a master wallet (agent-wallet model)
    master_addr = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip() or None
    exchange = Exchange(
        wallet=account,
        base_url=constants.MAINNET_API_URL,
        account_address=master_addr,
    )
    if master_addr:
        print(f"  Trading on behalf of master: {master_addr}")
    else:
        print(f"  No HL_ACCOUNT_ADDRESS set; orders will execute against agent's own account ({addr})")

    # Inject every live HIP-4 outcome (Yes + No coins) into the SDK's name lookup,
    # because hyperliquid-python-sdk only knows spot+perp asset ids out of the box.
    injected = 0
    for entry in om.get("outcomes", []):
        try:
            oid = int(entry["outcome"])
        except Exception:
            continue
        for side in (0, 1):
            cs = f"#{10 * oid + side}"
            aid = 100_000_000 + 10 * oid + side
            exchange.info.name_to_coin[cs] = cs
            exchange.info.coin_to_asset[cs] = aid
            injected += 1
    print(f"  Injected {injected} HIP-4 outcome coins into SDK name_to_coin / coin_to_asset")

    try:
        result = exchange.order(
            yes_coin,                       # name (positional, was `coin=` in old SDK)
            True,                           # is_buy
            TEST_SIZE,                      # sz
            TEST_LIMIT_PRICE,               # limit_px
            {"limit": {"tif": "Gtc"}},      # order_type
            False,                          # reduce_only
        )
        print(f"  Order response:\n{json.dumps(result, indent=2)}")
    except KeyError as e:
        print(f"  KEYERROR: {e}")
        print(f"  → SDK injection didn't catch this. Coin: {yes_coin}, asset_id should be {100_000_000 + 10 * o}")
        return 4
    except Exception as e:
        print(f"  EXCEPTION {type(e).__name__}: {e}")
        return 4

    # Did HL accept it? Pull oid out of the response, or bubble the typed error.
    oid = None
    embedded_errors = []
    if isinstance(result, dict):
        try:
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            for s in statuses:
                if isinstance(s, dict):
                    if "resting" in s:
                        oid = s["resting"].get("oid")
                    elif "filled" in s:
                        oid = s["filled"].get("oid")
                        print(f"  WARNING: order filled instead of resting — totalSz={s['filled'].get('totalSz')} avgPx={s['filled'].get('avgPx')}")
                    elif "error" in s:
                        embedded_errors.append(s["error"])
        except Exception:
            pass
    if embedded_errors:
        for err in embedded_errors:
            print(f"  HL REJECTED: {err}")
        return 4

    # ── 5. openOrders ──
    banner("[5/7] openOrders — verify the limit landed on the book")
    time.sleep(1.0)
    # Query master account if set (where the order actually rests), else the agent
    query_addr = master_addr or addr
    oo = post_info({"type": "openOrders", "user": query_addr})
    print(f"  openOrders for {query_addr}: {json.dumps(oo, indent=2)}")
    if oid is None and isinstance(oo, list):
        for ord_ in oo:
            if ord_.get("coin") == yes_coin and abs(float(ord_.get("limitPx", 0)) - TEST_LIMIT_PRICE) < 1e-9:
                oid = ord_.get("oid")
                break

    # ── 6. cancel ──
    banner(f"[6/7] Cancel the test order (oid={oid})")
    if oid is None:
        print("  No oid found. Skipping cancel.")
        print("  If you see a resting order in step 5, cancel it manually in the HL UI.")
        return 5
    try:
        cancel_result = exchange.cancel(yes_coin, oid)
        print(f"  Cancel response:\n{json.dumps(cancel_result, indent=2)}")
    except Exception as e:
        print(f"  CANCEL EXCEPTION {type(e).__name__}: {e}")
        return 6

    # ── 7. verify ──
    banner("[7/7] Re-read openOrders to verify cancel")
    time.sleep(1.0)
    oo2 = post_info({"type": "openOrders", "user": query_addr})
    still_there = any(
        ord_.get("oid") == oid for ord_ in (oo2 if isinstance(oo2, list) else [])
    )
    print(f"  openOrders after cancel: {json.dumps(oo2, indent=2)}")
    if still_there:
        print(f"  WARNING: order {oid} still resting after cancel call")
        return 7
    print("  Order cancelled cleanly.")

    banner("✅ SMOKE TEST PASSED — production code path works end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
