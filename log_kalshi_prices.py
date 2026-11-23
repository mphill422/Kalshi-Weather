"""
log_kalshi_prices.py — record what the Kalshi weather market is doing, every 30 min.

WHY THIS EXISTS
    obs_live records what the weather did. settlements records what was forecast.
    NOTHING has ever recorded what the MARKET did. That gap is why every
    analysis in this project dead-ends: the only prices anywhere in the system
    are in `bets`, which stops 2026-09-07 and is 97% paper trades duplicated 3x.

    The 2026-11-21 bias test showed the market going 7-for-14 on 2F brackets
    while an NWS-based forecast should hit about a third. The market forecasts
    BETTER than the public sources. So the edge, if one exists, is not
    "forecast better" -- it is "know something before the market prices it."
    Testing that requires knowing what the market charged, minute by minute.
    Hence this file.

WHAT IT IS NOT
    Not a model. Predicts nothing, takes no position, cannot be wrong.
    Read-only: it calls GET endpoints only. There is no code path that trades.

HOW IT FINDS MARKETS
    It does NOT build event tickers by string formatting -- an earlier draft
    guessed "KXHIGHNY-26NOV08" and that guess was unverified. This follows the
    pattern already proven in kalshi_settlements.py:
        /series?category=Climate and Weather   -> every KXHIGH* series
        /events?series_ticker=...              -> event_ticker + strike_date
        /markets?event_ticker=...              -> the bracket contracts
    The weather day is strike_date MINUS ONE, same as kalshi_settlements.py.

ENV (all already set in this repo)
    KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

EXIT CODES
    0  wrote rows for a healthy number of series
    1  wrote nothing, or coverage collapsed
    Deliberate. obs_live went dark for 41 days because a degraded job kept
    exiting green.
"""

import base64
import datetime as dt
import json
import os
import sys
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASE = "https://api.elections.kalshi.com"
TABLE = "kalshi_prices"
MIN_SERIES_OK = 10
_SAMPLE_PRINTED = False          # below this during US hours, something is wrong

KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "").strip()
PEM = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
SB_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SB_KEY = (os.environ.get("SUPABASE_SERVICE_KEY")
          or os.environ.get("SUPABASE_KEY") or "").strip()

# series ticker -> the city name used everywhere else in this project.
# Best effort only: unknown series still get logged, with city = stripped ticker.
CITY_BY_SERIES = {
    "KXHIGHNY": "New York",        "KXHIGHPHIL": "Philadelphia",
    "KXHIGHTDC": "Washington DC",  "KXHIGHTBOS": "Boston",
    "KXHIGHTATL": "Atlanta",       "KXHIGHMIA": "Miami",
    "KXHIGHCHI": "Chicago",        "KXHIGHTDAL": "Dallas",
    "KXHIGHTHOU": "Houston",       "KXHIGHAUS": "Austin",
    "KXHIGHTSATX": "San Antonio",  "KXHIGHTNOLA": "New Orleans",
    "KXHIGHTOKC": "Oklahoma City", "KXHIGHTMIN": "Minneapolis",
    "KXHIGHDEN": "Denver",         "KXHIGHTPHX": "Phoenix",
    "KXHIGHTLV": "Las Vegas",      "KXHIGHLAX": "Los Angeles",
    "KXHIGHTSEA": "Seattle",       "KXHIGHTSFO": "San Francisco",
}

_pk = serialization.load_pem_private_key(PEM.encode(), password=None) if PEM else None


def sign(ts, method, path):
    return base64.b64encode(_pk.sign(
        f"{ts}{method}{path}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()


def kget(path, params=None):
    """Signed GET with the same retry policy as kalshi_settlements.py."""
    for attempt in range(4):
        ts = str(int(time.time() * 1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID,
             "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path)}
        r = requests.get(BASE + path, headers=h, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        print(f"    ERR {r.status_code} {path} {r.text[:160]}")
        return None
    return None


def sb_insert(rows):
    """Insert ignoring duplicates on (market_ticker, snap_slot)."""
    if not rows:
        return 0, None
    url = f"{SB_URL}/rest/v1/{TABLE}?on_conflict=market_ticker,snap_slot"
    headers = {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}",
               "Content-Type": "application/json",
               "Prefer": "resolution=ignore-duplicates,return=minimal"}
    written = 0
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        r = requests.post(url, headers=headers, data=json.dumps(chunk), timeout=60)
        if r.status_code >= 300:
            return written, f"HTTP {r.status_code}: {r.text[:300]}"
        written += len(chunk)
    return written, None


def discover_series():
    out, cursor = [], None
    while True:
        p = {"limit": 200, "category": "Climate and Weather"}
        if cursor:
            p["cursor"] = cursor
        d = kget("/trade-api/v2/series", p)
        if not d:
            break
        out += [s["ticker"] for s in d.get("series", [])
                if s.get("ticker", "").startswith("KXHIGH")]
        cursor = d.get("cursor")
        if not cursor:
            break
    return sorted(set(out))


def live_events(series, today):
    """Events whose weather day (strike_date - 1) is today or adjacent."""
    found = []
    for status in ("open", None):
        p = {"series_ticker": series, "limit": 200}
        if status:
            p["status"] = status
        d = kget("/trade-api/v2/events", p)
        if not d:
            continue
        for ev in d.get("events", []):
            sd = (ev.get("strike_date") or "")[:10]
            if not sd:
                continue
            try:
                wday = dt.date.fromisoformat(sd) - dt.timedelta(days=1)
            except ValueError:
                continue
            if abs((wday - today).days) <= 1:
                found.append((ev.get("event_ticker", ""), wday))
        if found:
            return found           # "open" worked; don't fall through
    return found


def _num(mk, *keys):
    """First present key as a float. Kalshi returns prices as DOLLAR STRINGS
    ('0.4900') under *_dollars names; older shapes used integer cents under
    bare names. Handles both."""
    for k in keys:
        v = mk.get(k)
        if v is None or v == "":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        return f * 100.0 if k.endswith("_dollars") else f
    return None


def _cents(mk, *keys):
    f = _num(mk, *keys)
    return int(round(f)) if f is not None else None


def _int(mk, *keys):
    for k in keys:
        v = mk.get(k)
        if v is None or v == "":
            continue
        try:
            return int(float(v))
        except (TypeError, ValueError):
            continue
    return None


def floor_30(ts):
    return ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)


def main():
    now = dt.datetime.now(dt.timezone.utc)
    slot = floor_30(now)
    today = now.date()

    print("=" * 72)
    print(f"log_kalshi_prices | {now:%Y-%m-%d %H:%M:%S} UTC | slot {slot:%H:%M} | READ-ONLY")
    print("=" * 72)

    missing = [n for n, v in (("KALSHI_API_KEY_ID", KEY_ID),
                              ("KALSHI_PRIVATE_KEY", PEM),
                              ("SUPABASE_URL", SB_URL),
                              ("SUPABASE_SERVICE_KEY", SB_KEY)) if not v]
    if missing:
        print("FATAL: missing env -> " + ", ".join(missing))
        return 1

    series = discover_series()
    print(f"discovered {len(series)} KXHIGH series\n")
    if not series:
        print("FAIL: no series returned. Check the key, or the category name.")
        return 1

    rows, ok, empty = [], [], []

    for s in series:
        city = CITY_BY_SERIES.get(s, s.replace("KXHIGH", ""))
        evs = live_events(s, today)
        if not evs:
            empty.append(city)
            print(f"  {city:<15} {s:<14} no live event")
            continue

        n = 0
        for et, wday in evs:
            m = kget("/trade-api/v2/markets", {"event_ticker": et, "limit": 200})
            if not m:
                continue
            for mk in m.get("markets", []):
                global _SAMPLE_PRINTED
                if not _SAMPLE_PRINTED:
                    print("    market object keys: " + ", ".join(sorted(mk.keys())))
                    print()
                    _SAMPLE_PRINTED = True
                bid = _cents(mk, "yes_bid_dollars", "yes_bid")
                ask = _cents(mk, "yes_ask_dollars", "yes_ask")
                # A binary market's YES side is the mirror of its NO side.
                # Derive it when only the NO side is published.
                if bid is None:
                    na = _cents(mk, "no_ask_dollars", "no_ask")
                    bid = (100 - na) if na is not None else None
                if ask is None:
                    nb = _cents(mk, "no_bid_dollars", "no_bid")
                    ask = (100 - nb) if nb is not None else None
                last = _cents(mk, "last_price_dollars", "last_price")
                both = bid is not None and ask is not None
                rows.append({
                    "snap_utc": now.isoformat(),
                    "snap_slot": slot.isoformat(),
                    "local_date": wday.isoformat(),
                    "city": city,
                    "series": s,
                    "event_ticker": et,
                    "market_ticker": mk.get("ticker"),
                    "bracket": (mk.get("yes_sub_title") or mk.get("subtitle")
                                or (mk.get("ticker") or "").split("-")[-1]),
                    "status": mk.get("status"),
                    "yes_bid": bid,
                    "yes_ask": ask,
                    "mid": round((bid + ask) / 2.0, 1) if both else None,
                    "spread": (ask - bid) if both else None,
                    "last_price": last,
                    "volume": _int(mk, "volume", "volume_24h"),
                    "open_interest": _int(mk, "open_interest"),
                })
                n += 1
        ok.append(city)
        print(f"  {city:<15} {s:<14} {n:>3} brackets across {len(evs)} event(s)")
        time.sleep(0.2)

    print("-" * 72)
    written, err = sb_insert(rows)
    if err:
        print(f"SUPABASE WRITE FAILED after {written} rows: {err}")
        print("  42P01 => table missing; run kalshi_prices_table.sql first.")
        print("  42703 => column mismatch; compare the table to the row keys above.")
        return 1

    print(f"wrote {written} rows | {len(ok)} series with live markets, {len(empty)} without")

    priced = sum(1 for r in rows if r["yes_ask"] is not None)
    print(f"rows with a usable yes_ask: {priced} of {len(rows)}")
    if priced == 0 and rows:
        print("  ^^ NO PRICES CAPTURED. Read the SAMPLE MARKET OBJECT above and")
        print("     compare its key names to yes_bid / yes_ask / volume / open_interest.")

    if written == 0:
        print("FAIL: nothing written.")
        return 1
    if len(ok) < MIN_SERIES_OK:
        print(f"FAIL: only {len(ok)} series had live markets (expected >= {MIN_SERIES_OK}).")
        print("  Overnight this can be legitimate. During US daytime, investigate.")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
