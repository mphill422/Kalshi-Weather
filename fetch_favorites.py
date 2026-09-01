"""
fetch_favorites.py — FAV V1. Buy the market's own favorite, in band, at set times.

WHAT THIS IS, AND WHY IT IS NOT THE WEATHER MODEL
==================================================
This script does NO forecasting. It never fetches NWS, GFS, or NBM. It does not
compute a consensus, a sigma, a bias correction, or a trust score. It has no
opinion about tomorrow's temperature.

It does exactly this:
    1. pull the Kalshi ladder
    2. take the highest-priced bracket (the market's own favorite)
    3. if the ask is inside the band, log a bet
    4. next day, settle against Kalshi's own result

That is the entire strategy.

THE REASON THIS EXISTS
----------------------
The weather model (V5.31.x, ~2,000 lines) forecasts a temperature and finds the
bracket that best matches it. Measured across 72 days and 20 cities, when that
model DISAGREED with the market, the market was right 762 times to 171 — 81.7%.
Every line of forecasting machinery in that file exists to produce a
disagreement with the market, and disagreement is the losing side of this trade.

Meanwhile the manual research found something that requires no forecast at all:
the market's favorite, bought in a specific price band at specific hours, has
won well above its implied probability. From kalshi_candles, 25 days
(2026-08-02 to 08-26), 19 US cities, top bracket by ASK:

    10:30 ET, 58-69c   n=93    81.7% win at 63.3c   =  +18.45c / contract
    12:00 ET, 58-69c   n=~90   ~73%   at ~63c       =  +10 to +12c
    16:00 ET, 58-79c   n=260   81.5% at 74.2c       =   +7.33c  (70-79 slice)

These two things are not the same strategy at different levels of polish. They
are opposite bets. This file implements the one that measured positive.

⚠️ THIS HAS NOT BEEN VALIDATED FORWARD. Everything above is backtest on 25
summer days. Backtests on a fixed window are exactly how the last four models
looked good before losing money. Run it on paper, per-tag, and judge it on
settled rows.

BAND DEFINITIONS — where the numbers came from
-----------------------------------------------
FLOOR = 58, not 60, and not 55. Measured 10am-1pm ET, ask price:
    55-57c   n=246   53.7% win at 56.0c   =  -2.38
    58-59c   n=174   64.9% win at 58.5c   =  +6.44
    60-64c   n=392   68.6% win at 61.9c   =  +6.69
58-59 behaves like 60-64; 55-57 is a different population and loses. Confirmed
independently at 16:00 ET (55-57 = -2.09, everything >=58 positive), so the 58
line holds at two different times of day. Per-city at 10:30, the 58-69 band beat
55-57 in 12 of 13 cities that had both.

CEILING = 69 in the morning, 79 in the afternoon. Checked 10:00-12:30 ET:
    58-69   n=597   72.9%   +10.16
    70      n=31    71.0%    +0.97
    71-72   n=45    80.0%    +8.62
    73-79   n=117   73.5%    -2.20
    80+     n=57    84.2%    -2.60
71-72 looks good but sits between +0.97 and -2.20 on cells of 31-45 — that is
noise, not a ceiling worth extending. But at 16:00 the 70-79 band is the BEST
band (+7.33, n=260), because by then the high has largely happened and a 74c
favorite is nearly resolved. Hence the wider afternoon band.

NEVER use an open-ended floor. "58 and over" drags in the 80+ tier, which wins
85-96% and pays nothing at 87-94c. Open-ended floors were negative at 6-7 of 11
hours tested. The ceiling does as much work as the floor.

TIMES — no earlier pass is worth adding
----------------------------------------
58-69 band by half hour (n=89-101 each, so comparable):
    08:00 +1.54   08:30 +0.92   09:00 +0.73   09:30 +7.61
    10:00 +4.73   10:30 +18.45  11:00 +13.77  11:30 +11.04
The count of qualifying picks is FLAT across all of these (~95), so it is not
that fewer cities qualify early — the same picks are available at 8am and are
simply wrong more often. The information that makes the band work arrives
between 09:00 and 10:30. Do not add an earlier window.

The 10:30 spike is partly luck; 10:30-11:30 averages ~+14 and these cells are
correlated (same city-days 30 min apart). Treat ~+14 as the morning estimate.

⚠️ DST. Windows are defined in EASTERN LOCAL TIME and resolved through pytz, not
hardcoded as UTC. All the research above was done in EDT (UTC-4). From early
November the same ET times are UTC-5. Hardcoding UTC would silently shift every
window by an hour halfway through the season.

⚠️ SEASONALITY — the biggest open risk. Every number above is from 25 days in
August. Summer highs are solar-driven and boringly predictable, which is
probably WHY the afternoon certainty curve exists. Winter highs are driven by
frontal timing. The bands, the hours, and the per-city ordering may all move.
Re-run the band analysis each season rather than treating 58-79 as permanent.

CITIES
------
No forecast means no per-city calibration, which means no reason to restrict the
roster to cities we have weather data for. Seattle and San Francisco — dropped
from the weather model in V5.30 for forecast quality — are fine here, because
this strategy never forecasts them.

EXCLUDED: San Diego (KXHIGHTSAN). Structurally odd in the candle data: 11.8% win
at 90.4c average, which is not a real market's behavior. Never explained, so it
stays out. Trenton and Newark had ~150 candle rows total across 25 days — too
thin to have been tested at all. International series (Paris, Geneva, Tokyo,
Seoul, Hong Kong, Mumbai, Singapore, Sydney, Sao Paulo, Dubai, Beijing,
Shanghai) exist in Kalshi's settlement data but never appeared in the candle
backfill and are not visible in the app — no measured basis, so excluded.

SETTLEMENT
----------
Settles against KALSHI'S OWN `result` field, not Iowa CLI. This is deliberate:
we are scoring a contract, not a temperature, so any bracket-boundary or
rounding disagreement between our arithmetic and Kalshi's settlement is removed
entirely. The weather model's CLI-based settlement stays as it is; that file
answers a different question.

CREATE THE TABLE ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.favorites_bets (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    city TEXT NOT NULL,
    series TEXT,
    event_ticker TEXT,
    market_ticker TEXT,
    bracket TEXT,
    window_label TEXT NOT NULL,
    yes_ask_cents INTEGER,
    sigma_p NUMERIC,
    n_brackets INTEGER,
    strategy_tag TEXT NOT NULL,
    amount NUMERIC NOT NULL DEFAULT 5.0,
    result TEXT NOT NULL DEFAULT 'Pending',
    profit NUMERIC,
    settled_at TIMESTAMPTZ,
    placed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, city, window_label)
  );
  ALTER TABLE public.favorites_bets ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.favorites_bets
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_fav_date ON public.favorites_bets (date);
  CREATE INDEX IF NOT EXISTS idx_fav_tag ON public.favorites_bets (strategy_tag);

The UNIQUE (date, city, window_label) is what makes a double workflow run safe.

⚠️ FEES ARE NOT MODELLED. Every P&L number in this file and in the table is
gross. Kalshi charges per contract and it is not trivial at these prices. Before
any of this becomes a real-money decision, compute the fee on a 5-contract
$5 ticket at 63c and subtract it from the +14 morning estimate. That calculation
has not been done.

Secrets: SUPABASE_URL, and SUPABASE_SERVICE_KEY or SUPABASE_KEY.
No Kalshi credentials needed — the markets endpoint is public.
"""

import os
import re
import time
import requests
import datetime as dt

import pytz

SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
HEADERS = {"User-Agent": "kalshi-favorites/1.0", "Accept": "application/json"}

ET = pytz.timezone("America/New_York")

STAKE = 5.0
TAG_PREFIX = "FAV_V1"

# Windows in EASTERN LOCAL TIME. pytz resolves DST, so these stay correct in
# November when ET moves from UTC-4 to UTC-5.
#   (hour, minute, label, band_low_cents, band_high_cents_exclusive)
WINDOWS = [
    (10, 30, "MORNING",   58, 70),
    (12,  0, "MIDDAY",    58, 70),
    (16,  0, "AFTERNOON", 58, 80),
]
WINDOW_TOLERANCE_MIN = 10

SERIES = {
    "New York":       "KXHIGHNY",
    "Miami":          "KXHIGHMIA",
    "Atlanta":        "KXHIGHTATL",
    "Philadelphia":   "KXHIGHPHIL",
    "Washington DC":  "KXHIGHTDC",
    "Boston":         "KXHIGHTBOS",
    "Chicago":        "KXHIGHCHI",
    "Austin":         "KXHIGHAUS",
    "Dallas":         "KXHIGHTDAL",
    "Houston":        "KXHIGHTHOU",
    "Oklahoma City":  "KXHIGHTOKC",
    "Minneapolis":    "KXHIGHTMIN",
    "San Antonio":    "KXHIGHTSATX",
    "New Orleans":    "KXHIGHTNOLA",
    "Los Angeles":    "KXHIGHLAX",
    "Phoenix":        "KXHIGHTPHX",
    "Seattle":        "KXHIGHTSEA",
    "San Francisco":  "KXHIGHTSFO",
    "Denver":         "KXHIGHDEN",
    "Las Vegas":      "KXHIGHTLV",
}


# ── Supabase ─────────────────────────────────────────────────────────────────
def sb_headers(prefer="return=representation"):
    return {
        "apikey": SB_KEY,
        "Authorization": "Bearer " + SB_KEY,
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def sb_url(table):
    return f"{SB_URL}/rest/v1/{table}"


def insert_bet(row):
    """UNIQUE (date, city, window_label) makes a repeat run a no-op."""
    try:
        r = requests.post(
            sb_url("favorites_bets") + "?on_conflict=date,city,window_label",
            headers=sb_headers("return=minimal,resolution=ignore-duplicates"),
            json=row, timeout=15)
        return r.status_code in (200, 201, 204)
    except Exception as e:
        print(f"    insert failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def fetch_pending():
    try:
        r = requests.get(
            sb_url("favorites_bets"),
            headers=sb_headers(),
            params={"result": "eq.Pending", "order": "date.asc", "limit": "500"},
            timeout=20)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def update_bet(bet_id, updates):
    try:
        r = requests.patch(
            sb_url("favorites_bets") + "?id=eq." + str(bet_id),
            headers=sb_headers(), json=updates, timeout=15)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── Kalshi ───────────────────────────────────────────────────────────────────
def event_ticker_for(series, when_et):
    return series + "-" + when_et.strftime("%y%b%d").upper()


def kalshi_markets(params):
    for attempt in (1, 2, 3):
        try:
            r = requests.get(KALSHI, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json().get("markets", [])
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * attempt)
                continue
            return []
        except Exception:
            if attempt == 3:
                return []
            time.sleep(1.5 * attempt)
    return []


def label_of(m):
    for f in ("yes_sub_title", "subtitle", "title"):
        s = (m.get(f) or "").replace("\u00b0", "").strip()
        if s:
            return s
    return ""


def ask_cents(m):
    """Ask price in cents. ASK, not mid, not bid — you pay the ask.

    An earlier analysis of this same data used mid and produced 55-59 numbers
    that did not reproduce; the band a contract falls into changes depending on
    which price you classify by, so this is not a 1-2c adjustment.
    """
    v = m.get("yes_ask_dollars")
    if v:
        try:
            return int(round(float(v) * 100))
        except Exception:
            pass
    v = m.get("yes_ask")
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    return None


def top_bracket(markets):
    """The market's favorite: highest ask on the ladder.

    Returns (market, ask_cents, sigma_p, n_priced) or None.
    sigma_p is the sum of implied probabilities across the ladder — logged as
    context, NOT used as a filter here. The weather model gates on Sigma-p >
    1.15; whether that helps this strategy is untested, so it is measured
    rather than assumed.
    """
    priced = [(m, ask_cents(m)) for m in markets]
    priced = [(m, a) for m, a in priced if a is not None and 0 < a < 100]
    if len(priced) < 2:
        return None
    sigma_p = round(sum(a for _, a in priced) / 100.0, 4)
    m, a = max(priced, key=lambda x: x[1])
    return m, a, sigma_p, len(priced)


# ── Logging pass ─────────────────────────────────────────────────────────────
def run_window(label, band_lo, band_hi, now_et):
    today = now_et.strftime("%Y-%m-%d")
    tag = f"{TAG_PREFIX}_{label}"
    print(f"\n=== {label} window | band {band_lo}-{band_hi - 1}c | {today} ===")

    logged, skipped_band, skipped_nomarket = [], 0, 0

    for city, series in SERIES.items():
        et_ticker = event_ticker_for(series, now_et)
        markets = kalshi_markets({"event_ticker": et_ticker, "limit": 40})
        if not markets:
            markets = kalshi_markets(
                {"series_ticker": series, "status": "open", "limit": 40})
            markets = [m for m in markets
                       if et_ticker.upper() in (m.get("event_ticker") or "").upper()]

        top = top_bracket(markets)
        if top is None:
            skipped_nomarket += 1
            print(f"  {city:<15} no ladder")
            continue

        m, ask, sigma_p, n_priced = top
        bracket = label_of(m)

        if not (band_lo <= ask < band_hi):
            skipped_band += 1
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  — out of band")
            continue

        row = {
            "date": today,
            "city": city,
            "series": series,
            "event_ticker": m.get("event_ticker"),
            "market_ticker": m.get("ticker"),
            "bracket": bracket,
            "window_label": label,
            "yes_ask_cents": ask,
            "sigma_p": sigma_p,
            "n_brackets": n_priced,
            "strategy_tag": tag,
            "amount": STAKE,
            "result": "Pending",
            "placed_at": now_et.isoformat(),
        }
        if insert_bet(row):
            logged.append(f"{city} {bracket} @ {ask}c")
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  ✅ LOGGED  (Σp {sigma_p:.2f})")
        else:
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  insert failed")

        time.sleep(0.25)

    print(f"\n  logged {len(logged)} | out of band {skipped_band} | no ladder {skipped_nomarket}")
    return logged


# ── Settlement pass ──────────────────────────────────────────────────────────
def settle():
    """Score against Kalshi's own result field, not a temperature we computed."""
    print("\n=== Settlement ===")
    pending = fetch_pending()
    if not pending:
        print("  nothing pending")
        return

    today = dt.datetime.now(ET).strftime("%Y-%m-%d")
    by_event = {}
    for b in pending:
        if b.get("date", "") >= today:
            continue
        by_event.setdefault(b.get("event_ticker"), []).append(b)

    if not by_event:
        print(f"  {len(pending)} pending, none from a prior day yet")
        return

    won = lost = 0
    for et_ticker, bets in by_event.items():
        if not et_ticker:
            continue
        markets = kalshi_markets({"event_ticker": et_ticker, "limit": 40})
        results = {m.get("ticker"): m.get("result") for m in markets}
        for b in bets:
            res = results.get(b.get("market_ticker"))
            if res not in ("yes", "no"):
                continue
            price = float(b.get("yes_ask_cents") or 0)
            amount = float(b.get("amount") or STAKE)
            if res == "yes" and price > 0:
                profit = round(amount * (100.0 - price) / price, 2)
                won += 1
            else:
                profit = round(-amount, 2)
                lost += 1
            update_bet(b["id"], {
                "result": "Won" if res == "yes" else "Lost",
                "profit": profit,
                "settled_at": dt.datetime.now(ET).isoformat(),
            })
        time.sleep(0.25)

    n = won + lost
    if n:
        print(f"  settled {n}: {won} won, {lost} lost ({100.0*won/n:.1f}%)")
    else:
        print("  no results available yet")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_et = dt.datetime.now(ET)
    print(f"FAV V1 | {now_et:%Y-%m-%d %H:%M} ET | {len(SERIES)} cities")
    print("no forecast — buying the market's own favorite, in band\n")

    fired = False
    for hh, mm, label, lo, hi in WINDOWS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = abs((now_et - target).total_seconds()) / 60.0
        if delta_min <= WINDOW_TOLERANCE_MIN:
            run_window(label, lo, hi, now_et)
            fired = True

    if not fired:
        nearest = min(
            WINDOWS,
            key=lambda w: abs((now_et - now_et.replace(
                hour=w[0], minute=w[1], second=0, microsecond=0)).total_seconds()))
        print(f"(no window active — nearest is {nearest[2]} "
              f"at {nearest[0]:02d}:{nearest[1]:02d} ET, "
              f"tolerance ±{WINDOW_TOLERANCE_MIN} min)")

    settle()

    print("\nPer-tag results (never pool tags):")
    print("  select strategy_tag, count(*) n,")
    print("         sum((result='Won')::int) wins,")
    print("         round(100.0*avg((result='Won')::int),1) win_pct,")
    print("         round(sum(profit),2) total_profit,")
    print("         round(avg(yes_ask_cents),1) avg_ask")
    print("  from favorites_bets where result<>'Pending'")
    print("  group by strategy_tag order by strategy_tag;")


if __name__ == "__main__":
    main()
