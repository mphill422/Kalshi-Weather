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

⚠️ SET THE KILL LINE BEFORE THE DATA ARRIVES. The 10:30 cell (+18.45) is the
most extreme number in the set and rests on the fewest rows (n=93). If forward
results come in materially lower there, that is regression toward the mean and
was the expected outcome, not a surprise. Decide now what win rate over what
sample retires MORNING, so the threshold is not chosen after seeing the result.

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

THE LATCH — why the window is one-sided, not ±10 minutes
---------------------------------------------------------
GitHub Actions scheduled runs do not start on time. On shared runners they
start LATE — commonly 5-20 minutes, occasionally more — and never early. A
symmetric ±10 minute tolerance therefore discards the entire late half of a
distribution that is entirely late, and does so silently: the run succeeds, logs
"no window active", and nothing is recorded.

So the check is one-sided: fire if ET time is at or after the window target and
within WINDOW_LATCH_MIN of it. Negative deltas (early) never fire, which also
means the off-season DST twin cron declines cleanly on its own.

The cost of a latch is drift. A 16:00 bet placed at 16:24 is not the same bet —
the afternoon certainty curve moves fast, and price at entry is the whole
strategy. Two guards:
  1. minutes_late is stored on every row. If drift is routine, it is visible
     in the data rather than inferred, and the band analysis can be re-cut
     against actual entry time.
  2. a window that already has rows for today is skipped entirely, so a retry
     cron cannot append later-priced entries alongside on-time ones.
Guard 2 has one hole: if a window fires on time and NOTHING qualified (all
cities out of band), there are no rows, and a retry 12 minutes later will run
and may log at the later price. This is rare and self-limiting at a 25-minute
latch, but it is a known way for a few late entries to enter the sample.
minutes_late is what makes those findable.

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

FEES
----
`profit` remains GROSS, exactly as before, so every number already in the table
stays comparable. `fee_dollars` and `net_profit` are stored alongside it.

FEE_CENTS = 3.6 is backed out from a SINGLE Kalshi ticket. One observation is
not a fee schedule. Kalshi's fee is a function of price, so a flat cent figure
is an approximation that will be least accurate at the ends of the band —
exactly where the afternoon numbers live. Confirm against a second settled
ticket at a different price before treating any net figure as decided. The
afternoon band nets ~+3.7 on this assumption, which is thin enough that a
half-cent error changes the conclusion.

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

RUN THIS ONCE MORE for the new columns (safe on an existing table):

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS minutes_late INTEGER,
    ADD COLUMN IF NOT EXISTS fee_dollars  NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS net_profit   NUMERIC(10,4);

The UNIQUE (date, city, window_label) is what makes a double workflow run safe.

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

# Per-contract fee, backed out from ONE Kalshi ticket. See FEES in the docstring.
# Applied to net_profit only; `profit` stays gross.
FEE_CENTS = 3.6

# Windows in EASTERN LOCAL TIME. pytz resolves DST, so these stay correct in
# November when ET moves from UTC-4 to UTC-5.
#   (hour, minute, label, band_low_cents, band_high_cents_exclusive)
WINDOWS = [
    (10, 30, "MORNING",   58, 70),
    (12,  0, "MIDDAY",    58, 70),
    (16,  0, "AFTERNOON", 58, 80),
]

# One-sided latch, in minutes AFTER the window target. Runner queue delay is
# always late, never early, so an early delta must never fire. See THE LATCH.
WINDOW_LATCH_MIN = 25

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


def window_already_logged(date_str, label):
    """True if this (date, window) already has rows.

    Fails OPEN on any error: if we cannot tell, we proceed and let the UNIQUE
    constraint do the work. That is the safer default — a missed window loses a
    day of sample, a duplicate insert is simply ignored by the constraint.
    """
    try:
        r = requests.get(
            sb_url("favorites_bets"),
            headers=sb_headers(),
            params={"date": f"eq.{date_str}", "window_label": f"eq.{label}",
                    "select": "id", "limit": "1"},
            timeout=15)
        if r.status_code == 200:
            return len(r.json()) > 0
    except Exception:
        pass
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


def fee_for(amount, price_cents):
    """Estimated round-trip fee in dollars on a stake at a given ask.

    contracts = stake / (price in dollars). Flat per-contract approximation —
    see FEES. Returns 0.0 on a nonsense price rather than raising.
    """
    try:
        p = float(price_cents)
        if p <= 0:
            return 0.0
        contracts = float(amount) * 100.0 / p
        return round(contracts * (FEE_CENTS / 100.0), 4)
    except Exception:
        return 0.0


# ── Logging pass ─────────────────────────────────────────────────────────────
def run_window(label, band_lo, band_hi, now_et, minutes_late):
    today = now_et.strftime("%Y-%m-%d")
    tag = f"{TAG_PREFIX}_{label}"
    print(f"\n=== {label} window | band {band_lo}-{band_hi - 1}c | {today} "
          f"| {minutes_late} min after target ===")
    if minutes_late >= 10:
        print(f"  ⚠️ entry drift: {minutes_late} min late. Price at entry is the "
              f"strategy; check minutes_late across the sample.")

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
            "minutes_late": minutes_late,
            "fee_dollars": fee_for(STAKE, ask),
        }
        if insert_bet(row):
            logged.append(f"{city} {bracket} @ {ask}c")
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  ✅ LOGGED  (Σp {sigma_p:.2f})")
        else:
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  insert failed")

        time.sleep(0.25)

    print(f"\n  logged {len(logged)} | out of band {skipped_band} | no ladder {skipped_nomarket}")
    if not logged:
        print("  (no rows written — a retry cron may fire this window again; "
              "any such entry will carry a larger minutes_late)")
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
    gross = net = 0.0
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

            fee = b.get("fee_dollars")
            if fee is None:
                fee = fee_for(amount, price)
            fee = float(fee)
            net_profit = round(profit - fee, 4)

            gross += profit
            net += net_profit

            update_bet(b["id"], {
                "result": "Won" if res == "yes" else "Lost",
                "profit": profit,
                "fee_dollars": fee,
                "net_profit": net_profit,
                "settled_at": dt.datetime.now(ET).isoformat(),
            })
        time.sleep(0.25)

    n = won + lost
    if n:
        print(f"  settled {n}: {won} won, {lost} lost ({100.0*won/n:.1f}%)")
        print(f"  gross ${gross:+.2f} | net ${net:+.2f} "
              f"(fee est. {FEE_CENTS}c/contract, one-ticket basis)")
    else:
        print("  no results available yet")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_et = dt.datetime.now(ET)
    print(f"FAV V1 | {now_et:%Y-%m-%d %H:%M} ET | {len(SERIES)} cities")
    print("no forecast — buying the market's own favorite, in band")
    print(f"latch: fires 0 to +{WINDOW_LATCH_MIN} min after target, never early\n")

    today = now_et.strftime("%Y-%m-%d")
    fired = False

    for hh, mm, label, lo, hi in WINDOWS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = (now_et - target).total_seconds() / 60.0

        # One-sided: early never fires. This is also what makes the off-season
        # DST twin cron decline cleanly.
        if not (0 <= delta_min <= WINDOW_LATCH_MIN):
            continue

        if window_already_logged(today, label):
            print(f"({label} already has rows for {today} — skipping, "
                  f"this run is a retry that arrived after a successful one)")
            fired = True
            continue

        run_window(label, lo, hi, now_et, int(round(delta_min)))
        fired = True

    if not fired:
        def mins_until(w):
            t = now_et.replace(hour=w[0], minute=w[1], second=0, microsecond=0)
            d = (t - now_et).total_seconds() / 60.0
            return d if d >= 0 else d + 1440.0
        nxt = min(WINDOWS, key=mins_until)
        print(f"(no window active — next is {nxt[2]} at "
              f"{nxt[0]:02d}:{nxt[1]:02d} ET, in {mins_until(nxt):.0f} min; "
              f"latch is 0 to +{WINDOW_LATCH_MIN} min after target)")

    settle()

    print("\nPer-tag results (never pool tags):")
    print("  select strategy_tag, count(*) n,")
    print("         sum((result='Won')::int) wins,")
    print("         round(100.0*avg((result='Won')::int),1) win_pct,")
    print("         round(sum(profit),2) gross_profit,")
    print("         round(sum(net_profit),2) net_profit,")
    print("         round(avg(yes_ask_cents),1) avg_ask,")
    print("         round(avg(minutes_late),1) avg_min_late")
    print("  from favorites_bets where result<>'Pending'")
    print("  group by strategy_tag order by strategy_tag;")


if __name__ == "__main__":
    main()
