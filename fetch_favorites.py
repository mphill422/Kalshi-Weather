"""
fetch_favorites.py — FAV V1.2. Buy the market's own favorite, in band, at set times.

WHAT THIS IS
============
No forecasting. Never fetches NWS, GFS, or NBM. No consensus, no sigma, no
trust score, no opinion about tomorrow's temperature.

    1. pull the Kalshi ladder
    2. take the highest-priced bracket (the market's own favorite)
    3. if the ask is inside the band, log it
    4. next day, settle against Kalshi's own result

That is the entire strategy. It is the automated version of the Daily Capture
Grid — same three times, same bands, same pick.

V1.2 CHANGES (2026-09-08)
=========================
REMOVED: the consensus agreement tag added in V1.1.

    It read settlements.consensus and stored whether the weather model's
    temperature fell inside the market's bracket. Backtest said AGREE won 80%
    vs 76% overall, and turned midday and afternoon from negative to positive.

    First full day of live data: 4 AGREE, 20 DISAGREE. Inspecting the
    afternoon disagreements showed what it was actually measuring:

        Austin       100-101   consensus 99.20   short 0.80
        Denver        86-87    consensus 85.20   short 0.80
        San Antonio   99-100   consensus 98.50   short 0.50
        New York      80-81    consensus 81.50   over  0.50
        Minneapolis   83-84    consensus 81.60   short 1.40
        Las Vegas    100-101   consensus 97.50   short 2.50

    Five of six within 1.4F, four within 0.8F, and FIVE OF SIX with consensus
    running LOW. That is not the model disagreeing with the market. That is the
    model's documented cold bias landing a fraction under a bracket edge, and
    strict containment calling it conflict.

    The tag was measuring bias, not agreement. Removed rather than retuned —
    the backtest was run at strict containment, so changing the tolerance would
    have invalidated the comparison anyway.

    The consensus_f and agrees_with_consensus columns are left in the table
    with the 09-07/09-08 rows in them. Nothing reads them.

ADDED: the runner-up bracket and its price.

    Not a filter and not an improvement — it makes the one directional finding
    in the data testable. Measured on 97 settled bets:

        losses where settlement came in ABOVE the bracket : 18
        losses where settlement came in BELOW the bracket :  7
        average miss                                      : 1.39F

    72/28 split, and the misses are small — most losses are ONE BRACKET HIGH,
    not a blown call. This matches the market's favorite running -0.44F vs
    actual across 72 days in the older snapshot data.

    But with only the favorite stored, there is no way to tell whether the
    settlement landed on the ADJACENT bracket or somewhere else. Storing
    rank 2 answers that directly: if losses consistently settle in the
    runner-up bracket AND the runner-up is consistently the one above, that is
    a systematic mispricing. If they scatter, it is noise.

    Also stored: bracket_lo / bracket_hi parsed at write time, so miss analysis
    is a subtraction instead of a regex over label strings, and gap_cents (the
    favorite's lead over rank 2) which has never been tested forward.

    ⚠️ NONE OF THIS CHANGES A BET. Every in-band favorite is still taken,
    exactly as before. These are columns, not logic.

WHERE THE STRATEGY ACTUALLY STANDS (97 settled, 4 days)
=======================================================
        window      n    win%    break-even    net
        AFTERNOON  30    73.3      72.5      +$2.83
        MIDDAY     40    67.5      66.5      +$3.38
        MORNING    27    63.0      65.8      -$7.09
        TOTAL      97    68.0      68.2      -$0.89

Gross +$26.32, net -$0.89. Sixty-eight percent of bets win and the fee takes
all of it. Afternoon and midday clear break-even by under a point; morning is
below it.

Day by day: +0.87, +21.65, -13.75, -9.64. The entire four-day P&L is one day.
Three of four days finished under the 66% break-even.

⚠️ THE FEE IS THE BINDING CONSTRAINT, NOT THE STRATEGY. FEE_CENTS = 3.6 is
backed out from ONE Kalshi ticket. At a 0.2-point margin the difference between
3.0c and 4.2c decides whether this works at all. Nothing in this file can fix
that — CONFIRM THE FEE against a second settled ticket at a different price
before drawing any conclusion from the numbers above.

⚠️ KILL LINE, written before the data arrived: below 66% win rate at n=50 per
window, that window retires. Morning is at 63.0% on n=27 — the closest to it
and the furthest from a verdict.

BAND DEFINITIONS
================
FLOOR = 58, not 60, not 55. Measured 10am-1pm ET:
    55-57c   n=246   53.7%   -2.38
    58-59c   n=174   64.9%   +6.44
    60-64c   n=392   68.6%   +6.69
58-59 behaves like 60-64; 55-57 is a different population and loses. Confirmed
independently at 16:00 ET. Forward data agrees there is no ordering within the
band — the price tiers scatter with no monotonic trend, which is what you get
if price inside the band carries no information.

The floor survived two further tests. City-days where the favorite NEVER
cleared 58c all day (Seattle 59.4% of days, SF 46.9%, Denver 42.9%) scored
45.3% at 47.8c ask, -6.11 net. Persistent cheapness is correctly priced.

CEILING = 69 morning and midday, 79 afternoon. 71-72 looked good at +8.62 but
sits between +0.97 and -2.20 on cells of 31-45 — noise, not a ceiling worth
extending. The 70-79 extension was tested across all 19 cities and CLOSED:
n=90, ~44% win against a ~75.5% break-even, roughly -27c/contract.

NEVER an open-ended floor. "58 and over" drags in the 80+ tier, which wins
85-96% and pays nothing at 87-94c.

NO BETS ARE STRUCTURALLY DEAD. Every cell with n>100 is negative across all
ranks and price bands. At ranks 4-12 in the 80+ band the break-even required
exceeds 100% after fees — arithmetic impossibility, not a bad bet.

TIMES
=====
58-69 band by half hour (n=89-101 each):
    08:00 +1.54   08:30 +0.92   09:00 +0.73   09:30 +7.61
    10:00 +4.73   10:30 +18.45  11:00 +13.77  11:30 +11.04
The count of qualifying picks is FLAT across all of these — the same picks are
available at 8am and are simply wrong more often. Do not add an earlier window.

⚠️ Those figures are from the exact-minute filter and are STALE. Corrected with
ladder reconstruction (626 city-days): MORNING 58-69 ran 77.4% / +10.87 in
August and 74.3% / +8.14 in September. Read the corrected numbers.

⚠️ A 17:00 window was considered and not built. By 4pm many cities are already
above 79c; by 5pm more are. It would likely produce FEWER qualifying bets, not
more, and the +10.41 figure for that hour carries the same ladder bug.

⚠️ DST. Windows are EASTERN LOCAL TIME through pytz. From early November the
same ET times are UTC-5. Hardcoding UTC would silently shift every window.

⚠️ SEASONALITY. Every band figure is from summer days. Summer highs are
solar-driven and predictable; winter highs are driven by frontal timing. Re-run
the band analysis each season.

THE LATCH
=========
Scheduled runs start LATE and never early, so a symmetric tolerance discards
the entire late half of a distribution that is entirely late — silently.

⚠️ THE REAL CAUSE WAS WORSE THAN QUEUE DELAY. GitHub delayed this repo's
scheduled runs by ~3 HOURS on 2026-09-03. No latch width fixes that. Scheduling
runs through cron-job.org -> workflow_dispatch, America/New_York. favorites.yml
has `workflow_dispatch` ONLY. Do not re-add `on: schedule:`.

The latch stays because it is correct for ordinary delay and makes a stray
manual dispatch a safe no-op. minutes_late is stored on every row — it has read
0 on all four days so far. A window that already has rows for today is skipped.

CITIES
======
No forecast means no per-city calibration. Seattle and San Francisco are fine
here despite being dropped from the weather model — this never forecasts them.

EXCLUDED: San Diego (KXHIGHTSAN, 11.8% win at 90.4c — not a real market's
behavior), Louisville (KXHIGHTSDF, 98c average with a 14.3% win rate on 7
city-days), Trenton and Newark (~150 candle rows total).

⚠️ NO PER-CITY FILTER IS SUPPORTED. City selection has failed four separate
tests now: the tier-3 scan, lock-in timing, persistent uncertainty, and the
per-city breakdown of this table. The band is the edge.

SETTLEMENT
==========
Against KALSHI'S OWN `result` field, not Iowa CLI. We are scoring a contract,
not a temperature, so any bracket-boundary disagreement between our arithmetic
and Kalshi's settlement is removed entirely.

CREATE THE TABLE ONCE:

  CREATE TABLE IF NOT EXISTS public.favorites_bets (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL, city TEXT NOT NULL, series TEXT,
    event_ticker TEXT, market_ticker TEXT, bracket TEXT,
    window_label TEXT NOT NULL, yes_ask_cents INTEGER,
    sigma_p NUMERIC, n_brackets INTEGER, strategy_tag TEXT NOT NULL,
    amount NUMERIC NOT NULL DEFAULT 5.0,
    result TEXT NOT NULL DEFAULT 'Pending',
    profit NUMERIC, settled_at TIMESTAMPTZ,
    placed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, city, window_label)
  );
  ALTER TABLE public.favorites_bets ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.favorites_bets
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_fav_date ON public.favorites_bets (date);
  CREATE INDEX IF NOT EXISTS idx_fav_tag ON public.favorites_bets (strategy_tag);

RUN THIS BEFORE DEPLOYING V1.2 (safe to re-run):

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS minutes_late   INTEGER,
    ADD COLUMN IF NOT EXISTS fee_dollars    NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS net_profit     NUMERIC(10,4),
    ADD COLUMN IF NOT EXISTS bracket_lo     INTEGER,
    ADD COLUMN IF NOT EXISTS bracket_hi     INTEGER,
    ADD COLUMN IF NOT EXISTS runner_bracket TEXT,
    ADD COLUMN IF NOT EXISTS runner_ask     INTEGER,
    ADD COLUMN IF NOT EXISTS runner_lo      INTEGER,
    ADD COLUMN IF NOT EXISTS runner_hi      INTEGER,
    ADD COLUMN IF NOT EXISTS gap_cents      INTEGER;

THE QUERY V1.2 EXISTS FOR (run at ~60 settled losses):

  -- when the favorite loses, does the settlement land on the runner-up?
  select f.window_label,
         case when f.runner_lo > f.bracket_lo then 'runner is ABOVE'
              when f.runner_lo < f.bracket_lo then 'runner is BELOW'
              else 'same/unparsed' end as runner_side,
         count(*) n,
         count(*) filter (where s.actual::numeric between f.runner_lo and f.runner_hi)
           as settled_in_runner,
         round(avg(f.gap_cents),1) avg_gap
  from favorites_bets f
  join settlements s on s.city = f.city and s.date = f.date::text
  where f.result = 'Lost' and s.actual is not null and f.runner_lo is not null
  group by f.window_label, runner_side
  order by f.window_label, runner_side;

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
HEADERS = {"User-Agent": "kalshi-favorites/1.2", "Accept": "application/json"}

ET = pytz.timezone("America/New_York")

STAKE = 5.0
TAG_PREFIX = "FAV_V1"

# Per-contract fee, backed out from ONE Kalshi ticket. See the FEE warning in
# the docstring — at the current margin this number decides everything.
# `profit` stays gross; this applies to net_profit only.
FEE_CENTS = 3.6

# Windows in EASTERN LOCAL TIME. pytz resolves DST.
#   (hour, minute, label, band_low_cents, band_high_cents_exclusive)
WINDOWS = [
    (10, 30, "MORNING",   58, 70),
    (12,  0, "MIDDAY",    58, 70),
    (16,  0, "AFTERNOON", 58, 80),
]

# One-sided latch, minutes AFTER the target. Delay is always late, never early.
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

    Fails OPEN on error: a missed window loses a day of sample, a duplicate
    insert is ignored by the constraint. The safer default is to proceed.
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

    An earlier analysis used mid and produced 55-59 numbers that did not
    reproduce; the band a contract falls into changes with which price you
    classify by, so this is not a 1-2c adjustment.
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


def bracket_bounds(label):
    """Parse a Kalshi bracket label into (lo, hi). None means unbounded.

        "94 to 95" / "94-95"  -> (94, 95)
        "97 or below"         -> (None, 97)
        "106 or above"        -> (106, None)
        unparseable           -> (None, None)

    Stored at write time so miss analysis is a subtraction rather than a regex
    over label strings months later.

    ⚠️ Despite the range naming, Kalshi brackets settle EXCLUSIVE — verified by
    counting yes-per-event, which came back exactly 1 of 6 on every event
    checked. A temperature belongs to exactly one bracket.
    """
    if not label:
        return None, None
    s = label.replace("\u00b0", "").replace("deg", "").strip()
    low = s.lower()
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return None, None
    if "below" in low or "under" in low:
        return None, nums[0]
    if "above" in low or "over" in low:
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def top_two(markets):
    """The market's favorite and its runner-up.

    Returns (fav_market, fav_ask, runner_market, runner_ask, sigma_p, n_priced)
    or None. runner_* may be None if the ladder has only one priced bracket
    (it needs 2 to qualify at all, so in practice they are populated).

    sigma_p is the sum of implied probabilities across the ladder — logged as
    context, NOT used as a filter. The weather model gated on sigma_p > 1.15;
    whether that helps THIS strategy is untested. Forward data so far shows no
    ordering by sigma tier.
    """
    priced = [(m, ask_cents(m)) for m in markets]
    priced = [(m, a) for m, a in priced if a is not None and 0 < a < 100]
    if len(priced) < 2:
        return None
    sigma_p = round(sum(a for _, a in priced) / 100.0, 4)
    priced.sort(key=lambda x: x[1], reverse=True)
    (m1, a1), (m2, a2) = priced[0], priced[1]
    return m1, a1, m2, a2, sigma_p, len(priced)


def fee_for(amount, price_cents):
    """Estimated round-trip fee in dollars on a stake at a given ask.

    contracts = stake / (price in dollars). Flat per-contract approximation.
    ⚠️ One observed ticket. At the current margin this is the number that
    decides whether the strategy works. Confirm it.
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
    print(f"\n=== {label} | band {band_lo}-{band_hi - 1}c | {today} "
          f"| {minutes_late} min after target ===")
    if minutes_late >= 10:
        print(f"  ⚠️ entry drift {minutes_late} min. Price at entry is the "
              f"strategy — check minutes_late across the sample.")

    logged, skipped_band, skipped_nomarket = [], 0, 0

    for city, series in SERIES.items():
        et_ticker = event_ticker_for(series, now_et)
        markets = kalshi_markets({"event_ticker": et_ticker, "limit": 40})
        if not markets:
            markets = kalshi_markets(
                {"series_ticker": series, "status": "open", "limit": 40})
            markets = [m for m in markets
                       if et_ticker.upper() in (m.get("event_ticker") or "").upper()]

        top = top_two(markets)
        if top is None:
            skipped_nomarket += 1
            print(f"  {city:<15} no ladder")
            continue

        m1, ask, m2, runner_ask, sigma_p, n_priced = top
        bracket = label_of(m1)
        runner_bracket = label_of(m2)

        if not (band_lo <= ask < band_hi):
            skipped_band += 1
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  — out of band")
            continue

        lo, hi = bracket_bounds(bracket)
        r_lo, r_hi = bracket_bounds(runner_bracket)

        # which side is the runner-up on? this is the V1.2 question.
        side = ""
        if lo is not None and r_lo is not None:
            side = " ↑" if r_lo > lo else " ↓" if r_lo < lo else ""

        row = {
            "date": today,
            "city": city,
            "series": series,
            "event_ticker": m1.get("event_ticker"),
            "market_ticker": m1.get("ticker"),
            "bracket": bracket,
            "bracket_lo": lo,
            "bracket_hi": hi,
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
            "runner_bracket": runner_bracket,
            "runner_ask": runner_ask,
            "runner_lo": r_lo,
            "runner_hi": r_hi,
            "gap_cents": ask - runner_ask,
        }
        if insert_bet(row):
            logged.append(f"{city} {bracket} @ {ask}c")
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  ✅ LOGGED  "
                  f"(Σp {sigma_p:.2f} · runner {runner_bracket} @ {runner_ask}c"
                  f"{side} · gap {ask - runner_ask}c)")
        else:
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  insert failed")

        time.sleep(0.25)

    print(f"\n  logged {len(logged)} | out of band {skipped_band} | "
          f"no ladder {skipped_nomarket}")
    if not logged:
        print("  (no rows written — a retry cron may fire this window again; "
              "any such entry carries a larger minutes_late)")
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
              f"(fee est. {FEE_CENTS}c/contract, ONE-ticket basis — unconfirmed)")
    else:
        print("  no results available yet")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_et = dt.datetime.now(ET)
    print(f"FAV V1.2 | {now_et:%Y-%m-%d %H:%M} ET | {len(SERIES)} cities")
    print("no forecast — buying the market's own favorite, in band")
    print(f"latch: fires 0 to +{WINDOW_LATCH_MIN} min after target, never early")
    print("logging runner-up bracket — columns only, no bet is filtered\n")

    today = now_et.strftime("%Y-%m-%d")
    fired = False

    for hh, mm, label, lo, hi in WINDOWS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = (now_et - target).total_seconds() / 60.0

        # one-sided: early never fires
        if not (0 <= delta_min <= WINDOW_LATCH_MIN):
            continue

        if window_already_logged(today, label):
            print(f"({label} already has rows for {today} — skipping, "
                  f"this run is a retry after a successful one)")
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
              f"{nxt[0]:02d}:{nxt[1]:02d} ET, in {mins_until(nxt):.0f} min)")

    settle()

    print("\nThe tally:")
    print("  select coalesce(window_label,'TOTAL') win_label, count(*) n,")
    print("         sum((result='Won')::int) wins,")
    print("         round(100.0*avg((result='Won')::int),1) win_pct,")
    print("         round(avg(yes_ask_cents),1) avg_ask,")
    print("         round(avg(yes_ask_cents)+3.6,1) break_even_pct,")
    print("         round(sum(net_profit),2) net")
    print("  from favorites_bets where result in ('Won','Lost')")
    print("  group by rollup (window_label) order by win_label nulls last;")


if __name__ == "__main__":
    main()
