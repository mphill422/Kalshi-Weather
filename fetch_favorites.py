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

That is the entire strategy. The consensus READ added in V1.1 does not change
any of it — see AGREEMENT TAG below.

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

⚠️ THOSE THREE NUMBERS ARE STALE — kept only as the historical record of why
this file was written. They came from an exact-minute filter on kalshi_candles,
and kalshi_candles only has a row for a bracket in a given minute if that
bracket TRADED in that minute. Observed ladder completeness at 14:30 ranged
from 1 to 6 of 6 brackets. So "the top bracket" was frequently the max of a
partial ladder, biased toward actively-traded (more certain) days.

Corrected by reconstructing each ladder from the last observed ask within 90
minutes — 626 city-days per window at 5.96 avg brackets, ~1 min avg quote age:

    MORNING   58-69   Aug n=133  77.4%  +10.87   |  Sep n=35  74.3%  +8.14
    MIDDAY    58-69   Aug n=144  70.1%   +4.16   |  Sep n=46  50.0% -16.27
    AFTERNOON 58-79   Aug n=174  ~72%    ~+1     |  Sep n=72  ~68%   ~-3.5

Read the corrected numbers, not the header ones.

FORWARD RESULTS (real, 2026-09-04 to 09-06, 47 settled)
--------------------------------------------------------
    AFTERNOON  n=14  85.7%  net +12.80
    MIDDAY     n=21  71.4%  net  +8.88
    MORNING    n=12  66.7%  net  +0.83
    TOTAL      35/47 (74.5%)  net +$22.51

Note this INVERTS the backtest ordering — afternoon was the worst window in the
corrected September cut and is the best forward; morning was the best and is now
the weakest. 47 bets across 3 days is far too thin to act on, and the windows
are correlated (same city-days). Do not cut a window on this. It is recorded
because it is the only real money either way.

AGREEMENT TAG (V1.1) — what it is and what it is NOT
-----------------------------------------------------
Measured on reconstructed ladders, 33 days, comparing the market's favorite
against the bracket the weather model's CONSENSUS falls into:

    window     FAVORITE (all)        AGREE (consensus confirms)
    MORNING    n=165  76.4%  +9.83   n=51  80.4%  +14.69
    MIDDAY     n=186  65.1%  -1.02   n=50  80.0%  +13.56
    AFTERNOON  n=234  71.4%  -0.73   n=71  77.5%   +5.32

Midday and afternoon are NEGATIVE unfiltered and POSITIVE filtered. The
DISAGREE rows are where the losses sit: midday -6.38, afternoon -3.37.

⚠️ THIS FILE DOES NOT FILTER ON IT. Every in-band bet is still taken. The
agreement result is a 33-day backtest and it CONTRADICTS the 47 bets of forward
data above, which have all three windows positive unfiltered. Filtering now
would discard ~69% of volume on the strength of a backtest that the live data
disagrees with, and would make the two impossible to compare.

So: TAG, do not gate. Every row gets `agrees_with_consensus` and the consensus
value that produced it. In ~50 more bets the forward data answers it directly,
with no opportunity cost and no guessing.

⚠️ STRICT CONTAINMENT, deliberately. fetch_weather.py has
bracket_contains_consensus(..., tolerance=1.0), which counts a bracket as
containing consensus if consensus is within 1F of either edge. The head-to-head
numbers above were measured with STRICT containment — consensus inside the
bracket, no tolerance. Using the model's looser test here would match more often
and the forward numbers would not be comparable to the backtest that motivated
them. CONSENSUS_TOLERANCE_F is exposed below if that ever needs revisiting;
changing it invalidates the comparison.

⚠️ 18 CITIES, NOT 20. settlements only carries the weather model's roster.
Seattle and San Francisco were dropped in V5.30 for forecast quality and will
therefore ALWAYS have agrees_with_consensus = NULL here. That is expected, not a
bug. They are still bet normally — this strategy never needed a forecast.

⚠️ TIMING IS TIGHT IN THE MORNING. fetch_weather.py's ET EDGE window fires at
14:00 UTC and consensus rows land 14:01-14:09 UTC. The MORNING window here is
14:30 UTC — about 22 minutes of margin. If the weather model runs late or fails,
the morning read returns nothing and the row is written with a NULL tag. The bet
still logs and still settles. Never let a missing consensus block a bet.

BAND DEFINITIONS — where the numbers came from
-----------------------------------------------
FLOOR = 58, not 60, and not 55. Measured 10am-1pm ET, ask price:
    55-57c   n=246   53.7% win at 56.0c   =  -2.38
    58-59c   n=174   64.9% win at 58.5c   =  +6.44
    60-64c   n=392   68.6% win at 61.9c   =  +6.69
58-59 behaves like 60-64; 55-57 is a different population and loses. Confirmed
independently at 16:00 ET (55-57 = -2.09, everything >=58 positive), so the 58
line holds at two different times of day.

The floor has since survived two more tests. Cities that sit in the 40s and 50s
all day (Seattle 59.4% of days never in band, SF 46.9%, Denver 42.9%) were
checked as a separate population: NEVER_IN_BAND scored 45.3% at 47.8c ask,
-6.11 net. Persistent cheapness is correctly priced, not an opportunity.

CEILING = 69 in the morning, 79 in the afternoon. Checked 10:00-12:30 ET:
    58-69   n=597   72.9%   +10.16
    70      n=31    71.0%    +0.97
    71-72   n=45    80.0%    +8.62
    73-79   n=117   73.5%    -2.20
    80+     n=57    84.2%    -2.60
71-72 looks good but sits between +0.97 and -2.20 on cells of 31-45 — that is
noise, not a ceiling worth extending.

The 70-79 extension was tested properly across ALL 19 cities and CLOSED: the
70-74 slice is n=90, ~44% win against a ~75.5% break-even, roughly -27c/contract
net. San Antonio (+16.76 on n=14) and Vegas (+24.65 on n=8) look good and are
the top of a losing distribution — Miami is -41.82 in BOTH halves on n=9 and
Phoenix sign-flips +25.40 / -49.85. Do not reopen this.

NEVER use an open-ended floor. "58 and over" drags in the 80+ tier, which wins
85-96% and pays nothing at 87-94c.

NO BETS ARE STRUCTURALLY DEAD. Every cell with n>100 is negative across all
bracket ranks and price bands. At ranks 4-12 in the 80+ band the break-even
required exceeds 100% after fees — an arithmetic impossibility, not a bad bet.
Do not revisit.

TIMES — no earlier pass is worth adding
----------------------------------------
58-69 band by half hour (n=89-101 each, so comparable):
    08:00 +1.54   08:30 +0.92   09:00 +0.73   09:30 +7.61
    10:00 +4.73   10:30 +18.45  11:00 +13.77  11:30 +11.04
The count of qualifying picks is FLAT across all of these (~95), so it is not
that fewer cities qualify early — the same picks are available at 8am and are
simply wrong more often. The information that makes the band work arrives
between 09:00 and 10:30. Do not add an earlier window.

⚠️ DST. Windows are defined in EASTERN LOCAL TIME and resolved through pytz, not
hardcoded as UTC. From early November the same ET times are UTC-5.

⚠️ SEASONALITY — the biggest open risk. Summer highs are solar-driven and
boringly predictable. Winter highs are driven by frontal timing. Re-run the band
analysis each season rather than treating 58-79 as permanent.

THE LATCH — why the window is one-sided, not ±10 minutes
---------------------------------------------------------
Scheduled runs do not start on time. They start LATE and never early, so a
symmetric tolerance discards the entire late half of a distribution that is
entirely late — silently: the run succeeds, logs "no window active", and nothing
is recorded.

⚠️ THE REAL CAUSE WAS WORSE THAN QUEUE DELAY. GitHub delayed this repo's
scheduled runs by ~3 HOURS on 2026-09-03 (crons set for 20:00 UTC fired at
23:02, retry spacing preserved exactly). No latch width fixes that. Scheduling
now runs through cron-job.org -> workflow_dispatch, America/New_York, which
fires within ~1 second. favorites.yml has `workflow_dispatch` ONLY. Do not
re-add `on: schedule:`.

The latch stays because it is still correct for ordinary delay and it makes a
stray manual dispatch a safe no-op. minutes_late is stored on every row so
drift is visible rather than inferred. A window that already has rows for today
is skipped, so a retry cannot append later-priced entries beside on-time ones.

CITIES
------
No forecast means no per-city calibration. Seattle and San Francisco are fine
here despite being dropped from the weather model — this strategy never
forecasts them. (They will carry NULL agreement tags; see above.)

EXCLUDED from the research, and not in this roster: San Diego (KXHIGHTSAN,
11.8% win at 90.4c — not a real market's behavior), Louisville (KXHIGHTSDF,
same signature: 98c average with a 14.3% win rate on 7 city-days), Trenton and
Newark (~150 candle rows total). International series never appeared in the
candle backfill.

⚠️ NO PER-CITY FILTER IS SUPPORTED. Only four cities have n>=9 in-band history
and all four are positive. Every city improved from first half to second half,
which means the PERIOD was easier, not that particular cities are good. City
selection has now failed three separate tests. The band is the edge.

SETTLEMENT
----------
Settles against KALSHI'S OWN `result` field, not Iowa CLI. We are scoring a
contract, not a temperature, so any bracket-boundary disagreement between our
arithmetic and Kalshi's settlement is removed entirely.

FEES
----
`profit` remains GROSS. `fee_dollars` and `net_profit` are stored alongside it.

FEE_CENTS = 3.6 is backed out from a SINGLE Kalshi ticket. One observation is
not a fee schedule. Kalshi's fee is a function of price, so a flat cent figure
is least accurate at the ends of the band. Confirm against a second settled
ticket at a different price before treating any net figure as decided.

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

RUN THIS BEFORE DEPLOYING V1.1 (safe to re-run):

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS minutes_late          INTEGER,
    ADD COLUMN IF NOT EXISTS fee_dollars           NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS net_profit            NUMERIC(10,4),
    ADD COLUMN IF NOT EXISTS consensus_f           NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS agrees_with_consensus BOOLEAN;

THE QUERY THIS IS ALL FOR (run at ~50 settled rows per group):

  select window_label, agrees_with_consensus,
         count(*) n, sum((result='Won')::int) wins,
         round(100.0*avg((result='Won')::int),1) win_pct,
         round(avg(yes_ask_cents),1) avg_ask,
         round(sum(net_profit),2) net
  from favorites_bets
  where result <> 'Pending' and agrees_with_consensus is not null
  group by window_label, agrees_with_consensus
  order by window_label, agrees_with_consensus;

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
HEADERS = {"User-Agent": "kalshi-favorites/1.1", "Accept": "application/json"}

ET = pytz.timezone("America/New_York")

STAKE = 5.0
TAG_PREFIX = "FAV_V1"

# Per-contract fee, backed out from ONE Kalshi ticket. See FEES in the docstring.
# Applied to net_profit only; `profit` stays gross.
FEE_CENTS = 3.6

# V1.1: how far outside a bracket consensus may sit and still count as
# "contained". 0.0 = STRICT, which is what the head-to-head backtest measured.
# fetch_weather.py uses 1.0 for its own gate — a different question, and a
# looser test. Changing this invalidates the comparison to the backtest.
CONSENSUS_TOLERANCE_F = 0.0

# Windows in EASTERN LOCAL TIME. pytz resolves DST, so these stay correct in
# November when ET moves from UTC-4 to UTC-5.
#   (hour, minute, label, band_low_cents, band_high_cents_exclusive)
WINDOWS = [
    (10, 30, "MORNING",   58, 70),
    (12,  0, "MIDDAY",    58, 70),
    (16,  0, "AFTERNOON", 58, 80),
]

# One-sided latch, in minutes AFTER the window target. Delay is always late,
# never early, so an early delta must never fire. See THE LATCH.
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


def fetch_consensus_today(date_str):
    """V1.1: read today's consensus per city from settlements. READ ONLY.

    This is the ONLY dependency this file has on the weather model, and it is a
    table read rather than an import or a call. If fetch_weather.py is ever
    stripped down or deleted, keep whatever writes settlements.consensus and
    this keeps working.

    Returns {} on ANY failure. A missing consensus must never block a bet — the
    row is written with a NULL tag and settles normally.

    Only 18 cities are in settlements; Seattle and San Francisco are not.
    """
    try:
        r = requests.get(
            sb_url("settlements"),
            headers=sb_headers(),
            params={"date": f"eq.{date_str}", "select": "city,consensus"},
            timeout=15)
        if r.status_code != 200:
            print(f"  consensus read HTTP {r.status_code} — tagging disabled this run")
            return {}
        out = {}
        for row in r.json():
            c = row.get("consensus")
            if c is not None:
                try:
                    out[row.get("city")] = float(c)
                except Exception:
                    pass
        return out
    except Exception as e:
        print(f"  consensus read failed: {type(e).__name__} — tagging disabled this run")
        return {}


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


def bracket_bounds(label):
    """Parse a Kalshi bracket label into (lo, hi). None means unbounded.

        "94 to 95" / "94-95"  -> (94, 95)
        "97 or below"         -> (None, 97)
        "106 or above"        -> (106, None)
        anything unparseable  -> (None, None)

    ⚠️ Despite the range naming, Kalshi brackets settle as EXCLUSIVE ranges —
    verified by counting yes-per-event, which came back exactly 1 of 6 on every
    event checked. So a temperature belongs to exactly one bracket.

    Falls back to floor_strike / cap_strike is NOT done here on purpose: this
    parses the same label string that gets stored in `bracket`, so the stored
    row and the tag always describe the same thing.
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


def bracket_contains(label, consensus, tolerance=CONSENSUS_TOLERANCE_F):
    """Does this bracket contain the consensus temperature?

    Returns None when it cannot be determined (no consensus, unparseable
    label). None is NOT False — a bet with an unknown tag must not be counted
    as a disagreement, or the forward comparison is poisoned.

    Default tolerance is 0.0 (strict). See CONSENSUS_TOLERANCE_F.
    """
    if consensus is None:
        return None
    lo, hi = bracket_bounds(label)
    if lo is None and hi is None:
        return None
    if lo is None:
        return consensus <= hi + tolerance
    if hi is None:
        return consensus >= lo - tolerance
    return (lo - tolerance) <= consensus <= (hi + tolerance)


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
def run_window(label, band_lo, band_hi, now_et, minutes_late, consensus_map):
    today = now_et.strftime("%Y-%m-%d")
    tag = f"{TAG_PREFIX}_{label}"
    print(f"\n=== {label} window | band {band_lo}-{band_hi - 1}c | {today} "
          f"| {minutes_late} min after target ===")
    if minutes_late >= 10:
        print(f"  ⚠️ entry drift: {minutes_late} min late. Price at entry is the "
              f"strategy; check minutes_late across the sample.")
    if consensus_map:
        print(f"  consensus available for {len(consensus_map)} cities "
              f"(tagging only — no bet is filtered on it)")
    else:
        print("  ⚠️ no consensus available — rows will carry a NULL agreement tag")

    logged, skipped_band, skipped_nomarket = [], 0, 0
    n_agree = n_disagree = n_untagged = 0

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

        # V1.1: tag only. This never gates the bet.
        consensus = consensus_map.get(city)
        agrees = bracket_contains(bracket, consensus)
        if agrees is True:
            n_agree += 1
            agree_str = f"  ✓ cons {consensus:.1f}"
        elif agrees is False:
            n_disagree += 1
            agree_str = f"  ✗ cons {consensus:.1f}"
        else:
            n_untagged += 1
            agree_str = "  · no cons"

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
            "consensus_f": round(consensus, 2) if consensus is not None else None,
            "agrees_with_consensus": agrees,
        }
        if insert_bet(row):
            logged.append(f"{city} {bracket} @ {ask}c")
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  ✅ LOGGED  "
                  f"(Σp {sigma_p:.2f}){agree_str}")
        else:
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  insert failed")

        time.sleep(0.25)

    print(f"\n  logged {len(logged)} | out of band {skipped_band} | "
          f"no ladder {skipped_nomarket}")
    if logged:
        print(f"  agreement: {n_agree} agree | {n_disagree} disagree | "
              f"{n_untagged} untagged  (all {len(logged)} were taken)")
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
    agree_w = agree_l = solo_w = solo_l = 0

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

            # running tally of the thing V1.1 exists to answer
            ag = b.get("agrees_with_consensus")
            if ag is True:
                if res == "yes":
                    agree_w += 1
                else:
                    agree_l += 1
            elif ag is False:
                if res == "yes":
                    solo_w += 1
                else:
                    solo_l += 1

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
        if (agree_w + agree_l) or (solo_w + solo_l):
            a_n, s_n = agree_w + agree_l, solo_w + solo_l
            a_pct = f"{100.0*agree_w/a_n:.0f}%" if a_n else "—"
            s_pct = f"{100.0*solo_w/s_n:.0f}%" if s_n else "—"
            print(f"  this batch — AGREE {agree_w}/{a_n} ({a_pct}) | "
                  f"DISAGREE {solo_w}/{s_n} ({s_pct})")
            print("  (one batch proves nothing; run the grouped query at ~50 each)")
    else:
        print("  no results available yet")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_et = dt.datetime.now(ET)
    print(f"FAV V1.1 | {now_et:%Y-%m-%d %H:%M} ET | {len(SERIES)} cities")
    print("no forecast — buying the market's own favorite, in band")
    print(f"latch: fires 0 to +{WINDOW_LATCH_MIN} min after target, never early")
    print(f"consensus agreement: TAGGED, never gated "
          f"(tolerance {CONSENSUS_TOLERANCE_F}F, strict)\n")

    today = now_et.strftime("%Y-%m-%d")
    fired = False

    for hh, mm, label, lo, hi in WINDOWS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = (now_et - target).total_seconds() / 60.0

        # One-sided: early never fires. This is also what makes an off-season
        # DST twin decline cleanly.
        if not (0 <= delta_min <= WINDOW_LATCH_MIN):
            continue

        if window_already_logged(today, label):
            print(f"({label} already has rows for {today} — skipping, "
                  f"this run is a retry that arrived after a successful one)")
            fired = True
            continue

        # Read consensus only when a window is actually firing.
        consensus_map = fetch_consensus_today(today)
        run_window(label, lo, hi, now_et, int(round(delta_min)), consensus_map)
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

    print("\nAgreement split (the V1.1 question — needs ~50 settled per group):")
    print("  select window_label, agrees_with_consensus,")
    print("         count(*) n, sum((result='Won')::int) wins,")
    print("         round(100.0*avg((result='Won')::int),1) win_pct,")
    print("         round(avg(yes_ask_cents),1) avg_ask,")
    print("         round(sum(net_profit),2) net")
    print("  from favorites_bets")
    print("  where result <> 'Pending' and agrees_with_consensus is not null")
    print("  group by window_label, agrees_with_consensus")
    print("  order by window_label, agrees_with_consensus;")


if __name__ == "__main__":
    main()
