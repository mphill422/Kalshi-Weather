"""
fetch_weather.py — MPH Weather Model V5.31.1

V5.31.1 changes from V5.31.0:
  - STORE NBM p50 in settlements.nbm_p50.

    V5.31.0 re-centered the bracket ladder on consensus because the chosen
    bracket sat 1-2F BELOW the model's own consensus in 18 of 20 cities. The
    working explanation is that NBM's central estimate runs cold. But that is
    an INFERENCE, not a measurement — NBM percentiles were fetched live and
    discarded, so NBM p50 has never once been scored against a settled actual.

    That is a weak foundation under a change that affects every bet. This
    stores p50 alongside consensus so the question becomes answerable in about
    ten days:

        select city,
               round(avg(actual::numeric - consensus::numeric), 2) as consensus_err,
               round(avg(actual::numeric - nbm_p50::numeric), 2)   as nbm_err,
               count(*) as n
        from settlements
        where actual is not null and nbm_p50 is not null
        group by city order by nbm_err desc;

    Read it with the same sign convention as everything else: POSITIVE means
    actual came in warmer than the estimate, i.e. that estimate runs COLD.
    If nbm_err is consistently more positive than consensus_err, NBM does run
    colder and V5.31 is anchoring on the better of the two. If they are
    similar, the offset came from somewhere else and the anchor is treating a
    symptom.

    RUN THIS FIRST or every upsert 400s:
        ALTER TABLE public.settlements ADD COLUMN IF NOT EXISTS nbm_p50 NUMERIC(6,2);

    CAVEAT: NBM is only fetched when a paper-bet window is firing (see main()),
    so nbm_p50 is NULL on the other runs. That still yields ~5 readings per
    city per day, which is more than enough. Do not read a NULL as "NBM was
    unavailable" — check whether a window was firing first.

    This is storage only. It does not touch the ladder, the gates, or any
    probability. It does not reset the V5.31 validation window.

V5.31.0 changes from V5.30.0:
  - NBM CONSENSUS ANCHOR. This is the big one, and it changes every bet.

    The bracket ladder was built ENTIRELY from NBM percentiles. `consensus`
    was passed into bracket_probs() but only reached the math when
    nbm_bracket_prob() returned None for an individual bracket. Net effect:
    the NWS/GFS blend, the per-city warm offsets, the bias correction, and
    the V5.30 trimmed-mean fix all moved a number that got logged, displayed,
    and stored in settlements.consensus — but NEVER reached bet selection.
    NBM alone picked the bracket.

    Measured on kalshi_snapshots joined to settlements (72 days, 20 cities):
      * The chosen bracket midpoint sat BELOW the model's own consensus in
        18 of 20 cities. LA -2.34F, DC -1.93F, Atlanta -1.89F, Denver -1.76F,
        Philadelphia -1.42F, Phoenix -1.41F, Austin -1.41F. Only Dallas
        (+0.21) and New Orleans (+0.12) were positive.
      * The cities with the worst ladder offset were the SAME cities with the
        worst bracket-vs-actual error — near one-to-one, so the offset
        explains most of the cold bias rather than being a separate problem.
      * Consensus itself is well calibrated. After the V5.30 trimmed mean,
        actual-minus-consensus ran +1.23 (Las Vegas) to -0.69 (New York),
        11 cities positive and 7 negative, mean near +0.18.
      * The market's own favorite ran -0.44F vs actual — near unbiased.
      * When model and market disagreed, the market won 762 to 171 (81.7%).

    Ruled out before landing here: BASE_SIGMA (no correlation — Boston at
    sigma 3.5 offsets -1.00 while Phoenix at 0.9 offsets -1.41),
    CITY_PREDICTION_MODE (both modes appear in both groups), and
    NWS_BIAS_BOOST_CITIES (San Antonio is in the set at -0.17 while LA and
    Atlanta are not in it and are the worst two).

    Fix: nbm_bracket_prob() now takes an `anchor` and shifts the whole
    percentile distribution so its p50 lands on consensus. NBM still supplies
    the SHAPE — the spread, and therefore the uncertainty structure — while
    consensus supplies the CENTER.

    NBM_CONSENSUS_ANCHOR controls it:
      1.0 = fully centered on consensus   (V5.31 default)
      0.5 = halfway blend
      0.0 = exact V5.30 behavior          (rollback — change the number only)

  - PAPER TAGS BUMPED V530_ → V531_. Required, not cosmetic. Picks made
    under two different bracket-selection models must not share a strategy
    tag, or the corpus mixes populations and no per-tag comparison is valid.
    The V530_ rows stay where they are as the pre-change baseline.

V5.30.0 changes from V5.29.F:
  - ROSTER CONTRACTION. San Francisco (KSFO) and Seattle (KSEA) REMOVED.
    20 cities → 18 cities. Reason: measured performance. Across settled data
    SF ran a -2.14F residual bias (worst on the board by 4x) with its best
    available model at 1.8F MAE against narrow marine brackets, and Seattle
    scattered badly every day of manual-grid logging. Neither would clear a
    60c price floor often, and when they did we would be betting the two
    least reliable forecasts we have. They were also never in streamlit_app.py
    CITIES, so they logged predictions that could never become bets — 20 rows
    written, 18 bet-eligible.

  - CONSENSUS FLOOR FIX (the important one). An observed high is a
    MEASUREMENT, not a forecast base. compute_consensus() previously did
    `consensus = obs_high` and then added the city warm offset on top, and
    main() then added bias_correction on top of THAT. Phoenix showed
    consensus 114.35F (floor 113.0 + offset 1.0 + bias 0.35) while every
    forecast input read below 111F. Now the floor clamps from below via
    max(), applied AFTER offset and AFTER bias, so adjustments shape the
    forecast estimate and the measurement can only raise the result, never
    be inflated by it.

  - LADDER NORMALIZATION. apply_prob_floor() did not normalize. Observed a
    Phoenix ladder summing to 13.9% (0.0 + 1.6 + 4.1 + 4.1 + 4.1 + 4.1) —
    four identical values because the floor set them all to 0.05 and the
    old rescale only divided by (1 + boost_total). Every downstream edge
    number was computed against a distribution missing ~86% of its mass.
    Now sums to 1.0, with busted brackets held at zero and excluded from
    redistribution.

  - BIAS CORRECTION: median → trimmed mean.
    VERIFIED WORKING 2026-08-29 on 9 settled days per city under V5.30.0:
    the one-directional bias is gone, mean residual now ~+0.18F. This fix
    did its job. Note the sign convention: errors are actual MINUS
    consensus, so a POSITIVE residual means settlement came in WARMER than
    predicted — the model runs COLD, not warm.

  - PRICE_FLOOR_CENTS DELIBERATELY LEFT AT 30 HERE. The paper tags this file
    writes are the RESEARCH corpus. The 60c floor lives in streamlit_app.py
    where the AUTO_GATED_V2 strategy tag runs. Two different jobs: this file
    measures, that file bets.

V5.29.F changes from V5.29.E:
  - HOUSTON COORDINATE FIX. CITIES['Houston'] was 29.9902/-95.3368 — those
    are BUSH INTERCONTINENTAL (KIAH) coordinates. Kalshi settles Houston on
    HOBBY (KHOU), and both WETHR_STATIONS and CLI_STATIONS correctly said
    KHOU. So settlement was right, but the NWS grid fallback and BOTH
    Open-Meteo calls were pulling forecast data for the WRONG AIRPORT.
  - Corrected to 29.6459/-95.2769 (KHOU Hobby).
  - Known trap set (per Kalshi market rules): O'Hare vs Midway (we use KMDW,
    verified), Bush vs Hobby (KHOU, fixed), Love Field vs DFW (we use
    KDFW — STILL UNVERIFIED, check KXHIGHTDAL market rules).

V5.29.D: ensemble-aware consensus correction (>3F gap → half-shift, ±2F cap).
V5.29.C: GFS /v1/forecast fallback when /v1/ensemble fails.
V5.29.B: per-city sigma recalibration from observed 28-day data.
V5.29.A: Σp gate — skip the city if the YES ladder sums over 1.15.
V5.28.4 / .3 / .2 / .1: event_ticker date fix, raw API dump, dedup, snapshots.
V5.28: BLOCK NO bets on the model's TOP bracket. Busted brackets exempt.
"""

import math
import os
import re
import requests
import statistics
import time
from datetime import datetime, timedelta

import pytz

# ── Credentials ──────────────────────────────────────────────────────────────
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
SUPABASE_URL  = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY  = os.environ.get('SUPABASE_KEY', '')

WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS       = {'User-Agent': 'kalshi-weather-fetcher/5.31.1', 'Accept': 'application/json'}

# ── Validator Configuration ──────────────────────────────────────────────────
PAPER_BET_STAKE       = 3.0
# NOTE (V5.30.0): stays at 30 on purpose. See header. This file generates the
# research corpus across ALL price bands; the 60c betting floor lives in
# streamlit_app.py under the AUTO_GATED_V2 tag.
PRICE_FLOOR_CENTS     = 30
CONSENSUS_TOP_N       = 2
TRUST_THRESHOLDS      = [75, 80]
WINDOW_TOLERANCE_MIN  = 6

SIGMA_P_GATE_MAX      = 1.15

# V5.31: how much the NBM ladder is re-centered on consensus.
#   1.0 = fully centered on consensus (NBM supplies shape only)   [default]
#   0.5 = halfway blend
#   0.0 = original V5.30 behavior (NBM owns placement entirely)   [rollback]
# MUST MATCH the value in streamlit_app.py. If the two files disagree, the
# app and the cron will pick different brackets for the same city on the same
# day, and the paper corpus becomes a mixture of two models.
NBM_CONSENSUS_ANCHOR  = 1.0

# V5.31: paper tag prefix. Bumped V530_ → V531_ because the bracket-selection
# math changed. Rows written under a different prefix are a different
# population and must not be pooled with these.
PAPER_TAG_PREFIX      = 'V531'

WINDOW_SCHEDULE = {
    (14,  0): [('ET', 'EDGE')],
    (14, 30): [('CT', 'EDGE')],
    (15, 30): [('ET', 'CONVICTION'), ('PT', 'EDGE')],
    (16, 30): [('CT', 'CONVICTION'), ('MT', 'EDGE')],
    (18, 30): [('MT', 'CONVICTION'), ('PT', 'CONVICTION')],
}

TZ_CITIES = {
    'ET': ['New York', 'Boston', 'Philadelphia', 'Washington DC', 'Atlanta', 'Miami'],
    'CT': ['Chicago', 'Dallas', 'Austin', 'Houston', 'San Antonio',
           'New Orleans', 'Oklahoma City', 'Minneapolis'],
    'MT': ['Denver'],
    'PT': ['Phoenix', 'Las Vegas', 'Los Angeles'],
}

CITY_TZ = {
    'Phoenix': 'America/Phoenix', 'Las Vegas': 'America/Los_Angeles',
    'Los Angeles': 'America/Los_Angeles', 'Dallas': 'America/Chicago',
    'Austin': 'America/Chicago', 'Houston': 'America/Chicago',
    'Atlanta': 'America/New_York', 'Miami': 'America/New_York',
    'New York': 'America/New_York', 'San Antonio': 'America/Chicago',
    'New Orleans': 'America/Chicago', 'Philadelphia': 'America/New_York',
    'Boston': 'America/New_York', 'Denver': 'America/Denver',
    'Oklahoma City': 'America/Chicago', 'Minneapolis': 'America/Chicago',
    'Washington DC': 'America/New_York', 'Chicago': 'America/Chicago',
}

CITIES = {
    'Phoenix':       {'lat': 33.4342, 'lon': -112.0116},
    'Las Vegas':     {'lat': 36.0840, 'lon': -115.1537},
    'Los Angeles':   {'lat': 33.9416, 'lon': -118.4085},
    'Dallas':        {'lat': 32.8998, 'lon': -97.0403},
    'Austin':        {'lat': 30.1945, 'lon': -97.6699},
    'Houston':       {'lat': 29.6459, 'lon': -95.2769},  # V5.29.F: KHOU Hobby (was Bush/KIAH coords)
    'Atlanta':       {'lat': 33.6407, 'lon': -84.4277},
    'Miami':         {'lat': 25.7959, 'lon': -80.2870},
    'New York':      {'lat': 40.7812, 'lon': -73.9665},
    'San Antonio':   {'lat': 29.5337, 'lon': -98.4698},
    'New Orleans':   {'lat': 29.9934, 'lon': -90.2580},
    'Philadelphia':  {'lat': 39.8744, 'lon': -75.2424},
    'Boston':        {'lat': 42.3656, 'lon': -71.0096},
    'Denver':        {'lat': 39.8561, 'lon': -104.6737},
    'Oklahoma City': {'lat': 35.3931, 'lon': -97.6007},
    'Minneapolis':   {'lat': 44.8848, 'lon': -93.2223},
    'Washington DC': {'lat': 38.8512, 'lon': -77.0402},
    'Chicago':       {'lat': 41.7868, 'lon': -87.7522},
}

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

CLI_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

SERIES = {
    'Phoenix': 'KXHIGHTPHX', 'Las Vegas': 'KXHIGHTLV',
    'Los Angeles': 'KXHIGHLAX', 'Dallas': 'KXHIGHTDAL',
    'Austin': 'KXHIGHAUS', 'Houston': 'KXHIGHTHOU',
    'Atlanta': 'KXHIGHTATL', 'Miami': 'KXHIGHMIA',
    'New York': 'KXHIGHNY', 'San Antonio': 'KXHIGHTSATX',
    'New Orleans': 'KXHIGHTNOLA', 'Philadelphia': 'KXHIGHPHIL',
    'Boston': 'KXHIGHTBOS', 'Denver': 'KXHIGHDEN',
    'Oklahoma City': 'KXHIGHTOKC', 'Minneapolis': 'KXHIGHTMIN',
    'Washington DC': 'KXHIGHTDC', 'Chicago': 'KXHIGHCHI',
}

CITY_PREDICTION_MODE = {
    'New York':      'full_blend', 'Houston':       'full_blend',
    'Dallas':        'full_blend', 'Los Angeles':   'full_blend',
    'Phoenix':       'full_blend', 'Las Vegas':     'full_blend',
    'Boston':        'full_blend', 'Philadelphia':  'full_blend',
    'Miami':         'nws_only',   'New Orleans':   'nws_only',
    'Washington DC': 'nws_only',   'Atlanta':       'nws_only',
    'Oklahoma City': 'nws_only',   'Chicago':       'nws_only',
    'Denver':        'nws_only',   'Austin':        'nws_only',
    'Minneapolis':   'nws_only',   'San Antonio':   'nws_only',
}

CITY_WARM_OFFSET = {'Phoenix': 1.0, 'Las Vegas': -1.0}

FORECAST_HEAVY_CITIES = {'Dallas', 'Austin', 'Houston', 'San Antonio', 'Oklahoma City'}
DESERT_CITIES         = {'Phoenix', 'Las Vegas'}
NORTHEAST_CITIES      = {'New York', 'Philadelphia', 'Boston', 'Washington DC'}
REGIONAL_PRIOR_BIAS   = {'Chicago': 'Minneapolis'}

GFS_CITY_WEIGHT = {
    'Houston': 0.18, 'Phoenix': 0.0, 'Las Vegas': 0.0, 'Los Angeles': 0.0,
    'Miami': 0.0, 'New Orleans': 0.0, 'Dallas': 0.0, 'Austin': 0.0,
    'San Antonio': 0.0, 'Oklahoma City': 0.0, 'Atlanta': 0.0, 'Denver': 0.0,
    'Minneapolis': 0.0, 'Chicago': 0.0, 'New York': 0.0, 'Philadelphia': 0.0,
    'Boston': 0.0, 'Washington DC': 0.0,
}

# V5.29.B per-city sigma recalibration from observed 28-day data (Jun 17 2026).
# Capped at 3.5F to prevent single-day data feed failures (e.g. OKC's 22F
# outlier) from over-inflating. Re-run quarterly or after structural changes.
#
# V5.31 NOTE: sigma only affects the FALLBACK path (sigma_bracket_prob), used
# when NBM is unavailable or returns None for a bracket. It was tested as a
# candidate explanation for the ladder offset and ruled out — no correlation.
BASE_SIGMA = {
    'Boston': 3.5,        # was 1.9 — observed σ 3.66 (capped)
    'Oklahoma City': 3.5, # was 2.5 — observed σ 4.35 (capped, outlier-inflated)
    'San Antonio': 3.2,   # was 2.3 — observed σ 3.19
    'New York': 3.0,      # was 1.8 — observed σ 3.03
    'Dallas': 3.0,        # was 2.3 — observed σ 3.02
    'Miami': 3.0,         # was 2.0 — observed σ 2.99
    'Atlanta': 2.8,       # was 2.3 — observed σ 2.77
    'Los Angeles': 2.7,   # was 1.7 — observed σ 2.69
    'Austin': 2.6,        # was 2.3 — observed σ 2.64
    'Washington DC': 2.6, # was 1.9 — observed σ 2.60
    'Chicago': 2.3,       # was 2.1 — observed σ 2.31
    'Minneapolis': 2.2,   # was 2.1 — observed σ 2.20
    'Denver': 2.1,        # was 1.9 — observed σ 2.12
    'Houston': 2.1,       # was 2.3 — observed σ 2.06 (slight tighten)
    'New Orleans': 2.0,   # was 2.1 — observed σ 1.95
    'Philadelphia': 1.7,  # was 1.8 — observed σ 1.74
    'Las Vegas': 1.2,     # was 2.2 — observed σ 1.24 (major tighten)
    'Phoenix': 0.9,       # was 2.2 — observed σ 0.94 (major tighten)
}

NWS_BIAS_BOOST_CITIES = {'Washington DC', 'Oklahoma City', 'Denver', 'Austin', 'San Antonio'}
NWS_BIAS_BOOST_MULTIPLIER = 1.2

OBS_HIGH_TRUST_HOUR              = 13
OBS_HIGH_MAX_OVERSHOOT           = 10.0
OBS_HIGH_OVER_CURRENT_THRESHOLD  = 10.0
OBS_HIGH_OVER_FORECAST_THRESHOLD = 12.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_eastern_date():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')

def get_local_hour(city):
    return datetime.now(pytz.timezone(CITY_TZ.get(city, 'America/New_York'))).hour

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def c_to_f(c):
    return c * 9 / 5 + 32

def safe_get(url, params=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  safe_get failed {url}: {e}')
        return None

def normalize_label(label):
    if not label:
        return ''
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)', lambda m: m.group(1)+'-'+m.group(2), label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)', lambda m: m.group(1)+'-'+m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above').strip()

def label_to_numeric_key(label):
    label = normalize_label(label)
    nums = [int(x) for x in re.findall(r'\d+', label)]
    low = label.lower()
    if not nums:
        return None, None
    if 'below' in low:
        return None, nums[0]
    if 'above' in low:
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def labels_match(a, b):
    return label_to_numeric_key(a) == label_to_numeric_key(b)


# ── Supabase ──────────────────────────────────────────────────────────────────
def sb_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': 'Bearer ' + SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }

def sb_url(table):
    return SUPABASE_URL + '/rest/v1/' + table

def sb_fetch_city(city):
    try:
        r = requests.get(sb_url('settlements'), headers=sb_headers(),
                         params={'city': 'eq.' + city, 'order': 'date.asc', 'limit': '200'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def sb_fetch_today(city):
    today = get_eastern_date()
    try:
        r = requests.get(sb_url('settlements'), headers=sb_headers(),
                         params={'date': 'eq.' + today, 'city': 'eq.' + city}, timeout=10)
        rows = r.json() if r.status_code == 200 else []
        return rows[0] if rows else None
    except Exception:
        return None

def sb_upsert(city, consensus, forecast, ensemble_mean, source_gap,
              high_uncertainty, obs_high, bias_correction, nbm_p50=None):
    """V5.31.1: nbm_p50 added so NBM's central estimate can eventually be
    scored against actual. Storage only — nothing reads it yet.

    Requires: ALTER TABLE public.settlements ADD COLUMN IF NOT EXISTS
              nbm_p50 NUMERIC(6,2);

    On an existing row nbm_p50 is only overwritten when a non-null value is
    supplied. That matters because NBM is fetched only during firing windows:
    a later non-window run of the same day must not blank out a p50 that an
    earlier window run already stored.
    """
    today = get_eastern_date()
    existing = sb_fetch_today(city)
    row = {
        'date': today, 'city': city,
        'consensus': round(consensus, 2),
        'forecast': round(forecast, 2) if forecast else None,
        'ensemble_mean': round(ensemble_mean, 2) if ensemble_mean else None,
        'source_gap': round(source_gap, 2) if source_gap else None,
        'high_uncertainty': bool(high_uncertainty),
        'obs_high': round(obs_high, 2) if obs_high else None,
        'bias_correction': round(bias_correction, 2),
        'nbm_p50': round(float(nbm_p50), 2) if nbm_p50 is not None else None,
        'actual': None, 'error': None,
    }
    if existing:
        update = {k: v for k, v in row.items() if k not in ('date', 'city')}
        if existing.get('actual') is not None:
            update.pop('actual', None)
            update.pop('error', None)
        # Do not overwrite a stored p50 with a null from a non-window run.
        if nbm_p50 is None:
            update.pop('nbm_p50', None)
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(existing['id']),
            headers=sb_headers(), json=update, timeout=10)
        return r.status_code in (200, 204)
    else:
        r = requests.post(sb_url('settlements'), headers=sb_headers(), json=row, timeout=10)
        return r.status_code in (200, 201)


def sb_insert_paper_bet(bet_row):
    try:
        r = requests.post(sb_url('bets'), headers=sb_headers(), json=bet_row, timeout=10)
        return r.status_code in (200, 201)
    except Exception as e:
        print(f'    ❌ Paper-bet insert exception: {e}')
        return False


def sb_count_paper_bets_today(city, strategy_tag):
    today = get_eastern_date()
    try:
        r = requests.get(
            sb_url('bets'),
            headers=sb_headers(),
            params={
                'date': 'eq.' + today,
                'city': 'eq.' + city,
                'strategy_tag': 'eq.' + strategy_tag,
                'select': 'id',
            },
            timeout=10,
        )
        if r.status_code == 200:
            return len(r.json())
        return 0
    except Exception:
        return 0


# ── V5.28.1: Kalshi market snapshot logger ───────────────────────────────────
def snapshot_kalshi_market(city, kalshi_markets, window_tz, window_label,
                           model_top_bracket, utc_now):
    """Snapshot the full Kalshi ladder for one city to kalshi_snapshots table.

    V5.31 NOTE: this table is what made the ladder-offset diagnosis possible.
    model_top_bracket joined to settlements.consensus is how we found the
    chosen bracket sitting 1-2F below the model's own forecast, and joined to
    settlements.actual is how we found the market winning disagreements
    762-171. Keep writing it. Two snapshots per city-day (EDGE and CONVICTION)
    is thin for intraday questions but sufficient for this one.
    """
    if not kalshi_markets:
        return 0
    try:
        valid_yes = [m[1] for m in kalshi_markets if m[1] is not None]
        if not valid_yes:
            return 0
        sigma_p = round(sum(valid_yes) / 100.0, 4)

        ranked = sorted(
            [(idx, m) for idx, m in enumerate(kalshi_markets) if m[1] is not None],
            key=lambda x: x[1][1],
            reverse=True,
        )
        rank_by_idx = {orig_idx: rank for rank, (orig_idx, _) in enumerate(ranked, start=1)}

        today = get_eastern_date()
        snap_ts = utc_now.replace(microsecond=0).isoformat() + 'Z'

        rows = []
        for idx, (label, yes_ask, no_ask) in enumerate(kalshi_markets):
            rows.append({
                'snapshot_time': snap_ts,
                'date': today,
                'city': city,
                'bracket_label': normalize_label(label),
                'yes_price_cents': yes_ask,
                'no_price_cents': no_ask,
                'bracket_rank': rank_by_idx.get(idx),
                'sigma_p': sigma_p,
                'window_tz': window_tz or None,
                'window_label': window_label or None,
                'utc_hour': utc_now.hour,
                'model_top_bracket': normalize_label(model_top_bracket) if model_top_bracket else None,
            })

        r = requests.post(
            sb_url('kalshi_snapshots'),
            headers=sb_headers(),
            json=rows,
            timeout=10,
        )
        if r.status_code in (200, 201):
            return len(rows)
        else:
            print(f'    ⚠️ Snapshot insert non-200 for {city}: HTTP {r.status_code}')
            return 0
    except Exception as e:
        print(f'    ⚠️ Snapshot exception for {city} (non-fatal): {type(e).__name__}: {str(e)[:120]}')
        return 0


# ── V5.28.3-diag: Raw Kalshi API dump (diagnostic only) ──────────────────────
def dump_raw_kalshi_response(city, series, raw_data, endpoint_used,
                              market_count, filtered_count, final_count):
    """Dump raw Kalshi API response to kalshi_api_dumps table for diagnosis."""
    if not raw_data:
        return
    try:
        today = get_eastern_date()
        row = {
            'date': today,
            'city': city,
            'series': series,
            'endpoint_used': endpoint_used,
            'market_count': market_count,
            'filtered_count': filtered_count,
            'final_count': final_count,
            'raw_response': raw_data,
        }
        r = requests.post(
            sb_url('kalshi_api_dumps'),
            headers=sb_headers(),
            json=row,
            timeout=10,
        )
        if r.status_code not in (200, 201):
            print(f'    ⚠️ API dump insert non-200 for {city}: HTTP {r.status_code}')
    except Exception as e:
        print(f'    ⚠️ API dump exception for {city} (non-fatal): {type(e).__name__}: {str(e)[:120]}')


# ── Bias correction (V5.30.0: trimmed mean) ──────────────────────────────────
def _trimmed_mean(errors):
    """Drop the single highest and lowest error, average the rest.

    V5.30.0 BIAS FIX. statistics.median() discarded the systematic middle of a
    skewed error distribution. Across 2,514 settlements the residual bias AFTER
    correction was still positive in 17 of 18 cities (Boston +1.07, Atlanta
    +1.06, Miami +1.02, DC +0.91, Chicago +0.77 ... Houston -0.14, NYC -0.07).

    Trimmed mean keeps outlier robustness (a +12F sensor-spike style error is
    still dropped, where a plain mean would blow the correction up to +1.6)
    without throwing away the persistent drift the median was compressing.

    VERIFIED WORKING 2026-08-29 on 9 settled days per city under V5.30.0:
    residuals now run +1.23 (Las Vegas) to -0.69 (New York), 11 positive and
    7 negative, mean ~+0.18F. The one-directional bias is gone.

    SIGN CONVENTION: errors are actual MINUS consensus. A POSITIVE value means
    settlement came in WARMER than the model predicted — the model runs COLD.
    Misreading this cost a wrong call on 2026-08-24.
    """
    if not errors:
        return 0.0
    if len(errors) < 4:
        return sum(errors) / len(errors)
    s = sorted(errors)[1:-1]
    return sum(s) / len(s)


def compute_bias_correction(city, n_recent=10):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        prior_city = REGIONAL_PRIOR_BIAS.get(city)
        if prior_city:
            prior_rows = sb_fetch_city(prior_city)
            prior_complete = [r for r in prior_rows
                              if r.get('actual') is not None and r.get('consensus') is not None]
            if len(prior_complete) >= 3:
                recent = prior_complete[-n_recent:]
                errors = [r['actual'] - r['consensus'] for r in recent]
                med_error = _trimmed_mean(errors)
                return round(max(-3.0, min(3.0, med_error)), 2), len(complete)
        return 0.0, len(complete)
    recent = complete[-n_recent:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    med_error = _trimmed_mean(errors)
    if city in NWS_BIAS_BOOST_CITIES:
        med_error = med_error * NWS_BIAS_BOOST_MULTIPLIER
    return round(max(-3.0, min(3.0, med_error)), 2), len(recent)


def get_city_mae_color(city, n_recent=14):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('error') is not None]
    if len(complete) < 3:
        return 'green'
    recent = complete[-n_recent:]
    errors = [abs(r['error']) for r in recent]
    mae = sum(errors) / len(errors)
    if mae < 2.5: return 'green'
    if mae < 4.0: return 'yellow'
    return 'red'


# ── NWS forecast ──────────────────────────────────────────────────────────────
_NWS_GRID_CACHE = {}

def fetch_nws_grid(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _NWS_GRID_CACHE:
        return _NWS_GRID_CACHE[key]
    try:
        r = requests.get(f'https://api.weather.gov/points/{lat},{lon}',
                         headers=HEADERS, timeout=12)
        r.raise_for_status()
        props = r.json().get('properties', {})
        office = props.get('gridId')
        gx = props.get('gridX')
        gy = props.get('gridY')
        if not all([office, gx is not None, gy is not None]):
            return None
        result = (office, gx, gy)
        _NWS_GRID_CACHE[key] = result
        return result
    except Exception:
        return None

def fetch_nws_forecast(city):
    station = WETHR_STATIONS.get(city)
    today = get_eastern_date()
    if station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/nws_forecasts.php',
                params={'station_code': station, 'date': today, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json()
                high = data.get('high')
                if high is not None:
                    return round(float(high), 1)
        except Exception:
            pass
    coords = CITIES[city]
    grid = fetch_nws_grid(coords['lat'], coords['lon'])
    if not grid:
        return None
    office, gx, gy = grid
    hourly_url = f'https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast/hourly'
    try:
        r = requests.get(hourly_url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        periods = r.json().get('properties', {}).get('periods', [])
        today_highs = []
        for period in periods:
            start = period.get('startTime', '')
            temp = period.get('temperature')
            unit = period.get('temperatureUnit', 'F')
            is_day = period.get('isDaytime', True)
            if not start.startswith(today):
                continue
            if temp is not None and is_day:
                temp_f = float(temp) if unit == 'F' else float(temp) * 9/5 + 32
                today_highs.append(temp_f)
        if today_highs:
            return round(max(today_highs), 1)
    except Exception:
        pass
    return None


def fetch_current_temp(city):
    station = WETHR_STATIONS.get(city)
    if station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': station, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json()
                temp = data.get('temperature_display')
                if temp is not None:
                    return round(float(temp), 1)
        except Exception:
            pass
    return None


def fetch_obs_high(city):
    station = WETHR_STATIONS.get(city)
    if station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': station, 'mode': 'wethr_high', 'logic': 'nws'},
                headers=WETHR_HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json()
                wethr_high = data.get('wethr_high')
                if wethr_high is not None:
                    return round(float(wethr_high), 1)
        except Exception:
            pass
    return None


# ── GFS ensemble (V5.27.2: 45s timeout + retry) ──────────────────────────────
def fetch_gfs_forecast_fallback(city):
    """V5.29.C: Single-point GFS forecast fallback when ensemble fails."""
    coords = CITIES[city]
    lat, lon = coords['lat'], coords['lon']
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'auto', 'forecast_days': 2,
        'models': 'gfs_seamless',
    }
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'    ⚠️ [{city}] GFS fallback /v1/forecast FAILED: '
              f'{type(e).__name__}: {str(e)[:120]}')
        return None, None

    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    temps = hourly.get('temperature_2m', [])
    if not times or not temps:
        return None, None

    today_temps = []
    for i, t in enumerate(times):
        if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21:
            if i < len(temps) and temps[i] is not None:
                try:
                    today_temps.append(float(temps[i]))
                except Exception:
                    pass
    if not today_temps:
        return None, None
    max_temp = round(max(today_temps), 1)
    return [max_temp], max_temp


def fetch_gfs_ensemble(city):
    coords = CITIES[city]
    lat, lon = coords['lat'], coords['lon']
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'auto', 'forecast_days': 2,
        'models': 'gfs_seamless',
    }
    data = None
    for attempt in (1, 2):
        try:
            r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble',
                             params=params, headers=HEADERS, timeout=45)
            r.raise_for_status()
            data = r.json()
            if attempt == 2:
                print(f'    ✅ [{city}] GFS ensemble recovered on retry')
            break
        except requests.exceptions.Timeout as e:
            if attempt == 1:
                print(f'    … [{city}] GFS ensemble timeout (attempt 1/2), retrying once')
                continue
            else:
                print(f'    ⚠️ [{city}] GFS ensemble fetch FAILED (timeout after retry): '
                      f'{type(e).__name__}: {str(e)[:120]}')
                fb_members, fb_mean = fetch_gfs_forecast_fallback(city)
                if fb_members is not None:
                    print(f'    🔁 [{city}] GFS fallback fired (timeout): mean={fb_mean}F')
                return fb_members, fb_mean
        except Exception as e:
            print(f'    ⚠️ [{city}] GFS ensemble fetch FAILED (network/HTTP): '
                  f'{type(e).__name__}: {str(e)[:120]}')
            fb_members, fb_mean = fetch_gfs_forecast_fallback(city)
            if fb_members is not None:
                print(f'    🔁 [{city}] GFS fallback fired (network): mean={fb_mean}F')
            return fb_members, fb_mean
    if data is None:
        fb_members, fb_mean = fetch_gfs_forecast_fallback(city)
        if fb_members is not None:
            print(f'    🔁 [{city}] GFS fallback fired (no_data): mean={fb_mean}F')
        return fb_members, fb_mean
    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    today_indices = [i for i, t in enumerate(times)
                     if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21]
    if not today_indices:
        today_indices = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_indices:
        sample_times = times[:3] if times else []
        print(f'    ⚠️ [{city}] GFS ensemble NO TODAY MATCH: looking for "{today}", '
              f'series has {len(times)} entries, sample: {sample_times}')
        fb_members, fb_mean = fetch_gfs_forecast_fallback(city)
        if fb_members is not None:
            print(f'    🔁 [{city}] GFS fallback fired (no_today_match): mean={fb_mean}F')
        return fb_members, fb_mean
    member_maxes = []
    for key, vals in hourly.items():
        if key == 'time' or 'temperature_2m' not in key or not isinstance(vals, list):
            continue
        today_vals = [vals[i] for i in today_indices if i < len(vals) and vals[i] is not None]
        if today_vals:
            try:
                member_maxes.append(round(max(float(v) for v in today_vals), 1))
            except Exception:
                pass
    if len(member_maxes) < 3:
        member_key_count = sum(1 for k in hourly.keys()
                               if k != 'time' and 'temperature_2m' in k)
        print(f'    ⚠️ [{city}] GFS ensemble SPARSE GRID: got {len(member_maxes)} member maxes '
              f'(need ≥3); response had {member_key_count} ensemble keys total')
        fb_members, fb_mean = fetch_gfs_forecast_fallback(city)
        if fb_members is not None:
            print(f'    🔁 [{city}] GFS fallback fired (sparse_grid): mean={fb_mean}F')
        return fb_members, fb_mean
    return member_maxes, round(sum(member_maxes) / len(member_maxes), 1)


# ── NBM percentile fetch ──────────────────────────────────────────────────────
def fetch_nbm_percentiles(city):
    """NBM run-to-run spread, expressed as percentiles of the daily max.

    V5.31.1: p50 from this function is now STORED in settlements.nbm_p50 so it
    can eventually be scored against actual. Until enough settled rows
    accumulate, "NBM runs cold" remains an inference from the ladder offset
    rather than a measurement.

    Note these are percentiles across recent NBM RUNS, not an NBM-native
    probabilistic product. The spread reflects run-to-run disagreement, which
    is a reasonable proxy for uncertainty but is not the same thing.
    """
    station = WETHR_STATIONS.get(city)
    if not station:
        return None
    city_tz = pytz.timezone(CITY_TZ.get(city, 'America/New_York'))
    today_local = datetime.now(city_tz).strftime('%Y-%m-%d')
    now_utc = datetime.utcnow()
    start_utc = (now_utc - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_utc   = (now_utc + timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        r = requests.get(
            'https://wethr.net/api/v2/forecasts.php',
            params={'location_name': station, 'start_valid_time': start_utc,
                    'end_valid_time': end_utc, 'model': 'NBM'},
            headers=WETHR_HEADERS, timeout=15
        )
        if r.status_code != 200:
            return None
        forecasts = r.json()
        if not forecasts or not isinstance(forecasts, list):
            return None
        run_highs = {}
        for f in forecasts:
            valid_time_str = f.get('valid_time', '')
            temp_f = f.get('temperature_f')
            run_time = f.get('run_time', '')
            if temp_f is None:
                continue
            try:
                vt_utc = datetime.strptime(valid_time_str, '%Y-%m-%d %H:%M:%S')
                vt_utc = pytz.utc.localize(vt_utc)
                vt_local = vt_utc.astimezone(city_tz)
                if not (6 <= vt_local.hour <= 21):
                    continue
                if vt_local.strftime('%Y-%m-%d') != today_local:
                    continue
            except Exception:
                continue
            run_highs.setdefault(run_time, []).append(float(temp_f))
        if not run_highs:
            return None
        run_max_temps = sorted([max(t) for t in run_highs.values() if t])
        if len(run_max_temps) < 1:
            return None
        def pct(data, p):
            idx = (p / 100) * (len(data) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(data) - 1)
            return round(data[lo] + (idx - lo) * (data[hi] - data[lo]), 1)
        return {
            'p10': pct(run_max_temps, 10), 'p25': pct(run_max_temps, 25),
            'p50': pct(run_max_temps, 50), 'p75': pct(run_max_temps, 75),
            'p90': pct(run_max_temps, 90),
        }
    except Exception:
        return None


# ── Consensus ────────────────────────────────────────────────────────────────
def choose_sigma(city, obs_high=None, forecast=None):
    s = BASE_SIGMA.get(city, 2.1)
    local_hour = get_local_hour(city)
    s *= 1.00 if local_hour < 11 else 0.94 if local_hour < 14 else 0.90 if local_hour < 16 else 0.86
    if city in DESERT_CITIES:
        s *= 0.92
    if obs_high is not None and forecast is not None:
        gap = abs(forecast - obs_high)
        if gap < 2:   s *= 0.80
        elif gap < 4: s *= 0.90
    return max(1.30, min(2.80, s))


def late_day_floor(fc, obs, local_hour, city=''):
    gap = max(0.0, fc - obs)
    if city in NORTHEAST_CITIES:
        frac = 0.50 if local_hour < 12 else 0.75 if local_hour < 14 else 0.88 if local_hour < 16 else 0.93
    else:
        frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap


def compute_consensus(fc, cur, noaa, city, obs_high=None):
    """V5.30.0 CONSENSUS FLOOR FIX.

    An observed high is a MEASUREMENT, not a forecast. Previously this
    function did `consensus = obs_high` and then added the city warm offset
    on top, and main() added bias_correction on top of that — so Phoenix
    reported consensus 114.35F from a 113.0F floor while every forecast
    input read below 111F.

    Now: decide whether the observed high is trustworthy, but apply it as a
    max() clamp AFTER the warm offset. The offset shapes the forecast
    estimate; the measurement can only raise the result from below, never
    become the base that adjustments stack on. Returns a flag-free value —
    main() re-clamps after bias for the same reason.

    V5.31 NOTE: the value this function returns is now what ANCHORS the
    bracket ladder, not merely what gets logged. Every adjustment here
    reaches bet selection.
    """
    mode = CITY_PREDICTION_MODE.get(city, 'full_blend')
    obs_locked = False

    if mode == 'nws_only':
        consensus = float(fc)
        local_hour = get_local_hour(city)
        obs = noaa if noaa is not None else cur
        if obs_high is not None and obs_high > consensus:
            trusted = True
            if local_hour < OBS_HIGH_TRUST_HOUR:
                trusted = False
            current_for_check = obs if obs is not None else cur
            if current_for_check is not None and obs_high > current_for_check + OBS_HIGH_MAX_OVERSHOOT:
                trusted = False
            if current_for_check is not None and obs_high < current_for_check:
                trusted = False
            if trusted:
                obs_locked = True
    else:
        local_hour = get_local_hour(city)
        is_fc_heavy = city in FORECAST_HEAVY_CITIES
        if is_fc_heavy and local_hour < 10:
            base = fc * 0.95 + (noaa if noaa is not None else cur) * 0.05 if (noaa is not None or cur is not None) else fc
        elif is_fc_heavy and local_hour < 14:
            obs_val = noaa if noaa is not None else cur
            base = fc * 0.90 + obs_val * 0.10 if obs_val is not None else fc
        elif is_fc_heavy and local_hour < 16:
            obs_val = noaa if noaa is not None else cur
            base = fc * 0.75 + obs_val * 0.25 if obs_val is not None else fc
        elif local_hour < 10:
            base = fc * 0.90 + cur * 0.07 + (noaa * 0.03 if noaa is not None else 0) if noaa is not None else fc * 0.93 + cur * 0.07
        elif local_hour < 12:
            base = fc * 0.80 + cur * 0.12 + (noaa * 0.08 if noaa is not None else 0) if noaa is not None else fc * 0.85 + cur * 0.15
        elif local_hour < 14:
            base = fc * 0.65 + cur * 0.18 + noaa * 0.17 if noaa is not None else fc * 0.78 + cur * 0.22
        else:
            base = fc * 0.45 + cur * 0.25 + noaa * 0.30 if noaa is not None else fc * 0.60 + cur * 0.40
        if abs(base - fc) > 4.0:
            base = fc - 4.0 if base < fc else fc + 4.0
        obs = noaa if noaa is not None else cur
        if obs is not None:
            consensus = max(base, late_day_floor(fc, obs, local_hour, city))
        else:
            consensus = base
        if obs_high is not None and obs_high > consensus:
            trusted = True
            if local_hour < OBS_HIGH_TRUST_HOUR:
                trusted = False
            current_for_check = obs if obs is not None else cur
            if current_for_check is not None and obs_high > current_for_check + OBS_HIGH_MAX_OVERSHOOT:
                trusted = False
            if current_for_check is not None and obs_high < current_for_check:
                trusted = False
            if trusted:
                obs_locked = True

    warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
    if warm_offset != 0.0:
        consensus += warm_offset

    # V5.30.0: measurement clamps from below, AFTER the offset.
    if obs_locked:
        consensus = max(consensus, obs_high)

    return consensus


# ── Bracket math ──────────────────────────────────────────────────────────────
def parse_ladder(text):
    if not text:
        return []
    out = []
    for p in text.split('|'):
        p = normalize_label(p)
        nums = [int(x) for x in re.findall(r'\d+', p)]
        if not nums:
            continue
        low = p.lower()
        if 'below' in low:
            out.append((p, None, nums[0]))
        elif 'above' in low:
            out.append((p, nums[0], None))
        elif len(nums) >= 2:
            out.append((p, nums[0], nums[1]))
    return out


def sigma_bracket_prob(mu, lo, hi, sigma, obs_high=None):
    if obs_high is not None and hi is not None and obs_high > hi + 0.4:
        return 0.0
    if lo is None:
        return normal_cdf(hi + 0.5, mu, sigma)
    elif hi is None:
        return 1 - normal_cdf(lo - 0.5, mu, sigma)
    else:
        return normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)


def nbm_bracket_prob(nbm, lo, hi, obs_high=None, anchor=None):
    """Bracket probability from the NBM percentile distribution.

    V5.31: `anchor` (consensus) optionally re-centers the distribution.

    Before V5.31 the ladder was placed entirely by NBM percentiles, and
    consensus only entered when this function returned None for an individual
    bracket. That meant every forecasting improvement — the NWS/GFS blend, the
    per-city warm offsets, the bias correction, the V5.30 trimmed mean —
    affected a number that was logged and stored but never reached bet
    selection. Measured across 72 days and 20 cities: the chosen bracket
    midpoint sat 1-2F BELOW the model's own consensus in 18 of 20 cities, and
    the market beat the model 762-171 on disagreements.

    With NBM_CONSENSUS_ANCHOR = 1.0 the percentile spread — and therefore the
    uncertainty structure — still comes from NBM, but every point is shifted
    so that p50 lands exactly on consensus. Set the constant to 0.0 to restore
    V5.30 behavior exactly.

    The asymmetric tail extrapolation below is UNCHANGED from V5.30: the lower
    tail ramps over a fixed 5F while the upper ramps over (p90 - p75). That is
    a real asymmetry and it affects tail brackets, but it is not what produced
    the center offset, and it is left alone deliberately — one variable at a
    time.
    """
    if not nbm:
        return None
    cdf_points = []
    pct_map = {'p10': 0.10, 'p25': 0.25, 'p50': 0.50, 'p75': 0.75, 'p90': 0.90}
    for key, prob in sorted(pct_map.items(), key=lambda x: x[1]):
        if key in nbm:
            cdf_points.append((nbm[key], prob))
    if len(cdf_points) < 2:
        return None
    cdf_points.sort(key=lambda x: x[0])

    # V5.31: shift the whole distribution so its median lands on consensus.
    if anchor is not None and NBM_CONSENSUS_ANCHOR > 0:
        p50 = nbm.get('p50')
        if p50 is not None:
            try:
                shift = (float(anchor) - float(p50)) * NBM_CONSENSUS_ANCHOR
                cdf_points = [(t + shift, p) for t, p in cdf_points]
            except Exception:
                pass

    def cdf(t):
        if t <= cdf_points[0][0]:
            return max(0.0, cdf_points[0][1] * (t - (cdf_points[0][0] - 5)) / 5)
        if t >= cdf_points[-1][0]:
            remaining = 1.0 - cdf_points[-1][1]
            span = max(cdf_points[-1][0] - cdf_points[-2][0], 1.0)
            return min(1.0, cdf_points[-1][1] + remaining * (t - cdf_points[-1][0]) / span)
        for i in range(len(cdf_points) - 1):
            t0, p0 = cdf_points[i]
            t1, p1 = cdf_points[i + 1]
            if t0 <= t <= t1:
                return p0 + (t - t0) / max(t1 - t0, 0.001) * (p1 - p0)
        return 0.5

    if lo is None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4:
            return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5)))
    elif hi is None and lo is not None:
        return max(0.0, min(1.0, 1.0 - cdf(lo - 0.5)))
    elif lo is not None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4:
            return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5) - cdf(lo - 0.5)))
    return None


def bracket_probs(consensus, ladder_text, city, nbm, obs_high=None, forecast=None):
    used_nbm = bool(nbm and len(nbm) >= 2)
    rows = []
    if used_nbm:
        sigma = choose_sigma(city, obs_high=obs_high, forecast=forecast)
        for label, lo, hi in parse_ladder(ladder_text):
            # V5.31: consensus is now the ANCHOR for the NBM distribution, not
            # merely a fallback when NBM has no answer for this bracket.
            p = nbm_bracket_prob(nbm, lo, hi, obs_high=obs_high, anchor=consensus)
            if p is None:
                p = sigma_bracket_prob(consensus, lo, hi, sigma, obs_high)
            rows.append((label, max(0.0, min(1.0, p))))
    else:
        sigma = choose_sigma(city, obs_high=obs_high, forecast=forecast)
        for label, lo, hi in parse_ladder(ladder_text):
            p = sigma_bracket_prob(consensus, lo, hi, sigma, obs_high)
            rows.append((label, max(0.0, min(1.0, p))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, used_nbm


def ensemble_bracket_prob(members, lo, hi):
    if not members:
        return None
    return sum(1 for m in members
               if (lo is None or m >= lo - 0.5) and (hi is None or m <= hi + 0.5)) / len(members)


def blend_probs(sigma_prob, ensemble_prob, members, city='', nbm_active=False):
    if ensemble_prob is None or members is None:
        return sigma_prob
    base_weight = GFS_CITY_WEIGHT.get(city, 0.20)
    ensemble_weight = base_weight * 0.5 if nbm_active else base_weight
    return round((1.0 - ensemble_weight) * sigma_prob + ensemble_weight * ensemble_prob, 4)


def apply_prob_floor(prob_rows, consensus, ladder_text):
    """V5.30.0 LADDER NORMALIZATION FIX.

    The old version applied a probability floor to near-consensus brackets and
    then rescaled by 1/(1+boost_total), which does NOT normalize. Observed a
    Phoenix ladder summing to 13.9% with four identical 4.1% values — the floor
    had set several brackets to 0.05 and the rescale left the total far from 1.
    Every edge number downstream was computed against a distribution missing
    most of its mass.

    Now: floor as before, then divide by the true total so the ladder sums to
    1.0. Busted brackets (already zeroed by the obs_high check upstream) stay
    at exactly zero and are excluded from redistribution — a bracket the
    observed high has already passed is impossible, not merely unlikely.
    """
    if not prob_rows or consensus is None:
        return prob_rows
    parsed = {lbl: (lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)}
    adjusted = []
    for label, prob in prob_rows:
        lo, hi = parsed.get(label, (None, None))
        if lo is None and hi is None:
            adjusted.append((label, prob))
            continue
        if prob <= 0.0:
            adjusted.append((label, 0.0))
            continue
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
        elif lo is not None:
            mid = lo + 1.0
        else:
            mid = hi - 1.0
        distance = abs(mid - consensus)
        new_prob = prob
        if distance <= 4.0 and prob < 0.05:
            new_prob = 0.05
        elif distance <= 6.0 and prob < 0.02:
            new_prob = 0.02
        adjusted.append((label, new_prob))
    total = sum(p for _, p in adjusted)
    if total <= 0:
        return adjusted
    adjusted = [(lbl, round(p / total, 4)) for lbl, p in adjusted]
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def bracket_contains_consensus(label, consensus, ladder_text, tolerance=1.0):
    if consensus is None:
        return True
    for lbl, lo, hi in parse_ladder(ladder_text):
        if not labels_match(lbl, label):
            continue
        if lo is None and hi is not None:
            return consensus <= hi + tolerance
        if hi is None and lo is not None:
            return consensus >= lo - tolerance
        if lo is not None and hi is not None:
            return (lo - tolerance) <= consensus <= (hi + tolerance)
    return False


def two_degree_call(mu, ladder_text, obs_high=None):
    best_label, best_dist = None, float('inf')
    for label, lo, hi in parse_ladder(ladder_text):
        if obs_high is not None and hi is not None and obs_high > hi + 0.4:
            continue
        mid = (hi - 1.0 if lo is None and hi is not None else
               lo + 1.0 if hi is None and lo is not None else
               (lo + hi) / 2 if lo is not None and hi is not None else None)
        if mid is None:
            continue
        dist = abs(mid - mu)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


# ── Kalshi market fetch ──────────────────────────────────────────────────────
def get_eastern_datetime():
    return datetime.now(pytz.timezone('America/New_York'))


def get_event_ticker(series):
    # V5.28.4 fix: Kalshi uses YY-MON-DD format (e.g., '26JUN15'), not DD-MON-YY.
    return series + '-' + get_eastern_datetime().strftime('%y%b%d').upper()


def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = normalize_label((m.get(field) or '').replace('\u00b0', '').replace('deg', '').strip())
        if s:
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below: return below.group(1)+' or below', int(below.group(1))-10000
            if above: return above.group(1)+' or above', int(above.group(1))+10000
            if rng:   return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
    title = (m.get('title') or '').replace('\u00b0', '').replace('**', '').replace('deg', '')
    if title:
        ma = re.search(r'be\s*[>=]+\s*(\d+)', title, re.I)
        if ma: n = int(ma.group(1)); return str(n)+' or above', n+10000
        mb = re.search(r'be\s*[<=]+\s*(\d+)', title, re.I)
        if mb: n = int(mb.group(1)); return str(n)+' or below', n-10000
        mr = re.search(r'be\s*(\d+)\s*(?:to|-)\s*(\d+)', title, re.I)
        if mr:
            lo, hi = int(mr.group(1)), int(mr.group(2))
            return str(lo)+'-'+str(hi), lo
        nums = re.findall(r'\d+', title)
        if len(nums) >= 2:
            lo, hi = int(nums[-2]), int(nums[-1])
            if 0 < hi-lo <= 5:
                return str(lo)+'-'+str(hi), lo
    cap, floor_s = m.get('cap_strike'), m.get('floor_strike')
    if cap is not None and floor_s is not None:
        try:
            lo, hi = int(float(floor_s)), int(float(cap))
            return str(lo)+'-'+str(hi), lo
        except Exception:
            pass
    if cap is not None:
        try:
            n = int(float(cap))
            return str(n)+' or below', n-10000
        except Exception:
            pass
    return None, None


def get_price_cents(m):
    yes_ask = no_ask = None
    for f in ['yes_ask_dollars', 'yes_bid_dollars']:
        v = m.get(f)
        if v:
            try: yes_ask = round(float(v)*100); break
            except Exception: pass
    for f in ['no_ask_dollars', 'no_bid_dollars']:
        v = m.get(f)
        if v:
            try: no_ask = round(float(v)*100); break
            except Exception: pass
    if yes_ask is None:
        raw = m.get('yes_ask') or m.get('yes_bid')
        if raw is not None:
            try: yes_ask = int(raw)
            except Exception: pass
    if no_ask is None:
        raw = m.get('no_ask') or m.get('no_bid')
        if raw is not None:
            try: no_ask = int(raw)
            except Exception: pass
    return yes_ask, no_ask


def fetch_kalshi_brackets(series, city=''):
    """Fetch and parse Kalshi market ladder for a city."""
    url = 'https://api.elections.kalshi.com/trade-api/v2/markets'
    event_ticker = get_event_ticker(series)
    today_date = get_eastern_date()

    today_et_dt = get_eastern_datetime()
    today_kalshi_fmt = today_et_dt.strftime('%y%b%d').upper()  # '26JUN15'
    tomorrow_utc_date = (today_et_dt + timedelta(days=1)).strftime('%Y-%m-%d')

    def _try(params):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    endpoint_used = None
    data = _try({'event_ticker': event_ticker, 'limit': 30})
    if data and data.get('markets'):
        endpoint_used = 'event_ticker'
    else:
        data = _try({'series_ticker': series, 'status': 'open', 'limit': 30})
        if data and data.get('markets'):
            endpoint_used = 'series_status_open'
        else:
            data = _try({'series_ticker': series, 'limit': 30})
            if data and data.get('markets'):
                endpoint_used = 'series_only'
    if not data or not data.get('markets'):
        return None

    all_markets = data['markets']
    raw_market_count = len(all_markets)

    markets = [m for m in all_markets if today_kalshi_fmt in (m.get('event_ticker') or '').upper()]
    if not markets:
        markets = [m for m in all_markets if (m.get('close_time') or '').startswith(tomorrow_utc_date)]
    if not markets:
        markets = [m for m in all_markets if today_kalshi_fmt in (m.get('ticker') or '').upper()]
    if not markets:
        print(f'    ⚠️ [{series}] No today-match ({today_kalshi_fmt}/{tomorrow_utc_date}) — falling back to all {len(all_markets)} markets')
        markets = all_markets

    filtered_count = len(markets)

    parsed = []
    seen_labels = set()
    duplicates_dropped = 0
    for m in markets:
        label, key = parse_market_label(m)
        if label is None:
            continue
        norm_label = normalize_label(label)
        if norm_label in seen_labels:
            duplicates_dropped += 1
            continue
        seen_labels.add(norm_label)
        yes_ask, no_ask = get_price_cents(m)
        parsed.append((key, label, yes_ask, no_ask))

    if duplicates_dropped > 0:
        print(f'    ⚠️ [{series}] Dropped {duplicates_dropped} duplicate bracket(s) after dedup')

    print(f'    📋 [{series}] endpoint={endpoint_used}, raw={raw_market_count}, '
          f'filtered={filtered_count}, final={len(parsed)}')
    dump_raw_kalshi_response(
        city=city or series,
        series=series,
        raw_data=data,
        endpoint_used=endpoint_used,
        market_count=raw_market_count,
        filtered_count=filtered_count,
        final_count=len(parsed),
    )

    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[0])
    return [(label, yes_ask, no_ask) for _, label, yes_ask, no_ask in parsed]


def get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N):
    if not kalshi_markets:
        return []
    priced = [(label, yes_ask) for label, yes_ask, _ in kalshi_markets if yes_ask is not None]
    if not priced:
        return []
    priced.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in priced[:n]]


# ── Trust score ──────────────────────────────────────────────────────────────
try:
    from trust_score import SignalInputs, compute_trust_score, bracket_midpoint_from_label
    _TRUST_AVAILABLE = True
except Exception as e:
    print(f'⚠️ Could not import trust_score module: {e}')
    print('   Paper-bet validator will be DISABLED for this run (predictions still fetch).')
    _TRUST_AVAILABLE = False


def compute_row_trust(city, bracket_label, direction, model_pct, ensemble_tier,
                      two_degree_call_str, mae_color, nbm_active,
                      nws_forecast_f, gfs_ensemble_f, bias_adj_f):
    if not _TRUST_AVAILABLE:
        return None
    try:
        inp = SignalInputs(
            city=str(city or ''),
            bracket_label=str(bracket_label or ''),
            direction=str(direction or 'YES'),
            two_degree_call=str(two_degree_call_str or ''),
            bracket_midpoint=bracket_midpoint_from_label(bracket_label),
            twodc_midpoint=bracket_midpoint_from_label(two_degree_call_str),
            model_pct=float(model_pct or 0),
            edge_cents=0.0,
            ensemble_tier=str(ensemble_tier or ''),
            mae_color=str(mae_color or 'green'),
            nbm_active=bool(nbm_active),
            nws_forecast_f=float(nws_forecast_f) if nws_forecast_f is not None else None,
            gfs_ensemble_f=float(gfs_ensemble_f) if gfs_ensemble_f is not None else None,
            bias_adj_f=float(bias_adj_f or 0),
        )
        result = compute_trust_score(inp)
        return result.composite if result else None
    except Exception as e:
        print(f'    ⚠️ Trust score compute failed for {city}/{bracket_label}: {e}')
        return None


def ensemble_tier_from_prob(prob):
    if prob is None:
        return ''
    if prob >= 0.80 or prob <= 0.20:
        return 'HIGH'
    if prob >= 0.65 or prob <= 0.35:
        return 'MED'
    return 'LOW'


# ── Auto-Settlement ──────────────────────────────────────────────────────────
_CLI_CACHE = {}


def fetch_cli_max_temp(city, target_date_str):
    station = CLI_STATIONS.get(city)
    if not station:
        return None
    year = target_date_str[:4]
    cache_key = station + '_' + year
    if cache_key not in _CLI_CACHE:
        try:
            url = ('https://mesonet.agron.iastate.edu/json/cli.py'
                   '?station=' + station + '&year=' + year)
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            lookup = {}
            for entry in data.get('results', []):
                valid = entry.get('valid', '')
                high = entry.get('high')
                if valid and high is not None:
                    try:
                        lookup[valid] = float(high)
                    except Exception:
                        pass
            _CLI_CACHE[cache_key] = lookup
        except Exception:
            return None
    return _CLI_CACHE.get(cache_key, {}).get(target_date_str)


def bracket_hits(actual_temp, lo, hi):
    if actual_temp is None:
        return None
    rounded = int(math.floor(float(actual_temp) + 0.5))
    if lo is None and hi is not None:
        return rounded <= hi
    if hi is None and lo is not None:
        return rounded >= lo
    if lo is not None and hi is not None:
        return lo <= rounded <= hi
    return None


def sb_fetch_unsettled_settlements():
    try:
        r = requests.get(
            sb_url('settlements'),
            headers=sb_headers(),
            params={'actual': 'is.null', 'order': 'date.asc'},
            timeout=15,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def sb_update_settlement_actual(row_id, actual, error):
    try:
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(row_id),
            headers=sb_headers(),
            json={'actual': round(actual, 2), 'error': error},
            timeout=10,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def sb_fetch_pending_bets():
    try:
        r = requests.get(
            sb_url('bets'),
            headers=sb_headers(),
            params={'or': '(result.eq.Pending,result.is.null)',
                    'order': 'date.asc'},
            timeout=15,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def sb_update_bet_settle(bet_id, updates):
    try:
        r = requests.patch(
            sb_url('bets') + '?id=eq.' + str(bet_id),
            headers=sb_headers(),
            json=updates,
            timeout=10,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def run_settlement_pass():
    print(f'\n=== Settlement Pass ===')
    try:
        today = get_eastern_date()
        unsettled = sb_fetch_unsettled_settlements()
        if not unsettled:
            print('  No unsettled settlement rows.')
        else:
            settled_now = []
            for row in unsettled:
                row_date = row.get('date', '')
                if not row_date or row_date >= today:
                    continue
                city = row.get('city')
                if not city:
                    continue
                actual = fetch_cli_max_temp(city, row_date)
                if actual is None:
                    continue
                consensus = row.get('consensus')
                error = round(actual - consensus, 2) if consensus is not None else None
                if sb_update_settlement_actual(row['id'], actual, error):
                    settled_now.append({'city': city, 'date': row_date, 'actual': actual})
            if settled_now:
                print(f'  Settled {len(settled_now)} prediction(s):')
                for s in settled_now:
                    print(f"    + {s['city']} {s['date']} → actual {s['actual']}°F")
            else:
                print(f'  No new predictions settleable yet '
                      f'({len(unsettled)} pending, CLI data not available).')

        pending_bets = sb_fetch_pending_bets()
        if not pending_bets:
            print('  No pending bets.')
            return

        try:
            r = requests.get(
                sb_url('settlements'),
                headers=sb_headers(),
                params={'actual': 'not.is.null', 'order': 'date.desc',
                        'limit': '500'},
                timeout=15,
            )
            settled_rows = r.json() if r.status_code == 200 else []
        except Exception:
            settled_rows = []

        actuals_map = {}
        for row in settled_rows:
            key = (row.get('city'), row.get('date'))
            actuals_map[key] = row.get('actual')

        settled_bet_count = 0
        won_count = 0
        lost_count = 0
        settled_ts = datetime.now(pytz.timezone('America/New_York')).isoformat()

        for bet in pending_bets:
            bet_city = bet.get('city')
            bet_date = bet.get('date')
            actual = actuals_map.get((bet_city, bet_date))
            if actual is None:
                continue
            bracket = bet.get('bracket') or bet.get('bracket_label')
            if not bracket:
                continue
            lo, hi = label_to_numeric_key(bracket)
            hit = bracket_hits(actual, lo, hi)
            if hit is None:
                continue

            direction = (bet.get('direction') or 'YES').upper()
            won = hit if direction == 'YES' else (not hit)

            amount = float(bet.get('amount', 0) or 0)
            price = float(bet.get('price', 0) or 0)
            if won and price > 0:
                profit = round(amount * (100.0 - price) / price, 2)
                payout = round(amount + profit, 2)
            else:
                profit = round(-amount, 2)
                payout = 0.0

            updates = {
                'result': 'Won' if won else 'Lost',
                'profit': profit,
                'payout': payout,
                'actual': round(float(actual), 2),
                'settled_at': settled_ts,
            }
            if sb_update_bet_settle(bet['id'], updates):
                settled_bet_count += 1
                if won:
                    won_count += 1
                else:
                    lost_count += 1

        if settled_bet_count:
            print(f'  Settled {settled_bet_count} paper bet(s): '
                  f'{won_count} won, {lost_count} lost.')
        else:
            print(f'  No paper bets settleable yet '
                  f'({len(pending_bets)} pending).')

    except Exception as e:
        print(f'  ⚠️ Settlement pass error (non-fatal): {e}')


# ── Paper-Bet Validator (V5.28: NO blocked on top bracket) ───────────────────
def is_window_time(utc_now):
    firing = []
    for (h, m), tz_window_list in WINDOW_SCHEDULE.items():
        window_dt = utc_now.replace(hour=h, minute=m, second=0, microsecond=0)
        delta_min = abs((utc_now - window_dt).total_seconds()) / 60.0
        if delta_min <= WINDOW_TOLERANCE_MIN:
            firing.extend(tz_window_list)
    return firing


def evaluate_city_for_paper_bet(city, weather_data, consensus, bias_correction,
                                window_tz='', window_label='', utc_now=None):
    nws_fc = weather_data.get('nws_fc')
    obs_high = weather_data.get('obs_high')
    ensemble_members = weather_data.get('ensemble_members')
    ensemble_mean = weather_data.get('ensemble_mean')
    nbm = weather_data.get('nbm')

    series = SERIES.get(city)
    if not series:
        return None
    kalshi_markets = fetch_kalshi_brackets(series, city=city)
    if not kalshi_markets or len(kalshi_markets) < 2:
        return None

    valid_yes_prices = [m[1] for m in kalshi_markets if m[1] is not None]
    sigma_p = round(sum(valid_yes_prices) / 100.0, 4) if valid_yes_prices else None

    ladder_text = ' | '.join(normalize_label(m[0]) for m in kalshi_markets)

    prob_rows, used_nbm = bracket_probs(consensus, ladder_text, city, nbm,
                                         obs_high=obs_high, forecast=nws_fc)
    prob_rows = apply_prob_floor(prob_rows, consensus, ladder_text)
    if not prob_rows:
        return None

    model_pick_label = prob_rows[0][0]
    target_base_prob = prob_rows[0][1]

    if utc_now is not None:
        snapshot_kalshi_market(
            city=city,
            kalshi_markets=kalshi_markets,
            window_tz=window_tz,
            window_label=window_label,
            model_top_bracket=model_pick_label,
            utc_now=utc_now,
        )

    if sigma_p is not None and sigma_p > SIGMA_P_GATE_MAX:
        return {
            'gate1_pass': False,
            'sigma_p_pass': False,
            'sigma_p': sigma_p,
            'reason': f'Σp gate fail: {sigma_p:.3f} > {SIGMA_P_GATE_MAX} (ladder overpriced)',
        }

    market_top = get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N)
    in_top = any(labels_match(mt, model_pick_label) for mt in market_top)
    if not in_top:
        return {
            'gate1_pass': False,
            'sigma_p_pass': True,
            'sigma_p': sigma_p,
            'reason': f'Gate 1 fail: model pick "{model_pick_label}" not in market top {CONSENSUS_TOP_N}: {market_top}',
        }

    target_market = next((m for m in kalshi_markets if labels_match(m[0], model_pick_label)), None)
    if not target_market:
        return None
    yes_ask, no_ask = target_market[1], target_market[2]

    bracket_lo = bracket_hi = None
    for lbl, lo, hi in parse_ladder(ladder_text):
        if labels_match(lbl, model_pick_label):
            bracket_lo, bracket_hi = lo, hi
            break

    ens_prob = ensemble_bracket_prob(ensemble_members, bracket_lo, bracket_hi) if ensemble_members else None
    final_prob = blend_probs(target_base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)

    busted = obs_high is not None and bracket_hi is not None and obs_high > bracket_hi + 0.4

    contains_consensus = bracket_contains_consensus(model_pick_label, consensus, ladder_text, tolerance=1.0)

    ens_tier = ensemble_tier_from_prob(ens_prob) if ens_prob is not None else ''
    call_str = two_degree_call(consensus, ladder_text, obs_high=obs_high)
    mae_color = get_city_mae_color(city)

    trust_yes = compute_row_trust(
        city=city, bracket_label=model_pick_label, direction='YES',
        model_pct=final_prob * 100, ensemble_tier=ens_tier,
        two_degree_call_str=call_str or '', mae_color=mae_color,
        nbm_active=used_nbm, nws_forecast_f=nws_fc,
        gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
    )
    trust_no = compute_row_trust(
        city=city, bracket_label=model_pick_label, direction='NO',
        model_pct=(1.0 - final_prob) * 100, ensemble_tier=ens_tier,
        two_degree_call_str=call_str or '', mae_color=mae_color,
        nbm_active=used_nbm, nws_forecast_f=nws_fc,
        gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
    )

    return {
        'gate1_pass': True,
        'sigma_p_pass': True,
        'sigma_p': sigma_p,
        'bracket': model_pick_label,
        'yes_ask': yes_ask,
        'no_ask': no_ask,
        'final_prob': final_prob,
        'busted': busted,
        'contains_consensus': contains_consensus,
        'trust_yes': trust_yes,
        'trust_no': trust_no,
        'used_nbm': used_nbm,
    }


def log_paper_bets_for_window(tz_key, window_label, weather_results, run_iso):
    """V5.29.A Σp gate + V5.28 NO-bet block, both carried forward unchanged.

    V5.31: strategy tags use PAPER_TAG_PREFIX (now 'V531'). Rows written under
    the old 'V530' prefix were produced by a different bracket-selection model
    and must be analyzed separately, never pooled with these.
    """
    cities = TZ_CITIES.get(tz_key, [])
    today = get_eastern_date()
    logged = []
    no_blocks = 0
    sigma_p_blocks = 0
    utc_now = datetime.utcnow()

    for city in cities:
        wx_data = weather_results.get(city)
        if not wx_data or not wx_data.get('ok'):
            continue
        consensus = wx_data.get('consensus')
        bias_correction = wx_data.get('bias_correction', 0.0)
        if consensus is None:
            continue

        eval_result = evaluate_city_for_paper_bet(
            city, wx_data, consensus, bias_correction,
            window_tz=tz_key, window_label=window_label, utc_now=utc_now,
        )
        if eval_result is None:
            continue
        if eval_result.get('sigma_p_pass') is False:
            sigma_p_blocks += 1
            sp = eval_result.get('sigma_p')
            print(f'    🚫 V5.29.A Σp gate: {city} skipped (Σp={sp:.3f} > {SIGMA_P_GATE_MAX})')
            continue
        if not eval_result.get('gate1_pass'):
            continue

        bracket = eval_result['bracket']
        yes_ask = eval_result['yes_ask']
        no_ask  = eval_result['no_ask']
        final_prob = eval_result['final_prob']
        busted = eval_result['busted']
        contains_consensus = eval_result['contains_consensus']
        trust_yes = eval_result['trust_yes']
        trust_no  = eval_result['trust_no']
        sigma_p   = eval_result.get('sigma_p')

        yes_qualifies = (
            yes_ask is not None
            and yes_ask >= PRICE_FLOOR_CENTS
            and yes_ask < 99
            and not busted
            and contains_consensus
            and final_prob >= 0.10
        )

        no_qualifies = False
        if no_ask is not None and no_ask < 99:
            if busted and no_ask <= 5:
                no_qualifies = True
            elif (1.0 - final_prob) >= 0.10 and not busted:
                no_blocks += 1

        if yes_qualifies and trust_yes is not None:
            for threshold in TRUST_THRESHOLDS:
                if trust_yes >= threshold:
                    tag = f'{PAPER_TAG_PREFIX}_PAPER_YES_{window_label}_T{threshold}'
                    if sb_count_paper_bets_today(city, tag) > 0:
                        continue
                    sp_str = f' | Σp {sigma_p:.3f}' if sigma_p is not None else ''
                    bet_row = {
                        'date': today, 'city': city, 'bracket': bracket,
                        'direction': 'YES', 'price': yes_ask, 'amount': PAPER_BET_STAKE,
                        'result': 'Pending', 'profit': 0.0, 'payout': 0.0,
                        'actual': None, 'strategy_tag': tag,
                        'notes': f'Trust 🎯 {trust_yes:.1f} | Model {final_prob*100:.1f}%{sp_str} | {tz_key} {window_label}',
                        'placed_at': run_iso,
                    }
                    if sb_insert_paper_bet(bet_row):
                        logged.append(f'{city} {bracket} YES @ {yes_ask}c [{tag}] Trust {trust_yes:.1f}')

        if no_qualifies and trust_no is not None:
            for threshold in TRUST_THRESHOLDS:
                if trust_no >= threshold:
                    tag = f'{PAPER_TAG_PREFIX}_PAPER_NO_{window_label}_T{threshold}'
                    if sb_count_paper_bets_today(city, tag) > 0:
                        continue
                    bust_note = ' [BUSTED]' if busted else ''
                    sp_str = f' | Σp {sigma_p:.3f}' if sigma_p is not None else ''
                    bet_row = {
                        'date': today, 'city': city, 'bracket': bracket,
                        'direction': 'NO', 'price': no_ask, 'amount': PAPER_BET_STAKE,
                        'result': 'Pending', 'profit': 0.0, 'payout': 0.0,
                        'actual': None, 'strategy_tag': tag,
                        'notes': f'Trust 🎯 {trust_no:.1f} | Model {(1.0-final_prob)*100:.1f}%{bust_note}{sp_str} | {tz_key} {window_label}',
                        'placed_at': run_iso,
                    }
                    if sb_insert_paper_bet(bet_row):
                        logged.append(f'{city} {bracket} NO @ {no_ask}c [{tag}] Trust {trust_no:.1f}')

    if no_blocks > 0:
        print(f'    🚫 V5.28: blocked {no_blocks} NO bet(s) on model top brackets (calibration fix)')
    if sigma_p_blocks > 0:
        print(f'    🚫 V5.29.A: skipped {sigma_p_blocks} city/cities (Σp > {SIGMA_P_GATE_MAX})')

    return logged


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = get_eastern_date()
    now_et = datetime.now(pytz.timezone('America/New_York'))
    utc_now = datetime.utcnow()

    print(f'\n=== V5.31.1 Weather Fetch Run ===')
    print(f'Date: {today} | ET: {now_et.strftime("%I:%M %p ET")} | UTC: {utc_now.strftime("%H:%M")}')
    print(f'Cities: {len(CITIES)} | NBM anchor: {NBM_CONSENSUS_ANCHOR} | Paper tag: {PAPER_TAG_PREFIX}_\n')

    firing_windows = is_window_time(utc_now)
    if firing_windows:
        print(f'🎯 Paper-bet window(s) firing this run: {firing_windows}\n')
    else:
        print('(No paper-bet window active this run — predictions only, NBM not fetched)\n')

    weather_results = {}
    save_ok_count = 0
    save_fail = []

    for city in CITIES.keys():
        print(f'  [{city}]')
        try:
            nws_fc = fetch_nws_forecast(city)
            print(f'    NWS forecast: {nws_fc}F')
            if nws_fc is None:
                print(f'    ⚠️ No NWS forecast — skipping {city}')
                weather_results[city] = {'ok': False, 'reason': 'no_nws'}
                save_fail.append(city)
                continue

            current_temp = fetch_current_temp(city)
            obs_high_raw = fetch_obs_high(city)
            ensemble_members, ensemble_mean = fetch_gfs_ensemble(city)
            nbm = None
            if firing_windows:
                nbm = fetch_nbm_percentiles(city)
            print(f'    Current: {current_temp}F | Obs high: {obs_high_raw}F | '
                  f'GFS: {ensemble_mean}F | NBM: {"yes" if nbm else "no"}')

            obs_high = obs_high_raw
            if (obs_high_raw is not None and current_temp is not None
                    and obs_high_raw > current_temp + OBS_HIGH_OVER_CURRENT_THRESHOLD):
                print(f'    ⚠️ Obs high discarded — {obs_high_raw}F is '
                      f'{obs_high_raw - current_temp:.1f}F above current ({current_temp}F)')
                obs_high = None
            elif (obs_high_raw is not None and nws_fc is not None
                    and obs_high_raw > nws_fc + OBS_HIGH_OVER_FORECAST_THRESHOLD):
                print(f'    ⚠️ Obs high discarded — {obs_high_raw}F is '
                      f'{obs_high_raw - nws_fc:.1f}F above NWS forecast')
                obs_high = None

            if (ensemble_mean is not None and nws_fc is not None
                    and abs(ensemble_mean - nws_fc) > 8.0):
                print(f'    ⚠️ GFS discarded — {abs(ensemble_mean - nws_fc):.1f}F gap from NWS')
                ensemble_members = None
                ensemble_mean = None

            source_gap = None
            high_uncertainty = False
            if nws_fc is not None and ensemble_mean is not None:
                source_gap = abs(nws_fc - ensemble_mean)
                high_uncertainty = source_gap > 5.0

            bias_correction, bias_n = compute_bias_correction(city)
            print(f'    Bias: {bias_correction:+.2f}F (n={bias_n})')

            cur = current_temp if current_temp is not None else nws_fc
            consensus_raw = compute_consensus(nws_fc, cur, current_temp, city, obs_high=obs_high)

            # V5.30.0: if compute_consensus locked onto the observed high, the
            # bias correction must not inflate a measurement. Re-clamp after.
            _obs_locked = (obs_high is not None and abs(consensus_raw - obs_high) < 0.05)
            consensus = round(consensus_raw + bias_correction, 1)
            if _obs_locked:
                consensus = round(max(consensus, obs_high), 1)

            warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
            lock_note = ' [obs-locked]' if _obs_locked else ''
            print(f'    Consensus: {consensus}F (raw={consensus_raw:.1f}, '
                  f'bias={bias_correction:+.2f}, offset={warm_offset:+.1f}){lock_note}')

            # V5.29.D: ensemble-aware consensus correction.
            if ensemble_mean is not None and nws_fc is not None:
                gap = ensemble_mean - consensus
                nws_ensemble_gap = abs(ensemble_mean - nws_fc)
                obs_locked = (obs_high is not None and abs(consensus - obs_high) < 0.1)
                ensemble_wildly_off = nws_ensemble_gap > 8.0
                if abs(gap) > 3.0 and not obs_locked and not ensemble_wildly_off:
                    adjustment = max(-2.0, min(2.0, 0.5 * gap))
                    new_consensus = round(consensus + adjustment, 1)
                    print(f'    🎯 V5.29.D consensus shift: ensemble={ensemble_mean}F vs '
                          f'consensus={consensus}F (gap={gap:+.1f}F) → {new_consensus}F '
                          f'(adj={adjustment:+.1f}F)')
                    consensus = new_consensus

            # V5.31: log the anchor shift so the effect is visible in run output.
            nbm_p50 = nbm.get('p50') if nbm else None
            if nbm_p50 is not None and NBM_CONSENSUS_ANCHOR > 0:
                _shift = round((consensus - float(nbm_p50)) * NBM_CONSENSUS_ANCHOR, 1)
                print(f'    🎯 V5.31 NBM anchor: p50={nbm_p50}F → consensus={consensus}F '
                      f'(shift {_shift:+.1f}F, weight {NBM_CONSENSUS_ANCHOR})')

            ok = sb_upsert(
                city=city, consensus=consensus, forecast=nws_fc,
                ensemble_mean=ensemble_mean, source_gap=source_gap,
                high_uncertainty=high_uncertainty, obs_high=obs_high,
                bias_correction=bias_correction, nbm_p50=nbm_p50,
            )
            print(f'    {"✅ Saved" if ok else "❌ Save failed"}')

            if ok:
                save_ok_count += 1
            else:
                save_fail.append(city)

            weather_results[city] = {
                'ok': ok, 'consensus': consensus, 'bias_correction': bias_correction,
                'nws_fc': nws_fc, 'current_temp': current_temp, 'obs_high': obs_high,
                'ensemble_members': ensemble_members, 'ensemble_mean': ensemble_mean,
                'nbm': nbm,
            }

        except Exception as e:
            print(f'    ❌ Error: {e}')
            weather_results[city] = {'ok': False, 'reason': str(e)}
            save_fail.append(city)

        time.sleep(0.3)

    print(f'\n=== Predictions Summary ===')
    print(f'Saved {save_ok_count}/{len(CITIES)} cities')
    if save_fail:
        print(f'Failed: {", ".join(save_fail)}')

    if firing_windows and _TRUST_AVAILABLE:
        print(f'\n=== Paper-Bet Validator (V5.31.1) ===')
        run_iso = datetime.now(pytz.timezone('America/New_York')).isoformat()
        all_logged = []
        for tz_key, window_label in firing_windows:
            print(f'\n  Window: {tz_key} {window_label}')
            logged = log_paper_bets_for_window(tz_key, window_label, weather_results, run_iso)
            if logged:
                print(f'    Logged {len(logged)} paper bet(s):')
                for line in logged:
                    print(f'      + {line}')
                all_logged.extend(logged)
            else:
                print(f'    No paper bets qualified (Gate 1/2/Trust thresholds blocked all candidates)')

        print(f'\nTotal paper bets logged this run: {len(all_logged)}')
    elif firing_windows and not _TRUST_AVAILABLE:
        print('\n⚠️ Paper-bet window firing but trust_score module unavailable — SKIPPED')

    run_settlement_pass()

    if save_fail:
        print(f'\n⚠️ Partial failure: {len(save_fail)} of {len(CITIES)} cities did not save.')
        print('   (Run completes successfully — predictions for working cities are live.)')
    else:
        print(f'\n✅ All {len(CITIES)} cities saved successfully.')


if __name__ == '__main__':
    main()
