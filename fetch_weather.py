"""
fetch_weather.py — V6.0 CONSENSUS WRITER

WHAT THIS FILE IS NOW
=====================
It writes ONE NUMBER per city per day to settlements.consensus, and settles
yesterday's rows against Iowa State CLI. That is all. It places no bets, picks
no brackets, computes no trust score, and has no gates.

WHAT WAS DELETED, AND WHY
=========================
V5.31.1 was 2,091 lines. It placed ~40 paper bets a day across four tags. All
four tags were negative on 205 settled bets. The deletions below are not
cleanup — each one is a thing that was measured and found to destroy value.

  - NBM PERCENTILE LADDER (nbm_bracket_prob, fetch_nbm_percentiles)
    The ladder was built entirely from NBM percentiles. Measured on 626
    reconstructed city-days, comparing the same days at the same prices:

        naked consensus picks the bracket : 74.3% win, +7.75c net
        the model's actual pick           : 55.2% win, -10.96c net

    Nineteen points, same days, same prices. The forecast was good and the
    machinery downstream threw it away. V5.31's anchor narrowed the gap
    (bracket offset -0.62 -> -0.25) and the tags still lost.

    NBM itself was also measured: 1.85F cold with 2.80 MAE against consensus's
    0.82, on 54 city-days, 16 of 18 cities. It was the weakest input in the
    file and it was placing every bet.

  - SIGMA-CDF FALLBACK (choose_sigma, sigma_bracket_prob, BASE_SIGMA)
    Only reached when NBM returned None. With no ladder there is nothing to
    fall back to. NOTE: the per-city sigma constants were shown to be wrong in
    a specific way worth remembering — measured 2026-09-07 on 368 city-days,
    consensus MAE runs 0.73F on a city's clearest third of days and 1.10F on
    its cloudiest third, terciled WITHIN each city. Uncertainty is not a
    per-city constant. If a distribution is ever rebuilt, that is the finding
    to build it on, not BASE_SIGMA.

  - TRUST SCORE (compute_row_trust, TRUST_THRESHOLDS, trust_score import)
    The T75/T80 tags were a strict subset of each other by construction — the
    loop wrote a row at every threshold the score cleared, so T80 rows were
    also T75 rows. Reading them as independent confirmation double-counted.
    Neither tier separated winners from losers.

  - THE THREE GATES (Sigma-p, market-consensus, price floor)
    Gate 1 required the model pick to be in the market's top 2, which is close
    to requiring it to BE the market's favorite — which is what FAV V1 buys
    directly, without a forecast.

  - PAPER BET LOGGING (log_paper_bets_for_window, evaluate_city_for_paper_bet,
    WINDOW_SCHEDULE, TZ_CITIES, PAPER_TAG_PREFIX)
    205 bets, four tags, all negative.

  - KALSHI ENTIRELY (fetch_kalshi_brackets, snapshot_kalshi_market,
    dump_raw_kalshi_response, parse_market_label, get_price_cents)
    This file no longer touches a market. kalshi_snapshots stops being written
    here; the 626 city-days already collected are what the analyses above ran
    on and they remain in the table.

WHAT SURVIVED, AND WHY
======================
  - compute_consensus() and its bias correction. Consensus runs 0.82F MAE.
    That is the good part and it is the only part with a measured edge.
  - The CLI settlement pass. It fills settlements.actual, which is what every
    accuracy measurement in this project is scored against.
  - sb_upsert. FAV V1.1 READS settlements.consensus for its agreement tag:

        window     FAVORITE (all)        AGREE (consensus confirms)
        MORNING    n=165  76.4%  +9.83   n=51  80.4%  +14.69
        MIDDAY     n=186  65.1%  -1.02   n=50  80.0%  +13.56
        AFTERNOON  n=234  71.4%  -0.73   n=71  77.5%   +5.32

    Midday and afternoon are NEGATIVE unfiltered and POSITIVE filtered. That
    is what consensus is for now — a filter on bets FAV V1 already places, not
    a bracket picker. It is the reason this file still exists.

⚠️ TIMING MATTERS. FAV V1's MORNING window fires at 14:30 UTC and reads
settlements.consensus. This file's cron fires at 14:00 UTC and rows land
14:01-14:09. That is ~22 minutes of margin. If this file runs late or fails,
FAV V1 logs the bet anyway with a NULL agreement tag — never blocking a bet on
a missing forecast was deliberate — but the tag is lost for that window.

⚠️ 18 CITIES. San Francisco and Seattle were dropped in V5.30 for forecast
quality (SF ran -2.14F residual bias, 4x worse than any other city). FAV V1
covers 20, so those two permanently carry NULL agreement tags. Expected.

SIGN CONVENTION — read before interpreting any bias number.
    errors = actual - consensus
    POSITIVE means settlement came in WARMER than predicted, i.e. the model
    runs COLD. Misreading this caused a wrong call on 2026-08-24 and it is why
    the convention is written in three places.

HISTORY WORTH KEEPING
=====================
V5.30.0 CONSENSUS FLOOR FIX. An observed high is a MEASUREMENT, not a forecast
base. compute_consensus() used to do `consensus = obs_high` then add the city
warm offset on top, and main() added bias on top of THAT — Phoenix showed
114.35F from a 113.0F floor while every forecast input read below 111F. The
floor now clamps from below via max(), applied AFTER offset and AFTER bias.
Both clamp sites are preserved below. Do not "simplify" them back.

V5.30.0 BIAS: median -> trimmed mean. The median discarded the systematic
middle of a skewed error distribution; residual bias stayed positive in 17 of
18 cities. Trimmed mean keeps outlier robustness without compressing the
drift. Verified 2026-08-29: residuals now +1.23 to -0.69, mean ~+0.18.

V5.29.F HOUSTON COORDINATES. CITIES['Houston'] was Bush Intercontinental
(KIAH). Kalshi settles Houston on HOBBY (KHOU). Settlement was right but every
forecast call pulled the wrong airport. Fixed to 29.6459/-95.2769. The same
trap exists for Chicago — KMDW (Midway), not KORD.

Secrets: WETHR_API_KEY, SUPABASE_URL, SUPABASE_KEY.
"""

import math
import os
import re
import requests
import time
from datetime import datetime, timedelta

import pytz

# ── Credentials ──────────────────────────────────────────────────────────────
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
SUPABASE_URL  = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY  = os.environ.get('SUPABASE_KEY', '')

WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS       = {'User-Agent': 'kalshi-weather-fetcher/6.0', 'Accept': 'application/json'}

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
    'Dallas':        {'lat': 32.8998, 'lon':  -97.0403},
    'Austin':        {'lat': 30.1945, 'lon':  -97.6699},
    'Houston':       {'lat': 29.6459, 'lon':  -95.2769},  # KHOU Hobby, NOT Bush
    'Atlanta':       {'lat': 33.6407, 'lon':  -84.4277},
    'Miami':         {'lat': 25.7959, 'lon':  -80.2870},
    'New York':      {'lat': 40.7812, 'lon':  -73.9665},
    'San Antonio':   {'lat': 29.5337, 'lon':  -98.4698},
    'New Orleans':   {'lat': 29.9934, 'lon':  -90.2580},
    'Philadelphia':  {'lat': 39.8744, 'lon':  -75.2424},
    'Boston':        {'lat': 42.3656, 'lon':  -71.0096},
    'Denver':        {'lat': 39.8561, 'lon': -104.6737},
    'Oklahoma City': {'lat': 35.3931, 'lon':  -97.6007},
    'Minneapolis':   {'lat': 44.8848, 'lon':  -93.2223},
    'Washington DC': {'lat': 38.8512, 'lon':  -77.0402},
    'Chicago':       {'lat': 41.7868, 'lon':  -87.7522},  # KMDW Midway, NOT O'Hare
}

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

CLI_STATIONS = dict(WETHR_STATIONS)

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
NORTHEAST_CITIES      = {'New York', 'Philadelphia', 'Boston', 'Washington DC'}
REGIONAL_PRIOR_BIAS   = {'Chicago': 'Minneapolis'}

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


def c_to_f(c):
    return c * 9 / 5 + 32


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
                         params={'city': 'eq.' + city, 'order': 'date.asc',
                                 'limit': '200'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def sb_fetch_today(city):
    today = get_eastern_date()
    try:
        r = requests.get(sb_url('settlements'), headers=sb_headers(),
                         params={'date': 'eq.' + today, 'city': 'eq.' + city},
                         timeout=10)
        rows = r.json() if r.status_code == 200 else []
        return rows[0] if rows else None
    except Exception:
        return None


def sb_upsert(city, consensus, forecast, ensemble_mean, source_gap,
              high_uncertainty, obs_high, bias_correction):
    """One row per city per day. This is the file's entire output.

    FAV V1.1 reads settlements.consensus for its agreement tag. Nothing else
    downstream depends on this file.
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
        'actual': None, 'error': None,
    }
    if existing:
        update = {k: v for k, v in row.items() if k not in ('date', 'city')}
        # never blank a settled actual on a re-run
        if existing.get('actual') is not None:
            update.pop('actual', None)
            update.pop('error', None)
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(existing['id']),
            headers=sb_headers(), json=update, timeout=10)
        return r.status_code in (200, 204)
    r = requests.post(sb_url('settlements'), headers=sb_headers(),
                      json=row, timeout=10)
    return r.status_code in (200, 201)


# ── Bias correction ───────────────────────────────────────────────────────────
def _trimmed_mean(errors):
    """Drop the single highest and lowest error, average the rest.

    statistics.median() discarded the systematic middle of a skewed error
    distribution — across 2,514 settlements the residual bias AFTER correction
    was still positive in 17 of 18 cities. Trimmed mean keeps outlier
    robustness (a +12F sensor spike is still dropped) without throwing away the
    persistent drift the median was compressing.

    VERIFIED 2026-08-29: residuals now +1.23 (Las Vegas) to -0.69 (New York),
    11 positive and 7 negative, mean ~+0.18F. The one-directional bias is gone.
    """
    if not errors:
        return 0.0
    if len(errors) < 4:
        return sum(errors) / len(errors)
    s = sorted(errors)[1:-1]
    return sum(s) / len(s)


def compute_bias_correction(city, n_recent=10):
    rows = sb_fetch_city(city)
    complete = [r for r in rows
                if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        prior_city = REGIONAL_PRIOR_BIAS.get(city)
        if prior_city:
            prior_rows = sb_fetch_city(prior_city)
            prior_complete = [r for r in prior_rows
                              if r.get('actual') is not None
                              and r.get('consensus') is not None]
            if len(prior_complete) >= 3:
                recent = prior_complete[-n_recent:]
                errors = [r['actual'] - r['consensus'] for r in recent]
                return round(max(-3.0, min(3.0, _trimmed_mean(errors))), 2), len(complete)
        return 0.0, len(complete)
    recent = complete[-n_recent:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    med_error = _trimmed_mean(errors)
    if city in NWS_BIAS_BOOST_CITIES:
        med_error = med_error * NWS_BIAS_BOOST_MULTIPLIER
    return round(max(-3.0, min(3.0, med_error)), 2), len(recent)


# ── NWS forecast + observations ───────────────────────────────────────────────
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
        office, gx, gy = props.get('gridId'), props.get('gridX'), props.get('gridY')
        if not all([office, gx is not None, gy is not None]):
            return None
        _NWS_GRID_CACHE[key] = (office, gx, gy)
        return _NWS_GRID_CACHE[key]
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
                high = r.json().get('high')
                if high is not None:
                    return round(float(high), 1)
        except Exception:
            pass
    coords = CITIES[city]
    grid = fetch_nws_grid(coords['lat'], coords['lon'])
    if not grid:
        return None
    office, gx, gy = grid
    try:
        r = requests.get(
            f'https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast/hourly',
            headers=HEADERS, timeout=12)
        r.raise_for_status()
        periods = r.json().get('properties', {}).get('periods', [])
        highs = []
        for p in periods:
            if not (p.get('startTime') or '').startswith(today):
                continue
            temp = p.get('temperature')
            if temp is not None and p.get('isDaytime', True):
                unit = p.get('temperatureUnit', 'F')
                highs.append(float(temp) if unit == 'F' else c_to_f(float(temp)))
        if highs:
            return round(max(highs), 1)
    except Exception:
        pass
    return None


def fetch_current_temp(city):
    station = WETHR_STATIONS.get(city)
    if not station:
        return None
    try:
        r = requests.get(
            'https://wethr.net/api/v2/observations.php',
            params={'station_code': station, 'mode': 'latest'},
            headers=WETHR_HEADERS, timeout=12)
        if r.status_code == 200:
            temp = r.json().get('temperature_display')
            if temp is not None:
                return round(float(temp), 1)
    except Exception:
        pass
    return None


def fetch_obs_high(city):
    """⚠️ This reads Wethr's summary field, not the station directly.

    On 2026-09-06 it reported an obs high of 79.0F on a day the actual high was
    78, which eliminated "78 or below" from the model. fetch_obs_live.py now
    computes a running max from raw observations and from hourly METAR T-groups
    for the same stations. If this ever becomes load-bearing again, read
    obs_live.precise_max_f instead.
    """
    station = WETHR_STATIONS.get(city)
    if not station:
        return None
    try:
        r = requests.get(
            'https://wethr.net/api/v2/observations.php',
            params={'station_code': station, 'mode': 'wethr_high', 'logic': 'nws'},
            headers=WETHR_HEADERS, timeout=12)
        if r.status_code == 200:
            wethr_high = r.json().get('wethr_high')
            if wethr_high is not None:
                return round(float(wethr_high), 1)
    except Exception:
        pass
    return None


# ── GFS ensemble ──────────────────────────────────────────────────────────────
def fetch_gfs_forecast_fallback(city):
    coords = CITIES[city]
    params = {
        'latitude': coords['lat'], 'longitude': coords['lon'],
        'hourly': 'temperature_2m', 'temperature_unit': 'fahrenheit',
        'timezone': 'auto', 'forecast_days': 2, 'models': 'gfs_seamless',
    }
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'    ⚠️ [{city}] GFS fallback FAILED: {type(e).__name__}')
        return None, None

    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times, temps = hourly.get('time', []), hourly.get('temperature_2m', [])
    vals = []
    for i, t in enumerate(times):
        if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21:
            if i < len(temps) and temps[i] is not None:
                try:
                    vals.append(float(temps[i]))
                except Exception:
                    pass
    if not vals:
        return None, None
    mx = round(max(vals), 1)
    return [mx], mx


def fetch_gfs_ensemble(city):
    """GFS weight is 0.0 for 17 of 18 cities (Houston 0.18), so this is close to
    decorative — but source_gap feeds the high_uncertainty flag, which is
    stored. Kept, with the fallback chain intact."""
    coords = CITIES[city]
    params = {
        'latitude': coords['lat'], 'longitude': coords['lon'],
        'hourly': 'temperature_2m', 'temperature_unit': 'fahrenheit',
        'timezone': 'auto', 'forecast_days': 2, 'models': 'gfs_seamless',
    }
    data = None
    for attempt in (1, 2):
        try:
            r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble',
                             params=params, headers=HEADERS, timeout=45)
            r.raise_for_status()
            data = r.json()
            break
        except requests.exceptions.Timeout:
            if attempt == 1:
                print(f'    … [{city}] GFS ensemble timeout, retrying once')
                continue
            return fetch_gfs_forecast_fallback(city)
        except Exception as e:
            print(f'    ⚠️ [{city}] GFS ensemble failed: {type(e).__name__}')
            return fetch_gfs_forecast_fallback(city)
    if data is None:
        return fetch_gfs_forecast_fallback(city)

    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    idx = [i for i, t in enumerate(times)
           if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21]
    if not idx:
        idx = [i for i, t in enumerate(times) if t.startswith(today)]
    if not idx:
        return fetch_gfs_forecast_fallback(city)

    member_maxes = []
    for key, vals in hourly.items():
        if key == 'time' or 'temperature_2m' not in key or not isinstance(vals, list):
            continue
        day_vals = [vals[i] for i in idx if i < len(vals) and vals[i] is not None]
        if day_vals:
            try:
                member_maxes.append(round(max(float(v) for v in day_vals), 1))
            except Exception:
                pass
    if len(member_maxes) < 3:
        return fetch_gfs_forecast_fallback(city)
    return member_maxes, round(sum(member_maxes) / len(member_maxes), 1)


# ── Consensus ─────────────────────────────────────────────────────────────────
def late_day_floor(fc, obs, local_hour, city=''):
    gap = max(0.0, fc - obs)
    if city in NORTHEAST_CITIES:
        frac = 0.50 if local_hour < 12 else 0.75 if local_hour < 14 else 0.88 if local_hour < 16 else 0.93
    else:
        frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap


def compute_consensus(fc, cur, noaa, city, obs_high=None):
    """The one number this file exists to produce. 0.82F MAE.

    V5.30.0 CONSENSUS FLOOR FIX. An observed high is a MEASUREMENT, not a
    forecast base. This used to do `consensus = obs_high` and then add the city
    warm offset on top, and main() added bias on top of THAT — Phoenix reported
    114.35F from a 113.0F floor while every forecast input read below 111F.

    Now: decide whether the observed high is trustworthy, but apply it as a
    max() clamp AFTER the warm offset. The offset shapes the forecast estimate;
    the measurement can only raise the result from below, never become the base
    that adjustments stack on. main() re-clamps after bias for the same reason.
    """
    mode = CITY_PREDICTION_MODE.get(city, 'full_blend')
    local_hour = get_local_hour(city)
    obs_locked = False

    if mode == 'nws_only':
        consensus = float(fc)
        obs = noaa if noaa is not None else cur
    else:
        is_fc_heavy = city in FORECAST_HEAVY_CITIES
        obs_val = noaa if noaa is not None else cur
        if is_fc_heavy and local_hour < 10:
            base = fc * 0.95 + obs_val * 0.05 if obs_val is not None else fc
        elif is_fc_heavy and local_hour < 14:
            base = fc * 0.90 + obs_val * 0.10 if obs_val is not None else fc
        elif is_fc_heavy and local_hour < 16:
            base = fc * 0.75 + obs_val * 0.25 if obs_val is not None else fc
        elif local_hour < 10:
            base = (fc * 0.90 + cur * 0.07 + noaa * 0.03) if noaa is not None else fc * 0.93 + cur * 0.07
        elif local_hour < 12:
            base = (fc * 0.80 + cur * 0.12 + noaa * 0.08) if noaa is not None else fc * 0.85 + cur * 0.15
        elif local_hour < 14:
            base = (fc * 0.65 + cur * 0.18 + noaa * 0.17) if noaa is not None else fc * 0.78 + cur * 0.22
        else:
            base = (fc * 0.45 + cur * 0.25 + noaa * 0.30) if noaa is not None else fc * 0.60 + cur * 0.40
        if abs(base - fc) > 4.0:
            base = fc - 4.0 if base < fc else fc + 4.0
        obs = obs_val
        consensus = max(base, late_day_floor(fc, obs, local_hour, city)) if obs is not None else base

    if obs_high is not None and obs_high > consensus:
        trusted = True
        if local_hour < OBS_HIGH_TRUST_HOUR:
            trusted = False
        check = obs if obs is not None else cur
        if check is not None and obs_high > check + OBS_HIGH_MAX_OVERSHOOT:
            trusted = False
        if check is not None and obs_high < check:
            trusted = False
        if trusted:
            obs_locked = True

    warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
    if warm_offset != 0.0:
        consensus += warm_offset

    # measurement clamps from below, AFTER the offset
    if obs_locked:
        consensus = max(consensus, obs_high)

    return consensus


# ── Settlement ────────────────────────────────────────────────────────────────
_CLI_CACHE = {}


def fetch_cli_max_temp(city, target_date_str):
    station = CLI_STATIONS.get(city)
    if not station:
        return None
    year = target_date_str[:4]
    key = station + '_' + year
    if key not in _CLI_CACHE:
        try:
            r = requests.get(
                'https://mesonet.agron.iastate.edu/json/cli.py'
                f'?station={station}&year={year}',
                headers=HEADERS, timeout=15)
            r.raise_for_status()
            lookup = {}
            for entry in r.json().get('results', []):
                valid, high = entry.get('valid', ''), entry.get('high')
                if valid and high is not None:
                    try:
                        lookup[valid] = float(high)
                    except Exception:
                        pass
            _CLI_CACHE[key] = lookup
        except Exception:
            return None
    return _CLI_CACHE.get(key, {}).get(target_date_str)


def sb_fetch_unsettled():
    try:
        r = requests.get(sb_url('settlements'), headers=sb_headers(),
                         params={'actual': 'is.null', 'order': 'date.asc'},
                         timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def sb_update_actual(row_id, actual, error):
    try:
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(row_id),
            headers=sb_headers(),
            json={'actual': round(actual, 2), 'error': error}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


def run_settlement_pass():
    """Fill settlements.actual from Iowa State CLI.

    This is what every accuracy number in the project is scored against —
    consensus MAE, the bias correction, the cloud-variance finding. Do not
    settle against an observation feed; CLI is the official climate record and
    it is what Kalshi resolves on.
    """
    print('\n=== Settlement Pass ===')
    today = get_eastern_date()
    unsettled = sb_fetch_unsettled()
    if not unsettled:
        print('  No unsettled rows.')
        return

    settled = []
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
        if sb_update_actual(row['id'], actual, error):
            settled.append((city, row_date, actual, error))

    if settled:
        print(f'  Settled {len(settled)}:')
        for city, d, actual, err in settled:
            sign = '+' if (err or 0) >= 0 else ''
            print(f'    {city:<15} {d}  actual {actual}F  error {sign}{err}F')
    else:
        print(f'  Nothing settleable yet ({len(unsettled)} pending, '
              f'CLI may not be published).')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = get_eastern_date()
    now_et = datetime.now(pytz.timezone('America/New_York'))

    print(f'\n=== V6.0 Consensus Writer ===')
    print(f'{today} | {now_et:%I:%M %p ET} | {len(CITIES)} cities')
    print('writes settlements.consensus. places no bets, picks no brackets.\n')

    ok_count = 0
    failed = []

    for city in CITIES:
        print(f'  [{city}]')
        try:
            nws_fc = fetch_nws_forecast(city)
            if nws_fc is None:
                print('    ⚠️ No NWS forecast — skipping')
                failed.append(city)
                continue

            current_temp = fetch_current_temp(city)
            obs_high_raw = fetch_obs_high(city)
            ensemble_members, ensemble_mean = fetch_gfs_ensemble(city)

            obs_high = obs_high_raw
            if (obs_high_raw is not None and current_temp is not None
                    and obs_high_raw > current_temp + OBS_HIGH_OVER_CURRENT_THRESHOLD):
                print(f'    ⚠️ Obs high {obs_high_raw}F discarded — '
                      f'{obs_high_raw - current_temp:.1f}F above current')
                obs_high = None
            elif (obs_high_raw is not None
                    and obs_high_raw > nws_fc + OBS_HIGH_OVER_FORECAST_THRESHOLD):
                print(f'    ⚠️ Obs high {obs_high_raw}F discarded — '
                      f'{obs_high_raw - nws_fc:.1f}F above NWS forecast')
                obs_high = None

            if (ensemble_mean is not None and abs(ensemble_mean - nws_fc) > 8.0):
                print(f'    ⚠️ GFS discarded — {abs(ensemble_mean - nws_fc):.1f}F from NWS')
                ensemble_members = ensemble_mean = None

            source_gap = abs(nws_fc - ensemble_mean) if ensemble_mean is not None else None
            high_uncertainty = source_gap is not None and source_gap > 5.0

            bias_correction, bias_n = compute_bias_correction(city)

            cur = current_temp if current_temp is not None else nws_fc
            consensus_raw = compute_consensus(nws_fc, cur, current_temp, city,
                                              obs_high=obs_high)

            # re-clamp after bias: a measurement must not be inflated
            _obs_locked = (obs_high is not None
                           and abs(consensus_raw - obs_high) < 0.05)
            consensus = round(consensus_raw + bias_correction, 1)
            if _obs_locked:
                consensus = round(max(consensus, obs_high), 1)

            # V5.29.D ensemble-aware correction
            if ensemble_mean is not None:
                gap = ensemble_mean - consensus
                locked = obs_high is not None and abs(consensus - obs_high) < 0.1
                wild = abs(ensemble_mean - nws_fc) > 8.0
                if abs(gap) > 3.0 and not locked and not wild:
                    adj = max(-2.0, min(2.0, 0.5 * gap))
                    consensus = round(consensus + adj, 1)
                    print(f'    ensemble shift {adj:+.1f}F (GFS {ensemble_mean}F)')

            lock_note = ' [obs-locked]' if _obs_locked else ''
            print(f'    NWS {nws_fc}F | cur {current_temp}F | obs high {obs_high}F '
                  f'| GFS {ensemble_mean}F | bias {bias_correction:+.2f} (n={bias_n})')
            print(f'    → consensus {consensus}F{lock_note}')

            if sb_upsert(city, consensus, nws_fc, ensemble_mean, source_gap,
                         high_uncertainty, obs_high, bias_correction):
                ok_count += 1
                print('    ✅ saved')
            else:
                failed.append(city)
                print('    ❌ save failed')

        except Exception as e:
            print(f'    ❌ {type(e).__name__}: {str(e)[:120]}')
            failed.append(city)

        time.sleep(0.3)

    print(f'\n=== Summary ===')
    print(f'Saved {ok_count}/{len(CITIES)}')
    if failed:
        print(f'Failed: {", ".join(failed)}')

    run_settlement_pass()

    print('\nConsensus accuracy:')
    print('  select round(avg(abs(actual::numeric - consensus::numeric)),2) mae,')
    print('         round(avg(actual::numeric - consensus::numeric),2) mean_err,')
    print('         count(*) n')
    print('  from settlements where actual is not null;')


if __name__ == '__main__':
    main()
