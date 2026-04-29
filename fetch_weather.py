"""
fetch_weather.py — MPH Weather Model V5.19
Scheduled weather fetcher for GitHub Actions.

Runs every 25 minutes from 9am-5pm ET via GitHub Actions cron.
Fetches NWS forecast, current temp, obs high, GFS ensemble, and NBM
for all 10 visible cities, computes consensus + bias correction,
and upserts to Supabase settlements table.

This keeps Supabase always fresh so the Streamlit app loads instantly
without waiting for live API calls on every page load.
"""

import math
import os
import re
import requests
import statistics
import time
from datetime import datetime, timedelta

import pytz

# ── Credentials from GitHub Secrets ──────────────────────────────────────────
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
SUPABASE_URL  = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY  = os.environ.get('SUPABASE_KEY', '')

WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS       = {'User-Agent': 'kalshi-weather-fetcher/5.19', 'Accept': 'application/json'}

# ── Only fetch visible (non-hidden) cities ────────────────────────────────────
VISIBLE_CITIES = [
    'New York', 'Miami', 'Atlanta', 'Washington DC',  # ET
    'Dallas', 'Houston', 'New Orleans', 'Oklahoma City',  # CT
    'Phoenix', 'Las Vegas',  # PT
]

CITY_TZ = {
    'Phoenix': 'America/Phoenix', 'Las Vegas': 'America/Los_Angeles',
    'Dallas': 'America/Chicago', 'Houston': 'America/Chicago',
    'Atlanta': 'America/New_York', 'Miami': 'America/New_York',
    'New York': 'America/New_York', 'New Orleans': 'America/Chicago',
    'Oklahoma City': 'America/Chicago', 'Washington DC': 'America/New_York',
}

CITIES = {
    'Phoenix':       {'lat': 33.4342, 'lon': -112.0116},
    'Las Vegas':     {'lat': 36.0840, 'lon': -115.1537},
    'Dallas':        {'lat': 32.8998, 'lon': -97.0403},
    'Houston':       {'lat': 29.9902, 'lon': -95.3368},
    'Atlanta':       {'lat': 33.6407, 'lon': -84.4277},
    'Miami':         {'lat': 25.7959, 'lon': -80.2870},
    'New York':      {'lat': 40.7812, 'lon': -73.9665},
    'New Orleans':   {'lat': 29.9934, 'lon': -90.2580},
    'Oklahoma City': {'lat': 35.3931, 'lon': -97.6007},
    'Washington DC': {'lat': 38.8512, 'lon': -77.0402},
}

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Dallas': 'KDFW',
    'Houston': 'KHOU', 'Atlanta': 'KATL', 'Miami': 'KMIA',
    'New York': 'KNYC', 'New Orleans': 'KMSY', 'Oklahoma City': 'KOKC',
    'Washington DC': 'KDCA',
}

OBHISTORY_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Dallas': 'KDFW',
    'Houston': 'KHOU', 'Atlanta': 'KATL', 'Miami': 'KMIA',
    'New York': 'KNYC', 'New Orleans': 'KMSY', 'Oklahoma City': 'KOKC',
    'Washington DC': 'KDCA',
}

CLI_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Dallas': 'KDFW',
    'Houston': 'KHOU', 'Atlanta': 'KATL', 'Miami': 'KMIA',
    'New York': 'KNYC', 'New Orleans': 'KMSY', 'Oklahoma City': 'KOKC',
    'Washington DC': 'KDCA',
}

# V5.19 city warm offsets (mirrors streamlit_app.py)
CITY_WARM_OFFSET = {
    'Miami':         2.5,   # avg error +2.59F over 27 days
    'Phoenix':       1.0,   # avg error +0.94F
    'Las Vegas':    -1.0,   # avg error -1.00F — runs cold
}

CITY_PREDICTION_MODE = {
    'New York':      'full_blend',
    'Houston':       'full_blend',
    'Dallas':        'full_blend',
    'Miami':         'full_blend',
    'Phoenix':       'full_blend',
    'Las Vegas':     'full_blend',
    'New Orleans':   'nws_only',
    'Washington DC': 'nws_only',
    'Atlanta':       'nws_only',
    'Oklahoma City': 'nws_only',
}

FORECAST_HEAVY_CITIES = {'Dallas', 'Houston', 'Oklahoma City'}
DESERT_CITIES         = {'Phoenix', 'Las Vegas'}
NORTHEAST_CITIES      = {'New York', 'Washington DC'}

GFS_CITY_WEIGHT = {
    'Phoenix': 0.10, 'Las Vegas': 0.10,
    'Miami': 0.18, 'Houston': 0.18, 'New Orleans': 0.18,
    'Dallas': 0.25, 'Oklahoma City': 0.25,
    'Atlanta': 0.22,
    'New York': 0.15, 'Washington DC': 0.15,
}

BASE_SIGMA = {
    'New York': 1.8, 'Washington DC': 1.9, 'Miami': 2.0,
    'New Orleans': 2.1, 'Phoenix': 2.2, 'Las Vegas': 2.2, 'Atlanta': 2.3,
    'Dallas': 2.3, 'Houston': 2.3, 'Oklahoma City': 2.5,
}

OBS_HIGH_TRUST_HOUR   = 13
OBS_HIGH_MAX_OVERSHOOT = 10.0


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


# ── Supabase helpers ──────────────────────────────────────────────────────────
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
              high_uncertainty, obs_high, bias_correction):
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
        # Preserve actual/error if already settled
        if existing.get('actual') is not None:
            update.pop('actual', None)
            update.pop('error', None)
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(existing['id']),
            headers=sb_headers(), json=update, timeout=10)
        return r.status_code in (200, 204)
    else:
        r = requests.post(sb_url('settlements'), headers=sb_headers(), json=row, timeout=10)
        return r.status_code in (200, 201)


# ── Bias correction ───────────────────────────────────────────────────────────
def compute_bias_correction(city, n_recent=14):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        return 0.0, len(complete)
    recent = complete[-n_recent:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    med_error = statistics.median(errors)
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(abs_errors)
    if mae > 4.0:
        med_error *= 0.5
    return round(max(-3.0, min(3.0, med_error)), 2), len(recent)


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
    # NWS fallback
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


# ── Current temp ──────────────────────────────────────────────────────────────
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


# ── Obs high today ────────────────────────────────────────────────────────────
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


# ── GFS ensemble ──────────────────────────────────────────────────────────────
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
    try:
        r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble',
                         params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None
    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    today_indices = [i for i, t in enumerate(times)
                     if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21]
    if not today_indices:
        today_indices = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_indices:
        return None, None
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
        return None, None
    return member_maxes, round(sum(member_maxes) / len(member_maxes), 1)


# ── Consensus computation (mirrors streamlit_app.py) ─────────────────────────
def choose_sigma(city, obs_high=None, forecast=None):
    s = BASE_SIGMA.get(city, 2.1)
    local_hour = get_local_hour(city)
    s *= 1.00 if local_hour < 11 else 0.94 if local_hour < 14 else 0.90 if local_hour < 16 else 0.86
    if city in DESERT_CITIES:
        s *= 0.92
    if obs_high is not None and forecast is not None:
        gap = abs(forecast - obs_high)
        if gap < 2:
            s *= 0.80
        elif gap < 4:
            s *= 0.90
    return max(1.30, min(2.80, s))

def late_day_floor(fc, obs, local_hour, city=''):
    gap = max(0.0, fc - obs)
    if city in NORTHEAST_CITIES:
        frac = 0.50 if local_hour < 12 else 0.75 if local_hour < 14 else 0.88 if local_hour < 16 else 0.93
    else:
        frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap

def compute_consensus(fc, cur, noaa, city, obs_high=None):
    mode = CITY_PREDICTION_MODE.get(city, 'full_blend')
    if mode == 'nws_only':
        consensus = float(fc)
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
        # obs_high override
        if obs_high is not None and obs_high > consensus:
            obs_high_trusted = True
            if local_hour < OBS_HIGH_TRUST_HOUR:
                obs_high_trusted = False
            current_for_check = obs if obs is not None else cur
            if current_for_check is not None and obs_high > current_for_check + OBS_HIGH_MAX_OVERSHOOT:
                obs_high_trusted = False
            if current_for_check is not None and obs_high < current_for_check:
                obs_high_trusted = False
            if obs_high_trusted:
                consensus = obs_high

    # V5.19: city warm offset
    warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
    if warm_offset != 0.0:
        consensus += warm_offset

    return consensus


# ── Main fetch loop ───────────────────────────────────────────────────────────
def main():
    today = get_eastern_date()
    now_et = datetime.now(pytz.timezone('America/New_York'))
    et_hour = now_et.hour

    print(f'\n=== Weather Fetch Run ===')
    print(f'Date: {today} | ET time: {now_et.strftime("%I:%M %p ET")}')
    print(f'Fetching {len(VISIBLE_CITIES)} cities...\n')

    results = []
    for city in VISIBLE_CITIES:
        print(f'  [{city}]')
        try:
            # 1. NWS forecast
            nws_fc = fetch_nws_forecast(city)
            print(f'    NWS forecast: {nws_fc}F')
            if nws_fc is None:
                print(f'    ⚠️ No NWS forecast — skipping {city}')
                continue

            # 2. Current temp
            current_temp = fetch_current_temp(city)
            print(f'    Current temp: {current_temp}F')

            # 3. Obs high
            obs_high_raw = fetch_obs_high(city)
            print(f'    Obs high: {obs_high_raw}F')

            # 4. GFS ensemble
            ensemble_members, ensemble_mean = fetch_gfs_ensemble(city)
            print(f'    GFS ensemble: {ensemble_mean}F ({len(ensemble_members) if ensemble_members else 0} members)')

            # 5. Sanity checks
            obs_high = obs_high_raw
            if obs_high_raw is not None and current_temp is not None and obs_high_raw > current_temp + 15.0:
                print(f'    ⚠️ Obs high discarded — {obs_high_raw}F is {obs_high_raw - current_temp:.1f}F above current')
                obs_high = None
            if obs_high_raw is not None and nws_fc is not None and obs_high is not None and obs_high_raw > nws_fc + 12.0:
                print(f'    ⚠️ Obs high discarded — {obs_high_raw}F is {obs_high_raw - nws_fc:.1f}F above NWS')
                obs_high = None
            if ensemble_mean is not None and nws_fc is not None and abs(ensemble_mean - nws_fc) > 8.0:
                print(f'    ⚠️ GFS discarded — {abs(ensemble_mean - nws_fc):.1f}F gap from NWS')
                ensemble_members = None
                ensemble_mean = None

            # 6. Source gap / uncertainty
            source_gap = None
            high_uncertainty = False
            if nws_fc is not None and ensemble_mean is not None:
                source_gap = abs(nws_fc - ensemble_mean)
                high_uncertainty = source_gap > 5.0

            # 7. Bias correction
            bias_correction, bias_n = compute_bias_correction(city)
            print(f'    Bias correction: {bias_correction:+.2f}F ({bias_n} days)')

            # 8. Consensus
            cur = current_temp if current_temp is not None else nws_fc
            consensus_raw = compute_consensus(nws_fc, cur, current_temp, city, obs_high=obs_high)
            consensus = round(consensus_raw + bias_correction, 1)
            warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
            print(f'    Consensus: {consensus}F (raw={consensus_raw:.1f}, bias={bias_correction:+.2f}, offset={warm_offset:+.1f})')

            # 9. Upsert to Supabase
            ok = sb_upsert(
                city=city, consensus=consensus, forecast=nws_fc,
                ensemble_mean=ensemble_mean, source_gap=source_gap,
                high_uncertainty=high_uncertainty, obs_high=obs_high,
                bias_correction=bias_correction,
            )
            status = '✅ Saved' if ok else '❌ Save failed'
            print(f'    {status}')
            results.append({'city': city, 'consensus': consensus, 'ok': ok})

        except Exception as e:
            print(f'    ❌ Error: {e}')
            results.append({'city': city, 'consensus': None, 'ok': False})

        time.sleep(0.5)  # be polite to APIs

    # Summary
    print(f'\n=== Summary ===')
    ok_count = sum(1 for r in results if r['ok'])
    print(f'Saved {ok_count}/{len(VISIBLE_CITIES)} cities successfully')
    for r in results:
        status = '✅' if r['ok'] else '❌'
        consensus_str = f"{r['consensus']}F" if r['consensus'] else 'failed'
        print(f'  {status} {r["city"]}: {consensus_str}')

    if ok_count < len(VISIBLE_CITIES):
        print(f'\n⚠️ {len(VISIBLE_CITIES) - ok_count} cities failed — check logs above')
        exit(1)  # fail the Action so GitHub sends a notification


if __name__ == '__main__':
    main()
