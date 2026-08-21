"""
fetch_intraday.py — Intraday atmospheric collector (reversal-strategy foundation)
=================================================================================
Captures the LIVE atmospheric features that let us predict whether the market's
current temperature bracket is about to flip before settlement. This is the
ingredient the Kalshi weather market underuses — most bettors watch the morning
point forecast; this watches the atmosphere evolve through the heating window.

Runs every 15-30 min on a cron. Writes to a NEW table (intraday_atmospherics).
Touches nothing in the existing highs model. Read-only w.r.t. everything else.

OVERNIGHT COLLECTION (V2): window widened past midnight to support the V5.30
lows model. Minimum temperature locks in near sunrise, so the 8am-10pm heating
window captured none of the relevant decay curve. Two changes support this:
  - Open-Meteo now requests past_days=1, forecast_days=2 so the hourly array
    always brackets 'now' regardless of UTC date rollover.
  - _nearest_hour_index enforces MAX_HOUR_GAP; a match further than that from
    'now' is treated as no data rather than silently logged as stale.
  - local_date recorded alongside ET date — overnight captures belong to the
    city's own calendar date, which diverges from ET after midnight local.

FEATURES CAPTURED (per city, per run):
  From Open-Meteo (the reversal signal — atmospheric profile the market ignores):
    - temperature_2m        (current modeled surface temp)
    - temperature_925hPa    (925mb temp — warm air aloft that can mix down)
    - temperature_850hPa    (850mb temp — mid-level thermal signal)
    - shortwave_radiation    (solar irradiance — the heating engine)
    - cloud_cover           (suppresses/allows heating)
    - wind_speed_10m, wind_direction_10m  (advection / sea-breeze)
    - apparent_temperature
  From Wethr.net (YOUR surface-truth advantage over free-data competitors):
    - wethr_obs_temp        (live professional observation)

Create the table once (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.intraday_atmospherics (
    id BIGSERIAL PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date TEXT NOT NULL,               -- ET date 'YYYY-MM-DD'
    city TEXT NOT NULL,
    local_hour INTEGER,               -- city-local hour (heating window = 10-15)
    temp_2m NUMERIC(6,2),
    temp_925 NUMERIC(6,2),
    temp_850 NUMERIC(6,2),
    solar NUMERIC(8,2),
    cloud_cover NUMERIC(5,1),
    wind_speed NUMERIC(6,2),
    wind_dir NUMERIC(6,1),
    apparent_temp NUMERIC(6,2),
    wethr_obs NUMERIC(6,2),
    source TEXT
  );
  ALTER TABLE public.intraday_atmospherics ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.intraday_atmospherics
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_intraday_city_date
    ON public.intraday_atmospherics (city, date);

MIGRATION for V2 (run once, additive — does not touch existing rows):

  ALTER TABLE public.intraday_atmospherics
    ADD COLUMN IF NOT EXISTS local_date TEXT;

Secrets needed (already in your repo): SUPABASE_URL, SUPABASE_KEY, WETHR_API_KEY.
ALWAYS exits 0 — a collection hiccup must never spam failure emails.
"""

import os
import sys
from datetime import datetime

import requests
import pytz

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
HEADERS = {'User-Agent': 'intraday-collector/2.0', 'Accept': 'application/json'}

# Reject any Open-Meteo hour further than this from 'now'. Without this the
# nearest-hour search silently clamps to the end of the array and logs stale
# data as if it were live — the exact failure mode overnight collection hits.
MAX_HOUR_GAP_SECONDS = 5400  # 90 min

# The focused 4 — coastal-east, coastal-west, continental, gulf.
# lat, lon, IANA tz, wethr station code (for surface obs).
CITIES = {
    'New York':    (40.7812, -73.9665, 'America/New_York',    'KNYC'),
    'Los Angeles': (33.9416, -118.4085, 'America/Los_Angeles', 'KLAX'),
    'Chicago':     (41.7868, -87.7522, 'America/Chicago',      'KMDW'),
    'Houston':     (29.9902, -95.3368, 'America/Chicago',      'KHOU'),
}

OPEN_METEO = 'https://api.open-meteo.com/v1/forecast'
HOURLY_VARS = ('temperature_2m,temperature_925hPa,temperature_850hPa,'
               'shortwave_radiation,cloud_cover,wind_speed_10m,'
               'wind_direction_10m,apparent_temperature')


def et_date():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')


def local_date(tz_name):
    """City-local calendar date. Diverges from ET date overnight — a 1am PT
    capture is still 'yesterday' locally while ET has already rolled over."""
    return datetime.now(pytz.timezone(tz_name)).strftime('%Y-%m-%d')


def local_hour(tz_name):
    return datetime.now(pytz.timezone(tz_name)).hour


def _nearest_hour_index(times):
    """Open-Meteo returns hourly arrays; pick the index nearest to 'now' UTC.
    Returns None if the nearest hour is further than MAX_HOUR_GAP_SECONDS —
    that means the array does not bracket 'now' and the data is stale."""
    now = datetime.utcnow()
    best_i, best_gap = None, 1e9
    for i, t in enumerate(times):
        try:
            # times look like '2026-07-08T14:00'
            dt = datetime.strptime(t[:16], '%Y-%m-%dT%H:%M')
            gap = abs((dt - now).total_seconds())
            if gap < best_gap:
                best_gap, best_i = gap, i
        except Exception:
            continue
    if best_i is None or best_gap > MAX_HOUR_GAP_SECONDS:
        print(f'      open-meteo: nearest hour is {best_gap/3600:.1f}h from now '
              f'— stale, rejecting')
        return None
    return best_i


def fetch_open_meteo(lat, lon):
    """Pull the atmospheric reversal features for the hour nearest now.
    Returns dict of features or {} on failure (logs loudly)."""
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly': HOURLY_VARS,
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
        'past_days': 1,
        'forecast_days': 2,
    }
    try:
        r = requests.get(OPEN_METEO, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f'      open-meteo HTTP {r.status_code}: {r.text[:120]}')
            return {}
        h = r.json().get('hourly', {})
        times = h.get('time', [])
        if not times:
            print('      open-meteo: no hourly.time in response')
            return {}
        i = _nearest_hour_index(times)
        if i is None:
            return {}

        def g(key):
            arr = h.get(key)
            if isinstance(arr, list) and i < len(arr) and arr[i] is not None:
                return round(float(arr[i]), 2)
            return None

        return {
            'temp_2m': g('temperature_2m'),
            'temp_925': g('temperature_925hPa'),
            'temp_850': g('temperature_850hPa'),
            'solar': g('shortwave_radiation'),
            'cloud_cover': g('cloud_cover'),
            'wind_speed': g('wind_speed_10m'),
            'wind_dir': g('wind_direction_10m'),
            'apparent_temp': g('apparent_temperature'),
        }
    except Exception as e:
        print(f'      open-meteo exc: {type(e).__name__}: {str(e)[:100]}')
        return {}


def fetch_wethr_obs(station):
    """Live surface observation from Wethr — your data advantage. Best-effort."""
    if not WETHR_API_KEY:
        return None
    try:
        r = requests.get('https://wethr.net/api/v2/observations.php',
                         params={'station_code': station, 'mode': 'latest',
                                 'api_key': WETHR_API_KEY},
                         headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            rec = data[0] if isinstance(data, list) and data else data
            if isinstance(rec, dict):
                for k in ('temperature', 'temp', 'current_temp'):
                    if rec.get(k) is not None:
                        return round(float(rec[k]), 2)
        return None
    except Exception as e:
        print(f'      wethr exc: {type(e).__name__}: {str(e)[:80]}')
        return None


def sb_insert(row):
    try:
        headers = {'apikey': SUPABASE_KEY, 'Authorization': 'Bearer ' + SUPABASE_KEY,
                   'Content-Type': 'application/json', 'Prefer': 'return=minimal'}
        r = requests.post(SUPABASE_URL + '/rest/v1/intraday_atmospherics',
                          headers=headers, json=row, timeout=15)
        return r.status_code in (200, 201, 204)
    except Exception as e:
        print(f'      sb_insert exc: {type(e).__name__}: {str(e)[:80]}')
        return False


def main():
    today = et_date()
    print(f'=== intraday collector v2 | ET {today} | '
          f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC ===')
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing logged (exit 0).')
        sys.exit(0)

    logged = 0
    for city, (lat, lon, tz, station) in CITIES.items():
        atmo = fetch_open_meteo(lat, lon)
        if not atmo or atmo.get('temp_2m') is None:
            print(f'  [{city}] no atmospheric data — skipped')
            continue
        obs = fetch_wethr_obs(station)
        row = {
            'date': today, 'local_date': local_date(tz), 'city': city,
            'local_hour': local_hour(tz),
            'temp_2m': atmo['temp_2m'], 'temp_925': atmo['temp_925'],
            'temp_850': atmo['temp_850'], 'solar': atmo['solar'],
            'cloud_cover': atmo['cloud_cover'], 'wind_speed': atmo['wind_speed'],
            'wind_dir': atmo['wind_dir'], 'apparent_temp': atmo['apparent_temp'],
            'wethr_obs': obs, 'source': 'open-meteo+wethr',
        }
        if sb_insert(row):
            logged += 1
            print(f'  [{city}] ld={row["local_date"]} lh={row["local_hour"]} '
                  f't2m={atmo["temp_2m"]} 925={atmo["temp_925"]} '
                  f'solar={atmo["solar"]} cloud={atmo["cloud_cover"]} '
                  f'obs={obs} ✅')
        else:
            print(f'  [{city}] captured but DB write failed')

    print(f'\nLogged {logged}/{len(CITIES)} cities.')
    sys.exit(0)


if __name__ == '__main__':
    main()
