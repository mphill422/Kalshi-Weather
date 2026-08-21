"""
fetch_intraday.py — Intraday atmospheric collector (reversal-strategy foundation)
=================================================================================
Captures the LIVE atmospheric features that let us predict whether the market's
current temperature bracket is about to flip before settlement. This is the
ingredient the Kalshi weather market underuses — most bettors watch the morning
point forecast; this watches the atmosphere evolve through the heating window.

Runs every 15-30 min on a cron. Writes to a NEW table (intraday_atmospherics).
Touches nothing in the existing highs model. Read-only w.r.t. everything else.

V2 CHANGES:
  - Expanded 4 -> 18 cities, matching fetch_weather.py CITIES exactly.
    Houston coords corrected to KHOU Hobby (29.6459/-95.2769); the prior
    29.9902/-95.3368 was Bush/KIAH, so all Houston rows before this version
    profile the wrong airport and should be excluded from Houston analysis.
  - Overnight-safe Open-Meteo window: past_days=1, forecast_days=2 so the
    hourly array always brackets 'now' regardless of UTC date rollover. The
    old forecast_days=1 returned only today's UTC hours, and the nearest-hour
    search would silently clamp to the array edge and log stale values as live.
  - MAX_HOUR_GAP_SECONDS staleness guard on that search — a match further than
    90 min from 'now' is treated as no data rather than logged.
  - local_date recorded alongside ET date. Overnight captures belong to the
    city's own calendar date, which diverges from ET after local midnight.
    Group lows analysis on local_date, not date.

STILL OPEN: wethr_obs has returned NULL on 100% of rows since inception
(4,215/4,215). The call below is unchanged and still expected to fail — see
wethr_probe.py and the working implementation in fetch_weather.py.

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
  CREATE INDEX IF NOT EXISTS idx_intraday_city_localdate
    ON public.intraday_atmospherics (city, local_date);

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

# All 18 cities. Coords, tz, and station live together so they cannot drift
# apart. Mirrors CITIES / CITY_TZ / WETHR_STATIONS in fetch_weather.py — if
# that file changes, change this too.
CITIES = {
    'Phoenix':       {'lat': 33.4342, 'lon': -112.0116, 'tz': 'America/Phoenix',     'station': 'KPHX'},
    'Las Vegas':     {'lat': 36.0840, 'lon': -115.1537, 'tz': 'America/Los_Angeles', 'station': 'KLAS'},
    'Los Angeles':   {'lat': 33.9416, 'lon': -118.4085, 'tz': 'America/Los_Angeles', 'station': 'KLAX'},
    'Dallas':        {'lat': 32.8998, 'lon':  -97.0403, 'tz': 'America/Chicago',     'station': 'KDFW'},
    'Austin':        {'lat': 30.1945, 'lon':  -97.6699, 'tz': 'America/Chicago',     'station': 'KAUS'},
    'Houston':       {'lat': 29.6459, 'lon':  -95.2769, 'tz': 'America/Chicago',     'station': 'KHOU'},
    'Atlanta':       {'lat': 33.6407, 'lon':  -84.4277, 'tz': 'America/New_York',    'station': 'KATL'},
    'Miami':         {'lat': 25.7959, 'lon':  -80.2870, 'tz': 'America/New_York',    'station': 'KMIA'},
    'New York':      {'lat': 40.7812, 'lon':  -73.9665, 'tz': 'America/New_York',    'station': 'KNYC'},
    'San Antonio':   {'lat': 29.5337, 'lon':  -98.4698, 'tz': 'America/Chicago',     'station': 'KSAT'},
    'New Orleans':   {'lat': 29.9934, 'lon':  -90.2580, 'tz': 'America/Chicago',     'station': 'KMSY'},
    'Philadelphia':  {'lat': 39.8744, 'lon':  -75.2424, 'tz': 'America/New_York',    'station': 'KPHL'},
    'Boston':        {'lat': 42.3656, 'lon':  -71.0096, 'tz': 'America/New_York',    'station': 'KBOS'},
    'Denver':        {'lat': 39.8561, 'lon': -104.6737, 'tz': 'America/Denver',      'station': 'KDEN'},
    'Oklahoma City': {'lat': 35.3931, 'lon':  -97.6007, 'tz': 'America/Chicago',     'station': 'KOKC'},
    'Minneapolis':   {'lat': 44.8848, 'lon':  -93.2223, 'tz': 'America/Chicago',     'station': 'KMSP'},
    'Washington DC': {'lat': 38.8512, 'lon':  -77.0402, 'tz': 'America/New_York',    'station': 'KDCA'},
    'Chicago':       {'lat': 41.7868, 'lon':  -87.7522, 'tz': 'America/Chicago',     'station': 'KMDW'},
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
        print(f'      open-meteo: nearest hour is {best_gap / 3600:.1f}h from now '
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
    """Live surface observation from Wethr — your data advantage. Best-effort.

    KNOWN BROKEN: returns None on 100% of calls. Left unchanged pending a port
    of the working implementation from fetch_weather.py (which authenticates
    via WETHR_HEADERS rather than an api_key query param)."""
    if not WETHR_API_KEY:
        print('      wethr: WETHR_API_KEY not set in this environment')
        return None
    try:
        r = requests.get('https://wethr.net/api/v2/observations.php',
                         params={'station_code': station, 'mode': 'latest',
                                 'api_key': WETHR_API_KEY},
                         headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'      wethr HTTP {r.status_code}: {r.text[:120]}')
            return None
        data = r.json()
        rec = data[0] if isinstance(data, list) and data else data
        if isinstance(rec, dict):
            for k in ('temperature', 'temp', 'current_temp'):
                if rec.get(k) is not None:
                    return round(float(rec[k]), 2)
            print(f'      wethr: no temp key in response; keys={list(rec)[:8]}')
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
        if r.status_code not in (200, 201, 204):
            print(f'      sb_insert HTTP {r.status_code}: {r.text[:140]}')
            return False
        return True
    except Exception as e:
        print(f'      sb_insert exc: {type(e).__name__}: {str(e)[:80]}')
        return False


def main():
    today = et_date()
    print(f'=== intraday collector v2 | ET {today} | '
          f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC | '
          f'{len(CITIES)} cities ===')
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing logged (exit 0).')
        sys.exit(0)

    logged = 0
    for city, cfg in CITIES.items():
        atmo = fetch_open_meteo(cfg['lat'], cfg['lon'])
        if not atmo or atmo.get('temp_2m') is None:
            print(f'  [{city}] no atmospheric data — skipped')
            continue
        obs = fetch_wethr_obs(cfg['station'])
        row = {
            'date': today, 'local_date': local_date(cfg['tz']), 'city': city,
            'local_hour': local_hour(cfg['tz']),
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
