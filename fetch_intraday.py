"""
fetch_intraday.py — Intraday atmospheric collector (reversal-strategy foundation)
=================================================================================
Captures the LIVE atmospheric features that let us predict whether the market's
current temperature bracket is about to flip before settlement. This is the
ingredient the Kalshi weather market underuses — most bettors watch the morning
point forecast; this watches the atmosphere evolve through the heating window.

Runs every 20 min on a cron. Writes to intraday_atmospherics.
Touches nothing in the existing highs model. Read-only w.r.t. everything else.

V2.3 CHANGES — THE FORECAST CLOUD COLUMNS (2026-09-07)
------------------------------------------------------
This version exists because of a measured finding, and it is worth writing the
finding down here so the columns are not a mystery in three months.

MEASURED 2026-09-07, 368 city-days (Jul 12 - Sep 7), 17 cities, joined to
settlements. Cloud cover averaged over LOCAL HOURS 10-15, terciled WITHIN each
city so city composition cannot drive the result:

    tercile   avg cloud   mean_err    MAE     sd
      1          9.6%       0.00      0.73   1.30
      2         36.5%      +0.07      0.89   1.46
      3         73.0%      -0.20      1.10   1.82

mean_err is FLAT. MAE and sd rise ~50% and ~40%, monotonically.

  -> Cloud does NOT bias the forecast. Consensus already accounts for it in the
     central estimate. Cloud predicts how WIDE the error distribution is.
  -> 12 of 17 cities show the effect individually, including Phoenix (+0.74),
     which kills the "dry cities are just easier" confound.

This is the only hypothesis out of seven tested this week that survived its
control. It matters because choose_sigma() in fetch_weather.py currently treats
uncertainty as a per-city CONSTANT (BASE_SIGMA, recalibrated quarterly) scaled
by hour of day. The actual uncertainty is ~1.4x wider on a city's own cloudy
days than its clear ones, and that is knowable in advance.

THE PROBLEM THIS VERSION SOLVES
--------------------------------
The finding uses cloud OBSERVED during local hours 10-15. The model bets at
14:00 UTC — 10am ET, 7am PT. The heating window has barely started, and for
Pacific cities has not begun. So the model cannot see the variable the finding
depends on.

Morning cloud was tested as a proxy. It works, but weakly:

    early tercile   cloud 6-10   cloud 10-15    MAE
        1              5.5%        16.2%       0.70
        2             23.6%        29.7%       0.76
        3             68.2%        60.0%       0.90

    correlation(early, late) = 0.38 - 0.51

MAE still climbs, so morning cloud carries real information — but the spread is
0.20 instead of 0.37, roughly half the signal. Clear mornings cloud up: tercile
1 goes from 5.5% to 16.2%.

The right variable is the FORECAST for the afternoon, available at bet time.
Open-Meteo already returns it in the hourly array this script fetches — and
_nearest_hour_index() throws all of it away except the single hour nearest now.

So: keep storing the nearest-hour observation exactly as before, and ALSO store
the forecast for local hours 12-15 from the same response. Nothing else changes.

WHAT THIS BUYS, AND WHAT IT DOES NOT
-------------------------------------
It does NOT immediately improve anything. It starts accumulating the pairs
needed to answer one question: does the morning FORECAST of afternoon cloud
predict consensus error as well as the afternoon OBSERVATION does?

  - If yes, choose_sigma() gets the full ~0.37 MAE spread instead of ~0.20.
  - If no, the finding is real but only half-usable, and that is worth knowing
    before writing sigma code around it.

It also measures how good Open-Meteo's cloud forecast actually is, by comparing
fc_cloud_12_15 against the observations this same table collects later that day.
That is worth having on its own.

⚠️ Give it ~2 weeks before querying. One week of city-days will not separate a
real relationship from noise, and seven hypotheses died this week on exactly
that mistake.

The test, when there is enough data:

    -- forecast cloud (stored at the morning capture) vs consensus error
    select ntile(3) over (partition by city order by fc_cloud_12_15) as tercile,
           count(*), round(avg(fc_cloud_12_15),1),
           round(avg(abs(actual - consensus)),2) as mae
    from intraday_atmospherics i
    join settlements s on s.city = i.city
                      and s.date = coalesce(i.local_date, i.date)
    where i.local_hour between 8 and 11
      and i.fc_cloud_12_15 is not null
      and s.actual is not null
    group by tercile;

RUN THIS FIRST or every insert 400s:

  ALTER TABLE public.intraday_atmospherics
    ADD COLUMN IF NOT EXISTS fc_cloud_12_15 NUMERIC(5,1),
    ADD COLUMN IF NOT EXISTS fc_solar_12_15 NUMERIC(8,2),
    ADD COLUMN IF NOT EXISTS fc_tmax_today  NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS fc_hours_used  INTEGER;

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

    ⚠️ local_date is NULL on every row written BEFORE V2. Any query joining
    this table to settlements must coalesce(local_date, date) or it silently
    drops most of the history — that mistake returned zero rows on the first
    attempt at the cloud analysis above.

V2.1 CHANGES:
  - Wethr removed. It returned HTTP 401 "API key missing" on every call since
    inception (wethr_obs null on 4,215/4,215 rows). For the lows model the
    useful observation is the settled minimum, which is collected elsewhere,
    so the surface obs is not worth fixing here. The wethr_obs column is left
    in place but is no longer written.
  - Rate-limit handling: REQUEST_SPACING_SECONDS between cities, plus one
    application-level retry on timeout.

V2.2 CHANGES:
  - Pooled session. The application-level retry from V2.1 never succeeded once
    across two runs (11/18 then 14/18) — and raising the timeout from 20s to
    45s changed nothing. A request that fails identically at both limits is
    not slow, it is not completing. Failures were scattered across cities with
    no geographic, timezone, or loop-position pattern, which points at the
    connection rather than the endpoint: the old code opened a fresh TLS
    connection per city, 18 per run, from a shared Actions runner.
    SESSION now reuses one keep-alive connection with urllib3 transport-level
    retries underneath the application retry.
  - If this still drops cities, the next move is Open-Meteo's multi-coordinate
    form — one request carrying all 18 lat/lons — which removes the per-city
    request entirely.

FEATURES CAPTURED (per city, per run), all from Open-Meteo:
    - temperature_2m        (current modeled surface temp)
    - temperature_925hPa    (925mb temp — warm air aloft that can mix down)
    - temperature_850hPa    (850mb temp — mid-level thermal signal)
    - shortwave_radiation    (solar irradiance — the heating engine)
    - cloud_cover           (suppresses/allows heating)
    - wind_speed_10m, wind_direction_10m  (advection / sea-breeze)
    - apparent_temperature
    - V2.3: fc_cloud_12_15, fc_solar_12_15, fc_tmax_today (FORECAST, not obs)

Secrets needed: SUPABASE_URL, SUPABASE_KEY. (WETHR_API_KEY no longer used.)
ALWAYS exits 0 — a collection hiccup must never spam failure emails.
"""

import os
import sys
import time
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
HEADERS = {'User-Agent': 'intraday-collector/2.3', 'Accept': 'application/json'}

# Reject any Open-Meteo hour further than this from 'now'. Without this the
# nearest-hour search silently clamps to the end of the array and logs stale
# data as if it were live — the exact failure mode overnight collection hits.
MAX_HOUR_GAP_SECONDS = 5400  # 90 min

REQUEST_SPACING_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 45
RETRY_BACKOFF_SECONDS = 5

# V2.3: the heating window, in CITY-LOCAL hours. This is the window the cloud
# finding was measured over (10-15); 12-15 is used for the forecast because it
# is the part still ahead of the model at its 14:00 UTC bet time in every
# timezone, and because the high is made in the back half of that window.
FC_WINDOW_LOCAL = (12, 15)


def _make_session():
    """One pooled, keep-alive connection reused across all 18 cities.

    The per-request retry in V2.1 never once succeeded — a call that times out
    at 20s also times out at 45s, which means the request is not slow, it is
    not completing. Opening 18 fresh TLS connections from a shared Actions
    runner is the likely cause. A single pooled session with transport-level
    retries handles the connect/read failures below the requests layer."""
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(['GET']),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount('https://', adapter)
    s.headers.update(HEADERS)
    return s


SESSION = _make_session()

# All 18 cities. Coords, tz, and station live together so they cannot drift
# apart. Mirrors CITIES / CITY_TZ / WETHR_STATIONS in fetch_weather.py — if
# that file changes, change this too. (station retained for reference only.)
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


def _forecast_window(h, times, tz_name, today_local):
    """V2.3: average the FORECAST across local hours 12-15 of TODAY.

    The array is in UTC (timezone=UTC is requested). Convert each timestamp to
    the city's local time and keep the hours inside the window that fall on the
    city's own calendar day. Doing this in UTC hours instead would silently
    grab the wrong part of the day for Pacific and Mountain cities.

    Returns (cloud, solar, tmax, n_hours). Any of the first three may be None.
    n_hours is stored so a partial window is visible in the data rather than
    averaged as if complete — late in the day the window has already passed and
    only a couple of hours remain ahead.
    """
    tz = pytz.timezone(tz_name)
    lo, hi = FC_WINDOW_LOCAL

    clouds, solars, temps = [], [], []
    for i, t in enumerate(times):
        try:
            naive = datetime.strptime(t[:16], '%Y-%m-%dT%H:%M')
        except Exception:
            continue
        # times are UTC (timezone=UTC in params); localize then convert
        utc_dt = pytz.utc.localize(naive)
        loc = utc_dt.astimezone(tz)
        if loc.strftime('%Y-%m-%d') != today_local:
            continue
        if not (lo <= loc.hour <= hi):
            continue

        def val(key):
            arr = h.get(key)
            if isinstance(arr, list) and i < len(arr) and arr[i] is not None:
                try:
                    return float(arr[i])
                except Exception:
                    return None
            return None

        c = val('cloud_cover')
        s = val('shortwave_radiation')
        tt = val('temperature_2m')
        if c is not None:
            clouds.append(c)
        if s is not None:
            solars.append(s)
        if tt is not None:
            temps.append(tt)

    return (
        round(sum(clouds) / len(clouds), 1) if clouds else None,
        round(sum(solars) / len(solars), 2) if solars else None,
        round(max(temps), 2) if temps else None,
        len(clouds),
    )


def _open_meteo_once(lat, lon, tz_name, today_local):
    """Single attempt. Returns parsed features, or raises on timeout so the
    caller can retry, or returns {} on a non-retryable failure."""
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly': HOURLY_VARS,
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
        'past_days': 1,
        'forecast_days': 2,
    }
    r = SESSION.get(OPEN_METEO, params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS)
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

    # V2.3: the same response already contains the whole day. Previously
    # everything except the nearest hour was discarded.
    fc_cloud, fc_solar, fc_tmax, fc_hours = _forecast_window(
        h, times, tz_name, today_local)

    return {
        'temp_2m': g('temperature_2m'),
        'temp_925': g('temperature_925hPa'),
        'temp_850': g('temperature_850hPa'),
        'solar': g('shortwave_radiation'),
        'cloud_cover': g('cloud_cover'),
        'wind_speed': g('wind_speed_10m'),
        'wind_dir': g('wind_direction_10m'),
        'apparent_temp': g('apparent_temperature'),
        'fc_cloud_12_15': fc_cloud,
        'fc_solar_12_15': fc_solar,
        'fc_tmax_today': fc_tmax,
        'fc_hours_used': fc_hours,
    }


def fetch_open_meteo(lat, lon, tz_name, today_local):
    """Pull the atmospheric reversal features for the hour nearest now.
    Application-level retry sits on top of the session's transport retries.
    Returns {} on failure (logs loudly)."""
    for attempt in (1, 2):
        try:
            return _open_meteo_once(lat, lon, tz_name, today_local)
        except requests.exceptions.Timeout:
            if attempt == 1:
                print(f'      open-meteo timeout — retrying in '
                      f'{RETRY_BACKOFF_SECONDS}s')
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            print('      open-meteo timeout on retry — giving up')
            return {}
        except Exception as e:
            print(f'      open-meteo exc: {type(e).__name__}: {str(e)[:100]}')
            return {}
    return {}


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
    print(f'=== intraday collector v2.3 | ET {today} | '
          f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC | '
          f'{len(CITIES)} cities ===')
    print('  V2.3: also storing FORECAST cloud/solar for local hours '
          f'{FC_WINDOW_LOCAL[0]}-{FC_WINDOW_LOCAL[1]}')
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing logged (exit 0).')
        sys.exit(0)

    logged = 0
    for n, (city, cfg) in enumerate(CITIES.items()):
        if n:
            time.sleep(REQUEST_SPACING_SECONDS)
        ld = local_date(cfg['tz'])
        atmo = fetch_open_meteo(cfg['lat'], cfg['lon'], cfg['tz'], ld)
        if not atmo or atmo.get('temp_2m') is None:
            print(f'  [{city}] no atmospheric data — skipped')
            continue
        row = {
            'date': today, 'local_date': ld, 'city': city,
            'local_hour': local_hour(cfg['tz']),
            'temp_2m': atmo['temp_2m'], 'temp_925': atmo['temp_925'],
            'temp_850': atmo['temp_850'], 'solar': atmo['solar'],
            'cloud_cover': atmo['cloud_cover'], 'wind_speed': atmo['wind_speed'],
            'wind_dir': atmo['wind_dir'], 'apparent_temp': atmo['apparent_temp'],
            'fc_cloud_12_15': atmo['fc_cloud_12_15'],
            'fc_solar_12_15': atmo['fc_solar_12_15'],
            'fc_tmax_today': atmo['fc_tmax_today'],
            'fc_hours_used': atmo['fc_hours_used'],
            'source': 'open-meteo',
        }
        if sb_insert(row):
            logged += 1
            fc = atmo['fc_cloud_12_15']
            fc_str = (f'fc_cloud={fc} ({atmo["fc_hours_used"]}h)'
                      if fc is not None else 'fc_cloud=—')
            print(f'  [{city}] ld={ld} lh={row["local_hour"]} '
                  f't2m={atmo["temp_2m"]} 925={atmo["temp_925"]} '
                  f'solar={atmo["solar"]} cloud={atmo["cloud_cover"]} '
                  f'{fc_str} ✅')
        else:
            print(f'  [{city}] captured but DB write failed')

    print(f'\nLogged {logged}/{len(CITIES)} cities.')
    sys.exit(0)


if __name__ == '__main__':
    main()
