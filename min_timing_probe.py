"""
min_timing_probe.py — when do the observed-min fields populate? (DIAGNOSTIC)
===========================================================================
The wethr observations.php mode=latest response carries the observed-min
fields we need for low settlement / obs-min tracking:

    cli_low              official CLI minimum (the Kalshi settlement value)
    dsm_low              daily summary message low
    twentyfour_hour_low  rolling 24h observed min
    six_hour_low         rolling 6h observed min

At midday they read None (the min reports after it's locked). We don't know
WHEN each flips to a value, and it may differ by city. That populate-time is
exactly what sets the V5.30 cron windows — fire a window before the field is
live and you log nothing (same failure as the afternoon hourly-min garbage).

This probe prints, per city, the local time and the current state of each min
field (None vs value), plus the forecast low for reference. Run it hourly
across tomorrow morning→afternoon; read the run history to find the first hour
each field goes non-null per city. That gives the windows.

Needs WETHR_API_KEY (already wired in lows_smoke.yml). Read-only, no Supabase.
"""

import os
from datetime import datetime

import requests
import pytz

WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
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

# Our tight-low allowlist leads first, so they're easy to scan in the log
PRIORITY = ['Miami', 'New Orleans', 'Los Angeles', 'Houston']
ORDER = PRIORITY + [c for c in WETHR_STATIONS if c not in PRIORITY]


def fmt(v):
    return 'None' if v is None else str(v)


def get_obs(station):
    try:
        r = requests.get('https://wethr.net/api/v2/observations.php',
                         params={'station_code': station, 'mode': 'latest'},
                         headers=WETHR_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_fcst_low(station):
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        r = requests.get('https://wethr.net/api/v2/nws_forecasts.php',
                         params={'station_code': station, 'date': today, 'mode': 'latest'},
                         headers=WETHR_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get('low')
    except Exception:
        pass
    return None


def main():
    utc = datetime.utcnow()
    print(f'=== min-timing probe | UTC {utc.strftime("%Y-%m-%d %H:%M")} | '
          f'key present: {bool(WETHR_API_KEY)} ===')
    print(f'{"city":14s} {"local":>6s}  {"fcst":>4s} | {"cli_low":>8s} '
          f'{"dsm_low":>8s} {"24h_low":>8s} {"6h_low":>7s}')
    print('-' * 72)

    populated = []
    for city in ORDER:
        station = WETHR_STATIONS[city]
        lt = datetime.now(pytz.timezone(CITY_TZ[city])).strftime('%H:%M')
        obs = get_obs(station)
        fcst = get_fcst_low(station)

        if obs is None:
            print(f'{city:14s} {lt:>6s}  {fmt(fcst):>4s} | (obs fetch failed)')
            continue

        cli_low = obs.get('cli_low_f') if obs.get('cli_low_f') is not None else obs.get('cli_low')
        dsm_low = obs.get('dsm_low_f') if obs.get('dsm_low_f') is not None else obs.get('dsm_low')
        h24_low = obs.get('twentyfour_hour_low')
        h6_low  = obs.get('six_hour_low_f') if obs.get('six_hour_low_f') is not None else obs.get('six_hour_low')

        if cli_low is not None or h24_low is not None:
            populated.append(city)

        print(f'{city:14s} {lt:>6s}  {fmt(fcst):>4s} | {fmt(cli_low):>8s} '
              f'{fmt(dsm_low):>8s} {fmt(h24_low):>8s} {fmt(h6_low):>7s}')

    print('-' * 72)
    if populated:
        print(f'✅ min field LIVE this run for: {", ".join(populated)}')
    else:
        print('… no city has cli_low / 24h_low populated yet this run')
    print('(Run hourly. First run a city appears in the LIVE line = its populate '
          'time = its cron window.)')


if __name__ == '__main__':
    main()
