"""
wethr_probe.py — resolve wethr low-side data sourcing (DIAGNOSTIC, freeze-safe)
========================================================================
The lows smoke test got wethr=None on all 18 cities. The highs model pulls
'high' from the SAME endpoint with the SAME key successfully — so key + endpoint
are fine. The only open question: does the response carry a LOW value, and under
what field name?

This probe dumps the RAW wethr JSON for 3 spread-out cities so we can read the
actual field names.

RESOLVED 2026-08-22 (run #12):
  - nws_forecasts.php DOES carry a 'low' key, and it IS the calendar-day
    minimum: it equals min(hourly_temps) for all three probe cities.
  - observations.php mode='wethr_low' DOES NOT EXIST. Server enumerates its
    modes: 'latest', 'wethr_high', or omitted for history.
  - observations.php raw 'temperature' is INTEGER CELSIUS (Miami 28 vs
    temperature_display 82.4). Any min reconstruction must use 'temperature_f',
    not 'temperature', or it silently yields Celsius at ~0.9F granularity.
  - All precomputed extremes fields (cli_high/low, dsm_high/low, six_hour_*,
    twentyfour_hour_*) return None on the observations endpoint.

OPEN — what block 3 below tests:
  wethr.net/market/{city} displays "Est. NWS CLI high/low" with timestamps, and
  a 30-day "typical low" window. DevTools shows that page calling
  /api/v2/daily_extremes_api.php?station=KATL&days=31&logic=nws — a documented
  /api/v2/ path we have never called. If it returns dated high/low pairs, the
  §2 obs-history reconstruction (chunked <=24h pulls, local-day grouping,
  min of temperature_f) becomes unnecessary: one call per station returns a
  month of computed extremes, which can be diffed directly against Iowa CLI.

  Note the same page also calls marketv2.php?action=fetch_asos_data_dev — that
  one is an internal _dev path, NOT under /api/v2/, and should not be built on.

Needs WETHR_API_KEY in env (already wired in lows_smoke.yml). No model logic.
"""

import json
import os
from datetime import datetime

import requests

WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}

# 3 cities spread across timezones; Miami + LA are our tight-low allowlist leads
PROBE = {'Miami': 'KMIA', 'Chicago': 'KMDW', 'Los Angeles': 'KLAX'}

TODAY = datetime.utcnow().strftime('%Y-%m-%d')


def show(label, resp):
    print(f'    {label}: HTTP {resp.status_code}')
    if resp.status_code != 200:
        print(f'      body: {resp.text[:200]}')
        return
    try:
        data = resp.json()
    except Exception:
        print(f'      non-JSON body: {resp.text[:200]}')
        return
    if isinstance(data, dict):
        print(f'      keys: {list(data.keys())}')
        # surface anything low/min-ish
        for k, v in data.items():
            if any(t in k.lower() for t in ('low', 'min', 'high', 'temp')):
                print(f'        {k} = {v}')
    elif isinstance(data, list):
        print(f'      list of {len(data)} items')
        if data and isinstance(data[0], dict):
            print(f'      item[0] keys: {list(data[0].keys())}')
            print(f'      item[0]: {json.dumps(data[0])[:300]}')
    else:
        print(f'      {str(data)[:200]}')


def show_extremes(label, resp):
    """daily_extremes_api returns a dated series — print several entries whole
    rather than just item[0], so the date field and the high/low field names are
    both readable and we can see whether it covers past days or only today."""
    print(f'    {label}: HTTP {resp.status_code}')
    if resp.status_code != 200:
        print(f'      body: {resp.text[:300]}')
        return
    try:
        data = resp.json()
    except Exception:
        print(f'      non-JSON body: {resp.text[:300]}')
        return
    # response may be a bare list, or a dict wrapping one
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        print(f'      top-level keys: {list(data.keys())}')
        for k, v in data.items():
            if isinstance(v, list) and v:
                print(f'      (series under key {k!r})')
                rows = v
                break
        if rows is None:
            print(f'      {json.dumps(data)[:400]}')
            return
    if not rows:
        print('      empty series')
        return
    print(f'      {len(rows)} entries')
    if isinstance(rows[0], dict):
        print(f'      entry keys: {list(rows[0].keys())}')
    for row in rows[:5]:
        print(f'        {json.dumps(row)[:220]}')


def main():
    print(f'=== wethr probe ({TODAY}) | key present: {bool(WETHR_API_KEY)} ===\n')
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY is EMPTY in env — secret not reaching the run.')
        print('   (If this prints, the problem is the workflow env wiring, not wethr.)\n')

    for city, station in PROBE.items():
        print(f'[{city}]  station={station}')

        # 1) FORECAST endpoint — same call the highs model uses for 'high'.
        #    RESOLVED: 'low' here is the calendar-day min (== min(hourly_temps)).
        try:
            r = requests.get(
                'https://wethr.net/api/v2/nws_forecasts.php',
                params={'station_code': station, 'date': TODAY, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=15)
            show('nws_forecasts.php', r)
        except Exception as e:
            print(f'    nws_forecasts.php EXC: {type(e).__name__}: {str(e)[:120]}')

        # 2) latest observation — confirms the obs endpoint works and that raw
        #    'temperature' is integer Celsius while 'temperature_f' is usable.
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': station, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=15)
            show('observations.php mode=latest', r)
        except Exception as e:
            print(f'    observations latest EXC: {type(e).__name__}: {str(e)[:120]}')

        # 3) THE OPEN QUESTION — wethr's own computed daily high/low history.
        #    days=5 keeps the log readable; the page itself requests days=31.
        try:
            r = requests.get(
                'https://wethr.net/api/v2/daily_extremes_api.php',
                params={'station': station, 'days': 5, 'logic': 'nws'},
                headers=WETHR_HEADERS, timeout=15)
            show_extremes('daily_extremes_api.php', r)
        except Exception as e:
            print(f'    daily_extremes EXC: {type(e).__name__}: {str(e)[:120]}')

        print()

    print('=== Read: does daily_extremes_api return DATED high/low pairs, and do '
          'the dates go BACK several days? If yes, that is the CLI-validation '
          'set — one call per station, no obs reconstruction needed. ===')


if __name__ == '__main__':
    main()
