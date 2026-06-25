"""
wethr_probe.py — resolve the wethr=None blocker (DIAGNOSTIC, freeze-safe)
========================================================================
The lows smoke test got wethr=None on all 18 cities. The highs model pulls
'high' from the SAME endpoint with the SAME key successfully — so key + endpoint
are fine. The only open question: does the response carry a LOW value, and under
what field name?

This probe dumps the RAW wethr JSON for 3 spread-out cities so we can read the
actual field names. It checks two things:

  1. FORECAST source — wethr.net/api/v2/nws_forecasts.php
     Highs read data['high']. Is there a 'low' / 'min' / 'low_temp' key?
     → answers: where does the low FORECAST come from.

  2. OBSERVED-MIN source — wethr.net/api/v2/observations.php
     Highs use mode='wethr_high' to get the locked observed high. By symmetry
     there may be mode='wethr_low'. The tight coastal cities (Miami/LA/NOLA)
     bottom out near dawn and barely move, so an observed-min read is the most
     reliable path for them.
     → answers: where does the observed/locked low come from.

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


def main():
    print(f'=== wethr probe ({TODAY}) | key present: {bool(WETHR_API_KEY)} ===\n')
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY is EMPTY in env — secret not reaching the run.')
        print('   (If this prints, the problem is the workflow env wiring, not wethr.)\n')

    for city, station in PROBE.items():
        print(f'[{city}]  station={station}')

        # 1) FORECAST endpoint — same call the highs model uses for 'high'
        try:
            r = requests.get(
                'https://wethr.net/api/v2/nws_forecasts.php',
                params={'station_code': station, 'date': TODAY, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=15)
            show('nws_forecasts.php', r)
        except Exception as e:
            print(f'    nws_forecasts.php EXC: {type(e).__name__}: {str(e)[:120]}')

        # 2) OBSERVED-MIN — guess mode='wethr_low' by symmetry with wethr_high
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': station, 'mode': 'wethr_low', 'logic': 'nws'},
                headers=WETHR_HEADERS, timeout=15)
            show("observations.php mode=wethr_low", r)
        except Exception as e:
            print(f'    observations wethr_low EXC: {type(e).__name__}: {str(e)[:120]}')

        # 2b) fallback: plain latest observation, to confirm the obs endpoint works
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': station, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=15)
            show('observations.php mode=latest', r)
        except Exception as e:
            print(f'    observations latest EXC: {type(e).__name__}: {str(e)[:120]}')

        print()

    print('=== Read: in nws_forecasts.php keys, look for low/min. In wethr_low, '
          'look for a low value. That tells us forecast-source + obs-min-source. ===')


if __name__ == '__main__':
    main()
