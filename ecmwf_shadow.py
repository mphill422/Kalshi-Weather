"""
ecmwf_shadow.py — ECMWF shadow logger (SHADOW ONLY, freeze-safe)
================================================================
Pulls the ECMWF ensemble daily-high forecast for all 18 cities once a day and
writes it to its OWN table (ecmwf_shadow). It does NOT touch fetch_weather.py,
consensus, bracket selection, bets, or settlements. The live model is
completely unaware this exists. Purpose: accumulate ECMWF-vs-actual data so the
integration decision can be made on evidence, not speculation.

Compare later (when back at a Mac):
    select e.city, e.date, e.ecmwf_high_mean, s.consensus, s.actual,
           round((e.ecmwf_high_mean - s.actual)::numeric,2)  as ecmwf_err,
           round((s.consensus       - s.actual)::numeric,2)  as model_err
    from ecmwf_shadow e
    join settlements s on e.city=s.city and e.date=s.date
    where s.actual is not null
    order by e.date, e.city;
  → if |ecmwf_err| is consistently smaller than |model_err|, ECMWF earns a
    place in the blend. If not, the idea is dead and you stop wondering.

CRITICAL DESIGN: this script ALWAYS exits 0, even on total failure. It must
never fail the workflow, because a failed workflow emails you — and you're on
vacation. A dead ECMWF log is harmless; a spammed phone is not.

Create the table once before first run (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.ecmwf_shadow (
    id BIGSERIAL PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date TEXT NOT NULL,                 -- ET date 'YYYY-MM-DD'
    city TEXT NOT NULL,
    ecmwf_high_mean NUMERIC(6,2),       -- mean of member daily-maxes
    ecmwf_high_spread NUMERIC(6,2),     -- stdev across members (uncertainty)
    ecmwf_high_min NUMERIC(6,2),
    ecmwf_high_max NUMERIC(6,2),
    n_members INTEGER,
    source TEXT,                        -- which endpoint/model produced it
    UNIQUE (date, city)
  );
  ALTER TABLE public.ecmwf_shadow ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.ecmwf_shadow
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

Needs SUPABASE_URL + SUPABASE_KEY (already in repo secrets). No wethr, no Kalshi.
"""

import os
import sys
import statistics
from datetime import datetime

import requests
import pytz

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
HEADERS = {'User-Agent': 'ecmwf-shadow/1.0', 'Accept': 'application/json'}

CITIES = {
    'Phoenix': (33.4342, -112.0116), 'Las Vegas': (36.0840, -115.1537),
    'Los Angeles': (33.9416, -118.4085), 'Dallas': (32.8998, -97.0403),
    'Austin': (30.1945, -97.6699), 'Houston': (29.9902, -95.3368),
    'Atlanta': (33.6407, -84.4277), 'Miami': (25.7959, -80.2870),
    'New York': (40.7812, -73.9665), 'San Antonio': (29.5337, -98.4698),
    'New Orleans': (29.9934, -90.2580), 'Philadelphia': (39.8744, -75.2424),
    'Boston': (42.3656, -71.0096), 'Denver': (39.8561, -104.6737),
    'Oklahoma City': (35.3931, -97.6007), 'Minneapolis': (44.8848, -93.2223),
    'Washington DC': (38.8512, -77.0402), 'Chicago': (41.7868, -87.7522),
}

# ECMWF model strings to try, in order. First that returns ≥3 member maxes wins.
ECMWF_ENSEMBLE_MODELS = ['ecmwf_ifs025', 'ecmwf_ifs04']


def et_date():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')


def _member_maxes_from_hourly(hourly, today):
    """Per-member daily max over today's daytime hours (6-21 local)."""
    times = hourly.get('time', [])
    today_idx = [i for i, t in enumerate(times)
                 if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21]
    if not today_idx:
        today_idx = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_idx:
        return []
    maxes = []
    for key, vals in hourly.items():
        if key == 'time' or 'temperature_2m' not in key or not isinstance(vals, list):
            continue
        day_vals = [vals[i] for i in today_idx if i < len(vals) and vals[i] is not None]
        if day_vals:
            try:
                maxes.append(round(max(float(v) for v in day_vals), 1))
            except Exception:
                pass
    return maxes


def fetch_ecmwf(lat, lon, today):
    """Try ECMWF ensemble models, then deterministic ECMWF as last resort.
    Returns (member_maxes_list, source_str) or ([], reason_str)."""
    for model in ECMWF_ENSEMBLE_MODELS:
        params = {'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m',
                  'temperature_unit': 'fahrenheit', 'timezone': 'auto',
                  'forecast_days': 2, 'models': model}
        try:
            r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble',
                             params=params, headers=HEADERS, timeout=45)
            if r.status_code == 200:
                maxes = _member_maxes_from_hourly(r.json().get('hourly', {}), today)
                if len(maxes) >= 3:
                    return maxes, f'ensemble:{model}'
        except Exception as e:
            print(f'      {model} ensemble exc: {type(e).__name__}: {str(e)[:80]}')

    # last resort: deterministic ECMWF single forecast (1 "member")
    params = {'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m',
              'temperature_unit': 'fahrenheit', 'timezone': 'auto',
              'forecast_days': 2, 'models': 'ecmwf_ifs025'}
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            maxes = _member_maxes_from_hourly(r.json().get('hourly', {}), today)
            if maxes:
                return maxes, 'forecast:ecmwf_ifs025'
    except Exception as e:
        print(f'      deterministic exc: {type(e).__name__}: {str(e)[:80]}')
    return [], 'no_data'


def sb_upsert(row):
    """Upsert on (date,city). Best-effort; never raises."""
    try:
        headers = {'apikey': SUPABASE_KEY, 'Authorization': 'Bearer ' + SUPABASE_KEY,
                   'Content-Type': 'application/json',
                   'Prefer': 'resolution=merge-duplicates,return=minimal'}
        r = requests.post(SUPABASE_URL + '/rest/v1/ecmwf_shadow?on_conflict=date,city',
                          headers=headers, json=row, timeout=15)
        return r.status_code in (200, 201, 204)
    except Exception as e:
        print(f'      sb_upsert exc: {type(e).__name__}: {str(e)[:80]}')
        return False


def main():
    today = et_date()
    print(f'=== ECMWF shadow | ET {today} | {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC ===')
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing logged (exiting 0 anyway).')
        sys.exit(0)

    logged = 0
    for city, (lat, lon) in CITIES.items():
        maxes, source = fetch_ecmwf(lat, lon, today)
        if not maxes:
            print(f'  [{city}] no ECMWF data ({source})')
            continue
        mean = round(statistics.mean(maxes), 2)
        spread = round(statistics.stdev(maxes), 2) if len(maxes) > 1 else 0.0
        row = {'date': today, 'city': city, 'ecmwf_high_mean': mean,
               'ecmwf_high_spread': spread, 'ecmwf_high_min': round(min(maxes), 2),
               'ecmwf_high_max': round(max(maxes), 2), 'n_members': len(maxes),
               'source': source}
        if sb_upsert(row):
            logged += 1
            print(f'  [{city}] mean={mean}F spread={spread} n={len(maxes)} ({source}) ✅')
        else:
            print(f'  [{city}] mean={mean}F but DB write failed ({source})')

    print(f'\nLogged {logged}/{len(CITIES)} cities.')
    # ALWAYS exit 0 — never spam a vacationing phone with workflow-failure emails.
    sys.exit(0)


if __name__ == '__main__':
    main()
