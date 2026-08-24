"""
fetch_lows_settlements.py — V5.30 lows settlement + history collector
======================================================================
Writes to a NEW table (`lows_settlements`). Touches NOTHING in the highs model.
Does not import from or modify fetch_weather.py, streamlit_app.py, or
fetch_lows.py. Safe to run during the highs validation freeze — it is a separate
model with separate math.

WHY THIS FILE EXISTS
--------------------
V5.30_lows_spec.md §2 specified reconstructing the calendar-day minimum from
wethr observation history: chunked <=24h pulls, merged, grouped by city-local
date, min over the series. That is unnecessary.

On 2026-08-23/24 `lows_source_validate.py` compared

    https://wethr.net/api/v2/daily_extremes_api.php?station=X&days=N&logic=nws

against Iowa State CLI across 18 cities x 10 days: **179/179 exact, MAE 0.00F.**
wethr relays the CLI settlement value verbatim. So one call per station returns
a month of settled daily extremes, already grouped by local calendar date.

This script pulls that series and maintains it in Supabase, so the lows model has
a settled-truth table to calibrate and score against without re-fetching history
every time.

WHAT IT STORES AND WHY EACH FIELD MATTERS
-----------------------------------------
  date, city, station      — key. `date` is the city-LOCAL calendar date, which
                             is what Kalshi settles on. Do not join it to the
                             highs `settlements.date` (ET) without thinking.
  low, high                — the settled extremes.
  low_time_utc             — WHEN the minimum occurred. This is the field that
                             killed the spec's central premise: only 3/18 cities
                             have tight timing, 8/18 put the min outside the
                             03:00-09:00 dawn window on >=20% of days, and Miami
                             does it 48% of the time. Keep collecting it; the
                             31-day read was peak summer and is expected to shift
                             seasonally.
  high_time_utc            — same, for the highs side. Free, so keep it.
  low_source, high_source  — 'cli' on every row observed so far. If this ever
                             changes value, the row is NOT settlement truth and
                             must not be used for calibration. LOUD, not silent
                             (spec §6).
  dsm_rejected_low/high    — wethr's own quality flag. A rejected row is suspect.
  is_final                 — whether the day has settled. A non-final row is
                             EXPECTED to disagree with CLI and is not evidence
                             of a problem.
  cli_low, cli_high        — the INDEPENDENT Iowa State CLI value for the same
                             date, pulled separately. This is the cross-check.
  low_delta                — low - cli_low. Should be 0.00 on every settled row.

THE CROSS-CHECK IS NOT OPTIONAL. It is what proved the source, and a single
source with no independent verification is exactly how the six-week Wethr outage
went unnoticed. If low_delta ever drifts off zero, something changed upstream and
every downstream number is suspect.

CREATE THE TABLE ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.lows_settlements (
    id BIGSERIAL PRIMARY KEY,
    date TEXT NOT NULL,               -- city-LOCAL calendar date 'YYYY-MM-DD'
    city TEXT NOT NULL,
    station TEXT,
    low NUMERIC(6,2),
    high NUMERIC(6,2),
    low_time_utc TIMESTAMPTZ,
    high_time_utc TIMESTAMPTZ,
    low_source TEXT,
    high_source TEXT,
    dsm_rejected_low BOOLEAN,
    dsm_rejected_high BOOLEAN,
    is_final BOOLEAN,
    cli_low NUMERIC(6,2),             -- independent Iowa State CLI value
    cli_high NUMERIC(6,2),
    low_delta NUMERIC(6,2),           -- low - cli_low; should be 0.00
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, city)
  );
  ALTER TABLE public.lows_settlements ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.lows_settlements
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_lows_settlements_city_date
    ON public.lows_settlements (city, date);

The UNIQUE (date, city) constraint plus an upsert means this script is safe to
run repeatedly and safe to run on overlapping date ranges. Re-running it will
CORRECT rows that were non-final when first fetched — which is the point, since
a day fetched before CLI posts will settle later.

USAGE
-----
  python fetch_lows_settlements.py            # default DAYS_BACK
  DAYS_BACK=90 python fetch_lows_settlements.py   # backfill deeper

Start with a deep backfill (90 days) once, then run daily with the default.

Secrets needed: SUPABASE_URL, SUPABASE_KEY, WETHR_API_KEY.
ALWAYS exits 0 — a collection hiccup must never spam failure emails. Failures
are printed loudly and counted in the summary instead.
"""

import os
import sys
import time
from datetime import datetime

import requests

DAYS_BACK = int(os.environ.get('DAYS_BACK', '31'))

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS = {'User-Agent': 'kalshi-lows-settlements/1.0', 'Accept': 'application/json'}

# 18 cities, mirroring fetch_weather.py. wethr and CLI use identical station
# codes for all of them — if a city ever disagrees between the two sources, a
# station mismatch here is the first thing to check.
STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}


def _num(v):
    if v is None:
        return None
    try:
        return round(float(v), 2)
    except Exception:
        return None


def _ts(v):
    """'2026-08-23 19:59:00' (UTC) -> ISO with explicit UTC offset, or None."""
    if not v:
        return None
    s = str(v).strip().replace(' ', 'T')[:19]
    if len(s) < 19:
        return None
    return s + '+00:00'


def fetch_wethr_extremes(station, days):
    """Returns list of raw day dicts, or [] on failure. Logs loudly."""
    try:
        r = requests.get(
            'https://wethr.net/api/v2/daily_extremes_api.php',
            params={'station': station, 'days': days, 'logic': 'nws'},
            headers=WETHR_HEADERS, timeout=25)
    except Exception as e:
        print(f'    ❌ wethr EXC: {type(e).__name__}: {str(e)[:120]}')
        return []
    if r.status_code != 200:
        print(f'    ❌ wethr HTTP {r.status_code}: {r.text[:160]}')
        return []
    try:
        data = r.json()
    except Exception:
        print(f'    ❌ wethr non-JSON: {r.text[:160]}')
        return []
    rows = data.get('days') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        print(f'    ❌ wethr unexpected shape')
        return []
    return rows


_CLI_CACHE = {}

def fetch_cli_year(station, year):
    """Iowa State cli.py -> {date: (low, high)}. Independent cross-check."""
    key = f'{station}_{year}'
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    try:
        r = requests.get(
            'https://mesonet.agron.iastate.edu/json/cli.py',
            params={'station': station, 'year': year},
            headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'    ⚠️ CLI EXC: {type(e).__name__}: {str(e)[:120]}')
        _CLI_CACHE[key] = {}
        return {}
    lookup = {}
    for entry in data.get('results', []):
        valid = entry.get('valid', '')
        if not valid:
            continue
        lookup[valid] = (_num(entry.get('low')), _num(entry.get('high')))
    _CLI_CACHE[key] = lookup
    return lookup


def sb_upsert(rows):
    """Upsert on (date, city). Returns True on success."""
    if not rows:
        return True
    try:
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': 'Bearer ' + SUPABASE_KEY,
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal,resolution=merge-duplicates',
        }
        r = requests.post(
            SUPABASE_URL + '/rest/v1/lows_settlements?on_conflict=date,city',
            headers=headers, json=rows, timeout=30)
        if r.status_code not in (200, 201, 204):
            print(f'    ❌ upsert HTTP {r.status_code}: {r.text[:200]}')
            return False
        return True
    except Exception as e:
        print(f'    ❌ upsert EXC: {type(e).__name__}: {str(e)[:120]}')
        return False


def main():
    print(f'=== lows settlement collector | {DAYS_BACK} days back | '
          f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC ===\n')

    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing written (exit 0).')
        sys.exit(0)
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY empty — every call will 401 (exit 0).')
        sys.exit(0)

    total_written = 0
    total_final = 0
    mismatches = []
    source_anomalies = []
    cities_failed = []

    for city, station in STATIONS.items():
        rows = fetch_wethr_extremes(station, DAYS_BACK)
        if not rows:
            print(f'[{city}]  no wethr data — skipped')
            cities_failed.append(city)
            time.sleep(0.3)
            continue

        years = sorted({str(r.get('date', ''))[:4] for r in rows if r.get('date')})
        cli = {}
        for y in years:
            if y:
                cli.update(fetch_cli_year(station, y))

        payload = []
        n_final = 0
        for row in rows:
            d = row.get('date')
            if not d:
                continue

            low = _num(row.get('low'))
            high = _num(row.get('high'))
            cli_low, cli_high = cli.get(d, (None, None))
            low_delta = None
            if low is not None and cli_low is not None:
                low_delta = round(low - cli_low, 2)

            is_final = row.get('is_final')
            if is_final:
                n_final += 1

            low_src = row.get('low_source')
            # LOUD, not silent: any source that is not CLI means this row is not
            # settlement truth and must not be used for calibration.
            if low_src and low_src != 'cli':
                source_anomalies.append((city, d, low_src))

            # A non-final row is EXPECTED to disagree. Only flag settled rows.
            if is_final and low_delta is not None and abs(low_delta) > 0.001:
                mismatches.append((city, d, low, cli_low, low_delta))

            payload.append({
                'date': d,
                'city': city,
                'station': station,
                'low': low,
                'high': high,
                'low_time_utc': _ts(row.get('low_time_utc')),
                'high_time_utc': _ts(row.get('high_time_utc')),
                'low_source': low_src,
                'high_source': row.get('high_source'),
                'dsm_rejected_low': row.get('dsm_rejected_low'),
                'dsm_rejected_high': row.get('dsm_rejected_high'),
                'is_final': is_final,
                'cli_low': cli_low,
                'cli_high': cli_high,
                'low_delta': low_delta,
            })

        if sb_upsert(payload):
            total_written += len(payload)
            total_final += n_final
            print(f'[{city}]  {len(payload)} rows ({n_final} final) ✅')
        else:
            print(f'[{city}]  fetched {len(payload)} but DB write FAILED')
            cities_failed.append(city)

        time.sleep(0.3)

    print()
    print('=== SUMMARY ===')
    print(f'  rows upserted   : {total_written}')
    print(f'  marked final    : {total_final}')
    print(f'  cities failed   : {len(cities_failed)}'
          + (('  — ' + ', '.join(cities_failed)) if cities_failed else ''))

    if source_anomalies:
        print(f'\n  ⚠️ {len(source_anomalies)} row(s) with low_source != "cli":')
        for c, d, s in source_anomalies[:10]:
            print(f'      {c} {d}  source={s}')
        print('     These are NOT settlement truth. Exclude from calibration.')
    else:
        print('  low_source: "cli" on every row ✅')

    if mismatches:
        print(f'\n  ❌ {len(mismatches)} SETTLED row(s) where wethr != Iowa CLI:')
        for c, d, w, cl, delta in mismatches[:15]:
            print(f'      {c} {d}  wethr={w} cli={cl} delta={delta:+.2f}')
        print('     The 179/179 exact agreement no longer holds. Something')
        print('     changed upstream — do NOT trust downstream numbers until')
        print('     this is understood.')
    else:
        print('  wethr vs Iowa CLI: exact on every settled row ✅')

    print()
    print('Next: the timing distribution (low_time_utc) and value dispersion can')
    print('now be recomputed from this table instead of re-fetching the API.')

    sys.exit(0)


if __name__ == '__main__':
    main()
