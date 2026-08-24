"""
lows_source_validate.py — does wethr daily_extremes_api agree with Iowa CLI?
============================================================================
DIAGNOSTIC ONLY. Read-only. No Supabase writes, no model logic, no bets.
Safe to run during the highs validation freeze.

WHY THIS EXISTS
---------------
V5.30_lows_spec.md §2 specifies reconstructing the calendar-day minimum from
wethr's observation history: chunked <=24h pulls (wide windows return "Range
too large"), merged, grouped by city-local date, min taken over the series.
That was scoped in June off a 12-city-day hand check that matched Iowa CLI to
avg 0.32F.

On 2026-08-23 a different endpoint turned up in the wethr.net/market/{city}
page's network traffic:

    https://wethr.net/api/v2/daily_extremes_api.php?station=KATL&days=31&logic=nws

It is under the documented /api/v2/ namespace (same as observations.php and
nws_forecasts.php, both of which the highs model already uses successfully with
Bearer auth). It returns a dated series, one entry per day:

    date, high, low, high_time_utc, low_time_utc,
    high_source, low_source, dsm_rejected_high, dsm_rejected_low, is_final

For Miami it returned low_source="cli" on every row, and its 2026-08-22 low of
80 matched the Iowa CLI min of 80.0 that fetch_lows.py pulled independently the
same run.

If that agreement holds across all 18 cities and several days, then:
  - §2's reconstruction is unnecessary. Delete it. One call per station returns
    a month of settled extremes.
  - low_time_utc gives the per-city distribution of WHEN minima occur — which
    is exactly the timing characterization §5 step 4 says to do BEFORE setting
    cron windows, and it can now be done on history instead of by waiting.
  - is_final becomes the loud-not-silent settled flag §6 demands.

If it does NOT hold, we learn that here for ~$0 instead of after building on it.

WHAT THIS SCRIPT DOES
---------------------
For each of the 18 cities:
  1. GET daily_extremes_api.php (N days back)
  2. GET Iowa State cli.py for the same station-year
  3. Join on date, report per-day delta between wethr 'low' and CLI 'low'

Then prints a per-city summary (n compared, exact matches, mean abs delta, worst
delta) and a corpus total.

READ THE OUTPUT LIKE THIS
-------------------------
  - Exact match on ~all rows      → wethr IS relaying CLI. Use it as the source.
  - Small consistent deltas       → different rounding or a different instrument.
                                    Investigate before trusting; do NOT average
                                    the difference away.
  - Divergence on specific dates  → check is_final and dsm_rejected_low on those
                                    rows first; a non-final day is expected to
                                    disagree and is not evidence against the
                                    endpoint.
  - Divergence in specific cities → station mismatch. Check WETHR_STATIONS vs
                                    CLI_STATIONS for that city.

CAVEAT CARRIED FROM THE SPEC (§6): every fallback LOUD, never silent. This
script prints the reason for every missing comparison rather than skipping it.

Needs WETHR_API_KEY in env. Run via the Lows Smoke Test workflow.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import requests

# How many days of history to request and compare.
DAYS_BACK = 10

WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS = {'User-Agent': 'kalshi-lows-validate/1.0', 'Accept': 'application/json'}

# Mirrored from fetch_weather.py. wethr and CLI use the same station codes for
# all 18 — if a city diverges below, a station mismatch is the first suspect.
STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}


def fetch_wethr_extremes(station, days):
    """Returns {date_str: entry_dict} or {} on failure. Logs loudly."""
    try:
        r = requests.get(
            'https://wethr.net/api/v2/daily_extremes_api.php',
            params={'station': station, 'days': days, 'logic': 'nws'},
            headers=WETHR_HEADERS, timeout=20)
    except Exception as e:
        print(f'    ❌ wethr EXC: {type(e).__name__}: {str(e)[:120]}')
        return {}
    if r.status_code != 200:
        print(f'    ❌ wethr HTTP {r.status_code}: {r.text[:160]}')
        return {}
    try:
        data = r.json()
    except Exception:
        print(f'    ❌ wethr non-JSON: {r.text[:160]}')
        return {}

    rows = data.get('days') if isinstance(data, dict) else data
    if not isinstance(rows, list):
        print(f'    ❌ wethr unexpected shape: {json.dumps(data)[:200]}')
        return {}

    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get('date'):
            out[row['date']] = row
    return out


_CLI_CACHE = {}

def fetch_cli_year(station, year):
    """Iowa State cli.py for a station-year → {date_str: low_float}."""
    key = f'{station}_{year}'
    if key in _CLI_CACHE:
        return _CLI_CACHE[key]
    try:
        r = requests.get(
            'https://mesonet.agron.iastate.edu/json/cli.py',
            params={'station': station, 'year': year},
            headers=HEADERS, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'    ❌ CLI EXC: {type(e).__name__}: {str(e)[:120]}')
        _CLI_CACHE[key] = {}
        return {}

    lookup = {}
    for entry in data.get('results', []):
        valid = entry.get('valid', '')
        low = entry.get('low')
        if valid and low is not None:
            try:
                lookup[valid] = float(low)
            except Exception:
                pass
    _CLI_CACHE[key] = lookup
    return lookup


def main():
    print('=== lows source validation: wethr daily_extremes vs Iowa CLI ===')
    print(f'    days back: {DAYS_BACK} | key present: {bool(WETHR_API_KEY)}\n')
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY empty — wethr calls will 401. Fix workflow env.\n')

    today = datetime.utcnow().date()
    want_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d')
                  for i in range(1, DAYS_BACK + 1)]
    years = sorted({d[:4] for d in want_dates})

    corpus_n = 0
    corpus_exact = 0
    corpus_abs_sum = 0.0
    corpus_worst = (0.0, '', '')
    city_rows = []

    for city, station in STATIONS.items():
        print(f'[{city}]  station={station}')

        wethr = fetch_wethr_extremes(station, DAYS_BACK + 2)
        cli = {}
        for y in years:
            cli.update(fetch_cli_year(station, y))

        if not wethr:
            print('    → no wethr data, city skipped\n')
            city_rows.append((city, 0, 0, None, None))
            time.sleep(0.3)
            continue
        if not cli:
            print('    → no CLI data, city skipped\n')
            city_rows.append((city, 0, 0, None, None))
            time.sleep(0.3)
            continue

        n = exact = 0
        abs_sum = 0.0
        worst = (0.0, '')
        for d in want_dates:
            w_entry = wethr.get(d)
            c_low = cli.get(d)

            if w_entry is None and c_low is None:
                continue
            if w_entry is None:
                print(f'    {d}  wethr=—      CLI={c_low:>5}   (wethr missing)')
                continue
            if c_low is None:
                print(f'    {d}  wethr={w_entry.get("low"):>5}  CLI=—       (CLI missing)')
                continue

            w_low = w_entry.get('low')
            if w_low is None:
                print(f'    {d}  wethr low is null  CLI={c_low}')
                continue

            delta = float(w_low) - c_low
            n += 1
            abs_sum += abs(delta)
            if abs(delta) < 0.01:
                exact += 1
            if abs(delta) > abs(worst[0]):
                worst = (delta, d)

            flags = []
            if not w_entry.get('is_final', True):
                flags.append('NOT FINAL')
            if w_entry.get('dsm_rejected_low'):
                flags.append('dsm_rejected_low')
            src = w_entry.get('low_source')
            if src and src != 'cli':
                flags.append(f'source={src}')
            flag_s = ('  ⚠️ ' + ', '.join(flags)) if flags else ''

            mark = '✅' if abs(delta) < 0.01 else '❌'
            print(f'    {d}  wethr={w_low:>5}  CLI={c_low:>5}  '
                  f'Δ={delta:+.1f}  {mark}{flag_s}')

        if n:
            mae = abs_sum / n
            print(f'    → {exact}/{n} exact | MAE {mae:.2f}F | '
                  f'worst {worst[0]:+.1f}F on {worst[1]}')
            city_rows.append((city, n, exact, mae, worst))
            corpus_n += n
            corpus_exact += exact
            corpus_abs_sum += abs_sum
            if abs(worst[0]) > abs(corpus_worst[0]):
                corpus_worst = (worst[0], worst[1], city)
        else:
            print('    → no overlapping dates to compare')
            city_rows.append((city, 0, 0, None, None))

        print()
        time.sleep(0.3)

    print('=== SUMMARY ===')
    print(f'{"city":<16}{"n":>4}{"exact":>7}{"MAE":>8}   worst')
    for city, n, exact, mae, worst in city_rows:
        if n:
            print(f'{city:<16}{n:>4}{exact:>7}{mae:>8.2f}   '
                  f'{worst[0]:+.1f} on {worst[1]}')
        else:
            print(f'{city:<16}{"—":>4}{"—":>7}{"—":>8}   —')

    print()
    if corpus_n:
        print(f'CORPUS: {corpus_exact}/{corpus_n} exact '
              f'({100.0 * corpus_exact / corpus_n:.1f}%) | '
              f'MAE {corpus_abs_sum / corpus_n:.2f}F')
        if corpus_worst[1]:
            print(f'worst single day: {corpus_worst[0]:+.1f}F  '
                  f'{corpus_worst[2]} {corpus_worst[1]}')
        print()
        if corpus_exact == corpus_n:
            print('→ EXACT across the corpus. wethr is relaying CLI. Use it as')
            print('  the lows settlement source and delete §2 reconstruction.')
        elif corpus_exact >= corpus_n * 0.95:
            print('→ Near-exact. Inspect the mismatched rows above for is_final /')
            print('  dsm_rejected_low before concluding anything — a non-final day')
            print('  is EXPECTED to disagree and is not evidence against wethr.')
        else:
            print('→ Material disagreement. Do NOT build on this endpoint yet.')
            print('  Check per-city column: divergence isolated to a few cities is')
            print('  a station mismatch; divergence everywhere means it is not CLI.')
    else:
        print('CORPUS: nothing compared — see the loud failures above.')

    sys.exit(0)


if __name__ == '__main__':
    main()
