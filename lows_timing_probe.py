"""
lows_timing_probe.py — WHEN does the calendar-day minimum actually occur?
==========================================================================
DIAGNOSTIC ONLY. Read-only. No Supabase writes, no model logic, no bets.
Safe to run during the highs validation freeze.

WHY THIS EXISTS
---------------
V5.30_lows_spec.md §5 step 4 says: "Set cron windows LAST — only after observed-
min timing is characterized. Do NOT guess windows; that recreates the silent-
misalignment risk."

That was written assuming timing could only be learned by accumulating data
going forward. It can't be learned from the intraday collector either — that
only started 24h coverage on 2026-08-22, and temp_2m is MODELED surface temp,
not observation.

daily_extremes_api.php changes this. It returns low_time_utc per day per
station, on history, from the settled CLI record. So the timing distribution
can be measured RIGHT NOW over a month instead of waited for.

WHAT THIS ANSWERS
-----------------
1. Median local hour of the daily minimum, per city.
2. Spread — how tightly clustered is it? A city whose min always lands at
   06:00-07:00 local can be treated as locked after sunrise. A city with a
   10-hour spread cannot, and no cron window will fix that.
3. How often the minimum lands OUTSIDE the dawn window entirely. This is the
   real question. On 2026-08-23 Miami's temp_2m bottomed near 19:00 local,
   ~5F below anything the night produced, after an afternoon convective
   collapse. If that is chronic rather than a one-off storm, the "minimum is a
   ceiling that locks at sunrise" premise in the spec is FALSE for Miami and
   any city like it, and those cities need a convective gate rather than a
   timing window.

READ THE OUTPUT LIKE THIS
-------------------------
  - Tight IQR around dawn        → the spec's sunrise premise holds. A cron
                                   window just after the p90 hour is safe.
  - Wide IQR                     → timing is not predictable. Do not gate on
                                   time-of-day for this city; gate on observed
                                   value instead.
  - High late-day %              → convective risk city. The minimum can arrive
                                   after the market has been pricing a locked
                                   low all afternoon. This is an EDGE if modeled
                                   and a trap if not.

Cities with flat diurnal range (Boston showed 0.9F across ten hours on 08-23)
will show noisy timing that DOES NOT MATTER — when the whole day sits within a
degree, the min TIME is meaningless but the min VALUE is highly predictable.
Timing spread and value predictability are separate properties; do not read a
wide IQR as "unbettable" without checking the range column.

Needs WETHR_API_KEY in env. Run via the Lows Smoke Test workflow.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime

import pytz
import requests

# Days of history to request. The endpoint accepts at least 31.
DAYS_BACK = 31

# Local hours considered the "dawn window" — the span the spec assumes the
# minimum falls in. Used only to compute the outside-window percentage.
DAWN_START, DAWN_END = 3, 9

WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}

STATIONS = {
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


def fetch_extremes(station, days):
    try:
        r = requests.get(
            'https://wethr.net/api/v2/daily_extremes_api.php',
            params={'station': station, 'days': days, 'logic': 'nws'},
            headers=WETHR_HEADERS, timeout=20)
    except Exception as e:
        print(f'    ❌ EXC: {type(e).__name__}: {str(e)[:120]}')
        return []
    if r.status_code != 200:
        print(f'    ❌ HTTP {r.status_code}: {r.text[:160]}')
        return []
    try:
        data = r.json()
    except Exception:
        print(f'    ❌ non-JSON: {r.text[:160]}')
        return []
    rows = data.get('days') if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def utc_to_local_hour(utc_str, tz_name):
    """'2026-08-23 19:59:00' (UTC) → local hour as a float, e.g. 15.98."""
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(utc_str[:19], fmt)
            break
        except Exception:
            continue
    else:
        return None
    dt = pytz.utc.localize(dt).astimezone(pytz.timezone(tz_name))
    return dt.hour + dt.minute / 60.0


def fmt_hour(h):
    """15.98 → '15:59'."""
    if h is None:
        return '  —  '
    hh = int(h) % 24
    mm = int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = (hh + 1) % 24, 0
    return f'{hh:02d}:{mm:02d}'


def main():
    print('=== lows timing characterization: when does the daily min occur? ===')
    print(f'    {DAYS_BACK} days | dawn window = {DAWN_START:02d}:00-{DAWN_END:02d}:00 local'
          f' | key present: {bool(WETHR_API_KEY)}\n')
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY empty — calls will 401. Fix workflow env.\n')

    results = []

    for city, station in STATIONS.items():
        tz = CITY_TZ[city]
        rows = fetch_extremes(station, DAYS_BACK)
        if not rows:
            print(f'[{city}]  no data\n')
            continue

        hours, ranges, late_days = [], [], []
        skipped = 0
        for row in rows:
            lt = row.get('low_time_utc')
            lo = row.get('low')
            hi = row.get('high')
            if not lt:
                skipped += 1
                continue
            h = utc_to_local_hour(lt, tz)
            if h is None:
                skipped += 1
                continue
            hours.append(h)
            if lo is not None and hi is not None:
                try:
                    ranges.append(float(hi) - float(lo))
                except Exception:
                    pass
            if not (DAWN_START <= h < DAWN_END):
                late_days.append((row.get('date'), h, lo))

        if not hours:
            print(f'[{city}]  no usable low_time_utc ({skipped} skipped)\n')
            continue

        hours_sorted = sorted(hours)
        n = len(hours_sorted)
        median_h = statistics.median(hours_sorted)
        p10 = hours_sorted[max(0, int(n * 0.10) - 1)]
        p90 = hours_sorted[min(n - 1, int(n * 0.90))]
        iqr = p90 - p10
        avg_range = statistics.mean(ranges) if ranges else None
        late_pct = 100.0 * len(late_days) / n

        print(f'[{city}]  n={n}' + (f'  ({skipped} skipped)' if skipped else ''))
        print(f'    median min time : {fmt_hour(median_h)} local')
        print(f'    p10-p90 spread  : {fmt_hour(p10)} - {fmt_hour(p90)}  ({iqr:.1f}h)')
        if avg_range is not None:
            print(f'    avg diurnal rng : {avg_range:.1f}F')
        print(f'    outside dawn    : {len(late_days)}/{n}  ({late_pct:.0f}%)')
        for d, h, lo in late_days[:5]:
            print(f'        {d}  min at {fmt_hour(h)} local  (low {lo})')
        if len(late_days) > 5:
            print(f'        ... and {len(late_days) - 5} more')
        print()

        results.append({
            'city': city, 'n': n, 'median': median_h, 'p10': p10, 'p90': p90,
            'iqr': iqr, 'range': avg_range, 'late_pct': late_pct,
        })
        time.sleep(0.3)

    if not results:
        print('No results.')
        sys.exit(0)

    print('=== SUMMARY (sorted by timing tightness) ===')
    print(f'{"city":<16}{"n":>4}{"median":>9}{"p10-p90":>16}{"spread":>9}'
          f'{"rng":>7}{"late%":>8}')
    for r in sorted(results, key=lambda x: x['iqr']):
        rng = f'{r["range"]:.1f}' if r['range'] is not None else '—'
        print(f'{r["city"]:<16}{r["n"]:>4}{fmt_hour(r["median"]):>9}'
              f'{fmt_hour(r["p10"]) + "-" + fmt_hour(r["p90"]):>16}'
              f'{r["iqr"]:>8.1f}h{rng:>7}{r["late_pct"]:>7.0f}%')

    print()
    tight = [r for r in results if r['iqr'] <= 3.0]
    loose = [r for r in results if r['iqr'] > 6.0]
    convective = [r for r in results if r['late_pct'] >= 20.0]

    print(f'Tight timing (spread <=3h): {len(tight)}/{len(results)}')
    if tight:
        print('  ' + ', '.join(r['city'] for r in tight))
    print(f'Loose timing (spread >6h) : {len(loose)}/{len(results)}')
    if loose:
        print('  ' + ', '.join(r['city'] for r in loose))
    print(f'Late-min >=20% of days    : {len(convective)}/{len(results)}')
    if convective:
        print('  ' + ', '.join(f'{r["city"]} ({r["late_pct"]:.0f}%)'
                               for r in convective))
    print()
    print('→ Tight + narrow spread: the sunrise premise holds; a cron window')
    print('  just past p90 is safe for that city.')
    print('→ Loose spread but SMALL diurnal range: timing is meaningless but the')
    print('  VALUE is predictable. Gate on observed value, not on the clock.')
    print('→ High late%: convective-risk city. The min can arrive after the')
    print('  market has priced a locked low all afternoon. Needs its own gate.')

    sys.exit(0)


if __name__ == '__main__':
    main()
