"""
lows_value_probe.py — how tightly does the daily MINIMUM cluster, per city?
===========================================================================
DIAGNOSTIC ONLY. Read-only. No Supabase writes, no model logic, no bets.
Safe to run during the highs validation freeze.

WHY THIS EXISTS
---------------
lows_timing_probe.py (2026-08-23) killed the spec's central premise. §5/§6
assumed the calendar-day minimum is "a ceiling that locks at sunrise," so the
model could watch it form and fire when locked — the low-side version of the
afternoon certainty curve.

The 31-day timing data says otherwise:
  - Only 3/18 cities have tight timing (Dallas, Houston, New Orleans, <=3h).
  - 9/18 have p10-p90 spreads over 6h. Phoenix is 23.0h — its minimum has
    landed at essentially every hour of the day.
  - 8/18 put the minimum OUTSIDE the 03-09 dawn window on >=20% of days.
    Miami does it 48% of the time.

So there is no reliable "locked" moment for most cities, and a timing-gated
lows model would be wrong about half the time in the worst cases.

BUT the same run showed something that points the other way. Timing spread and
diurnal range are INVERSELY related:

    tight timing, big range : Dallas 20.9F, OKC 27.4F, Denver 31.2F
    loose timing, small rng : LA 11.1F, Miami 14.1F, Boston 14.1F, Chicago 14.3F

That is physically sensible — a large diurnal swing has a sharp, well-defined
bottom; a compressed one is flat overnight and any hour can win by a tenth of a
degree. It also means the loose-timing cities may still be highly predictable in
VALUE even though the timing is noise. Miami's lows over five days were
79, 80, 81, 79, 76 — a tight value cluster arriving at unpredictable hours.

THE QUESTION THIS ANSWERS
-------------------------
If the value is predictable, the model does not need timing at all. It can gate
on forecast value and ignore the clock entirely — no certainty curve, no
observed-min tracker, no cron window tuned to sunrise.

What decides that is dispersion RELATIVE TO BRACKET WIDTH. Kalshi low brackets
are 2F wide (e.g. "73-74", "75-76"). So:

  - day-over-day |change| small vs 2F  → consecutive days land in the same or
    adjacent bracket. Persistence alone is informative and the market is
    probably pricing it too.
  - stdev small vs 2F                  → the whole month sits in a couple of
    brackets. Bettable, but likely priced — check the Kalshi ladder before
    assuming edge exists.
  - stdev large vs 2F                  → wide outcome distribution. Needs a real
    forecast, not persistence. This is where a good model can actually pay.

WHAT IT REPORTS, PER CITY
-------------------------
  n, mean low, stdev, min, max, spread (max-min)
  mean |day-over-day change|, and how often that change is <=1F and <=2F
  brackets_spanned: (max-min)/2, i.e. how many 2F brackets the month covered

READ THE OUTPUT LIKE THIS
-------------------------
Low stdev is NOT automatically edge. A city whose low is the same 2F bracket
every day will have that bracket priced at 90c+, which is the same trap the
manual-grid work already found on the highs side: "a stable city that wins
every day at 90c has no edge left to capture." The interesting cities are the
ones where dispersion is moderate — wide enough that the market is uncertain,
tight enough that a model can beat it.

CAVEAT: this window is ~Jul 24 - Aug 23, peak summer. Both the timing tails and
the value dispersion are seasonal. Do not treat these numbers as fixed; re-run
in October and January before trusting any per-city tiering built on them.

Needs WETHR_API_KEY in env. Run via the Lows Smoke Test workflow.
"""

import os
import statistics
import sys
import time

import requests

DAYS_BACK = 31
BRACKET_WIDTH = 2.0  # Kalshi KXLOWT brackets are 2F wide

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


def main():
    print('=== lows VALUE dispersion: how predictable is the minimum itself? ===')
    print(f'    {DAYS_BACK} days | bracket width {BRACKET_WIDTH:.0f}F'
          f' | key present: {bool(WETHR_API_KEY)}\n')
    if not WETHR_API_KEY:
        print('⚠️ WETHR_API_KEY empty — calls will 401. Fix workflow env.\n')

    results = []

    for city, station in STATIONS.items():
        rows = fetch_extremes(station, DAYS_BACK)
        if not rows:
            print(f'[{city}]  no data\n')
            continue

        # sort by date so day-over-day deltas are real consecutive days
        dated = []
        for row in rows:
            d, lo = row.get('date'), row.get('low')
            if d and lo is not None:
                try:
                    dated.append((d, float(lo)))
                except Exception:
                    pass
        dated.sort(key=lambda x: x[0])
        lows = [v for _, v in dated]

        if len(lows) < 5:
            print(f'[{city}]  only {len(lows)} usable rows — skipped\n')
            continue

        n = len(lows)
        mean_low = statistics.mean(lows)
        stdev = statistics.pstdev(lows)
        lo_min, lo_max = min(lows), max(lows)
        spread = lo_max - lo_min

        deltas = [abs(lows[i] - lows[i - 1]) for i in range(1, n)]
        mean_delta = statistics.mean(deltas) if deltas else 0.0
        within_1 = 100.0 * sum(1 for d in deltas if d <= 1.0) / len(deltas) if deltas else 0.0
        within_2 = 100.0 * sum(1 for d in deltas if d <= 2.0) / len(deltas) if deltas else 0.0
        brackets_spanned = spread / BRACKET_WIDTH

        print(f'[{city}]  n={n}')
        print(f'    mean low {mean_low:.1f}F | stdev {stdev:.2f}F | '
              f'range {lo_min:.0f}-{lo_max:.0f}F (spread {spread:.0f}F)')
        print(f'    day-over-day |Δ| mean {mean_delta:.2f}F | '
              f'<=1F on {within_1:.0f}% of days | <=2F on {within_2:.0f}%')
        print(f'    month spanned ~{brackets_spanned:.1f} brackets '
              f'({BRACKET_WIDTH:.0f}F each)')
        print()

        results.append({
            'city': city, 'n': n, 'mean': mean_low, 'stdev': stdev,
            'spread': spread, 'delta': mean_delta, 'w1': within_1,
            'w2': within_2, 'brackets': brackets_spanned,
        })
        time.sleep(0.3)

    if not results:
        print('No results.')
        sys.exit(0)

    print('=== SUMMARY (sorted by value dispersion, tightest first) ===')
    print(f'{"city":<16}{"n":>4}{"mean":>7}{"stdev":>8}{"range":>11}'
          f'{"d/d |Δ|":>9}{"<=1F":>7}{"<=2F":>7}{"brkts":>7}')
    for r in sorted(results, key=lambda x: x['stdev']):
        rng = f'{r["mean"] - 0:.0f}'  # placeholder not used
        print(f'{r["city"]:<16}{r["n"]:>4}{r["mean"]:>7.1f}{r["stdev"]:>8.2f}'
              f'{r["spread"]:>10.0f}F{r["delta"]:>9.2f}{r["w1"]:>6.0f}%'
              f'{r["w2"]:>6.0f}%{r["brackets"]:>7.1f}')

    print()
    tight = [r for r in results if r['stdev'] <= 2.0]
    wide = [r for r in results if r['stdev'] >= 4.0]
    persistent = [r for r in results if r['w2'] >= 80.0]

    print(f'Tight value (stdev <=2F)   : {len(tight)}/{len(results)}')
    if tight:
        print('  ' + ', '.join(f'{r["city"]} ({r["stdev"]:.1f})' for r in tight))
    print(f'Wide value (stdev >=4F)    : {len(wide)}/{len(results)}')
    if wide:
        print('  ' + ', '.join(f'{r["city"]} ({r["stdev"]:.1f})' for r in wide))
    print(f'Persistent (d/d <=2F, 80%+): {len(persistent)}/{len(results)}')
    if persistent:
        print('  ' + ', '.join(r['city'] for r in persistent))

    print()
    print('→ Tight stdev + high persistence: the low is highly predictable and')
    print('  the market almost certainly knows. Check the Kalshi ladder before')
    print('  assuming edge — a 90c favorite that wins daily is not edge.')
    print('→ Wide stdev: outcome genuinely uncertain. This is where a forecast')
    print('  model can pay, and where sigma calibration actually matters.')
    print('→ Compare this ordering against lows_timing_probe: cities that are')
    print('  LOOSE on timing but TIGHT on value are the ones the timing-based')
    print('  premise would have wrongly excluded.')

    sys.exit(0)


if __name__ == '__main__':
    main()
