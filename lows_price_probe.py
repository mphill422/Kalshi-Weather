"""
lows_price_probe.py — is the predictable low already PRICED?
=============================================================
DIAGNOSTIC ONLY. Read-only. No Supabase writes, no model logic, no bets.
Safe to run during the highs validation freeze.

WHY THIS EXISTS
---------------
lows_value_probe.py (2026-08-23) found four cities whose daily minimum is
extremely predictable over 31 days:

    Los Angeles   stdev 1.06F   d/d 0.67F   within 2F on 100% of days   1.5 brackets
    San Antonio   stdev 1.29F   d/d 1.07F   within 2F on  93% of days   2.5 brackets
    Houston       stdev 1.46F   d/d 1.47F   within 2F on  80% of days   4.0 brackets
    New Orleans   stdev 1.62F   d/d 1.30F   within 2F on  93% of days   3.0 brackets

Predictable is NOT the same as profitable. The manual-grid work on the highs
side already established the trap in one line: "a stable city that wins every
day at 90c has no edge left to capture." If LA's low sits in the same couple of
brackets all month, the market has 31 days of the same evidence we do, and the
favorite will be priced at 90c+ where there is nothing left to take.

The interesting band is the MIDDLE — dispersed enough that the market cannot
just price persistence, tight enough that a forecast can beat it:

    Atlanta 2.09 | Dallas 2.35 | Austin 2.46 | Miami 2.66

And the WIDE tier (Denver 5.51, Las Vegas 5.90, Phoenix 5.26, Minneapolis 4.95)
is where a genuinely good forecast plus honest sigma would have to earn its
keep — high uncertainty means cheap brackets, but also means being right is
hard.

WHAT THIS MEASURES
------------------
For each city, right now, off the live Kalshi KXLOWT ladder:

  top_price      — yes-ask on the most expensive bracket. This is the market's
                   confidence in its favorite. High = priced, little left.
  top2_sum       — combined price of the two most expensive brackets. The
                   highs-side manual research used <=85c sum-of-top-two as an
                   entry filter against a ~94.8% top-two hit rate. Same
                   arithmetic applies here.
  n_above_10c    — how many brackets the market considers live at all. A city
                   pricing 1-2 brackets above 10c is a market that has made up
                   its mind; 4+ means genuine uncertainty.
  implied_spread — a crude read on how much probability mass sits outside the
                   favorite.

Then it joins that against the 31-day dispersion so the two can be read
together in one table.

READ THE OUTPUT LIKE THIS
-------------------------
  - tight stdev + top_price >=85c   → PRICED. The predictability is already in
                                      the market. Skip, or look only for the
                                      rare mispriced day.
  - tight stdev + top_price <=70c   → the market is MORE uncertain than the
                                      history says it should be. This is the
                                      signal worth chasing.
  - wide stdev + top_price >=80c    → market is more confident than the history
                                      justifies. Potential NO-side edge, but
                                      also the most likely place for the model
                                      to be the one that is wrong.
  - wide stdev + cheap ladder       → everyone agrees it is uncertain. Needs
                                      real forecast skill, not structure.

CAVEATS
-------
1. This is ONE snapshot, at whatever time of day the workflow runs. Prices move
   through the day exactly as they do on the highs side. A single reading is a
   sanity check, not a price history. Run it at several times before drawing
   conclusions about any specific city.
2. Late-evening runs are the WORST case for lows: for most cities tomorrow's
   minimum is hours away and the ladder is thin and wide. Early-morning runs,
   near the median 06:00-07:30 minimum, will look completely different.
3. Dispersion numbers are ~Jul 24 - Aug 23, peak summer. Seasonal.

Needs WETHR_API_KEY (unused here, kept for workflow symmetry). Uses the same
public Kalshi endpoint and parsing as fetch_lows.py.

Run via the Lows Smoke Test workflow.
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta

import pytz
import requests

HEADERS = {'User-Agent': 'kalshi-lows-price-probe/1.0', 'Accept': 'application/json'}
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}

DAYS_BACK = 31

# 31-day dispersion, measured 2026-08-23 by lows_value_probe.py. Hardcoded so
# this probe reads as one table instead of two. Re-measure seasonally.
DISPERSION = {
    'Los Angeles': 1.06, 'San Antonio': 1.29, 'Houston': 1.46,
    'New Orleans': 1.62, 'Atlanta': 2.09, 'Dallas': 2.35,
    'Austin': 2.46, 'Miami': 2.66, 'Chicago': 3.12, 'New York': 3.28,
    'Washington DC': 3.33, 'Philadelphia': 3.69, 'Oklahoma City': 4.01,
    'Boston': 4.37, 'Minneapolis': 4.95, 'Phoenix': 5.26,
    'Denver': 5.51, 'Las Vegas': 5.90,
}

LOW_SERIES = {
    'Phoenix': 'KXLOWTPHX', 'Las Vegas': 'KXLOWTLV',
    'Los Angeles': 'KXLOWTLAX', 'Dallas': 'KXLOWTDAL',
    'Austin': 'KXLOWTAUS', 'Houston': 'KXLOWTHOU',
    'Atlanta': 'KXLOWTATL', 'Miami': 'KXLOWTMIA',
    'New York': 'KXLOWTNYC', 'San Antonio': 'KXLOWTSATX',
    'New Orleans': 'KXLOWTNOLA', 'Philadelphia': 'KXLOWTPHIL',
    'Boston': 'KXLOWTBOS', 'Denver': 'KXLOWTDEN',
    'Oklahoma City': 'KXLOWTOKC', 'Minneapolis': 'KXLOWTMIN',
    'Washington DC': 'KXLOWTDC', 'Chicago': 'KXLOWTCHI',
}


def get_eastern_datetime():
    return datetime.now(pytz.timezone('America/New_York'))


def normalize_label(label):
    if not label:
        return ''
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)', lambda m: m.group(1) + '-' + m.group(2),
                   label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)',
                   lambda m: m.group(1) + '-' + m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above').strip()


def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = normalize_label((m.get(field) or '').replace('\u00b0', '').strip())
        if s:
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below:
                return below.group(1) + ' or below', int(below.group(1)) - 10000
            if above:
                return above.group(1) + ' or above', int(above.group(1)) + 10000
            if rng:
                return rng.group(1) + '-' + rng.group(2), int(rng.group(1))
    cap, floor_s = m.get('cap_strike'), m.get('floor_strike')
    if cap is not None and floor_s is not None:
        try:
            lo, hi = int(float(floor_s)), int(float(cap))
            return f'{lo}-{hi}', lo
        except Exception:
            pass
    return None, None


def get_yes_cents(m):
    for f in ['yes_ask_dollars', 'yes_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                return round(float(v) * 100)
            except Exception:
                pass
    raw = m.get('yes_ask') or m.get('yes_bid')
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    return None


def fetch_ladder(series):
    """Returns [(label, yes_cents), ...] sorted by price desc, or []."""
    url = 'https://api.elections.kalshi.com/trade-api/v2/markets'
    et = get_eastern_datetime()
    event_ticker = series + '-' + et.strftime('%y%b%d').upper()
    today_fmt = et.strftime('%y%b%d').upper()
    tomorrow_utc = (et + timedelta(days=1)).strftime('%Y-%m-%d')

    def _try(params):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    data = _try({'event_ticker': event_ticker, 'limit': 30})
    if not (data and data.get('markets')):
        data = _try({'series_ticker': series, 'status': 'open', 'limit': 30})
    if not (data and data.get('markets')):
        data = _try({'series_ticker': series, 'limit': 30})
    if not (data and data.get('markets')):
        return []

    all_markets = data['markets']
    markets = [m for m in all_markets
               if today_fmt in (m.get('event_ticker') or '').upper()]
    if not markets:
        markets = [m for m in all_markets
                   if (m.get('close_time') or '').startswith(tomorrow_utc)]
    if not markets:
        markets = all_markets

    out, seen = [], set()
    for m in markets:
        label, _ = parse_market_label(m)
        if not label:
            continue
        norm = normalize_label(label)
        if norm in seen:
            continue
        seen.add(norm)
        cents = get_yes_cents(m)
        if cents is not None:
            out.append((label, cents))
    out.sort(key=lambda x: -x[1])
    return out


def main():
    et = get_eastern_datetime()
    print('=== lows ladder pricing vs 31-day dispersion ===')
    print(f'    snapshot {et.strftime("%Y-%m-%d %H:%M")} ET — ONE reading, not a series\n')

    rows = []
    for city, series in LOW_SERIES.items():
        ladder = fetch_ladder(series)
        if not ladder:
            print(f'[{city}]  no ladder')
            rows.append((city, DISPERSION.get(city), None, None, None))
            time.sleep(0.3)
            continue

        top_label, top_price = ladder[0]
        top2 = sum(c for _, c in ladder[:2])
        live = sum(1 for _, c in ladder if c >= 10)

        print(f'[{city}]  top {top_label!r} @ {top_price}c | '
              f'top2 {top2}c | {live} brackets >=10c')
        print(f'    ladder: ' + ', '.join(f'{l}@{c}c' for l, c in ladder[:5]))
        rows.append((city, DISPERSION.get(city), top_price, top2, live))
        time.sleep(0.3)

    print()
    print('=== SUMMARY (sorted by 31-day value stdev, tightest first) ===')
    print(f'{"city":<16}{"stdev":>7}{"top":>7}{"top2":>7}{"live":>6}   read')
    for city, sd, top, top2, live in sorted(
            rows, key=lambda x: (x[1] if x[1] is not None else 99)):
        if top is None:
            print(f'{city:<16}{sd if sd else 0:>7.2f}{"—":>7}{"—":>7}{"—":>6}   no ladder')
            continue
        if sd is not None and sd <= 2.0 and top >= 85:
            read = 'PRICED — predictability already in the market'
        elif sd is not None and sd <= 2.0 and top <= 70:
            read = 'market less certain than history — worth a look'
        elif sd is not None and sd >= 4.0 and top >= 80:
            read = 'market more confident than history justifies'
        elif sd is not None and sd >= 4.0:
            read = 'genuinely uncertain — needs forecast skill'
        else:
            read = 'middle band'
        print(f'{city:<16}{sd if sd else 0:>7.2f}{top:>6}c{top2:>6}c{live:>6}   {read}')

    print()
    print('→ REMEMBER: one snapshot at one time of day. Lows ladders late in the')
    print('  evening are thin and wide because the minimum is still ~8h away.')
    print('  Re-run near 05:00-07:00 local, around the median minimum time, before')
    print('  concluding anything about whether a city is priced.')

    sys.exit(0)


if __name__ == '__main__':
    main()
