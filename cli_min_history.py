"""
cli_min_history.py — V5.30 lows design data (ANALYSIS ONLY, freeze-safe)
========================================================================
Pulls ~30 days of NWS Climatological Report MINIMUM (the Kalshi KXLOWT
settlement value) for all 18 cities from the Iowa State CLI JSON API, and
reports per-city behavior:

    n            settled days in window
    min / max    coldest / warmest daily low in window
    range        max - min  (how wide the low swings)
    mean         average daily low
    stdev        raw spread of daily lows (includes seasonal drift)
    dod          mean |day-over-day change| (cleaner volatility signal —
                 strips most of the seasonal trend)

WHY: before designing low brackets or per-city sigma, we need the relative
volatility tiering. The highs model learned the hard way that Phoenix/Vegas
run tight and Boston/Denver/OKC run wide — sigma has to be per-city. This
gives the lows the same map up front instead of discovering it in losses.

NOTE: stdev/dod here are RAW low variability, NOT forecast-error sigma (we
have no low forecast source yet). They're a starting prior for relative city
tiering and bracket width, to be replaced by observed forecast-error sigma
once the lows have a forecast source and settled bets. Re-run periodically.

No Supabase, no Kalshi, no model logic. Reads Iowa CLI and prints. Safe anytime.
"""

import statistics
from datetime import datetime, timedelta

import requests

HEADERS = {'User-Agent': 'cli-min-history/1.0', 'Accept': 'application/json'}

WINDOW_DAYS = 30

CLI_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

_CLI_CACHE = {}


def fetch_cli_year(station, year):
    """Return {valid_date_str: low_float} for a station/year. Cached."""
    cache_key = f'{station}_{year}'
    if cache_key in _CLI_CACHE:
        return _CLI_CACHE[cache_key]
    lookup = {}
    try:
        url = ('https://mesonet.agron.iastate.edu/json/cli.py'
               f'?station={station}&year={year}')
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        for entry in r.json().get('results', []):
            valid = entry.get('valid', '')
            low = entry.get('low')
            if valid and low is not None:
                try:
                    lookup[valid] = float(low)
                except Exception:
                    pass
    except Exception as e:
        print(f'    ⚠️ CLI fetch failed {station} {year}: {type(e).__name__}')
    _CLI_CACHE[cache_key] = lookup
    return lookup


def window_dates(days):
    """List of 'YYYY-MM-DD' for the last `days` days ending yesterday."""
    end = datetime.utcnow().date() - timedelta(days=1)
    return [(end - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]


def main():
    dates = window_dates(WINDOW_DAYS)
    years = sorted({d[:4] for d in dates})
    print(f'=== CLI MINIMUM history — last {WINDOW_DAYS} days '
          f'({dates[-1]} → {dates[0]}) ===\n')

    rows = []
    for city, station in CLI_STATIONS.items():
        merged = {}
        for y in years:
            merged.update(fetch_cli_year(station, y))
        series = [(d, merged[d]) for d in sorted(dates) if d in merged]
        lows = [v for _, v in series]
        if len(lows) < 5:
            print(f'[{city:14s}] only {len(lows)} days — skipping stats')
            continue

        lo, hi = min(lows), max(lows)
        mean = statistics.mean(lows)
        stdev = statistics.pstdev(lows)
        # mean |day-over-day change| over consecutive available days
        dod_changes = [abs(lows[i] - lows[i - 1]) for i in range(1, len(lows))]
        dod = statistics.mean(dod_changes) if dod_changes else 0.0
        last7 = [v for _, v in series[-7:]]

        rows.append({
            'city': city, 'n': len(lows), 'min': lo, 'max': hi,
            'range': hi - lo, 'mean': mean, 'stdev': stdev, 'dod': dod,
            'last7': last7,
        })

    # Sort by day-over-day volatility (cleanest tiering signal)
    rows.sort(key=lambda r: r['dod'], reverse=True)

    hdr = (f"{'city':14s} {'n':>3s} {'min':>5s} {'max':>5s} {'rng':>5s} "
           f"{'mean':>6s} {'stdev':>6s} {'dod':>5s}  recent 7 lows")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        l7 = ' '.join(f'{v:.0f}' for v in r['last7'])
        print(f"{r['city']:14s} {r['n']:>3d} {r['min']:>5.0f} {r['max']:>5.0f} "
              f"{r['range']:>5.0f} {r['mean']:>6.1f} {r['stdev']:>6.2f} "
              f"{r['dod']:>5.2f}  {l7}")

    if rows:
        print('\n=== Tiering (by day-over-day volatility) ===')
        tight = [r['city'] for r in rows if r['dod'] < 2.0]
        mid   = [r['city'] for r in rows if 2.0 <= r['dod'] < 3.5]
        wide  = [r['city'] for r in rows if r['dod'] >= 3.5]
        print(f'  TIGHT  (dod < 2.0F) : {", ".join(tight) or "none"}')
        print(f'  MID    (2.0-3.5F)   : {", ".join(mid) or "none"}')
        print(f'  WIDE   (dod >= 3.5F): {", ".join(wide) or "none"}')
        print('\n  → TIGHT cities are the low-side analogue of Phoenix/Vegas on '
              'highs: candidates for the first low allowlist.')


if __name__ == '__main__':
    main()
