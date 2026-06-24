"""
fetch_lows.py — MPH Weather Model V5.30 (LOWS) — FOUNDATION SCAFFOLD
=====================================================================
Parallel build to fetch_weather.py (V5.29.D, highs). This file does NOT touch
the high-temp model. It is a standalone scaffold for the three foundation
pieces of the low-temperature build:

    1. fetch_kalshi_low_brackets(series, city='')   — Kalshi KXLOWT ladder
    2. fetch_nws_low(city)                           — NWS calendar-day MIN
    3. fetch_cli_min_temp(city, target_date_str)     — Iowa CLI MINIMUM parser

All three mirror their highs counterparts exactly (same filtering chain, same
endpoint fallbacks, same caching) with three lows-specific deltas:

    Δ1  Ticker carries the T for ALL cities: KXLOWT{CITY}-{YYMMMDD}.
        (Highs are inconsistent: KXHIGHTPHX has T, KXHIGHLAX does not. Per the
        scoping note, lows use KXLOWT uniformly. The LOW_SERIES dict below is
        CONSTRUCTED from the highs city codes — see "FIRST-RUN VERIFICATION".)

    Δ2  fetch_nws_low takes the MIN over the WHOLE calendar day (midnight→
        midnight LOCAL), not the daytime max. No isDaytime filter. Filters
        hourly periods by the CITY-LOCAL date, not the ET date — the highs file
        uses ET date which is a latent bug for Pacific cities, masked because
        highs land midday. Lows land ~4-7am where the date boundary matters.

    Δ3  CLI parser extracts entry['low'] instead of entry['high']. Same call,
        same JSON — the Iowa cli.py response carries both MAXIMUM and MINIMUM.

────────────────────────────────────────────────────────────────────────────
FIRST-RUN VERIFICATION (3 things this scaffold cannot confirm offline):
  V1  LOW_SERIES tickers. Constructed as KXLOWT + highs city code. Run the
      smoke test (`python fetch_lows.py`) — any city whose bracket count comes
      back 0 / endpoint=None has the wrong ticker; pull the real one off Kalshi
      and correct that one line.
  V2  wethr.net `low` field semantics. fetch_nws_low tries wethr's `low` first
      (mirroring highs). UNKNOWN whether wethr returns the OVERNIGHT low or the
      CALENDAR-DAY min. These differ on frontal-passage days. The smoke test
      prints BOTH the wethr value and the hourly-derived calendar-day min so you
      can compare them against CLI MINIMUM over a few days before trusting either.
  V3  CLI `low` field name. Iowa cli.py is expected to key the daily minimum as
      'low'. Confirm on first settlement (smoke test pulls yesterday's CLI min).
────────────────────────────────────────────────────────────────────────────
NOT in this scaffold (later V5.30 pieces, intentionally out of scope tonight):
  - Supabase persistence / snapshot / settlement passes
  - Observed-min tracker (the calendar-day min has usually ALREADY occurred by
    the time afternoon crons run, so it's mostly an OBSERVED quantity — needs an
    obs source analogous to obs_high, flipped. Separate piece.)
  - Consensus / sigma / bracket-probability math for lows
  - Raw Kalshi API dump to Supabase (console diagnostic kept; DB hook omitted)
"""

import math
import os
import re
import requests
import time
from datetime import datetime, timedelta

import pytz

# ── Credentials (same env vars as the highs model) ───────────────────────────
WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS       = {'User-Agent': 'kalshi-lows-fetcher/5.30.0', 'Accept': 'application/json'}


# ── City geography / stations (mirrored from highs) ──────────────────────────
CITIES = {
    'Phoenix':       {'lat': 33.4342, 'lon': -112.0116},
    'Las Vegas':     {'lat': 36.0840, 'lon': -115.1537},
    'Los Angeles':   {'lat': 33.9416, 'lon': -118.4085},
    'Dallas':        {'lat': 32.8998, 'lon': -97.0403},
    'Austin':        {'lat': 30.1945, 'lon': -97.6699},
    'Houston':       {'lat': 29.9902, 'lon': -95.3368},
    'Atlanta':       {'lat': 33.6407, 'lon': -84.4277},
    'Miami':         {'lat': 25.7959, 'lon': -80.2870},
    'New York':      {'lat': 40.7812, 'lon': -73.9665},
    'San Antonio':   {'lat': 29.5337, 'lon': -98.4698},
    'New Orleans':   {'lat': 29.9934, 'lon': -90.2580},
    'Philadelphia':  {'lat': 39.8744, 'lon': -75.2424},
    'Boston':        {'lat': 42.3656, 'lon': -71.0096},
    'Denver':        {'lat': 39.8561, 'lon': -104.6737},
    'Oklahoma City': {'lat': 35.3931, 'lon': -97.6007},
    'Minneapolis':   {'lat': 44.8848, 'lon': -93.2223},
    'Washington DC': {'lat': 38.8512, 'lon': -77.0402},
    'Chicago':       {'lat': 41.7868, 'lon': -87.7522},
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

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

CLI_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

# Δ1 — CONSTRUCTED low-market series tickers: KXLOWT + (highs city code).
# Highs city codes, for reference: PHX LV LAX DAL AUS HOU ATL MIA NY SATX NOLA
# PHIL BOS DEN OKC MIN DC CHI. See FIRST-RUN VERIFICATION V1 — confirm each
# resolves on the smoke test; correct any individual line that returns 0 brackets.
LOW_SERIES = {
    'Phoenix': 'KXLOWTPHX', 'Las Vegas': 'KXLOWTLV',
    'Los Angeles': 'KXLOWTLAX', 'Dallas': 'KXLOWTDAL',
    'Austin': 'KXLOWTAUS', 'Houston': 'KXLOWTHOU',
    'Atlanta': 'KXLOWTATL', 'Miami': 'KXLOWTMIA',
    'New York': 'KXLOWTNY', 'San Antonio': 'KXLOWTSATX',
    'New Orleans': 'KXLOWTNOLA', 'Philadelphia': 'KXLOWTPHIL',
    'Boston': 'KXLOWTBOS', 'Denver': 'KXLOWTDEN',
    'Oklahoma City': 'KXLOWTOKC', 'Minneapolis': 'KXLOWTMIN',
    'Washington DC': 'KXLOWTDC', 'Chicago': 'KXLOWTCHI',
}


# ── Date / time helpers (mirrored) ───────────────────────────────────────────
def get_eastern_date():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')

def get_eastern_datetime():
    return datetime.now(pytz.timezone('America/New_York'))

def get_city_local_date(city):
    """Δ2 — local calendar date for the city. NWS clim day (and CLI MINIMUM)
    is local midnight→midnight, so the calendar-day min must be filtered by the
    city's LOCAL date, not the ET date the highs file uses."""
    return datetime.now(pytz.timezone(CITY_TZ.get(city, 'America/New_York'))).strftime('%Y-%m-%d')


# ── Label parsing (mirrored verbatim — lows brackets are temps too) ──────────
def normalize_label(label):
    if not label:
        return ''
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)', lambda m: m.group(1)+'-'+m.group(2), label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)', lambda m: m.group(1)+'-'+m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above').strip()


def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = normalize_label((m.get(field) or '').replace('\u00b0', '').replace('deg', '').strip())
        if s:
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below: return below.group(1)+' or below', int(below.group(1))-10000
            if above: return above.group(1)+' or above', int(above.group(1))+10000
            if rng:   return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
    title = (m.get('title') or '').replace('\u00b0', '').replace('**', '').replace('deg', '')
    if title:
        ma = re.search(r'be\s*[>=]+\s*(\d+)', title, re.I)
        if ma: n = int(ma.group(1)); return str(n)+' or above', n+10000
        mb = re.search(r'be\s*[<=]+\s*(\d+)', title, re.I)
        if mb: n = int(mb.group(1)); return str(n)+' or below', n-10000
        mr = re.search(r'be\s*(\d+)\s*(?:to|-)\s*(\d+)', title, re.I)
        if mr:
            lo, hi = int(mr.group(1)), int(mr.group(2))
            return str(lo)+'-'+str(hi), lo
        nums = re.findall(r'\d+', title)
        if len(nums) >= 2:
            lo, hi = int(nums[-2]), int(nums[-1])
            if 0 < hi-lo <= 5:
                return str(lo)+'-'+str(hi), lo
    cap, floor_s = m.get('cap_strike'), m.get('floor_strike')
    if cap is not None and floor_s is not None:
        try:
            lo, hi = int(float(floor_s)), int(float(cap))
            return str(lo)+'-'+str(hi), lo
        except Exception:
            pass
    if cap is not None:
        try:
            n = int(float(cap))
            return str(n)+' or below', n-10000
        except Exception:
            pass
    return None, None


def get_price_cents(m):
    yes_ask = no_ask = None
    for f in ['yes_ask_dollars', 'yes_bid_dollars']:
        v = m.get(f)
        if v:
            try: yes_ask = round(float(v)*100); break
            except Exception: pass
    for f in ['no_ask_dollars', 'no_bid_dollars']:
        v = m.get(f)
        if v:
            try: no_ask = round(float(v)*100); break
            except Exception: pass
    if yes_ask is None:
        raw = m.get('yes_ask') or m.get('yes_bid')
        if raw is not None:
            try: yes_ask = int(raw)
            except Exception: pass
    if no_ask is None:
        raw = m.get('no_ask') or m.get('no_bid')
        if raw is not None:
            try: no_ask = int(raw)
            except Exception: pass
    return yes_ask, no_ask


# ── 1) Kalshi low-bracket fetch (mirrors fetch_kalshi_brackets) ──────────────
def get_low_event_ticker(series):
    """KXLOWT{CITY}-{YYMMMDD}. Same date format as the highs V5.28.4 fix:
    strftime('%y%b%d').upper() → e.g. '26JUN23'. ET datetime, mirroring highs
    (Kalshi labels a city's daily markets by a single calendar date)."""
    return series + '-' + get_eastern_datetime().strftime('%y%b%d').upper()


def fetch_kalshi_low_brackets(series, city=''):
    """Fetch + parse the Kalshi LOW-temperature ladder for a city.

    Mirrors fetch_kalshi_brackets (highs) exactly — same three-endpoint
    fallback, same V5.28.4 four-layer filtering chain, same dedup. Differences:
      - uses get_low_event_ticker (KXLOWT series)
      - omits the Supabase kalshi_api_dumps write (console diagnostic kept);
        add a low-side dump table later if you want it.

    Returns list of (label, yes_ask, no_ask) sorted by bracket, or None.
    """
    url = 'https://api.elections.kalshi.com/trade-api/v2/markets'
    event_ticker = get_low_event_ticker(series)

    today_et_dt = get_eastern_datetime()
    today_kalshi_fmt = today_et_dt.strftime('%y%b%d').upper()             # '26JUN23'
    tomorrow_utc_date = (today_et_dt + timedelta(days=1)).strftime('%Y-%m-%d')

    def _try(params):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    endpoint_used = None
    data = _try({'event_ticker': event_ticker, 'limit': 30})
    if data and data.get('markets'):
        endpoint_used = 'event_ticker'
    else:
        data = _try({'series_ticker': series, 'status': 'open', 'limit': 30})
        if data and data.get('markets'):
            endpoint_used = 'series_status_open'
        else:
            data = _try({'series_ticker': series, 'limit': 30})
            if data and data.get('markets'):
                endpoint_used = 'series_only'
    if not data or not data.get('markets'):
        print(f'    📋 [{series}] endpoint=None (no markets returned)')
        return None

    all_markets = data['markets']
    raw_market_count = len(all_markets)

    # V5.28.4 four-layer filter chain (verbatim logic from highs):
    markets = [m for m in all_markets if today_kalshi_fmt in (m.get('event_ticker') or '').upper()]
    if not markets:
        markets = [m for m in all_markets if (m.get('close_time') or '').startswith(tomorrow_utc_date)]
    if not markets:
        markets = [m for m in all_markets if today_kalshi_fmt in (m.get('ticker') or '').upper()]
    if not markets:
        print(f'    ⚠️ [{series}] No today-match ({today_kalshi_fmt}/{tomorrow_utc_date}) — '
              f'falling back to all {len(all_markets)} markets')
        markets = all_markets

    filtered_count = len(markets)

    parsed = []
    seen_labels = set()
    duplicates_dropped = 0
    for m in markets:
        label, key = parse_market_label(m)
        if label is None:
            continue
        norm_label = normalize_label(label)
        if norm_label in seen_labels:
            duplicates_dropped += 1
            continue
        seen_labels.add(norm_label)
        yes_ask, no_ask = get_price_cents(m)
        parsed.append((key, label, yes_ask, no_ask))

    if duplicates_dropped > 0:
        print(f'    ⚠️ [{series}] Dropped {duplicates_dropped} duplicate bracket(s) after dedup')

    print(f'    📋 [{series}] endpoint={endpoint_used}, raw={raw_market_count}, '
          f'filtered={filtered_count}, final={len(parsed)}')

    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[0])
    return [(label, yes_ask, no_ask) for _, label, yes_ask, no_ask in parsed]


# ── NWS grid cache + 2) calendar-day MIN fetch ───────────────────────────────
_NWS_GRID_CACHE = {}

def fetch_nws_grid(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _NWS_GRID_CACHE:
        return _NWS_GRID_CACHE[key]
    try:
        r = requests.get(f'https://api.weather.gov/points/{lat},{lon}',
                         headers=HEADERS, timeout=12)
        r.raise_for_status()
        props = r.json().get('properties', {})
        office = props.get('gridId')
        gx = props.get('gridX')
        gy = props.get('gridY')
        if not all([office, gx is not None, gy is not None]):
            return None
        result = (office, gx, gy)
        _NWS_GRID_CACHE[key] = result
        return result
    except Exception:
        return None


def fetch_nws_low_hourly_min(city):
    """The semantically-correct path: lowest NWS hourly temp across the city's
    LOCAL calendar day (Δ2). Filters by local date; takes min over ALL hours.
    Returns float or None. Used as the hourly fallback AND printed standalone in
    the smoke test for the wethr-vs-hourly comparison (V2)."""
    coords = CITIES[city]
    grid = fetch_nws_grid(coords['lat'], coords['lon'])
    if not grid:
        return None
    office, gx, gy = grid
    today_local = get_city_local_date(city)
    hourly_url = f'https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast/hourly'
    try:
        r = requests.get(hourly_url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        periods = r.json().get('properties', {}).get('periods', [])
        today_temps = []
        for period in periods:
            start = period.get('startTime', '')   # local ISO, e.g. 2026-06-23T04:00:00-05:00
            temp = period.get('temperature')
            unit = period.get('temperatureUnit', 'F')
            if not start.startswith(today_local):  # Δ2: LOCAL date, no isDaytime filter
                continue
            if temp is not None:
                temp_f = float(temp) if unit == 'F' else float(temp) * 9 / 5 + 32
                today_temps.append(temp_f)
        if today_temps:
            return round(min(today_temps), 1)
    except Exception:
        pass
    return None


def fetch_nws_low(city):
    """Forecast calendar-day MINIMUM for a city.

    Mirrors fetch_nws_forecast (highs): wethr.net primary, NWS hourly fallback.
    Deltas: pulls wethr's `low` (not `high`), and the hourly fallback is the
    LOCAL calendar-day MIN over all hours (fetch_nws_low_hourly_min).

    ⚠️ V2 — wethr `low` semantics unverified (overnight low vs calendar-day min).
    The smoke test prints both wethr `low` and the hourly min so you can pick the
    right source after comparing to CLI MINIMUM. If wethr `low` turns out to be
    the overnight low, flip this to call fetch_nws_low_hourly_min first.

    ⚠️ By afternoon the true calendar-day min has usually ALREADY occurred
    (~4-7am). This returns the FORECAST-side min only; an observed-min tracker
    (flipped obs_high) is a separate later piece.

    Returns float (rounded 1dp) or None.
    """
    station = WETHR_STATIONS.get(city)
    today_local = get_city_local_date(city)   # Δ2: local calendar date
    if station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/nws_forecasts.php',
                params={'station_code': station, 'date': today_local, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json()
                low = data.get('low')
                if low is not None:
                    return round(float(low), 1)
        except Exception:
            pass
    return fetch_nws_low_hourly_min(city)


# ── 3) Iowa CLI MINIMUM parser (mirrors fetch_cli_max_temp) ──────────────────
_CLI_MIN_CACHE = {}

def fetch_cli_min_temp(city, target_date_str):
    """Iowa State CLI daily MINIMUM for settlement. Mirrors fetch_cli_max_temp
    exactly — same cli.py call, same per-station-year cache — but reads
    entry['low'] instead of entry['high'] (Δ3). The cli.py response carries both
    MAXIMUM and MINIMUM from the same NWS Climatological Report.

    ⚠️ V3 — confirm the field is keyed 'low' on first settlement run.

    Returns float or None.
    """
    station = CLI_STATIONS.get(city)
    if not station:
        return None
    year = target_date_str[:4]
    cache_key = station + '_' + year
    if cache_key not in _CLI_MIN_CACHE:
        try:
            url = ('https://mesonet.agron.iastate.edu/json/cli.py'
                   '?station=' + station + '&year=' + year)
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            lookup = {}
            for entry in data.get('results', []):
                valid = entry.get('valid', '')
                low = entry.get('low')          # Δ3
                if valid and low is not None:
                    try:
                        lookup[valid] = float(low)
                    except Exception:
                        pass
            _CLI_MIN_CACHE[cache_key] = lookup
        except Exception:
            return None
    return _CLI_MIN_CACHE.get(cache_key, {}).get(target_date_str)


# ── Smoke test — read-only, no Supabase, safe to run anytime ──────────────────
def _smoke_test():
    print('=== fetch_lows.py V5.30 foundation smoke test ===')
    print(f'ET date: {get_eastern_date()} | low ticker date fmt: '
          f'{get_eastern_datetime().strftime("%y%b%d").upper()}\n')

    yest = (get_eastern_datetime() - timedelta(days=1)).strftime('%Y-%m-%d')

    kalshi_ok, nws_ok, cli_ok = [], [], []
    for city in CITIES:
        series = LOW_SERIES.get(city, '')
        print(f'[{city}]  series={series}')

        # 1) Kalshi ladder (V1 ticker verification)
        brackets = fetch_kalshi_low_brackets(series, city=city)
        if brackets:
            kalshi_ok.append(city)
            print(f'    ✅ Kalshi: {len(brackets)} brackets, '
                  f'e.g. {brackets[0][0]!r} @ yes {brackets[0][1]}c')
        else:
            print(f'    ❌ Kalshi: no ladder (check LOW_SERIES ticker — V1)')

        # 2) NWS — print BOTH sources for the V2 comparison
        wethr_low = None
        st = WETHR_STATIONS.get(city)
        if st:
            try:
                r = requests.get('https://wethr.net/api/v2/nws_forecasts.php',
                                 params={'station_code': st,
                                         'date': get_city_local_date(city),
                                         'mode': 'latest'},
                                 headers=WETHR_HEADERS, timeout=12)
                if r.status_code == 200:
                    v = r.json().get('low')
                    wethr_low = round(float(v), 1) if v is not None else None
            except Exception:
                pass
        hourly_min = fetch_nws_low_hourly_min(city)
        if wethr_low is not None or hourly_min is not None:
            nws_ok.append(city)
        flag = ''
        if wethr_low is not None and hourly_min is not None and abs(wethr_low - hourly_min) >= 2.0:
            flag = '  ⚠️ DIVERGES ≥2F (V2)'
        print(f'    NWS low → wethr={wethr_low}F | hourly-cal-day-min={hourly_min}F{flag}')

        # 3) CLI MINIMUM for yesterday (V3 field check)
        cli_min = fetch_cli_min_temp(city, yest)
        if cli_min is not None:
            cli_ok.append(city)
            print(f'    ✅ CLI min {yest}: {cli_min}F')
        else:
            print(f'    ⚠️ CLI min {yest}: none (data lag or field≠"low" — V3)')

        print()
        time.sleep(0.3)

    n = len(CITIES)
    print('=== Summary ===')
    print(f'  Kalshi ladders resolved : {len(kalshi_ok)}/{n}')
    print(f'  NWS low available       : {len(nws_ok)}/{n}')
    print(f'  CLI min ({yest}) : {len(cli_ok)}/{n}')
    missing_kalshi = [c for c in CITIES if c not in kalshi_ok]
    if missing_kalshi:
        print(f'  ⚠️ Kalshi tickers to fix (V1): {", ".join(missing_kalshi)}')


if __name__ == '__main__':
    _smoke_test()
