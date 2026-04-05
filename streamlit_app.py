# Kalshi High Temperature Model - V5.1
#
# Changes from V5.0:
# 1. Discarded obs high now shows as WARNING instead of silently dropping
#    "⚠️ Obs high of XX.XF discarded — verify manually before betting"
# 2. NBM fix — replaced broken NWS gridpoint API fetch (field names don't exist)
#    with Iowa State AFOS archive fetch of actual NBM text bulletin (NBP product)
#    Correct NBM field names (v4.x): TXNP1(p10), TXNP2(p25), TXNP5(p50),
#    TXNP7(p75), TXNP9(p90), TXNMN(mean) — previous version had wrong names
#    Also adds visible NBM/Sigma status indicator to All Cities panel
# 3. Cold front detection warning — when temp trending down significantly from obs high,
#    flags city as "⚠️ Peak may already be in — verify before betting"
# 4. Bracket boundary warning — flags any consensus within 0.5F of a bracket
#    ceiling or floor as "⚠️ Boundary risk — treat as next bracket up"

import math, re, json, time, requests
import streamlit as st
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title='Kalshi High Temp V5.1', layout='wide')
st.title('Kalshi High Temperature Model - V5.1')

SAVE_FILE = Path('saved_ladders.json')
LAST_SYNC_FILE = Path('last_sync.json')
PRICE_CACHE_FILE = Path('price_cache.json')
PRICE_CACHE_MINUTES = 10

MIN_EDGE = 8
HEADERS = {'User-Agent': 'kalshi-temp-model/5.1', 'Accept': 'application/geo+json, application/json, text/html'}

CITY_TZ = {
    'Phoenix': 'America/Phoenix', 'Las Vegas': 'America/Los_Angeles',
    'Los Angeles': 'America/Los_Angeles', 'Dallas': 'America/Chicago',
    'Austin': 'America/Chicago', 'Houston': 'America/Chicago',
    'Atlanta': 'America/New_York', 'Miami': 'America/New_York',
    'New York': 'America/New_York', 'San Antonio': 'America/Chicago',
    'New Orleans': 'America/Chicago', 'Philadelphia': 'America/New_York',
    'Boston': 'America/New_York', 'Denver': 'America/Denver',
    'Oklahoma City': 'America/Chicago', 'Minneapolis': 'America/Chicago',
    'Washington DC': 'America/New_York',
}

SERIES = {
    'Phoenix': 'KXHIGHTPHX', 'Las Vegas': 'KXHIGHTLV',
    'Los Angeles': 'KXHIGHLAX', 'Dallas': 'KXHIGHTDAL',
    'Austin': 'KXHIGHAUS', 'Houston': 'KXHIGHTHOU',
    'Atlanta': 'KXHIGHTATL', 'Miami': 'KXHIGHMIA',
    'New York': 'KXHIGHNY', 'San Antonio': 'KXHIGHTSATX',
    'New Orleans': 'KXHIGHTNOLA', 'Philadelphia': 'KXHIGHPHIL',
    'Boston': 'KXHIGHTBOS', 'Denver': 'KXHIGHDEN',
    'Oklahoma City': 'KXHIGHTOKC', 'Minneapolis': 'KXHIGHTMIN',
    'Washington DC': 'KXHIGHTDC',
}

STATIONS = {
    'Phoenix': 'CLIPHX', 'Las Vegas': 'CLILAS', 'Los Angeles': 'CLILAX',
    'Dallas': 'CLIDFW', 'Austin': 'CLIAUS', 'Houston': 'CLIHOU',
    'Atlanta': 'CLIATL', 'Miami': 'CLIMIA', 'New York': 'KNYC',
    'San Antonio': 'CLISAT', 'New Orleans': 'CLIMSY', 'Philadelphia': 'CLIPHL',
    'Boston': 'CLIBOS', 'Denver': 'CLIDEN', 'Oklahoma City': 'CLIOKC',
    'Minneapolis': 'CLIMSP', 'Washington DC': 'CLIDCA',
}

OBHISTORY_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA',
}

WUNDERGROUND_URLS = {
    'Phoenix':       'https://www.wunderground.com/weather/KPHX',
    'Las Vegas':     'https://www.wunderground.com/weather/KLAS',
    'Los Angeles':   'https://www.wunderground.com/weather/KLAX',
    'Dallas':        'https://www.wunderground.com/weather/KDFW',
    'Austin':        'https://www.wunderground.com/weather/KAUS',
    'Houston':       'https://www.wunderground.com/weather/KHOU',
    'Atlanta':       'https://www.wunderground.com/weather/KATL',
    'Miami':         'https://www.wunderground.com/weather/KMIA',
    'New York':      'https://www.wunderground.com/weather/KNYC',
    'San Antonio':   'https://www.wunderground.com/weather/KSAT',
    'New Orleans':   'https://www.wunderground.com/weather/KMSY',
    'Philadelphia':  'https://www.wunderground.com/weather/KPHL',
    'Boston':        'https://www.wunderground.com/weather/KBOS',
    'Denver':        'https://www.wunderground.com/weather/KDEN',
    'Oklahoma City': 'https://www.wunderground.com/weather/KOKC',
    'Minneapolis':   'https://www.wunderground.com/weather/KMSP',
    'Washington DC': 'https://www.wunderground.com/weather/KDCA',
}

SETTLEMENT_LOCATION = {
    'Phoenix': 'Phoenix Sky Harbor Airport', 'Las Vegas': 'Las Vegas Harry Reid Airport',
    'Los Angeles': 'LA International Airport', 'Dallas': 'Dallas/Fort Worth Airport',
    'Austin': 'Austin-Bergstrom Airport', 'Houston': 'Houston Hobby Airport',
    'Atlanta': 'Atlanta Hartsfield Airport', 'Miami': 'Miami International Airport',
    'New York': 'Central Park, Manhattan', 'San Antonio': 'San Antonio International Airport',
    'New Orleans': 'New Orleans Armstrong Airport', 'Philadelphia': 'Philadelphia International Airport',
    'Boston': 'Boston Logan Airport', 'Denver': 'Denver International Airport',
    'Oklahoma City': 'Oklahoma City Will Rogers Airport', 'Minneapolis': 'Minneapolis-St. Paul Airport',
    'Washington DC': 'Reagan National Airport',
}

CITIES = {
    'Phoenix': {'lat': 33.4342, 'lon': -112.0116}, 'Las Vegas': {'lat': 36.0840, 'lon': -115.1537},
    'Los Angeles': {'lat': 33.9416, 'lon': -118.4085}, 'Dallas': {'lat': 32.8998, 'lon': -97.0403},
    'Austin': {'lat': 30.1945, 'lon': -97.6699}, 'Houston': {'lat': 29.9902, 'lon': -95.3368},
    'Atlanta': {'lat': 33.6407, 'lon': -84.4277}, 'Miami': {'lat': 25.7959, 'lon': -80.2870},
    'New York': {'lat': 40.7812, 'lon': -73.9665}, 'San Antonio': {'lat': 29.5337, 'lon': -98.4698},
    'New Orleans': {'lat': 29.9934, 'lon': -90.2580}, 'Philadelphia': {'lat': 39.8744, 'lon': -75.2424},
    'Boston': {'lat': 42.3656, 'lon': -71.0096}, 'Denver': {'lat': 39.8561, 'lon': -104.6737},
    'Oklahoma City': {'lat': 35.3931, 'lon': -97.6007}, 'Minneapolis': {'lat': 44.8848, 'lon': -93.2223},
    'Washington DC': {'lat': 38.8512, 'lon': -77.0402},
}

DEFAULT_LADDERS = {
    'Phoenix': '74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above',
    'Las Vegas': '74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above',
    'Los Angeles': '66 or below | 67-68 | 69-70 | 71-72 | 73-74 | 75 or above',
    'Dallas': '78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above',
    'Austin': '78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above',
    'Houston': '79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above',
    'Atlanta': '74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above',
    'Miami': '76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above',
    'New York': '46 or below | 47-48 | 49-50 | 51-52 | 53-54 | 55 or above',
    'San Antonio': '78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above',
    'New Orleans': '80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above',
    'Philadelphia': '73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above',
    'Boston': '48 or below | 49-50 | 51-52 | 53-54 | 55-56 | 57 or above',
    'Denver': '65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above',
    'Oklahoma City': '75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above',
    'Minneapolis': '65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above',
    'Washington DC': '76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above',
}

BASE_SIGMA = {
    'New York': 1.8, 'Philadelphia': 1.8, 'Washington DC': 1.9, 'Boston': 1.9,
    'Los Angeles': 1.7, 'Denver': 1.9, 'Miami': 2.0, 'Minneapolis': 2.1,
    'New Orleans': 2.1, 'Phoenix': 2.2, 'Las Vegas': 2.2, 'Atlanta': 2.3,
    'Dallas': 2.3, 'Austin': 2.3, 'Houston': 2.3, 'San Antonio': 2.3, 'Oklahoma City': 2.5,
}

DESERT_CITIES = {'Phoenix', 'Las Vegas'}
FORECAST_HEAVY_CITIES = {'Dallas', 'Austin', 'Houston', 'San Antonio', 'Oklahoma City'}

GFS_CITY_WEIGHT = {
    'Phoenix': 0.10, 'Las Vegas': 0.10,
    'Los Angeles': 0.12,
    'Miami': 0.18, 'Houston': 0.18, 'New Orleans': 0.18,
    'Dallas': 0.25, 'Austin': 0.25, 'San Antonio': 0.25, 'Oklahoma City': 0.25,
    'Atlanta': 0.22, 'Washington DC': 0.22,
    'New York': 0.25, 'Philadelphia': 0.25, 'Boston': 0.25,
    'Denver': 0.22, 'Minneapolis': 0.22,
}

# ── Supabase Client ───────────────────────────────────────────────────────────
_SB_URL = 'https://oirnfhhuyjuotkrlymxd.supabase.co'
_SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pcm5maGh1eWp1b3Rrcmx5bXhkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIzMDYyMjAsImV4cCI6MjA1NzgyNjIyMH0.3Mp81UjdxkpAYq_cuaOa-0Vqo1LkMgxawOM1gWF6TJ0'

def get_sb_headers():
    try:
        key = st.secrets['supabase']['key']
    except Exception:
        key = _SB_KEY
    return {
        'apikey': key,
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }

def sb_url(table):
    try:
        url = st.secrets['supabase']['url']
    except Exception:
        url = _SB_URL
    return url + '/rest/v1/' + table

def sb_insert(row):
    try:
        r = requests.post(sb_url('settlements'), headers=get_sb_headers(), json=row, timeout=10)
        return r.status_code in (200, 201)
    except Exception:
        return False

def sb_fetch_all():
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'order': 'date.asc', 'limit': '1000'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def sb_fetch_city(city):
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'city': 'eq.' + city, 'order': 'date.asc', 'limit': '200'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def sb_fetch_unsettled():
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'actual': 'is.null', 'order': 'date.asc'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def sb_update_actual(row_id, actual, error):
    try:
        r = requests.patch(
            sb_url('settlements') + '?id=eq.' + str(row_id),
            headers=get_sb_headers(),
            json={'actual': actual, 'error': round(error, 2)},
            timeout=10
        )
        return r.status_code in (200, 204)
    except Exception:
        return False

def sb_fetch_today(city):
    today = get_eastern_date()
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'date': 'eq.' + today, 'city': 'eq.' + city}, timeout=10)
        rows = r.json() if r.status_code == 200 else []
        return rows[0] if rows else None
    except Exception:
        return None

def sb_upsert_prediction(city, consensus, forecast, ensemble_mean, source_gap, high_uncertainty, obs_high, bias_correction):
    today = get_eastern_date()
    existing = sb_fetch_today(city)
    row = {
        'date': today,
        'city': city,
        'consensus': round(consensus, 2),
        'forecast': round(forecast, 2) if forecast else None,
        'ensemble_mean': round(ensemble_mean, 2) if ensemble_mean else None,
        'source_gap': round(source_gap, 2) if source_gap else None,
        'high_uncertainty': bool(high_uncertainty),
        'obs_high': round(obs_high, 2) if obs_high else None,
        'bias_correction': round(bias_correction, 2),
        'actual': None,
        'error': None,
    }
    if existing:
        try:
            r = requests.patch(
                sb_url('settlements') + '?id=eq.' + str(existing['id']),
                headers=get_sb_headers(),
                json={k: v for k, v in row.items() if k not in ('date', 'city')},
                timeout=10
            )
            return r.status_code in (200, 204)
        except Exception:
            return False
    else:
        return sb_insert(row)

# ── Auto-Settlement ───────────────────────────────────────────────────────────
def fetch_obs_high_for_date(icao, target_date_str):
    url = 'https://forecast.weather.gov/data/obhistory/' + icao + '.html'
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    table = max(tables, key=lambda t: len(t.find_all('tr')), default=None) if tables else None
    if not table:
        return None
    target_day = str(datetime.strptime(target_date_str, '%Y-%m-%d').day)
    highs = []
    for row in table.find_all('tr'):
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cols or len(cols) < 9 or cols[0] != target_day:
            continue
        try:
            t = float(cols[8])
            if 0 < t < 130:
                highs.append(t)
        except Exception:
            pass
    return round(max(highs), 1) if highs else None

@st.cache_data(ttl=3600)
def run_auto_settlement():
    unsettled = sb_fetch_unsettled()
    if not unsettled:
        return 0, []
    settled = []
    for row in unsettled:
        row_date = row.get('date', '')
        if row_date >= get_eastern_date():
            continue
        city = row.get('city')
        icao = OBHISTORY_STATIONS.get(city)
        if not icao:
            continue
        actual = fetch_obs_high_for_date(icao, row_date)
        if actual is None:
            continue
        consensus = row.get('consensus')
        error = round(actual - consensus, 2) if consensus is not None else None
        ok = sb_update_actual(row['id'], actual, error)
        if ok:
            settled.append({'city': city, 'date': row_date, 'actual': actual, 'error': error})
    return len(settled), settled

# ── Bias Correction ───────────────────────────────────────────────────────────
def compute_bias_correction_db(city, n_recent=10):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        return 0.0, len(complete)
    recent = complete[-n_recent:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    mean_error = sum(errors) / len(errors)
    correction = max(-5.0, min(5.0, mean_error))
    return round(correction, 2), len(recent)

# ── V5.1 NBM via Iowa State AFOS — CORRECTED field names ─────────────────────
@st.cache_data(ttl=1800)
def fetch_nbm_percentiles(lat, lon):
    """
    V5.1: Fetch NBM percentile forecasts from Iowa State AFOS archive (NBP bulletin).

    Correct NBM v4.x field names:
      TXNMN = mean max/min temp (F)
      TXNP1 = 10th percentile
      TXNP2 = 25th percentile
      TXNP5 = 50th percentile  ← was wrongly TXNP3 in first attempt
      TXNP7 = 75th percentile  ← was wrongly TXNP3
      TXNP9 = 90th percentile  ← was wrongly TXNP4

    Max temp (daytime high) is listed at 00z column.
    NBP bulletin PIL format: NBP + 3-letter station (e.g. NBPDFW for KDFW)

    Returns dict with keys: p10, p25, p50, p75, p90 or None if unavailable.
    """
    # Find closest city to get ICAO station
    city_name = None
    best_dist = float('inf')
    for c, coords in CITIES.items():
        dist = abs(coords['lat'] - lat) + abs(coords['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            city_name = c

    icao = OBHISTORY_STATIONS.get(city_name, '')
    if not icao:
        return None

    # NBP PIL: NBP + 3-letter station (drop K prefix)
    station_3 = icao[1:] if icao.startswith('K') else icao[:3]
    pil = 'NBP' + station_3

    try:
        url = f'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil={pil}&fmt=text&limit=1'
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        text = r.text
    except Exception:
        return None

    if not text or len(text) < 100:
        return None

    lines = text.split('\n')

    def parse_nbm_line(lines, key):
        """
        Find line starting with key and return all valid temperature values.
        NBP format: "TXNP1 51| 62 47| 53 44| 62..."
        Values before pipe = 12z (overnight low), after pipe = 00z (daytime max).
        We extract all numbers and take the max of the first few as today's high.
        """
        for line in lines:
            stripped = line.strip()
            if re.match(r'^' + re.escape(key) + r'\s', stripped, re.IGNORECASE):
                remainder = stripped[len(key):]
                nums = re.findall(r'\d+', remainder)
                vals = []
                for n in nums:
                    try:
                        v = float(n)
                        if 30 < v < 130:  # valid Fahrenheit temp range
                            vals.append(v)
                    except Exception:
                        pass
                return vals
        return []

    def get_today_high(vals):
        """Take max of first 4 values — covers today's min/max pair and tomorrow's min."""
        if not vals:
            return None
        return round(max(vals[:4]), 1)

    p10_vals  = parse_nbm_line(lines, 'TXNP1')
    p25_vals  = parse_nbm_line(lines, 'TXNP2')
    p50_vals  = parse_nbm_line(lines, 'TXNP5')
    p75_vals  = parse_nbm_line(lines, 'TXNP7')
    p90_vals  = parse_nbm_line(lines, 'TXNP9')
    mean_vals = parse_nbm_line(lines, 'TXNMN')

    result = {}
    if p10_vals:
        result['p10'] = get_today_high(p10_vals)
    if p25_vals:
        result['p25'] = get_today_high(p25_vals)
    if p50_vals:
        result['p50'] = get_today_high(p50_vals)
    elif mean_vals:
        result['p50'] = get_today_high(mean_vals)
    if p75_vals:
        result['p75'] = get_today_high(p75_vals)
    if p90_vals:
        result['p90'] = get_today_high(p90_vals)

    # Remove any None values
    result = {k: v for k, v in result.items() if v is not None}

    if len(result) >= 2 and ('p50' in result or 'p25' in result or 'p75' in result):
        return result
    return None

def nbm_bracket_prob(nbm_percentiles, lo, hi, obs_high=None):
    """Compute bracket probability from NBM percentiles using piecewise linear CDF."""
    if not nbm_percentiles:
        return None

    cdf_points = []
    pct_map = {'p10': 0.10, 'p25': 0.25, 'p50': 0.50, 'p75': 0.75, 'p90': 0.90}
    for key, prob in sorted(pct_map.items(), key=lambda x: x[1]):
        if key in nbm_percentiles:
            cdf_points.append((nbm_percentiles[key], prob))

    if len(cdf_points) < 2:
        return None

    cdf_points.sort(key=lambda x: x[0])

    def cdf(t):
        if t <= cdf_points[0][0]:
            return max(0.0, cdf_points[0][1] * (t - (cdf_points[0][0] - 5)) / 5)
        if t >= cdf_points[-1][0]:
            remaining = 1.0 - cdf_points[-1][1]
            span = max(cdf_points[-1][0] - cdf_points[-2][0], 1.0)
            return min(1.0, cdf_points[-1][1] + remaining * (t - cdf_points[-1][0]) / span)
        for i in range(len(cdf_points) - 1):
            t0, p0 = cdf_points[i]
            t1, p1 = cdf_points[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / max(t1 - t0, 0.001)
                return p0 + frac * (p1 - p0)
        return 0.5

    if lo is None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4:
            return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5)))
    elif hi is None and lo is not None:
        return max(0.0, min(1.0, 1.0 - cdf(lo - 0.5)))
    elif lo is not None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4:
            return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5) - cdf(lo - 0.5)))
    return None

def bracket_probs_nbm(consensus, ladder_text, city, nbm_percentiles, obs_high=None, forecast=None):
    """Compute bracket probabilities using NBM as primary, sigma as fallback."""
    if nbm_percentiles and len(nbm_percentiles) >= 2:
        rows = []
        for label, lo, hi in parse_ladder(ladder_text):
            p = nbm_bracket_prob(nbm_percentiles, lo, hi, obs_high=obs_high)
            if p is None:
                sigma = choose_sigma(city, obs_high=obs_high, forecast=forecast)
                p = _sigma_bracket_prob(consensus, lo, hi, sigma, obs_high)
            rows.append((label, max(0.0, min(1.0, p))))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows, 'NBM', True
    else:
        sigma_rows, sigma = bracket_probs(consensus, ladder_text, city, obs_high=obs_high, forecast=forecast)
        return sigma_rows, sigma, False

def _sigma_bracket_prob(mu, lo, hi, sigma, obs_high=None):
    if obs_high is not None and hi is not None and obs_high > hi + 0.4:
        return 0.0
    if lo is None:
        return normal_cdf(hi + 0.5, mu, sigma)
    elif hi is None:
        return 1 - normal_cdf(lo - 0.5, mu, sigma)
    else:
        return normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)

# ── V5.1: Bracket Boundary Warning ───────────────────────────────────────────
def check_bracket_boundary(consensus, ladder_text, boundary_threshold=0.5):
    """
    V5.1: Flag when consensus falls within 0.5F of any bracket ceiling or floor.
    NWS rounds to whole numbers so 73.9 -> 74, meaning boundary proximity is real risk.
    """
    warnings = []
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and lo is not None:
            if abs(consensus - hi) <= boundary_threshold:
                warnings.append(
                    f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of '
                    f'{hi}F ceiling ({label}) — NWS rounding could push to next bracket up. Verify before betting.'
                )
            if abs(consensus - lo) <= boundary_threshold:
                warnings.append(
                    f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of '
                    f'{lo}F floor ({label}) — NWS rounding could push to bracket below. Verify before betting.'
                )
        elif hi is not None and lo is None:
            if abs(consensus - hi) <= boundary_threshold:
                warnings.append(
                    f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of '
                    f'{hi}F ceiling ({label}) — NWS rounding could push to next bracket up. Verify before betting.'
                )
        elif lo is not None and hi is None:
            if abs(consensus - lo) <= boundary_threshold:
                warnings.append(
                    f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of '
                    f'{lo}F floor ({label}) — NWS rounding could push to bracket below. Verify before betting.'
                )
    return warnings

# ── V5.1: Cold Front Detection ────────────────────────────────────────────────
def check_cold_front_warning(obs_high, current_temp, nws_forecast):
    """
    V5.1: Detect when daily high may already be in due to cold front passage.
    Triggers when current temp drops 5F+ below obs high.
    """
    if obs_high is None or current_temp is None:
        return None
    temp_drop = obs_high - current_temp
    if temp_drop >= 5.0:
        msg = (
            f'⚠️ Peak may already be in: Obs high {obs_high}F but current temp is '
            f'{round(current_temp, 1)}F ({round(temp_drop, 1)}F drop). '
            f'Cold front passage likely — verify before betting.'
        )
        if nws_forecast is not None and current_temp < nws_forecast - 3.0:
            msg += f' NWS forecast ({nws_forecast}F) also now above current temp.'
        return msg
    return None

# ── NWS Grid Cache ────────────────────────────────────────────────────────────
_NWS_GRID_CACHE = {}

def fetch_nws_grid(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _NWS_GRID_CACHE:
        return _NWS_GRID_CACHE[key]
    try:
        r = requests.get(f'https://api.weather.gov/points/{lat},{lon}', headers=HEADERS, timeout=12)
        r.raise_for_status()
        props = r.json().get('properties', {})
        office = props.get('gridId')
        gx = props.get('gridX')
        gy = props.get('gridY')
        fc_url = props.get('forecast')
        if not all([office, gx is not None, gy is not None, fc_url]):
            return None
        result = (office, gx, gy, fc_url)
        _NWS_GRID_CACHE[key] = result
        return result
    except Exception:
        return None

def fetch_nws_forecast(lat, lon):
    grid = fetch_nws_grid(lat, lon)
    if not grid:
        return None, None
    _, _, _, fc_url = grid
    try:
        r = requests.get(fc_url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        periods = r.json().get('properties', {}).get('periods', [])
    except Exception:
        return None, None
    today = get_eastern_date()
    for period in periods:
        start = period.get('startTime', '')
        is_day = period.get('isDaytime', False)
        temp = period.get('temperature')
        unit = period.get('temperatureUnit', 'F')
        if start.startswith(today) and is_day and temp is not None:
            temp_f = float(temp) if unit == 'F' else float(temp) * 9/5 + 32
            return round(temp_f, 1), fc_url
    for period in periods[:2]:
        temp = period.get('temperature')
        unit = period.get('temperatureUnit', 'F')
        if temp is not None:
            temp_f = float(temp) if unit == 'F' else float(temp) * 9/5 + 32
            return round(temp_f, 1), fc_url
    return None, None

def fetch_nws_current(lat, lon, station_id):
    if station_id:
        obs = safe_get('https://api.weather.gov/stations/' + station_id + '/observations/latest')
        if obs:
            temp_c = obs.get('properties', {}).get('temperature', {}).get('value')
            if temp_c is not None:
                return station_id, float(c_to_f(temp_c))
    points = safe_get('https://api.weather.gov/points/' + str(lat) + ',' + str(lon))
    if not points:
        return station_id, None
    stations_url = points.get('properties', {}).get('observationStations')
    if not stations_url:
        return station_id, None
    stations = safe_get(stations_url)
    if not stations or not stations.get('observationStations'):
        return station_id, None
    first = stations['observationStations'][0]
    sid = first.rstrip('/').split('/')[-1]
    obs = safe_get(first + '/observations/latest')
    if not obs:
        return sid, None
    temp_c = obs.get('properties', {}).get('temperature', {}).get('value')
    if temp_c is None:
        return sid, None
    return sid, float(c_to_f(temp_c))

# ── Kelly Criterion ───────────────────────────────────────────────────────────
def kelly_bet(model_prob, market_price_cents, bankroll, fractional=0.15, max_pct=0.05, max_dollars=100):
    if market_price_cents is None or market_price_cents <= 0 or market_price_cents >= 100:
        return 0.0
    p = model_prob
    q = 1.0 - p
    price = market_price_cents / 100.0
    odds = (1.0 - price) / price
    kelly_full = (p * odds - q) / odds
    if kelly_full <= 0:
        return 0.0
    kelly_frac = kelly_full * fractional
    raw = kelly_frac * bankroll
    capped = min(raw, max_pct * bankroll, max_dollars)
    return round(max(0.0, capped), 2)

def edge_cents(model_prob, market_price_cents):
    if market_price_cents is None:
        return None
    return round(model_prob * 100 - market_price_cents, 1)

def edge_signal(e, high_uncertainty=False):
    if e is None:
        return '⚪', 'No price'
    if high_uncertainty:
        if e >= MIN_EDGE:
            return '🟡', 'SKIP (uncertain)'
        if e >= 3:
            return '🟡', 'SKIP'
        return '🔴', 'AVOID'
    if e >= MIN_EDGE:
        return '🟢', 'BET'
    if e >= 3:
        return '🟡', 'SKIP'
    return '🔴', 'AVOID'

def no_edge_cents(model_prob, no_ask_cents):
    if no_ask_cents is None:
        return None
    return round((1.0 - model_prob) * 100 - no_ask_cents, 1)

def no_signal(no_edge, busted=False, model_prob=None, no_ask=None, high_uncertainty=False):
    if busted:
        if no_ask is not None and no_ask <= 5:
            return '🟢', 'BET NO (busted)'
        return '🟡', 'CONSIDER NO (busted)'
    if no_edge is None:
        return '⚪', 'No price'
    if high_uncertainty:
        if no_edge >= MIN_EDGE:
            return '🟡', 'SKIP NO (uncertain)'
        if no_edge >= 0:
            return '🔴', 'AVOID'
        return '🔴', 'AVOID'
    if no_edge >= MIN_EDGE:
        return '🟢', 'BET NO'
    if no_edge >= 3:
        return '🟡', 'SKIP NO'
    if no_edge >= 0:
        return '🔴', 'AVOID'
    return '🔴', 'AVOID'

def kelly_bet_no(model_prob, no_ask_cents, bankroll, fractional=0.15, max_pct=0.05, max_dollars=100):
    if no_ask_cents is None or no_ask_cents <= 0 or no_ask_cents >= 100:
        return 0.0
    p = 1.0 - model_prob
    q = 1.0 - p
    price = no_ask_cents / 100.0
    odds = (1.0 - price) / price
    kelly_full = (p * odds - q) / odds
    if kelly_full <= 0:
        return 0.0
    kelly_frac = kelly_full * fractional
    raw = kelly_frac * bankroll
    capped = min(raw, max_pct * bankroll, max_dollars)
    return round(max(0.0, capped), 2)

def get_city_best_signals(city, consensus, ladder_text, ensemble_members, kalshi_markets_data,
                          obs_high, high_uncertainty, bankroll, nbm_percentiles=None):
    if consensus is None or not ladder_text:
        return '—', '—'
    try:
        prob_rows, _, used_nbm = bracket_probs_nbm(
            consensus, ladder_text, city, nbm_percentiles, obs_high=obs_high
        )
        best_yes = None
        best_yes_edge = -999
        best_no = None
        best_no_edge = -999

        for label, base_prob in prob_rows:
            ens_prob = None
            for lbl, lo, hi in parse_ladder(ladder_text):
                if labels_match(lbl, label):
                    ens_prob = ensemble_bracket_prob(ensemble_members, lo, hi)
                    break
            if used_nbm:
                final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=True)
            else:
                final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city)

            yes_ask = no_ask = None
            if kalshi_markets_data:
                match = next((m for m in kalshi_markets_data if labels_match(m[0], label)), None)
                if match:
                    yes_ask, no_ask = match[1], match[2]

            busted = False
            if obs_high is not None:
                for lbl, lo, hi in parse_ladder(ladder_text):
                    if labels_match(lbl, label) and hi is not None and obs_high > hi + 0.4:
                        busted = True

            e = edge_cents(final_prob, yes_ask)
            if e is not None and e > best_yes_edge and not busted:
                best_yes_edge = e
                icon, _ = edge_signal(e, high_uncertainty)
                if icon == '🟢':
                    kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0
                    best_yes = f'🟢 {label} | +{e}c | ${kelly}'

            no_e = no_edge_cents(final_prob, no_ask)
            no_icon, _ = no_signal(no_e, busted=busted, model_prob=final_prob,
                                   no_ask=no_ask, high_uncertainty=high_uncertainty)
            if no_icon == '🟢' and no_e is not None and no_e > best_no_edge:
                best_no_edge = no_e
                kelly_no = kelly_bet_no(final_prob, no_ask, bankroll) if no_ask else 0.0
                best_no = f'🟢 {label} NO | +{no_e}c | ${kelly_no}'

        return best_yes or '—', best_no or '—'
    except Exception:
        return '—', '—'

def fetch_gfs_ensemble(lat, lon):
    url = 'https://ensemble-api.open-meteo.com/v1/ensemble'
    params = {
        'latitude': lat, 'longitude': lon,
        'hourly': 'temperature_2m',
        'temperature_unit': 'fahrenheit',
        'timezone': 'auto',
        'forecast_days': 2,
        'models': 'gfs_seamless',
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None, None
    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    today_indices = [
        i for i, t in enumerate(times)
        if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21
    ]
    if not today_indices:
        today_indices = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_indices:
        return None, None
    member_maxes = []
    for key, vals in hourly.items():
        if key == 'time' or 'temperature_2m' not in key:
            continue
        if not isinstance(vals, list):
            continue
        today_vals = [vals[i] for i in today_indices if i < len(vals) and vals[i] is not None]
        if today_vals:
            try:
                member_maxes.append(round(max(float(v) for v in today_vals), 1))
            except Exception:
                pass
    if len(member_maxes) < 3:
        return None, None
    mean = round(sum(member_maxes) / len(member_maxes), 1)
    return member_maxes, mean

def ensemble_bracket_prob(members, lo, hi):
    if not members:
        return None
    count = sum(
        1 for m in members
        if (lo is None or m >= lo - 0.5) and (hi is None or m <= hi + 0.5)
    )
    return count / len(members)

def ensemble_confidence(prob):
    if prob is None:
        return ''
    if prob >= 0.80 or prob <= 0.20:
        return '🔵 HIGH'
    if prob >= 0.65 or prob <= 0.35:
        return '🟡 MED'
    return '⚪ LOW'

def ensemble_overall_confidence(members, consensus, ladder_text):
    if not members or not ladder_text:
        return ''
    try:
        best_ens_prob = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            mid = None
            if lo is None and hi is not None:
                mid = hi - 1.0
            elif hi is None and lo is not None:
                mid = lo + 1.0
            elif lo is not None and hi is not None:
                mid = (lo + hi) / 2.0
            if mid is not None and consensus is not None and abs(mid - consensus) <= 2.0:
                prob = ensemble_bracket_prob(members, lo, hi)
                if prob is not None:
                    best_ens_prob = prob
                    break
        if best_ens_prob is None:
            probs = []
            for lbl, lo, hi in parse_ladder(ladder_text):
                p = ensemble_bracket_prob(members, lo, hi)
                if p is not None:
                    probs.append(p)
            best_ens_prob = max(probs) if probs else None
        return ensemble_confidence(best_ens_prob)
    except Exception:
        return ''

def blend_probs(sigma_prob, ensemble_prob, members, city='', nbm_active=False):
    if ensemble_prob is None or members is None:
        return sigma_prob
    base_weight = GFS_CITY_WEIGHT.get(city, 0.20)
    ensemble_weight = base_weight * 0.5 if nbm_active else base_weight
    sigma_weight = 1.0 - ensemble_weight
    return round(sigma_weight * sigma_prob + ensemble_weight * ensemble_prob, 4)

# ── Core Math ─────────────────────────────────────────────────────────────────
def get_eastern_date():
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern).strftime('%Y-%m-%d')

def get_eastern_datetime():
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern)

def get_local_hour(city):
    tz_name = CITY_TZ.get(city, 'America/New_York')
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).hour

def get_event_ticker(series):
    return series + '-' + get_eastern_datetime().strftime('%d%b%y').upper()

def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def safe_get_with_retry(url, params=None, retries=3, delay=2.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def c_to_f(c):
    return c * 9 / 5 + 32

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def normalize_label(label):
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)', lambda m: m.group(1)+'-'+m.group(2), label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)', lambda m: m.group(1)+'-'+m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    label = label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above')
    return label.strip()

def label_to_numeric_key(label):
    label = normalize_label(label)
    nums = [int(x) for x in re.findall(r'\d+', label)]
    low = label.lower()
    if not nums:
        return None, None
    if 'below' in low:
        return None, nums[0]
    if 'above' in low:
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def labels_match(label_a, label_b):
    lo_a, hi_a = label_to_numeric_key(label_a)
    lo_b, hi_b = label_to_numeric_key(label_b)
    return lo_a == lo_b and hi_a == hi_b

def parse_ladder(text):
    out = []
    for p in text.split('|'):
        p = normalize_label(p)
        nums = [int(x) for x in re.findall(r'\d+', p)]
        if not nums:
            continue
        low = p.lower()
        if 'below' in low:
            out.append((p, None, nums[0]))
        elif 'above' in low:
            out.append((p, nums[0], None))
        elif len(nums) >= 2:
            out.append((p, nums[0], nums[1]))
    return out

def choose_sigma(city, obs_high=None, forecast=None):
    s = BASE_SIGMA.get(city, 2.1)
    local_hour = get_local_hour(city)
    s *= 1.00 if local_hour < 11 else 0.94 if local_hour < 14 else 0.90 if local_hour < 16 else 0.86
    if city in DESERT_CITIES:
        s *= 0.92
    if obs_high is not None and forecast is not None:
        gap = abs(forecast - obs_high)
        if gap < 2:
            s *= 0.80
        elif gap < 4:
            s *= 0.90
    return max(1.30, min(2.80, s))

def late_day_floor(fc, obs, local_hour):
    gap = max(0.0, fc - obs)
    frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap

def compute_consensus(fc, cur, noaa, city, obs_high=None):
    local_hour = get_local_hour(city)
    is_fc_heavy = city in FORECAST_HEAVY_CITIES
    if is_fc_heavy and local_hour < 10:
        base = fc * 0.95 + (noaa if noaa is not None else cur) * 0.05 if (noaa is not None or cur is not None) else fc
    elif is_fc_heavy and local_hour < 14:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.90 + obs_val * 0.10 if obs_val is not None else fc
    elif is_fc_heavy and local_hour < 16:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.75 + obs_val * 0.25 if obs_val is not None else fc
    elif local_hour < 10:
        base = fc * 0.90 + cur * 0.07 + (noaa * 0.03 if noaa is not None else 0) if noaa is not None else fc * 0.93 + cur * 0.07
    elif local_hour < 12:
        base = fc * 0.80 + cur * 0.12 + (noaa * 0.08 if noaa is not None else 0) if noaa is not None else fc * 0.85 + cur * 0.15
    elif local_hour < 14:
        base = fc * 0.65 + cur * 0.18 + noaa * 0.17 if noaa is not None else fc * 0.78 + cur * 0.22
    else:
        base = fc * 0.45 + cur * 0.25 + noaa * 0.30 if noaa is not None else fc * 0.60 + cur * 0.40
    if abs(base - fc) > 4.0:
        base = fc - 4.0 if base < fc else fc + 4.0
    obs = noaa if noaa is not None else cur
    if obs is not None:
        consensus = max(base, late_day_floor(fc, obs, local_hour))
    else:
        consensus = base
    if obs_high is not None and obs_high > consensus:
        consensus = obs_high
    return consensus

def bracket_probs(mu, ladder_text, city, obs_high=None, forecast=None):
    sigma = choose_sigma(city, obs_high=obs_high, forecast=forecast)
    rows = []
    for label, lo, hi in parse_ladder(ladder_text):
        if obs_high is not None and hi is not None and obs_high > hi + 0.4:
            rows.append((label, 0.0))
            continue
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((label, max(0.0, min(1.0, p))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, sigma

def two_degree_call(mu, ladder_text, obs_high=None):
    best_label, best_dist = None, float('inf')
    for label, lo, hi in parse_ladder(ladder_text):
        if obs_high is not None and hi is not None and obs_high > hi + 0.4:
            continue
        if lo is None and hi is not None:
            mid = hi - 1.0
        elif hi is None and lo is not None:
            mid = lo + 1.0
        elif lo is not None and hi is not None:
            mid = (lo + hi) / 2
        else:
            continue
        dist = abs(mid - mu)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label

def ladder_to_boxes(text):
    parts = [normalize_label(p) for p in text.split('|')]
    while len(parts) < 6:
        parts.append('')
    return parts[:6]

def boxes_to_ladder(parts):
    cleaned = []
    for i, p in enumerate(parts):
        t = normalize_label(p)
        if not t:
            continue
        nums = re.findall(r'\d+', t)
        low = t.lower()
        if 'below' in low or 'above' in low or '-' in t:
            cleaned.append(t)
        elif len(nums) == 1:
            n = int(nums[0])
            cleaned.append(str(n) + (' or below' if i == 0 else ' or above' if i == 5 else ''))
        else:
            cleaned.append(t)
    return ' | '.join(cleaned)

# ── Data Fetchers ─────────────────────────────────────────────────────────────
def fetch_obs_high_today(icao):
    eastern = pytz.timezone('America/New_York')
    today_day = str(datetime.now(eastern).day)
    url = 'https://forecast.weather.gov/data/obhistory/' + icao + '.html'
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
    except Exception:
        return None, url
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    table = max(tables, key=lambda t: len(t.find_all('tr')), default=None) if tables else None
    if not table:
        return None, url
    highs = []
    for row in table.find_all('tr'):
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cols or len(cols) < 9 or cols[0] != today_day:
            continue
        try:
            t = float(cols[8])
            if 0 < t < 130:
                highs.append(t)
        except Exception:
            pass
    return (round(max(highs), 1), url) if highs else (None, url)

def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = (m.get(field) or '').replace('\u00b0', '').replace('deg', '').strip()
        if s:
            s = normalize_label(s)
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below:
                return below.group(1)+' or below', int(below.group(1))-10000
            if above:
                return above.group(1)+' or above', int(above.group(1))+10000
            if rng:
                return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
    title = (m.get('title') or '').replace('\u00b0', '').replace('**', '').replace('deg', '')
    if title:
        ma = re.search(r'be\s*[>=]+\s*(\d+)', title, re.I)
        if ma:
            n = int(ma.group(1))
            return str(n)+' or above', n+10000
        mb = re.search(r'be\s*[<=]+\s*(\d+)', title, re.I)
        if mb:
            n = int(mb.group(1))
            return str(n)+' or below', n-10000
        mr = re.search(r'be\s*(\d+)\s*(?:to|-)\s*(\d+)', title, re.I)
        if mr:
            lo, hi = int(mr.group(1)), int(mr.group(2))
            return str(lo)+'-'+str(hi), lo
        nums = re.findall(r'\d+', title)
        if len(nums) >= 2:
            lo, hi = int(nums[-2]), int(nums[-1])
            if 0 < hi-lo <= 5:
                return str(lo)+'-'+str(hi), lo
    cap = m.get('cap_strike')
    floor_s = m.get('floor_strike')
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
    for field in ['short_title', 'market_title', 'name']:
        val = (m.get(field) or '').replace('\u00b0', '').strip()
        if val:
            val = normalize_label(val)
            rng = re.match(r'^(\d+)-(\d+)$', val)
            below = re.match(r'^(\d+)\s*or\s*below$', val, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', val, re.I)
            if rng:
                return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
            if below:
                return below.group(1)+' or below', int(below.group(1))-10000
            if above:
                return above.group(1)+' or above', int(above.group(1))+10000
    return None, None

def get_price_cents(m):
    yes_ask = no_ask = None
    for f in ['yes_ask_dollars', 'yes_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                yes_ask = round(float(v)*100)
                break
            except Exception:
                pass
    for f in ['no_ask_dollars', 'no_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                no_ask = round(float(v)*100)
                break
            except Exception:
                pass
    if yes_ask is None:
        raw = m.get('yes_ask') or m.get('yes_bid')
        if raw is not None:
            try:
                yes_ask = int(raw)
            except Exception:
                pass
    if no_ask is None:
        raw = m.get('no_ask') or m.get('no_bid')
        if raw is not None:
            try:
                no_ask = int(raw)
            except Exception:
                pass
    return yes_ask, no_ask

def fetch_kalshi_brackets(series, retries=3):
    url = 'https://api.elections.kalshi.com/trade-api/v2/markets'
    event_ticker = get_event_ticker(series)
    today_date = get_eastern_date()
    today_upper = get_eastern_datetime().strftime('%y%b%d').upper()
    today_upper2 = get_eastern_datetime().strftime('%d%b%y').upper()
    today_upper3 = get_eastern_datetime().strftime('%d%b%Y').upper()
    data = safe_get_with_retry(url, {'event_ticker': event_ticker, 'limit': 30}, retries=retries, delay=2.0)
    if not data or not data.get('markets'):
        data = safe_get_with_retry(url, {'series_ticker': series, 'status': 'open', 'limit': 30}, retries=retries, delay=2.0)
    if not data or not data.get('markets'):
        data = safe_get_with_retry(url, {'series_ticker': series, 'limit': 30}, retries=retries, delay=2.0)
    if not data or not data.get('markets'):
        return None
    all_markets = data['markets']
    markets = [m for m in all_markets if
               today_upper in (m.get('ticker') or '').upper() or
               today_upper2 in (m.get('ticker') or '').upper() or
               today_upper3 in (m.get('ticker') or '').upper() or
               today_upper2 in (m.get('event_ticker') or '').upper() or
               today_upper3 in (m.get('event_ticker') or '').upper()]
    if not markets:
        markets = [m for m in all_markets if (m.get('close_time') or '').startswith(today_date)]
    if not markets:
        markets = all_markets
    parsed = []
    for m in markets:
        label, key = parse_market_label(m)
        if label is None:
            continue
        yes_ask, no_ask = get_price_cents(m)
        parsed.append((key, label, yes_ask, no_ask))
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[0])
    return [(label, yes_ask, no_ask) for _, label, yes_ask, no_ask in parsed]

def get_cached_prices(city):
    cache = load_json(PRICE_CACHE_FILE)
    entry = cache.get(city)
    if not entry:
        return None, None
    if (time.time() - entry.get('fetched_at', 0)) / 60 > PRICE_CACHE_MINUTES:
        return None, None
    return entry.get('markets'), entry.get('fetched_at')

def save_cached_prices(city, markets):
    cache = load_json(PRICE_CACHE_FILE)
    cache[city] = {'fetched_at': time.time(), 'markets': markets}
    save_json(PRICE_CACHE_FILE, cache)

def clear_city_cache(city):
    cache = load_json(PRICE_CACHE_FILE)
    if city in cache:
        del cache[city]
    save_json(PRICE_CACHE_FILE, cache)

def sync_all_ladders(saved_ladders, force=False):
    today = get_eastern_date()
    last_sync = load_json(LAST_SYNC_FILE)
    if not force and last_sync.get('date') == today:
        return saved_ladders, None
    cities = list(SERIES.keys())
    progress = st.progress(0, text='Syncing all city ladders from Kalshi...')
    synced, failed = [], []
    for i, c in enumerate(cities):
        progress.progress((i+1)/len(cities), text='Syncing ' + c + '...')
        markets = fetch_kalshi_brackets(SERIES[c], retries=3)
        if markets:
            labels = [normalize_label(m[0]) for m in markets]
            while len(labels) < 6:
                labels.append('')
            saved_ladders[c] = ' | '.join(labels[:6])
            save_cached_prices(c, markets)
            synced.append(c)
        else:
            failed.append(c)
        time.sleep(0.5)
    save_json(SAVE_FILE, saved_ladders)
    save_json(LAST_SYNC_FILE, {'date': today, 'synced': synced, 'failed': failed})
    progress.empty()
    return saved_ladders, {'synced': synced, 'failed': failed}

# ── Cached per-city weather fetch ────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_city_weather(city):
    coords = CITIES[city]
    lat, lon = coords['lat'], coords['lon']
    station = STATIONS[city]
    obs_icao = OBHISTORY_STATIONS[city]

    nws_fc, _ = fetch_nws_forecast(lat, lon)
    _, current_temp = fetch_nws_current(lat, lon, station)
    obs_high_raw, _ = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)

    obs_high_final = obs_high_raw
    obs_high_discarded = False
    obs_high_discard_reason = None

    if obs_high_raw is not None and current_temp is not None:
        if obs_high_raw > current_temp + 15.0:
            obs_high_final = None
            obs_high_discarded = True
            obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - current_temp, 1)}F above current temp (likely wrong-day data)'
    if obs_high_raw is not None and nws_fc is not None and not obs_high_discarded:
        if obs_high_raw > nws_fc + 12.0:
            obs_high_final = None
            obs_high_discarded = True
            obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - nws_fc, 1)}F above NWS forecast (implausible)'

    if ensemble_mean is not None and nws_fc is not None:
        if abs(ensemble_mean - nws_fc) > 8.0:
            ensemble_members = None
            ensemble_mean = None

    source_gap = None
    high_uncertainty = False
    if nws_fc is not None and ensemble_mean is not None:
        source_gap = abs(nws_fc - ensemble_mean)
        uncertainty_threshold = 6.0 if city in DESERT_CITIES else 5.0
        high_uncertainty = source_gap > uncertainty_threshold

    return {
        'nws_fc': nws_fc,
        'current_temp': current_temp,
        'obs_high': obs_high_final,
        'obs_high_raw': obs_high_raw,
        'obs_high_discarded': obs_high_discarded,
        'obs_high_discard_reason': obs_high_discard_reason,
        'ensemble_members': ensemble_members,
        'ensemble_mean': ensemble_mean,
        'source_gap': source_gap,
        'high_uncertainty': high_uncertainty,
        'nbm_percentiles': nbm_percentiles,
    }

def save_city_prediction(city, weather, saved_ladders):
    nws_fc = weather['nws_fc']
    if nws_fc is None:
        return None, False
    current_temp = weather['current_temp']
    obs_high = weather['obs_high']
    ensemble_mean = weather['ensemble_mean']
    source_gap = weather['source_gap']
    high_uncertainty = weather['high_uncertainty']

    cur = current_temp if current_temp is not None else nws_fc
    consensus_raw = compute_consensus(nws_fc, cur, current_temp, city, obs_high=obs_high)
    bias_correction, _ = compute_bias_correction_db(city)
    consensus = round(consensus_raw + bias_correction, 1)

    save_ok = sb_upsert_prediction(
        city=city, consensus=consensus, forecast=nws_fc,
        ensemble_mean=ensemble_mean, source_gap=source_gap,
        high_uncertainty=high_uncertainty, obs_high=obs_high,
        bias_correction=bias_correction,
    )
    return consensus, save_ok

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Kelly Settings')
    bankroll = st.number_input('My Bankroll ($)', min_value=10.0, max_value=100000.0,
                               value=500.0, step=10.0)
    st.caption('Used to calculate optimal bet sizes.')
    st.markdown('---')
    st.markdown('**Edge threshold:** ' + str(MIN_EDGE) + 'c minimum to bet')
    st.markdown('**Kelly fraction:** 15% (conservative)')
    st.markdown('**Max per trade:** min(5% bankroll, $100)')
    st.markdown('---')
    st.markdown('**Signal Key**')
    st.markdown('🟢 Edge >=8c — **BET**')
    st.markdown('🟡 Edge 3-7c — **SKIP**')
    st.markdown('🔴 Edge <3c — **AVOID**')
    st.markdown('🟡 SKIP (uncertain) — NWS vs Ensemble >5F')
    st.markdown('🔵 Ensemble HIGH confidence')
    st.markdown('---')
    st.markdown('**V5.1 Changes**')
    st.markdown('- Discarded obs high now shows ⚠️ warning (not silent)')
    st.markdown('- NBM fix: Iowa State AFOS, correct field names (TXNP1/2/5/7/9)')
    st.markdown('- Cold front detection: ⚠️ Peak may already be in')
    st.markdown('- Bracket boundary warning: ⚠️ within 0.5F of ceiling/floor')
    st.markdown('- NBM/Sigma status visible in All Cities panel')

# ── Main App ──────────────────────────────────────────────────────────────────
saved_ladders = load_json(SAVE_FILE)
today_str = get_eastern_date()
last_sync_data = load_json(LAST_SYNC_FILE)

if 'auto_settled' not in st.session_state:
    with st.spinner('Checking for unsettled predictions...'):
        n_settled, settled_rows = run_auto_settlement()
    st.session_state.auto_settled = True
    if n_settled > 0:
        for s in settled_rows:
            direction = '✅' if abs(s['error']) <= 1.5 else '⚠️'
            st.success(f"{direction} Auto-settled {s['city']} ({s['date']}): actual={s['actual']}F | error={s['error']:+.1f}F")

if last_sync_data.get('date') != today_str:
    saved_ladders, results = sync_all_ladders(saved_ladders)
    if results:
        n = len(results.get('synced', []))
        st.success('Morning sync complete — ' + str(n) + '/' + str(len(SERIES)) + ' city ladders loaded from Kalshi')
        if results.get('failed'):
            st.warning('Could not fetch: ' + ', '.join(results['failed']) + ' — using saved ladders')
else:
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.caption('Ladders auto-synced from Kalshi today (' + today_str + ') — ' +
                   str(len(last_sync_data.get('synced', []))) + ' cities loaded')
    with col_btn:
        if st.button('Refresh All'):
            saved_ladders, results = sync_all_ladders(saved_ladders, force=True)
            st.success('Re-synced ' + str(len(results.get('synced', []))) + '/' + str(len(SERIES)) + ' city ladders')
            if results.get('failed'):
                st.warning('Could not fetch: ' + ', '.join(results['failed']))
            st.rerun()

city_list = list(CITIES.keys())
default_idx = city_list.index('New York')
city = st.selectbox('City', city_list, index=default_idx)

if 'last_city' not in st.session_state:
    st.session_state.last_city = city

if st.session_state.last_city != city:
    st.session_state.last_city = city
    clear_city_cache(city)
    st.rerun()

lat, lon = CITIES[city]['lat'], CITIES[city]['lon']
station = STATIONS[city]
series = SERIES[city]
obs_icao = OBHISTORY_STATIONS[city]
obs_url = 'https://forecast.weather.gov/data/obhistory/' + obs_icao + '.html'
local_hour = get_local_hour(city)
tz_name = CITY_TZ[city]

st.caption('Settlement: ' + station + ' — ' + SETTLEMENT_LOCATION[city] + ' — Series: ' + series)
st.caption('Local time: ' + str(local_hour) + ':00 ' + tz_name)
if city in FORECAST_HEAVY_CITIES and local_hour < 16:
    st.caption('Forecast-heavy mode active (Texas/OKC heat lag correction)')

bias_correction, bias_n = compute_bias_correction_db(city)
if bias_n >= 3:
    direction = 'warm' if bias_correction > 0 else 'cold'
    st.info(f'Bias correction active: +{bias_correction}F applied to consensus '
            f'(model ran {direction} by avg {abs(bias_correction)}F over last {bias_n} days)')
elif bias_n > 0:
    st.caption(f'Bias correction: {bias_n} settlement(s) logged for {city} — need 3+ for correction')
else:
    st.caption(f'Bias correction: no history yet for {city} — will activate after 3 settled days')

if city not in saved_ladders:
    saved_ladders[city] = DEFAULT_LADDERS.get(city, '')

kalshi_markets, fetched_at = get_cached_prices(city)
if kalshi_markets is None:
    with st.spinner('Fetching live Kalshi prices for ' + city + '...'):
        kalshi_markets = fetch_kalshi_brackets(series, retries=3)
        if kalshi_markets:
            save_cached_prices(city, kalshi_markets)
            labels = [normalize_label(m[0]) for m in kalshi_markets]
            while len(labels) < 6:
                labels.append('')
            saved_ladders[city] = ' | '.join(labels[:6])
            save_json(SAVE_FILE, saved_ladders)
            fetched_at = time.time()

st.subheader('Kalshi Ladder')
if kalshi_markets:
    age_min = round((time.time() - fetched_at) / 60) if fetched_at else 0
    age_str = 'just now' if age_min < 1 else str(age_min) + ' min ago'
    st.success('Live prices loaded — ' + str(len(kalshi_markets)) + ' brackets (fetched ' + age_str + ')')
    for m in kalshi_markets:
        st.caption(' ' + m[0] + ' | YES: ' + (str(m[1])+'c' if m[1] else 'no price') +
                   ' | NO: ' + (str(m[2])+'c' if m[2] else 'no price'))
else:
    st.warning('Could not fetch live prices from Kalshi. Using saved ladder.')

if st.button('Refresh Prices'):
    clear_city_cache(city)
    st.rerun()

box_values = ladder_to_boxes(saved_ladders[city])
with st.expander('Edit Brackets', expanded=False):
    cols = st.columns(6)
    new_boxes = []
    for i, col in enumerate(cols):
        with col:
            new_boxes.append(st.text_input('Box '+str(i+1), value=box_values[i], key=city+'_b'+str(i)))
    if st.button('Save Ladder'):
        saved_ladders[city] = boxes_to_ladder(new_boxes)
        save_json(SAVE_FILE, saved_ladders)
        st.success('Saved')
        st.rerun()

ladder_text = saved_ladders[city]
st.caption('Current ladder: ' + ladder_text)

st.subheader('Live Weather')
with st.spinner('Fetching weather data...'):
    nws_forecast, nws_fc_url = fetch_nws_forecast(lat, lon)
    noaa_station, noaa_obs = fetch_nws_current(lat, lon, station)
    obs_high_raw, obs_high_url = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)

sanity_warnings = []
obs_high_today = obs_high_raw
obs_high_suspect = False
obs_high_discard_reason = None

if obs_high_raw is not None:
    if noaa_obs is not None and obs_high_raw > noaa_obs + 15.0:
        obs_high_today = None
        obs_high_suspect = True
        obs_high_discard_reason = f'Obs high of {obs_high_raw}F discarded — {round(obs_high_raw - noaa_obs, 1)}F above current temp ({round(noaa_obs, 1)}F). Likely wrong-day data. Verify manually before betting.'
        sanity_warnings.append(obs_high_discard_reason)
    elif nws_forecast is not None and obs_high_raw > nws_forecast + 12.0:
        obs_high_today = None
        obs_high_suspect = True
        obs_high_discard_reason = f'Obs high of {obs_high_raw}F discarded — {round(obs_high_raw - nws_forecast, 1)}F above NWS forecast ({nws_forecast}F). Implausible. Verify manually before betting.'
        sanity_warnings.append(obs_high_discard_reason)

nws_stale = False
if nws_forecast is not None and noaa_obs is not None:
    if noaa_obs > nws_forecast + 5.0:
        nws_stale = True
        sanity_warnings.append(
            f'NWS forecast ({nws_forecast}F) is {round(noaa_obs - nws_forecast, 1)}F below current temp '
            f'({round(noaa_obs, 1)}F) — forecast may be stale.')

ensemble_suspect = False
if ensemble_mean is not None and nws_forecast is not None:
    if abs(ensemble_mean - nws_forecast) > 8.0:
        ensemble_suspect = True
        sanity_warnings.append(
            f'GFS ensemble ({ensemble_mean}F) differs from NWS by '
            f'{round(abs(ensemble_mean - nws_forecast), 1)}F — discarded (>8F threshold).')
        ensemble_members = None
        ensemble_mean = None

high_uncertainty = False
source_gap = None
if nws_forecast is not None and ensemble_mean is not None:
    source_gap = abs(nws_forecast - ensemble_mean)
    uncertainty_threshold = 6.0 if city in DESERT_CITIES else 5.0
    high_uncertainty = source_gap > uncertainty_threshold

col1, col2, col3, col4 = st.columns(4)
with col1:
    if nws_forecast:
        st.metric('NWS Forecast', str(nws_forecast)+' F')
        st.caption('Primary — settlement source' + (' (stale?)' if nws_stale else ''))
    else:
        st.metric('NWS Forecast', 'Unavailable')
with col2:
    if noaa_obs is not None:
        st.metric('Current Temp', str(round(noaa_obs, 1))+' F')
        st.caption('Station: ' + noaa_station)
    else:
        st.metric('Current Temp', 'Unavailable')
with col3:
    if obs_high_today is not None:
        st.metric('Obs High Today', str(obs_high_today)+' F', delta='floor active')
        wu_url = WUNDERGROUND_URLS.get(city, '')
        st.caption('[NWS table](' + obs_url + ')' + (' · [Wunderground ↗](' + wu_url + ')' if wu_url else ''))
    elif obs_high_suspect:
        st.metric('Obs High Today', str(obs_high_raw)+'F ⚠️')
        wu_url = WUNDERGROUND_URLS.get(city, '')
        st.caption('⚠️ Discarded — verify manually' + (' · [Wunderground ↗](' + wu_url + ')' if wu_url else ''))
    else:
        st.metric('Obs High Today', 'Unavailable')
        wu_url = WUNDERGROUND_URLS.get(city, '')
        st.caption('[NWS table](' + obs_url + ')' + (' · [Wunderground ↗](' + wu_url + ')' if wu_url else ''))
with col4:
    if ensemble_mean is not None:
        n_members = len(ensemble_members) if ensemble_members else 0
        gfs_weight_pct = int(GFS_CITY_WEIGHT.get(city, 0.20) * 100)
        gap_str = ''
        if nws_forecast is not None:
            gap_val = round(nws_forecast - ensemble_mean, 1)
            gap_str = f' | NWS gap: {gap_val:+.1f}F'
        st.metric('GFS Ensemble', str(ensemble_mean)+' F', delta=str(n_members)+' members')
        st.caption(f'Weight: {gfs_weight_pct}%{gap_str}')
    elif ensemble_suspect:
        st.metric('GFS Ensemble', 'Discarded')
        st.caption('Failed sanity check (>8F from NWS)')
    else:
        st.metric('GFS Ensemble', 'Unavailable')

if nbm_percentiles:
    nbm_p50 = nbm_percentiles.get('p50', nbm_percentiles.get('p25', '—'))
    nbm_p10 = nbm_percentiles.get('p10', '—')
    nbm_p90 = nbm_percentiles.get('p90', '—')
    st.success(f'✅ NBM active — p10:{nbm_p10}F | p50:{nbm_p50}F | p90:{nbm_p90}F | bracket probs from real percentile distribution')
else:
    st.warning('⚠️ NBM unavailable — using sigma/normal distribution fallback')

for w in sanity_warnings:
    st.error('⚠️ ' + w)

if nws_forecast is None:
    st.error('NWS forecast unavailable — cannot run model.')
elif high_uncertainty and source_gap is not None:
    st.warning(f'HIGH UNCERTAINTY: NWS ({nws_forecast}F) vs GFS ({ensemble_mean}F) gap = {round(source_gap, 1)}F. Green signals suppressed.')
elif source_gap is not None and source_gap > 2.5:
    st.info(f'Source gap: NWS vs Ensemble = {round(source_gap, 1)}F — moderate divergence.')

cold_front_warning = check_cold_front_warning(obs_high_raw, noaa_obs, nws_forecast)
if cold_front_warning:
    st.warning(cold_front_warning)

if obs_high_today is not None:
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and obs_high_today > hi + 0.4:
            st.warning('BUST: ' + label + ' eliminated — obs high ' + str(obs_high_today) + 'F exceeds ' + str(hi) + 'F')

with st.expander('Override weather inputs', expanded=False):
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        override_fc = st.number_input('Forecast High F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_fc')
    with ov2:
        override_cur = st.number_input('Current Temp F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_cur')
    with ov3:
        override_noaa = st.number_input('NOAA Obs F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_noaa')
    with ov4:
        override_obs_high = st.number_input('Obs High Override F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_obs')

if override_fc > 0 or override_cur > 0 or override_obs_high > 0:
    st.info('Using manual overrides — set back to 0.0 to use auto values')

forecast = override_fc if override_fc > 0 else nws_forecast
current = override_cur if override_cur > 0 else noaa_obs
noaa_final = override_noaa if override_noaa > 0 else noaa_obs
obs_high_final = override_obs_high if override_obs_high > 0 else obs_high_today

if forecast is not None and current is not None:
    consensus_raw = compute_consensus(forecast, current, noaa_final, city, obs_high=obs_high_final)
    consensus = round(consensus_raw + bias_correction, 1)

    boundary_warnings = check_bracket_boundary(consensus, ladder_text)
    for bw in boundary_warnings:
        st.warning(bw)

    prob_rows, prob_label, used_nbm = bracket_probs_nbm(
        consensus, ladder_text, city, nbm_percentiles,
        obs_high=obs_high_final, forecast=forecast
    )
    _, sigma = bracket_probs(consensus, ladder_text, city, obs_high=obs_high_final, forecast=forecast)
    call = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)

    save_ok = sb_upsert_prediction(
        city=city, consensus=consensus, forecast=forecast,
        ensemble_mean=ensemble_mean, source_gap=source_gap,
        high_uncertainty=high_uncertainty, obs_high=obs_high_final,
        bias_correction=bias_correction,
    )
    if not save_ok:
        st.caption('⚠️ Could not save prediction to database')

    st.subheader('Model Output')
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric('Consensus High', str(round(consensus, 1))+' F')
        st.caption('Saved to DB ✓' if save_ok else 'DB save failed')
    with c2:
        st.metric('2 Degree Call', call or 'none')
    with c3:
        if used_nbm:
            p50 = nbm_percentiles.get('p50', nbm_percentiles.get('p25', '—'))
            st.metric('NBM p50', str(p50)+' F')
            st.caption('Bracket probs from NBM percentiles')
        else:
            st.metric('Sigma', str(round(sigma, 2))+' F')
            st.caption('Fallback: sigma/normal distribution')
    with c4:
        if obs_high_final is not None:
            st.metric('Obs Floor', str(obs_high_final)+' F',
                      delta='controlling' if obs_high_final >= consensus-0.1 else 'not binding')
    with c5:
        if bias_correction != 0.0:
            st.metric('Bias Adj', ('+' if bias_correction > 0 else '')+str(bias_correction)+' F',
                      delta='from '+str(bias_n)+' days')

    if ensemble_mean is not None:
        gfs_weight_pct = int(GFS_CITY_WEIGHT.get(city, 0.20) * 100)
        effective_weight = int(gfs_weight_pct * 0.5) if used_nbm else gfs_weight_pct
        st.caption(f'GFS ensemble: {ensemble_mean}F | {len(ensemble_members)} members | weight {effective_weight}%' +
                   (' (halved — NBM active)' if used_nbm else ''))
    if high_uncertainty:
        st.caption('High uncertainty mode — green signals suppressed')

    import pandas as pd
    yes_rows = []
    no_rows = []
    best_bet = None
    best_edge = -999
    best_no_bet = None
    best_no_edge = -999

    for label, base_prob in prob_rows:
        ens_prob = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            if labels_match(lbl, label):
                ens_prob = ensemble_bracket_prob(ensemble_members, lo, hi)
                break
        final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)
        fair = round(final_prob * 100)
        yes_ask = no_ask = None
        if kalshi_markets:
            match = next((m for m in kalshi_markets if labels_match(m[0], label)), None)
            if match:
                yes_ask, no_ask = match[1], match[2]

        e = edge_cents(final_prob, yes_ask)
        signal_icon, signal_text = edge_signal(e, high_uncertainty=high_uncertainty)
        kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0

        busted = False
        if obs_high_final is not None:
            for lbl, lo, hi in parse_ladder(ladder_text):
                if labels_match(lbl, label) and hi is not None and obs_high_final > hi + 0.4:
                    busted = True
        no_e = no_edge_cents(final_prob, no_ask)
        no_icon, no_text = no_signal(no_e, busted=busted, model_prob=final_prob,
                                     no_ask=no_ask, high_uncertainty=high_uncertainty)
        kelly_no = kelly_bet_no(final_prob, no_ask, bankroll) if no_ask and no_icon == '🟢' else 0.0

        ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''
        edge_str = ('+'+str(e)+'c') if e and e > 0 else (str(e)+'c' if e is not None else 'none')
        no_edge_str = ('+'+str(no_e)+'c') if no_e and no_e > 0 else (str(no_e)+'c' if no_e is not None else 'none')

        yes_signal_str = signal_icon + ' ' + signal_text if signal_text else ''
        no_signal_str = (no_icon + ' ' + no_text) if no_icon and no_text else '—'

        yes_rows.append({
            'Signal': yes_signal_str,
            'Bracket': label + (' BUSTED' if busted else ''),
            'Model %': str(round(final_prob*100, 1))+'%',
            'Mkt Implied %': str(round(yes_ask, 1))+'%' if yes_ask else '—',
            'Fair': str(fair)+'c',
            'YES ask': str(yes_ask)+'c' if yes_ask is not None else '—',
            'Edge': edge_str,
            'Kelly': ('$'+str(kelly)) if kelly > 0 else '—',
            'Ensemble': ens_conf,
        })

        no_rows.append({
            'NO Signal': no_signal_str,
            'Bracket': label + (' BUSTED' if busted else ''),
            'Model %': str(round(final_prob*100, 1))+'%',
            'Mkt Implied %': str(round(no_ask, 1))+'%' if no_ask else '—',
            'NO ask': str(no_ask)+'c' if no_ask is not None else '—',
            'NO Edge': no_edge_str,
            'Kelly NO': ('$'+str(kelly_no)) if kelly_no > 0 else '—',
            'Ensemble': ens_conf,
        })

        if e is not None and e > best_edge and not busted:
            best_edge = e
            best_bet = {'label': label, 'edge': e, 'kelly': kelly,
                        'signal': signal_icon, 'uncertain': high_uncertainty}

        if busted and no_ask is not None and no_ask <= 5:
            no_e_for_rank = no_e if no_e is not None else 95
            if no_e_for_rank > best_no_edge:
                best_no_edge = no_e_for_rank
                best_no_bet = {'label': label, 'edge': no_e_for_rank, 'kelly': kelly_no,
                               'busted': True, 'no_ask': no_ask}
        elif no_e is not None and no_e > best_no_edge and no_icon == '🟢':
            best_no_edge = no_e
            best_no_bet = {'label': label, 'edge': no_e, 'kelly': kelly_no,
                           'busted': False, 'no_ask': no_ask}

    prob_source = '(NBM percentiles)' if used_nbm else '(sigma/normal fallback)'
    st.markdown(f'#### 🟢 YES Signals {prob_source}')
    st.dataframe(pd.DataFrame(yes_rows), use_container_width=True, hide_index=True)

    if best_bet and best_bet['edge'] >= MIN_EDGE and not best_bet['uncertain']:
        st.success('🟢 Best YES Bet: **' + best_bet['label'] + '** | Edge: +' +
                   str(best_bet['edge']) + 'c | Kelly: $' + str(best_bet['kelly']))
    elif best_bet and best_bet['edge'] >= MIN_EDGE and best_bet['uncertain']:
        st.warning('Best YES edge: **' + best_bet['label'] + '** (+' + str(best_bet['edge']) +
                   'c) but HIGH UNCERTAINTY — consider skipping.')
    elif best_bet:
        st.warning('No YES bracket meets the ' + str(MIN_EDGE) + 'c minimum. Best: ' +
                   best_bet['label'] + ' (+' + str(best_bet['edge']) + 'c)')

    st.markdown(f'#### 🔴 NO Signals {prob_source}')
    st.dataframe(pd.DataFrame(no_rows), use_container_width=True, hide_index=True)

    if best_no_bet:
        if best_no_bet['busted']:
            st.success('🟢 Best NO Bet: **' + best_no_bet['label'] + ' NO** | BUSTED bracket | ' +
                       'NO ask: ' + str(best_no_bet['no_ask']) + 'c | Kelly: $' + str(best_no_bet['kelly']))
        else:
            st.success('🟢 Best NO Bet: **' + best_no_bet['label'] + ' NO** | Edge: +' +
                       str(best_no_bet['edge']) + 'c | NO ask: ' + str(best_no_bet['no_ask']) +
                       'c | Kelly: $' + str(best_no_bet['kelly']))

    parsed = parse_ladder(ladder_text)
    top_b = next((b for b in parsed if b[2] is None), None)
    bot_b = next((b for b in parsed if b[1] is None), None)
    if (top_b and consensus > top_b[1]+5) or (bot_b and bot_b[2] is not None and consensus < bot_b[2]-5):
        st.warning('Ladder does not cover consensus of '+str(round(consensus, 1))+'F — update brackets.')

else:
    if forecast is None:
        st.error('NWS forecast unavailable. Use manual override or try refreshing.')
    else:
        st.error('Current temperature unavailable — cannot compute consensus.')

# ── Calibration Panel ─────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Calibration & Settlement History')

with st.expander('View history for ' + city, expanded=False):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None]
    pending = [r for r in rows if r.get('actual') is None]

    if complete:
        import pandas as pd
        errors = [r['error'] for r in complete if r.get('error') is not None]
        mae = round(sum(abs(e) for e in errors) / len(errors), 2) if errors else None
        avg_err = round(sum(errors) / len(errors), 2) if errors else None
        within_2 = round(100 * sum(1 for e in errors if abs(e) <= 2.0) / len(errors)) if errors else None

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric('Settled Days', len(complete))
        with m2:
            st.metric('MAE', str(mae)+'F' if mae else 'n/a')
        with m3:
            st.metric('Avg Error', ('+' if avg_err and avg_err > 0 else '')+str(avg_err)+'F' if avg_err else 'n/a',
                      delta='warm bias' if avg_err and avg_err < -0.5 else 'cold bias' if avg_err and avg_err > 0.5 else 'calibrated')
        with m4:
            st.metric('Within ±2F', str(within_2)+'%' if within_2 else 'n/a')

        hist_df = pd.DataFrame([{
            'Date': r['date'],
            'Consensus': r.get('consensus'),
            'Actual': r.get('actual'),
            'Error': ('+' if r['error'] > 0 else '') + str(r['error']) + 'F' if r.get('error') is not None else '',
            'Ensemble': r.get('ensemble_mean'),
            'Uncertain': '⚠️' if r.get('high_uncertainty') else '',
        } for r in sorted(complete, key=lambda x: x['date'], reverse=True)])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info('No settled history yet for ' + city + '.')

    if pending:
        st.caption(str(len(pending)) + ' prediction(s) pending settlement: ' +
                   ', '.join(r['date'] for r in pending))

# ── All Cities Summary Panel ──────────────────────────────────────────────────
st.markdown('---')
with st.expander('All Cities — Today\'s Predictions', expanded=True):
    import pandas as pd

    all_rows = sb_fetch_all()
    today_rows = [r for r in all_rows if r.get('date') == today_str]
    saved_cities = {r['city'] for r in today_rows}
    missing_cities = [c for c in CITIES.keys() if c not in saved_cities]

    if missing_cities:
        fill_status = st.empty()
        newly_saved = []
        for c in missing_cities:
            fill_status.caption(f'Fetching {c}...')
            try:
                weather = fetch_city_weather(c)
                consensus, save_ok = save_city_prediction(c, weather, saved_ladders)
                if save_ok and consensus is not None:
                    bc, _ = compute_bias_correction_db(c)
                    today_rows.append({
                        'city': c,
                        'date': today_str,
                        'consensus': consensus,
                        'forecast': weather['nws_fc'],
                        'ensemble_mean': weather['ensemble_mean'],
                        'source_gap': weather['source_gap'],
                        'high_uncertainty': weather['high_uncertainty'],
                        'bias_correction': bc,
                    })
                    newly_saved.append(c)
            except Exception:
                pass
            time.sleep(0.4)
        fill_status.empty()
        if newly_saved:
            st.caption(f'✅ Filled in {len(newly_saved)} missing cities: {", ".join(newly_saved)}')

    if today_rows:
        summary_rows = []
        for r in sorted(today_rows, key=lambda x: x['city']):
            c = r['city']
            consensus_val = r.get('consensus')
            ladder = saved_ladders.get(c, DEFAULT_LADDERS.get(c, ''))
            cached_markets, _ = get_cached_prices(c)
            obs_h = r.get('obs_high')
            high_unc = r.get('high_uncertainty', False)
            bc_val = r.get('bias_correction', 0.0)

            members = None
            nbm_pcts = None
            try:
                cached_wx = fetch_city_weather(c)
                if cached_wx:
                    members = cached_wx.get('ensemble_members')
                    nbm_pcts = cached_wx.get('nbm_percentiles')
            except Exception:
                pass

            ens_key = ensemble_overall_confidence(members, consensus_val, ladder)
            nbm_status = '✅ NBM' if nbm_pcts else '📊 Sigma'

            best_yes, best_no = get_city_best_signals(
                c, consensus_val, ladder, members,
                cached_markets, obs_h, high_unc, bankroll,
                nbm_percentiles=nbm_pcts
            )

            if bc_val and bc_val != 0.0:
                bias_str = ('+' if bc_val > 0 else '') + str(bc_val) + 'F'
            else:
                bias_str = '—'

            summary_rows.append({
                'City': c,
                'Consensus': str(consensus_val)+'F' if consensus_val else '—',
                'NWS': str(r.get('forecast', ''))+'F' if r.get('forecast') else '—',
                'GFS': str(r.get('ensemble_mean', ''))+'F' if r.get('ensemble_mean') else '—',
                'Gap': str(round(r['source_gap'], 1))+'F' if r.get('source_gap') else '—',
                '⚠️': '⚠️' if r.get('high_uncertainty') else '✅',
                'Ens Key': ens_key if ens_key else '—',
                'Prob Src': nbm_status,
                'Bias Adj': bias_str,
                'Best YES': best_yes,
                'Best NO': best_no,
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        n_saved_total = len(today_rows)
        n_yes = sum(1 for r in summary_rows if r['Best YES'] != '—')
        n_no = sum(1 for r in summary_rows if r['Best NO'] != '—')
        n_nbm = sum(1 for r in summary_rows if '✅' in r.get('Prob Src', ''))
        status_icon = '✅' if n_saved_total == 17 else '⏳'
        st.caption(f'{status_icon} {n_saved_total}/17 cities | 🟢 {n_yes} YES signals | 🟢 {n_no} NO signals | ✅ {n_nbm} NBM active today')
    else:
        st.info('Loading predictions for all cities — this panel fills itself in automatically.')
