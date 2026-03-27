# Kalshi High Temperature Model - V4.23
#
# Changes from V4.22:
# 1. Raised high uncertainty threshold from 3.0F to 5.0F — signals no longer suppressed on normal divergence
# 2. Desert cities (Phoenix, Las Vegas) get even higher threshold of 6.0F
# 3. Moderate divergence info message raised to 2.5F
# 4. NWS stale detection improved — when NWS is 10F+ below current temp, model trusts current temp more

import math, re, json, time, requests
import streamlit as st
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title='Kalshi High Temp V4.23', layout='wide')
st.title('Kalshi High Temperature Model - V4.23')

SAVE_FILE = Path('saved_ladders.json')
LAST_SYNC_FILE = Path('last_sync.json')
PRICE_CACHE_FILE = Path('price_cache.json')
PRICE_CACHE_MINUTES = 10

MIN_EDGE = 8
HEADERS = {'User-Agent': 'kalshi-temp-model/4.20', 'Accept': 'application/geo+json, application/json, text/html'}

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
    """Rows where actual is null — pending auto-settlement."""
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
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'date': 'eq.' + today, 'city': 'eq.' + city}, timeout=10)
        rows = r.json() if r.status_code == 200 else []
        return rows[0] if rows else None
    except Exception:
        return None

def sb_upsert_prediction(city, consensus, forecast, ensemble_mean, source_gap, high_uncertainty, obs_high, bias_correction):
    today = datetime.now().strftime('%Y-%m-%d')
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
    """Fetch the obs high from NWS obhistory table for a specific date."""
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
    """Called once per hour. Finds unsettled rows and fills in actuals from NWS."""
    unsettled = sb_fetch_unsettled()
    if not unsettled:
        return 0, []
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    settled = []
    for row in unsettled:
        row_date = row.get('date', '')
        if row_date >= datetime.now().strftime('%Y-%m-%d'):
            continue  # Don't try to settle today
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
    correction = max(-3.0, min(3.0, mean_error))
    return round(correction, 2), len(recent)

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
    today = datetime.now().strftime('%Y-%m-%d')
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

# ── GFS Ensemble ──────────────────────────────────────────────────────────────
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
    today = datetime.now().strftime('%Y-%m-%d')
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
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

def blend_probs(sigma_prob, ensemble_prob, members):
    if ensemble_prob is None or members is None:
        return sigma_prob
    n = len(members)
    ensemble_weight = min(0.50, 0.35 + (n / 200.0))
    sigma_weight = 1.0 - ensemble_weight
    return round(sigma_weight * sigma_prob + ensemble_weight * ensemble_prob, 4)

# ── Core Math ─────────────────────────────────────────────────────────────────
def get_local_hour(city):
    tz_name = CITY_TZ.get(city, 'America/New_York')
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).hour

def get_event_ticker(series):
    return series + '-' + datetime.now().strftime('%d%b%y').upper()

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
    """Convert label to (lo, hi) for numeric matching — avoids string fragility."""
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
    """Numeric key comparison — immune to formatting differences."""
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
    if is_fc_heavy and local_hour < 14:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.85 + obs_val * 0.15 if obs_val is not None else fc
    elif is_fc_heavy and local_hour < 16:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.70 + obs_val * 0.30 if obs_val is not None else fc
    elif local_hour < 10:
        # Early morning — trust NWS forecast heavily, current temp is cold/misleading
        base = fc * 0.85 + cur * 0.10 + (noaa * 0.05 if noaa is not None else 0) if noaa is not None else fc * 0.90 + cur * 0.10
    elif local_hour < 12:
        # Late morning — still lean on forecast
        base = fc * 0.75 + cur * 0.15 + (noaa * 0.10 if noaa is not None else 0) if noaa is not None else fc * 0.80 + cur * 0.20
    elif local_hour < 14:
        # Early afternoon — more balanced
        base = fc * 0.60 + cur * 0.20 + noaa * 0.20 if noaa is not None else fc * 0.75 + cur * 0.25
    else:
        # Late afternoon — obs matters most
        base = fc * 0.45 + cur * 0.25 + noaa * 0.30 if noaa is not None else fc * 0.60 + cur * 0.40
    if abs(base - fc) > 3.0:
        base = fc - 3.0 if base < fc else fc + 3.0
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
        # Handle open-ended brackets properly
        if lo is None and hi is not None:
            mid = hi - 1.0  # "X or below"
        elif hi is None and lo is not None:
            mid = lo + 1.0  # "X or above"
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
    today_day = str(datetime.now().day)
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
    today_date = datetime.now().strftime('%Y-%m-%d')
    today_upper = datetime.now().strftime('%y%b%d').upper()
    today_upper2 = datetime.now().strftime('%d%b%y').upper()
    today_upper3 = datetime.now().strftime('%d%b%Y').upper()
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
    today = datetime.now().strftime('%Y-%m-%d')
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
    st.markdown('🟡 SKIP (uncertain) — NWS vs Ensemble >3F')
    st.markdown('🔵 Ensemble HIGH confidence')
    st.markdown('---')
    st.markdown('**V4.23 Changes**')
    st.markdown('- Uncertainty threshold raised to 5.0F (6.0F for desert cities)')
    st.markdown('- Green BET signals now show on normal morning divergence')
    st.markdown('- Moderate divergence info raised to 2.5F')
    st.markdown('- Supabase fallback hardcoded')

# ── Main App ──────────────────────────────────────────────────────────────────
saved_ladders = load_json(SAVE_FILE)
today_str = datetime.now().strftime('%Y-%m-%d')
last_sync_data = load_json(LAST_SYNC_FILE)

# ── Auto-Settlement (runs silently on load) ───────────────────────────────────
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
        st.success('Morning sync complete - ' + str(n) + '/' + str(len(SERIES)) + ' city ladders loaded from Kalshi')
        if results.get('failed'):
            st.warning('Could not fetch: ' + ', '.join(results['failed']) + ' - using saved ladders')
else:
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.caption('Ladders auto-synced from Kalshi today (' + today_str + ') - ' +
                   str(len(last_sync_data.get('synced', []))) + ' cities loaded')
    with col_btn:
        if st.button('Refresh All'):
            saved_ladders, results = sync_all_ladders(saved_ladders, force=True)
            st.success('Re-synced ' + str(len(results.get('synced', []))) + '/' + str(len(SERIES)) + ' city ladders')
            if results.get('failed'):
                st.warning('Could not fetch: ' + ', '.join(results['failed']))
            st.rerun()

# ── City Selection ────────────────────────────────────────────────────────────
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

st.caption('Settlement: ' + station + ' - ' + SETTLEMENT_LOCATION[city] + ' - Series: ' + series)
st.caption('Local time: ' + str(local_hour) + ':00 ' + tz_name)
if city in FORECAST_HEAVY_CITIES and local_hour < 16:
    st.caption('Forecast-heavy mode active (Texas/OKC heat lag correction)')

# ── Bias Correction (from Supabase) ──────────────────────────────────────────
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

# ── Kalshi Prices ─────────────────────────────────────────────────────────────
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
    st.success('Live prices loaded - ' + str(len(kalshi_markets)) + ' brackets (fetched ' + age_str + ')')
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

# ── Live Weather ──────────────────────────────────────────────────────────────
st.subheader('Live Weather')
with st.spinner('Fetching weather data...'):
    nws_forecast, nws_fc_url = fetch_nws_forecast(lat, lon)
    noaa_station, noaa_obs = fetch_nws_current(lat, lon, station)
    obs_high_raw, obs_high_url = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)

# ── Sanity Checks ─────────────────────────────────────────────────────────────
sanity_warnings = []
obs_high_today = obs_high_raw
obs_high_suspect = False

if obs_high_raw is not None:
    if noaa_obs is not None and obs_high_raw > noaa_obs + 15.0:
        obs_high_today = None
        obs_high_suspect = True
        sanity_warnings.append(
            f'Obs high ({obs_high_raw}F) is {round(obs_high_raw - noaa_obs, 1)}F above current temp '
            f'({round(noaa_obs, 1)}F) - likely wrong-day data. Discarded.')
    elif nws_forecast is not None and obs_high_raw > nws_forecast + 12.0:
        obs_high_today = None
        obs_high_suspect = True
        sanity_warnings.append(
            f'Obs high ({obs_high_raw}F) is {round(obs_high_raw - nws_forecast, 1)}F above NWS forecast '
            f'({nws_forecast}F) - implausible. Discarded.')

nws_stale = False
if nws_forecast is not None and noaa_obs is not None:
    if noaa_obs > nws_forecast + 5.0:
        nws_stale = True
        sanity_warnings.append(
            f'NWS forecast ({nws_forecast}F) is {round(noaa_obs - nws_forecast, 1)}F below current temp '
            f'({round(noaa_obs, 1)}F) - forecast may be stale.')

ensemble_suspect = False
if ensemble_mean is not None and nws_forecast is not None:
    if abs(ensemble_mean - nws_forecast) > 10.0:
        ensemble_suspect = True
        sanity_warnings.append(
            f'GFS ensemble ({ensemble_mean}F) differs from NWS by '
            f'{round(abs(ensemble_mean - nws_forecast), 1)}F - discarded.')
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
        st.caption('Primary - settlement source' + (' (stale?)' if nws_stale else ''))
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
        st.caption('[NWS table](' + obs_url + ')')
    elif obs_high_suspect:
        st.metric('Obs High Today', str(obs_high_raw)+'F')
        st.caption('Discarded - failed sanity check')
    else:
        st.metric('Obs High Today', 'Unavailable')
        st.caption('[NWS table](' + obs_url + ')')
with col4:
    if ensemble_mean is not None:
        n_members = len(ensemble_members) if ensemble_members else 0
        st.metric('GFS Ensemble', str(ensemble_mean)+' F', delta=str(n_members)+' members')
        st.caption('Weight: 45-50%')
    elif ensemble_suspect:
        st.metric('GFS Ensemble', 'Discarded')
        st.caption('Failed sanity check')
    else:
        st.metric('GFS Ensemble', 'Unavailable')

for w in sanity_warnings:
    st.error(w)

if nws_forecast is None:
    st.error('NWS forecast unavailable - cannot run model.')
elif high_uncertainty and source_gap is not None:
    st.warning(f'HIGH UNCERTAINTY: NWS ({nws_forecast}F) vs GFS ({ensemble_mean}F) gap = {round(source_gap, 1)}F. Green signals suppressed.')
elif source_gap is not None and source_gap > 2.5:
    st.info(f'Source gap: NWS vs Ensemble = {round(source_gap, 1)}F - moderate divergence.')

if obs_high_today is not None:
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and obs_high_today > hi + 0.4:
            st.warning('BUST: ' + label + ' eliminated - obs high ' + str(obs_high_today) + 'F exceeds ' + str(hi) + 'F')

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
    st.info('Using manual overrides - set back to 0.0 to use auto values')

forecast = override_fc if override_fc > 0 else nws_forecast
current = override_cur if override_cur > 0 else noaa_obs
noaa_final = override_noaa if override_noaa > 0 else noaa_obs
obs_high_final = override_obs_high if override_obs_high > 0 else obs_high_today

# ── Model Output ──────────────────────────────────────────────────────────────
if forecast is not None and current is not None:
    consensus_raw = compute_consensus(forecast, current, noaa_final, city, obs_high=obs_high_final)
    consensus = round(consensus_raw + bias_correction, 1)
    sigma_rows, sigma = bracket_probs(consensus, ladder_text, city, obs_high=obs_high_final, forecast=forecast)
    call = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)

    # ── Auto-save prediction to Supabase ─────────────────────────────────────
    save_ok = sb_upsert_prediction(
        city=city,
        consensus=consensus,
        forecast=forecast,
        ensemble_mean=ensemble_mean,
        source_gap=source_gap,
        high_uncertainty=high_uncertainty,
        obs_high=obs_high_final,
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
        st.metric('Sigma', str(round(sigma, 2))+' F')
    with c4:
        if obs_high_final is not None:
            st.metric('Obs Floor', str(obs_high_final)+' F',
                      delta='controlling' if obs_high_final >= consensus-0.1 else 'not binding')
    with c5:
        if bias_correction != 0.0:
            st.metric('Bias Adj', ('+' if bias_correction > 0 else '')+str(bias_correction)+' F',
                      delta='from '+str(bias_n)+' days')

    if ensemble_mean is not None:
        st.caption('GFS ensemble: '+str(ensemble_mean)+'F | '+str(len(ensemble_members))+
                   ' members | weight 45-50%')
    if high_uncertainty:
        st.caption('High uncertainty mode - green signals suppressed')

    import pandas as pd
    df_rows = []
    best_bet = None
    best_edge = -999

    for label, sigma_prob in sigma_rows:
        ens_prob = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            if labels_match(lbl, label):
                ens_prob = ensemble_bracket_prob(ensemble_members, lo, hi)
                break
        final_prob = blend_probs(sigma_prob, ens_prob, ensemble_members)
        fair = round(final_prob * 100)
        yes_ask = no_ask = None
        if kalshi_markets:
            # Numeric key matching — immune to string formatting differences
            match = next((m for m in kalshi_markets if labels_match(m[0], label)), None)
            if match:
                yes_ask, no_ask = match[1], match[2]
        e = edge_cents(final_prob, yes_ask)
        signal_icon, signal_text = edge_signal(e, high_uncertainty=high_uncertainty)
        kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0
        ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''
        busted = False
        if obs_high_final is not None:
            for lbl, lo, hi in parse_ladder(ladder_text):
                if labels_match(lbl, label) and hi is not None and obs_high_final > hi + 0.4:
                    busted = True
        edge_str = ('+'+str(e)+'c') if e and e > 0 else (str(e)+'c' if e is not None else 'none')
        df_rows.append({
            'Signal': signal_icon + ' ' + signal_text,
            'Bracket': label + (' BUSTED' if busted else ''),
            'Model %': str(round(final_prob*100, 1))+'%',
            'Fair': str(fair)+'c',
            'YES ask': str(yes_ask)+'c' if yes_ask is not None else 'none',
            'NO ask': str(no_ask)+'c' if no_ask is not None else 'none',
            'Edge': edge_str,
            'Kelly Bet': ('$'+str(kelly)) if kelly > 0 else '-',
            'Ensemble': ens_conf,
        })
        if e is not None and e > best_edge and not busted:
            best_edge = e
            best_bet = {'label': label, 'edge': e, 'kelly': kelly,
                        'signal': signal_icon, 'uncertain': high_uncertainty}

    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    if best_bet and best_bet['edge'] >= MIN_EDGE and not best_bet['uncertain']:
        st.success('Best Bet: **' + best_bet['label'] + '** | Edge: +' +
                   str(best_bet['edge']) + 'c | Kelly: $' + str(best_bet['kelly']))
    elif best_bet and best_bet['edge'] >= MIN_EDGE and best_bet['uncertain']:
        st.warning('Best edge: **' + best_bet['label'] + '** (+' + str(best_bet['edge']) +
                   'c) but HIGH UNCERTAINTY - consider skipping.')
    elif best_bet:
        st.warning('No bracket meets the ' + str(MIN_EDGE) + 'c minimum. Best: ' +
                   best_bet['label'] + ' (+' + str(best_bet['edge']) + 'c)')

    parsed = parse_ladder(ladder_text)
    top_b = next((b for b in parsed if b[2] is None), None)
    bot_b = next((b for b in parsed if b[1] is None), None)
    if (top_b and consensus > top_b[1]+5) or (bot_b and bot_b[2] is not None and consensus < bot_b[2]-5):
        st.warning('Ladder does not cover consensus of '+str(round(consensus, 1))+'F - update brackets.')

else:
    if forecast is None:
        st.error('NWS forecast unavailable. Use manual override or try refreshing.')
    else:
        st.error('Current temperature unavailable - cannot compute consensus.')

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
        st.info('No settled history yet for ' + city + '. Predictions auto-save daily and settle the next morning.')

    if pending:
        st.caption(str(len(pending)) + ' prediction(s) pending settlement: ' +
                   ', '.join(r['date'] for r in pending))
