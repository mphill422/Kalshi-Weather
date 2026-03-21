# Kalshi High Temperature Model - V4.14

# New in V4.13:
# - GFS 31-member ensemble forecasts from Open-Meteo
#   Counts how many of 31 model runs agree on temperature outcome
#   Ensemble probability blended with sigma model for final signal
# - Kelly Criterion position sizing (15% fractional Kelly)
#   kelly = (win_prob * odds - lose_prob) / odds
#   Capped at 5% of bankroll and $100 per trade
# - 8-cent minimum edge threshold
#   Green = strong edge (bet), Yellow = weak (skip), Red = no edge (avoid)
# - Bankroll input in sidebar powers all Kelly recommendations
# - Ensemble confidence label per bracket
# - All V4.12 features retained

import math, re, json, time, requests
import streamlit as st
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import pytz

st.set_page_config(page_title='Kalshi High Temp V4.14', layout='wide')
st.title('Kalshi High Temperature Model - V4.14')

SAVE_FILE = Path('saved_ladders.json')
HISTORY_FILE = Path('settlement_history.json')
LAST_SYNC_FILE = Path('last_sync.json')
PRICE_CACHE_FILE = Path('price_cache.json')
PRICE_CACHE_MINUTES = 10
MIN_EDGE = 8

HEADERS = {'User-Agent': 'kalshi-temp-model/4.14', 'Accept': 'application/geo+json, application/json, text/html'}

CITY_TZ = {
    'Phoenix': 'America/Phoenix',
    'Las Vegas': 'America/Los_Angeles',
    'Los Angeles': 'America/Los_Angeles',
    'Dallas': 'America/Chicago',
    'Austin': 'America/Chicago',
    'Houston': 'America/Chicago',
    'Atlanta': 'America/New_York',
    'Miami': 'America/New_York',
    'New York': 'America/New_York',
    'San Antonio': 'America/Chicago',
    'New Orleans': 'America/Chicago',
    'Philadelphia': 'America/New_York',
    'Boston': 'America/New_York',
    'Denver': 'America/Denver',
    'Oklahoma City': 'America/Chicago',
    'Minneapolis': 'America/Chicago',
    'Washington DC': 'America/New_York',
}

SERIES = {
    'Phoenix': 'KXHIGHTPHX', 'Las Vegas': 'KXHIGHTLV',
    'Los Angeles': 'KXHIGHLAX', 'Dallas': 'KXHIGHTDAL',
    'Austin': 'KXHIGHAUS', 'Houston': 'KXHIGHTHOU',
    'Atlanta': 'KXHIGHTATL', 'Miami': 'KXHIGHTMIA',
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
    'New York': 1.5, 'Philadelphia': 1.5, 'Washington DC': 1.6, 'Boston': 1.6,
    'Los Angeles': 1.4, 'Denver': 1.6, 'Miami': 1.7, 'Minneapolis': 1.7,
    'New Orleans': 1.8, 'Phoenix': 1.9, 'Las Vegas': 1.9, 'Atlanta': 2.0,
    'Dallas': 2.0, 'Austin': 2.0, 'Houston': 2.0, 'San Antonio': 2.0, 'Oklahoma City': 2.1,
}

DESERT_CITIES = {'Phoenix', 'Las Vegas'}
FORECAST_HEAVY_CITIES = {'Dallas', 'Austin', 'Houston', 'San Antonio', 'Oklahoma City'}

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

def edge_signal(e):
    if e is None:
        return '⚪', 'No price'
    if e >= MIN_EDGE:
        return '🟢', 'BET'
    if e >= 3:
        return '🟡', 'SKIP'
    return '🔴', 'AVOID'

# ── GFS 31-Member Ensemble ────────────────────────────────────────────────────

def fetch_gfs_ensemble(lat, lon):
    """
    Fetch GFS ensemble hourly temps from Open-Meteo ensemble API.
    Gets max temp for today across all ensemble members.
    Returns (list_of_member_max_temps, ensemble_mean) or (None, None).
    """
    url = 'https://ensemble-api.open-meteo.com/v1/ensemble'
    params = {
        'latitude': lat,
        'longitude': lon,
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
    except:
        return None, None

    today = datetime.now().strftime('%Y-%m-%d')
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])

    # Find indices for today only
    today_indices = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_indices:
        return None, None

    # Each key like 'temperature_2m_member01' is one ensemble member
    member_maxes = []
    for key, vals in hourly.items():
        if key == 'time':
            continue
        if 'temperature_2m' not in key:
            continue
        if not isinstance(vals, list):
            continue
        today_vals = [vals[i] for i in today_indices if i < len(vals) and vals[i] is not None]
        if today_vals:
            try:
                member_maxes.append(round(max(float(v) for v in today_vals), 1))
            except:
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
    """
    Blend sigma-model probability with ensemble probability.
    Weight ensemble more heavily when it has more members and agreement.
    """
    if ensemble_prob is None or members is None:
        return sigma_prob
    n = len(members)
    ensemble_weight = min(0.55, 0.30 + (n / 100.0))
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
        except:
            return {}
    return {}

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return r.json()
    except:
        return None

def safe_get_with_retry(url, params=None, retries=3, delay=2.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except:
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
    label = re.sub(r'(\d+)\s*-\s*(\d+)', lambda m: m.group(1)+'-'+m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').strip()

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
    s = BASE_SIGMA.get(city, 1.8)
    local_hour = get_local_hour(city)
    s *= 1.00 if local_hour < 11 else 0.92 if local_hour < 14 else 0.86 if local_hour < 16 else 0.80
    if city in DESERT_CITIES:
        s *= 0.90
    if obs_high is not None and forecast is not None:
        gap = abs(forecast - obs_high)
        if gap < 2:
            s *= 0.75
        elif gap < 4:
            s *= 0.85
    return max(1.10, min(2.40, s))

def late_day_floor(fc, obs, local_hour):
    gap = max(0.0, fc - obs)
    frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap

def compute_consensus(fc, cur, noaa, city, obs_high=None):
    local_hour = get_local_hour(city)
    is_fc_heavy = city in FORECAST_HEAVY_CITIES
    if is_fc_heavy and local_hour < 14:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.80 + obs_val * 0.20 if obs_val is not None else fc
    elif is_fc_heavy and local_hour < 16:
        obs_val = noaa if noaa is not None else cur
        base = fc * 0.65 + obs_val * 0.35 if obs_val is not None else fc
    else:
        base = fc * 0.55 + cur * 0.20 + noaa * 0.25 if noaa is not None else fc * 0.70 + cur * 0.30
    if abs(base - fc) > 2:
        base = fc - 1 if base < fc else fc + 1
    obs = noaa if noaa is not None else cur
    if obs is not None:
        consensus = max(base, late_day_floor(fc, obs, local_hour))
    else:
        consensus = base
    if consensus > fc + 0.6:
        consensus = fc + 0.6
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
        if lo is None or hi is None:
            continue
        if obs_high is not None and obs_high > hi + 0.4:
            continue
        dist = abs((lo + hi) / 2 - mu)
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
    except:
        return None, url
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'class': 'observations'})
    if not table:
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
        except:
            pass
        if len(cols) > 10:
            try:
                t6 = float(cols[10])
                if 0 < t6 < 130:
                    highs.append(t6)
            except:
                pass
    return (round(max(highs), 1), url) if highs else (None, url)

def fetch_open_meteo(lat, lon):
    data = safe_get('https://api.open-meteo.com/v1/forecast', {
        'latitude': lat, 'longitude': lon, 'daily': 'temperature_2m_max',
        'current': 'temperature_2m', 'temperature_unit': 'fahrenheit',
        'timezone': 'auto', 'forecast_days': 2,
    })
    if not data:
        return None, None
    today = datetime.now().strftime('%Y-%m-%d')
    times = data.get('daily', {}).get('time', [])
    idx = next((i for i, t in enumerate(times) if t.startswith(today)), 0)
    fc = data.get('daily', {}).get('temperature_2m_max', [None])[idx]
    cur = data.get('current', {}).get('temperature_2m')
    return (float(fc) if fc is not None else None, float(cur) if cur is not None else None)

def fetch_noaa(lat, lon, station_id):
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
        except:
            pass
    if cap is not None:
        try:
            n = int(float(cap))
            return str(n)+' or below', n-10000
        except:
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
            except:
                pass
    for f in ['no_ask_dollars', 'no_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                no_ask = round(float(v)*100)
                break
            except:
                pass
    if yes_ask is None:
        raw = m.get('yes_ask') or m.get('yes_bid')
        if raw is not None:
            try:
                yes_ask = int(raw)
            except:
                pass
    if no_ask is None:
        raw = m.get('no_ask') or m.get('no_bid')
        if raw is not None:
            try:
                no_ask = int(raw)
            except:
                pass
    return yes_ask, no_ask

def fetch_kalshi_brackets(series, retries=3):
    url = 'https://api.elections.kalshi.com/trade-api/v2/markets'
    event_ticker = get_event_ticker(series)
    today_date = datetime.now().strftime('%Y-%m-%d')
    today_upper = datetime.now().strftime('%y%b%d').upper()
    today_upper2 = datetime.now().strftime('%d%b%y').upper()
    data = safe_get_with_retry(url, {'event_ticker': event_ticker, 'limit': 30}, retries=retries, delay=2.0)
    if not data or not data.get('markets'):
        data = safe_get_with_retry(url, {'series_ticker': series, 'status': 'open', 'limit': 30}, retries=retries, delay=2.0)
    if not data or not data.get('markets'):
        return None
    all_markets = data['markets']
    markets = [m for m in all_markets if
               today_upper in (m.get('ticker') or '').upper() or
               today_upper2 in (m.get('ticker') or '').upper() or
               today_upper2 in (m.get('event_ticker') or '').upper()]
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
    st.markdown('🟢 Edge ≥8c — **BET**')
    st.markdown('🟡 Edge 3-7c — **SKIP**')
    st.markdown('🔴 Edge <3c — **AVOID**')
    st.markdown('🔵 Ensemble HIGH confidence')
    st.markdown('---')
    st.markdown('**Ensemble**')
    st.markdown('GFS 31-member forecast blended with sigma model.')

# ── Main App ──────────────────────────────────────────────────────────────────

saved_ladders = load_json(SAVE_FILE)
history = load_json(HISTORY_FILE)
if not isinstance(history, list):
    history = []

today_str = datetime.now().strftime('%Y-%m-%d')
last_sync_data = load_json(LAST_SYNC_FILE)

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
        st.caption('Ladders auto-synced from Kalshi today (' + today_str + ') - ' + str(len(last_sync_data.get('synced', []))) + ' cities loaded')
    with col_btn:
        if st.button('Refresh All'):
            saved_ladders, results = sync_all_ladders(saved_ladders, force=True)
            st.success('Re-synced ' + str(len(results.get('synced', []))) + '/' + str(len(SERIES)) + ' city ladders')
            if results.get('failed'):
                st.warning('Could not fetch: ' + ', '.join(results['failed']))
            st.rerun()

city = st.selectbox('City', list(CITIES.keys()), index=list(CITIES.keys()).index('New York'))
lat, lon = CITIES[city]['lat'], CITIES[city]['lon']
station, series = STATIONS[city], SERIES[city]
obs_icao = OBHISTORY_STATIONS.get(city, 'KNYC')
obs_url = 'https://forecast.weather.gov/data/obhistory/' + obs_icao + '.html'
local_hour = get_local_hour(city)
tz_name = CITY_TZ.get(city, 'America/New_York')

st.caption('Settlement: ' + station + ' - ' + SETTLEMENT_LOCATION[city] + ' - Series: ' + series)
st.caption('Local time: ' + str(local_hour) + ':00 ' + tz_name)
if city in FORECAST_HEAVY_CITIES and local_hour < 16:
    st.caption('Forecast-heavy mode active (Texas/OKC heat lag correction)')

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
    st.success('Live prices loaded - ' + str(len(kalshi_markets)) + ' brackets (fetched ' + age_str + ')')
    for m in kalshi_markets:
        st.caption('  ' + m[0] + ' | YES: ' + (str(m[1])+'c' if m[1] else 'no price') + ' | NO: ' + (str(m[2])+'c' if m[2] else 'no price'))
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

# ── Live Weather + Ensemble ───────────────────────────────────────────────────

st.subheader('Live Weather')
with st.spinner('Fetching weather, ensemble, and observed high...'):
    forecast_auto, current_auto = fetch_open_meteo(lat, lon)
    noaa_station, noaa_obs = fetch_noaa(lat, lon, station)
    obs_high_today, obs_high_url = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric('Forecast High', str(round(forecast_auto, 1))+' F' if forecast_auto else 'unavailable')
with col2:
    st.metric('Current Temp', str(round(current_auto, 1))+' F' if current_auto else 'unavailable')
with col3:
    if noaa_obs is not None:
        st.metric('NOAA Obs', str(round(noaa_obs, 1))+' F')
        st.caption('Station: '+noaa_station)
    else:
        st.metric('NOAA Obs', 'Unavailable')
with col4:
    if obs_high_today is not None:
        st.metric('Obs High So Far', str(obs_high_today)+' F', delta='floor active')
        st.caption('[NWS table]('+obs_url+')')
    else:
        st.metric('Obs High So Far', 'Unavailable')
        st.caption('[NWS table]('+obs_url+')')
with col5:
    if ensemble_mean is not None:
        n_members = len(ensemble_members) if ensemble_members else 0
        st.metric('GFS Ensemble', str(ensemble_mean)+' F', delta=str(n_members)+' members')
        st.caption('Blended into model')
    else:
        st.metric('GFS Ensemble', 'Unavailable')
        st.caption('Sigma-only mode')

if obs_high_today is not None:
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and obs_high_today > hi + 0.4:
            st.warning('BUST: '+label+' eliminated - obs high '+str(obs_high_today)+'F exceeds '+str(hi)+'F')

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

forecast = override_fc if override_fc > 0 else forecast_auto
current = override_cur if override_cur > 0 else current_auto
noaa_final = override_noaa if override_noaa > 0 else noaa_obs
obs_high_final = override_obs_high if override_obs_high > 0 else obs_high_today

# ── Model Output ──────────────────────────────────────────────────────────────

if forecast is not None and current is not None:
    consensus = compute_consensus(forecast, current, noaa_final, city, obs_high=obs_high_final)
    sigma_rows, sigma = bracket_probs(consensus, ladder_text, city, obs_high=obs_high_final, forecast=forecast)
    call = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)

    st.subheader('Model Output')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Consensus High', str(round(consensus, 1))+' F')
    with c2:
        st.metric('2 Degree Call', call or 'none')
    with c3:
        st.metric('Sigma', str(round(sigma, 2))+' F')
    with c4:
        if obs_high_final is not None:
            st.metric('Obs Floor', str(obs_high_final)+' F',
                      delta='controlling' if obs_high_final >= consensus-0.1 else 'not binding')

    if ensemble_mean is not None:
        st.caption('GFS ensemble mean: '+str(ensemble_mean)+'F with '+str(len(ensemble_members))+' members — blended into probabilities below')

    st.caption('Local time: '+str(local_hour)+':00 '+tz_name+' - Late-day floor active')

    import pandas as pd

    df_rows = []
    best_bet = None
    best_edge = -999

    for label, sigma_prob in sigma_rows:
        # Get ensemble prob for this bracket
        ens_prob = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            if normalize_label(lbl) == normalize_label(label):
                ens_prob = ensemble_bracket_prob(ensemble_members, lo, hi)
                break

        # Blend sigma + ensemble
        final_prob = blend_probs(sigma_prob, ens_prob, ensemble_members)

        fair = round(final_prob * 100)
        yes_ask = no_ask = None
        if kalshi_markets:
            match = next((m for m in kalshi_markets if normalize_label(m[0]) == normalize_label(label)), None)
            if match:
                yes_ask, no_ask = match[1], match[2]

        e = edge_cents(final_prob, yes_ask)
        signal_icon, signal_text = edge_signal(e)

        kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0

        ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''

        busted = False
        if obs_high_final is not None:
            for lbl, lo, hi in parse_ladder(ladder_text):
                if normalize_label(lbl) == normalize_label(label) and hi is not None and obs_high_final > hi + 0.4:
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
            best_bet = {'label': label, 'edge': e, 'kelly': kelly, 'signal': signal_icon}

    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    # Best bet callout
    if best_bet and best_bet['edge'] >= MIN_EDGE:
        st.success('🎯 Best Bet: **' + best_bet['label'] + '** | Edge: +' + str(best_bet['edge']) + 'c | Kelly: $' + str(best_bet['kelly']))
    elif best_bet:
        st.warning('⚠️ No bracket meets the ' + str(MIN_EDGE) + 'c minimum edge threshold. Best available: ' + best_bet['label'] + ' (+' + str(best_bet['edge']) + 'c) — consider skipping this market today.')

    parsed = parse_ladder(ladder_text)
    top_b = next((b for b in parsed if b[2] is None), None)
    bot_b = next((b for b in parsed if b[1] is None), None)
    if (top_b and consensus > top_b[1]+5) or (bot_b and consensus < bot_b[2]-5):
        st.warning('Ladder does not cover consensus of '+str(round(consensus, 1))+' F - update brackets.')
else:
    st.error('Weather data unavailable.')

# ── Settlement Log ────────────────────────────────────────────────────────────

st.subheader('Log Actual High (after settlement)')
with st.form('log_form'):
    actual = st.number_input('Actual NWS High F', min_value=0.0, max_value=130.0, step=0.1)
    if st.form_submit_button('Log Settlement') and forecast is not None:
        entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'city': city,
            'actual': actual,
            'forecast': round(forecast, 1),
            'consensus': round(consensus, 1),
            'ensemble_mean': ensemble_mean,
            'obs_high': obs_high_final,
            'error': round(actual-consensus, 1),
        }
        history.append(entry)
        save_json(HISTORY_FILE, history[-300:])
        st.success('Logged - actual='+str(actual)+'F  consensus='+str(round(consensus, 1))+'F  error='+str(entry['error'])+'F')

if history:
    st.subheader('Settlement History')
    df_h = pd.DataFrame(history[-50:][::-1])
    wd = [h for h in history if h.get('consensus') and h.get('actual')]
    if wd:
        mae = sum(abs(h['actual']-h['consensus']) for h in wd) / len(wd)
        w1 = sum(1 for h in wd if abs(h['actual']-h['consensus']) <= 1) / len(wd)
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.metric('Records', len(history))
        with hc2:
            st.metric('Model MAE', str(round(mae, 2))+' F')
        with hc3:
            st.metric('Within 1 degree', str(round(w1*100, 0))+'%')
    st.dataframe(df_h, use_container_width=True, hide_index=True)
