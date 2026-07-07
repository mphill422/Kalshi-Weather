# Kalshi High Temperature Model - V5.27.1
# V5.27 BUILD GOALS — bet frequency reduction (research-validated):
#   After 43 live bets resulted in -$75.02 / -21.5% ROI / 27.9% win rate,
#   public Kalshi weather strategies (gopher-lab/kalshi-go and
#   Oalkhadra/prediction-market-trading) showed three patterns the V5.26.2
#   model was violating:
#     1. Brackets <30c entry have ~0% historical win rate.
#     2. 2-of-3 consensus (model + market top 2) yields 82% win rate.
#     3. Profitable strategies skip ~50% of days.
#     4. Multi-bracket spreading is a documented failure (52% win rate).
#
# V5.27 implements three gates. A signal must pass ALL THREE to surface:
#   GATE 1 (Consensus): Model top pick must equal market #1 OR strict #2 by yes-ask.
#   GATE 2 (Price floor): Entry >= 30c on side bought.
#   GATE 3 (One bet/day): Single highest Trust accuracy bet across all cities.
#
# V5.27.1 HOTFIX (2026-05-12):
#   - Tightened obs-vs-current threshold 15F -> 10F (both backend and display)
#     to catch May 11 KNYC sensor-spike pattern
#   - Fixed NO Signals table Model% showing YES probability instead of NO
#   - Routed Miami full_blend -> nws_only (14-day MAE: -2.04F hot bias, 0.73F worse)
#   - Version string bumps throughout user-facing text
#
# All V5.22 - V5.26.2 forecasting/data/scoring infrastructure preserved exactly.
# Only the bet-selection / surfacing layer is gated.

import math, re, json, time, requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
import pytz

from trust_score import SignalInputs, compute_trust_score, bracket_midpoint_from_label

st.set_page_config(page_title='MPH Weather Model', layout='wide', page_icon='🌡️')

def _check_app_password():
    try:
        correct_pw = st.secrets.get('app_password', None)
    except Exception:
        correct_pw = None
    if not correct_pw:
        return True
    if st.session_state.get('_app_authed') is True:
        return True
    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .mph-login-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 70vh; }
    .mph-login-title { font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.25rem; }
    .mph-login-sub { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }
    .mph-login-badge { display: inline-block; padding: 0.2rem 0.7rem; background: rgba(0, 255, 136, 0.12); border: 1px solid rgba(0, 255, 136, 0.4); border-radius: 999px; color: #00ff88; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; margin-left: 0.5rem; }
    </style>
    <div class="mph-login-wrap">
      <div class="mph-login-title">🌡️ MPH Weather Model <span class="mph-login-badge">V5.27.1</span></div>
      <div class="mph-login-sub">Private — enter access password to continue</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        pw = st.text_input('Access password', type='password', key='_app_pw_input',
                           label_visibility='collapsed', placeholder='Enter password')
        if pw:
            if pw == correct_pw:
                st.session_state['_app_authed'] = True
                st.rerun()
            else:
                st.error('Incorrect password.')
    st.stop()

_check_app_password()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp { background: #0a0e1a; }
.mph-hero { background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 24px 32px; margin-bottom: 20px; position: relative; overflow: hidden; }
.mph-hero::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00ff88, #00b4d8, #00ff88); }
.mph-hero-title { font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0 0 4px 0; font-family: 'Inter', sans-serif; }
.mph-hero-sub { font-size: 13px; color: #64748b; font-family: 'JetBrains Mono', monospace; margin: 0; }
.mph-version-badge { display: inline-block; background: #00ff8820; border: 1px solid #00ff8840; color: #00ff88; font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 20px; font-family: 'JetBrains Mono', monospace; margin-left: 10px; vertical-align: middle; }
.mph-live-dot { display: inline-block; width: 8px; height: 8px; background: #00ff88; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
.mph-stats-bar { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.mph-stat { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 120px; text-align: center; }
.mph-stat-value { font-size: 22px; font-weight: 700; color: #00ff88; font-family: 'JetBrains Mono', monospace; display: block; line-height: 1.2; }
.mph-stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; display: block; margin-top: 4px; }
.mph-stat-warn .mph-stat-value { color: #f59e0b; }
.mph-stat-alert .mph-stat-value { color: #ef4444; }
.mph-stat-neutral .mph-stat-value { color: #94a3b8; }
.mph-section-header { font-size: 13px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.2px; padding: 0 0 8px 0; border-bottom: 1px solid #1e3a5f; margin-bottom: 16px; font-family: 'Inter', sans-serif; }
.stMetric { background: #0d1b2a !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; padding: 12px !important; }
.stMetric label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
.stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'JetBrains Mono', monospace !important; font-size: 20px !important; }
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 8px !important; overflow: hidden !important; }
.stButton > button { background: #1e3a5f !important; color: #00ff88 !important; border: 1px solid #00ff8840 !important; border-radius: 6px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; font-weight: 600 !important; letter-spacing: 0.5px !important; padding: 6px 16px !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #00ff8820 !important; border-color: #00ff88 !important; }
.stSelectbox > div > div { background: #0d1b2a !important; border: 1px solid #1e3a5f !important; border-radius: 6px !important; color: #ffffff !important; }
.stSuccess { background: #00ff8810 !important; border: 1px solid #00ff8840 !important; border-radius: 6px !important; color: #00ff88 !important; }
.stWarning { background: #f59e0b10 !important; border: 1px solid #f59e0b40 !important; border-radius: 6px !important; }
.stInfo { background: #00b4d810 !important; border: 1px solid #00b4d840 !important; border-radius: 6px !important; }
.stError { background: #ef444410 !important; border: 1px solid #ef444440 !important; border-radius: 6px !important; }
[data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid #1e3a5f !important; }
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }
.streamlit-expanderHeader { background: #0d1b2a !important; border: 1px solid #1e3a5f !important; border-radius: 6px !important; color: #94a3b8 !important; font-size: 13px !important; }
.stNumberInput > div > div > input { background: #0d1b2a !important; border: 1px solid #1e3a5f !important; color: #ffffff !important; border-radius: 6px !important; font-family: 'JetBrains Mono', monospace !important; }
@media (max-width: 768px) {
    .mph-hero { padding: 16px 20px; }
    .mph-hero-title { font-size: 20px; }
    .mph-stats-bar { gap: 8px; }
    .mph-stat { padding: 10px 12px; min-width: 80px; }
    .mph-stat-value { font-size: 18px; }
    .mph-stat-label { font-size: 10px; }
}
</style>
""", unsafe_allow_html=True)

SAVE_FILE = Path('saved_ladders.json')
LAST_SYNC_FILE = Path('last_sync.json')
PRICE_CACHE_FILE = Path('price_cache.json')
PRICE_CACHE_MINUTES = 10

MIN_EDGE = 8

# V5.27 Three research-validated gating constants
PRICE_FLOOR_CENTS = 30
CONSENSUS_TOP_N = 2
ONE_CITY_PER_DAY = False

HEADERS = {'User-Agent': 'kalshi-temp-model/5.27.1', 'Accept': 'application/geo+json, application/json, text/html'}
try:
    WETHR_API_KEY = st.secrets['wethr']['api_key']
except Exception:
    try:
        WETHR_API_KEY = st.secrets['api_keys']['WETHR_API_KEY']
    except Exception:
        WETHR_API_KEY = ''
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}

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

SERIES = {
    'Phoenix': 'KXHIGHTPHX', 'Las Vegas': 'KXHIGHTLV',
    'Los Angeles': 'KXHIGHLAX', 'Dallas': 'KXHIGHTDAL',
    'Austin': 'KXHIGHAUS', 'Houston': 'KXHIGHTHOU',
    'Atlanta': 'KXHIGHTATL', 'Miami': 'KXHIGHMIA',
    'New York': 'KXHIGHNY', 'San Antonio': 'KXHIGHTSATX',
    'New Orleans': 'KXHIGHTNOLA', 'Philadelphia': 'KXHIGHPHIL',
    'Boston': 'KXHIGHTBOS', 'Denver': 'KXHIGHDEN',
    'Oklahoma City': 'KXHIGHTOKC', 'Minneapolis': 'KXHIGHTMIN',
    'Washington DC': 'KXHIGHTDC', 'Chicago': 'KXHIGHCHI',
}

STATIONS = {
    'Phoenix': 'CLIPHX', 'Las Vegas': 'CLILAS', 'Los Angeles': 'CLILAX',
    'Dallas': 'CLIDFW', 'Austin': 'CLIAUS', 'Houston': 'CLIHOU',
    'Atlanta': 'CLIATL', 'Miami': 'CLIMIA', 'New York': 'KNYC',
    'San Antonio': 'CLISAT', 'New Orleans': 'CLIMSY', 'Philadelphia': 'CLIPHL',
    'Boston': 'CLIBOS', 'Denver': 'CLIDEN', 'Oklahoma City': 'CLIOKC',
    'Minneapolis': 'CLIMSP', 'Washington DC': 'CLIDCA', 'Chicago': 'KMDW',
}

OBHISTORY_STATIONS = {
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

WETHR_STATIONS = {
    'Phoenix': 'KPHX', 'Las Vegas': 'KLAS', 'Los Angeles': 'KLAX',
    'Dallas': 'KDFW', 'Austin': 'KAUS', 'Houston': 'KHOU',
    'Atlanta': 'KATL', 'Miami': 'KMIA', 'New York': 'KNYC',
    'San Antonio': 'KSAT', 'New Orleans': 'KMSY', 'Philadelphia': 'KPHL',
    'Boston': 'KBOS', 'Denver': 'KDEN', 'Oklahoma City': 'KOKC',
    'Minneapolis': 'KMSP', 'Washington DC': 'KDCA', 'Chicago': 'KMDW',
}

WUNDERGROUND_URLS = {
    'Phoenix': 'https://www.wunderground.com/weather/KPHX',
    'Las Vegas': 'https://www.wunderground.com/weather/KLAS',
    'Los Angeles': 'https://www.wunderground.com/weather/KLAX',
    'Dallas': 'https://www.wunderground.com/weather/KDFW',
    'Austin': 'https://www.wunderground.com/weather/KAUS',
    'Houston': 'https://www.wunderground.com/weather/KHOU',
    'Atlanta': 'https://www.wunderground.com/weather/KATL',
    'Miami': 'https://www.wunderground.com/weather/KMIA',
    'New York': 'https://www.wunderground.com/weather/KNYC',
    'San Antonio': 'https://www.wunderground.com/weather/KSAT',
    'New Orleans': 'https://www.wunderground.com/weather/KMSY',
    'Philadelphia': 'https://www.wunderground.com/weather/KPHL',
    'Boston': 'https://www.wunderground.com/weather/KBOS',
    'Denver': 'https://www.wunderground.com/weather/KDEN',
    'Oklahoma City': 'https://www.wunderground.com/weather/KOKC',
    'Minneapolis': 'https://www.wunderground.com/weather/KMSP',
    'Washington DC': 'https://www.wunderground.com/weather/KDCA',
    'Chicago': 'https://www.wunderground.com/weather/KMDW',
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
    'Washington DC': 'Reagan National Airport', 'Chicago': 'Chicago Midway Airport',
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
    'Washington DC': {'lat': 38.8512, 'lon': -77.0402}, 'Chicago': {'lat': 41.7868, 'lon': -87.7522},
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
    'Chicago': '46 or below | 47-48 | 49-50 | 51-52 | 53-54 | 55 or above',
}

BASE_SIGMA = {
    'New York': 1.8, 'Philadelphia': 1.8, 'Washington DC': 1.9, 'Boston': 1.9,
    'Los Angeles': 1.7, 'Denver': 1.9, 'Miami': 2.0, 'Minneapolis': 2.1,
    'New Orleans': 2.1, 'Phoenix': 2.2, 'Las Vegas': 2.2, 'Atlanta': 2.3,
    'Dallas': 2.3, 'Austin': 2.3, 'Houston': 2.3, 'San Antonio': 2.3,
    'Oklahoma City': 2.5, 'Chicago': 2.1,
}

DESERT_CITIES = {'Phoenix', 'Las Vegas'}
FORECAST_HEAVY_CITIES = {'Dallas', 'Austin', 'Houston', 'San Antonio', 'Oklahoma City'}

GFS_CITY_WEIGHT = {
    'Houston': 0.18, 'Phoenix': 0.0, 'Las Vegas': 0.0, 'Los Angeles': 0.0,
    'Miami': 0.0, 'New Orleans': 0.0, 'Dallas': 0.0, 'Austin': 0.0,
    'San Antonio': 0.0, 'Oklahoma City': 0.0, 'Atlanta': 0.0, 'Denver': 0.0,
    'Minneapolis': 0.0, 'Chicago': 0.0, 'New York': 0.0, 'Philadelphia': 0.0,
    'Boston': 0.0, 'Washington DC': 0.0,
}

HIDDEN_CITIES = set()


SPRING_WIDE_THRESHOLD_CITIES = {'New York', 'Philadelphia', 'Boston', 'Washington DC', 'Los Angeles'}
NORTHEAST_CITIES = {'New York', 'Philadelphia', 'Boston', 'Washington DC'}
DESERT_CITIES = {'Phoenix', 'Las Vegas'}
REGIONAL_PRIOR_BIAS = {'Chicago': 'Minneapolis'}

# Miami: 14-day MAE showed full_blend ran -2.04F hot bias and 0.73F worse
# than nws_only. Routed to nws_only 2026-05-12 (V5.27.1).
CITY_PREDICTION_MODE = {
    'New York': 'full_blend', 'Houston': 'full_blend', 'Dallas': 'full_blend',
    'Los Angeles': 'full_blend', 'Phoenix': 'full_blend',
    'Las Vegas': 'full_blend', 'Boston': 'full_blend', 'Philadelphia': 'full_blend',
    'Miami': 'nws_only', 'New Orleans': 'nws_only', 'Washington DC': 'nws_only',
    'Atlanta': 'nws_only', 'Oklahoma City': 'nws_only', 'Chicago': 'nws_only',
    'Denver': 'nws_only', 'Austin': 'nws_only', 'Minneapolis': 'nws_only',
    'San Antonio': 'nws_only',
}

CITY_WARM_OFFSET = {
    'Phoenix': 1.0,
    'Las Vegas': -1.0,
}

NWS_BIAS_BOOST_CITIES = {
    'Washington DC', 'Oklahoma City', 'Denver', 'Austin', 'San Antonio',
}
NWS_BIAS_BOOST_MULTIPLIER = 1.2

OBS_HIGH_TRUST_HOUR = 13
OBS_HIGH_MAX_OVERSHOOT = 10.0

try:
    _SB_URL = st.secrets['supabase']['url']
    _SB_KEY = st.secrets['supabase']['key']
except Exception:
    _SB_URL = 'https://oirnfhhuyjuotkrlymxd.supabase.co'
    _SB_KEY = ''

def get_sb_headers():
    return {'apikey': _SB_KEY, 'Authorization': 'Bearer ' + _SB_KEY,
            'Content-Type': 'application/json', 'Prefer': 'return=representation'}

def sb_url(table):
    return _SB_URL + '/rest/v1/' + table

def sb_insert(row):
    try:
        r = requests.post(sb_url('settlements'), headers=get_sb_headers(), json=row, timeout=10)
        return r.status_code in (200, 201)
    except Exception: return False

def _bust_bets_cache():
    try: _sb_fetch_bets_cached.clear()
    except Exception: pass

def _bust_settlements_cache():
    try:
        _sb_fetch_all_cached.clear()
        _sb_fetch_city_cached.clear()
    except Exception: pass

@st.cache_data(ttl=60)
def _sb_fetch_all_cached():
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'order': 'date.asc', 'limit': '1000'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_fetch_all():
    return _sb_fetch_all_cached()

@st.cache_data(ttl=60)
def _sb_fetch_city_cached(city):
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'city': 'eq.' + city, 'order': 'date.asc', 'limit': '200'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_fetch_city(city):
    return _sb_fetch_city_cached(city)

def sb_fetch_unsettled():
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'actual': 'is.null', 'order': 'date.asc'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_update_actual(row_id, actual, error):
    try:
        r = requests.patch(sb_url('settlements') + '?id=eq.' + str(row_id),
                           headers=get_sb_headers(),
                           json={'actual': actual, 'error': round(error, 2)}, timeout=10)
        return r.status_code in (200, 204)
    except Exception: return False

def sb_fetch_today(city):
    today = get_eastern_date()
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'date': 'eq.' + today, 'city': 'eq.' + city}, timeout=10)
        rows = r.json() if r.status_code == 200 else []
        return rows[0] if rows else None
    except Exception: return None

def sb_upsert_prediction(city, consensus, forecast, ensemble_mean, source_gap,
                          high_uncertainty, obs_high, bias_correction):
    today = get_eastern_date()
    existing = sb_fetch_today(city)
    row = {
        'date': today, 'city': city, 'consensus': round(consensus, 2),
        'forecast': round(forecast, 2) if forecast else None,
        'ensemble_mean': round(ensemble_mean, 2) if ensemble_mean else None,
        'source_gap': round(source_gap, 2) if source_gap else None,
        'high_uncertainty': bool(high_uncertainty),
        'obs_high': round(obs_high, 2) if obs_high else None,
        'bias_correction': round(bias_correction, 2),
        'actual': None, 'error': None,
    }
    if existing:
        try:
            r = requests.patch(sb_url('settlements') + '?id=eq.' + str(existing['id']),
                               headers=get_sb_headers(),
                               json={k: v for k, v in row.items() if k not in ('date', 'city')}, timeout=10)
            if r.status_code not in (200, 204):
                st.error(f'DB PATCH failed: {r.status_code} — {r.text[:200]}')
            return r.status_code in (200, 204)
        except Exception as e:
            st.error(f'DB PATCH exception: {str(e)[:200]}')
            return False
    else:
        try:
            r = requests.post(sb_url('settlements'), headers=get_sb_headers(), json=row, timeout=10)
            if r.status_code not in (200, 201):
                st.error(f'DB INSERT failed: {r.status_code} — {r.text[:200]}')
            return r.status_code in (200, 201)
        except Exception as e:
            st.error(f'DB INSERT exception: {str(e)[:200]}')
            return False

_CLI_CACHE = {}

def fetch_cli_max_temp(city, target_date_str):
    station = CLI_STATIONS.get(city)
    if not station: return None
    year = target_date_str[:4]
    cache_key = station + '_' + year
    if cache_key not in _CLI_CACHE:
        try:
            url = f'https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={year}'
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            lookup = {}
            for entry in data.get('results', []):
                valid = entry.get('valid', '')
                high = entry.get('high')
                if valid and high is not None:
                    try: lookup[valid] = float(high)
                    except Exception: pass
            _CLI_CACHE[cache_key] = lookup
        except Exception: return None
    return _CLI_CACHE.get(cache_key, {}).get(target_date_str)

def fetch_obs_high_for_date(icao, target_date_str):
    url = 'https://forecast.weather.gov/data/obhistory/' + icao + '.html'
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
    except Exception: return None
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    table = max(tables, key=lambda t: len(t.find_all('tr')), default=None) if tables else None
    if not table: return None
    target_day = str(datetime.strptime(target_date_str, '%Y-%m-%d').day)
    highs = []
    for row in table.find_all('tr'):
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cols or len(cols) < 9 or cols[0] != target_day: continue
        try:
            t = float(cols[8])
            if 0 < t < 130: highs.append(t)
        except Exception: pass
    return round(max(highs), 1) if highs else None

def run_auto_settlement():
    unsettled = sb_fetch_unsettled()
    if not unsettled: return 0, []
    settled = []
    today = get_eastern_date()
    for row in unsettled:
        row_date = row.get('date', '')
        if row_date >= today: continue
        city = row.get('city')
        icao = OBHISTORY_STATIONS.get(city)
        if not icao: continue
        actual = fetch_cli_max_temp(city, row_date)
        if actual is None:
            actual = fetch_obs_high_for_date(icao, row_date)
        if actual is None: continue
        consensus = row.get('consensus')
        error = round(actual - consensus, 2) if consensus is not None else None
        ok = sb_update_actual(row['id'], actual, error)
        if ok:
            settled.append({'city': city, 'date': row_date, 'actual': actual, 'error': error})
    return len(settled), settled

def compute_bias_correction_db(city, n_recent=10):
    import statistics
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        prior_city = REGIONAL_PRIOR_BIAS.get(city)
        if prior_city:
            prior_rows = sb_fetch_city(prior_city)
            prior_complete = [r for r in prior_rows if r.get('actual') is not None and r.get('consensus') is not None]
            if len(prior_complete) >= 3:
                recent = prior_complete[-n_recent:]
                errors = [r['actual'] - r['consensus'] for r in recent]
                med_error = statistics.median(errors)
                return round(max(-3.0, min(3.0, med_error)), 2), len(complete)
        return 0.0, len(complete)
    recent = complete[-n_recent:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    med_error = statistics.median(errors)
    if city in NWS_BIAS_BOOST_CITIES:
        med_error = med_error * NWS_BIAS_BOOST_MULTIPLIER
    return round(max(-3.0, min(3.0, med_error)), 2), len(recent)

def get_city_mae_and_color(city, n_recent=14):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('error') is not None]
    if len(complete) < 3:
        return None, 'green'
    recent = complete[-n_recent:]
    errors = [abs(r['error']) for r in recent]
    mae = round(sum(errors) / len(errors), 2)
    if mae < 2.5: color = 'green'
    elif mae < 4.0: color = 'yellow'
    else: color = 'red'
    return mae, color

def get_uncertainty_threshold(city):
    month = datetime.now().month
    if city == 'Los Angeles' and 3 <= month <= 5: return 7.0
    if city in SPRING_WIDE_THRESHOLD_CITIES and 3 <= month <= 5: return 6.5
    if city in DESERT_CITIES: return 6.0
    return 5.0


def compute_quality_score(city):
    import statistics
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    reasons = []
    n_complete = len(complete)
    if n_complete < 5:
        reasons.append(f'Only {n_complete} settlement(s) on record — insufficient for confident quality assessment')
        return 35, '🟡', reasons
    recent = complete[-7:]
    errors = [r['actual'] - r['consensus'] for r in recent]
    abs_errors = [abs(e) for e in errors]
    score = 100
    err_stdev = statistics.stdev(errors) if len(errors) >= 2 else 0
    if err_stdev > 3.0:
        score -= 35
        reasons.append(f'Errors very volatile (stdev {err_stdev:.1f}°F) — model unpredictable lately')
    elif err_stdev > 2.0:
        score -= 20
        reasons.append(f'Errors moderately volatile (stdev {err_stdev:.1f}°F)')
    elif err_stdev > 1.0:
        score -= 8
        reasons.append(f'Errors mildly volatile (stdev {err_stdev:.1f}°F) — normal range')
    else:
        reasons.append(f'Errors tightly clustered (stdev {err_stdev:.1f}°F) — strong calibration')
    err_range = max(errors) - min(errors)
    if err_range > 6.0:
        score -= 25
        reasons.append(f'Last 7 errors span {err_range:.1f}°F — wild swings')
    elif err_range > 4.0:
        score -= 12
        reasons.append(f'Last 7 errors span {err_range:.1f}°F — meaningful swings')
    if len(complete) >= 3:
        last_3 = complete[-3:]
        last_3_errors = [r['actual'] - r['consensus'] for r in last_3]
        if all(e > 1.5 for e in last_3_errors):
            score -= 25
            reasons.append(f'⚠️ REGIME SHIFT — last 3 errors all warm: {", ".join("+"+f"{e:.1f}" for e in last_3_errors)}°F. Model running cold.')
        elif all(e < -1.5 for e in last_3_errors):
            score -= 25
            reasons.append(f'⚠️ REGIME SHIFT — last 3 errors all cold: {", ".join(f"{e:.1f}" for e in last_3_errors)}°F. Model running warm.')
    if errors:
        last_err = errors[-1]
        if abs(last_err) > 3.0:
            score -= 15
            reasons.append(f'Yesterday errored {("+ " if last_err > 0 else "")}{last_err:.1f}°F — large miss reduces today confidence')
    if n_complete >= 14:
        reasons.append(f'{n_complete} historical settlements — ample sample size')
    elif n_complete < 7:
        score -= 8
        reasons.append(f'Only {n_complete} historical settlements — sample size limits confidence')
    score = max(0, min(100, score))
    if score >= 75: tier = '🟢'
    elif score >= 50: tier = '🟡'
    else: tier = '🔴'
    return score, tier, reasons


def compute_regime_indicator():
    yesterday_str = (datetime.now(pytz.timezone('America/New_York')) - timedelta(days=1)).strftime('%Y-%m-%d')
    all_rows = sb_fetch_all() or []
    yesterday_rows = [r for r in all_rows if r.get('date') == yesterday_str
                      and r.get('actual') is not None and r.get('consensus') is not None]
    if not yesterday_rows:
        return 0, 0, 0, 'No settled data from yesterday yet'
    warm_misses = sum(1 for r in yesterday_rows if (r['actual'] - r['consensus']) > 1.5)
    cold_misses = sum(1 for r in yesterday_rows if (r['actual'] - r['consensus']) < -1.5)
    total = len(yesterday_rows)
    if warm_misses >= 4 and warm_misses / total >= 0.4:
        msg = f'⚠️ WARM REGIME — {warm_misses}/{total} cities settled warmer than model yesterday. Today predictions may run cold.'
    elif cold_misses >= 4 and cold_misses / total >= 0.4:
        msg = f'⚠️ COLD REGIME — {cold_misses}/{total} cities settled cooler than model yesterday. Today predictions may run warm.'
    else:
        msg = f'Stable regime — yesterday {warm_misses} warm misses, {cold_misses} cold misses across {total} cities.'
    return warm_misses, cold_misses, total, msg


def compute_calibration_buckets():
    bets = sb_fetch_bets() or []
    settled = [b for b in bets if b.get('result') in ('Won', 'Lost')]
    buckets = {
        '90-100%': {'wins': 0, 'losses': 0, 'bets': []},
        '70-89%':  {'wins': 0, 'losses': 0, 'bets': []},
        '50-69%':  {'wins': 0, 'losses': 0, 'bets': []},
        '30-49%':  {'wins': 0, 'losses': 0, 'bets': []},
        '0-29%':   {'wins': 0, 'losses': 0, 'bets': []},
    }
    for bet in settled:
        price = bet.get('price')
        if price is None: continue
        try:
            p = float(price)
        except Exception:
            continue
        if bet.get('direction') == 'YES':
            implied = p / 100.0
        else:
            implied = 1.0 - (p / 100.0)
        implied_pct = implied * 100
        if implied_pct >= 90: bucket = '90-100%'
        elif implied_pct >= 70: bucket = '70-89%'
        elif implied_pct >= 50: bucket = '50-69%'
        elif implied_pct >= 30: bucket = '30-49%'
        else: bucket = '0-29%'
        if bet['result'] == 'Won':
            buckets[bucket]['wins'] += 1
        else:
            buckets[bucket]['losses'] += 1
        buckets[bucket]['bets'].append(bet)
    return buckets

@st.cache_data(ttl=300)
def fetch_nbm_percentiles(lat, lon):
    city_name = None
    best_dist = float('inf')
    for c, coords in CITIES.items():
        dist = abs(coords['lat'] - lat) + abs(coords['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            city_name = c
    station = WETHR_STATIONS.get(city_name, '')
    if not station: return None
    city_tz = pytz.timezone(CITY_TZ.get(city_name, 'America/New_York'))
    today_local = datetime.now(city_tz).strftime('%Y-%m-%d')
    now_utc = datetime.utcnow()
    start_utc = (now_utc - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_utc = (now_utc + timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        r = requests.get(
            'https://wethr.net/api/v2/forecasts.php',
            params={'location_name': station, 'start_valid_time': start_utc,
                    'end_valid_time': end_utc, 'model': 'NBM'},
            headers=WETHR_HEADERS, timeout=15
        )
        if r.status_code != 200: return None
        forecasts = r.json()
        if not forecasts or not isinstance(forecasts, list): return None
        run_highs = {}
        for f in forecasts:
            valid_time_str = f.get('valid_time', '')
            temp_f = f.get('temperature_f')
            run_time = f.get('run_time', '')
            if temp_f is None: continue
            try:
                vt_utc = datetime.strptime(valid_time_str, '%Y-%m-%d %H:%M:%S')
                vt_utc = pytz.utc.localize(vt_utc)
                vt_local = vt_utc.astimezone(city_tz)
                if not (6 <= vt_local.hour <= 21): continue
                if vt_local.strftime('%Y-%m-%d') != today_local: continue
            except Exception: continue
            if run_time not in run_highs: run_highs[run_time] = []
            run_highs[run_time].append(float(temp_f))
        if not run_highs: return None
        run_max_temps = [max(temps) for temps in run_highs.values() if temps]
        if len(run_max_temps) < 1: return None
        run_max_temps.sort()
        def percentile(data, p):
            idx = (p / 100) * (len(data) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(data) - 1)
            return round(data[lo] + (idx - lo) * (data[hi] - data[lo]), 1)
        return {
            'p10': percentile(run_max_temps, 10), 'p25': percentile(run_max_temps, 25),
            'p50': percentile(run_max_temps, 50), 'p75': percentile(run_max_temps, 75),
            'p90': percentile(run_max_temps, 90),
        }
    except Exception: return None

def nbm_bracket_prob(nbm_percentiles, lo, hi, obs_high=None):
    if not nbm_percentiles: return None
    cdf_points = []
    pct_map = {'p10': 0.10, 'p25': 0.25, 'p50': 0.50, 'p75': 0.75, 'p90': 0.90}
    for key, prob in sorted(pct_map.items(), key=lambda x: x[1]):
        if key in nbm_percentiles: cdf_points.append((nbm_percentiles[key], prob))
    if len(cdf_points) < 2: return None
    cdf_points.sort(key=lambda x: x[0])
    def cdf(t):
        if t <= cdf_points[0][0]: return max(0.0, cdf_points[0][1] * (t - (cdf_points[0][0] - 5)) / 5)
        if t >= cdf_points[-1][0]:
            remaining = 1.0 - cdf_points[-1][1]
            span = max(cdf_points[-1][0] - cdf_points[-2][0], 1.0)
            return min(1.0, cdf_points[-1][1] + remaining * (t - cdf_points[-1][0]) / span)
        for i in range(len(cdf_points) - 1):
            t0, p0 = cdf_points[i]; t1, p1 = cdf_points[i + 1]
            if t0 <= t <= t1:
                return p0 + (t - t0) / max(t1 - t0, 0.001) * (p1 - p0)
        return 0.5
    if lo is None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4: return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5)))
    elif hi is None and lo is not None:
        return max(0.0, min(1.0, 1.0 - cdf(lo - 0.5)))
    elif lo is not None and hi is not None:
        if obs_high is not None and obs_high > hi + 0.4: return 0.0
        return max(0.0, min(1.0, cdf(hi + 0.5) - cdf(lo - 0.5)))
    return None

def bracket_probs_nbm(consensus, ladder_text, city, nbm_percentiles, obs_high=None, forecast=None):
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
    if obs_high is not None and hi is not None and obs_high > hi + 0.4: return 0.0
    if lo is None: return normal_cdf(hi + 0.5, mu, sigma)
    elif hi is None: return 1 - normal_cdf(lo - 0.5, mu, sigma)
    else: return normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)

def check_market_conviction(kalshi_markets, ladder_text, conviction_threshold=70):
    if not kalshi_markets: return None
    for label, yes_ask, no_ask in kalshi_markets:
        if yes_ask is not None and yes_ask >= conviction_threshold:
            lo, hi = label_to_numeric_key(label)
            return (label, lo, hi,
                    f'⚠️ Market conviction: {label} is priced at {yes_ask}c ({yes_ask}% implied). '
                    f'Market strongly believes the high will be in this bracket. '
                    f'Verify before betting any other bracket.')
    return None

def is_conflicting_with_conviction(label, conviction_lo, conviction_hi, ladder_text):
    if conviction_lo is None and conviction_hi is None: return False
    parsed = parse_ladder(ladder_text)
    conviction_idx = label_idx = None
    for i, (lbl, lo, hi) in enumerate(parsed):
        lo_k, hi_k = label_to_numeric_key(lbl)
        lo_l, hi_l = label_to_numeric_key(label)
        if lo_k == conviction_lo and hi_k == conviction_hi: conviction_idx = i
        if lo_l == lo_k and hi_l == hi_k: label_idx = i
    if conviction_idx is None or label_idx is None: return False
    return abs(label_idx - conviction_idx) > 1

def check_bracket_boundary(consensus, ladder_text, boundary_threshold=0.5):
    warnings = []
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and abs(consensus - hi) <= boundary_threshold:
            warnings.append(f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of {hi}F ceiling ({label}) — NWS rounding could push to next bracket up. Verify before betting.')
        if lo is not None and abs(consensus - lo) <= boundary_threshold:
            warnings.append(f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of {lo}F floor ({label}) — NWS rounding could push to bracket below. Verify before betting.')
    return warnings

def check_cold_front_warning(obs_high, current_temp, nws_forecast, local_hour):
    # Cold front warning only fires after 1 PM local. Before that, obs_high vs current_temp
    # gaps are usually morning fluctuations or sensor blips, not actual frontal passages.
    if obs_high is not None and current_temp is not None and local_hour >= 13:
        temp_drop = obs_high - current_temp
        if temp_drop >= 5.0:
            msg = (f'⚠️ Peak may already be in: Obs high {obs_high}F but current temp is '
                   f'{round(current_temp, 1)}F ({round(temp_drop, 1)}F drop). Cold front passage likely — verify before betting.')
            if nws_forecast is not None and current_temp < nws_forecast - 3.0:
                msg += f' NWS forecast ({nws_forecast}F) also now above current temp.'
            return msg
    if obs_high is None and current_temp is not None and nws_forecast is not None:
        fc_gap = nws_forecast - current_temp
        if fc_gap >= 8.0 and local_hour < 12:
            return (f'⚠️ No obs high — current temp {round(current_temp, 1)}F is '
                    f'{round(fc_gap, 1)}F below NWS forecast ({nws_forecast}F) in morning hours. '
                    f'Possible overnight peak or front passage. Verify before betting.')
    return None

def check_morning_suppression(obs_high, current_temp, nws_forecast, local_hour):
    if obs_high is not None: return False, None
    if current_temp is None or nws_forecast is None: return False, None
    fc_gap = nws_forecast - current_temp
    if fc_gap >= 5.0 and local_hour < 12:
        return True, (f'⚠️ Signal suppression active: No obs high + current temp '
                      f'({round(current_temp, 1)}F) is {round(fc_gap, 1)}F below NWS forecast '
                      f'({nws_forecast}F) in morning hours. High may have already occurred. '
                      f'Green signals suppressed — verify manually before betting.')
    return False, None

_NWS_GRID_CACHE = {}

def fetch_nws_grid(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _NWS_GRID_CACHE: return _NWS_GRID_CACHE[key]
    try:
        r = requests.get(f'https://api.weather.gov/points/{lat},{lon}', headers=HEADERS, timeout=12)
        r.raise_for_status()
        props = r.json().get('properties', {})
        office, gx, gy, fc_url = props.get('gridId'), props.get('gridX'), props.get('gridY'), props.get('forecast')
        if not all([office, gx is not None, gy is not None, fc_url]): return None
        result = (office, gx, gy, fc_url)
        _NWS_GRID_CACHE[key] = result
        return result
    except Exception: return None

@st.cache_data(ttl=300)
def fetch_nws_forecast(lat, lon):
    city_name = None
    best_dist = float('inf')
    for c, coords in CITIES.items():
        dist = abs(coords['lat'] - lat) + abs(coords['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            city_name = c
    station = WETHR_STATIONS.get(city_name)
    today = get_eastern_date()
    if station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/nws_forecasts.php',
                params={'station_code': station, 'date': today, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=12
            )
            if r.status_code == 200:
                data = r.json()
                high = data.get('high')
                if high is not None:
                    return round(float(high), 1), 'wethr_nws_forecasts'
        except Exception: pass
    grid = fetch_nws_grid(lat, lon)
    if not grid: return None, None
    office, gx, gy, _ = grid
    hourly_url = f'https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast/hourly'
    try:
        r = requests.get(hourly_url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        periods = r.json().get('properties', {}).get('periods', [])
        today_highs = []
        for period in periods:
            start = period.get('startTime', '')
            temp = period.get('temperature')
            unit = period.get('temperatureUnit', 'F')
            is_day = period.get('isDaytime', True)
            if not start.startswith(today): continue
            if temp is not None and is_day:
                temp_f = float(temp) if unit == 'F' else float(temp) * 9/5 + 32
                today_highs.append(temp_f)
        if today_highs:
            return round(max(today_highs), 1), hourly_url
    except Exception: pass
    return None, None

@st.cache_data(ttl=300)
def fetch_nws_current(lat, lon, station_id):
    city_name = None
    best_dist = float('inf')
    for c, coords in CITIES.items():
        dist = abs(coords['lat'] - lat) + abs(coords['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            city_name = c
    wethr_station = WETHR_STATIONS.get(city_name)
    if station_id:
        obs = safe_get('https://api.weather.gov/stations/' + station_id + '/observations/latest')
        if obs:
            props = obs.get('properties', {})
            temp_c = props.get('temperature', {}).get('value')
            obs_ts = props.get('timestamp')
            if temp_c is not None:
                return station_id, float(c_to_f(temp_c)), 'nws_direct', obs_ts
    if station_id:
        try:
            now_utc = datetime.utcnow()
            sts_iso = (now_utc - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
            r = requests.get(
                'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                params={
                    'station': station_id, 'data': 'tmpf', 'sts': sts_iso,
                    'format': 'onlycomma', 'tz': 'UTC', 'missing': 'empty',
                    'latlon': 'no', 'report_type': '3,4',
                },
                timeout=10
            )
            if r.status_code == 200 and r.text:
                lines = [ln.strip() for ln in r.text.strip().split('\n') if ln.strip()]
                for line in reversed(lines[1:] if len(lines) > 1 else []):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        ts_str, _, tmpf_str = parts[1], parts[0], parts[2]
                        try:
                            tmpf = float(tmpf_str)
                            if -50 <= tmpf <= 140:
                                ts_iso = ts_str.replace(' ', 'T') + 'Z'
                                return station_id, round(tmpf, 1), 'iowa_mesonet', ts_iso
                        except (ValueError, IndexError):
                            continue
        except Exception: pass
    if wethr_station:
        try:
            r = requests.get(
                'https://wethr.net/api/v2/observations.php',
                params={'station_code': wethr_station, 'mode': 'latest'},
                headers=WETHR_HEADERS, timeout=12
            )
            if r.status_code == 200:
                data = r.json()
                temp_display = data.get('temperature_display')
                obs_ts = data.get('observation_time') or data.get('timestamp')
                if temp_display is not None:
                    return wethr_station, round(float(temp_display), 1), 'wethr_obs', obs_ts
        except Exception: pass
    points = safe_get('https://api.weather.gov/points/' + str(lat) + ',' + str(lon))
    if not points: return station_id, None, None, None
    stations_url = points.get('properties', {}).get('observationStations')
    if not stations_url: return station_id, None, None, None
    stations = safe_get(stations_url)
    if not stations or not stations.get('observationStations'): return station_id, None, None, None
    first = stations['observationStations'][0]
    sid = first.rstrip('/').split('/')[-1]
    obs = safe_get(first + '/observations/latest')
    if not obs: return sid, None, None, None
    props = obs.get('properties', {})
    temp_c = props.get('temperature', {}).get('value')
    obs_ts = props.get('timestamp')
    if temp_c is None: return sid, None, None, None
    return sid, float(c_to_f(temp_c)), 'nws_nearby', obs_ts

def kelly_bet(model_prob, market_price_cents, bankroll, fractional=0.15, max_pct=0.05, max_dollars=100):
    if market_price_cents is None or market_price_cents <= 0 or market_price_cents >= 100: return 0.0
    p, q = model_prob, 1.0 - model_prob
    price = market_price_cents / 100.0
    odds = (1.0 - price) / price
    kelly_full = (p * odds - q) / odds
    if kelly_full <= 0: return 0.0
    return round(max(0.0, min(kelly_full * fractional * bankroll, max_pct * bankroll, max_dollars)), 2)

def edge_cents(model_prob, market_price_cents):
    if market_price_cents is None: return None
    return round(model_prob * 100 - market_price_cents, 1)

def edge_signal(e, high_uncertainty=False, morning_suppressed=False, conviction_conflict=False):
    if e is None: return '⚪', 'No price'
    if morning_suppressed:
        return ('🟡', 'SKIP (no obs high — verify)') if e >= MIN_EDGE else ('🟡', 'SKIP') if e >= 3 else ('🔴', 'AVOID')
    if conviction_conflict:
        return ('🟡', 'SKIP (market conviction conflict)') if e >= MIN_EDGE else ('🟡', 'SKIP') if e >= 3 else ('🔴', 'AVOID')
    if high_uncertainty:
        return ('🟡', 'SKIP (uncertain)') if e >= MIN_EDGE else ('🟡', 'SKIP') if e >= 3 else ('🔴', 'AVOID')
    if e >= MIN_EDGE: return '🟢', 'BET'
    if e >= 3: return '🟡', 'SKIP'
    return '🔴', 'AVOID'

def no_edge_cents(model_prob, no_ask_cents):
    if no_ask_cents is None: return None
    return round((1.0 - model_prob) * 100 - no_ask_cents, 1)

def no_signal(no_edge, busted=False, model_prob=None, no_ask=None,
              high_uncertainty=False, morning_suppressed=False, conviction_conflict=False):
    if busted:
        return ('🟢', 'BET NO (busted)') if no_ask is not None and no_ask <= 5 else ('🟡', 'CONSIDER NO (busted)')
    if no_edge is None: return '⚪', 'No price'
    if morning_suppressed or conviction_conflict or high_uncertainty:
        return ('🟡', 'SKIP NO') if no_edge >= MIN_EDGE else ('🔴', 'AVOID')
    if no_edge >= MIN_EDGE: return '🟢', 'BET NO'
    if no_edge >= 3: return '🟡', 'SKIP NO'
    return '🔴', 'AVOID'

def kelly_bet_no(model_prob, no_ask_cents, bankroll, fractional=0.15, max_pct=0.05, max_dollars=100):
    if no_ask_cents is None or no_ask_cents <= 0 or no_ask_cents >= 100: return 0.0
    p, q = 1.0 - model_prob, model_prob
    price = no_ask_cents / 100.0
    odds = (1.0 - price) / price
    kelly_full = (p * odds - q) / odds
    if kelly_full <= 0: return 0.0
    return round(max(0.0, min(kelly_full * fractional * bankroll, max_pct * bankroll, max_dollars)), 2)

def compute_row_trust(city, bracket_label, direction, model_pct, ensemble_tier, two_degree_call_str,
                      mae_color, nbm_active, nws_forecast_f, gfs_ensemble_f, bias_adj_f):
    try:
        inp = SignalInputs(
            city=str(city or ''), bracket_label=str(bracket_label or ''),
            direction=str(direction or 'YES'), two_degree_call=str(two_degree_call_str or ''),
            bracket_midpoint=bracket_midpoint_from_label(bracket_label),
            twodc_midpoint=bracket_midpoint_from_label(two_degree_call_str),
            model_pct=float(model_pct or 0), edge_cents=0.0,
            ensemble_tier=str(ensemble_tier or ''), mae_color=str(mae_color or 'green'),
            nbm_active=bool(nbm_active),
            nws_forecast_f=float(nws_forecast_f) if nws_forecast_f is not None else None,
            gfs_ensemble_f=float(gfs_ensemble_f) if gfs_ensemble_f is not None else None,
            bias_adj_f=float(bias_adj_f or 0),
        )
        return compute_trust_score(inp)
    except Exception: return None

def compute_edge_trust(model_pct, yes_ask, ensemble_tier):
    if yes_ask is None or model_pct is None:
        return 0.0
    try:
        yes_ask = float(yes_ask)
        model_pct = float(model_pct)
        market_implied = yes_ask
        if yes_ask <= 5:   price_score = 50
        elif yes_ask <= 10: price_score = 40
        elif yes_ask <= 15: price_score = 30
        elif yes_ask <= 20: price_score = 15
        else:               price_score = 0
        gap = model_pct - market_implied
        if gap >= 25:   gap_score = 35
        elif gap >= 20: gap_score = 28
        elif gap >= 15: gap_score = 20
        elif gap >= 10: gap_score = 10
        else:           gap_score = 0
        ens = str(ensemble_tier or '').upper()
        if 'HIGH' in ens:   ens_score = 15
        elif 'MED' in ens:  ens_score = 8
        else:               ens_score = 0
        total = price_score + gap_score + ens_score
        return round(min(100.0, total), 1)
    except Exception:
        return 0.0

def trust_tier_icon(tier):
    if tier == 'BET': return '🟢'
    if tier == 'CAUTION': return '🟡'
    if tier == 'SKIP': return '⚪'
    return '—'

def ensemble_tier_from_confidence(conf_str):
    if not conf_str: return ''
    if 'HIGH' in conf_str: return 'HIGH'
    if 'MED' in conf_str: return 'MED'
    if 'LOW' in conf_str: return 'LOW'
    return ''


# V5.27 Three-gate logic ──────────────────────────────────────────────────────

def get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N):
    if not kalshi_markets:
        return []
    priced = [(label, yes_ask) for label, yes_ask, _ in kalshi_markets if yes_ask is not None]
    if not priced:
        return []
    priced.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in priced[:n]]


def model_top_pick(prob_rows):
    if not prob_rows:
        return None
    return prob_rows[0][0]


def passes_consensus_gate(model_pick_label, kalshi_markets):
    if not model_pick_label:
        return False, 'No model pick'
    market_top = get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N)
    if not market_top:
        return False, 'No market prices'
    for mt in market_top:
        if labels_match(mt, model_pick_label):
            rank = market_top.index(mt) + 1
            return True, f'Consensus #{rank}'
    return False, f'Model pick "{model_pick_label}" not in market top {CONSENSUS_TOP_N}: {market_top}'


def passes_price_floor(price_cents):
    if price_cents is None:
        return False, 'No price'
    if price_cents < PRICE_FLOOR_CENTS:
        return False, f'Price {price_cents}c < {PRICE_FLOOR_CENTS}c floor'
    if price_cents >= 99:
        return False, f'Price {price_cents}c too high — no edge'
    return True, 'Price floor OK'


def select_qualifying_bet_v527(city, prob_rows, kalshi_markets, ladder_text, ensemble_members,
                                obs_high, high_uncertainty, morning_suppressed, conviction_result,
                                consensus, used_nbm, nws_forecast, ensemble_mean, bias_correction,
                                city_mae_color, two_degree_call_str, bankroll):
    if not prob_rows or not kalshi_markets:
        return None

    model_pick = model_top_pick(prob_rows)
    consensus_pass, consensus_reason = passes_consensus_gate(model_pick, kalshi_markets)
    if not consensus_pass:
        return None

    target_label = model_pick
    bracket_match = next(((lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)
                          if labels_match(lbl, target_label)), (None, None))
    target_lo, target_hi = bracket_match
    target_market = next((m for m in kalshi_markets if labels_match(m[0], target_label)), None)
    if not target_market:
        return None
    yes_ask, no_ask = target_market[1], target_market[2]

    target_base_prob = next((p for lbl, p in prob_rows if labels_match(lbl, target_label)), None)
    if target_base_prob is None:
        return None
    ens_prob = ensemble_bracket_prob(ensemble_members, target_lo, target_hi) if ensemble_members else None
    final_prob = blend_probs(target_base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)

    busted = obs_high is not None and target_hi is not None and obs_high > target_hi + 0.4
    conviction_conflict = (conviction_result and
                           is_conflicting_with_conviction(target_label, conviction_result[1],
                                                          conviction_result[2], ladder_text))

    yes_e = edge_cents(final_prob, yes_ask)
    yes_icon, _ = edge_signal(yes_e, high_uncertainty, morning_suppressed, conviction_conflict)
    no_e = no_edge_cents(final_prob, no_ask)
    no_icon, _ = no_signal(no_e, busted=busted, model_prob=final_prob, no_ask=no_ask,
                           high_uncertainty=high_uncertainty, morning_suppressed=morning_suppressed,
                           conviction_conflict=conviction_conflict)

    contains_consensus = bracket_contains_consensus(target_label, consensus, ladder_text, tolerance=1.0)
    candidates = []

    if (yes_icon == '🟢' and not busted and contains_consensus
            and final_prob >= 0.10 and yes_ask is not None):
        floor_pass, _ = passes_price_floor(yes_ask)
        if floor_pass:
            ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''
            ens_tier = ensemble_tier_from_confidence(ens_conf)
            trust_yes = compute_row_trust(
                city=city, bracket_label=target_label, direction='YES',
                model_pct=final_prob * 100, ensemble_tier=ens_tier,
                two_degree_call_str=two_degree_call_str or '', mae_color=city_mae_color,
                nbm_active=used_nbm, nws_forecast_f=nws_forecast,
                gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
            )
            edge_trust_yes = compute_edge_trust(
                model_pct=final_prob * 100, yes_ask=yes_ask, ensemble_tier=ens_tier,
            )
            trust_acc = trust_yes.composite if trust_yes else 0.0
            kelly = kelly_bet(final_prob, yes_ask, bankroll)
            candidates.append({
                'side': 'YES', 'label': target_label, 'price': yes_ask,
                'edge': yes_e, 'kelly': kelly, 'model_prob': final_prob,
                'trust_accuracy': round(trust_acc, 1),
                'trust_edge': round(edge_trust_yes, 1),
                'gate_reason': consensus_reason,
            })

    no_qualifies = False
    if busted and no_ask is not None and no_ask <= 5:
        no_qualifies = True
    elif (no_icon == '🟢' and (1.0 - final_prob) >= 0.10
          and not busted and no_ask is not None):
        no_qualifies = True

    if no_qualifies:
        floor_pass, _ = passes_price_floor(no_ask)
        if floor_pass or (busted and no_ask is not None and no_ask <= 5):
            ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''
            ens_tier = ensemble_tier_from_confidence(ens_conf)
            trust_no = compute_row_trust(
                city=city, bracket_label=target_label, direction='NO',
                model_pct=(1.0 - final_prob) * 100, ensemble_tier=ens_tier,
                two_degree_call_str=two_degree_call_str or '', mae_color=city_mae_color,
                nbm_active=used_nbm, nws_forecast_f=nws_forecast,
                gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
            )
            edge_trust_no = compute_edge_trust(
                model_pct=(1.0 - final_prob) * 100, yes_ask=no_ask, ensemble_tier=ens_tier,
            )
            trust_acc = trust_no.composite if trust_no else 0.0
            kelly_no = kelly_bet_no(final_prob, no_ask, bankroll)
            no_edge_val = no_e if no_e is not None else (95 if busted else 0)
            candidates.append({
                'side': 'NO', 'label': target_label, 'price': no_ask,
                'edge': no_edge_val, 'kelly': kelly_no, 'model_prob': 1.0 - final_prob,
                'trust_accuracy': round(trust_acc, 1),
                'trust_edge': round(edge_trust_no, 1),
                'gate_reason': consensus_reason + (' (busted NO)' if busted else ''),
            })

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c['trust_accuracy'], c['edge']), reverse=True)
    return candidates[0]


def evaluate_all_cities_v527(saved_ladders, bankroll):
    today_rows = {r['city']: r for r in (sb_fetch_all() or []) if r.get('date') == get_eastern_date()}
    qualifying_picks = {}
    rejected = {}

    for c in CITIES.keys():
        if c in HIDDEN_CITIES:
            continue
        row = today_rows.get(c)
        if not row:
            rejected[c] = 'No prediction logged for today'
            continue
        consensus_val = row.get('consensus')
        if consensus_val is None:
            rejected[c] = 'Consensus unavailable'
            continue

        ladder = saved_ladders.get(c, DEFAULT_LADDERS.get(c, ''))
        cached_markets, _ = get_cached_prices(c)
        if not cached_markets:
            rejected[c] = 'No cached Kalshi prices'
            continue

        try:
            wx = fetch_city_weather(c)
        except Exception:
            wx = None

        members = wx.get('ensemble_members') if wx else None
        nbm_pcts = wx.get('nbm_percentiles') if wx else None
        c_temp = wx.get('current_temp') if wx else None
        c_fc = wx.get('nws_fc') if wx else None
        c_hour = wx.get('local_hour', 12) if wx else 12

        try:
            prob_rows, _, used_nbm = bracket_probs_nbm(
                consensus_val, ladder, c, nbm_pcts,
                obs_high=row.get('obs_high'), forecast=c_fc,
            )
            prob_rows = apply_prob_floor(prob_rows, consensus_val, ladder)
        except Exception:
            rejected[c] = 'Probability computation failed'
            continue

        morning_suppressed, _ = check_morning_suppression(
            row.get('obs_high'), c_temp, c_fc, c_hour,
        )
        conviction_result = check_market_conviction(cached_markets, ladder)
        bias_correction, _ = compute_bias_correction_db(c)
        mae_val, mae_color = get_city_mae_and_color(c)
        call = two_degree_call(consensus_val, ladder, obs_high=row.get('obs_high'))

        pick = select_qualifying_bet_v527(
            city=c, prob_rows=prob_rows, kalshi_markets=cached_markets,
            ladder_text=ladder, ensemble_members=members,
            obs_high=row.get('obs_high'),
            high_uncertainty=row.get('high_uncertainty', False),
            morning_suppressed=morning_suppressed,
            conviction_result=conviction_result,
            consensus=consensus_val, used_nbm=used_nbm,
            nws_forecast=c_fc, ensemble_mean=wx.get('ensemble_mean') if wx else None,
            bias_correction=bias_correction, city_mae_color=mae_color,
            two_degree_call_str=call, bankroll=bankroll,
        )

        if pick is None:
            model_pick = model_top_pick(prob_rows)
            cp, cr = passes_consensus_gate(model_pick, cached_markets)
            if not cp:
                rejected[c] = f'Gate 1 fail: {cr}'
            else:
                rejected[c] = 'Gate 2 / safety filter rejected (price floor / busted / conviction / morning)'
        else:
            pick['city'] = c
            pick['consensus'] = consensus_val
            qualifying_picks[c] = pick

    best_overall = None
    if qualifying_picks:
        ranked = sorted(qualifying_picks.values(),
                        key=lambda p: (p['trust_accuracy'], p['edge']), reverse=True)
        best_overall = ranked[0]

    return {
        'best_overall': best_overall,
        'qualifying_picks': qualifying_picks,
        'rejected': rejected,
    }


@st.cache_data(ttl=300)
def fetch_gfs_ensemble(lat, lon):
    params = {'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m',
               'temperature_unit': 'fahrenheit', 'timezone': 'auto', 'forecast_days': 2, 'models': 'gfs_seamless'}
    try:
        r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble', params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception: return None, None
    today = get_eastern_date()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    today_indices = [i for i, t in enumerate(times)
                     if t.startswith(today) and len(t) >= 13 and 6 <= int(t[11:13]) <= 21]
    if not today_indices: today_indices = [i for i, t in enumerate(times) if t.startswith(today)]
    if not today_indices: return None, None
    member_maxes = []
    for key, vals in hourly.items():
        if key == 'time' or 'temperature_2m' not in key or not isinstance(vals, list): continue
        today_vals = [vals[i] for i in today_indices if i < len(vals) and vals[i] is not None]
        if today_vals:
            try: member_maxes.append(round(max(float(v) for v in today_vals), 1))
            except Exception: pass
    if len(member_maxes) < 3: return None, None
    return member_maxes, round(sum(member_maxes) / len(member_maxes), 1)

def ensemble_bracket_prob(members, lo, hi):
    if not members: return None
    return sum(1 for m in members if (lo is None or m >= lo - 0.5) and (hi is None or m <= hi + 0.5)) / len(members)

def ensemble_confidence(prob):
    if prob is None: return ''
    if prob >= 0.80 or prob <= 0.20: return '🔵 HIGH'
    if prob >= 0.65 or prob <= 0.35: return '🟡 MED'
    return '⚪ LOW'

def ensemble_overall_confidence(members, consensus, ladder_text):
    if not members or not ladder_text: return ''
    try:
        best_ens_prob = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            mid = (hi - 1.0 if lo is None and hi is not None else
                   lo + 1.0 if hi is None and lo is not None else
                   (lo + hi) / 2.0 if lo is not None and hi is not None else None)
            if mid is not None and consensus is not None and abs(mid - consensus) <= 2.0:
                prob = ensemble_bracket_prob(members, lo, hi)
                if prob is not None: best_ens_prob = prob; break
        if best_ens_prob is None:
            probs = [p for p in (ensemble_bracket_prob(members, lo, hi) for _, lo, hi in parse_ladder(ladder_text)) if p is not None]
            best_ens_prob = max(probs) if probs else None
        return ensemble_confidence(best_ens_prob)
    except Exception: return ''

def blend_probs(sigma_prob, ensemble_prob, members, city='', nbm_active=False):
    if ensemble_prob is None or members is None: return sigma_prob
    base_weight = GFS_CITY_WEIGHT.get(city, 0.20)
    ensemble_weight = base_weight * 0.5 if nbm_active else base_weight
    return round((1.0 - ensemble_weight) * sigma_prob + ensemble_weight * ensemble_prob, 4)

def apply_prob_floor(prob_rows, consensus, ladder_text):
    if not prob_rows or consensus is None: return prob_rows
    parsed = {lbl: (lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)}
    adjusted = []
    boost_total = 0.0
    for label, prob in prob_rows:
        lo, hi = parsed.get(label, (None, None))
        if lo is not None and hi is not None: mid = (lo + hi) / 2.0
        elif lo is not None: mid = lo + 1.0
        elif hi is not None: mid = hi - 1.0
        else: adjusted.append((label, prob)); continue
        distance = abs(mid - consensus)
        new_prob = prob
        if distance <= 4.0 and prob < 0.05: new_prob = 0.05
        elif distance <= 6.0 and prob < 0.02: new_prob = 0.02
        boost_total += (new_prob - prob)
        adjusted.append((label, new_prob))
    if boost_total > 0:
        scale = 1.0 / (1.0 + boost_total)
        adjusted = [(lbl, round(p * scale, 4)) for lbl, p in adjusted]
    return adjusted


def get_eastern_date():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')

def get_eastern_datetime():
    return datetime.now(pytz.timezone('America/New_York'))

def get_local_hour(city):
    return datetime.now(pytz.timezone(CITY_TZ.get(city, 'America/New_York'))).hour

def get_event_ticker(series):
    return series + '-' + get_eastern_datetime().strftime('%d%b%y').upper()

def load_json(path):
    if path.exists():
        try: return json.loads(path.read_text())
        except Exception: return {}
    return {}

def save_json(path, data): path.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception: return None

def safe_get_with_retry(url, params=None, retries=3, delay=2.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1: time.sleep(delay)
    return None

def c_to_f(c): return c * 9 / 5 + 32

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def normalize_label(label):
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)', lambda m: m.group(1)+'-'+m.group(2), label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)', lambda m: m.group(1)+'-'+m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above').strip()

def label_to_numeric_key(label):
    label = normalize_label(label)
    nums = [int(x) for x in re.findall(r'\d+', label)]
    low = label.lower()
    if not nums: return None, None
    if 'below' in low: return None, nums[0]
    if 'above' in low: return nums[0], None
    if len(nums) >= 2: return nums[0], nums[1]
    return None, None

def labels_match(a, b):
    return label_to_numeric_key(a) == label_to_numeric_key(b)

def parse_ladder(text):
    out = []
    for p in text.split('|'):
        p = normalize_label(p)
        nums = [int(x) for x in re.findall(r'\d+', p)]
        if not nums: continue
        low = p.lower()
        if 'below' in low: out.append((p, None, nums[0]))
        elif 'above' in low: out.append((p, nums[0], None))
        elif len(nums) >= 2: out.append((p, nums[0], nums[1]))
    return out

def choose_sigma(city, obs_high=None, forecast=None):
    s = BASE_SIGMA.get(city, 2.1)
    local_hour = get_local_hour(city)
    s *= 1.00 if local_hour < 11 else 0.94 if local_hour < 14 else 0.90 if local_hour < 16 else 0.86
    if city in DESERT_CITIES: s *= 0.92
    if obs_high is not None and forecast is not None:
        gap = abs(forecast - obs_high)
        if gap < 2: s *= 0.80
        elif gap < 4: s *= 0.90
    return max(1.30, min(2.80, s))

def late_day_floor(fc, obs, local_hour, city=''):
    gap = max(0.0, fc - obs)
    if city in NORTHEAST_CITIES:
        frac = 0.50 if local_hour < 12 else 0.75 if local_hour < 14 else 0.88 if local_hour < 16 else 0.93
    else:
        frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap

def compute_consensus(fc, cur, noaa, city, obs_high=None):
    mode = CITY_PREDICTION_MODE.get(city, 'full_blend')
    if mode == 'nws_only':
        return float(fc)
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
    if abs(base - fc) > 4.0: base = fc - 4.0 if base < fc else fc + 4.0
    obs = noaa if noaa is not None else cur
    if obs is not None: consensus = max(base, late_day_floor(fc, obs, local_hour, city))
    else: consensus = base
    if obs_high is not None and obs_high > consensus:
        obs_high_trusted = True
        if local_hour < OBS_HIGH_TRUST_HOUR: obs_high_trusted = False
        current_for_check = obs if obs is not None else cur
        if current_for_check is not None and obs_high > current_for_check + OBS_HIGH_MAX_OVERSHOOT:
            obs_high_trusted = False
        if current_for_check is not None and obs_high < current_for_check:
            obs_high_trusted = False
        if obs_high_trusted: consensus = obs_high
    warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
    if warm_offset != 0.0:
        consensus = consensus + warm_offset
    return consensus

def bracket_probs(mu, ladder_text, city, obs_high=None, forecast=None):
    sigma = choose_sigma(city, obs_high=obs_high, forecast=forecast)
    rows = []
    for label, lo, hi in parse_ladder(ladder_text):
        if obs_high is not None and hi is not None and obs_high > hi + 0.4:
            rows.append((label, 0.0)); continue
        if lo is None: p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None: p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else: p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((label, max(0.0, min(1.0, p))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, sigma

def two_degree_call(mu, ladder_text, obs_high=None):
    best_label, best_dist = None, float('inf')
    for label, lo, hi in parse_ladder(ladder_text):
        if obs_high is not None and hi is not None and obs_high > hi + 0.4: continue
        mid = (hi - 1.0 if lo is None and hi is not None else
               lo + 1.0 if hi is None and lo is not None else
               (lo + hi) / 2 if lo is not None and hi is not None else None)
        if mid is None: continue
        dist = abs(mid - mu)
        if dist < best_dist: best_dist = dist; best_label = label
    return best_label

def bracket_contains_consensus(label, consensus, ladder_text, tolerance=1.0):
    if consensus is None: return True
    for lbl, lo, hi in parse_ladder(ladder_text):
        if not labels_match(lbl, label): continue
        if lo is None and hi is not None:
            return consensus <= hi + tolerance
        if hi is None and lo is not None:
            return consensus >= lo - tolerance
        if lo is not None and hi is not None:
            return (lo - tolerance) <= consensus <= (hi + tolerance)
    return False

def ladder_to_boxes(text):
    parts = [normalize_label(p) for p in text.split('|')]
    while len(parts) < 6: parts.append('')
    return parts[:6]

def boxes_to_ladder(parts):
    cleaned = []
    for i, p in enumerate(parts):
        t = normalize_label(p)
        if not t: continue
        nums = re.findall(r'\d+', t)
        low = t.lower()
        if 'below' in low or 'above' in low or '-' in t: cleaned.append(t)
        elif len(nums) == 1:
            n = int(nums[0])
            cleaned.append(str(n) + (' or below' if i == 0 else ' or above' if i == 5 else ''))
        else: cleaned.append(t)
    return ' | '.join(cleaned)

@st.cache_data(ttl=300)
def fetch_obs_high_today(icao):
    try:
        r = requests.get(
            'https://wethr.net/api/v2/observations.php',
            params={'station_code': icao, 'mode': 'wethr_high', 'logic': 'nws'},
            headers=WETHR_HEADERS, timeout=12
        )
        if r.status_code == 200:
            data = r.json()
            wethr_high = data.get('wethr_high')
            if wethr_high is not None:
                high_val = round(float(wethr_high), 1)
                return high_val, high_val, 'wethr_api_nws'
    except Exception: pass
    eastern = pytz.timezone('America/New_York')
    today_day = str(datetime.now(eastern).day)
    url = 'https://forecast.weather.gov/data/obhistory/' + icao + '.html'
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
    except Exception: return None, None, url
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    table = max(tables, key=lambda t: len(t.find_all('tr')), default=None) if tables else None
    if not table: return None, None, url
    highs = []
    six_hr_maxes = []
    for row in table.find_all('tr'):
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cols or len(cols) < 9 or cols[0] != today_day: continue
        try:
            t = float(cols[8])
            if 0 < t < 130: highs.append(t)
        except Exception: pass
        if len(cols) > 10:
            try:
                six_max = float(cols[10])
                if 0 < six_max < 130: six_hr_maxes.append(six_max)
            except Exception: pass
    obs_high = round(max(highs), 1) if highs else None
    six_hr_max = round(max(six_hr_maxes), 1) if six_hr_maxes else None
    true_high = max(filter(None, [obs_high, six_hr_max])) if (obs_high or six_hr_max) else None
    return true_high, six_hr_max, url

def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = normalize_label((m.get(field) or '').replace('\u00b0', '').replace('deg', '').strip())
        if s:
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below: return below.group(1)+' or below', int(below.group(1))-10000
            if above: return above.group(1)+' or above', int(above.group(1))+10000
            if rng: return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
    title = (m.get('title') or '').replace('\u00b0', '').replace('**', '').replace('deg', '')
    if title:
        ma = re.search(r'be\s*[>=]+\s*(\d+)', title, re.I)
        if ma: n = int(ma.group(1)); return str(n)+' or above', n+10000
        mb = re.search(r'be\s*[<=]+\s*(\d+)', title, re.I)
        if mb: n = int(mb.group(1)); return str(n)+' or below', n-10000
        mr = re.search(r'be\s*(\d+)\s*(?:to|-)\s*(\d+)', title, re.I)
        if mr: lo, hi = int(mr.group(1)), int(mr.group(2)); return str(lo)+'-'+str(hi), lo
        nums = re.findall(r'\d+', title)
        if len(nums) >= 2:
            lo, hi = int(nums[-2]), int(nums[-1])
            if 0 < hi-lo <= 5: return str(lo)+'-'+str(hi), lo
    cap, floor_s = m.get('cap_strike'), m.get('floor_strike')
    if cap is not None and floor_s is not None:
        try: lo, hi = int(float(floor_s)), int(float(cap)); return str(lo)+'-'+str(hi), lo
        except Exception: pass
    if cap is not None:
        try: n = int(float(cap)); return str(n)+' or below', n-10000
        except Exception: pass
    for field in ['short_title', 'market_title', 'name']:
        val = normalize_label((m.get(field) or '').replace('\u00b0', '').strip())
        if val:
            rng = re.match(r'^(\d+)-(\d+)$', val)
            below = re.match(r'^(\d+)\s*or\s*below$', val, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', val, re.I)
            if rng: return rng.group(1)+'-'+rng.group(2), int(rng.group(1))
            if below: return below.group(1)+' or below', int(below.group(1))-10000
            if above: return above.group(1)+' or above', int(above.group(1))+10000
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
    if not data or not data.get('markets'): return None
    all_markets = data['markets']
    markets = [m for m in all_markets if
               any(x in (m.get('ticker') or '').upper() for x in [today_upper, today_upper2, today_upper3]) or
               any(x in (m.get('event_ticker') or '').upper() for x in [today_upper2, today_upper3])]
    if not markets: markets = [m for m in all_markets if (m.get('close_time') or '').startswith(today_date)]
    if not markets: markets = all_markets
    parsed = []
    for m in markets:
        label, key = parse_market_label(m)
        if label is None: continue
        yes_ask, no_ask = get_price_cents(m)
        parsed.append((key, label, yes_ask, no_ask))
    if len(parsed) < 2: return None
    parsed.sort(key=lambda x: x[0])
    return [(label, yes_ask, no_ask) for _, label, yes_ask, no_ask in parsed]

def get_cached_prices(city):
    cache = load_json(PRICE_CACHE_FILE)
    entry = cache.get(city)
    if not entry: return None, None
    if (time.time() - entry.get('fetched_at', 0)) / 60 > PRICE_CACHE_MINUTES: return None, None
    return entry.get('markets'), entry.get('fetched_at')

def save_cached_prices(city, markets):
    cache = load_json(PRICE_CACHE_FILE)
    cache[city] = {'fetched_at': time.time(), 'markets': markets}
    save_json(PRICE_CACHE_FILE, cache)

def clear_city_cache(city):
    cache = load_json(PRICE_CACHE_FILE)
    if city in cache: del cache[city]
    save_json(PRICE_CACHE_FILE, cache)

def sync_all_ladders(saved_ladders, force=False):
    today = get_eastern_date()
    last_sync = load_json(LAST_SYNC_FILE)
    if not force and last_sync.get('date') == today: return saved_ladders, None
    cities = list(SERIES.keys())
    progress = st.progress(0, text='Syncing all city ladders from Kalshi...')
    synced, failed = [], []
    for i, c in enumerate(cities):
        progress.progress((i+1)/len(cities), text='Syncing ' + c + '...')
        markets = fetch_kalshi_brackets(SERIES[c], retries=3)
        if markets:
            labels = [normalize_label(m[0]) for m in markets]
            while len(labels) < 6: labels.append('')
            saved_ladders[c] = ' | '.join(labels[:6])
            save_cached_prices(c, markets)
            synced.append(c)
        else: failed.append(c)
        time.sleep(0.5)
    save_json(SAVE_FILE, saved_ladders)
    save_json(LAST_SYNC_FILE, {'date': today, 'synced': synced, 'failed': failed})
    progress.empty()
    return saved_ladders, {'synced': synced, 'failed': failed}

@st.cache_data(ttl=300)
def fetch_city_weather(city):
    coords = CITIES[city]
    lat, lon = coords['lat'], coords['lon']
    nws_fc, _ = fetch_nws_forecast(lat, lon)
    _, current_temp, _, _ = fetch_nws_current(lat, lon, STATIONS[city])
    obs_high_raw, six_hr_max, _ = fetch_obs_high_today(OBHISTORY_STATIONS[city])
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)
    obs_high_final = obs_high_raw
    obs_high_discarded = False
    obs_high_discard_reason = None
    if obs_high_raw is not None and current_temp is not None and obs_high_raw > current_temp + 10.0:
        obs_high_final = None; obs_high_discarded = True
        obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - current_temp, 1)}F above current temp'
    if obs_high_raw is not None and nws_fc is not None and not obs_high_discarded and obs_high_raw > nws_fc + 12.0:
        obs_high_final = None; obs_high_discarded = True
        obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - nws_fc, 1)}F above NWS forecast'
    if ensemble_mean is not None and nws_fc is not None and abs(ensemble_mean - nws_fc) > 8.0:
        ensemble_members = None; ensemble_mean = None
    source_gap = None; high_uncertainty = False
    if nws_fc is not None and ensemble_mean is not None:
        source_gap = abs(nws_fc - ensemble_mean)
        threshold = get_uncertainty_threshold(city)
        high_uncertainty = source_gap > threshold
    return {
        'nws_fc': nws_fc, 'current_temp': current_temp,
        'obs_high': obs_high_final, 'obs_high_raw': obs_high_raw,
        'obs_high_discarded': obs_high_discarded, 'obs_high_discard_reason': obs_high_discard_reason,
        'ensemble_members': ensemble_members, 'ensemble_mean': ensemble_mean,
        'source_gap': source_gap, 'high_uncertainty': high_uncertainty,
        'nbm_percentiles': nbm_percentiles, 'local_hour': get_local_hour(city),
    }

def save_city_prediction(city, weather, saved_ladders):
    nws_fc = weather['nws_fc']
    if nws_fc is None: return None, False
    current_temp = weather['current_temp']
    obs_high = weather['obs_high']
    cur = current_temp if current_temp is not None else nws_fc
    consensus_raw = compute_consensus(nws_fc, cur, current_temp, city, obs_high=obs_high)
    bias_correction, _ = compute_bias_correction_db(city)
    consensus = round(consensus_raw + bias_correction, 1)
    save_ok = sb_upsert_prediction(city=city, consensus=consensus, forecast=nws_fc,
                                    ensemble_mean=weather['ensemble_mean'], source_gap=weather['source_gap'],
                                    high_uncertainty=weather['high_uncertainty'], obs_high=obs_high,
                                    bias_correction=bias_correction)
    return consensus, save_ok

@st.cache_data(ttl=30)
def _sb_fetch_bets_cached():
    try:
        r = requests.get(sb_url('bets'), headers=get_sb_headers(),
                         params={'order': 'id.asc', 'limit': '1000'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_fetch_bets():
    return _sb_fetch_bets_cached()

def sb_insert_bet(bet_dict):
    try:
        r = requests.post(sb_url('bets'), headers=get_sb_headers(), json=bet_dict, timeout=10)
        if r.status_code in (200, 201):
            _bust_bets_cache()
            rows = r.json()
            return rows[0] if rows else None
        st.error(f'Bet insert failed: {r.status_code} — {r.text[:200]}')
        return None
    except Exception as e:
        st.error(f'Bet insert exception: {str(e)[:200]}')
        return None

def sb_update_bet(bet_id, updates):
    try:
        r = requests.patch(sb_url('bets') + '?id=eq.' + str(bet_id),
                           headers=get_sb_headers(), json=updates, timeout=10)
        if r.status_code in (200, 204):
            _bust_bets_cache()
            return True
        return False
    except Exception: return False

def sb_delete_bet(bet_id):
    try:
        r = requests.delete(sb_url('bets') + '?id=eq.' + str(bet_id),
                            headers=get_sb_headers(), timeout=10)
        if r.status_code in (200, 204):
            _bust_bets_cache()
            return True
        return False
    except Exception: return False

def load_bet_log(): return sb_fetch_bets()
def save_bet_log(log): pass

def bracket_hits(actual_temp, lo, hi):
    if actual_temp is None: return None
    rounded = int(math.floor(float(actual_temp) + 0.5))
    if lo is None and hi is not None: return rounded <= hi
    if hi is None and lo is not None: return rounded >= lo
    if lo is not None and hi is not None: return lo <= rounded <= hi
    return None

def settle_bet_log(settled_rows):
    if not settled_rows: return []
    bet_log = sb_fetch_bets()
    if not bet_log: return []
    now_iso = datetime.now(pytz.timezone('America/New_York')).isoformat()
    just_settled = []
    for s in settled_rows:
        s_city = s.get('city'); s_date = s.get('date'); s_actual = s.get('actual')
        if s_city is None or s_date is None or s_actual is None: continue
        for b in bet_log:
            if b.get('result') != 'Pending': continue
            if b.get('city') != s_city: continue
            if str(b.get('date')) != s_date: continue
            bracket_str = b.get('bracket', '')
            lo, hi = label_to_numeric_key(bracket_str)
            if lo is None and hi is None: continue
            hit = bracket_hits(s_actual, lo, hi)
            if hit is None: continue
            direction = (b.get('direction') or 'YES').upper()
            won = hit if direction == 'YES' else (not hit)
            amount = float(b.get('amount', 0) or 0)
            price = float(b.get('price', 0) or 0)
            if won and price > 0:
                profit = round(amount * (100 - price) / price, 2)
                payout = profit
            else:
                profit = -amount; payout = 0.0
            updates = {'result': 'Won' if won else 'Lost', 'profit': profit,
                       'payout': payout, 'actual': s_actual, 'settled_at': now_iso}
            ok = sb_update_bet(b['id'], updates)
            if ok:
                just_settled.append({'city': s_city, 'date': s_date, 'bracket': bracket_str,
                                     'direction': direction, 'actual': s_actual, 'won': won,
                                     'amount': amount, 'profit': profit})
    return just_settled

def settle_pending_bets_retroactive():
    bet_log = sb_fetch_bets()
    if not bet_log: return []
    pending = [b for b in bet_log if b.get('result') == 'Pending']
    if not pending: return []
    all_settlements = sb_fetch_all()
    settlement_map = {}
    for s in all_settlements:
        if s.get('actual') is not None:
            key = (s.get('city'), str(s.get('date')))
            settlement_map[key] = s['actual']
    now_iso = datetime.now(pytz.timezone('America/New_York')).isoformat()
    just_settled = []
    for b in pending:
        key = (b.get('city'), str(b.get('date')))
        actual = settlement_map.get(key)
        if actual is None: continue
        bracket_str = b.get('bracket', '')
        lo, hi = label_to_numeric_key(bracket_str)
        if lo is None and hi is None: continue
        hit = bracket_hits(actual, lo, hi)
        if hit is None: continue
        direction = (b.get('direction') or 'YES').upper()
        won = hit if direction == 'YES' else (not hit)
        amount = float(b.get('amount', 0) or 0)
        price = float(b.get('price', 0) or 0)
        if won and price > 0:
            profit = round(amount * (100 - price) / price, 2)
            payout = profit
        else:
            profit = -amount; payout = 0.0
        updates = {'result': 'Won' if won else 'Lost', 'profit': profit,
                   'payout': payout, 'actual': actual, 'settled_at': now_iso}
        if sb_update_bet(b['id'], updates):
            just_settled.append({'city': b.get('city'), 'date': b.get('date'),
                                 'bracket': bracket_str, 'direction': direction,
                                 'actual': actual, 'won': won, 'amount': amount, 'profit': profit})
    return just_settled


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="mph-section-header">⚙️ Kelly Settings</div>', unsafe_allow_html=True)
    bankroll = st.number_input('Bankroll ($)', min_value=10.0, max_value=100000.0, value=500.0, step=10.0)
    st.markdown('---')
    st.markdown(f'**Edge threshold:** {MIN_EDGE}c minimum')
    st.markdown('**Kelly fraction:** 15% (conservative)')
    st.markdown('**Max per trade:** min(5% bankroll, $100)')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🚦 V5.27.1 Three Gates</div>', unsafe_allow_html=True)
    st.markdown(f'**Gate 1 — Consensus:** Model pick must equal market #1 or #2 by yes-ask')
    st.markdown(f'**Gate 2 — Price floor:** Entry ≥{PRICE_FLOOR_CENTS}c on side bought')
    st.markdown('**Gate 3 — One bet/city:** Best side (YES or NO) per city only — no spreading')
    st.caption('Research basis: gopher-lab/kalshi-go + Oalkhadra. Brackets <30c = ~0% win rate. 2-of-3 consensus = 82% win rate. Skip ~50% of days.')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">📊 Signal Key</div>', unsafe_allow_html=True)
    st.markdown('🟢 BET · 🟡 SKIP · 🔴 AVOID · ⚪ No price')
    st.markdown('🚦 = bracket passes Gate 1 · 🚫 = blocked')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🎯 Trust Columns</div>', unsafe_allow_html=True)
    st.markdown('**Trust 💎** — Edge trust (0-100)')
    st.markdown('**Trust 🎯** — Accuracy trust ← V5.27.1 tie-breaker')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🔵 Ensemble</div>', unsafe_allow_html=True)
    st.markdown('🔵 HIGH · 🟡 MED · ⚪ LOW')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🔬 MAE Guide</div>', unsafe_allow_html=True)
    st.markdown('✅ <2.5F · 🟡 2.5-4F · 🔴 >4F')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🚦 V5.27.1</div>', unsafe_allow_html=True)
    st.markdown('Bet frequency reduction.')
    st.markdown('- Three research-validated gates')
    st.markdown('- Banner shows ALL qualifying picks (ranked by Trust 🎯)')
    st.markdown('- Per-city detail with Gate Status')
    st.markdown('- ~50% of days expected to be skips')
    st.markdown('- All forecasting infra unchanged')
    st.markdown('---')
    st.caption('V5.27.1 reduces bets — does not change predictions.')


# ── Main App ──────────────────────────────────────────────────────────────────
saved_ladders = load_json(SAVE_FILE)
today_str = get_eastern_date()
last_sync_data = load_json(LAST_SYNC_FILE)

st.markdown(f"""
<div class="mph-hero">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="mph-hero-title">
                🌡️ MPH Weather Model
                <span class="mph-version-badge">V5.27.1</span>
            </div>
            <div class="mph-hero-sub">
                <span class="mph-live-dot"></span>
                LIVE · Kalshi High Temperature · {today_str} · Three-Gate Selection Active
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:11px; color:#64748b; font-family:'JetBrains Mono',monospace;">
                SETTLEMENT SOURCE<br>
                <span style="color:#94a3b8; font-size:13px;">Iowa State CLI + Wethr.net</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner('Checking for unsettled predictions...'):
    n_settled, settled_rows = run_auto_settlement()
    bet_log_just_settled = settle_bet_log(settled_rows)
    retro_settled = settle_pending_bets_retroactive()
    bet_log_just_settled.extend(retro_settled)

if n_settled > 0:
    for s in settled_rows:
        direction = '✅' if abs(s['error']) <= 1.5 else '⚠️'
        st.success(f"{direction} Auto-settled {s['city']} ({s['date']}): actual={s['actual']}F | error={s['error']:+.1f}F")
else:
    unsettled_check = sb_fetch_unsettled()
    pending_past = [r for r in unsettled_check if r.get('date', '') < today_str]
    if pending_past:
        st.caption(f'⏳ {len(pending_past)} past predictions still pending — Iowa State CLI data may not be available yet.')

if bet_log_just_settled:
    for b in bet_log_just_settled:
        icon = '✅' if b['won'] else '❌'
        result_word = 'WON' if b['won'] else 'LOST'
        pnl_str = ('+' if b['profit'] >= 0 else '') + f"${b['profit']:.2f}"
        st.success(f"{icon} Bet {result_word}: {b['city']} {b['bracket']} {b['direction']} "
                   f"(${b['amount']:.0f}) — actual {b['actual']}F | {pnl_str}")

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
            try:
                st.cache_data.clear()
            except Exception: pass
            saved_ladders, results = sync_all_ladders(saved_ladders, force=True)
            st.success('Re-synced ' + str(len(results.get('synced', []))) + '/' + str(len(SERIES)) + ' city ladders + cleared weather cache')
            if results.get('failed'): st.warning('Could not fetch: ' + ', '.join(results['failed']))
            st.rerun()

_all_rows_stats = sb_fetch_all()
_today_rows_stats = [r for r in _all_rows_stats if r.get('date') == today_str]
_n_cities = len(_today_rows_stats)
_synced_count = len(last_sync_data.get('synced', []))
_last_sync_time = last_sync_data.get('date', '—')

st.markdown(f"""
<div class="mph-stats-bar">
    <div class="mph-stat">
        <span class="mph-stat-value">{_n_cities}/18</span>
        <span class="mph-stat-label">Cities Live</span>
    </div>
    <div class="mph-stat {'mph-stat-neutral' if _synced_count < 18 else ''}">
        <span class="mph-stat-value">{_synced_count}</span>
        <span class="mph-stat-label">Ladders Synced</span>
    </div>
    <div class="mph-stat">
        <span class="mph-stat-value" style="color:#00b4d8">{n_settled}</span>
        <span class="mph-stat-label">Settled Today</span>
    </div>
    <div class="mph-stat mph-stat-neutral">
        <span class="mph-stat-value" style="font-size:14px">{_last_sync_time}</span>
        <span class="mph-stat-label">Last Sync</span>
    </div>
</div>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html('<script>setTimeout(function(){window.location.reload();}, 600000);</script>', height=0)


# ── V5.27 Cross-City Evaluation ───────────────────────────────────────────────
with st.spinner('Running V5.27.1 three-gate evaluation across all cities...'):
    v527_eval = evaluate_all_cities_v527(saved_ladders, bankroll=bankroll)

best_overall = v527_eval['best_overall']
qualifying_picks = v527_eval['qualifying_picks']
rejected_cities = v527_eval['rejected']

st.markdown('<div class="mph-section-header">🚦 V5.27.1 — Today\'s Qualifying Bets</div>', unsafe_allow_html=True)

if not qualifying_picks:
    n_evaluated = len(qualifying_picks) + len(rejected_cities)
    skip_msg = (f'No bet passes all three gates today ({n_evaluated} cities evaluated). '
                'Skip day — preserve bankroll. Research expects ~50% of days to be skip days.')
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
        f'border:1px dashed #475569;border-radius:12px;padding:18px 24px;margin-bottom:18px;">'
        f'<div style="color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">⏸️ Skip Day</div>'
        f'<div style="color:#ffffff;font-size:18px;font-weight:600;margin-bottom:4px;">No qualifying bet today</div>'
        f'<div style="color:#64748b;font-size:13px;">{skip_msg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    # Sort all qualifying picks by Trust 🎯 (accuracy), edge as tiebreaker
    ranked_picks = sorted(qualifying_picks.values(),
                          key=lambda p: (p['trust_accuracy'], p['edge']), reverse=True)
    n_picks = len(ranked_picks)
    header_text = f'✅ {n_picks} qualifying bet{"s" if n_picks != 1 else ""} — all pass three-gate filter'
    st.markdown(
        f'<div style="color:#00ff88;font-size:13px;text-transform:uppercase;letter-spacing:1px;'
        f'margin-bottom:10px;font-family:\'JetBrains Mono\',monospace;">{header_text}</div>',
        unsafe_allow_html=True,
    )
    for idx, bet in enumerate(ranked_picks):
        side = bet['side']
        side_color = '#00ff88' if side == 'YES' else '#00b4d8'
        # Top pick (highest Trust 🎯) gets a star marker
        marker = '🏆 ' if idx == 0 else f'#{idx + 1} · '
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#052e16 0%,#064e3b 100%);'
            f'border:2px solid #00ff88;border-radius:12px;padding:14px 20px;margin-bottom:10px;">'
            f'<div style="color:#00ff88;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
            f'{marker}{bet["gate_reason"]}</div>'
            f'<div style="color:#ffffff;font-size:20px;font-weight:700;margin-bottom:4px;">'
            f'{bet["city"]} · {bet["label"]} · <span style="color:{side_color};">{side}</span> @ {bet["price"]}c</div>'
            f'<div style="color:#94a3b8;font-size:13px;font-family:\'JetBrains Mono\',monospace;">'
            f'Edge: <span style="color:#00ff88;">+{bet["edge"]}c</span> · '
            f'Kelly: <span style="color:#00ff88;">${bet["kelly"]}</span> · '
            f'Model: {round(bet["model_prob"]*100,1)}% · '
            f'Trust 🎯: <span style="color:#a78bfa;">{bet["trust_accuracy"]}</span> · '
            f'Trust 💎: {bet["trust_edge"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Audit panel
if rejected_cities or len(qualifying_picks) > 1:
    with st.expander(f'📋 V5.27.1 Gate Audit ({len(qualifying_picks)} passing · {len(rejected_cities)} rejected)', expanded=False):
        st.caption('Diagnostic visibility into why each city was selected or skipped. Verify the gate is firing correctly during the 2-week validation window.')
        if qualifying_picks:
            st.markdown('**Cities that passed all three gates:**')
            ranked_for_audit = sorted(qualifying_picks.items(),
                                      key=lambda x: (x[1]['trust_accuracy'], x[1]['edge']), reverse=True)
            for idx, (c, pick) in enumerate(ranked_for_audit):
                marker = '🏆 ' if idx == 0 else f'#{idx+1} '
                st.markdown(
                    f"{marker}**{c}** — {pick['label']} {pick['side']} @ {pick['price']}c · "
                    f"+{pick['edge']}c · ${pick['kelly']} · Trust 🎯 {pick['trust_accuracy']} · {pick['gate_reason']}"
                )
        if rejected_cities:
            st.markdown('**Cities rejected:**')
            for c, reason in sorted(rejected_cities.items()):
                st.markdown(f'- **{c}** — {reason}')


with st.expander('🔬 V5.27.1 Diagnostic Dashboard — Trust the Model?', expanded=False):
    st.caption('This panel measures whether the model deserves your trust today. '
               'It does NOT change any predictions — it tells you when predictions are likely reliable.')

    st.markdown('### 📡 Background Data Collection')
    _today_logged = sb_fetch_all() or []
    _today_logged = [r for r in _today_logged if r.get('date') == today_str]
    _logged_cities = set(r.get('city') for r in _today_logged)
    _missing_cities = [c for c in CITIES.keys() if c not in _logged_cities]

    bcol1, bcol2 = st.columns([1, 2])
    with bcol1:
        if st.button(f'📡 Collect All 18 Cities', key='_v527_collect_all', use_container_width=True):
            try:
                st.cache_data.clear()
            except Exception: pass
            progress = st.progress(0, text='Starting...')
            results = {'logged': [], 'failed': []}
            cities_list = list(CITIES.keys())
            for i, c in enumerate(cities_list):
                try:
                    progress.progress((i + 1) / len(cities_list),
                                      text=f'Fetching {c} ({i+1}/{len(cities_list)})...')
                    weather = fetch_city_weather(c)
                    consensus, ok = save_city_prediction(c, weather, saved_ladders)
                    if ok:
                        results['logged'].append(f'{c} (consensus {consensus}F)')
                    else:
                        results['failed'].append(c)
                except Exception as e:
                    results['failed'].append(f'{c} ({str(e)[:40]})')
            progress.empty()
            if results['logged']:
                st.success(f'✅ Logged predictions for {len(results["logged"])} cities')
                with st.expander('Cities logged', expanded=False):
                    for line in results['logged']:
                        st.markdown(f'- {line}')
            if results['failed']:
                st.warning(f'⚠️ Could not log {len(results["failed"])} cities: ' + ', '.join(results['failed']))
            st.rerun()
    with bcol2:
        if _missing_cities:
            st.warning(f'⚠️ {len(_missing_cities)} of 18 cities have no prediction logged for today: '
                       + ', '.join(_missing_cities[:8])
                       + ('...' if len(_missing_cities) > 8 else ''))
        else:
            st.success(f'✅ All 18 cities have predictions logged for {today_str}')
        st.caption('Press the button each morning to log predictions for all 18 cities.')

    st.markdown('---')
    st.markdown('### 🌡️ Today\'s Regime')
    warm_misses, cold_misses, total_yesterday, regime_msg = compute_regime_indicator()
    if total_yesterday == 0:
        st.info(regime_msg)
    elif '⚠️' in regime_msg:
        st.error(regime_msg + '  Recommendation: reduce stakes 50% or skip until pattern stabilizes.')
    else:
        st.success(regime_msg)
    st.caption(f'Yesterday: {warm_misses} warm misses, {cold_misses} cold misses out of {total_yesterday} settled cities.')

    st.markdown('---')
    st.markdown('### 🎯 Per-City Quality Scores (today)')
    st.caption('Quality measures whether THIS city\'s recent calibration justifies trusting today\'s prediction.')

    import pandas as pd
    quality_rows = []
    for c in CITIES.keys():
        score, tier, reasons = compute_quality_score(c)
        rows_c = sb_fetch_city(c)
        complete_c = [r for r in rows_c if r.get('actual') is not None and r.get('consensus') is not None]
        last_3_errs = []
        if len(complete_c) >= 1:
            last_3 = complete_c[-3:]
            last_3_errs = [r['actual'] - r['consensus'] for r in last_3]
        last_3_str = ', '.join(('+' if e > 0 else '') + f'{e:.1f}' for e in last_3_errs) if last_3_errs else '—'
        if last_3_errs:
            avg_recent = sum(last_3_errs) / len(last_3_errs)
            if avg_recent > 1.5: direction = '🔥 warm'
            elif avg_recent < -1.5: direction = '❄️ cold'
            else: direction = '✅ stable'
        else:
            direction = '—'
        quality_rows.append({
            'Tier': tier, 'City': c, 'Score': score,
            'Last 3 Errors': last_3_str, 'Recent Bias': direction,
            'N': len(complete_c),
        })
    quality_rows_sorted = sorted(quality_rows, key=lambda x: (-x['Score'],))
    st.dataframe(pd.DataFrame(quality_rows_sorted), use_container_width=True, hide_index=True)
    high_count = sum(1 for r in quality_rows if r['Tier'] == '🟢')
    med_count = sum(1 for r in quality_rows if r['Tier'] == '🟡')
    low_count = sum(1 for r in quality_rows if r['Tier'] == '🔴')
    st.caption(f'Today\'s breakdown: {high_count} 🟢 HIGH · {med_count} 🟡 MED · {low_count} 🔴 LOW.')

    st.markdown('---')
    st.markdown('### 📊 Bet Calibration — Does Confidence Predict Wins?')
    st.caption('When you bet at 70%+ market price, did you actually win that often?')

    buckets = compute_calibration_buckets()
    bucket_rows = []
    for label, data in buckets.items():
        n = data['wins'] + data['losses']
        if n == 0:
            win_rate_str = '—'
            calibration = '—'
        else:
            wr = (data['wins'] / n) * 100
            win_rate_str = f'{wr:.0f}% ({data["wins"]}/{n})'
            bucket_mid = {'90-100%': 95, '70-89%': 80, '50-69%': 60, '30-49%': 40, '0-29%': 15}[label]
            diff = wr - bucket_mid
            if abs(diff) < 10: calibration = '✅ Good'
            elif diff < -15: calibration = '🔴 Overconfident'
            elif diff > 15: calibration = '🟢 Underconfident'
            else: calibration = '🟡 Off'
        bucket_rows.append({
            'Confidence Bucket': label, 'Win Rate': win_rate_str,
            'Sample': n, 'Calibration': calibration,
        })
    st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)

view_mode = st.radio('View', ['📊 Mac', '📱 iPhone'], horizontal=True, label_visibility='collapsed')
is_mobile = view_mode == '📱 iPhone'


# ── Timezone Groups ───────────────────────────────────────────────────────────
TIMEZONE_GROUPS = {
    'ET': {
        'cities': ['New York', 'Boston', 'Philadelphia', 'Washington DC', 'Atlanta', 'Miami'],
        'cutoff_et_hour': 14, 'label': '🕙 ET Cities', 'closes': '2:00 PM ET',
    },
    'CT': {
        'cities': ['Chicago', 'Dallas', 'Austin', 'Houston', 'San Antonio', 'New Orleans', 'Oklahoma City', 'Minneapolis'],
        'cutoff_et_hour': 15, 'label': '🕙 CT Cities', 'closes': '3:00 PM ET',
    },
    'MT': {
        'cities': ['Denver'],
        'cutoff_et_hour': 16, 'label': '🕙 MT Cities', 'closes': '4:00 PM ET',
    },
    'PT': {
        'cities': ['Phoenix', 'Las Vegas', 'Los Angeles'],
        'cutoff_et_hour': 17, 'label': '🕙 PT Cities', 'closes': '5:00 PM ET',
    },
}

def get_et_hour():
    return datetime.now(pytz.timezone('America/New_York')).hour

def get_et_time_str():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%I:%M %p ET')

def minutes_until_close(cutoff_et_hour):
    now_et = datetime.now(pytz.timezone('America/New_York'))
    close_et = now_et.replace(hour=cutoff_et_hour, minute=0, second=0, microsecond=0)
    diff = (close_et - now_et).total_seconds() / 60
    return int(diff)

def window_status(cutoff_et_hour):
    mins = minutes_until_close(cutoff_et_hour)
    if mins <= 0: return '🔴 CLOSED', '#ef4444'
    if mins <= 30: return '⚠️ CLOSING SOON', '#f59e0b'
    return '✅ OPEN', '#00ff88'

TIMEZONE_STATIC_INFO = {
    'ET': {'sweet_spot': '9:30–10:30 AM ET', 'accuracy_window': '11:00 AM–12:00 PM ET', 'peak_heat': '2:00–4:00 PM ET'},
    'CT': {'sweet_spot': '10:00–11:00 AM ET', 'accuracy_window': '12:00–1:00 PM ET', 'peak_heat': '3:00–5:00 PM ET'},
    'MT': {'sweet_spot': '12:00–1:00 PM ET', 'accuracy_window': '2:00–3:00 PM ET', 'peak_heat': '4:00–6:00 PM ET'},
    'PT': {'sweet_spot': '11:00 AM–12:00 PM ET', 'accuracy_window': '2:00–3:00 PM ET', 'peak_heat': '5:00–7:00 PM ET'},
}

def get_phase_label(tz_key, et_hour):
    now_et = datetime.now(pytz.timezone('America/New_York'))
    et_hhmm = now_et.hour * 100 + now_et.minute
    phase_times = {
        'ET': {'bet': (930, 1030), 'peak': (1400, 1600)},
        'CT': {'bet': (1000, 1100), 'peak': (1500, 1700)},
        'MT': {'bet': (1200, 1300), 'peak': (1600, 1800)},
        'PT': {'bet': (1100, 1200), 'peak': (1700, 1900)},
    }
    conv_times = {
        'ET': (1100, 1200), 'CT': (1200, 1300),
        'MT': (1400, 1500), 'PT': (1400, 1500),
    }
    times = phase_times.get(tz_key, {})
    conv = conv_times.get(tz_key, (0, 0))
    bet_start, bet_end = times.get('bet', (0, 0))
    peak_start, peak_end = times.get('peak', (0, 0))
    conv_start, conv_end = conv
    if bet_start <= et_hhmm < bet_end: return '💎 EDGE WINDOW', '#00ff88'
    if conv_start <= et_hhmm < conv_end: return '🎯 CONVICTION WINDOW', '#a78bfa'
    if peak_start <= et_hhmm < peak_end: return '🌡️ PEAK HEAT', '#00b4d8'
    if et_hhmm < bet_start: return '⏳ EARLY', '#94a3b8'
    return '', '#64748b'

st.markdown('<div class="mph-section-header">🎯 Best Bets By Timezone Window (V5.27.1 Gated)</div>', unsafe_allow_html=True)
st.caption('Only V5.27.1-qualifying picks shown. Cities not listed did not pass all three gates — see V5.27.1 Gate Audit above.')

_et_hour_now = get_et_hour()

for tz_key, tz_info in TIMEZONE_GROUPS.items():
    visible_cities_in_tz = [c for c in tz_info['cities'] if c not in HIDDEN_CITIES]
    if not visible_cities_in_tz: continue
    status_text, status_color = window_status(tz_info['cutoff_et_hour'])
    mins_left = minutes_until_close(tz_info['cutoff_et_hour'])
    is_closed = mins_left <= 0
    phase_label, phase_color = get_phase_label(tz_key, _et_hour_now)

    tz_picks = []
    for c in visible_cities_in_tz:
        if c in qualifying_picks:
            tz_picks.append(qualifying_picks[c])
    tz_picks.sort(key=lambda p: p['trust_accuracy'], reverse=True)

    city_list_str = ' · '.join(visible_cities_in_tz)
    static = TIMEZONE_STATIC_INFO.get(tz_key, {})
    sweet_spot_str = static.get('sweet_spot', '')
    accuracy_window_str = static.get('accuracy_window', '')
    peak_heat_str = static.get('peak_heat', '')
    phase_html = f'<span style="color:{phase_color}; font-size:12px; font-weight:700; font-family:\'JetBrains Mono\',monospace;">{phase_label}</span>' if phase_label else ''

    if is_closed:
        picks_html = '<div style="color:#ef4444; font-size:12px;">🔴 Window Closed</div>'
    elif not tz_picks:
        picks_html = '<div style="color:#64748b; font-size:12px; font-style:italic;">— No qualifying bet (gates not satisfied)</div>'
    else:
        # Top pick across ALL cities (highest Trust 🎯) gets the trophy
        all_ranked = sorted(qualifying_picks.values(),
                            key=lambda p: (p['trust_accuracy'], p['edge']), reverse=True)
        top_city = all_ranked[0]['city'] if all_ranked else None
        rows = []
        for p in tz_picks:
            is_winner = top_city and p['city'] == top_city
            tag = '🏆 ' if is_winner else '🟢 '
            side_color = '#00ff88' if p['side'] == 'YES' else '#00b4d8'
            rows.append(
                f'<div style="color:#00ff88; font-size:12px; font-family:\'JetBrains Mono\',monospace; margin-bottom:3px;">'
                f'{tag}<strong>{p["city"]}</strong>: {p["label"]} '
                f'<span style="color:{side_color};">{p["side"]}</span> @ {p["price"]}c · '
                f'+{p["edge"]}c · ${p["kelly"]} · Trust 🎯 {p["trust_accuracy"]}'
                f'</div>'
            )
        picks_html = ''.join(rows)

    st.markdown(f"""
<div style="background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; flex-wrap:wrap; gap:6px;">
        <span style="color:#ffffff; font-weight:700; font-size:14px; font-family:'Inter',sans-serif;">{tz_info['label']} — closes {tz_info['closes']}</span>
        <span style="color:{status_color}; font-size:12px; font-family:'JetBrains Mono',monospace;">{status_text}</span>
    </div>
    <div style="margin-bottom:2px;">{phase_html}</div>
    <div style="color:#94a3b8; font-size:11px; margin-bottom:2px; font-family:'JetBrains Mono',monospace;">💎 Edge sweet spot: {sweet_spot_str}</div>
    <div style="color:#a78bfa; font-size:11px; margin-bottom:2px; font-family:'JetBrains Mono',monospace;">🎯 Accuracy sweet spot: {accuracy_window_str}</div>
    <div style="color:#00b4d8; font-size:11px; margin-bottom:8px; font-family:'JetBrains Mono',monospace;">🌡️ Peak heat: {peak_heat_str}</div>
    <div style="color:#64748b; font-size:11px; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">{city_list_str}</div>
    <div>
        <div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;">V5.27.1 Qualifying Picks</div>
        {picks_html}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('---')

city_list = [c for c in CITIES.keys() if c not in HIDDEN_CITIES]
default_idx = city_list.index('New York') if 'New York' in city_list else 0
city = st.selectbox('City', city_list, index=default_idx)

if 'last_city' not in st.session_state: st.session_state.last_city = city
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
    sign = '+' if bias_correction > 0 else ''
    boost_note = ' (1.2× boost active)' if city in NWS_BIAS_BOOST_CITIES else ''
    st.info(f'Bias correction active: {sign}{bias_correction}F applied to consensus '
            f'(model ran {direction} by avg {abs(bias_correction)}F over last {bias_n} days{boost_note})')
elif bias_n > 0:
    st.caption(f'Bias correction: {bias_n} settlement(s) logged — need 3+ for correction')
else:
    st.caption(f'Bias correction: no history yet for {city} — will activate after 3 settled days')

warm_offset = CITY_WARM_OFFSET.get(city, 0.0)
if warm_offset != 0.0:
    sign = '+' if warm_offset > 0 else ''
    st.info(f'🌡️ City offset active: {sign}{warm_offset}F structural adjustment')

if city not in saved_ladders:
    saved_ladders[city] = DEFAULT_LADDERS.get(city, '')

kalshi_markets, fetched_at = get_cached_prices(city)
if kalshi_markets is None:
    with st.spinner('Fetching live Kalshi prices for ' + city + '...'):
        kalshi_markets = fetch_kalshi_brackets(series, retries=3)
        if kalshi_markets:
            save_cached_prices(city, kalshi_markets)
            labels = [normalize_label(m[0]) for m in kalshi_markets]
            while len(labels) < 6: labels.append('')
            saved_ladders[city] = ' | '.join(labels[:6])
            save_json(SAVE_FILE, saved_ladders)
            fetched_at = time.time()

st.markdown('<div class="mph-section-header">📋 Kalshi Ladder</div>', unsafe_allow_html=True)
if kalshi_markets:
    age_min = round((time.time() - fetched_at) / 60) if fetched_at else 0
    age_str = 'just now' if age_min < 1 else str(age_min) + ' min ago'
    st.success('Live prices loaded — ' + str(len(kalshi_markets)) + ' brackets (fetched ' + age_str + ')')
    market_top = get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N)
    for m in kalshi_markets:
        rank_tag = ''
        for i, mt in enumerate(market_top):
            if labels_match(mt, m[0]):
                rank_tag = f' 🏷️ Mkt #{i+1}'
                break
        st.caption(' ' + m[0] + rank_tag + ' | YES: ' + (str(m[1])+'c' if m[1] else 'no price') +
                   ' | NO: ' + (str(m[2])+'c' if m[2] else 'no price'))
else:
    st.warning('Could not fetch live prices from Kalshi. Using saved ladder.')

if st.button('Refresh Prices'):
    clear_city_cache(city)
    try:
        st.cache_data.clear()
    except Exception: pass
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


st.markdown('<div class="mph-section-header">🌤️ Live Weather</div>', unsafe_allow_html=True)

fr_col1, fr_col2 = st.columns([1, 3])
with fr_col1:
    if st.button('🔄 Force Refresh Weather', key=f'force_wx_{city}', use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception: pass
        st.session_state[f'_wx_refresh_count_{city}'] = st.session_state.get(f'_wx_refresh_count_{city}', 0) + 1
        st.rerun()
with fr_col2:
    last_fetch_ts = st.session_state.get(f'_wx_last_fetch_{city}')
    if last_fetch_ts:
        age_sec = int(time.time() - last_fetch_ts)
        if age_sec < 60:
            age_str = f'{age_sec}s ago'
            age_color = '#22c55e'
        elif age_sec < 600:
            age_str = f'{age_sec//60}m {age_sec%60}s ago'
            age_color = '#22c55e'
        elif age_sec < 1800:
            age_str = f'{age_sec//60}m ago'
            age_color = '#eab308'
        else:
            age_str = f'{age_sec//60}m ago ⚠️ STALE'
            age_color = '#ef4444'
        st.markdown(f'<div style="padding:8px 0;color:{age_color};font-size:13px;">Weather last fetched: <strong>{age_str}</strong></div>', unsafe_allow_html=True)

with st.spinner('Fetching weather data...'):
    nws_forecast, _ = fetch_nws_forecast(lat, lon)
    noaa_station, noaa_obs, noaa_source, noaa_obs_ts = fetch_nws_current(lat, lon, station)
    obs_high_raw, six_hr_max, obs_high_url = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)
    st.session_state[f'_wx_last_fetch_{city}'] = time.time()
    _temp_hist_key = f'_temp_hist_{city}_{get_eastern_date()}'
    _temp_hist = st.session_state.get(_temp_hist_key, [])
    if noaa_obs is not None:
        _temp_hist.append({'t': time.time(), 'temp': noaa_obs})
        _temp_hist = _temp_hist[-20:]
        st.session_state[_temp_hist_key] = _temp_hist

nws_trend_key = f'nws_prev_{city}_{get_eastern_date()}'
nws_trend_up = False
nws_trend_delta = None
if nws_forecast is not None:
    prev_nws = st.session_state.get(nws_trend_key)
    if prev_nws is not None and nws_forecast > prev_nws + 1.0:
        nws_trend_up = True
        nws_trend_delta = round(nws_forecast - prev_nws, 1)
    st.session_state[nws_trend_key] = nws_forecast

sanity_warnings = []
obs_high_today = obs_high_raw
obs_high_suspect = False

if obs_high_raw is not None:
    if noaa_obs is not None and obs_high_raw > noaa_obs + 10.0:
        obs_high_today = None; obs_high_suspect = True
        sanity_warnings.append(f'Obs high of {obs_high_raw}F discarded — {round(obs_high_raw - noaa_obs, 1)}F above current temp ({round(noaa_obs, 1)}F). Verify manually before betting.')
    elif nws_forecast is not None and obs_high_raw > nws_forecast + 12.0:
        obs_high_today = None; obs_high_suspect = True
        sanity_warnings.append(f'Obs high of {obs_high_raw}F discarded — {round(obs_high_raw - nws_forecast, 1)}F above NWS forecast ({nws_forecast}F). Verify manually before betting.')

nws_stale = False
if nws_forecast is not None and noaa_obs is not None and noaa_obs > nws_forecast + 5.0:
    nws_stale = True
    sanity_warnings.append(f'NWS forecast ({nws_forecast}F) is {round(noaa_obs - nws_forecast, 1)}F below current temp ({round(noaa_obs, 1)}F) — forecast may be stale.')

ensemble_suspect = False
if ensemble_mean is not None and nws_forecast is not None and abs(ensemble_mean - nws_forecast) > 8.0:
    ensemble_suspect = True
    sanity_warnings.append(f'GFS ensemble ({ensemble_mean}F) differs from NWS by {round(abs(ensemble_mean - nws_forecast), 1)}F — discarded.')
    ensemble_members = None; ensemble_mean = None

high_uncertainty = False; source_gap = None
if nws_forecast is not None and ensemble_mean is not None:
    source_gap = abs(nws_forecast - ensemble_mean)
    threshold = get_uncertainty_threshold(city)
    high_uncertainty = source_gap > threshold

morning_suppressed, morning_warning = check_morning_suppression(obs_high_today, noaa_obs, nws_forecast, local_hour)
conviction_result = check_market_conviction(kalshi_markets, ladder_text)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if nws_forecast:
        st.metric('NWS Forecast', str(nws_forecast)+' F')
        st.caption('Primary — settlement source' + (' (stale?)' if nws_stale else ''))
    else: st.metric('NWS Forecast', 'Unavailable')
with col2:
    if noaa_obs is not None:
        _temp_hist_key = f'_temp_hist_{city}_{get_eastern_date()}'
        _temp_hist = st.session_state.get(_temp_hist_key, [])
        stall_warning = ''
        stall_age = 0
        if len(_temp_hist) >= 3:
            last_3 = _temp_hist[-3:]
            unique_temps = set(round(h['temp'], 1) for h in last_3)
            if len(unique_temps) == 1:
                stall_age = int(time.time() - last_3[0]['t'])
                if stall_age >= 600:
                    stall_warning = f' ⚠️ {stall_age//60}m stall'
        st.metric('Current Temp', str(round(noaa_obs, 1))+' F' + stall_warning)
        last_fetch_ts = st.session_state.get(f'_wx_last_fetch_{city}')
        age_str = ''
        if last_fetch_ts:
            age_sec = int(time.time() - last_fetch_ts)
            if age_sec < 60: age_str = f' · {age_sec}s old'
            else: age_str = f' · {age_sec//60}m old'
        source_badges = {
            'nws_direct':   ('🟢', 'NWS Direct',   'Fastest METAR ingestion (~1-3 min lag)'),
            'iowa_mesonet': ('🟢', 'Iowa Mesonet', 'Cross-check source (~1-5 min lag)'),
            'wethr_obs':    ('🟡', 'Wethr.net',    'Fallback source (~5-15 min lag)'),
            'nws_nearby':   ('🟡', 'NWS Nearby',   'Real obs from nearby NWS station — primary station unavailable'),
        }
        badge = source_badges.get(noaa_source, ('⚪', 'Unknown', ''))
        obs_age_str = ''
        if noaa_obs_ts:
            try:
                ts_clean = noaa_obs_ts.replace('Z', '+00:00')
                obs_dt = datetime.fromisoformat(ts_clean)
                if obs_dt.tzinfo is None:
                    obs_dt = obs_dt.replace(tzinfo=pytz.UTC)
                obs_age_sec = int((datetime.now(pytz.UTC) - obs_dt).total_seconds())
                if obs_age_sec < 60:
                    obs_age_str = f' · obs {obs_age_sec}s ago'
                elif obs_age_sec < 3600:
                    obs_age_str = f' · obs {obs_age_sec//60}m ago'
                else:
                    obs_age_str = f' · obs {obs_age_sec//3600}h{(obs_age_sec%3600)//60}m ago'
            except Exception:
                pass
        st.caption(f'{badge[0]} {badge[1]} · Station: {noaa_station}{age_str}{obs_age_str}')
        if noaa_source == 'nws_nearby':
            st.caption(f'🟡 {badge[2]}')
        elif noaa_source == 'wethr_obs':
            st.caption(f'🟡 {badge[2]}')
        if stall_warning:
            st.caption(f'⚠️ Same temp {stall_age//60} min — source API may be stale (METAR stations update hourly at :51)')
    else: st.metric('Current Temp', 'Unavailable')
with col3:
    if obs_high_today is not None:
        source_label = '✅ Wethr NWS' if obs_high_url == 'wethr_api_nws' else '[NWS table](' + obs_url + ')'
        st.metric('Obs High Today', str(obs_high_today)+' F', delta='floor active')
        wu_url = WUNDERGROUND_URLS.get(city, '')
        st.caption(source_label + (' · [Wunderground ↗](' + wu_url + ')' if wu_url else ''))
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
        gap_str = f' | NWS gap: {round(nws_forecast - ensemble_mean, 1):+.1f}F' if nws_forecast is not None else ''
        st.metric('GFS Ensemble', str(ensemble_mean)+' F', delta=str(n_members)+' members')
        st.caption(f'Weight: {gfs_weight_pct}%{gap_str}')
    elif ensemble_suspect:
        st.metric('GFS Ensemble', 'Discarded'); st.caption('Failed sanity check (>8F from NWS)')
    else:
        st.metric('GFS Ensemble', 'Unavailable')

if nbm_percentiles:
    nbm_p50 = nbm_percentiles.get('p50', nbm_percentiles.get('p25', '—'))
    st.success(f'✅ NBM active — p10:{nbm_percentiles.get("p10","—")}F | p50:{nbm_p50}F | p90:{nbm_percentiles.get("p90","—")}F | bracket probs from real percentile distribution')
else:
    st.caption('📊 NBM unavailable — sigma/normal fallback active')

for w in sanity_warnings: st.error('⚠️ ' + w)

if nws_trend_up and nws_trend_delta is not None:
    st.info(f'📈 NWS forecast trending UP +{nws_trend_delta}F since last fetch — model will boost consensus accordingly.')

if nws_forecast is None:
    st.error('NWS forecast unavailable — cannot run model.')
elif high_uncertainty and source_gap is not None:
    threshold = get_uncertainty_threshold(city)
    st.warning(f'HIGH UNCERTAINTY: NWS ({nws_forecast}F) vs GFS ({ensemble_mean}F) gap = {round(source_gap, 1)}F (threshold: {threshold}F). Green signals suppressed.')
elif source_gap is not None and source_gap > 4.0:
    st.info(f'Source gap: NWS vs Ensemble = {round(source_gap, 1)}F — moderate divergence.')

cold_front_warning = check_cold_front_warning(obs_high_raw, noaa_obs, nws_forecast, local_hour)
if morning_suppressed:
    combined = f'⚠️ Signal suppression active: No obs high + current temp ({round(noaa_obs, 1)}F) is {round(nws_forecast - noaa_obs, 1)}F below NWS forecast ({nws_forecast}F) in morning hours. High may have already occurred — green signals suppressed. Verify manually before betting.'
    if cold_front_warning: combined = cold_front_warning
    st.error(combined)
elif cold_front_warning:
    st.warning(cold_front_warning)
if conviction_result: st.warning(conviction_result[3])

if obs_high_today is not None:
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and obs_high_today > hi + 0.4:
            st.warning('BUST: ' + label + ' eliminated — obs high ' + str(obs_high_today) + 'F exceeds ' + str(hi) + 'F')

with st.expander('Override weather inputs', expanded=False):
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1: override_fc = st.number_input('Forecast High F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_fc')
    with ov2: override_cur = st.number_input('Current Temp F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_cur')
    with ov3: override_noaa = st.number_input('NOAA Obs F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_noaa')
    with ov4: override_obs_high = st.number_input('Obs High Override F', min_value=0.0, max_value=130.0, value=0.0, step=0.5, key='ov_obs')

if override_fc > 0 or override_cur > 0 or override_obs_high > 0:
    st.info('Using manual overrides — set back to 0.0 to use auto values')

forecast = override_fc if override_fc > 0 else nws_forecast
current = override_cur if override_cur > 0 else noaa_obs
noaa_final = override_noaa if override_noaa > 0 else noaa_obs
obs_high_final = override_obs_high if override_obs_high > 0 else obs_high_today


if forecast is not None:
    consensus_raw = compute_consensus(forecast, current if current is not None else forecast, noaa_final, city, obs_high=obs_high_final)
    consensus = round(consensus_raw + bias_correction, 1)
    save_city_prediction(city, {
        'nws_fc': forecast, 'current_temp': current, 'obs_high': obs_high_final,
        'ensemble_mean': ensemble_mean, 'source_gap': source_gap,
        'high_uncertainty': high_uncertainty,
    }, saved_ladders)

    st.markdown('<div class="mph-section-header">🎯 Model Output</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric('Consensus', str(consensus)+'F')
        st.caption('Top of stack')
    with m2:
        call = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)
        st.metric('2° Call', call or '—')
        st.caption('Nearest bracket')
    with m3:
        sigma_val = choose_sigma(city, obs_high=obs_high_final, forecast=forecast)
        st.metric('σ', f'{sigma_val:.2f}')
        st.caption('Bracket spread')
    with m4:
        if obs_high_final is not None:
            floor_val = late_day_floor(forecast, obs_high_final, local_hour, city)
            st.metric('Obs Floor', f'{floor_val:.1f}F')
            st.caption(f'h={local_hour}')
        else:
            st.metric('Obs Floor', '—')
    with m5:
        if bias_correction != 0:
            sign = '+' if bias_correction > 0 else ''
            st.metric('Bias Adj', f'{sign}{bias_correction}F')
            st.caption(f'n={bias_n}')
        else:
            st.metric('Bias Adj', '0.0F')
            st.caption('No history')

    boundary_warnings = check_bracket_boundary(consensus, ladder_text, boundary_threshold=0.5)
    for bw in boundary_warnings:
        st.warning(bw)

    prob_rows, sigma_val, used_nbm = bracket_probs_nbm(consensus, ladder_text, city, nbm_percentiles,
                                                       obs_high=obs_high_final, forecast=forecast)
    prob_rows = apply_prob_floor(prob_rows, consensus, ladder_text)

    if used_nbm:
        st.caption('Probabilities derived from NBM percentile distribution (real probabilistic forecast)')
    else:
        st.caption(f'Probabilities derived from sigma/normal model (σ={sigma_val:.2f}F) — NBM unavailable')

    overall_ens_conf = ensemble_overall_confidence(ensemble_members, consensus, ladder_text) if ensemble_members else ''
    if overall_ens_conf:
        st.caption('Overall ensemble confidence: ' + overall_ens_conf)

    # ── REALITY CHECK panel ──
    reality_warnings = []
    if obs_high_final is not None:
        for label, lo, hi in parse_ladder(ladder_text):
            for plbl, pprob in prob_rows:
                if labels_match(plbl, label) and pprob >= 0.10:
                    if hi is not None and obs_high_final > hi + 0.4:
                        reality_warnings.append(
                            f'⚠️ {label}: model probability {round(pprob*100,1)}% but obs high already {obs_high_final}F (above {hi}F ceiling) — busted bracket'
                        )

    if reality_warnings:
        st.warning('⚠️ Reality Check Warnings:')
        for w in reality_warnings: st.markdown('- ' + w)

    # ── YES + NO SIGNALS TABLES ──
    market_top_n = get_market_top_n(kalshi_markets, n=CONSENSUS_TOP_N) if kalshi_markets else []
    model_pick_label = model_top_pick(prob_rows)

    yes_rows = []
    no_rows = []
    for label, base_prob in prob_rows:
        bracket_lo = bracket_hi = None
        for lbl, lo, hi in parse_ladder(ladder_text):
            if labels_match(lbl, label):
                bracket_lo, bracket_hi = lo, hi
                break
        ens_prob = ensemble_bracket_prob(ensemble_members, bracket_lo, bracket_hi) if ensemble_members else None
        final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)

        # Find Kalshi market
        market_match = None
        if kalshi_markets:
            for m in kalshi_markets:
                if labels_match(m[0], label):
                    market_match = m
                    break
        yes_ask = market_match[1] if market_match else None
        no_ask = market_match[2] if market_match else None

        # Busted bracket
        busted = obs_high_final is not None and bracket_hi is not None and obs_high_final > bracket_hi + 0.4
        # Consensus containment
        contains_consensus = bracket_contains_consensus(label, consensus, ladder_text, tolerance=1.0)
        # Conviction conflict
        conviction_conflict = (conviction_result and is_conflicting_with_conviction(
            label, conviction_result[1], conviction_result[2], ladder_text))

        yes_e = edge_cents(final_prob, yes_ask)
        yes_icon, yes_action = edge_signal(yes_e, high_uncertainty, morning_suppressed, conviction_conflict)
        no_e = no_edge_cents(final_prob, no_ask)
        no_icon, no_action = no_signal(no_e, busted=busted, model_prob=final_prob, no_ask=no_ask,
                                        high_uncertainty=high_uncertainty,
                                        morning_suppressed=morning_suppressed,
                                        conviction_conflict=conviction_conflict)

        # Tighten green signal: require 30%+ model prob, HIGH ensemble for confidence, must contain consensus
        ens_conf_str = ensemble_confidence(ens_prob) if ens_prob is not None else ''
        ens_tier = ensemble_tier_from_confidence(ens_conf_str)
        if yes_icon == '🟢':
            if final_prob < 0.30 or 'HIGH' not in ens_conf_str or not contains_consensus:
                yes_icon = '🟡'
                yes_action = 'SKIP (gate)'

        # V5.27 GATE: bracket must be Model's #1 pick AND in market top 2
        in_market_top_n = any(labels_match(mt, label) for mt in market_top_n)
        is_model_pick = labels_match(label, model_pick_label) if model_pick_label else False
        passes_v527_gate1 = is_model_pick and in_market_top_n

        # V5.27 GATE 2: Price floor
        yes_passes_floor = yes_ask is not None and yes_ask >= PRICE_FLOOR_CENTS
        no_passes_floor = no_ask is not None and no_ask >= PRICE_FLOOR_CENTS

        # Tags for the table
        yes_v527_tag = '🚦' if passes_v527_gate1 and yes_passes_floor and yes_icon == '🟢' else ('🚫' if not passes_v527_gate1 else '')
        no_v527_tag = '🚦' if passes_v527_gate1 and (no_passes_floor or (busted and no_ask is not None and no_ask <= 5)) and no_icon == '🟢' else ('🚫' if not passes_v527_gate1 else '')

        # Trust scores
        call_str = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)
        _, mae_color = get_city_mae_and_color(city)
        trust_yes = compute_row_trust(
            city=city, bracket_label=label, direction='YES',
            model_pct=final_prob * 100, ensemble_tier=ens_tier,
            two_degree_call_str=call_str or '', mae_color=mae_color,
            nbm_active=used_nbm, nws_forecast_f=forecast,
            gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
        )
        edge_trust_yes = compute_edge_trust(model_pct=final_prob * 100, yes_ask=yes_ask, ensemble_tier=ens_tier)

        trust_no = compute_row_trust(
            city=city, bracket_label=label, direction='NO',
            model_pct=(1.0 - final_prob) * 100, ensemble_tier=ens_tier,
            two_degree_call_str=call_str or '', mae_color=mae_color,
            nbm_active=used_nbm, nws_forecast_f=forecast,
            gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction,
        )
        edge_trust_no = compute_edge_trust(model_pct=(1.0 - final_prob) * 100, yes_ask=no_ask, ensemble_tier=ens_tier)

        kelly = kelly_bet(final_prob, yes_ask, bankroll)
        kelly_no = kelly_bet_no(final_prob, no_ask, bankroll)

        # YES row
        yes_rows.append({
            'V5.27': yes_v527_tag,
            'Signal': yes_icon,
            'Bracket': label,
            'Model %': str(round(final_prob*100, 1)) + '%',
            'YES Ask': (str(yes_ask) + 'c') if yes_ask is not None else '—',
            'Edge': (('+' if yes_e and yes_e > 0 else '') + str(yes_e) + 'c') if yes_e is not None else '—',
            'Trust 💎': str(round(edge_trust_yes, 1)) if edge_trust_yes else '—',
            'Trust 🎯': str(round(trust_yes.composite, 1)) if trust_yes else '—',
            'Ens.': ens_conf_str or '—',
            'Kelly $': '$' + str(kelly) if kelly > 0 else '—',
            'Action': yes_action,
        })
        # NO row — CRITICAL: Model% must show (1.0 - final_prob), not final_prob
        no_rows.append({
            'V5.27': no_v527_tag,
            'Signal': no_icon,
            'Bracket': label,
            'Model %': str(round((1.0 - final_prob)*100, 1)) + '%',
            'NO Ask': (str(no_ask) + 'c') if no_ask is not None else '—',
            'Edge': (('+' if no_e and no_e > 0 else '') + str(no_e) + 'c') if no_e is not None else '—',
            'Trust 💎': str(round(edge_trust_no, 1)) if edge_trust_no else '—',
            'Trust 🎯': str(round(trust_no.composite, 1)) if trust_no else '—',
            'Ens.': ens_conf_str or '—',
            'Kelly $': '$' + str(kelly_no) if kelly_no > 0 else '—',
            'Action': no_action,
        })

    import pandas as pd
    st.markdown('<div class="mph-section-header">📈 YES Signals</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(yes_rows), use_container_width=True, hide_index=True)
    st.markdown('<div class="mph-section-header">📉 NO Signals</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(no_rows), use_container_width=True, hide_index=True)
    st.caption('🚦 = passes all three V5.27.1 gates · 🚫 = blocked by Gate 1 (not model+market consensus)')

    # ── V5.27.1 Per-City Gate Status ──
    city_pick = qualifying_picks.get(city)
    if city_pick:
        side_color = '#00ff88' if city_pick['side'] == 'YES' else '#00b4d8'
        st.success(
            f'✅ V5.27.1 GATE STATUS: PASS — {city_pick["label"]} {city_pick["side"]} @ {city_pick["price"]}c · '
            f'+{city_pick["edge"]}c · Kelly ${city_pick["kelly"]} · Trust 🎯 {city_pick["trust_accuracy"]} · {city_pick["gate_reason"]}'
        )
    else:
        rejected_reason = rejected_cities.get(city, 'Did not pass three-gate filter')
        st.warning(f'⏸️ V5.27.1 GATE STATUS: REJECTED — {rejected_reason}')


# ── Per-City Quality Score Panel ──
st.markdown('<div class="mph-section-header">🎯 Quality Score — Trust This City Today?</div>', unsafe_allow_html=True)
q_score, q_tier, q_reasons = compute_quality_score(city)
qc1, qc2 = st.columns([1, 2])
with qc1:
    if q_tier == '🟢':
        st.markdown(f'<div style="background:#052e16;border:2px solid #00ff88;border-radius:10px;padding:18px;text-align:center;">'
                    f'<div style="color:#00ff88;font-size:36px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{q_score}/100</div>'
                    f'<div style="color:#00ff88;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">🟢 HIGH QUALITY</div>'
                    f'</div>', unsafe_allow_html=True)
    elif q_tier == '🟡':
        st.markdown(f'<div style="background:#1a1500;border:2px solid #f59e0b;border-radius:10px;padding:18px;text-align:center;">'
                    f'<div style="color:#f59e0b;font-size:36px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{q_score}/100</div>'
                    f'<div style="color:#f59e0b;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">🟡 MED QUALITY</div>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#2d0a0a;border:2px solid #ef4444;border-radius:10px;padding:18px;text-align:center;">'
                    f'<div style="color:#ef4444;font-size:36px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{q_score}/100</div>'
                    f'<div style="color:#ef4444;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">🔴 LOW QUALITY</div>'
                    f'</div>', unsafe_allow_html=True)
with qc2:
    st.markdown('**Why this score?**')
    for r in q_reasons:
        st.markdown(f'- {r}')


# ── Calibration & Settlement History ──
with st.expander('📊 Calibration & Settlement History', expanded=False):
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if not complete:
        st.info(f'No settled predictions yet for {city}.')
    else:
        recent = complete[-20:]
        recent_rows = []
        for r in recent:
            err = r['actual'] - r['consensus']
            err_str = ('+' if err > 0 else '') + f'{err:.1f}F'
            recent_rows.append({
                'Date': r.get('date', '—'),
                'Consensus': str(r.get('consensus', '—')) + 'F',
                'Actual': str(r.get('actual', '—')) + 'F',
                'Error': err_str,
                'Bias': str(r.get('bias_correction', 0)) + 'F',
            })
        st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
        all_errors = [r['actual'] - r['consensus'] for r in complete]
        mae_all = sum(abs(e) for e in all_errors) / len(all_errors)
        mean_err = sum(all_errors) / len(all_errors)
        st.caption(f'All-time: MAE {mae_all:.2f}F · Mean error {mean_err:+.2f}F · N={len(complete)}')


# ── Source Accuracy Report ──
with st.expander('🔬 Source Accuracy Report', expanded=False):
    st.caption('How each source (NWS forecast, GFS ensemble, consensus) has scored over the recent settlements for this city.')
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None]
    if len(complete) < 3:
        st.info('Need 3+ settled predictions for source accuracy report.')
    else:
        recent = complete[-14:]
        nws_errors = [r['actual'] - r['forecast'] for r in recent if r.get('forecast') is not None]
        gfs_errors = [r['actual'] - r['ensemble_mean'] for r in recent if r.get('ensemble_mean') is not None]
        consensus_errors = [r['actual'] - r['consensus'] for r in recent if r.get('consensus') is not None]
        def stats(errs):
            if not errs:
                return '—', '—', '—'
            mae = sum(abs(e) for e in errs) / len(errs)
            mean = sum(errs) / len(errs)
            return f'{mae:.2f}F', ('+' if mean > 0 else '') + f'{mean:.2f}F', str(len(errs))
        nws_mae, nws_mean, nws_n = stats(nws_errors)
        gfs_mae, gfs_mean, gfs_n = stats(gfs_errors)
        con_mae, con_mean, con_n = stats(consensus_errors)
        st.markdown(f"""
| Source | MAE | Mean Error | N |
|---|---|---|---|
| **Consensus (model)** | {con_mae} | {con_mean} | {con_n} |
| NWS Forecast | {nws_mae} | {nws_mean} | {nws_n} |
| GFS Ensemble | {gfs_mae} | {gfs_mean} | {gfs_n} |
""")
        if consensus_errors and nws_errors and gfs_errors:
            best_mae = min(con_mae, nws_mae, gfs_mae)
            if con_mae == best_mae:
                st.success(f'✅ Consensus is the lowest-MAE source for {city} ({con_mae}).')
            else:
                st.warning(f'⚠️ Consensus is not the best source for {city}. Best: '
                           f'NWS ({nws_mae}) or GFS ({gfs_mae}). Consider per-city routing review.')


# ── Personal Bet Log (password-gated) ──
st.markdown('---')
st.markdown('<div class="mph-section-header">📋 Personal Bet Log</div>', unsafe_allow_html=True)

_BET_LOG_PASSWORD = None
try:
    _BET_LOG_PASSWORD = st.secrets.get('bet_log_password', None)
except Exception:
    _BET_LOG_PASSWORD = None

if _BET_LOG_PASSWORD:
    if not st.session_state.get('_bet_log_authed'):
        with st.expander('🔒 Bet Log — unlock', expanded=False):
            bp = st.text_input('Bet log password', type='password', key='_bet_log_pw')
            if bp:
                if bp == _BET_LOG_PASSWORD:
                    st.session_state['_bet_log_authed'] = True
                    st.rerun()
                else:
                    st.error('Incorrect password.')
        bet_log_unlocked = False
    else:
        bet_log_unlocked = True
else:
    bet_log_unlocked = True

if bet_log_unlocked:
    bet_log = load_bet_log()
    pending = [b for b in bet_log if b.get('result') == 'Pending']
    won = [b for b in bet_log if b.get('result') == 'Won']
    lost = [b for b in bet_log if b.get('result') == 'Lost']

    total_staked = sum(float(b.get('amount', 0) or 0) for b in (won + lost))
    total_pnl = sum(float(b.get('profit', 0) or 0) for b in (won + lost))
    win_rate = (len(won) / (len(won) + len(lost)) * 100) if (won or lost) else 0
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1: st.metric('Bets Settled', f'{len(won) + len(lost)}')
    with bc2: st.metric('Win Rate', f'{win_rate:.1f}%')
    with bc3:
        pnl_color = '#00ff88' if total_pnl >= 0 else '#ef4444'
        st.markdown(f'<div style="padding:12px;background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;">'
                    f'<div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;">Total P&L</div>'
                    f'<div style="color:{pnl_color};font-size:20px;font-family:\'JetBrains Mono\',monospace;font-weight:700;">'
                    f'{"+" if total_pnl >= 0 else ""}${total_pnl:.2f}</div></div>',
                    unsafe_allow_html=True)
    with bc4: st.metric('ROI', f'{roi:+.1f}%')

    st.caption(f'Pending: {len(pending)} · Won: {len(won)} · Lost: {len(lost)}')

    with st.expander('➕ Log a new bet', expanded=False):
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            log_city = st.selectbox('City', list(CITIES.keys()), key='_log_city')
            log_date = st.date_input('Date', value=datetime.now(pytz.timezone('America/New_York')).date(), key='_log_date')
        with bcol2:
            log_bracket = st.text_input('Bracket', placeholder='e.g. 79-80 or 81 or above', key='_log_bracket')
            log_direction = st.selectbox('Direction', ['YES', 'NO'], key='_log_dir')
        with bcol3:
            log_price = st.number_input('Price (¢)', min_value=1, max_value=99, value=25, key='_log_price')
            log_amount = st.number_input('Amount ($)', min_value=0.5, max_value=1000.0, value=5.0, step=0.5, key='_log_amount')
        if st.button('💾 Log Bet', type='primary'):
            if not log_bracket.strip():
                st.error('Bracket required.')
            else:
                new_bet = {
                    'date': str(log_date), 'city': log_city,
                    'bracket': normalize_label(log_bracket.strip()),
                    'direction': log_direction, 'price': log_price,
                    'amount': float(log_amount), 'result': 'Pending',
                    'profit': 0.0, 'payout': 0.0, 'actual': None,
                    'logged_at': datetime.now(pytz.timezone('America/New_York')).isoformat(),
                }
                inserted = sb_insert_bet(new_bet)
                if inserted:
                    st.success(f'✅ Logged: {log_city} {log_bracket} {log_direction} @ {log_price}c ${log_amount}')
                    st.rerun()
                else:
                    st.error('Failed to log bet.')

    if bet_log:
        st.markdown('**Recent bets:**')
        recent_bets = list(reversed(bet_log[-50:]))
        bet_rows = []
        for b in recent_bets:
            result = b.get('result', 'Pending')
            if result == 'Won': result_icon = '✅'
            elif result == 'Lost': result_icon = '❌'
            else: result_icon = '⏳'
            profit = b.get('profit', 0) or 0
            pnl_str = ('+' if profit > 0 else '') + f'${profit:.2f}' if result != 'Pending' else '—'
            bet_rows.append({
                'Date': b.get('date', '—'),
                'City': b.get('city', '—'),
                'Bracket': b.get('bracket', '—'),
                'Dir': b.get('direction', '—'),
                'Price': str(b.get('price', '—')) + 'c',
                'Amount': '$' + f'{float(b.get("amount", 0)):.2f}',
                'Actual': (str(b.get('actual')) + 'F') if b.get('actual') is not None else '—',
                'Result': result_icon + ' ' + result,
                'P&L': pnl_str,
            })
        st.dataframe(pd.DataFrame(bet_rows), use_container_width=True, hide_index=True)

        with st.expander('✏️ Edit / Delete a Bet', expanded=False):
            bet_ids = [str(b['id']) for b in bet_log if b.get('id')]
            if bet_ids:
                sel_id = st.selectbox('Select bet by ID', bet_ids, key='_edit_bet_id')
                sel_bet = next((b for b in bet_log if str(b.get('id')) == sel_id), None)
                if sel_bet:
                    st.caption(f"{sel_bet.get('date')} · {sel_bet.get('city')} · {sel_bet.get('bracket')} · "
                               f"{sel_bet.get('direction')} @ {sel_bet.get('price')}c · ${sel_bet.get('amount')}")
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1: new_price = st.number_input('Price', min_value=1, max_value=99, value=int(sel_bet.get('price', 25)), key='_edit_price')
                    with ec2: new_amount = st.number_input('Amount', min_value=0.5, max_value=1000.0, value=float(sel_bet.get('amount', 5.0)), step=0.5, key='_edit_amount')
                    with ec3: new_result = st.selectbox('Result', ['Pending', 'Won', 'Lost'],
                                                        index=['Pending', 'Won', 'Lost'].index(sel_bet.get('result', 'Pending')),
                                                        key='_edit_result')
                    ucol, dcol = st.columns(2)
                    with ucol:
                        if st.button('💾 Update', key='_update_bet'):
                            updates = {'price': new_price, 'amount': float(new_amount), 'result': new_result}
                            if sb_update_bet(int(sel_id), updates):
                                st.success('Updated.')
                                st.rerun()
                            else:
                                st.error('Update failed.')
                    with dcol:
                        if st.button('🗑️ Delete', key='_delete_bet'):
                            if sb_delete_bet(int(sel_id)):
                                st.success('Deleted.')
                                st.rerun()
                            else:
                                st.error('Delete failed.')

st.markdown('---')
st.caption('🌡️ MPH Weather Model V5.27.1 — Three research-validated gates active')
