# Kalshi High Temperature Model - V5.17
#
# Philosophy shift from V5.16 → V5.17:
# V5.16 was EDGE-FIRST: best bet = biggest mispricing vs market, even if
# bracket was 2-3 away from consensus. In practice this meant model picking
# low-probability long-shots with huge edges (like 78-79 YES when consensus
# was 80F) and losing when the actual high landed closer to consensus.
#
# V5.17 is ACCURACY-FIRST: best bet = model's honest prediction for which
# bracket will actually hit, with edge shown as secondary info. Goal: 60%+
# win rate on safer picks, 3-4 bets/day, 5-6 days/week.
#
# Changes from V5.16:
# 1. MINIMUM PROBABILITY FLOOR — fixes the 0.0% bug on brackets adjacent to
#    consensus. Every bracket within 4F of consensus gets ≥ 5% probability;
#    brackets touching the ladder get ≥ 2%. Stops model from declaring
#    outcomes "impossible" when they're actually 10-20% likely (e.g., today's
#    Miami 82-83 showing 0.0% when it was clearly in play).
#
# 2. DUAL-DOT SIGNAL SYSTEM:
#    - 🟢🔵 accuracy pick: highest Trust + Model% ≥ 35% + Trust ≥ 75 + HIGH
#      ensemble. "Model's honest pick for which bracket will actually hit."
#    - 🟣 edge pick: highest edge bracket per city. "Biggest mispricing —
#      use as sanity check / secondary info."
#    - When both point to the same bracket → strongest signal.
#    - When they differ → user uses judgment to decide.
#
# 3. SUMMARY PANEL AT TOP — one-line-per-city panel showing Model's best
#    guess before you scroll to detailed tables. "New Orleans: 82-83 (Trust
#    88, prob 42%) · Market agrees".
#
# 4. ACCURACY SWEET SPOTS — new time windows added under existing edge sweet
#    spots in each timezone banner. Accuracy sweet spot = ~2-3 hours before
#    typical peak heat for each city. That's when obs data is clearest.
#    Edge sweet spot stays as-is (for people hunting mispricings early).
#
# 5. REALITY-CHECK BANNER — shows when current temp > consensus - 3F, or
#    when NWS forecast is 3F+ above consensus. Catches days like today's
#    Dallas where model said 83F but current obs was already 89F.
#
# 6. NO-BET HIGHLIGHT SECTION — new callout per timezone window listing
#    market-overpriced brackets. Market % significantly exceeds model %
#    → potential NO bet opportunity. Labeled "🔻 Market-overpriced brackets".
#
# 7. TIER 3 HIDDEN — Philadelphia, Boston, San Antonio moved to HIDDEN_CITIES
#    (MAE 2.8-3.1F, not good enough for 60% win rate target). Now 10 visible
#    cities instead of 13. All 18 still predict in background for accuracy
#    report and potential recovery tracking.
#
# 8. IPHONE CLEANUP — when iPhone toggle is active, less-critical elements
#    hide to reduce visual clutter on small screens. Mac/iPhone toggle stays.
#
# All V5.16 logic preserved:
# - Per-city prediction mode (full_blend vs nws_only)
# - Time-gated obs_high (only after 1 PM local, sanity-checked)
# - Bias correction improvements (median, ±3F cap, reduced for unstable)
# - Supabase bet log with Edit/Delete, retroactive settlement
# - App password gate (st.secrets['app_password'])
# - Source Accuracy Report tab
# - Trust score module (trust_score.py, weights unchanged)
# - V5.13 security, V5.12 auto-settlement, V5.11 signal suppression

# ── V5.16.1 hotfix note ──
# V5.16.1 hotfix: Best Bets by Timezone banner was still showing hidden cities
# in the header city list (e.g., "Chicago · Dallas · Austin · Houston" with
# Chicago and Austin being hidden). Also the MT section showed up with just
# "Denver" as the only city — Denver is hidden so the whole MT section should
# disappear. Both bugs fixed.
#
# ── V5.16 original changelog ──
# Major changes from V5.15 (DATA-DRIVEN REDESIGN based on 21-24 days of real
# settlement data — MAE tier analysis):
#
# 1. PER-CITY PREDICTION MODE: New CITY_PREDICTION_MODE dict routes each city
#    through either 'full_blend' (current consensus formula) or 'nws_only'
#    (simpler NWS forecast + bias correction). Based on empirical data showing
#    that the blend HELPS NYC (-3.67F MAE vs NWS-alone) but HURTS Minneapolis
#    (+1.57F MAE worse than NWS-alone) and is roughly neutral for 8 other cities.
#
# 2. TIME-GATED OBS_HIGH: obs_high can only override consensus after 1 PM local
#    time (with sanity checks: obs_high must be ≥ current_temp and ≤ current_temp
#    + 10F). Morning obs_high readings are ADVISORY ONLY. This fixes the sensor
#    spike bug that poisoned Chicago, Minneapolis, Boston, Austin predictions.
#
# 3. HIDDEN CITIES: 5 low-performing cities (Minneapolis, Denver, Chicago, Los
#    Angeles, Austin — all with MAE > 3.0) are hidden from the main city
#    selector AND dashboard. They still run in the accuracy report so we can
#    track if V5.16 fixes recover them.
#
# 4. SOURCE ACCURACY REPORT TAB: New dashboard section showing per-city MAE
#    comparison (NWS-alone vs GFS-alone vs our blend). Auto-regenerates every
#    time so you can watch V5.16 changes take effect over 1-2 weeks.
#
# 5. TWO-DOT UI (reverted from V5.15 Verdict column): Back to separate Signal
#    column (🟢🟡⚪🔴) on the left and Ensemble column (🔵🟡⚪) on the right,
#    with Trust column between them. Matches what the user found readable.
#
# 6. SIMPLIFIED TRUST SCORE (3 factors instead of 7):
#    - Source agreement (40%): do NWS, GFS, NBM, consensus all agree on bracket?
#    - Bracket proximity to 2DC + obs trajectory (40%)
#    - City historical MAE (20%)
#
# 7. BIAS CORRECTION IMPROVEMENTS: Switch from mean to median (ignores outliers),
#    cap at ±3.0F, reduce weight 50% for cities with MAE > 4.0.
#
# 8. SETTLE_BET_LOG RETROACTIVE FIX: Bug where pending bets logged AFTER
#    auto-settlement ran would never settle. Now runs against ALL pending bets
#    with matching actual values, not just newly-settled rows.
#
# 9. APP PASSWORD GATE: New branded login screen at app entry. Password read
#    from st.secrets['app_password']. Keeps bet log password (2974) separate.
#
# All V5.15 logic retained:
# - Bet log in Supabase with Edit/Delete/Logged confirmation
# - V5.14 trust_score.py module (weights adjusted in app code, not module)
# - V5.13 security (st.secrets)
# - V5.12 auto-settlement
# - V5.11 signal suppression, MAE flags, gap warnings

import math, re, json, time, requests
import streamlit as st
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# V5.14: trust score module (file lives in repo root as trust_score.py)
from trust_score import SignalInputs, compute_trust_score, bracket_midpoint_from_label

st.set_page_config(page_title='MPH Weather Model', layout='wide', page_icon='🌡️')

# ══════════════════════════════════════════════════════════════════════════════
# V5.16 APP PASSWORD GATE (branded)
# ══════════════════════════════════════════════════════════════════════════════
# Read the app password from Streamlit secrets. Set it in your app's
# Streamlit Cloud Settings -> Secrets as:
#   app_password = "your-password-here"
# If the secret is missing, the gate is disabled (app opens normally) so a
# deploy without the secret set doesn't brick the app.
def _check_app_password():
    try:
        correct_pw = st.secrets.get('app_password', None)
    except Exception:
        correct_pw = None
    # If no password configured, skip the gate entirely
    if not correct_pw:
        return True
    # Already authenticated this session
    if st.session_state.get('_app_authed') is True:
        return True
    # Show branded login screen
    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .mph-login-wrap {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 70vh;
    }
    .mph-login-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .mph-login-sub {
        color: #888; font-size: 0.95rem; margin-bottom: 2rem;
    }
    .mph-login-badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        background: rgba(0, 255, 136, 0.12);
        border: 1px solid rgba(0, 255, 136, 0.4);
        border-radius: 999px; color: #00ff88; font-size: 0.75rem;
        font-weight: 600; letter-spacing: 0.05em; margin-left: 0.5rem;
    }
    </style>
    <div class="mph-login-wrap">
      <div class="mph-login-title">🌡️ MPH Weather Model <span class="mph-login-badge">V5.16</span></div>
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

# ── Custom CSS — V5.8 Trading Terminal Style ──────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main background ── */
.stApp {
    background: #0a0e1a;
}

/* ── Hero header ── */
.mph-hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.mph-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ff88, #00b4d8, #00ff88);
}
.mph-hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
    font-family: 'Inter', sans-serif;
}
.mph-hero-sub {
    font-size: 13px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
}
.mph-version-badge {
    display: inline-block;
    background: #00ff8820;
    border: 1px solid #00ff8840;
    color: #00ff88;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 10px;
    vertical-align: middle;
}
.mph-live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #00ff88;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}

/* ── Stats bar ── */
.mph-stats-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.mph-stat {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 18px;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.mph-stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    display: block;
    line-height: 1.2;
}
.mph-stat-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: block;
    margin-top: 4px;
}
.mph-stat-warn .mph-stat-value { color: #f59e0b; }
.mph-stat-alert .mph-stat-value { color: #ef4444; }
.mph-stat-neutral .mph-stat-value { color: #94a3b8; }

/* ── Section headers ── */
.mph-section-header {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 0 0 8px 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 16px;
    font-family: 'Inter', sans-serif;
}

/* ── Metric cards ── */
.stMetric {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.stMetric label {
    color: #64748b !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e3a5f !important;
    color: #00ff88 !important;
    border: 1px solid #00ff8840 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00ff8820 !important;
    border-color: #00ff88 !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #ffffff !important;
}

/* ── Success/warning/info/error boxes ── */
.stSuccess {
    background: #00ff8810 !important;
    border: 1px solid #00ff8840 !important;
    border-radius: 6px !important;
    color: #00ff88 !important;
}
.stWarning {
    background: #f59e0b10 !important;
    border: 1px solid #f59e0b40 !important;
    border-radius: 6px !important;
}
.stInfo {
    background: #00b4d810 !important;
    border: 1px solid #00b4d840 !important;
    border-radius: 6px !important;
}
.stError {
    background: #ef444410 !important;
    border: 1px solid #ef444440 !important;
    border-radius: 6px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #94a3b8 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Mobile responsive ── */
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
HEADERS = {'User-Agent': 'kalshi-temp-model/5.7', 'Accept': 'application/geo+json, application/json, text/html'}
# V5.13: read Wethr API key from Streamlit secrets (no longer hardcoded)
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

# V5.6: Northeast/coastal GFS weights reduced 0.25 → 0.15 (GFS runs cold in spring)
GFS_CITY_WEIGHT = {
    'Phoenix': 0.10, 'Las Vegas': 0.10, 'Los Angeles': 0.12,
    'Miami': 0.18, 'Houston': 0.18, 'New Orleans': 0.18,
    'Dallas': 0.25, 'Austin': 0.25, 'San Antonio': 0.25, 'Oklahoma City': 0.25,
    'Atlanta': 0.22, 'Denver': 0.22, 'Minneapolis': 0.22, 'Chicago': 0.22,
    # V5.6: reduced from 0.25 → 0.15 for northeast/coastal cities
    'New York': 0.15, 'Philadelphia': 0.15, 'Boston': 0.15, 'Washington DC': 0.15,
}

# V5.6: Cities that get a wider high-uncertainty threshold in spring (months 3-5)
# GFS spread is routinely wider for these cities in spring, 5F threshold too tight
SPRING_WIDE_THRESHOLD_CITIES = {'New York', 'Philadelphia', 'Boston', 'Washington DC', 'Los Angeles'}

# V5.7: Northeast cities get more aggressive late-day floor multipliers
NORTHEAST_CITIES = {'New York', 'Philadelphia', 'Boston', 'Washington DC'}

# V5.7: Regional prior bias — used for cities with < 3 settlements
# Chicago inherits Minneapolis prior until it has its own data
REGIONAL_PRIOR_BIAS = {
    'Chicago': 'Minneapolis',
}

# ══════════════════════════════════════════════════════════════════════════════
# V5.16 CITY TIERS (data-driven, based on 21-24 days of real settlement MAE)
# ══════════════════════════════════════════════════════════════════════════════
# Run this SQL to refresh the tiers:
#   SELECT city, ROUND(AVG(ABS(error))::numeric, 2) as mae FROM settlements
#   WHERE actual IS NOT NULL GROUP BY city ORDER BY mae;
# Then update these sets accordingly every few weeks.

# HIDDEN cities: MAE > ~2.7 AND either blend hurting or unstable.
# These are NOT removed from the predictions pipeline (we keep collecting data
# to see if V5.17 changes recover them). They are hidden from:
#   - Main city selector dropdown
#   - "All Cities Today's Predictions" dashboard
#   - "Best Bets by Timezone" callouts
#   - Summary panel at top
# They ARE visible in:
#   - Source Accuracy Report tab (so user can watch recovery)
#
# V5.17: expanded from 5 to 8 hidden cities (added Philadelphia, Boston,
# San Antonio) to focus main app on 10 high-accuracy cities for 60%+ win
# rate target.
HIDDEN_CITIES = {
    # V5.16 original hidden set (MAE > 3.0, blend hurting or unstable)
    'Minneapolis', 'Denver', 'Chicago', 'Los Angeles', 'Austin',
    # V5.17 additions (Tier 3 — MAE 2.8-3.1, not good enough for 60% target)
    'Philadelphia', 'Boston', 'San Antonio',
}

# CITY_PREDICTION_MODE: per-city pipeline routing based on empirical data.
# - 'full_blend' = current V5.15 consensus formula (fc/cur/noaa time-weighted
#                  + obs_high override + late_day_floor + bias)
# - 'nws_only'   = simpler: consensus = NWS forecast + bias_correction
#
# Rule: use 'full_blend' when blend beats NWS-alone by ≥0.3F MAE,
#       use 'nws_only'  when NWS-alone matches or beats blend.
CITY_PREDICTION_MODE = {
    # Blend is clearly helpful (lower MAE than NWS-only by 0.3F+)
    'New York':     'full_blend',  # blend -3.67F vs NWS (huge)
    'Houston':      'full_blend',  # blend -0.99F vs NWS
    'Dallas':       'full_blend',  # blend -0.83F vs NWS
    'Miami':        'full_blend',  # blend -0.58F vs NWS
    'Phoenix':      'full_blend',  # blend -0.55F vs NWS
    'Las Vegas':    'full_blend',  # blend -0.29F vs NWS (borderline, kept)
    'Boston':       'full_blend',  # blend -0.38F vs NWS
    'Philadelphia': 'full_blend',  # blend -0.18F vs NWS
    'San Antonio':  'full_blend',  # blend -0.19F vs NWS
    # NWS-only wins or ties — skip the blend entirely
    'New Orleans':   'nws_only',   # NWS 1.17 vs blend 1.23 (tied, NWS simpler)
    'Washington DC': 'nws_only',   # tied
    'Atlanta':       'nws_only',   # NWS +0.12F better
    'Oklahoma City': 'nws_only',   # NWS +0.16F better
    # Hidden cities — still predicted for data collection, mode chosen to help
    'Chicago':       'nws_only',   # NWS +0.15F better
    'Denver':        'nws_only',   # tied (blend marginally worse)
    'Austin':        'nws_only',   # blend worse by 0.40F — needs NWS-only
    'Minneapolis':   'nws_only',   # 🚨 blend worse by 1.57F — biggest fix
    'Los Angeles':   'full_blend', # blend better by 0.91F but still 3.14 MAE
}

# V5.16: time gate for obs_high override — only trust observed daily high
# after 1 PM local. Before that, obs_high is advisory only (shown but doesn't
# drive consensus). This prevents ASOS morning sensor spikes from hijacking
# the model.
OBS_HIGH_TRUST_HOUR = 13  # local hour cutoff
OBS_HIGH_MAX_OVERSHOOT = 10.0  # obs_high can't exceed current_temp by more than this

# ── Supabase ──────────────────────────────────────────────────────────────────
# V5.13: read Supabase credentials from Streamlit secrets (no longer hardcoded)
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

def sb_fetch_all():
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'order': 'date.asc', 'limit': '1000'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_fetch_city(city):
    try:
        r = requests.get(sb_url('settlements'), headers=get_sb_headers(),
                         params={'city': 'eq.' + city, 'order': 'date.asc', 'limit': '200'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

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

# ── V5.2 Auto-Settlement — Iowa State CLI JSON API ────────────────────────────
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

# ── Bias Correction ───────────────────────────────────────────────────────────
# V5.6: increased n_recent from 10 → 14 for more stable correction
# V5.7: regional prior bias for cities with < 3 settlements
def compute_bias_correction_db(city, n_recent=14):
    """
    V5.16: Changed from mean to median (ignores outlier days), tightened cap
    from ±5F to ±3F, and reduced weight 50% for unstable cities (MAE > 4F).
    This prevents a few catastrophic days from dragging the correction in a
    misleading direction (classic Minneapolis problem).
    """
    import statistics
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('consensus') is not None]
    if len(complete) < 3:
        # V5.7: use regional prior if available
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
    # V5.16: use median instead of mean — more robust to outlier days
    med_error = statistics.median(errors)
    # V5.16: reduce weight for unstable cities (MAE > 4F gets 50% of correction)
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(abs_errors)
    if mae > 4.0:
        med_error = med_error * 0.5
    # V5.16: tighter ±3F cap (was ±5F)
    return round(max(-3.0, min(3.0, med_error)), 2), len(recent)


# ── V5.14: City MAE helper for trust score ──────────────────────────────────
def get_city_mae_and_color(city, n_recent=14):
    """
    Returns (mae_float_or_None, color_string).
    color: 'green' if <2.5F, 'yellow' if 2.5-4F, 'red' if >=4F.
    When insufficient data, returns (None, 'green') — treat as neutral-trusting
    since unknown MAE shouldn't actively penalize trust score yet.
    """
    rows = sb_fetch_city(city)
    complete = [r for r in rows if r.get('actual') is not None and r.get('error') is not None]
    if len(complete) < 3:
        return None, 'green'
    recent = complete[-n_recent:]
    errors = [abs(r['error']) for r in recent]
    mae = round(sum(errors) / len(errors), 2)
    if mae < 2.5:
        color = 'green'
    elif mae < 4.0:
        color = 'yellow'
    else:
        color = 'red'
    return mae, color

# ── High Uncertainty Threshold ────────────────────────────────────────────────
def get_uncertainty_threshold(city):
    """
    V5.6: Wider threshold for northeast/coastal cities in spring (months 3-5).
    V5.7: LA gets 7.0F — marine layer causes routine GFS divergence, not real uncertainty.
    """
    month = datetime.now().month
    if city == 'Los Angeles' and 3 <= month <= 5:
        return 7.0
    if city in SPRING_WIDE_THRESHOLD_CITIES and 3 <= month <= 5:
        return 6.5
    if city in DESERT_CITIES:
        return 6.0
    return 5.0

# ── NBM Fetcher ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_nbm_percentiles(lat, lon):
    """
    V5.6 fix: Convert valid_time from UTC to city local time before applying
    daytime hour filter (6am-9pm). Previously filtered in UTC, which starved
    western/central cities of NBM data points and caused Sigma fallback.
    """
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
            params={
                'location_name': station,
                'start_valid_time': start_utc,
                'end_valid_time': end_utc,
                'model': 'NBM',
            },
            headers=WETHR_HEADERS, timeout=15
        )
        if r.status_code != 200: return None
        forecasts = r.json()
        if not forecasts or not isinstance(forecasts, list): return None

        # V5.6 fix: convert valid_time (UTC) → city local time before hour filtering
        run_highs = {}
        for f in forecasts:
            valid_time_str = f.get('valid_time', '')
            temp_f = f.get('temperature_f')
            run_time = f.get('run_time', '')
            if temp_f is None: continue
            try:
                # Parse as UTC, convert to city local time
                vt_utc = datetime.strptime(valid_time_str, '%Y-%m-%d %H:%M:%S')
                vt_utc = pytz.utc.localize(vt_utc)
                vt_local = vt_utc.astimezone(city_tz)
                # Only keep daytime hours (6am-9pm) in city local time
                if not (6 <= vt_local.hour <= 21): continue
                # Only keep today's local date
                if vt_local.strftime('%Y-%m-%d') != today_local: continue
            except Exception: continue
            if run_time not in run_highs:
                run_highs[run_time] = []
            run_highs[run_time].append(float(temp_f))

        if not run_highs: return None

        run_max_temps = [max(temps) for temps in run_highs.values() if temps]
        # V5.7: reduced from 2 → 1, accept single NBM run for cities with fewer runs available
        if len(run_max_temps) < 1: return None

        run_max_temps.sort()

        def percentile(data, p):
            idx = (p / 100) * (len(data) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(data) - 1)
            return round(data[lo] + (idx - lo) * (data[hi] - data[lo]), 1)

        return {
            'p10': percentile(run_max_temps, 10),
            'p25': percentile(run_max_temps, 25),
            'p50': percentile(run_max_temps, 50),
            'p75': percentile(run_max_temps, 75),
            'p90': percentile(run_max_temps, 90),
        }

    except Exception:
        return None

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

# ── V5.2: Market Conviction ───────────────────────────────────────────────────
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

# ── V5.2: Bracket Boundary Warning ───────────────────────────────────────────
def check_bracket_boundary(consensus, ladder_text, boundary_threshold=0.5):
    warnings = []
    for label, lo, hi in parse_ladder(ladder_text):
        if hi is not None and abs(consensus - hi) <= boundary_threshold:
            warnings.append(f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of {hi}F ceiling ({label}) — NWS rounding could push to next bracket up. Verify before betting.')
        if lo is not None and abs(consensus - lo) <= boundary_threshold:
            warnings.append(f'⚠️ Boundary risk: Consensus {consensus}F is within {boundary_threshold}F of {lo}F floor ({label}) — NWS rounding could push to bracket below. Verify before betting.')
    return warnings

# ── V5.2: Cold Front + Morning Suppression ────────────────────────────────────
def check_cold_front_warning(obs_high, current_temp, nws_forecast, local_hour):
    if obs_high is not None and current_temp is not None:
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

# ── NWS Fetchers ──────────────────────────────────────────────────────────────
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
        except Exception:
            pass

    # Fallback: NWS hourly endpoint
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
    except Exception:
        pass
    return None, None

def fetch_nws_current(lat, lon, station_id):
    city_name = None
    best_dist = float('inf')
    for c, coords in CITIES.items():
        dist = abs(coords['lat'] - lat) + abs(coords['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            city_name = c
    wethr_station = WETHR_STATIONS.get(city_name)

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
                if temp_display is not None:
                    return wethr_station, round(float(temp_display), 1)
        except Exception:
            pass

    # Fallback: NWS API
    if station_id:
        obs = safe_get('https://api.weather.gov/stations/' + station_id + '/observations/latest')
        if obs:
            temp_c = obs.get('properties', {}).get('temperature', {}).get('value')
            if temp_c is not None: return station_id, float(c_to_f(temp_c))
    points = safe_get('https://api.weather.gov/points/' + str(lat) + ',' + str(lon))
    if not points: return station_id, None
    stations_url = points.get('properties', {}).get('observationStations')
    if not stations_url: return station_id, None
    stations = safe_get(stations_url)
    if not stations or not stations.get('observationStations'): return station_id, None
    first = stations['observationStations'][0]
    sid = first.rstrip('/').split('/')[-1]
    obs = safe_get(first + '/observations/latest')
    if not obs: return sid, None
    temp_c = obs.get('properties', {}).get('temperature', {}).get('value')
    if temp_c is None: return sid, None
    return sid, float(c_to_f(temp_c))

# ── Kelly ─────────────────────────────────────────────────────────────────────
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

# ── V5.14: Trust Score helper ──────────────────────────────────────────────
def compute_row_trust(
    city, bracket_label, direction, model_pct,
    ensemble_tier, two_degree_call_str,
    mae_color, nbm_active, nws_forecast_f, gfs_ensemble_f, bias_adj_f,
):
    """
    Compute trust score for a single bracket row. Returns a TrustScoreResult.
    Inputs are the same values the existing code already computes —
    this just packages them into SignalInputs and calls compute_trust_score.
    """
    try:
        inp = SignalInputs(
            city=str(city or ''),
            bracket_label=str(bracket_label or ''),
            direction=str(direction or 'YES'),
            two_degree_call=str(two_degree_call_str or ''),
            bracket_midpoint=bracket_midpoint_from_label(bracket_label),
            twodc_midpoint=bracket_midpoint_from_label(two_degree_call_str),
            model_pct=float(model_pct or 0),
            edge_cents=0.0,  # not used by trust score
            ensemble_tier=str(ensemble_tier or ''),
            mae_color=str(mae_color or 'green'),
            nbm_active=bool(nbm_active),
            nws_forecast_f=float(nws_forecast_f) if nws_forecast_f is not None else None,
            gfs_ensemble_f=float(gfs_ensemble_f) if gfs_ensemble_f is not None else None,
            bias_adj_f=float(bias_adj_f or 0),
        )
        return compute_trust_score(inp)
    except Exception:
        return None

def trust_tier_icon(tier):
    """UI icon for a trust tier."""
    if tier == 'BET': return '🟢'
    if tier == 'CAUTION': return '🟡'
    if tier == 'SKIP': return '⚪'
    return '—'

def ensemble_tier_from_confidence(conf_str):
    """Extract HIGH/MED/LOW from strings like '🔵 HIGH' or '🟡 MED' or '⚪ LOW'."""
    if not conf_str: return ''
    if 'HIGH' in conf_str: return 'HIGH'
    if 'MED' in conf_str: return 'MED'
    if 'LOW' in conf_str: return 'LOW'
    return ''

def get_city_best_signals(city, consensus, ladder_text, ensemble_members, kalshi_markets_data,
                          obs_high, high_uncertainty, bankroll, nbm_percentiles=None,
                          current_temp=None, nws_forecast=None, local_hour=12,
                          min_prob=0.10):
    """
    V5.8: Added min_prob threshold (default 10%).
    Brackets where model probability < 10% are excluded from signal consideration
    regardless of edge size. Eliminates phantom signals on near-impossible brackets
    (e.g. Minneapolis 79 or below when consensus is 80F).
    """
    if consensus is None or not ladder_text: return '—', '—'
    try:
        prob_rows, _, used_nbm = bracket_probs_nbm(consensus, ladder_text, city, nbm_percentiles, obs_high=obs_high)
        # V5.17: apply probability floor to prevent 0.0% bug on adjacent brackets
        prob_rows = apply_prob_floor(prob_rows, consensus, ladder_text)
        morning_suppressed, _ = check_morning_suppression(obs_high, current_temp, nws_forecast, local_hour)
        conviction_result = check_market_conviction(kalshi_markets_data, ladder_text)
        best_yes = best_no = None
        best_yes_edge = best_no_edge = -999

        for label, base_prob in prob_rows:
            ens_prob = next((ensemble_bracket_prob(ensemble_members, lo, hi)
                             for lbl, lo, hi in parse_ladder(ladder_text) if labels_match(lbl, label)), None)
            final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)

            # V5.8: Skip brackets where model probability is too low — not a real signal
            if final_prob < min_prob: continue

            # V5.8: Skip YES signals on brackets clearly below consensus
            # If bracket ceiling is more than 2F below consensus, YES bet makes no sense
            bracket_parsed = next(((lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)
                                   if labels_match(lbl, label)), (None, None))
            b_lo, b_hi = bracket_parsed
            below_consensus = (b_hi is not None and consensus is not None and
                               b_hi < consensus - 2.0)
            above_consensus = (b_lo is not None and consensus is not None and
                               b_lo > consensus + 2.0)

            yes_ask = no_ask = None
            if kalshi_markets_data:
                match = next((m for m in kalshi_markets_data if labels_match(m[0], label)), None)
                if match: yes_ask, no_ask = match[1], match[2]
            busted = obs_high is not None and any(
                labels_match(lbl, label) and hi is not None and obs_high > hi + 0.4
                for lbl, lo, hi in parse_ladder(ladder_text))
            conviction_conflict = (conviction_result and
                                   is_conflicting_with_conviction(label, conviction_result[1], conviction_result[2], ladder_text))
            e = edge_cents(final_prob, yes_ask)
            icon, _ = edge_signal(e, high_uncertainty, morning_suppressed, conviction_conflict)
            # Only show YES signal if bracket is near or above consensus
            if (e is not None and e > best_yes_edge and not busted and icon == '🟢'
                    and not below_consensus):
                best_yes_edge = e
                kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0
                best_yes = f'🟢 {label} | +{e}c | ${kelly}'

            # V5.8: For NO signals, also require model prob of being ABOVE bracket > 10%
            no_model_prob = 1.0 - final_prob
            if no_model_prob < min_prob: continue

            no_e = no_edge_cents(final_prob, no_ask)
            no_icon, _ = no_signal(no_e, busted=busted, model_prob=final_prob, no_ask=no_ask,
                                   high_uncertainty=high_uncertainty, morning_suppressed=morning_suppressed,
                                   conviction_conflict=conviction_conflict)
            # Only show NO signal if bracket is near or below consensus
            if (no_icon == '🟢' and no_e is not None and no_e > best_no_edge
                    and not above_consensus):
                best_no_edge = no_e
                kelly_no = kelly_bet_no(final_prob, no_ask, bankroll) if no_ask else 0.0
                best_no = f'🟢 {label} NO | +{no_e}c | ${kelly_no}'
        return best_yes or '—', best_no or '—'
    except Exception: return '—', '—'

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
    """
    V5.17: Fix the 0.0% bug on brackets adjacent to consensus.
    Enforces minimum probability for brackets that could plausibly hit:
      - Brackets whose midpoint is within 4F of consensus → min 5%
      - Brackets whose midpoint is within 6F (touching ladder) → min 2%
      - Brackets further out stay at computed value (can be near-zero)
    Redistributes probability so the row still sums to approximately 1.0.

    Takes list of (label, prob) tuples, returns same shape.

    Why this exists: before V5.17, the model would declare brackets 2-3F
    away from consensus as 0.0% probability — which is effectively saying
    "impossible." In reality, forecast sigma is 1.5-2.5F and unexpected
    outcomes happen. The floor prevents the model from ruling out outcomes
    that are actually 10-20% likely.
    """
    if not prob_rows or consensus is None: return prob_rows
    parsed = {lbl: (lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)}
    adjusted = []
    boost_total = 0.0
    for label, prob in prob_rows:
        lo, hi = parsed.get(label, (None, None))
        # Compute bracket midpoint for distance check
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
        elif lo is not None:   # "X or above"
            mid = lo + 1.0
        elif hi is not None:   # "X or below"
            mid = hi - 1.0
        else:
            adjusted.append((label, prob)); continue
        distance = abs(mid - consensus)
        new_prob = prob
        if distance <= 4.0 and prob < 0.05:
            new_prob = 0.05
        elif distance <= 6.0 and prob < 0.02:
            new_prob = 0.02
        boost_total += (new_prob - prob)
        adjusted.append((label, new_prob))
    # Redistribute: if we added probability, scale everything slightly down
    # to keep sum reasonable. Only scale down brackets with room (prob > floor).
    if boost_total > 0:
        scale = 1.0 / (1.0 + boost_total)
        adjusted = [(lbl, round(p * scale, 4)) for lbl, p in adjusted]
    return adjusted

# ── Core ──────────────────────────────────────────────────────────────────────
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
    # V5.7: northeast cities get more aggressive floor — catches warm surges earlier
    if city in NORTHEAST_CITIES:
        frac = 0.50 if local_hour < 12 else 0.75 if local_hour < 14 else 0.88 if local_hour < 16 else 0.93
    else:
        frac = 0.45 if local_hour < 12 else 0.62 if local_hour < 14 else 0.78 if local_hour < 16 else 0.90
    return obs + frac * gap

def compute_consensus(fc, cur, noaa, city, obs_high=None):
    """
    V5.16 changes:
    1. Per-city prediction mode: cities in CITY_PREDICTION_MODE as 'nws_only'
       skip the blend entirely and return `fc` (NWS forecast) directly. Bias
       correction is still added on top by the caller.
    2. Time-gated obs_high: obs_high can only override consensus after 1 PM
       local, AND only if it passes sanity checks (obs_high ≥ current_temp,
       obs_high ≤ current_temp + 10F). Morning sensor spikes are ignored.
    """
    # V5.16: nws_only mode — skip the blend for cities where NWS alone is better
    mode = CITY_PREDICTION_MODE.get(city, 'full_blend')
    if mode == 'nws_only':
        # Caller adds bias_correction on top; we just return the raw forecast
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

    # V5.16: time-gated obs_high override (was: unconditional in V5.15)
    if obs_high is not None and obs_high > consensus:
        obs_high_trusted = True
        # Only trust obs_high after 1 PM local
        if local_hour < OBS_HIGH_TRUST_HOUR:
            obs_high_trusted = False
        # obs_high should never exceed current temp by more than 10F
        # (a legit peak shouldn't be 10F+ above what the station is reading now)
        current_for_check = obs if obs is not None else cur
        if current_for_check is not None and obs_high > current_for_check + OBS_HIGH_MAX_OVERSHOOT:
            obs_high_trusted = False
        # obs_high should never be below current temp (stale reading)
        if current_for_check is not None and obs_high < current_for_check:
            obs_high_trusted = False
        if obs_high_trusted:
            consensus = obs_high
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
    except Exception:
        pass

    # Fallback: NWS obs history table
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

@st.cache_data(ttl=1800)
def fetch_city_weather(city):
    coords = CITIES[city]
    lat, lon = coords['lat'], coords['lon']
    nws_fc, _ = fetch_nws_forecast(lat, lon)
    _, current_temp = fetch_nws_current(lat, lon, STATIONS[city])
    obs_high_raw, six_hr_max, _ = fetch_obs_high_today(OBHISTORY_STATIONS[city])
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)

    obs_high_final = obs_high_raw
    obs_high_discarded = False
    obs_high_discard_reason = None

    if obs_high_raw is not None and current_temp is not None and obs_high_raw > current_temp + 15.0:
        obs_high_final = None; obs_high_discarded = True
        obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - current_temp, 1)}F above current temp'
    if obs_high_raw is not None and nws_fc is not None and not obs_high_discarded and obs_high_raw > nws_fc + 12.0:
        obs_high_final = None; obs_high_discarded = True
        obs_high_discard_reason = f'Obs high {obs_high_raw}F discarded — {round(obs_high_raw - nws_fc, 1)}F above NWS forecast'

    if ensemble_mean is not None and nws_fc is not None and abs(ensemble_mean - nws_fc) > 8.0:
        ensemble_members = None; ensemble_mean = None

    # V5.6: use city-specific uncertainty threshold
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

# ── V5.15: Bet Log in Supabase ────────────────────────────────────────────────
# V5.15: Migrated from local bet_log.json to Supabase `bets` table.
#        - Persists across Streamlit Cloud redeploys
#        - Supports Edit/Delete via new CRUD helpers below
#        - Legacy load_bet_log() kept as a compatibility shim so existing
#          call sites (in settle_bet_log and elsewhere) continue to work.

def sb_fetch_bets():
    """Fetch all bets from Supabase, ordered by id ascending."""
    try:
        r = requests.get(sb_url('bets'), headers=get_sb_headers(),
                         params={'order': 'id.asc', 'limit': '1000'}, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception: return []

def sb_insert_bet(bet_dict):
    """Insert a single bet row. Returns the inserted row (with id) or None."""
    try:
        r = requests.post(sb_url('bets'), headers=get_sb_headers(),
                          json=bet_dict, timeout=10)
        if r.status_code in (200, 201):
            rows = r.json()
            return rows[0] if rows else None
        st.error(f'Bet insert failed: {r.status_code} — {r.text[:200]}')
        return None
    except Exception as e:
        st.error(f'Bet insert exception: {str(e)[:200]}')
        return None

def sb_update_bet(bet_id, updates):
    """Update columns on a bet row by id. `updates` is a dict of column→value."""
    try:
        r = requests.patch(sb_url('bets') + '?id=eq.' + str(bet_id),
                           headers=get_sb_headers(),
                           json=updates, timeout=10)
        return r.status_code in (200, 204)
    except Exception: return False

def sb_delete_bet(bet_id):
    """Delete a bet row by id."""
    try:
        r = requests.delete(sb_url('bets') + '?id=eq.' + str(bet_id),
                            headers=get_sb_headers(), timeout=10)
        return r.status_code in (200, 204)
    except Exception: return False

# Backwards-compatible shim — settle_bet_log() still calls load_bet_log()
def load_bet_log():
    return sb_fetch_bets()

def save_bet_log(log):
    # Legacy whole-log saves are no longer used. Kept as no-op for safety.
    # New writes go through sb_insert_bet / sb_update_bet / sb_delete_bet.
    pass

def bracket_hits(actual_temp, lo, hi):
    """
    V5.12: Decide whether an actual settled high falls within a bracket.
    Uses Kalshi-style rounding: .5 rounds up (math.floor(x + 0.5)).
    CLI returns whole numbers already; fallback NWS obs table can return decimals.
    """
    if actual_temp is None: return None
    rounded = int(math.floor(float(actual_temp) + 0.5))
    if lo is None and hi is not None:   # "X or below"
        return rounded <= hi
    if hi is None and lo is not None:   # "X or above"
        return rounded >= lo
    if lo is not None and hi is not None:  # "X-Y"
        return lo <= rounded <= hi
    return None

def settle_bet_log(settled_rows):
    """
    V5.15: For each city/date that just settled, find Pending bets in Supabase
    and mark them Won/Lost based on whether the bracket hit. Writes updates
    directly to Supabase per-row rather than rewriting a local JSON file.

    Returns list of bets that were just settled this run (for UI toasts).
    """
    if not settled_rows: return []
    bet_log = sb_fetch_bets()
    if not bet_log: return []
    now_iso = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')
    just_settled = []
    for s in settled_rows:
        s_city = s.get('city')
        s_date = s.get('date')
        s_actual = s.get('actual')
        if s_city is None or s_date is None or s_actual is None: continue
        for b in bet_log:
            if b.get('result') != 'Pending': continue
            if b.get('city') != s_city: continue
            # Supabase returns date as 'YYYY-MM-DD' string — compare as strings
            if str(b.get('date')) != s_date: continue
            bracket_str = b.get('bracket', '')
            lo, hi = label_to_numeric_key(bracket_str)
            if lo is None and hi is None:
                continue
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
                profit = -amount
                payout = 0.0
            updates = {
                'result': 'Won' if won else 'Lost',
                'profit': profit,
                'payout': payout,
                'actual': s_actual,
                'settled_at': now_iso,
            }
            ok = sb_update_bet(b['id'], updates)
            if ok:
                just_settled.append({
                    'city': s_city, 'date': s_date, 'bracket': bracket_str,
                    'direction': direction, 'actual': s_actual, 'won': won,
                    'amount': amount, 'profit': profit,
                })
    return just_settled

def settle_pending_bets_retroactive():
    """
    V5.16: Bug fix for bets logged AFTER their settlement row was already
    written. The old flow only settled bets when `run_auto_settlement` just
    populated a settlement row — but if the settlement already existed, new
    pending bets on that date never got matched.

    This scans ALL pending bets, looks up their matching settlement row
    (city + date) from Supabase, and settles them if the `actual` is
    available. Runs on every app load before `run_auto_settlement`.

    Returns list of bets settled this pass (for UI toasts).
    """
    bet_log = sb_fetch_bets()
    if not bet_log: return []
    pending = [b for b in bet_log if b.get('result') == 'Pending']
    if not pending: return []
    # Build dict of (city, date) -> actual from settlements
    all_settlements = sb_fetch_all()
    settlement_map = {}
    for s in all_settlements:
        if s.get('actual') is not None:
            key = (s.get('city'), str(s.get('date')))
            settlement_map[key] = s['actual']
    now_iso = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')
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
            profit = -amount
            payout = 0.0
        updates = {
            'result': 'Won' if won else 'Lost',
            'profit': profit,
            'payout': payout,
            'actual': actual,
            'settled_at': now_iso,
        }
        if sb_update_bet(b['id'], updates):
            just_settled.append({
                'city': b.get('city'), 'date': b.get('date'),
                'bracket': bracket_str, 'direction': direction,
                'actual': actual, 'won': won, 'amount': amount, 'profit': profit,
            })
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
    st.markdown('<div class="mph-section-header">📊 Signal Key</div>', unsafe_allow_html=True)
    st.markdown('🟢 **BET** — Edge ≥8c')
    st.markdown('🟡 **SKIP** — Edge 3-7c')
    st.markdown('🔴 **AVOID** — Edge <3c')
    st.markdown('⚪ **No price** — Unpriced bracket')
    st.markdown('🔵 **HIGH** — Ensemble confident')
    st.markdown('---')
    # V5.16: two-dot UI key
    st.markdown('<div class="mph-section-header">🎯 Signal Key</div>', unsafe_allow_html=True)
    st.markdown('**Column 1 (Signal):**')
    st.markdown('🟢 BET · 🟡 caution · ⚪ neutral · 🔴 avoid')
    st.markdown('**Column next-to-last (Trust 0-100):**')
    st.markdown('≥75 = BET · 55-74 = CAUTION · <55 = SKIP')
    st.markdown('**Last column (Ensemble):**')
    st.markdown('🔵 HIGH · 🟡 MED · ⚪ LOW')
    st.caption('Best bet = 🟢 + 🔵 with highest Trust')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🔬 MAE Guide</div>', unsafe_allow_html=True)
    st.markdown('✅ **<2.5F** — Well calibrated')
    st.markdown('🟡 **2.5-4F** — Acceptable')
    st.markdown('🔴 **>4F** — Needs attention')
    st.markdown('---')
    st.markdown('<div class="mph-section-header">🚀 V5.17</div>', unsafe_allow_html=True)
    st.markdown('- **Accuracy-first** (not edge-first) picks')
    st.markdown('- Dual dots: 🟢🔵 accuracy + 🟣 edge')
    st.markdown('- Probability floor (no more 0.0%)')
    st.markdown('- Summary panel at top')
    st.markdown('- Accuracy sweet spot times')
    st.markdown('- NO-bet highlight section')
    st.markdown('- Reality-check banner')
    st.markdown('- 3 more cities hidden (Philly/Boston/SA)')

# ── Main App ──────────────────────────────────────────────────────────────────
saved_ladders = load_json(SAVE_FILE)
today_str = get_eastern_date()
last_sync_data = load_json(LAST_SYNC_FILE)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mph-hero">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="mph-hero-title">
                🌡️ MPH Weather Model
                <span class="mph-version-badge">V5.17</span>
            </div>
            <div class="mph-hero-sub">
                <span class="mph-live-dot"></span>
                LIVE · Kalshi High Temperature · {today_str}
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
    # V5.12: after model settlements, cascade into the personal bet log
    bet_log_just_settled = settle_bet_log(settled_rows)
    # V5.16: also settle any pending bets whose settlement was already written
    # (fixes bug where bets logged AFTER auto-settlement ran never got matched)
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

# V5.12: Surface any personal bet log settlements from this run
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
            saved_ladders, results = sync_all_ladders(saved_ladders, force=True)
            st.success('Re-synced ' + str(len(results.get('synced', []))) + '/' + str(len(SERIES)) + ' city ladders')
            if results.get('failed'): st.warning('Could not fetch: ' + ', '.join(results['failed']))
            st.rerun()

# ── Live Stats Bar ────────────────────────────────────────────────────────────
_all_rows_stats = sb_fetch_all()
_today_rows_stats = [r for r in _all_rows_stats if r.get('date') == today_str]
_n_cities = len(_today_rows_stats)
_synced_count = len(last_sync_data.get('synced', []))
_last_sync_time = last_sync_data.get('date', '—')
_nbm_count = 0  # will be updated after weather fetch, show placeholder

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

# ── V5.9: Auto-refresh every 10 minutes ──────────────────────────────────────
import streamlit.components.v1 as components
components.html('<script>setTimeout(function(){window.location.reload();}, 600000);</script>', height=0)

# ── V5.9: Mac/iPhone View Toggle ─────────────────────────────────────────────
view_mode = st.radio('View', ['📊 Mac', '📱 iPhone'], horizontal=True, label_visibility='collapsed')
is_mobile = view_mode == '📱 iPhone'

# ── V5.9: Timezone Best Signal Banner ────────────────────────────────────────
TIMEZONE_GROUPS = {
    'ET': {
        'cities': ['New York', 'Boston', 'Philadelphia', 'Washington DC', 'Atlanta', 'Miami'],
        'cutoff_et_hour': 14,
        'label': '🕙 ET Cities',
        'closes': '2:00 PM ET',
    },
    'CT': {
        'cities': ['Chicago', 'Dallas', 'Austin', 'Houston', 'San Antonio', 'New Orleans', 'Oklahoma City', 'Minneapolis'],
        'cutoff_et_hour': 15,
        'label': '🕙 CT Cities',
        'closes': '3:00 PM ET',
    },
    'MT': {
        'cities': ['Denver'],
        'cutoff_et_hour': 16,
        'label': '🕙 MT Cities',
        'closes': '4:00 PM ET',
    },
    'PT': {
        'cities': ['Phoenix', 'Las Vegas', 'Los Angeles'],
        'cutoff_et_hour': 17,
        'label': '🕙 PT Cities',
        'closes': '5:00 PM ET',
    },
}

def get_et_hour():
    return datetime.now(pytz.timezone('America/New_York')).hour

def get_et_time_str():
    return datetime.now(pytz.timezone('America/New_York')).strftime('%I:%M %p ET')

def minutes_until_close(cutoff_et_hour):
    """Kept for backwards compatibility — used in a few places to gate logic."""
    now_et = datetime.now(pytz.timezone('America/New_York'))
    close_et = now_et.replace(hour=cutoff_et_hour, minute=0, second=0, microsecond=0)
    diff = (close_et - now_et).total_seconds() / 60
    return int(diff)

def window_status(cutoff_et_hour):
    """
    V5.15: Removed the misleading 'XYZ min left' countdown. The old version
    rolled over past the cutoff and showed tomorrow's countdown, which was
    confusing. Now just shows OPEN / CLOSING / CLOSED — the 'closes 2:00 PM ET'
    label in the header tells you exactly when the window ends.
    """
    mins = minutes_until_close(cutoff_et_hour)
    if mins <= 0: return '🔴 CLOSED', '#ef4444'
    if mins <= 30: return '⚠️ CLOSING SOON', '#f59e0b'
    return '✅ OPEN', '#00ff88'

TIMEZONE_STATIC_INFO = {
    # V5.17: Added 'accuracy_window' — time range when model is most likely
    # to be accurate on bracket placement. Typically ~2-3 hours before peak
    # heat, when obs data has clarified the day's trajectory but market
    # hasn't fully priced it in. Differs from 'sweet_spot' which is the
    # edge-hunting window (earlier, when markets are slowest).
    'ET': {'sweet_spot': '10:00–11:30 AM ET', 'accuracy_window': '12:00–2:00 PM ET', 'peak_heat': '2:00–4:00 PM ET'},
    'CT': {'sweet_spot': '11:00 AM–12:30 PM ET', 'accuracy_window': '1:00–3:00 PM ET', 'peak_heat': '3:00–5:00 PM ET'},
    'MT': {'sweet_spot': '12:00–1:30 PM ET', 'accuracy_window': '2:00–4:00 PM ET', 'peak_heat': '4:00–6:00 PM ET'},
    'PT': {'sweet_spot': '1:00–2:30 PM ET', 'accuracy_window': '3:00–5:00 PM ET', 'peak_heat': '5:00–7:00 PM ET'},
}

def get_phase_label(tz_key, et_hour):
    """Return phase label only during active windows — empty outside them."""
    now_et = datetime.now(pytz.timezone('America/New_York'))
    et_hhmm = now_et.hour * 100 + now_et.minute

    phase_times = {
        'ET': {'bet': (1000, 1130), 'peak': (1400, 1600)},
        'CT': {'bet': (1100, 1230), 'peak': (1500, 1700)},
        'MT': {'bet': (1200, 1330), 'peak': (1600, 1800)},
        'PT': {'bet': (1300, 1430), 'peak': (1700, 1900)},
    }
    times = phase_times.get(tz_key, {})
    bet_start, bet_end = times.get('bet', (0, 0))
    peak_start, peak_end = times.get('peak', (0, 0))

    if bet_start <= et_hhmm < bet_end:
        return '🟢 BET NOW', '#00ff88'
    if peak_start <= et_hhmm < peak_end:
        return '🌡️ PEAK HEAT', '#00b4d8'
    if et_hhmm < bet_start:
        return '⏳ EARLY', '#94a3b8'
    return '', '#64748b'

st.markdown('<div class="mph-section-header">🎯 Best Bets By Timezone Window</div>', unsafe_allow_html=True)

_all_rows_banner = sb_fetch_all()
_today_rows_banner = {r['city']: r for r in _all_rows_banner if r.get('date') == today_str}
_et_hour_now = get_et_hour()

# ── V5.17: Summary Panel ──────────────────────────────────────────────
# Shows one line per visible city with model's accuracy pick (the
# bracket the model thinks is most likely to hit), so user can see the
# full picture at a glance before diving into per-city details.
# Uses the cached banner data plus Best YES pick.
_summary_rows = []
for tz_key, tz_info in TIMEZONE_GROUPS.items():
    for c in tz_info['cities']:
        if c in HIDDEN_CITIES: continue
        row = _today_rows_banner.get(c)
        if not row: continue
        consensus_val = row.get('consensus')
        if consensus_val is None: continue
        ladder_banner = saved_ladders.get(c, DEFAULT_LADDERS.get(c, ''))
        cached_markets_banner, _ = get_cached_prices(c)
        try:
            best_yes_banner, best_no_banner = get_city_best_signals(
                city=c, consensus=consensus_val, ladder_text=ladder_banner,
                ensemble_members=[], kalshi_markets_data=cached_markets_banner,
                obs_high=row.get('obs_high'), high_uncertainty=row.get('high_uncertainty', False),
                bankroll=25.0, nbm_percentiles=None,
                current_temp=None, nws_forecast=row.get('forecast'),
                local_hour=get_local_hour(c), min_prob=0.10,
            )
        except Exception:
            best_yes_banner, best_no_banner = '—', '—'
        # Parse best YES into a compact display
        _summary_rows.append({
            'city': c,
            'consensus': round(consensus_val, 1),
            'best_yes': best_yes_banner,
            'best_no': best_no_banner,
            'obs_high': row.get('obs_high'),
        })

if _summary_rows and not is_mobile:
    with st.expander('📌 Today\'s Model Picks — Summary', expanded=False):
        st.caption('Quick view of each visible city\'s top signal. Use for morning overview. '
                   'Click into each timezone banner below for full detail and sanity checks.')
        import pandas as pd
        _df = pd.DataFrame([{
            'City': r['city'],
            'Consensus °F': r['consensus'],
            'Obs High °F': r['obs_high'] if r['obs_high'] is not None else '—',
            'Best YES pick': r['best_yes'],
            'Best NO pick': r['best_no'],
        } for r in _summary_rows])
        st.dataframe(_df, use_container_width=True, hide_index=True)

for tz_key, tz_info in TIMEZONE_GROUPS.items():
    # V5.16.1: skip the whole timezone block if every city is hidden
    # (e.g., MT is only Denver, which is hidden — so hide the MT section entirely)
    visible_cities_in_tz = [c for c in tz_info['cities'] if c not in HIDDEN_CITIES]
    if not visible_cities_in_tz:
        continue

    status_text, status_color = window_status(tz_info['cutoff_et_hour'])
    mins_left = minutes_until_close(tz_info['cutoff_et_hour'])
    is_closed = mins_left <= 0
    phase_label, phase_color = get_phase_label(tz_key, _et_hour_now)

    yes_signals = []
    no_signals = []

    for c in tz_info['cities']:
        if c in HIDDEN_CITIES: continue  # V5.16: skip hidden cities in banner
        row = _today_rows_banner.get(c)
        if not row: continue
        consensus_val = row.get('consensus')
        ladder = saved_ladders.get(c, DEFAULT_LADDERS.get(c, ''))
        cached_markets, _ = get_cached_prices(c)
        obs_h = row.get('obs_high')
        high_unc = row.get('high_uncertainty', False)
        try:
            cached_wx = fetch_city_weather(c)
            members = cached_wx.get('ensemble_members') if cached_wx else None
            nbm_pcts = cached_wx.get('nbm_percentiles') if cached_wx else None
            c_temp = cached_wx.get('current_temp') if cached_wx else None
            c_fc = cached_wx.get('nws_fc') if cached_wx else None
            c_hour = cached_wx.get('local_hour', 12) if cached_wx else 12
        except Exception:
            members = nbm_pcts = c_temp = c_fc = None
            c_hour = 12

        # V5.11: Get city MAE and settled days for flags
        city_hist = sb_fetch_city(c)
        city_complete = [r for r in city_hist if r.get('actual') is not None and r.get('error') is not None]
        settled_days = len(city_complete)
        if settled_days < 3:
            # Not ready — suppress from banner
            continue
        city_mae = None
        if city_complete:
            recent_14 = city_complete[-14:]
            city_errors = [abs(r['error']) for r in recent_14]
            city_mae = round(sum(city_errors) / len(city_errors), 1)
        red_mae = city_mae is not None and city_mae >= 4.0

        # V5.11: Consensus vs NWS gap warning
        c_nws = row.get('forecast')
        consensus_nws_gap = abs(consensus_val - c_nws) if consensus_val and c_nws else None
        big_gap = consensus_nws_gap is not None and consensus_nws_gap > 3.0

        _, price_fetched_at = get_cached_prices(c)
        price_age_min = round((time.time() - price_fetched_at) / 60) if price_fetched_at else 999
        if price_age_min > 120: continue

        b_yes, b_no = get_city_best_signals(
            c, consensus_val, ladder, members, cached_markets, obs_h, high_unc, 500,
            nbm_percentiles=nbm_pcts, current_temp=c_temp, nws_forecast=c_fc, local_hour=c_hour)

        # Build flag suffix
        flags = ''
        if red_mae: flags += f' ⚠️MAE:{city_mae}F'
        if big_gap: flags += f' ⚠️Gap:{round(consensus_nws_gap,1)}F'

        if b_yes and b_yes != '—' and '+' in b_yes:
            try:
                edge_val = float(b_yes.split('+')[1].split('c')[0])
                yes_signals.append((edge_val, f'{c}: {b_yes}{flags}', red_mae))
            except Exception: pass

        if b_no and b_no != '—' and '+' in b_no:
            try:
                edge_val = float(b_no.split('+')[1].split('c')[0])
                no_signals.append((edge_val, f'{c}: {b_no}{flags}', red_mae))
            except Exception: pass

    yes_signals.sort(key=lambda x: x[0], reverse=True)
    no_signals.sort(key=lambda x: x[0], reverse=True)

    # V5.16.1: filter the header city list to exclude hidden cities
    city_list_str = ' · '.join([c for c in tz_info['cities'] if c not in HIDDEN_CITIES])
    static = TIMEZONE_STATIC_INFO.get(tz_key, {})
    sweet_spot_str = static.get('sweet_spot', '')
    accuracy_window_str = static.get('accuracy_window', '')  # V5.17
    peak_heat_str = static.get('peak_heat', '')
    phase_label, phase_color = get_phase_label(tz_key, _et_hour_now)  # V5.10: 2 return values
    phase_html = f'<span style="color:{phase_color}; font-size:12px; font-weight:700; font-family:\'JetBrains Mono\',monospace;">{phase_label}</span>' if phase_label else ''

    if is_closed:
        yes_html = '<div style="color:#ef4444; font-size:12px;">🔴 Window Closed</div>'
        no_html = '<div style="color:#ef4444; font-size:12px;">🔴 Window Closed</div>'
    else:
        yes_html = ''.join([
            f'<div style="color:{"#f59e0b" if red_mae else "#00ff88"}; font-size:12px; font-family:\'JetBrains Mono\',monospace; margin-bottom:3px;">{"🟡" if red_mae else "🟢"} {sig}</div>'
            for _, sig, red_mae in yes_signals
        ]) or '<div style="color:#64748b; font-size:12px;">— No green YES signal</div>'
        no_html = ''.join([
            f'<div style="color:{"#f59e0b" if red_mae else "#00ff88"}; font-size:12px; font-family:\'JetBrains Mono\',monospace; margin-bottom:3px;">{"🟡" if red_mae else "🟢"} {sig}</div>'
            for _, sig, red_mae in no_signals
        ]) or '<div style="color:#64748b; font-size:12px;">— No green NO signal</div>'

    st.markdown(f"""
<div style="background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; flex-wrap:wrap; gap:6px;">
        <span style="color:#ffffff; font-weight:700; font-size:14px; font-family:'Inter',sans-serif;">{tz_info['label']} — closes {tz_info['closes']}</span>
        <span style="color:{status_color}; font-size:12px; font-family:'JetBrains Mono',monospace;">{status_text}</span>
    </div>
    <div style="margin-bottom:2px;">{phase_html}</div>
    <div style="color:#94a3b8; font-size:11px; margin-bottom:2px; font-family:'JetBrains Mono',monospace;">🟢 Sweet spot (edge): {sweet_spot_str}</div>
    <div style="color:#a78bfa; font-size:11px; margin-bottom:2px; font-family:'JetBrains Mono',monospace;">🎯 Accuracy window: {accuracy_window_str}</div>
    <div style="color:#00b4d8; font-size:11px; margin-bottom:8px; font-family:'JetBrains Mono',monospace;">🌡️ Peak heat: {peak_heat_str}</div>
    <div style="color:#64748b; font-size:11px; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">{city_list_str}</div>
    <div style="display:flex; gap:16px; flex-wrap:wrap;">
        <div style="flex:1; min-width:200px;">
            <div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;">🟢 YES Signals (Model thinks it WILL hit)</div>
            {yes_html}
        </div>
        <div style="flex:1; min-width:200px;">
            <div style="color:#94a3b8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;">🔻 NO Signals (Market overpriced · bet NO)</div>
            {no_html}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('---')

# V5.16: hide low-MAE cities from main selector (still predicted, just not shown)
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
    st.info(f'Bias correction active: {sign}{bias_correction}F applied to consensus '
            f'(model ran {direction} by avg {abs(bias_correction)}F over last {bias_n} days)')
elif bias_n > 0:
    st.caption(f'Bias correction: {bias_n} settlement(s) logged — need 3+ for correction')
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
            while len(labels) < 6: labels.append('')
            saved_ladders[city] = ' | '.join(labels[:6])
            save_json(SAVE_FILE, saved_ladders)
            fetched_at = time.time()

st.markdown('<div class="mph-section-header">📋 Kalshi Ladder</div>', unsafe_allow_html=True)
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

st.markdown('<div class="mph-section-header">🌤️ Live Weather</div>', unsafe_allow_html=True)
with st.spinner('Fetching weather data...'):
    nws_forecast, _ = fetch_nws_forecast(lat, lon)
    noaa_station, noaa_obs = fetch_nws_current(lat, lon, station)
    obs_high_raw, six_hr_max, obs_high_url = fetch_obs_high_today(obs_icao)
    ensemble_members, ensemble_mean = fetch_gfs_ensemble(lat, lon)
    nbm_percentiles = fetch_nbm_percentiles(lat, lon)

# V5.7: NWS forecast trend check
# Store previous NWS fetch per city in session state, detect upward trend
nws_trend_key = f'nws_prev_{city}_{get_eastern_date()}'
nws_trend_up = False
nws_trend_delta = None
if nws_forecast is not None:
    prev_nws = st.session_state.get(nws_trend_key)
    if prev_nws is not None and nws_forecast > prev_nws + 1.0:
        nws_trend_up = True
        nws_trend_delta = round(nws_forecast - prev_nws, 1)
    # Update stored value
    st.session_state[nws_trend_key] = nws_forecast

sanity_warnings = []
obs_high_today = obs_high_raw
obs_high_suspect = False

if obs_high_raw is not None:
    if noaa_obs is not None and obs_high_raw > noaa_obs + 15.0:
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

# V5.6: city-specific uncertainty threshold
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
        st.metric('Current Temp', str(round(noaa_obs, 1))+' F')
        st.caption('Station: ' + noaa_station)
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
    elif ensemble_suspect: st.metric('GFS Ensemble', 'Discarded'); st.caption('Failed sanity check (>8F from NWS)')
    else: st.metric('GFS Ensemble', 'Unavailable')

if nbm_percentiles:
    nbm_p50 = nbm_percentiles.get('p50', nbm_percentiles.get('p25', '—'))
    st.success(f'✅ NBM active — p10:{nbm_percentiles.get("p10","—")}F | p50:{nbm_p50}F | p90:{nbm_percentiles.get("p90","—")}F | bracket probs from real percentile distribution')
else:
    st.caption('📊 NBM unavailable — sigma/normal fallback active')

for w in sanity_warnings: st.error('⚠️ ' + w)

# V5.7: NWS trend up warning
if nws_trend_up and nws_trend_delta is not None:
    st.info(f'📈 NWS forecast trending UP +{nws_trend_delta}F since last fetch — model will boost consensus accordingly.')

if nws_forecast is None: st.error('NWS forecast unavailable — cannot run model.')
elif high_uncertainty and source_gap is not None:
    threshold = get_uncertainty_threshold(city)
    st.warning(f'HIGH UNCERTAINTY: NWS ({nws_forecast}F) vs GFS ({ensemble_mean}F) gap = {round(source_gap, 1)}F (threshold: {threshold}F). Green signals suppressed.')
elif source_gap is not None and source_gap > 4.0:
    st.info(f'Source gap: NWS vs Ensemble = {round(source_gap, 1)}F — moderate divergence.')

cold_front_warning = check_cold_front_warning(obs_high_raw, noaa_obs, nws_forecast, local_hour)
if morning_suppressed:
    combined = f'⚠️ Signal suppression active: No obs high + current temp ({round(noaa_obs, 1)}F) is {round(nws_forecast - noaa_obs, 1)}F below NWS forecast ({nws_forecast}F) in morning hours. High may have already occurred — green signals suppressed. Verify manually before betting.'
    if cold_front_warning:
        combined = cold_front_warning
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

if forecast is not None and current is not None:
    consensus_raw = compute_consensus(forecast, current, noaa_final, city, obs_high=obs_high_final)
    bias_correction, bias_n = compute_bias_correction_db(city)
    # V5.7: if NWS is trending up, apply a partial boost to consensus
    trend_boost = round(nws_trend_delta * 0.4, 1) if nws_trend_up and nws_trend_delta else 0.0
    consensus = round(consensus_raw + bias_correction + trend_boost, 1)

    prob_rows, prob_label, used_nbm = bracket_probs_nbm(consensus, ladder_text, city, nbm_percentiles, obs_high=obs_high_final, forecast=forecast)
    # V5.17: apply probability floor to prevent 0.0% declarations on brackets
    # adjacent to consensus. Fixes the bug where adjacent brackets got crushed
    # to zero and the model declared unlikely outcomes "impossible."
    prob_rows = apply_prob_floor(prob_rows, consensus, ladder_text)
    _, sigma = bracket_probs(consensus, ladder_text, city, obs_high=obs_high_final, forecast=forecast)
    call = two_degree_call(consensus, ladder_text, obs_high=obs_high_final)

    # V5.14: MAE color for trust score
    city_mae_val, city_mae_color = get_city_mae_and_color(city)

    save_ok = sb_upsert_prediction(city=city, consensus=consensus, forecast=forecast,
                                    ensemble_mean=ensemble_mean, source_gap=source_gap,
                                    high_uncertainty=high_uncertainty, obs_high=obs_high_final,
                                    bias_correction=bias_correction)
    if not save_ok: st.caption('⚠️ Could not save prediction to database')

    st.markdown('<div class="mph-section-header">🎯 Model Output</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric('Consensus High', str(round(consensus, 1))+' F')
        st.caption('Saved to DB ✓' if save_ok else 'DB save failed')
    with c2: st.metric('2 Degree Call', call or 'none')
    with c3:
        if used_nbm:
            p50 = nbm_percentiles.get('p50', nbm_percentiles.get('p25', '—'))
            st.metric('NBM p50', str(p50)+' F'); st.caption('Bracket probs from NBM percentiles')
        else:
            st.metric('Sigma', str(round(sigma, 2))+' F'); st.caption('Fallback: sigma/normal distribution')
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
        threshold = get_uncertainty_threshold(city)
        st.caption(f'High uncertainty mode — gap {round(source_gap,1)}F exceeds {threshold}F threshold — green signals suppressed')
    if morning_suppressed: st.caption('⚠️ Morning suppression active — no obs high + temp well below forecast')

    # ── V5.17: Reality-check banner ──────────────────────────────────────
    # Fires when current conditions suggest consensus is too low:
    #   - Current temp is already ≥ (consensus - 1F) — actual is overtaking prediction
    #   - obs_high so far ≥ (consensus - 1F) — today's peak has already exceeded us
    #   - NWS forecast is 3F+ above consensus — official forecaster disagrees
    # Catches days like the Dallas 84 vs actual 89 miss we saw.
    _reality_warnings = []
    if current is not None and consensus is not None and current >= consensus - 1.0:
        _reality_warnings.append(
            f"Current temp {round(current,1)}°F is already near/above model consensus {round(consensus,1)}°F "
            f"— actual high may exceed prediction."
        )
    if obs_high_final is not None and consensus is not None and obs_high_final >= consensus - 1.0:
        _reality_warnings.append(
            f"Obs high so far {round(obs_high_final,1)}°F is near/above consensus {round(consensus,1)}°F "
            f"— bracket may have already moved higher."
        )
    if forecast is not None and consensus is not None and forecast >= consensus + 3.0:
        _reality_warnings.append(
            f"NWS forecast {round(forecast,1)}°F is 3°F+ above model consensus {round(consensus,1)}°F "
            f"— official forecaster expects higher."
        )
    if _reality_warnings:
        st.warning('🚨 **Reality check — verify before betting:**\n\n' +
                   '\n\n'.join(['• ' + w for w in _reality_warnings]))

    import pandas as pd
    yes_rows = []
    no_rows = []
    best_bet = best_no_bet = None
    best_edge = best_no_edge = -999
    # V5.14: track trust-score best bets separately
    best_trust_yes = None  # (composite, label, edge, kelly, warnings, stake_label)
    best_trust_no = None

    for label, base_prob in prob_rows:
        ens_prob = next((ensemble_bracket_prob(ensemble_members, lo, hi)
                         for lbl, lo, hi in parse_ladder(ladder_text) if labels_match(lbl, label)), None)
        final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)
        fair = round(final_prob * 100)
        yes_ask = no_ask = None
        if kalshi_markets:
            match = next((m for m in kalshi_markets if labels_match(m[0], label)), None)
            if match: yes_ask, no_ask = match[1], match[2]
        e = edge_cents(final_prob, yes_ask)
        busted = obs_high_final is not None and any(
            labels_match(lbl, label) and hi is not None and obs_high_final > hi + 0.4
            for lbl, lo, hi in parse_ladder(ladder_text))
        conviction_conflict = (conviction_result and
                               is_conflicting_with_conviction(label, conviction_result[1], conviction_result[2], ladder_text))
        signal_icon, signal_text = edge_signal(e, high_uncertainty, morning_suppressed, conviction_conflict)
        kelly = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0
        no_e = no_edge_cents(final_prob, no_ask)
        no_icon, no_text = no_signal(no_e, busted=busted, model_prob=final_prob, no_ask=no_ask,
                                     high_uncertainty=high_uncertainty, morning_suppressed=morning_suppressed,
                                     conviction_conflict=conviction_conflict)
        kelly_no = kelly_bet_no(final_prob, no_ask, bankroll) if no_ask and no_icon == '🟢' else 0.0
        ens_conf = ensemble_confidence(ens_prob) if ens_prob is not None else ''
        ens_tier_for_trust = ensemble_tier_from_confidence(ens_conf)
        edge_str = ('+'+str(e)+'c') if e and e > 0 else (str(e)+'c' if e is not None else 'none')
        no_edge_str = ('+'+str(no_e)+'c') if no_e and no_e > 0 else (str(no_e)+'c' if no_e is not None else 'none')

        # ── V5.14: Trust score for YES and NO sides ─────────────────────
        trust_yes = compute_row_trust(
            city=city, bracket_label=label, direction='YES',
            model_pct=final_prob * 100,
            ensemble_tier=ens_tier_for_trust,
            two_degree_call_str=call or '',
            mae_color=city_mae_color,
            nbm_active=used_nbm,
            nws_forecast_f=nws_forecast, gfs_ensemble_f=ensemble_mean,
            bias_adj_f=bias_correction,
        )
        trust_no = compute_row_trust(
            city=city, bracket_label=label, direction='NO',
            model_pct=(1.0 - final_prob) * 100,
            ensemble_tier=ens_tier_for_trust,
            two_degree_call_str=call or '',
            mae_color=city_mae_color,
            nbm_active=used_nbm,
            nws_forecast_f=nws_forecast, gfs_ensemble_f=ensemble_mean,
            bias_adj_f=bias_correction,
        )
        trust_y_score = round(trust_yes.composite, 1) if trust_yes else None
        trust_y_tier = trust_yes.tier if trust_yes else ''
        trust_y_stake = trust_yes.stake_suggestion_label if trust_yes else ''
        trust_n_score = round(trust_no.composite, 1) if trust_no else None
        trust_n_tier = trust_no.tier if trust_no else ''
        trust_n_stake = trust_no.stake_suggestion_label if trust_no else ''

        # V5.16: two-dot UI
        #   Column 1: Signal (🟢/🟡/⚪/🔴) from edge_signal
        #   Column next-to-last: Trust score (number)
        #   Last column: Ensemble (🔵/🟡/⚪)
        signal_cell = (signal_icon + ' ' + signal_text) if (signal_icon and signal_text) else (signal_icon or '—')
        no_signal_cell = (no_icon + ' ' + no_text) if (no_icon and no_text) else (no_icon or '—')
        trust_y_cell = str(trust_y_score) if trust_y_score is not None else '—'
        trust_n_cell = str(trust_n_score) if trust_n_score is not None else '—'

        yes_rows.append({
            'Signal': signal_cell,
            'Bracket': label + (' BUSTED' if busted else ''),
            'Model %': str(round(final_prob*100, 1))+'%',
            'Mkt Implied %': str(round(yes_ask, 1))+'%' if yes_ask else '—',
            'Fair': str(fair)+'c',
            'YES ask': str(yes_ask)+'c' if yes_ask is not None else '—',
            'Edge': edge_str,
            'Kelly': ('$'+str(kelly)) if kelly > 0 else '—',
            'Trust': trust_y_cell,
            'Ensemble': ens_conf,
        })
        no_rows.append({
            'Signal': no_signal_cell,
            'Bracket': label + (' BUSTED' if busted else ''),
            'Model %': str(round(final_prob*100, 1))+'%',
            'Mkt Implied %': str(round(no_ask, 1))+'%' if no_ask else '—',
            'NO ask': str(no_ask)+'c' if no_ask is not None else '—',
            'NO Edge': no_edge_str,
            'Kelly NO': ('$'+str(kelly_no)) if kelly_no > 0 else '—',
            'Trust': trust_n_cell,
            'Ensemble': ens_conf,
        })

        # V5.8: consensus alignment — don't bet YES below consensus or NO above consensus
        b_lo_d, b_hi_d = next(((lo, hi) for lbl, lo, hi in parse_ladder(ladder_text)
                                if labels_match(lbl, label)), (None, None))
        below_consensus_d = b_hi_d is not None and consensus is not None and b_hi_d < consensus - 2.0
        above_consensus_d = b_lo_d is not None and consensus is not None and b_lo_d > consensus + 2.0

        if (e is not None and e > best_edge and not busted and signal_icon == '🟢'
                and final_prob >= 0.10 and not below_consensus_d):
            best_edge = e
            best_bet = {'label': label, 'edge': e, 'kelly': kelly, 'uncertain': high_uncertainty}
        if busted and no_ask is not None and no_ask <= 5:
            no_e_for_rank = no_e if no_e is not None else 95
            if no_e_for_rank > best_no_edge:
                best_no_edge = no_e_for_rank
                best_no_bet = {'label': label, 'edge': no_e_for_rank, 'kelly': kelly_no, 'busted': True, 'no_ask': no_ask}
        elif (no_e is not None and no_e > best_no_edge and no_icon == '🟢'
              and (1.0 - final_prob) >= 0.10 and not above_consensus_d):
            best_no_edge = no_e
            best_no_bet = {'label': label, 'edge': no_e, 'kelly': kelly_no, 'busted': False, 'no_ask': no_ask}

        # V5.14: trust-tier best bets
        if (trust_yes and trust_yes.tier == 'BET' and e is not None and e >= MIN_EDGE
                and not busted and not below_consensus_d):
            if best_trust_yes is None or trust_yes.composite > best_trust_yes[0]:
                best_trust_yes = (trust_yes.composite, label, e, kelly, trust_yes.warnings, trust_yes.stake_suggestion_label)
        if (trust_no and trust_no.tier == 'BET' and no_e is not None and no_e >= MIN_EDGE
                and not above_consensus_d):
            if best_trust_no is None or trust_no.composite > best_trust_no[0]:
                best_trust_no = (trust_no.composite, label, no_e, kelly_no, trust_no.warnings, trust_no.stake_suggestion_label)

    prob_source = '(NBM percentiles)' if used_nbm else '(sigma/normal fallback)'

    # V5.17: Annotate Signal column with 🟣 edge-pick dot and 🟢🔵 accuracy-pick
    # dot. This makes the dual-dot system visible inside the tables themselves,
    # not just in the best-bet callout below.
    def _annotate_dots(rows, best_edge_pick, accuracy_label=None):
        for r in rows:
            bracket = r['Bracket'].replace(' BUSTED', '')
            dots = ''
            # Edge dot: mark the highest-edge bracket
            if best_edge_pick and labels_match(bracket, best_edge_pick.get('label', '')):
                dots = '🟣 ' + dots
            # Accuracy dot marker (not overwriting — appended to signal icon)
            if accuracy_label and labels_match(bracket, accuracy_label):
                dots = '🎯 ' + dots
            if dots:
                r['Signal'] = dots + r['Signal']
        return rows

    # Find accuracy pick among yes_rows (mirrors _render_best logic for dot only)
    _yes_acc_label = None
    for r in yes_rows:
        sig = r.get('Signal', '')
        ens = r.get('Ensemble', '')
        if sig.startswith('🟢') and '🔵' in ens and 'HIGH' in ens:
            label_plain = r['Bracket'].replace(' BUSTED', '')
            model_pct_str = r.get('Model %', '0%').replace('%','')
            try: model_pct_val = float(model_pct_str)
            except: model_pct_val = 0
            trust_str = r.get('Trust', '—')
            try: trust_val = float(trust_str)
            except: trust_val = 0
            if trust_val >= 75 and model_pct_val >= 35:
                if _yes_acc_label is None: _yes_acc_label = label_plain
    _no_acc_label = None
    for r in no_rows:
        sig = r.get('Signal', '')
        ens = r.get('Ensemble', '')
        if sig.startswith('🟢') and '🔵' in ens and 'HIGH' in ens:
            label_plain = r['Bracket'].replace(' BUSTED', '')
            model_pct_str = r.get('Model %', '0%').replace('%','')
            try: model_pct_val = float(model_pct_str)
            except: model_pct_val = 0
            trust_str = r.get('Trust', '—')
            try: trust_val = float(trust_str)
            except: trust_val = 0
            if trust_val >= 75 and model_pct_val >= 35:
                if _no_acc_label is None: _no_acc_label = label_plain

    yes_rows = _annotate_dots(yes_rows, best_bet, _yes_acc_label)
    no_rows = _annotate_dots(no_rows, best_no_bet, _no_acc_label)

    # V5.16: display rows directly (no _Trust filtering anymore — it's a real column now)
    yes_display = pd.DataFrame(yes_rows)
    no_display = pd.DataFrame(no_rows)

    st.markdown(f'#### 🟢 YES Signals {prob_source}')
    st.dataframe(yes_display, use_container_width=True, hide_index=True)

    # V5.17: ACCURACY-FIRST best-bet callout.
    #
    # Philosophy: the 🟢🔵 combo means "model's honest pick for which bracket
    # will actually hit" — NOT "biggest edge vs market." A separate 🟣 tracks
    # the highest-edge bracket for reference.
    #
    # Accuracy pick criteria:
    #   - 🟢 Signal (positive edge, not busted, not suppressed)
    #   - 🔵 HIGH ensemble
    #   - Trust ≥ 75
    #   - Model probability ≥ 35%
    # When multiple brackets qualify, highest Trust wins.
    #
    # Edge pick criteria (shown as 🟣 dot + secondary info, not recommended):
    #   - Highest edge among brackets with positive edge
    #   - No Trust/Model% threshold
    #   - For sanity check only — can be far from consensus
    def _render_best(side_label, side_suffix, rows_list, trust_map, best_sig_by_edge):
        """
        V5.17: Accuracy-first recommendation.
        side_label: 'YES' or 'NO'
        rows_list: yes_rows or no_rows (the dicts we built)
        trust_map: dict of bracket_label -> (trust, tier, edge, kelly, warnings, model_pct)
        best_sig_by_edge: best_bet or best_no_bet dict (highest-edge green signal)
        """
        # V5.17: Accuracy filter — Trust ≥ 75, Model% ≥ 35, green signal, HIGH ensemble
        accuracy_candidates = []
        for r in rows_list:
            sig = r.get('Signal', '')
            ens = r.get('Ensemble', '')
            if not (sig.startswith('🟢') and '🔵' in ens and 'HIGH' in ens):
                continue
            label = r['Bracket'].replace(' BUSTED', '')
            tdata = trust_map.get(label)
            if not tdata: continue
            trust_score = tdata[0]
            model_pct = tdata[5] if len(tdata) > 5 else 0
            if trust_score >= 75 and model_pct >= 35:
                accuracy_candidates.append({
                    'label': label, 'trust': trust_score,
                    'edge': tdata[2], 'kelly': tdata[3],
                    'warns': tdata[4] or [], 'model_pct': model_pct,
                })

        # Identify the edge pick (for 🟣 dot) — highest edge on any positive-edge bracket
        edge_pick = best_sig_by_edge

        if accuracy_candidates:
            # Sort by Trust DESC
            accuracy_candidates.sort(key=lambda x: (x['trust'], x['model_pct']), reverse=True)
            top = accuracy_candidates[0]
            same_as_edge = (edge_pick and labels_match(top['label'], edge_pick['label']))
            if same_as_edge:
                # Accuracy + edge agree — strongest signal
                st.success(
                    f"🟢🔵🟣 **{top['label']}{side_suffix}** — accuracy AND edge agree · "
                    f"Trust {round(top['trust'],1)} · Model {round(top['model_pct'],1)}% · "
                    f"Edge +{top['edge']}c · Kelly ${top['kelly']}"
                )
            else:
                # Accuracy and edge differ — show both
                st.success(
                    f"🟢🔵 **Accuracy pick {side_label}: {top['label']}{side_suffix}** · "
                    f"Trust {round(top['trust'],1)} · Model {round(top['model_pct'],1)}% · "
                    f"Edge +{top['edge']}c · Kelly ${top['kelly']}"
                )
                if edge_pick and edge_pick.get('label'):
                    st.caption(
                        f"🟣 Edge pick (secondary): {edge_pick['label']}{side_suffix} "
                        f"(+{edge_pick['edge']}c) — mispriced but low Trust. Sanity check only."
                    )
            if len(accuracy_candidates) > 1:
                others = ', '.join([f"{o['label']} (T {round(o['trust'],1)})" for o in accuracy_candidates[1:]])
                st.caption(f"({len(accuracy_candidates)} brackets passed accuracy filter — Trust broke the tie. Also: {others})")
            if top['warns']:
                st.caption('⚠️ ' + ' | '.join(top['warns']))
            return

        # No accuracy pick. Show edge pick only as sanity-check info.
        if edge_pick:
            st.info(
                f"🟣 Edge pick {side_label} (no accuracy match): "
                f"**{edge_pick['label']}{side_suffix}** (+{edge_pick['edge']}c, "
                f"Kelly ${edge_pick['kelly']}) — market mispriced but model isn't "
                f"confident enough (Trust <75 or Model% <35). Verify with your own "
                f"sources before betting."
            )
            return

        st.caption(f"No {side_label} bracket qualifies — skip this run.")

    # Build trust map keyed by bracket label for the callout
    # We need to reconstruct this from the loop above — easiest: store trust tuples
    # per bracket during the loop. But since we already have best_trust_yes/no,
    # let's just build a simple map from yes_rows/no_rows + trust_map (cached).
    # Simpler approach: we kept trust scores in the yes_rows/no_rows' Trust column
    # as display strings, and we tracked best_trust_* separately.
    # Let's keep it simple: reuse best_trust_yes/no as our "ideal" candidate sources.
    # But for the multi-candidate case, we need to re-scan. So do that:
    yes_trust_map = {}
    no_trust_map = {}
    for label, base_prob in prob_rows:
        ens_prob = next((ensemble_bracket_prob(ensemble_members, lo, hi)
                         for lbl, lo, hi in parse_ladder(ladder_text) if labels_match(lbl, label)), None)
        final_prob = blend_probs(base_prob, ens_prob, ensemble_members, city, nbm_active=used_nbm)
        yes_ask = no_ask = None
        if kalshi_markets:
            match = next((m for m in kalshi_markets if labels_match(m[0], label)), None)
            if match: yes_ask, no_ask = match[1], match[2]
        e_yes = edge_cents(final_prob, yes_ask)
        e_no = no_edge_cents(final_prob, no_ask)
        k_yes = kelly_bet(final_prob, yes_ask, bankroll) if yes_ask else 0.0
        k_no_est = kelly_bet_no(final_prob, no_ask, bankroll) if no_ask else 0.0
        ens_prob_for_tier = ens_prob
        ens_conf_tmp = ensemble_confidence(ens_prob_for_tier) if ens_prob_for_tier is not None else ''
        ens_tier_tmp = ensemble_tier_from_confidence(ens_conf_tmp)
        ty = compute_row_trust(city=city, bracket_label=label, direction='YES',
                               model_pct=final_prob*100, ensemble_tier=ens_tier_tmp,
                               two_degree_call_str=call or '', mae_color=city_mae_color,
                               nbm_active=used_nbm, nws_forecast_f=nws_forecast,
                               gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction)
        tn = compute_row_trust(city=city, bracket_label=label, direction='NO',
                               model_pct=(1.0-final_prob)*100, ensemble_tier=ens_tier_tmp,
                               two_degree_call_str=call or '', mae_color=city_mae_color,
                               nbm_active=used_nbm, nws_forecast_f=nws_forecast,
                               gfs_ensemble_f=ensemble_mean, bias_adj_f=bias_correction)
        if ty:
            # V5.17: add model_pct (as percent, 0-100) at index 5 for accuracy filter
            yes_trust_map[label] = (ty.composite, ty.tier, e_yes or 0, round(k_yes,2), ty.warnings, final_prob*100)
        if tn:
            no_trust_map[label] = (tn.composite, tn.tier, e_no or 0, round(k_no_est,2), tn.warnings, (1.0-final_prob)*100)

    _render_best('YES', '', yes_rows, yes_trust_map, best_bet)

    st.markdown(f'#### 🔴 NO Signals {prob_source}')
    st.dataframe(no_display, use_container_width=True, hide_index=True)

    # Busted NO is a special-case fast path — shows classic green NO bet callout
    if best_no_bet and best_no_bet.get('busted'):
        st.success('🟢 Best NO Bet: **' + best_no_bet['label'] + ' NO** · BUSTED bracket · '
                   'NO ask: ' + str(best_no_bet['no_ask']) + 'c · Kelly: $' + str(best_no_bet['kelly']))
    else:
        _render_best('NO', ' NO', no_rows, no_trust_map, best_no_bet)

    # V5.16: Trust score detail expander stays available as a reference view
    with st.expander('🔐 View Trust Score Details', expanded=False):
        st.caption('Raw trust composite (0-100) per bracket. Tier: BET ≥75, CAUTION 55-74, SKIP <55.')
        trust_detail_rows = []
        for r_y, r_n in zip(yes_rows, no_rows):
            trust_detail_rows.append({
                'Bracket': r_y.get('Bracket', ''),
                'YES Trust': r_y.get('Trust', '—'),
                'NO Trust': r_n.get('Trust', '—') if r_n else '—',
                'Model %': r_y.get('Model %', '—'),
                'Ensemble': r_y.get('Ensemble', ''),
            })
        st.dataframe(pd.DataFrame(trust_detail_rows), use_container_width=True, hide_index=True)

    parsed = parse_ladder(ladder_text)
    top_b = next((b for b in parsed if b[2] is None), None)
    bot_b = next((b for b in parsed if b[1] is None), None)
    if (top_b and consensus > top_b[1]+5) or (bot_b and bot_b[2] is not None and consensus < bot_b[2]-5):
        st.warning('Ladder does not cover consensus of '+str(round(consensus, 1))+'F — update brackets.')

else:
    st.error('NWS forecast unavailable. Use manual override or try refreshing.' if forecast is None
             else 'Current temperature unavailable — cannot compute consensus.')

# ── Calibration Panel ─────────────────────────────────────────────────────────
st.markdown('---')
st.markdown('<div class="mph-section-header">📈 Calibration & Settlement History</div>', unsafe_allow_html=True)

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
        with m1: st.metric('Settled Days', len(complete))
        with m2: st.metric('MAE', str(mae)+'F' if mae else 'n/a')
        with m3:
            st.metric('Avg Error', ('+' if avg_err and avg_err > 0 else '')+str(avg_err)+'F' if avg_err else 'n/a',
                      delta='warm bias' if avg_err and avg_err < -0.5 else 'cold bias' if avg_err and avg_err > 0.5 else 'calibrated')
        with m4: st.metric('Within ±2F', str(within_2)+'%' if within_2 else 'n/a')
        hist_df = pd.DataFrame([{
            'Date': r['date'], 'Consensus': r.get('consensus'), 'Actual': r.get('actual'),
            'Error': ('+' if r['error'] > 0 else '') + str(r['error']) + 'F' if r.get('error') is not None else '',
            'Ensemble': r.get('ensemble_mean'), 'Uncertain': '⚠️' if r.get('high_uncertainty') else '',
        } for r in sorted(complete, key=lambda x: x['date'], reverse=True)])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info('No settled history yet for ' + city + '.')
    if pending:
        st.caption(str(len(pending)) + ' prediction(s) pending: ' + ', '.join(r['date'] for r in pending))

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
                    today_rows.append({'city': c, 'date': today_str, 'consensus': consensus,
                                       'forecast': weather['nws_fc'], 'ensemble_mean': weather['ensemble_mean'],
                                       'source_gap': weather['source_gap'], 'high_uncertainty': weather['high_uncertainty'],
                                       'bias_correction': bc})
                    newly_saved.append(c)
            except Exception: pass
            time.sleep(0.4)
        fill_status.empty()
        if newly_saved: st.caption(f'✅ Filled in {len(newly_saved)} missing cities: {", ".join(newly_saved)}')

    if today_rows:
        summary_rows = []
        # V5.16: sort all, but skip hidden cities from the dashboard display.
        # (Hidden cities still got predicted & saved in the missing_cities loop
        #  above — we just don't show them in the main dashboard table.)
        for r in sorted(today_rows, key=lambda x: x['city']):
            c = r['city']
            if c in HIDDEN_CITIES:
                continue
            consensus_val = r.get('consensus')
            ladder = saved_ladders.get(c, DEFAULT_LADDERS.get(c, ''))
            cached_markets, _ = get_cached_prices(c)
            obs_h = r.get('obs_high')
            high_unc = r.get('high_uncertainty', False)
            bc_val = r.get('bias_correction', 0.0)
            members = nbm_pcts = c_temp = c_fc = None
            c_hour = 12
            try:
                cached_wx = fetch_city_weather(c)
                if cached_wx:
                    members = cached_wx.get('ensemble_members')
                    nbm_pcts = cached_wx.get('nbm_percentiles')
                    c_temp = cached_wx.get('current_temp')
                    c_fc = cached_wx.get('nws_fc')
                    c_hour = cached_wx.get('local_hour', 12)
            except Exception: pass

            ens_key = ensemble_overall_confidence(members, consensus_val, ladder)
            nbm_status = '✅ NBM' if nbm_pcts else '📊 Sigma'

            # V5.8: Check price cache age — grey out signals if prices are stale (>2 hours)
            _, price_fetched_at = get_cached_prices(c)
            price_age_min = round((time.time() - price_fetched_at) / 60) if price_fetched_at else 999
            prices_stale = price_age_min > 120  # 2 hours

            if prices_stale:
                best_yes = f'⚪ stale ({price_age_min}m)'
                best_no = f'⚪ stale ({price_age_min}m)'
            else:
                best_yes, best_no = get_city_best_signals(
                    c, consensus_val, ladder, members, cached_markets, obs_h, high_unc, bankroll,
                    nbm_percentiles=nbm_pcts, current_temp=c_temp, nws_forecast=c_fc, local_hour=c_hour)

            # Add price age indicator to signal if fresh
            if not prices_stale and price_fetched_at:
                age_str = f'{price_age_min}m ago' if price_age_min > 0 else 'just now'
            else:
                age_str = None
            bias_str = ('+' if bc_val and bc_val > 0 else '') + str(bc_val) + 'F' if bc_val and bc_val != 0.0 else '—'

            # V5.7: MAE column — rolling 14-day window to match bias correction
            city_rows = sb_fetch_city(c)
            city_complete = [row for row in city_rows if row.get('actual') is not None and row.get('error') is not None]
            if city_complete:
                recent_14 = city_complete[-14:]
                city_errors = [abs(row['error']) for row in recent_14]
                city_mae = round(sum(city_errors) / len(city_errors), 1)
                mae_icon = '✅' if city_mae < 2.5 else '🟡' if city_mae < 4.0 else '🔴'
                mae_str = f'{mae_icon} {city_mae}F'
            else:
                mae_str = '—'

            summary_rows.append({
                'City': c, 'Consensus': str(consensus_val)+'F' if consensus_val else '—',
                'NWS': str(r.get('forecast', ''))+'F' if r.get('forecast') else '—',
                'GFS': str(r.get('ensemble_mean', ''))+'F' if r.get('ensemble_mean') else '—',
                'Gap': str(round(r['source_gap'], 1))+'F' if r.get('source_gap') else '—',
                '⚠️': '⚠️' if r.get('high_uncertainty') else '✅',
                'Ens Key': ens_key if ens_key else '—', 'Prob Src': nbm_status,
                'Bias Adj': bias_str, 'MAE': mae_str,
                'Prices': f'⚠️ {price_age_min}m' if prices_stale else (age_str or '—'),
                'Best YES': best_yes, 'Best NO': best_no,
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        n_saved_total = len(today_rows)
        n_yes = sum(1 for r in summary_rows if r['Best YES'] != '—' and '⚪' not in r['Best YES'])
        n_no = sum(1 for r in summary_rows if r['Best NO'] != '—' and '⚪' not in r['Best NO'])
        n_nbm = sum(1 for r in summary_rows if '✅' in r.get('Prob Src', ''))
        n_stale = sum(1 for r in summary_rows if '⚠️' in r.get('Prices', ''))
        status_icon = '✅' if n_saved_total == 18 else '⏳'
        stale_str = f' | ⚠️ {n_stale} stale prices' if n_stale > 0 else ''
        st.caption(f'{status_icon} {n_saved_total}/18 cities | 🟢 {n_yes} YES signals | 🟢 {n_no} NO signals | ✅ {n_nbm} NBM active today{stale_str}')
    else:
        st.info('Loading predictions for all cities — this panel fills itself in automatically.')

# ── V5.16: Source Accuracy Report ─────────────────────────────────────────────
# Compares MAE of each source (NWS-alone, GFS-alone, our consensus blend)
# per-city across all settled days. Shows which cities the blend helps,
# which cities NWS-alone wins, and which cities are unstable regardless.
# Updates every time the app loads — so you can watch V5.16 changes
# take effect on each city over 1-2 weeks.
st.markdown('---')
with st.expander('🔬 Source Accuracy Report — Per-City MAE', expanded=False):
    st.caption('Lower MAE is better. Compares NWS forecast alone vs GFS ensemble alone '
               'vs our current blend (after bias correction). Use this to see which '
               'cities the blend is actually helping.')
    import pandas as pd
    all_settled = [r for r in sb_fetch_all()
                   if r.get('actual') is not None
                   and r.get('forecast') is not None
                   and r.get('consensus') is not None]
    if len(all_settled) < 20:
        st.info('Not enough settled data yet for a meaningful accuracy report. '
                'Check back after ~1 week of data collection.')
    else:
        from collections import defaultdict
        per_city = defaultdict(list)
        for r in all_settled:
            per_city[r['city']].append(r)
        report_rows = []
        for city_name, rows in per_city.items():
            n = len(rows)
            if n < 5: continue
            nws_errors = [abs(r['actual'] - r['forecast']) for r in rows if r.get('forecast') is not None]
            gfs_errors = [abs(r['actual'] - (r['ensemble_mean'] or r['forecast'])) for r in rows]
            blend_errors = [abs(r['actual'] - r['consensus']) for r in rows]
            mean_bias = sum(r['actual'] - r['consensus'] for r in rows) / n
            within_2 = sum(1 for e in blend_errors if e <= 2)
            mae_nws = round(sum(nws_errors) / len(nws_errors), 2) if nws_errors else None
            mae_gfs = round(sum(gfs_errors) / len(gfs_errors), 2) if gfs_errors else None
            mae_blend = round(sum(blend_errors) / len(blend_errors), 2)
            mode = CITY_PREDICTION_MODE.get(city_name, 'full_blend')
            hidden = '🔴 HIDDEN' if city_name in HIDDEN_CITIES else ''
            tier = ('🟢' if mae_blend < 2.0 else
                    '🟡' if mae_blend < 2.5 else
                    '🟠' if mae_blend < 3.5 else '🔴')
            report_rows.append({
                'Tier': tier,
                'City': city_name + (' ' + hidden if hidden else ''),
                'Days': n,
                'MAE (blend)': mae_blend,
                'MAE (NWS alone)': mae_nws if mae_nws is not None else '—',
                'MAE (GFS alone)': mae_gfs if mae_gfs is not None else '—',
                'Bias': round(mean_bias, 2),
                'Within 2F': f'{within_2}/{n}',
                'Mode': mode,
            })
        if report_rows:
            df = pd.DataFrame(report_rows)
            df = df.sort_values('MAE (blend)', ascending=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption('**Tier legend:** 🟢 MAE <2F excellent · 🟡 <2.5F good · 🟠 <3.5F caution · 🔴 ≥3.5F unsafe')
            st.caption('**Mode legend:** `full_blend` = current V5.15 formula · `nws_only` = NWS forecast + bias correction only (simpler, used when blend hurts)')

# ── V5.15: Personal Bet Log (Supabase-backed) ────────────────────────────────
# Migrated from local JSON to Supabase `bets` table. Adds Edit and Delete
# buttons per row plus a persistent "logged" confirmation banner.
st.markdown('---')
st.markdown('<div class="mph-section-header">📒 Personal Bet Log</div>', unsafe_allow_html=True)

_bet_log_pw = st.text_input('Enter password to access bet log', type='password', key='bet_log_pw')
_correct_pw = '2974'

if _bet_log_pw == _correct_pw:
    # V5.15: persistent "logged" banner, survives the st.rerun() after saving
    if st.session_state.get('_last_logged_bet'):
        lb = st.session_state['_last_logged_bet']
        st.success(f"✅ Logged: **{lb['city']} {lb['bracket']} {lb['direction']}** · "
                   f"${lb['amount']} @ {lb['price']}c · {lb['result']}")
        st.caption('(Shows once per session after logging. Clear this or log another bet.)')
        if st.button('Clear confirmation', key='clear_log_confirm'):
            st.session_state['_last_logged_bet'] = None
            st.rerun()

    with st.expander('📒 Log a Bet', expanded=False):
        bl1, bl2, bl3 = st.columns(3)
        with bl1:
            log_city = st.selectbox('City', list(CITIES.keys()), key='log_city')
            log_direction = st.radio('Direction', ['YES', 'NO'], horizontal=True, key='log_dir')
        with bl2:
            log_bracket = st.text_input('Bracket (e.g. 77-78)', key='log_bracket')
            log_amount = st.number_input('Amount ($)', min_value=1.0, max_value=500.0, value=25.0, step=1.0, key='log_amount')
        with bl3:
            log_price = st.number_input('Price paid (cents)', min_value=1, max_value=99, value=40, key='log_price')
            log_result = st.radio('Result', ['Pending', 'Won', 'Lost'], horizontal=True, key='log_result')

        if st.button('Log Bet', type='primary'):
            # V5.15: validate + insert into Supabase
            if not log_bracket or not log_bracket.strip():
                st.error('Bracket is required (e.g. 77-78, or "74 or below").')
            else:
                profit_val = 0.0
                payout_val = 0.0
                if log_result == 'Won' and log_price > 0:
                    profit_val = round(log_amount * (100 - log_price) / log_price, 2)
                    payout_val = profit_val
                elif log_result == 'Lost':
                    profit_val = -log_amount
                bet_row = {
                    'date': get_eastern_date(),
                    'city': log_city,
                    'bracket': log_bracket.strip(),
                    'direction': log_direction,
                    'amount': float(log_amount),
                    'price': int(log_price),
                    'result': log_result,
                    'payout': payout_val,
                    'profit': profit_val,
                    'actual': None,
                    'settled_at': None,
                }
                inserted = sb_insert_bet(bet_row)
                if inserted:
                    st.session_state['_last_logged_bet'] = {
                        'city': log_city, 'bracket': log_bracket.strip(),
                        'direction': log_direction, 'amount': log_amount,
                        'price': log_price, 'result': log_result,
                    }
                    st.rerun()

    with st.expander('📊 Bet Log History', expanded=False):
        bet_log = sb_fetch_bets()
        if bet_log:
            import pandas as pd
            total_bets = len(bet_log)
            settled = [b for b in bet_log if b.get('result') != 'Pending']
            won = [b for b in settled if b.get('result') == 'Won']
            total_wagered = sum(float(b.get('amount') or 0) for b in settled)
            total_profit = sum(float(b.get('profit') or 0) for b in settled)
            win_rate = round(100 * len(won) / len(settled)) if settled else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.metric('Total Bets', total_bets)
            with m2: st.metric('Win Rate', f'{win_rate}%')
            with m3: st.metric('Total Wagered', f'${round(total_wagered, 2)}')
            with m4: st.metric('Total P&L', f'{"+" if total_profit >= 0 else ""}{round(total_profit, 2)}')
            with m5: st.metric('Pending', len([b for b in bet_log if b.get('result') == 'Pending']))

            log_df = pd.DataFrame([{
                'Date': b.get('date'),
                'City': b.get('city'),
                'Bracket': b.get('bracket'),
                'Dir': b.get('direction'),
                'Amount': f"${float(b.get('amount') or 0):g}",
                'Price': f"{int(float(b.get('price') or 0))}c",
                'Result': b.get('result'),
                'Actual': (f"{b.get('actual')}F" if b.get('actual') is not None else '—'),
                'P&L': (('+' if (b.get('profit') or 0) >= 0 else '') +
                        f"${round(float(b.get('profit') or 0), 2)}"
                        if b.get('result') != 'Pending' else '—'),
            } for b in reversed(bet_log)])
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            # V5.15: EDIT / DELETE controls per row
            st.markdown('**Edit or Delete bets:**')
            st.caption('💡 V5.15: Fix typos on pending bets. Delete ghost entries. '
                       'Changes save directly to Supabase.')

            for b in reversed(bet_log):
                bet_id = b.get('id')
                if bet_id is None: continue
                summary = (f"#{bet_id} — {b.get('date')} · {b.get('city')} · "
                           f"{b.get('bracket')} · {b.get('direction')} · "
                           f"${float(b.get('amount') or 0):g} @ {int(float(b.get('price') or 0))}c · "
                           f"{b.get('result')}")

                with st.expander(summary, expanded=False):
                    # --- Edit form ---
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        new_city = st.selectbox(
                            'City', list(CITIES.keys()),
                            index=list(CITIES.keys()).index(b.get('city')) if b.get('city') in CITIES else 0,
                            key=f'edit_city_{bet_id}')
                        new_direction = st.radio(
                            'Direction', ['YES', 'NO'], horizontal=True,
                            index=0 if b.get('direction') == 'YES' else 1,
                            key=f'edit_dir_{bet_id}')
                    with ec2:
                        new_bracket = st.text_input(
                            'Bracket', value=b.get('bracket', ''),
                            key=f'edit_bracket_{bet_id}')
                        new_amount = st.number_input(
                            'Amount ($)', min_value=1.0, max_value=500.0,
                            value=float(b.get('amount') or 1.0), step=1.0,
                            key=f'edit_amount_{bet_id}')
                    with ec3:
                        new_price = st.number_input(
                            'Price (cents)', min_value=1, max_value=99,
                            value=int(float(b.get('price') or 40)),
                            key=f'edit_price_{bet_id}')
                        new_result = st.radio(
                            'Result', ['Pending', 'Won', 'Lost'], horizontal=True,
                            index=['Pending', 'Won', 'Lost'].index(b.get('result')) if b.get('result') in ('Pending', 'Won', 'Lost') else 0,
                            key=f'edit_result_{bet_id}')

                    bc1, bc2, bc3 = st.columns([1, 1, 4])
                    with bc1:
                        if st.button('Save Edit', key=f'save_{bet_id}', type='primary'):
                            profit_val = 0.0
                            payout_val = 0.0
                            if new_result == 'Won' and new_price > 0:
                                profit_val = round(new_amount * (100 - new_price) / new_price, 2)
                                payout_val = profit_val
                            elif new_result == 'Lost':
                                profit_val = -new_amount
                            updates = {
                                'city': new_city,
                                'bracket': new_bracket.strip(),
                                'direction': new_direction,
                                'amount': float(new_amount),
                                'price': int(new_price),
                                'result': new_result,
                                'profit': profit_val,
                                'payout': payout_val,
                            }
                            if sb_update_bet(bet_id, updates):
                                st.success('✅ Bet updated.')
                                st.rerun()
                            else:
                                st.error('Update failed.')
                    with bc2:
                        # Two-step delete — confirm before actually deleting
                        confirm_key = f'confirm_del_{bet_id}'
                        if st.session_state.get(confirm_key):
                            if st.button('⚠️ Confirm Delete', key=f'del_{bet_id}'):
                                if sb_delete_bet(bet_id):
                                    st.session_state[confirm_key] = False
                                    st.success('🗑️ Bet deleted.')
                                    st.rerun()
                                else:
                                    st.error('Delete failed.')
                        else:
                            if st.button('🗑️ Delete', key=f'del_btn_{bet_id}'):
                                st.session_state[confirm_key] = True
                                st.rerun()
                    with bc3:
                        if st.session_state.get(f'confirm_del_{bet_id}'):
                            st.caption('⚠️ Click "Confirm Delete" to permanently remove this bet, or collapse this row to cancel.')
        else:
            st.info('No bets logged yet. Use the form above to log your first bet.')
elif _bet_log_pw:
    st.error('Incorrect password.')
