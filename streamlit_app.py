"""
app.py — MPH Weather, V6.4

V6.4 — FROM LOOKUP TOOL TO DECISION TOOL (2026-09-13)
======================================================
V6.3 was accurate and nearly useless. It answered "what has this station done
today?" for one city at a time — and by the time you read it, the market had
already priced the same observation. Reviewing four days of actual use, the app
got opened at one recurring moment: a bracket boundary was live and the question
was whether the peak was in.

Everything V6.4 adds serves that moment, or removes work being done by hand.

1. DECISION BOARD — all 20 cities at once, ranked by undecidedness.
   The old flow was: pick a city from a dropdown, read one number, repeat
   twenty times. The board computes, for every city, whether the quantization
   band STRADDLES a settlement boundary — whether today's number is still
   genuinely undecided — and sorts those to the top.

   ⚠️ STRADDLE IS THE WHOLE POINT. A feed max of 98.6F is 37C exactly, so the
   true peak is 97.7-99.5F. That spans 98 AND 99: two different brackets, and
   the station cannot tell you which. A max of 93.9 sits inside one integer and
   is not straddling anything. The board makes that distinction visible without
   twenty clicks.

2. MARKET TIMELINE — what the favorite did, hour by hour.
   favorites_snapshots has collected since 2026-09-11 at 11:00, 12:00, 13:00,
   14:00, 15:00 and 17:00, plus the three bet windows. That data produced the
   most interesting result in a fortnight and it took hand-written SQL to see.
   Now it is a table: bracket and ask at each hour, with the row marked where
   the favorite CHANGED.

   Observed 2026-09-12, which is why this exists: San Antonio flipped from
   101-102 to 99-100 between 11:00 and 13:00 and then sat at 63c for three
   straight hours. Atlanta flipped at 14:00 and was still 56c. Miami flipped at
   14:00 and was ALREADY 84c. Same event, completely different tradeability,
   invisible without the timeline.

3. DAILY SCORECARD — replaces counting wins by hand off a photograph.
   The paper Daily Capture Grid was being photographed each night and scored by
   eye. That produced at least two errors, both the same mistake:

   ⚠️ TAIL BRACKETS ARE NOT RANGES. `bracket_lo`/`bracket_hi` are null on one
   side for "X or below" and "X or above". Read off a photo, "63↓" looks like a
   63-64 RANGE and gets scored backwards — that turned a Seattle WIN into a
   loss on 2026-09-13 (settled 60, bracket was 63-or-below) and did the same to
   New Orleans on 2026-09-11. score_bracket() handles all three shapes
   explicitly and falls back to parsing the label when the bound columns are
   null, which they are on every row written before 2026-09-08.

4. KILL-LINE STATUS — the number that actually governs what happens next.
   Each window against its OWN break-even at n=50. This lived in an occasional
   query while morning sat near the line for a week.

⚠️ WHAT V6.4 DELIBERATELY DOES NOT DO. No recommendation, no edge score, no
"BET THIS" panel, no probability of any kind. The 3,356-line version that did
all of that produced 205 losing paper bets across four tags, and naked
consensus beat its own bracket picks by 19 points on 626 city-days. Every
filter tested since has failed: per-city four separate times, bracket-change
under price control, price bands on the daily split, dominance for lack of any
variation to test. This file shows what is true. It does not say what to do.

--- V6.3 documentation below, unchanged and still true ---

THE GREEN BOX THAT LIED (2026-09-09 night)
At 5:19pm ET the panel showed San Antonio 81.0F in a green box. The station had
reached the 98.6F step at ~4pm local. The row had been written at about 13:24Z
— 8:24am local — and had not moved in nine hours. At 8:39pm the SAME numbers
(81.0 / 80.1 / 17 obs / 11 METARs) appeared under Los Angeles.

1. NOTHING IN THIS FILE MEASURED FRESHNESS.
   `obs_age_min` is stamped by the poller at WRITE time. A row written at
   13:24Z saying "obs_age_min: 4.2" still says 4.2 at 22:00Z. The header clock
   meanwhile rendered datetime.now(ET), so the page showed 8:39pm above a
   reading from breakfast and called it live.

   ⚠️ THE .hero CLASS HARDCODED `border:2px solid #00ff88`. There was no code
   path in V6.2 that could produce a non-green headline. The box was green
   because it is always green, not because the data was good.

   V6.3 computes age from `updated_at` against now, EVERY RENDER. Past
   STALE_HARD_SEC the headline number is REPLACED by the word STALE. Not
   dimmed, not caveated — replaced. A number you cannot trust is worse than no
   number, because you will act on it.

2. THE OFF-GRID TEST WAS BACKWARDS — and it is why a corrupt value produced
   the app's MOST confident output. V6.2 said: on the grid -> quantized, show a
   window; off the grid -> "native Fahrenheit tenths, the max is exact".

   Real ASOS values land ON the grid. 81.0F is not on it (27C = 80.6). So a
   stale, duplicated or corrupt value is EXACTLY the kind that reads as
   off-grid — and V6.2 responded by dropping its error bars and printing a flat
   red BROKEN. Off-grid is a symptom of bad data, not evidence of precision.
   Only KBOS and KMSP genuinely transmit tenths.

3. DUPLICATE-ROW DETECTOR. The city selector was correct — it filters on city —
   so identical numbers under two cities means the DATABASE holds identical
   rows. This file cannot fix the poller but it refuses to pretend.

4. TIMEZONE BUG IN THE DATE FILTER. `local_date = eq. today_et()` filtered
   every city by the EASTERN date while the poller writes each row under the
   STATION's local date. Between 9pm ET and midnight PT the Pacific cities
   silently disappeared.

THE QUANTIZATION BUG (V6.2)
V6.1 shipped a bracket check that read a feed value as a measurement. On
Atlanta it said "BROKEN — max 89.6 is already above 89." False. 89.6F is
EXACTLY 32C, so the true peak was anywhere in 88.7-90.5F. The Kalshi ladder at
that moment: 88-89 at 51%, 90-91 at 49%. The market had it right and the app
was calling a coin flip a certainty.

⚠️ EVERY VALUE THAT LOOKED PRECISE WAS ON THE GRID.
    95.0 = 35C    96.8 = 36C    98.6 = 37C
    73.4 = 23C    75.2 = 24C    89.6 = 32C

⚠️ THE UNRESOLVED SHARE ASSUMES A UNIFORM DISTRIBUTION INSIDE THE BAND.
It is not uniform. If the bracketing hourly METARs both sit below the band, the
peak almost certainly clipped the BOTTOM of it rather than running to the top.
Read the share as an upper bound on the bad outcome, not a probability. San
Antonio 2026-09-09: this panel said 44%, the market said 26%, and the market
was closer.

THE FEED MAX LEADS
Boston 2026-09-09: feed max 73.4F on 201 obs vs precise max 69.98F on 9 METARs.
Nine hourly samples cannot catch a peak between :51 reports.

    day_max_f      every ~5 min, 200+ samples. CATCHES THE PEAK. Quantized to
                   whole degrees Celsius on most stations.
    precise_max_f  exact to a tenth, 9-14 samples a day. MISSES PEAKS.
                   It is a FLOOR, never the answer.

Secrets: supabase.url, supabase.key, app_password (optional).
"""

import re
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title='MPH Weather', layout='wide', page_icon='🌡️')

ET = pytz.timezone('America/New_York')

# ── Freshness thresholds ─────────────────────────────────────────────────────
# The poller runs every 5 minutes. Past SOFT is worth flagging; past HARD it is
# not a number, it is a memory.
STALE_SOFT_SEC = 8 * 60
STALE_HARD_SEC = 15 * 60

# Stations that genuinely transmit Fahrenheit tenths. Everything else landing
# off the Celsius grid is suspect, not precise.
NATIVE_TENTHS = {'KBOS', 'KMSP'}

# ⚠️ LOCAL TIME, NOT EASTERN. Peak is a local-clock phenomenon: the 17:00 ET
# snapshot is 2pm in Seattle and 5pm in Miami, and those are nothing alike.
# On 2026-09-11 the 17:00 snapshot caught five Eastern cities already resolved.
CITY_TZ = {
    'Atlanta': 'America/New_York',        'Austin': 'America/Chicago',
    'Boston': 'America/New_York',         'Washington DC': 'America/New_York',
    'Denver': 'America/Denver',           'Dallas': 'America/Chicago',
    'Houston': 'America/Chicago',         'Las Vegas': 'America/Los_Angeles',
    'Los Angeles': 'America/Los_Angeles', 'Chicago': 'America/Chicago',
    'Miami': 'America/New_York',          'Minneapolis': 'America/Chicago',
    'New Orleans': 'America/Chicago',     'New York': 'America/New_York',
    'Oklahoma City': 'America/Chicago',   'Philadelphia': 'America/New_York',
    'Phoenix': 'America/Phoenix',         'San Antonio': 'America/Chicago',
    'Seattle': 'America/Los_Angeles',     'San Francisco': 'America/Los_Angeles',
}

# Clock order for merging the two tables into one timeline. The three bet
# windows come from favorites_bets, the six T-labels from favorites_snapshots.
SLOT_ORDER = [
    ('MORNING', 1030), ('T1100', 1100), ('MIDDAY', 1200), ('T1200', 1201),
    ('T1300', 1300), ('T1400', 1400), ('T1500', 1500),
    ('AFTERNOON', 1600), ('T1700', 1700),
]
SLOT_RANK = {name: rank for name, rank in SLOT_ORDER}

# ⚠️ A LADDER SUMMING FAR BELOW 1.0 IS A DEAD MARKET, NOT A CHEAP ONE.
# 2026-09-11 T1700: Houston, Atlanta, Miami, New York and New Orleans all came
# back with sigma_p 0.05 and a 1c favorite. Those are resolved markets, and
# reading them as prices would drag any hour-vs-hour comparison.
LIVE_SIGMA_MIN = 0.80

# ⚠️ THE LINK MUST POINT AT THE SETTLEMENT STATION, NOT THE CITY.
# This is the Hobby-vs-Bush, Midway-vs-O'Hare trap in a different form. Proven
# on 2026-09-14: the phone app pinned to Miami Intl Airport (KMIA) read 82F
# with a forecast high of 90, while weather.com/us/florida/city/miami/today read
# 84F with a high of 88. Same company, same minute, two degrees apart on both
# numbers — because one is the station and one is downtown.
#
# ⚠️ AN EARLIER BUILD USED weather.com/weather/today/l/{lat},{lon} AND WAS WRONG.
# weather.com snaps a coordinate to its nearest NAMED PLACE, which for the
# station's own lat/lon is usually the city, not the airport. Those links looked
# authoritative and were off by a couple of degrees, which on a 2F bracket is
# the entire question. Removed rather than left in.

# NWS point forecast, keyed by ICAO. EXACT for every station, no guessing —
# the code IS the identifier. Same forecast source the consensus table uses.
CITY_STATION = {
    'Atlanta': 'KATL',        'Austin': 'KAUS',
    'Boston': 'KBOS',         'Washington DC': 'KDCA',
    'Denver': 'KDEN',         'Dallas': 'KDFW',
    'Houston': 'KHOU',        # HOBBY, not Bush
    'Las Vegas': 'KLAS',      'Los Angeles': 'KLAX',
    'Chicago': 'KMDW',        # MIDWAY, not O'Hare
    'Miami': 'KMIA',          'Minneapolis': 'KMSP',
    'New Orleans': 'KMSY',    'New York': 'KNYC',   # Central Park, not an airport
    'Oklahoma City': 'KOKC',  'Philadelphia': 'KPHL',
    'Phoenix': 'KPHX',        'San Antonio': 'KSAT',
    'Seattle': 'KSEA',        'San Francisco': 'KSFO',
}

# ⚠️ VERIFIED TWC URLS. Every one of these was opened and confirmed to name the
# settlement station, 2026-09-14. They are NOT generated and must not be.
#
# The slug is the airport's full formal name and it is unguessable. Three of
# these would have been WRONG under any naming convention a machine would pick:
#
#   Houston       william-p-hobby-airport        NOT houston-hobby / Bush
#   Chicago       chicago-midway-international   NOT O'Hare
#   New York      /poi/central-park              NOT an airport at all
#
# And the city segment is often not the city the market is named for:
#   Washington DC -> /us/virginia/arlington/       (Reagan National)
#   Boston        -> /us/massachusetts/east-boston/ (Logan)
#   Dallas        -> /us/texas/grapevine/           (DFW)
#   New Orleans   -> /us/louisiana/kenner/          (Louis Armstrong)
#   Seattle       -> /us/washington/seatac/         (Sea-Tac)
#
# Las Vegas is harry-reid-international, not McCarran — renamed in 2021.
#
# Store the BASE url with no view suffix; twc_url() appends /today,
# /hourbyhour or /tenday.
TWC_VERIFIED = {
    'Atlanta':       'https://weather.com/us/georgia/atlanta/airport/hartsfield-jackson-atlanta-international-airport',
    'Austin':        'https://weather.com/us/texas/austin/airport/austin-bergstrom-international-airport',
    'Boston':        'https://weather.com/us/massachusetts/east-boston/airport/logan-international-airport',
    'Chicago':       'https://weather.com/us/illinois/chicago/airport/chicago-midway-international-airport',
    'Dallas':        'https://weather.com/us/texas/grapevine/airport/dallas-fort-worth-international-airport',
    'Denver':        'https://weather.com/us/colorado/denver/airport/denver-international-airport',
    'Houston':       'https://weather.com/us/texas/houston/airport/william-p-hobby-airport',
    'Las Vegas':     'https://weather.com/us/nevada/las-vegas/airport/harry-reid-international-airport',
    'Los Angeles':   'https://weather.com/us/california/los-angeles/airport/los-angeles-international-airport',
    'Miami':         'https://weather.com/us/florida/miami/airport/miami-international-airport',
    'Minneapolis':   'https://weather.com/us/minnesota/minneapolis/airport/minneapolis-saint-paul-international-airport-wold-chamberlain-field',
    'New Orleans':   'https://weather.com/us/louisiana/kenner/airport/louis-armstrong-new-orleans-international-airport',
    'New York':      'https://weather.com/us/new-york/new-york-city/poi/central-park',
    'Oklahoma City': 'https://weather.com/us/oklahoma/oklahoma-city/airport/will-rogers-world-airport',
    'Philadelphia':  'https://weather.com/us/pennsylvania/philadelphia/airport/philadelphia-international-airport',
    'Phoenix':       'https://weather.com/us/arizona/phoenix/airport/phoenix-sky-harbor-international-airport',
    'San Antonio':   'https://weather.com/us/texas/san-antonio/airport/san-antonio-international-airport',
    'San Francisco': 'https://weather.com/us/california/san-francisco/airport/san-francisco-international-airport',
    'Seattle':       'https://weather.com/us/washington/seatac/airport/seattle-tacoma-international-airport',
    'Washington DC': 'https://weather.com/us/virginia/arlington/airport/ronald-reagan-washington-national-airport',
}


def nws_url(city):
    """NWS point forecast for the settlement station. Exact, via ICAO code."""
    stn = CITY_STATION.get(city)
    if not stn:
        return None
    return f'https://forecast.weather.gov/zipcity.php?inputstring={stn}'


def twc_url(city, view='today'):
    """weather.com link, ONLY for stations whose URL has been verified.

    Returns None for anything unverified — a missing link is better than one
    pointing at the wrong airport.
    """
    base = TWC_VERIFIED.get(city)
    if not base:
        return None
    return f'{base}/{view}'


KILL_LINE_N = 50

# ⚠️ THE POLLER ONLY RUNS 9AM-9PM ET. Outside that window every row is stale by
# design, and V6.4's first cut blanked all twenty with ⛔ and no explanation —
# a screen that looks broken but is merely closed. Staleness during trading
# hours is a fault; staleness at 10pm is the schedule.
#
# The distinction matters because the two call for opposite treatment. A frozen
# row at 2pm can get you into a trade on a number from breakfast, which is what
# happened on 2026-09-09. A frozen row at 10pm cannot mislead anyone into
# anything — the markets it describes have settled.
POLLER_START_HOUR = 9
POLLER_END_HOUR = 21


def poller_should_be_running(now=None):
    """True when obs_live is scheduled to be writing right now."""
    n = now or datetime.now(ET)
    return POLLER_START_HOUR <= n.hour < POLLER_END_HOUR


def today_et():
    return datetime.now(ET).strftime('%Y-%m-%d')


def yesterday_et():
    return (datetime.now(ET) - timedelta(days=1)).strftime('%Y-%m-%d')


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except Exception:
        return None


def age_seconds(ts):
    """⚠️ READ-TIME age. This is the whole point of V6.3.

    obs_age_min in the row is computed when the poller WRITES and does not age.
    Only this function knows what time it actually is now.
    """
    dtv = parse_ts(ts)
    if dtv is None:
        return None
    return (datetime.now(pytz.UTC) - dtv.astimezone(pytz.UTC)).total_seconds()


def fmt_age(sec):
    if sec is None:
        return 'unknown age'
    if sec < 90:
        return f'{sec:.0f}s ago'
    if sec < 5400:
        return f'{sec/60:.0f}m ago'
    return f'{sec/3600:.1f}h ago'


def local_hour(city):
    """Local clock hour for a city, or None if unmapped."""
    tzname = CITY_TZ.get(city)
    if not tzname:
        return None
    try:
        return datetime.now(pytz.timezone(tzname)).hour
    except Exception:
        return None


# ── Auth ─────────────────────────────────────────────────────────────────────
def check_password():
    try:
        correct = st.secrets.get('app_password', None)
    except Exception:
        correct = None
    if not correct or st.session_state.get('_authed'):
        return
    st.markdown('### 🌡️ MPH Weather')
    st.caption('Private — enter access password')
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        pw = st.text_input('Password', type='password',
                           label_visibility='collapsed', placeholder='Password')
        if pw:
            if pw == correct:
                st.session_state['_authed'] = True
                st.rerun()
            else:
                st.error('Incorrect.')
    st.stop()


check_password()

# ⚠️ .hero is no longer unconditionally green. V6.2 hardcoded
# `border:2px solid #00ff88` so no data condition could ever change it.
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #0a0e1a; }
.sec { font-size:12px; font-weight:600; color:#94a3b8; text-transform:uppercase;
       letter-spacing:1.2px; padding:0 0 8px 0; border-bottom:1px solid #1e3a5f;
       margin:18px 0 14px 0; }
.stMetric { background:#0d1b2a !important; border:1px solid #1e3a5f !important;
            border-radius:8px !important; padding:12px !important; }
.stMetric label { color:#64748b !important; font-size:11px !important;
                  text-transform:uppercase !important; letter-spacing:.8px !important; }
.stMetric [data-testid="stMetricValue"] { color:#fff !important;
    font-family:'JetBrains Mono',monospace !important; font-size:22px !important; }
.hero { background:#0d1b2a; border-radius:10px; padding:14px 20px;
        margin-bottom:10px; }
.hero-ok    { border:2px solid #00ff88; }
.hero-warn  { border:2px solid #fbbf24; }
.hero-dead  { border:2px solid #ef4444; background:#2a0d12; }
.hero-l { color:#64748b; font-size:11px; text-transform:uppercase;
          letter-spacing:1px; }
.hero-v { font-size:34px; font-weight:700;
          font-family:'JetBrains Mono',monospace; line-height:1.1; }
.v-ok   { color:#00ff88; }
.v-warn { color:#fbbf24; }
.v-dead { color:#ef4444; font-size:28px; }
.sub { color:#94a3b8; font-size:12px; font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ── Supabase ─────────────────────────────────────────────────────────────────
try:
    SB_URL = st.secrets['supabase']['url']
    SB_KEY = st.secrets['supabase']['key']
except Exception:
    st.error('Supabase credentials missing from secrets.')
    st.stop()


def sb_headers():
    return {'apikey': SB_KEY, 'Authorization': 'Bearer ' + SB_KEY,
            'Content-Type': 'application/json'}


def sb_get(table, params, timeout=15):
    """Every read goes through here. Returns [] on any failure — a dashboard
    that errors out is worse than one showing an empty section."""
    try:
        r = requests.get(f'{SB_URL}/rest/v1/{table}',
                         headers=sb_headers(), params=params, timeout=timeout)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


# ⚠️ 10s, not 60s. A tab left open since morning served an 81.0F reading that
# was hours stale and it read as live. A short cache does NOT make data fresh —
# it only re-reads a possibly-frozen row more often. Freshness is age_seconds().
@st.cache_data(ttl=10)
def fetch_obs_live():
    """⚠️ TWO DATES, NOT ONE. The poller writes each row under the STATION's
    local date, so a single Eastern-date filter loses the Pacific cities
    between 9pm ET and midnight PT."""
    rows = sb_get('obs_live', {
        'local_date': f'in.({yesterday_et()},{today_et()})',
        'order': 'city.asc', 'limit': '100'})
    newest = {}
    for r in rows:
        c = r.get('city')
        if not c:
            continue
        prev = newest.get(c)
        if prev is None:
            newest[c] = r
            continue
        a, b = parse_ts(r.get('updated_at')), parse_ts(prev.get('updated_at'))
        if a and b and a > b:
            newest[c] = r
        elif (r.get('local_date') or '') > (prev.get('local_date') or ''):
            newest[c] = r
    return sorted(newest.values(), key=lambda x: x.get('city') or '')


@st.cache_data(ttl=120)
def fetch_today_consensus():
    return sb_get('settlements', {'date': 'eq.' + today_et(),
                                  'order': 'city.asc', 'limit': '50'})


@st.cache_data(ttl=300)
def fetch_settled(days=30):
    cutoff = (datetime.now(ET) - timedelta(days=days)).strftime('%Y-%m-%d')
    return sb_get('settlements', {'actual': 'not.is.null',
                                  'date': 'gte.' + cutoff,
                                  'order': 'date.desc', 'limit': '2000'})


@st.cache_data(ttl=60)
def fetch_favorites():
    return sb_get('favorites_bets', {'order': 'date.desc', 'limit': '2000'})


@st.cache_data(ttl=60)
def fetch_snapshots(days=10):
    """⚠️ SEPARATE TABLE, NEVER POOLED WITH BETS. Snapshots carry no stake, no
    fee and no band filter. They are observations, not trades."""
    cutoff = (datetime.now(ET) - timedelta(days=days)).strftime('%Y-%m-%d')
    return sb_get('favorites_snapshots', {'date': 'gte.' + cutoff,
                                          'order': 'date.desc', 'limit': '4000'})


def kalshi_fee_cents(price_cents):
    """Per-contract fee in cents, per Kalshi's published schedule:
        fee = round up(M x 0.07 x C x P x (1-P)),  M defaults to 1
    Confirmed 2026-09-08 against a real fill (30 contracts @ 65c -> $0.48)."""
    p = price_cents / 100.0
    return 0.07 * p * (1 - p) * 100


def break_even(price_cents):
    """Win rate needed to break even, holding to settlement (no exit fee)."""
    return round(price_cents + kalshi_fee_cents(price_cents), 1)


def quantization_band(f):
    """What a 5-minute feed reading ACTUALLY tells you.

    Most ASOS stations transmit whole degrees CELSIUS between the hourly :51
    METARs. 89.6F is not a measurement of 89.6 — it is the station saying
    "32C", so the true temperature was anywhere in 88.7 to 90.5F.

    Returns (lo_f, hi_f, celsius_int, is_on_grid).
    """
    try:
        c = (float(f) - 32.0) * 5.0 / 9.0
    except Exception:
        return f, f, None, False
    c_round = round(c)
    on_grid = abs(c - c_round) < 0.02
    if not on_grid:
        return f, f, None, False
    lo = (c_round - 0.5) * 9.0 / 5.0 + 32.0
    hi = (c_round + 0.5) * 9.0 / 5.0 + 32.0
    return round(lo, 1), round(hi, 1), c_round, True


@st.cache_data(ttl=900)
def fetch_calibration(days=60):
    """MEASURED (CLI actual − feed max), from this account's own history.

    ⚠️ THE YARDSTICK IS NWS CLI; KALSHI SETTLES ON THE WEATHER COMPANY.
    Confirmed 2026-09-14 by reading a live KXHIGH market's Rules tab: "the
    maximum temperature recorded at Miami (CLIMIA) ... according to The Weather
    Company. Outcome verified from The Weather Company." Kalshi's own help
    centre still said NWS Daily Climate Report, so it was out of date. Note the
    station identifier is CLIMIA, Kalshi's code, not KMIA.

    `settlements.actual` still comes from Iowa State CLI — i.e. NWS — so this
    calibration measures the feed against NWS, not against what actually pays.
    That was checked rather than assumed: across all 155 settled bets where a
    comparison was possible, Kalshi's own `result` field and an NWS-CLI scoring
    of the same bracket AGREED 155 times and disagreed ZERO times. The two
    sources are interchangeable for bracket outcomes, so this yardstick stands.

    ⚠️ Re-check if that ever stops being true. A source change that moved a
    bracket would silently corrupt every share on the board.

    ⚠️ READS THE `max_vs_settled` VIEW. An earlier version did this join in
    Python — obs_live keyed on (city, local_date), settlements on (city, date)
    — and silently matched only 8 of 109 rows. `settlements.date` is TEXT
    while `obs_live.local_date` is a DATE, so the string keys disagreed on
    most rows and the board calibrated on eight observations. That produced
    suspiciously round 50/50 and 62/38 splits, which is what gave it away.

    Postgres casts the two correctly. Do the join there, not here:

        create or replace view public.max_vs_settled as
        select o.city, o.local_date, o.day_max_f, s.actual,
               round((s.actual - o.day_max_f)::numeric, 1) as diff
        from obs_live o
        join settlements s
          on s.city = o.city and s.date::date = o.local_date
        where s.actual is not null and o.day_max_f is not null;

    ⚠️ THIS REPLACED A TEXTBOOK ASSUMPTION THAT WAS DEMONSTRABLY WRONG.
    V6.4's first cuts modelled the settle distribution as UNIFORM across the
    1.8F quantization band. Scored against 2026-09-13's settled column that
    produced six misses LOW, one high, three exact:

        Austin 100 -> settled 101      Denver 93 -> settled 94
        Dallas 100 -> settled 101      Miami  91 -> settled 92
        OKC    100 -> settled 101      N.O.   91 -> settled 92

    The measured distribution over 109 city-days explains it. It is NOT
    symmetric:

        left tail stops at -0.8   (pure quantization: a 37C feed max means the
                                   true value can sit 0.9F below the reading)
        right tail runs to +1.9   (the FLOOR effect: CLI is built from the
                                   station's own max and catches peaks that
                                   fall between 5-minute samples — nothing
                                   bounds this side)

        mean +0.15, median ~0, and 50 of 109 days came in ABOVE the feed max
        against 38 below.

    ⚠️ THE SAMPLE IS CAPPED BY obs_live, NOT settlements. As of 2026-09-13
    settlements holds 1,094 rows back to 2026-07-16 but obs_live only 140 rows
    back to 2026-09-07, so the join can never exceed ~140. It grows by 20/day.
    Do not read a small n here as a bug.

    Returns a list of diffs (floats). Empty -> callers fall back to the uniform
    band, which is known to skew LOW.
    """
    cutoff = (datetime.now(ET) - timedelta(days=days)).strftime('%Y-%m-%d')
    rows = sb_get('max_vs_settled', {'local_date': 'gte.' + cutoff,
                                     'select': 'diff', 'limit': '5000'})
    diffs = []
    for r in rows:
        d = r.get('diff')
        if d is None:
            continue
        try:
            diffs.append(float(d))
        except Exception:
            continue
    return diffs


def settle_distribution(feed_max, station=None, diffs=None):
    """Every integer degree this reading could settle at, with its share.

    Returns [(degree, share), ...] sorted by share descending, or [].

    ⚠️ V6.4 GOT THIS WRONG THREE TIMES. Each fix is worth keeping written down
    because each mistake looked reasonable.

    MISTAKE 1 — only the endpoints. Returned (lo, hi) and printed "lo or hi",
    silently dropping the middle when a band spans THREE integers:

        Las Vegas 102.2F = 39.0C, band 101.3-103.1
            101 : 11%      102 : 56%  <- omitted      103 : 33%

    MISTAKE 2 — OFF-GRID TREATED AS PRECISE. The identical inversion V6.3 was
    written to kill. The board printed "88 (100%) locked" for Washington DC on
    2026-09-13; it settled 89. 88.0F is NOT on the Celsius grid (31C = 87.8,
    32C = 89.6), and V6.3's finding is that real ASOS values land ON the grid,
    so off-grid means stale or misparsed — a WIDER band, not a narrower one.
    Only KBOS and KMSP genuinely transmit tenths.

    MISTAKE 3 — ASSUMING THE BAND IS UNIFORM. It is not, and the error has a
    direction: six of ten cities settled ABOVE the board's estimate on
    2026-09-13. See fetch_calibration() for the measured distribution.

    When `diffs` is supplied, the distribution is built by applying every
    historically observed (actual − feed max) to this reading and tallying
    where it rounds. That absorbs quantization AND the floor effect at once,
    measured rather than assumed.
    """
    if feed_max is None:
        return []
    try:
        fv = float(feed_max)
    except Exception:
        return []

    # ── Preferred path: the account's own measured error distribution ──
    if diffs:
        tally = {}
        for d in diffs:
            k = int(fv + d + 0.5)
            tally[k] = tally.get(k, 0) + 1
        tot = sum(tally.values())
        out = [(k, c / tot) for k, c in tally.items() if c / tot >= 0.01]
        rescale = sum(s for _, s in out)
        out = [(k, s / rescale) for k, s in out]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    # ── Fallback: uniform across the quantization band ──
    # ⚠️ Known to skew LOW. Used only before any settled history exists.
    lo, hi, _c, on_grid = quantization_band(fv)
    if not on_grid:
        if (station or '').upper() in NATIVE_TENTHS:
            return [(int(fv + 0.5), 1.0)]
        lo, hi = fv - 0.9, fv + 0.9
    width = hi - lo
    if width <= 0:
        return [(int(lo + 0.5), 1.0)]
    out = []
    k = int(lo + 0.5)
    k_hi = int(hi + 0.5 - 1e-9)
    while k <= k_hi:
        overlap = min(hi, k + 0.5) - max(lo, k - 0.5)
        if overlap > 1e-9:
            out.append((k, overlap / width))
        k += 1
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def settle_text(dist, compact=True):
    """Render a settle distribution. Most likely first, share in parentheses."""
    if not dist:
        return '—'
    if len(dist) == 1:
        return f'{dist[0][0]}'
    if compact:
        head = f'{dist[0][0]} ({dist[0][1]*100:.0f}%)'
        rest = ' · '.join(f'{d}={s*100:.0f}%' for d, s in dist[1:])
        return f'{head} · {rest}'
    return ' · '.join(f'{d}={s*100:.0f}%' for d, s in dist)


def find_duplicate_cities(rows):
    """Cities whose rows look like the SAME row written twice, not two cities
    that merely happen to read alike.

    ⚠️ THE FIRST VERSION KEYED ON TEMPERATURES AND CRIED WOLF CONSTANTLY.
    It flagged any two cities sharing (day_max_f, temp_f, n_obs_today,
    n_metars_today). That fires on coincidence all day, because the feed
    transmits whole degrees CELSIUS — there are only ~15 possible values across
    a September afternoon and twenty cities to spread over them. Observed
    2026-09-14 10:29: New Orleans and Oklahoma City both 84.20/84.20/121/4,
    flagged as corrupt. On the same screen Minneapolis and San Francisco were
    both 60.8/57.2, Houston and Miami both 86.0/86.0, Atlanta and Austin both
    84.2/84.2. All four pairs were real, distinct, correctly-polled cities.

    ⚠️ WHAT THE REAL BUG LOOKED LIKE. On 2026-09-09 San Antonio and Los Angeles
    showed identical rows because the poller had FROZEN — one write, replayed.
    The tell was never the temperature. It was that both rows carried the same
    stale `updated_at` while the clock moved on, and both were hours old.

    So the test is: same values AND the same write timestamp AND that timestamp
    is already stale. A live run writes every city in the same pass, so a shared
    fresh timestamp is normal and means nothing.
    """
    sig = {}
    for r in rows:
        key = (r.get('day_max_f'), r.get('temp_f'),
               r.get('n_obs_today'), r.get('n_metars_today'),
               str(r.get('updated_at')))
        if key[:4] == (None, None, None, None):
            continue
        sig.setdefault(key, []).append(r)
    dupes = set()
    for key, group in sig.items():
        if len(group) < 2:
            continue
        # Only suspicious once the shared row has gone stale. Identical values
        # written in the same LIVE pass are a coincidence, not corruption.
        age = age_seconds(group[0].get('updated_at'))
        if age is None or age > STALE_HARD_SEC:
            dupes.update(r.get('city') for r in group if r.get('city'))
    return dupes


def score_bracket(lo, hi, label, actual):
    """Did this bracket win? True / False / None (unscoreable).

    ⚠️ TAIL BRACKETS ARE NOT RANGES, AND THIS IS WHERE HAND-SCORING FAILS.
    Kalshi's ladder has three shapes and only one of them is a range:

        "97 to 98"     -> lo=97,   hi=98    win if 97 <= actual <= 98
        "63 or below"  -> lo=None, hi=63    win if actual <= 63
        "103 or above" -> lo=103,  hi=None  win if actual >= 103

    Read by eye off a photographed grid, "63↓" looks like a 63-64 range and
    gets scored backwards. That turned a Seattle WIN into a loss on 2026-09-13
    (settled 60) and did the same to New Orleans on 2026-09-11.

    Falls back to parsing the label when bracket_lo/bracket_hi are null — those
    columns only exist on rows written from 2026-09-08 onward, so most of the
    early record has neither.
    """
    if actual is None:
        return None
    try:
        a = float(actual)
    except Exception:
        return None

    if lo is None and hi is None and label:
        s = str(label).replace('\u00b0', '').strip().lower()
        nums = [int(x) for x in re.findall(r'\d+', s)]
        if not nums:
            return None
        if 'below' in s or 'under' in s:
            lo, hi = None, nums[0]
        elif 'above' in s or 'over' in s:
            lo, hi = nums[0], None
        elif len(nums) >= 2:
            lo, hi = nums[0], nums[1]
        else:
            return None

    if lo is None and hi is None:
        return None
    if lo is None:
        return a <= float(hi)
    if hi is None:
        return a >= float(lo)
    return float(lo) <= a <= float(hi)


# ── Header ───────────────────────────────────────────────────────────────────
now_et = datetime.now(ET)
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744,#0d1b2a);
            border:1px solid #1e3a5f;border-radius:12px;padding:16px 24px;">
  <div style="font-size:22px;font-weight:700;color:#fff;">🌡️ MPH Weather
    <span style="font-size:11px;color:#00ff88;border:1px solid #00ff8840;
                 background:#00ff8820;padding:2px 9px;border-radius:20px;
                 margin-left:8px;vertical-align:middle;
                 font-family:'JetBrains Mono',monospace;">V6.4</span></div>
  <div style="font-size:12px;color:#64748b;font-family:'JetBrains Mono',monospace;">
    {now_et:%Y-%m-%d %I:%M:%S %p ET} · settles on The Weather Company</div>
</div>
""", unsafe_allow_html=True)
with h2:
    st.write('')
    if st.button('🔄 Refresh', use_container_width=True):
        st.cache_data.clear()
        st.rerun()


obs_rows = fetch_obs_live()
dupe_cities = find_duplicate_cities(obs_rows)
favs_all = fetch_favorites()

if dupe_cities:
    st.error(
        f'⚠️ DUPLICATE ROWS IN obs_live — {len(dupe_cities)} cities share an '
        f'identical STALE row: {", ".join(sorted(dupe_cities))}. Same values '
        f'AND the same write timestamp, already past '
        f'{STALE_HARD_SEC // 60} minutes — the signature of a frozen poller '
        f'replaying one write, not two cities that happen to read alike. '
        f'Check GitHub Actions → obs_live.yml.')


# ── 0. DECISION BOARD ────────────────────────────────────────────────────────
# ⚠️ V6.4. Exists because the old flow was twenty clicks to answer one question.
st.markdown('<div class="sec">🎯 Decision Board — who is still undecided</div>',
            unsafe_allow_html=True)

if not obs_rows:
    st.caption('No obs_live rows. The poller runs every 5 min, 9am–9pm ET via '
               'cron-job.org → obs_live.yml.')
else:
    live_hours = poller_should_be_running()
    # ⚠️ Measured from this account's own obs_live vs settlements history, via
    # the max_vs_settled view. Empty -> settle_distribution falls back to the
    # uniform band, which is known to skew LOW.
    cal_diffs = fetch_calibration(60)
    if len(cal_diffs) < 30:
        st.warning(
            f'⚠️ **Calibration is running on only {len(cal_diffs)} city-days.** '
            f'Shares below are coarse and will look suspiciously round '
            f'(50/50, 62/38) because a small sample can only produce a few '
            f'distinct values. If this reads under ~20 when it should be over '
            f'100, check that the `max_vs_settled` view exists — an earlier '
            f'build did this join in Python and matched 8 of 109 rows because '
            f'`settlements.date` is TEXT and `obs_live.local_date` is a DATE.')
    board = []
    for r in obs_rows:
        city = r.get('city')
        ra = age_seconds(r.get('updated_at'))
        is_dupe_row = city in dupe_cities
        too_old = (ra is None or ra > STALE_HARD_SEC)

        # ⚠️ TWO DIFFERENT KINDS OF STALE, TREATED DIFFERENTLY.
        #   during 9am-9pm : a stale row is a FAULT. Blank it. This is the
        #                    2026-09-09 failure — an 8:24am reading shown in
        #                    green at 5:19pm, which drove a real decision.
        #   outside those  : a stale row is the SCHEDULE. Show it, labelled
        #                    "closed", because the day it describes is over
        #                    and a settled number cannot mislead anyone.
        # A duplicate row is untrustworthy at any hour — that is corruption,
        # not timing.
        dead = is_dupe_row or (too_old and live_hours)
        closed = (too_old and not live_hours and not is_dupe_row)
        fmax = r.get('day_max_f')
        trend = r.get('trend_30min')
        lh = local_hour(city)

        dist = settle_distribution(fmax, r.get('station'), cal_diffs)
        straddles = len(dist) > 1
        # ⚠️ "Close" means the top outcome does not dominate. A band whose
        # most likely integer holds 56% is a coin flip with a lean; one that
        # holds 90% is effectively decided even though it technically straddles.
        top_share = dist[0][1] if dist else 0.0

        # ⚠️ OFF-GRID AT A STATION THAT SHOULD BE ON IT IS A DATA SMELL.
        # Flagged so a suspect reading is visible rather than silently
        # widening the band. DC read 88.0 on 2026-09-13 and settled 89.
        # Off-grid is still worth surfacing as a data smell even when the
        # empirical model is in use — it flags a reading that should not exist.
        _lo, _hi, _cc, _og = (quantization_band(fmax) if fmax is not None
                              else (0, 0, None, False))
        off_grid_suspect = (fmax is not None and not _og
                            and (r.get('station') or '').upper() not in NATIVE_TENTHS)

        # ⚠️ PEAK STATUS IS A HEURISTIC, NOT A MEASUREMENT. It reads the local
        # clock and the 30-minute trend. A flat trend does NOT mean flat
        # weather — the feed steps 1.8F at a time and sits still in between, so
        # +0.0 is the normal reading for most of any given hour.
        if dead:
            peak = '—'
        elif closed:
            peak = '🔒 closed'
        elif lh is None:
            peak = 'unknown tz'
        elif lh < 12:
            peak = '🔺 early'
        elif trend is not None and trend > 0:
            peak = '🔺 climbing'
        elif lh >= 17:
            peak = '✅ peak likely in'
        elif lh >= 15:
            peak = '🟡 near peak'
        else:
            peak = '🔺 mid-day'

        if dead or not dist:
            likely, alts, closeness = '—', '—', ''
        else:
            likely = f'{dist[0][0]}  ({dist[0][1]*100:.0f}%)'
            # ⚠️ NEVER SAY "LOCKED". The feed max is the max of what was
            # SAMPLED; CLI is built from the station's own max and can come in
            # higher. DC: feed 88.0, CLI 89.
            if len(dist) > 1:
                alts = ' · '.join(f'{d}={s*100:.0f}%' for d, s in dist[1:])
            elif off_grid_suspect:
                alts = '⚠️ off-grid'
            else:
                alts = 'or higher'
            # Closeness flag: how much room the runner-up has. Suppressed once
            # the day is closed — nothing is undecided about a finished day.
            if closed:
                closeness = ''
            elif off_grid_suspect:
                closeness = '⚠️ suspect'
            elif len(dist) == 1:
                closeness = ''
            elif dist[0][1] < 0.45:
                closeness = '🔴 wide open'
            elif dist[0][1] < 0.62:
                closeness = '🟠 close'
            else:
                closeness = '🟡 leaning'

        board.append({
            # Sort: live rows first, then by how UNDECIDED they are — a band
            # whose top outcome holds only 40% is more interesting than one
            # holding 85%, even though both technically straddle. Closed rows
            # sink below live ones, dead rows below those.
            '_sort': (2 if dead else (1 if closed else 0),
                      top_share if not (dead or closed) else 9,
                      -(fmax or 0)),
            'City': city,
            'Local': f'{lh:02d}:00' if lh is not None else '—',
            'Day Max': '⛔' if dead else (f'{fmax:.1f}' if fmax is not None else '—'),
            'Most likely': likely,
            'Also possible': alts,
            'How close': closeness,
            'Trend 30m': ('—' if (dead or closed or trend is None)
                          else f'{trend:+.1f}'),
            'Peak': peak,
            'Age': fmt_age(ra),
            'NWS': nws_url(city),
            'TWC': twc_url(city),
        })
    board.sort(key=lambda x: x['_sort'])
    for b in board:
        b.pop('_sort', None)
    st.dataframe(
        pd.DataFrame(board), use_container_width=True, hide_index=True,
        column_config={
            # ⚠️ Points at the SETTLEMENT STATION's coordinates, not the city.
            # A link to "Houston" gives Bush; Kalshi settles Hobby.
            # ⚠️ Both point at the SETTLEMENT STATION. NWS is keyed by ICAO
            # so it is exact everywhere; TWC only appears for stations whose
            # URL has been verified by hand.
            'NWS': st.column_config.LinkColumn(
                'NWS', display_text='open', width='small',
                help='NWS point forecast for this settlement station, by ICAO '
                     'code. Exact station, no guessing.'),
            'TWC': st.column_config.LinkColumn(
                'TWC', display_text='open', width='small',
                help='The Weather Company forecast. Only shown for stations '
                     'whose URL has been verified — blank means not yet '
                     'added, NOT that it does not exist.'),
        })

    n_undec = sum(1 for b in board if b['How close'])
    n_closed = sum(1 for b in board if b['Peak'] == '🔒 closed')
    n_dead = sum(1 for b in board if b['Peak'] == '—')

    # ⚠️ SAY WHY THE BOARD IS EMPTY. Twenty blank rows with no explanation look
    # like a fault; the usual cause is simply that it is 10pm. The two cases
    # need opposite responses, so they get different messages.
    if not live_hours:
        st.info(
            f'🔒 **Poller window closed.** obs_live runs '
            f'{POLLER_START_HOUR}:00–{POLLER_END_HOUR}:00 ET via cron-job.org '
            f'→ obs_live.yml, so nothing has been written since about '
            f'{POLLER_END_HOUR}:00 and nothing will until '
            f'{POLLER_START_HOUR}:00 tomorrow. The figures above are each '
            f'city\'s FINAL reading for the day, not a live one — shown rather '
            f'than blanked because a settled number cannot mislead you into a '
            f'trade.')
    elif n_dead:
        st.error(
            f'⛔ **{n_dead} row(s) are stale DURING the poller window** — that '
            f'is a fault, not the schedule. Check GitHub Actions → obs_live.yml '
            f'for failed runs, and Supabase for PGRST204 column errors, which '
            f'let the job exit green while writing nothing.')

    st.caption(
        f'**{n_undec} of {len(board)} cities are still undecided.**'
        + (f' {n_closed} closed for the day.' if n_closed else '')
        + ' Sorted by how close, not alphabetically — the top row is the one '
          'the station can tell you least about.\n\n'
        '**Most likely** is the whole degree the reading is most likely to '
        'settle at. Shares come from **this account\'s own measured error** — '
        f'every (CLI actual − feed max) over the last 60 days ({len(cal_diffs)} '
        'city-days) applied to today\'s reading, not from a textbook '
        'assumption.\n\n'
        '⚠️ **The error is not symmetric, and that matters.** Measured over 108 '
        'city-days the left tail stops at −0.8°F while the right runs to '
        '+1.9°F, and 50 days came in ABOVE the feed max against 38 below. Two '
        'effects stacked: quantization can put the true value up to 0.9°F '
        'BELOW a reading, while CLI is built from the station\'s own max and '
        'catches peaks that fall between 5-minute samples — nothing bounds '
        'that side.\n\n'
        '⚠️ An earlier version assumed a uniform band and was wrong in a '
        'direction: on 2026-09-13 it missed LOW on six of ten cities (Austin, '
        'Dallas, OKC, Denver, Miami, New Orleans all settled one degree above '
        'its estimate). The empirical model absorbs that skew instead of '
        'modelling it.\n\n'
        '⚠️ **off-grid / suspect** means the reading does not sit on the '
        'station\'s Celsius transmission grid, at a station that is not KBOS '
        'or KMSP. Real ASOS values land ON the grid, so off-grid is a sign of '
        'stale, mixed or misparsed data. Treating off-grid as precise is what '
        'printed "88 (100%) locked" for DC the day it settled 89.\n\n'
        '⚠️ Shares assume the true value is spread evenly across the band. It '
        'is not. If the hourly METARs either side sit below the band, the peak '
        'most likely clipped its BOTTOM and the low end is underweighted here. '
        'San Antonio 2026-09-09: uniform said 44%, the market said 26%, and '
        'the market was closer.\n\n'
        '⚠️ Peak status reads the local clock and the 30-minute trend. A +0.0 '
        'trend is normal between 1.8°F steps and does NOT mean the temperature '
        'is flat.')


# ── 1. LIVE OBS ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📡 Live Obs — Settlement Station</div>',
            unsafe_allow_html=True)

sel = None
if obs_rows:
    cities = sorted(r['city'] for r in obs_rows if r.get('city'))
    default = cities.index('New York') if 'New York' in cities else 0
    sel = st.selectbox('City', cities, index=default, key='_obs_city')
    row = next((r for r in obs_rows if r.get('city') == sel), None)

    if row:
        feed_now = row.get('temp_f')
        feed_max = row.get('day_max_f')
        nxt = row.get('next_step_f')
        met_now = row.get('metar_temp_f')
        prec_max = row.get('precise_max_f')
        n_obs = row.get('n_obs_today')
        n_met = row.get('n_metars_today')
        station = (row.get('station') or '').upper()

        # ⚠️ AGE IS COMPUTED HERE, FROM updated_at, AGAINST NOW.
        row_age = age_seconds(row.get('updated_at'))
        poller_says_stale = bool(row.get('is_stale'))
        stale_reason = row.get('stale_reason')
        is_dupe = sel in dupe_cities

        hard_stale = (row_age is None) or (row_age > STALE_HARD_SEC)
        soft_stale = (row_age is not None) and (row_age > STALE_SOFT_SEC)
        untrustworthy = hard_stale or is_dupe

        # ── THE HEADLINE ────────────────────────────────────────────────
        # If the row is untrustworthy the NUMBER IS NOT SHOWN. V6.2 rendered
        # 81.0 in green at 5:19pm from a row written at 8:24am, and it drove a
        # real decision.
        if untrustworthy:
            why = []
            if hard_stale:
                why.append(f'last written {fmt_age(row_age)}')
            if is_dupe:
                why.append('duplicate of another city')
            if stale_reason:
                why.append(stale_reason)
            st.markdown(
                f'<div class="hero hero-dead">'
                f'<div class="hero-l">Day Max — 5-min feed</div>'
                f'<div class="hero-v v-dead">STALE — DO NOT USE</div>'
                f'<div class="sub">{" · ".join(why)}</div></div>',
                unsafe_allow_html=True)
            st.error('This row is not live. Check the obs_live poller before '
                     'reading anything on this page. Pull the raw METAR '
                     'directly if you need a number right now.')
        else:
            _band_note = ''
            if feed_max is not None:
                _lo, _hi, _c, _og = quantization_band(feed_max)
                if _og:
                    _band_note = f' · true peak {_lo}–{_hi}°F'
            klass = 'hero-warn' if (soft_stale or poller_says_stale) else 'hero-ok'
            vklass = 'v-warn' if (soft_stale or poller_says_stale) else 'v-ok'
            st.markdown(
                f'<div class="hero {klass}">'
                f'<div class="hero-l">Day Max — 5-min feed</div>'
                f'<div class="hero-v {vklass}">{feed_max:.1f}°F</div>'
                f'<div class="sub">{n_obs or 0} obs today · '
                f'written {fmt_age(row_age)}{_band_note}</div></div>',
                unsafe_allow_html=True)
            if soft_stale:
                st.warning(f'Row is {fmt_age(row_age)} — the poller runs every '
                           f'5 minutes, so this is behind. Treat as indicative.')
            if poller_says_stale and stale_reason:
                st.warning(f'Poller flagged this row: {stale_reason}')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Now', '—' if untrustworthy or feed_now is None else f'{feed_now:.1f}')
        c2.metric('Next possible', '—' if untrustworthy or nxt is None else f'{nxt:.1f}')
        c3.metric('Trend 30m',
                  '—' if untrustworthy or row.get('trend_30min') is None
                  else f"{row.get('trend_30min'):+.1f}")
        _lh = local_hour(sel)
        c4.metric('Local time', '—' if _lh is None else f'{_lh:02d}:00')

        # ── SECONDARY: the exact hourly reading ─────────────────────────
        met_age = age_seconds(row.get('metar_time_utc'))
        m1, m2 = st.columns([1, 3])
        with m1:
            st.metric('METAR (exact)',
                      '—' if untrustworthy or met_now is None else f'{met_now:.1f}')
        with m2:
            st.write('')
            if untrustworthy:
                st.markdown('<div class="sub" style="padding-top:14px;">'
                            'withheld — row not live</div>',
                            unsafe_allow_html=True)
            else:
                gap_note = ''
                if feed_max is not None and prec_max is not None:
                    g = round(feed_max - prec_max, 1)
                    gap_note = f' · METAR max {prec_max:.1f} ({g:+.1f} vs feed)'
                st.markdown(
                    f'<div class="sub" style="padding-top:14px;">'
                    f'{fmt_age(met_age)} · {n_met or 0} today{gap_note}</div>',
                    unsafe_allow_html=True)

        if (not untrustworthy and prec_max is not None and feed_max is not None
                and (feed_max - prec_max) >= 1.5):
            st.caption(f'⚠️ The hourly METAR record tops out {feed_max - prec_max:.1f}F '
                       f'below the 5-minute feed. With {n_met or 0} METARs against '
                       f'{n_obs or 0} feed obs, the peak fell between :51 reports. '
                       f'Trust the feed max.')

        # ⚠️ V6.4: STEPS, NOT DEGREES. "2.9F to go" is not actionable, because
        # the station cannot transmit 2.9F of change — it moves in 1.8F jumps.
        if not untrustworthy and nxt is not None and feed_max is not None:
            cur_d = settle_distribution(feed_max, station, fetch_calibration(60))
            nxt_d = settle_distribution(nxt, station, fetch_calibration(60))
            st.caption(
                f'Feed steps 1.8°F — nothing exists between **{feed_max:.1f}** '
                f'and **{nxt:.1f}**.\n\n'
                f'· As it stands: {settle_text(cur_d)}\n\n'
                f'· One more step: {settle_text(nxt_d)}')

        # ── BRACKET CHECK ───────────────────────────────────────────────
        st.markdown('<div class="sub">Bracket check — enter the ceiling you '
                    'care about</div>', unsafe_allow_html=True)

        if untrustworthy:
            st.info('Bracket check disabled — the underlying row is not live. '
                    'A verdict computed from a stale reading is worse than no '
                    'verdict.')
        else:
            b1, b2 = st.columns([1, 4])
            with b1:
                ceiling = st.number_input('Ceiling °F', min_value=0, max_value=130,
                                          value=int(feed_max) if feed_max else 80,
                                          step=1, label_visibility='collapsed')
            with b2:
                st.write('')
                if feed_max is not None:
                    lo, hi, c_round, on_grid = quantization_band(feed_max)
                    native_tenths = station in NATIVE_TENTHS

                    if not on_grid and not native_tenths:
                        # ⚠️ THE INVERSION THAT MADE A BAD ROW LOOK CERTAIN.
                        # Real ASOS values land ON the grid. Off-grid is a
                        # symptom of bad data, not evidence of precision.
                        st.warning(
                            f'⚠️ {feed_max:.1f}°F does not sit on this '
                            f'station\'s transmission grid, and '
                            f'{station or "this station"} is not one of the '
                            f'native-tenths sites. That is a sign of a bad or '
                            f'stale reading, not a precise one. No verdict.')
                    else:
                        settle_lo = int(lo + 0.5)
                        settle_hi = int(hi + 0.5 - 1e-9)
                        if settle_lo > ceiling:
                            st.error(f'BROKEN — the peak settles {settle_lo} at '
                                     f'best, above your {ceiling} ceiling.')
                        elif settle_hi <= ceiling:
                            st.success(f'SAFE so far — the peak settles '
                                       f'{settle_hi} at worst, at or below '
                                       f'{ceiling}.')
                        else:
                            share = ((ceiling + 0.5 - lo) / (hi - lo)) if hi > lo else 0.5
                            share = max(0.0, min(1.0, share))
                            st.warning(
                                f'UNRESOLVED — the true peak was '
                                f'**{lo:.1f}–{hi:.1f}°F**. On a flat assumption '
                                f'about **{share*100:.0f}%** of that range '
                                f'settles {ceiling} or below.')
                            st.caption(
                                '⚠️ That share assumes the true value is spread '
                                'evenly across the band. It is not. If the '
                                'hourly METARs either side sit below the band, '
                                'the peak most likely clipped its BOTTOM and '
                                'the real odds are better than this figure. '
                                'Treat it as an upper bound on the bad outcome. '
                                '(San Antonio 2026-09-09: this said 44%, the '
                                'market said 26%, and the market was closer.)')

                    if on_grid:
                        st.caption(f'{feed_max:.1f}°F is a quantized '
                                   f'transmission, not a measurement — the real '
                                   f'value is somewhere in a 1.8°F window.')
                    elif native_tenths:
                        st.caption(f'{station} transmits Fahrenheit tenths — '
                                   f'this max is exact.')

    with st.expander('Forecast links — settlement stations', expanded=False):
        # ⚠️ FORECAST, NOT SETTLEMENT. The Weather Company settles these markets
        # on the RECORDED max for the station (confirmed 2026-09-14 from a live
        # KXHIGH Rules tab). What these pages show is the FORECAST — same
        # company, different product. And forecasting is where this project has
        # already found no edge: 205 losing paper bets, with naked market
        # consensus beating the model 74.3% to 55.2% on 626 city-days.
        lrows = []
        for c in sorted(CITY_STATION):
            lrows.append({
                'City': c,
                'Station': CITY_STATION[c],
                'NWS': nws_url(c),
                'TWC today': twc_url(c, 'today'),
                'TWC hourly': twc_url(c, 'hourbyhour'),
            })
        st.dataframe(
            pd.DataFrame(lrows), use_container_width=True, hide_index=True,
            column_config={
                'NWS': st.column_config.LinkColumn('NWS', display_text='open'),
                'TWC today': st.column_config.LinkColumn('TWC today', display_text='open'),
                'TWC hourly': st.column_config.LinkColumn('TWC hourly', display_text='open'),
            })
        n_twc = len(TWC_VERIFIED)
        st.caption(
            f'All {len(CITY_STATION)} stations, both sources. **NWS** is keyed '
            'by ICAO code so it is exact by construction; **TWC** links were '
            f'each opened and confirmed by hand ({n_twc}/{len(CITY_STATION)}).'
            '\n\n'
            '⚠️ **These are FORECASTS, not the settlement number.** The Weather '
            'Company settles these markets on the RECORDED max for the station '
            '(confirmed 2026-09-14 from a live KXHIGH Rules tab: "according to '
            'The Weather Company"). The forecast on these pages is a different '
            'product from the number that pays. TWC\'s forecast has never been '
            'scored against NWS or GFS here — it is not in the consensus '
            'table.\n\n'
            '⚠️ **Never generate these URLs.** Three would be wrong under any '
            'naming rule a machine would pick: Houston is '
            '`william-p-hobby-airport` (not Bush), Chicago is '
            '`chicago-midway-international-airport` (not O\'Hare), and New '
            'York is `/poi/central-park` — not an airport at all. The city '
            'segment often is not the market\'s city either: DC sits under '
            'Arlington VA, Boston under East Boston, Dallas under Grapevine, '
            'New Orleans under Kenner, Seattle under SeaTac.\n\n'
            '⚠️ An earlier build generated these from station lat/lon and was '
            'wrong — weather.com snaps a coordinate to its nearest named '
            'place, which is the city, not the airport. Measured 2026-09-14: '
            'the app pinned to KMIA read 82°F / high 90°, the city page read '
            '84°F / high 88°. Two degrees on both, which on a 2°F bracket is '
            'the whole question.')

    with st.expander('All cities — raw obs', expanded=False):
        tbl = []
        for r in sorted(obs_rows,
                        key=lambda x: (x.get('day_max_f') is None,
                                       -(x.get('day_max_f') or 0))):
            ra = age_seconds(r.get('updated_at'))
            ma = age_seconds(r.get('metar_time_utc'))
            fx, px = r.get('day_max_f'), r.get('precise_max_f')
            dead = (ra is None or ra > STALE_HARD_SEC or r.get('city') in dupe_cities)
            tbl.append({
                'City': r.get('city', '—'),
                'Stn': r.get('station', '—'),
                'DAY MAX': '⛔ stale' if dead else (f"{fx:.1f}" if fx is not None else '—'),
                'Now': '—' if dead else (f"{r['temp_f']:.1f}" if r.get('temp_f') is not None else '—'),
                'Row age': fmt_age(ra) + (' ⛔' if dead else (' ⚠️' if ra and ra > STALE_SOFT_SEC else '')),
                'METAR age': fmt_age(ma),
                'METAR max': '—' if dead else (f"{px:.1f}" if px is not None else '—'),
                'gap': '—' if dead else (f"{fx - px:+.1f}" if (fx is not None and px is not None) else '—'),
                'obs/met': f"{r.get('n_obs_today','—')}/{r.get('n_metars_today','—')}",
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        st.caption('**Row age** is measured from updated_at against now, every '
                   'render. It is the only number on this page that cannot go '
                   'stale without saying so. ⛔ means the row has not been '
                   'rewritten in 15 minutes, or it duplicates another city. '
                   '⚠️ Preliminary, pre-QC. Kalshi settles on official CLI.')


# ── 2. MARKET TIMELINE ───────────────────────────────────────────────────────
# ⚠️ V6.4. Turns favorites_snapshots from a table you query by hand into
# something you glance at.
st.markdown('<div class="sec">🕐 Market Timeline — what the favorite did, '
            'hour by hour</div>', unsafe_allow_html=True)

snaps = fetch_snapshots(10)

if not snaps:
    st.caption('No snapshot rows yet. favorites_snapshots fills at 11:00, 12:00, '
               '13:00, 14:00, 15:00 and 17:00 ET via cron-job.org → '
               'favorites.yml. Collection began 2026-09-11.')
else:
    snap_dates = sorted({s.get('date') for s in snaps if s.get('date')},
                        reverse=True)
    tl1, tl2 = st.columns([1, 2])
    with tl1:
        tl_date = st.selectbox('Date', snap_dates, index=0, key='_tl_date')
    day_snaps = [s for s in snaps if s.get('date') == tl_date]
    day_bets = [b for b in favs_all if b.get('date') == tl_date]
    tl_cities = sorted({s.get('city') for s in day_snaps if s.get('city')})
    with tl2:
        _idx = tl_cities.index(sel) if (sel and sel in tl_cities) else 0
        tl_city = st.selectbox('City', tl_cities, index=_idx, key='_tl_city')

    # Merge both tables into one clock-ordered timeline.
    merged = []
    for s in day_snaps:
        if s.get('city') != tl_city:
            continue
        merged.append({
            'slot': s.get('snap_label'),
            'rank': SLOT_RANK.get(s.get('snap_label'), 9999),
            'bracket': s.get('bracket'),
            'ask': s.get('yes_ask_cents'),
            'sigma': s.get('sigma_p'),
            'src': 'snap',
            'in_band': bool(s.get('in_band')),
        })
    for b in day_bets:
        if b.get('city') != tl_city:
            continue
        merged.append({
            'slot': b.get('window_label'),
            'rank': SLOT_RANK.get(b.get('window_label'), 9999),
            'bracket': b.get('bracket'),
            'ask': b.get('yes_ask_cents'),
            'sigma': b.get('sigma_p'),
            'src': 'BET',
            'in_band': True,
        })
    merged.sort(key=lambda x: x['rank'])

    if not merged:
        st.caption('No rows for that city and date.')
    else:
        tbl = []
        prev_bracket = None
        n_changes = 0
        for m in merged:
            changed = (prev_bracket is not None and m['bracket'] != prev_bracket)
            if changed:
                n_changes += 1
            sig = m.get('sigma')
            dead_mkt = False
            try:
                dead_mkt = (sig is not None and float(sig) < LIVE_SIGMA_MIN)
            except Exception:
                dead_mkt = False
            tbl.append({
                'Slot': m['slot'],
                'Bracket': m['bracket'] or '—',
                'Ask': ('⛔ dead' if dead_mkt else
                        (f"{m['ask']}c" if m['ask'] is not None else '—')),
                'Σp': f"{float(sig):.2f}" if sig is not None else '—',
                'In band': '✅' if m.get('in_band') else '',
                'Source': m['src'],
                'Changed': '🔄 CHANGED' if changed else '',
            })
            prev_bracket = m['bracket']
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        if n_changes:
            st.caption(
                f'**{n_changes} bracket change(s) on this city-day.** The 🔄 row '
                'is where the market revised. What matters is the ASK on that '
                'row and the next: a revision that arrives already priced at '
                '84c is not tradeable, one that sits at 56c for an hour is. '
                '(2026-09-12: Miami flipped at 14:00 already at 84c; Atlanta '
                'flipped at 14:00 at 56c; San Antonio flipped by 13:00 and held '
                '63c for three hours.) ⛔ dead means the ladder summed below '
                f'{LIVE_SIGMA_MIN:.2f} — a resolved market, not a cheap one.')
        else:
            st.caption('No bracket change on this city-day — the market picked '
                       'one bracket and kept it.')

    with st.expander('Bracket changes — all cities, this date', expanded=False):
        rows = []
        for c in tl_cities:
            seq = sorted([s for s in day_snaps if s.get('city') == c],
                         key=lambda x: SLOT_RANK.get(x.get('snap_label'), 9999))
            brs = [x.get('bracket') for x in seq if x.get('bracket')]
            uniq = len(set(brs))
            first_change = ''
            for i in range(1, len(seq)):
                if seq[i].get('bracket') != seq[i - 1].get('bracket'):
                    first_change = (f"{seq[i].get('snap_label')} "
                                    f"@ {seq[i].get('yes_ask_cents')}c")
                    break
            rows.append({
                'City': c,
                'Obs': len(seq),
                'Distinct brackets': uniq,
                'Changed': '🔄' if uniq > 1 else '',
                'First change': first_change or '—',
            })
        rows.sort(key=lambda x: (-x['Distinct brackets'], x['City']))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            '⚠️ Read this as description, not signal. Bracket-change as a BET '
            'filter was tested and REVERSED under price control. Pooled across '
            '170 settled bets, city-days where the bracket never moved won '
            '70.8% (n=154) against 43.8% where it did (n=16) — but you only '
            'know a day was stable after it ends, and n=16 cannot support a '
            '27-point claim.')


# ── 3. DAILY SCORECARD ───────────────────────────────────────────────────────
# ⚠️ V6.4. Replaces counting wins by hand off a photograph of the paper grid.
st.markdown('<div class="sec">✅ Daily Scorecard — in-band results by pass</div>',
            unsafe_allow_html=True)

settled_all = fetch_settled(30)
actual_by = {}
for s in settled_all:
    if s.get('actual') is not None and s.get('city') and s.get('date'):
        actual_by[(str(s['date'])[:10], s['city'])] = s['actual']

bet_dates = sorted({b.get('date') for b in favs_all if b.get('date')},
                   reverse=True)
if not bet_dates:
    st.caption('No bets logged yet.')
else:
    sc_date = st.selectbox('Date', bet_dates, index=0, key='_sc_date')
    day = [b for b in favs_all if b.get('date') == sc_date]

    rows = []
    unscored = 0
    for w in ('MORNING', 'MIDDAY', 'AFTERNOON'):
        g = [b for b in day if b.get('window_label') == w]
        if not g:
            continue
        n = wins = 0
        net = 0.0
        asks = []
        for b in g:
            # Kalshi's own result field takes precedence — it is what the
            # contract actually settled at. Fall back to scoring the bracket
            # against the settled temperature when the bet is still Pending.
            res = b.get('result')
            if res in ('Won', 'Lost'):
                ok = (res == 'Won')
            else:
                ok = score_bracket(b.get('bracket_lo'), b.get('bracket_hi'),
                                   b.get('bracket'),
                                   actual_by.get((sc_date, b.get('city'))))
            if ok is None:
                unscored += 1
                continue
            n += 1
            wins += 1 if ok else 0
            if b.get('yes_ask_cents'):
                asks.append(float(b['yes_ask_cents']))
            net += float(b.get('net_profit') or 0)
        if not n:
            continue
        avg_ask = sum(asks) / len(asks) if asks else 0
        wp = 100.0 * wins / n
        be = break_even(avg_ask) if avg_ask else 0
        rows.append({
            'Pass': w, 'n': n, 'Wins': wins,
            'Win %': f'{wp:.1f}',
            'Avg Ask': f'{avg_ask:.1f}c',
            'Break-even': f'{be:.1f}%',
            'Margin': f'{wp - be:+.1f}',
            'Net': f'${net:+.2f}',
        })

    if rows:
        tn = sum(r['n'] for r in rows)
        tw = sum(r['Wins'] for r in rows)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric('Day total', f'{tw}/{tn}  ({100.0*tw/tn:.1f}%)')
    else:
        st.caption('Nothing scoreable for that date yet.')
    if unscored:
        st.caption(f'⚠️ {unscored} bet(s) not scoreable — no settled temperature '
                   f'and no Kalshi result yet.')
    st.caption(
        '⚠️ Scored from bracket bounds against the settled temperature, with '
        'Kalshi\'s own `result` field taking precedence. Tail brackets '
        '("63 or below", "103 or above") are handled explicitly — read by eye '
        'off the paper grid they look like ranges and get scored BACKWARDS, '
        'which turned a Seattle win into a loss on 2026-09-13 and a New '
        'Orleans win into a loss on 2026-09-11.')


# ── 4. RESULTS ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Results</div>', unsafe_allow_html=True)

settled_fav = [b for b in favs_all if b.get('result') in ('Won', 'Lost')]

tab_kill, tab_win, tab_day, tab_acc = st.tabs(
    ['Kill line', 'By window', 'By day', 'Consensus accuracy'])

with tab_kill:
    # ⚠️ V6.4. The kill line is the only number that governs what happens next,
    # and it was living in a query run occasionally.
    if not settled_fav:
        st.caption('No settled FAV V1 bets yet.')
    else:
        cols = st.columns(3)
        for i, w in enumerate(('MORNING', 'MIDDAY', 'AFTERNOON')):
            g = [b for b in settled_fav if b.get('window_label') == w]
            with cols[i]:
                if not g:
                    st.metric(w, '—')
                    continue
                n = len(g)
                wins = sum(1 for b in g if b['result'] == 'Won')
                asks = [float(b['yes_ask_cents']) for b in g
                        if b.get('yes_ask_cents')]
                avg_ask = sum(asks) / len(asks) if asks else 0
                wp = 100.0 * wins / n
                be = break_even(avg_ask) if avg_ask else 0
                margin = wp - be
                st.metric(w, f'{margin:+.1f} pts',
                          delta=f'n={n} of {KILL_LINE_N}', delta_color='off')
                if n >= KILL_LINE_N and margin < 0:
                    st.error(f'⛔ AT THE LINE — {wp:.1f}% against a '
                             f'{be:.1f}% break-even on n={n}.')
                elif n >= KILL_LINE_N:
                    st.success(f'✅ Clears its break-even at n={n}.')
                else:
                    st.info(f'{KILL_LINE_N - n} more settled bets to the '
                            f'decision point.')
        st.caption(
            '**A window retires when its win rate sits below its OWN average '
            'break-even at n=50 settled.** Break-even is entry price plus '
            'Kalshi\'s fee at that price — a curve, not a number: 58c needs '
            '59.7%, 69c needs 70.5%, 79c needs 80.2%. A 79c bet winning 75% of '
            'the time is LOSING; a 58c bet winning 62% is WINNING. One fee at '
            'entry; holding a winner to settlement costs nothing extra.')

with tab_win:
    if not settled_fav:
        st.caption('No settled FAV V1 bets yet.')
    else:
        rows = []
        for w in ('MORNING', 'MIDDAY', 'AFTERNOON'):
            g = [b for b in settled_fav if b.get('window_label') == w]
            if not g:
                continue
            n = len(g)
            wins = sum(1 for b in g if b['result'] == 'Won')
            asks = [float(b['yes_ask_cents']) for b in g if b.get('yes_ask_cents')]
            avg_ask = sum(asks) / len(asks) if asks else 0
            net = sum(float(b.get('net_profit') or 0) for b in g)
            wp = 100.0 * wins / n
            be = break_even(avg_ask)
            rows.append({
                'Window': w, 'n': n, 'Wins': wins,
                'Win %': f'{wp:.1f}',
                'Avg Ask': f'{avg_ask:.1f}c',
                'Break-even': f'{be:.1f}%',
                'Margin': f'{wp - be:+.1f}',
                'Net': f'${net:+.2f}',
            })
        tot_n = len(settled_fav)
        tot_w = sum(1 for b in settled_fav if b['result'] == 'Won')
        tot_net = sum(float(b.get('net_profit') or 0) for b in settled_fav)
        all_asks = [float(b['yes_ask_cents']) for b in settled_fav
                    if b.get('yes_ask_cents')]
        tot_ask = sum(all_asks) / len(all_asks) if all_asks else 0
        tot_wp = 100.0 * tot_w / tot_n
        tot_be = break_even(tot_ask)
        rows.append({'Window': 'TOTAL', 'n': tot_n, 'Wins': tot_w,
                     'Win %': f'{tot_wp:.1f}', 'Avg Ask': f'{tot_ask:.1f}c',
                     'Break-even': f'{tot_be:.1f}%',
                     'Margin': f'{tot_wp - tot_be:+.1f}',
                     'Net': f'${tot_net:+.2f}'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_day:
    if not settled_fav:
        st.caption('No settled bets yet.')
    else:
        by_day = {}
        for b in settled_fav:
            by_day.setdefault(b.get('date'), []).append(b)
        rows = []
        for d in sorted(by_day, reverse=True):
            g = by_day[d]
            n = len(g)
            wins = sum(1 for x in g if x['result'] == 'Won')
            net = sum(float(x.get('net_profit') or 0) for x in g)
            late = max((x.get('minutes_late') or 0) for x in g)
            rows.append({
                'Date': d, 'n': n, 'Wins': wins,
                'Win %': f'{100.0*wins/n:.1f}',
                'Net': f'${net:+.2f}',
                'Worst late': f'{late}m',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption('Worst late should read 0m — that is the cron-job.org dispatch '
                   'landing on time. If it climbs, entry prices are not what the '
                   'band was measured on.')

with tab_acc:
    if not settled_all:
        st.caption('No settled rows in the last 30 days.')
    else:
        errs = [float(r['actual']) - float(r['consensus']) for r in settled_all
                if r.get('actual') is not None and r.get('consensus') is not None]
        if errs:
            mae = sum(abs(e) for e in errs) / len(errs)
            mean = sum(errs) / len(errs)
            a1, a2, a3 = st.columns(3)
            a1.metric('MAE', f'{mae:.2f} F')
            a2.metric('Mean Error', f'{mean:+.2f} F')
            a3.metric('N', str(len(errs)))
            st.caption('error = actual − consensus. **POSITIVE means settlement '
                       'came in WARMER than predicted — the model runs COLD.**')

        by_city = {}
        for r in settled_all:
            if r.get('actual') is None or r.get('consensus') is None:
                continue
            by_city.setdefault(r['city'], []).append(
                float(r['actual']) - float(r['consensus']))
        rows = []
        for city, e in sorted(by_city.items(),
                              key=lambda kv: sum(abs(x) for x in kv[1]) / len(kv[1])):
            rows.append({
                'City': city, 'n': len(e),
                'MAE': f'{sum(abs(x) for x in e)/len(e):.2f}',
                'Mean Err': f'{sum(e)/len(e):+.2f}',
                'Worst': f'{max(e, key=abs):+.1f}',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── 5. TODAY'S CONSENSUS ─────────────────────────────────────────────────────
st.markdown('<div class="sec">🎯 Today\'s Consensus</div>', unsafe_allow_html=True)

cons_rows = fetch_today_consensus()

if not cons_rows:
    st.caption('No consensus rows today. fetch_weather.py V6 writes these at '
               '14:00 UTC.')
else:
    obs_by_city = {r.get('city'): r for r in obs_rows} if obs_rows else {}
    tbl = []
    for r in sorted(cons_rows, key=lambda x: x.get('city') or ''):
        city = r.get('city')
        c = r.get('consensus')
        o = obs_by_city.get(city, {})
        o_age = age_seconds(o.get('updated_at')) if o else None
        o_dead = (o_age is None or o_age > STALE_HARD_SEC or city in dupe_cities)
        fmax = None if o_dead else o.get('day_max_f')
        togo = round(c - fmax, 1) if (c is not None and fmax is not None) else None
        tbl.append({
            'City': city,
            'Consensus': f'{c:.1f}' if c is not None else '—',
            'Day Max': '⛔' if o_dead else (f'{fmax:.1f}' if fmax is not None else '—'),
            'To Go': f'{togo:+.1f}' if togo is not None else '—',
            'NWS': f"{r['forecast']:.1f}" if r.get('forecast') is not None else '—',
            'GFS': f"{r['ensemble_mean']:.1f}" if r.get('ensemble_mean') is not None else '—',
            'Bias': f"{r['bias_correction']:+.2f}" if r.get('bias_correction') is not None else '—',
            '⚠️': '⚠️' if r.get('high_uncertainty') else '',
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.caption(
        '**To Go** = consensus minus the day\'s feed max. Negative means the '
        'station has already passed the forecast. ⛔ means the obs row is stale '
        'so To Go cannot be computed. ⚠️ = NWS and GFS disagree by more than '
        '5°F. **Consensus has never picked a profitable bracket** — 205 paper '
        'bets, all negative, and naked consensus beat the model\'s own pick '
        '74.3% to 55.2% on 626 city-days. This is for seeing where the day '
        'stands, not for betting.')

st.markdown('---')
st.caption('V6.4 — decision board, market timeline, auto-scored grid, kill-line '
           'status. Freshness measured at read time, off-grid treated as '
           'suspect, duplicate rows surfaced. No gates, no trust scores, no bet '
           'selection. FAV V1 places the bets; this reads the tables.')
