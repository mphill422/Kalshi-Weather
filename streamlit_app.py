"""
app.py — MPH Weather, V6.3

V6.3 — THE GREEN BOX THAT LIED (2026-09-09 night)
==================================================
At 5:19pm ET the panel showed San Antonio:

    DAY MAX — 5-MIN FEED   81.0°F
    17 obs today · obs 24m ago

in a green box. The station had actually reached the 98.6F step at ~4pm local.
The row on screen had been written at about 13:24Z — 8:24am local — and had not
moved in nine hours. At 8:39pm the SAME numbers (81.0 / 80.1 / 17 obs / 11
METARs) appeared under **Los Angeles**.

Two separate defects produced that, and V6.2 had no defence against either.

1. NOTHING IN THIS FILE MEASURED FRESHNESS.
   `feed_age = row.get('obs_age_min')` is stamped by the poller at WRITE time.
   A row written at 13:24Z saying "obs_age_min: 4.2" still says 4.2 at 22:00Z.
   The header clock, meanwhile, renders `datetime.now(ET)` — so the page showed
   8:39pm above a reading from breakfast and called it live.

   ⚠️ THE .hero CLASS HARDCODED `border:2px solid #00ff88`. There was no code
   path in V6.2 that could produce a non-green headline. The box was green
   because it is always green, not because the data was good.

   V6.3 computes age from `updated_at` against now, EVERY RENDER. Past
   STALE_HARD_SEC the headline number is replaced by the word STALE. It is not
   dimmed, not caveated — replaced. A number you cannot trust is worse than no
   number, because you will act on it.

2. THE OFF-GRID TEST WAS BACKWARDS — and it is why a corrupt value produced
   the app's MOST confident output.
   V6.2 said: on the Celsius grid -> quantized, show a 1.8F window. Off the
   grid -> `st.caption('This station reports native Fahrenheit tenths — the max
   is exact, no quantization window.')`

   Real ASOS values land ON the grid. 81.0F is not on it (27C = 80.6). So a
   stale, duplicated or corrupt value is EXACTLY the kind that reads as
   off-grid — and V6.2 responded by dropping its error bars and printing a flat
   red BROKEN. Off-grid is a symptom of bad data, not evidence of precision.

   V6.3 treats off-grid as SUSPICIOUS unless the station is on the known
   native-tenths list (Boston, Minneapolis). Otherwise it warns and withholds
   the verdict.

3. NEW: DUPLICATE-ROW DETECTOR.
   The city selector in V6.2 is correct — it filters `r['city'] == sel`. So
   identical numbers under two different cities means the DATABASE holds
   identical rows, which the poller should never write. This file cannot fix
   the poller, but it can refuse to pretend. If two or more cities share the
   same day_max_f AND n_obs_today AND n_metars_today, every affected row is
   flagged and the bracket check is disabled for all of them.

4. TIMEZONE BUG IN THE DATE FILTER.
   `local_date = eq. today_et()` filtered every city by the EASTERN date. The
   poller writes each row under the STATION's local date. Between 9pm ET and
   midnight PT the Pacific cities' rows silently disappear from the dashboard.
   V6.3 queries today AND yesterday and keeps the newest row per city.

--- V6.2 documentation below, unchanged and still true ---

THE QUANTIZATION BUG
V6.1 shipped a bracket check that read a feed value as a measurement. On
Atlanta it said "BROKEN — max 89.6 is already above 89." That was false. 89.6F
is EXACTLY 32C. The station transmits whole degrees Celsius between hourly
METARs, so "89.6" means "somewhere that rounds to 32C", which is 88.7F to
90.5F. Roughly 44% of that window settles 89, 56% settles 90. The Kalshi ladder
at that moment: 88-89 at 51%, 90-91 at 49%. The market had it right and the app
was calling a coin flip a certainty.

⚠️ EVERY VALUE THAT LOOKED PRECISE WAS ON THE GRID.
    95.0 = 35C    96.8 = 36C    98.6 = 37C
    73.4 = 23C    75.2 = 24C    89.6 = 32C

So the bracket check returns one of three answers:
    BROKEN     — even the LOW end of the window settles above your ceiling
    SAFE       — even the HIGH end settles at or below it
    UNRESOLVED — the window straddles the boundary, with the share of it that
                 settles at or below your ceiling

⚠️ THE UNRESOLVED SHARE ASSUMES A UNIFORM DISTRIBUTION INSIDE THE BAND.
It is not uniform. If the bracketing hourly METARs both sit below the band, the
true peak almost certainly clipped the BOTTOM of it rather than running to the
top, and the uniform figure overstates the upside. Read the share as an upper
bound on the bad outcome, not a probability. Observed live on San Antonio
2026-09-09: this panel said 44%, the market said 26%.

THE FEED MAX LEADS
Boston, 2026-09-09: feed max 73.4F (201 obs) vs precise max 69.98F (9 METARs).
Nine hourly samples cannot catch a peak between :51 reports.

    day_max_f      every ~5 min, 200+ samples. CATCHES THE PEAK.
                   Quantized to whole degrees Celsius on most stations.
    precise_max_f  exact to a tenth, 9-14 samples a day. MISSES PEAKS.
                   It is a FLOOR, never the answer.

WHAT THIS FILE DOES NOT DO
It places no bets, picks no brackets, computes no probabilities, and has no
gates or trust scores. The 3,356-line version that did all of that produced 205
losing paper bets. FAV V1 places the bets; this reads the tables.

Secrets: supabase.url, supabase.key, app_password (optional).
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title='MPH Weather', layout='wide', page_icon='🌡️')

ET = pytz.timezone('America/New_York')

# ── Freshness thresholds ─────────────────────────────────────────────────────
# The poller runs every 5 minutes. Anything past SOFT is worth flagging;
# anything past HARD is not a number, it is a memory.
STALE_SOFT_SEC = 8 * 60
STALE_HARD_SEC = 15 * 60

# Stations that genuinely transmit Fahrenheit tenths. Everything else that
# lands off the Celsius grid is suspect, not precise.
NATIVE_TENTHS = {'KBOS', 'KMSP'}


def today_et():
    return datetime.now(ET).strftime('%Y-%m-%d')


def yesterday_et():
    return (datetime.now(ET) - timedelta(days=1)).strftime('%Y-%m-%d')


def parse_ts(s):
    """Supabase timestamptz -> aware datetime, or None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except Exception:
        return None


def age_seconds(ts):
    """⚠️ READ-TIME age. This is the whole point of V6.3.

    obs_age_min in the row is computed when the poller WRITES. It does not age.
    A row written at 13:24Z claiming to be 4 minutes old still claims that at
    22:00Z. Only this function knows what time it actually is.
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
# was hours stale and it read as live. If you are watching a position, cache is
# risk. NOTE: a short cache does NOT make data fresh — it only makes the app
# re-read a possibly-frozen row more often. Freshness is age_seconds().
@st.cache_data(ttl=10)
def fetch_obs_live():
    """⚠️ TWO DATES, NOT ONE.

    V6.2 filtered `local_date = eq. today_et()`. The poller writes each row
    under the STATION's local date, so between 9pm ET and midnight PT every
    Pacific city vanished from the dashboard. Query both days, keep the newest
    row per city.
    """
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
    return sb_get('favorites_bets', {'order': 'date.desc', 'limit': '1000'})


def kalshi_fee_cents(price_cents):
    """Per-contract fee in cents, per Kalshi's published schedule:
        fee = round up(M x 0.07 x C x P x (1-P)),  M defaults to 1
    Confirmed 2026-09-08 against a real fill (30 contracts @ 65c -> $0.48).
    Weather series are not in the non-standard multiplier table, so M = 1."""
    p = price_cents / 100.0
    return 0.07 * p * (1 - p) * 100


def break_even(price_cents):
    """Win rate needed to break even, holding to settlement (no exit fee)."""
    return round(price_cents + kalshi_fee_cents(price_cents), 1)


def quantization_band(f):
    """What a 5-minute feed reading ACTUALLY tells you.

    Most ASOS stations transmit whole degrees CELSIUS between the hourly :51
    METARs. So a feed value of 89.6F is not a measurement of 89.6 — it is the
    station saying "32C", and the true temperature was anywhere that rounds to
    32C: 88.7 to 90.5F. A 1.8F window.

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


def find_duplicate_cities(rows):
    """⚠️ Cities whose readings are byte-identical to another city's.

    On 2026-09-09 San Antonio and Los Angeles both showed 81.0 / 80.1 / 17 obs
    / 11 METARs. The selector in V6.2 was correct — it filters on city — which
    means the DATABASE held identical rows. The poller should never write that.
    This cannot fix it, but it refuses to display it as if it were real.
    """
    sig = {}
    for r in rows:
        key = (r.get('day_max_f'), r.get('temp_f'),
               r.get('n_obs_today'), r.get('n_metars_today'))
        if key == (None, None, None, None):
            continue
        sig.setdefault(key, []).append(r.get('city'))
    dupes = set()
    for key, cities in sig.items():
        if len(cities) > 1:
            dupes.update(c for c in cities if c)
    return dupes


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
                 font-family:'JetBrains Mono',monospace;">V6.3</span></div>
  <div style="font-size:12px;color:#64748b;font-family:'JetBrains Mono',monospace;">
    {now_et:%Y-%m-%d %I:%M:%S %p ET} · settles on Iowa State CLI</div>
</div>
""", unsafe_allow_html=True)
with h2:
    st.write('')
    if st.button('🔄 Refresh', use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── 1. LIVE OBS ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📡 Live Obs — Settlement Station</div>',
            unsafe_allow_html=True)

obs_rows = fetch_obs_live()
dupe_cities = find_duplicate_cities(obs_rows)

if dupe_cities:
    st.error(
        f'⚠️ DUPLICATE ROWS IN obs_live — {len(dupe_cities)} cities are '
        f'reporting identical readings: {", ".join(sorted(dupe_cities))}. '
        f'That is a poller fault, not a display fault. Readings for these '
        f'cities cannot be trusted and the bracket check is disabled for them.')

if not obs_rows:
    st.caption('No obs_live rows today. The poller runs every 5 min, 9am–9pm ET '
               'via cron-job.org → obs_live.yml.')
else:
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
        # Never from obs_age_min — that is a write-time stamp and does not age.
        row_age = age_seconds(row.get('updated_at'))
        poller_says_stale = bool(row.get('is_stale'))
        stale_reason = row.get('stale_reason')
        is_dupe = sel in dupe_cities

        hard_stale = (row_age is None) or (row_age > STALE_HARD_SEC)
        soft_stale = (row_age is not None) and (row_age > STALE_SOFT_SEC)
        untrustworthy = hard_stale or is_dupe

        # ── THE HEADLINE ────────────────────────────────────────────────
        # If the row is untrustworthy the NUMBER IS NOT SHOWN. Not greyed,
        # not asterisked — replaced. V6.2 rendered 81.0 in green at 5:19pm
        # from a row written at 8:24am, and it drove a real decision.
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
                f'<div class="hero-v {vklass}">'
                f'{feed_max:.1f}°F</div>'
                f'<div class="sub">{n_obs or 0} obs today · '
                f'written {fmt_age(row_age)}{_band_note}</div></div>',
                unsafe_allow_html=True)
            if soft_stale:
                st.warning(f'Row is {fmt_age(row_age)} — the poller runs every '
                           f'5 minutes, so this is behind. Treat as indicative.')
            if poller_says_stale and stale_reason:
                st.warning(f'Poller flagged this row: {stale_reason}')

        c1, c2, c3 = st.columns(3)
        c1.metric('Now', '—' if untrustworthy or feed_now is None else f'{feed_now:.1f}')
        c2.metric('Next possible', '—' if untrustworthy or nxt is None else f'{nxt:.1f}')
        c3.metric('Trend 30m',
                  '—' if untrustworthy or row.get('trend_30min') is None
                  else f"{row.get('trend_30min'):+.1f}")

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

        if not untrustworthy and nxt is not None and feed_max is not None:
            st.caption(f'Feed steps 1.8°F. Nothing exists between '
                       f'**{feed_max:.1f}** and **{nxt:.1f}**.')

        # ── BRACKET CHECK ───────────────────────────────────────────────
        st.markdown('<div class="sub">Bracket check — enter the ceiling you '
                    'care about</div>', unsafe_allow_html=True)

        if untrustworthy:
            # ⚠️ V6.2 computed a verdict from whatever was in the row. With a
            # stale 81.0 and an 80 ceiling it printed a flat red BROKEN, with
            # no hedge, because 81.0 is off the Celsius grid — see below.
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
                        # V6.2 said off-grid -> "native Fahrenheit tenths, the
                        # max is exact, no quantization window", and then
                        # printed a hard verdict. But real ASOS values land ON
                        # the grid. 81.0F is not on it (27C = 80.6). Off-grid
                        # is a SYMPTOM OF BAD DATA at every station except the
                        # two that genuinely send tenths.
                        st.warning(
                            f'⚠️ {feed_max:.1f}°F does not sit on this '
                            f'station\'s transmission grid, and {station or "this station"} '
                            f'is not one of the native-tenths sites. That is a '
                            f'sign of a bad or stale reading, not a precise '
                            f'one. No verdict given.')
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
                                'market said 26%.)')

                    if on_grid:
                        st.caption(f'{feed_max:.1f}°F is a quantized '
                                   f'transmission, not a measurement — the real '
                                   f'value is somewhere in a 1.8°F window.')
                    elif native_tenths:
                        st.caption(f'{station} transmits Fahrenheit tenths — '
                                   f'this max is exact.')

    with st.expander('All cities', expanded=False):
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


# ── 2. TODAY'S CONSENSUS ─────────────────────────────────────────────────────
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
    st.caption('**To Go** = consensus minus the day\'s feed max. Negative means '
               'the station has already passed the forecast. ⛔ means the obs '
               'row is stale so To Go cannot be computed. ⚠️ = NWS and GFS '
               'disagree by more than 5°F. Consensus has never picked a '
               'profitable bracket — this is for seeing where the day stands, '
               'not for betting.')


# ── 3. RESULTS ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Results</div>', unsafe_allow_html=True)

fav = fetch_favorites()
settled_fav = [b for b in fav if b.get('result') in ('Won', 'Lost')]

tab_win, tab_day, tab_acc = st.tabs(['By window', 'By day', 'Consensus accuracy'])

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
        all_asks = [float(b['yes_ask_cents']) for b in settled_fav if b.get('yes_ask_cents')]
        tot_ask = sum(all_asks) / len(all_asks) if all_asks else 0
        tot_wp = 100.0 * tot_w / tot_n
        tot_be = break_even(tot_ask)
        rows.append({'Window': 'TOTAL', 'n': tot_n, 'Wins': tot_w,
                     'Win %': f'{tot_wp:.1f}', 'Avg Ask': f'{tot_ask:.1f}c',
                     'Break-even': f'{tot_be:.1f}%',
                     'Margin': f'{tot_wp - tot_be:+.1f}',
                     'Net': f'${tot_net:+.2f}'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption('Break-even is entry price + Kalshi\'s fee at that price — it is '
                   'a curve, not a number (58c needs 59.7%, 79c needs 80.2%). '
                   '**Kill line: a window retires when its win rate sits below its '
                   'own break-even at n=50.** One fee at entry; holding a winner '
                   'to settlement costs nothing extra.')

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
    settled = fetch_settled(30)
    if not settled:
        st.caption('No settled rows in the last 30 days.')
    else:
        errs = [float(r['actual']) - float(r['consensus']) for r in settled
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
        for r in settled:
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

st.markdown('---')
st.caption('V6.3 — freshness measured at read time, off-grid treated as suspect, '
           'duplicate rows surfaced. No gates, no trust scores, no bet '
           'selection. FAV V1 places the bets; this reads the tables.')
