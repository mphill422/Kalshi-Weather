"""
app.py — MPH Weather, V6.2

V6.2 — THE QUANTIZATION BUG (2026-09-09 evening)
=================================================
V6.1 shipped a bracket check that read a feed value as a measurement. On
Atlanta that evening it said:

    BROKEN — max 89.6 is already above 89.

That was false. 89.6F is EXACTLY 32C. The station transmits whole degrees
Celsius between hourly METARs, so "89.6" means "somewhere that rounds to 32C",
which is 88.7F to 90.5F. Roughly 44% of that window settles 89, 56% settles 90.

The Kalshi ladder at that moment: 88-89 at 51%, 90-91 at 49%. The market had it
right and the app was calling a coin flip a certainty.

⚠️ EVERY VALUE THAT LOOKED PRECISE WAS ON THE GRID.
    95.0 = 35C    96.8 = 36C    98.6 = 37C
    73.4 = 23C    75.2 = 24C    89.6 = 32C
Boston's 81.0F, the one reading that was NOT on the grid, turned out to be a
stale row from an earlier day. Do not conclude a station reports Fahrenheit
tenths because one number looked like it did.

So the bracket check now returns one of three answers:
    BROKEN     — even the LOW end of the window settles above your ceiling
    SAFE       — even the HIGH end settles at or below it
    UNRESOLVED — the window straddles the boundary, with the share of it that
                 settles at or below your ceiling

and the quantization window is printed under the headline number rather than
buried. A 1.8F ambiguity on a 2F bracket is not a footnote.

WHAT CHANGED IN V6.1, AND WHY
===============================
V6.0 showed the 5-minute feed and the hourly METAR side by side as equals.
2026-09-09 proved they are not equals, and which one leads matters when you are
watching a live position.

Boston, that afternoon:

    feed max     73.4F   (201 observations)
    precise max  69.98F  (9 METARs)
    19:54 METAR  21.7C = 71.1F
    20:15 feed   24.0C = 75.2F

The METAR record was 2.3F BELOW the feed and four degrees below where the day
actually got to. Nine hourly samples cannot catch a peak that happens between
:51 reports. Meanwhile the Kalshi ladder had already moved — 75-or-below fell
from 16c to 6c — because the market was reading the same 5-minute data the
feed was.

⚠️ SO THE FEED MAX LEADS NOW. It is the headline number. The T-group is a
secondary reading, useful for one job: telling you exactly where you sit when
the running max is parked on a bracket boundary. That is what it did on
2026-09-08 in New York, where the display said 79.0 and the station had
transmitted 25.6C = 78.1F — a full degree, sitting on the 79/80 line.

Both readings, ranked by what they are good for:

    day_max_f      every ~5 min, 200+ samples. CATCHES THE PEAK.
                   Quantized to whole degrees Celsius on most stations, so it
                   can read up to 0.9F low. Boston and Minneapolis report
                   native Fahrenheit tenths and do not have this problem.

    precise_max_f  exact to a tenth, 9-14 samples a day. MISSES PEAKS between
                   :51 reports. It is a FLOOR, never the answer.

OTHER CHANGES
  - Cache dropped 60s -> 10s on obs_live. On 2026-09-09 a tab left open since
    morning served an 81.0F reading that was hours stale while the real max was
    73.4. If you are watching a position, near-zero is the only safe cache.
  - Every panel stamps how old its data is, in seconds.
  - Bracket proximity is stated in values the station can actually send.

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


def today_et():
    return datetime.now(ET).strftime('%Y-%m-%d')


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
.hero { background:#0d1b2a; border:2px solid #00ff88; border-radius:10px;
        padding:14px 20px; margin-bottom:10px; }
.hero-l { color:#64748b; font-size:11px; text-transform:uppercase;
          letter-spacing:1px; }
.hero-v { color:#00ff88; font-size:34px; font-weight:700;
          font-family:'JetBrains Mono',monospace; line-height:1.1; }
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


# ⚠️ 10s, not 60s. A tab left open since morning served a 3-hour-stale max on
# 2026-09-09 and it read as live. If you are watching a position, cache is risk.
@st.cache_data(ttl=10)
def fetch_obs_live():
    return sb_get('obs_live', {'local_date': 'eq.' + today_et(),
                               'order': 'city.asc', 'limit': '50'})


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
    32C: 31.5 to 32.5C, or 88.7 to 90.5F. A 1.8F window.

    Returns (lo_f, hi_f, celsius_int, is_on_grid).

    ⚠️ VERIFIED THE HARD WAY. Every value that looked like a precise Fahrenheit
    reading turned out to be on the grid:
        95.0 = 35C   96.8 = 36C   98.6 = 37C
        73.4 = 23C   75.2 = 24C   89.6 = 32C
    The one Boston reading that was NOT on the grid (81.0F) was a stale row
    from an earlier day. Do not assume a station reports tenths because one
    number looked like it did.

    Off-grid values are returned as a zero-width band — nothing to widen.
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
                 font-family:'JetBrains Mono',monospace;">V6.2</span></div>
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
        feed_age = row.get('obs_age_min')
        feed_max = row.get('day_max_f')
        nxt = row.get('next_step_f')
        met_now = row.get('metar_temp_f')
        met_age = row.get('metar_age_min')
        prec_max = row.get('precise_max_f')
        n_obs = row.get('n_obs_today')
        n_met = row.get('n_metars_today')
        updated = row.get('updated_at')

        # THE HEADLINE. 200+ samples, catches the peak. This is the number that
        # matters for a bracket, and it is the one the market is reading.
        # But it is quantized — say so right under it, not three panels down.
        _band_note = ''
        if feed_max is not None:
            _lo, _hi, _c, _og = quantization_band(feed_max)
            if _og:
                _band_note = f' · true peak {_lo}–{_hi} ({_c}°C)'

        hero_l, hero_r = st.columns([2, 3])
        with hero_l:
            st.markdown(
                f'<div class="hero">'
                f'<div class="hero-l">Day Max — 5-min feed</div>'
                f'<div class="hero-v">{feed_max:.1f}°F</div>'
                f'<div class="sub">{n_obs or 0} obs today · '
                f'{"obs " + format(feed_age, ".0f") + "m ago" if feed_age is not None else "—"}'
                f'{_band_note}</div></div>',
                unsafe_allow_html=True)
        with hero_r:
            c1, c2, c3 = st.columns(3)
            c1.metric('Now', f'{feed_now:.1f}' if feed_now is not None else '—')
            c2.metric('Next possible', f'{nxt:.1f}' if nxt is not None else '—')
            c3.metric('Trend 30m',
                      f"{row.get('trend_30min'):+.1f}"
                      if row.get('trend_30min') is not None else '—')

        # SECONDARY. Exact, but samples too rarely to be a max. Its one job is
        # telling you where you sit when the max is parked on a boundary.
        with st.container():
            m1, m2 = st.columns([1, 3])
            with m1:
                st.metric('METAR (exact)',
                          f'{met_now:.1f}' if met_now is not None else '—')
            with m2:
                st.write('')
                gap_note = ''
                if feed_max is not None and prec_max is not None:
                    g = round(feed_max - prec_max, 1)
                    gap_note = (f' · METAR max {prec_max:.1f} '
                                f'({g:+.1f} vs feed)')
                st.markdown(
                    f'<div class="sub" style="padding-top:14px;">'
                    f'{f"{met_age:.0f}m ago" if met_age is not None else "no METAR yet"} · '
                    f'{n_met or 0} today{gap_note}</div>',
                    unsafe_allow_html=True)

        if prec_max is not None and feed_max is not None and (feed_max - prec_max) >= 1.5:
            st.caption(f'⚠️ The hourly METAR record tops out {feed_max - prec_max:.1f}F '
                       f'below the 5-minute feed. With {n_met or 0} METARs against '
                       f'{n_obs or 0} feed obs, the peak fell between :51 reports. '
                       f'Trust the feed max.')

        if nxt is not None and feed_max is not None:
            st.caption(f'Feed steps 1.8°F (whole °C). Nothing exists between '
                       f'**{feed_max:.1f}** and **{nxt:.1f}**.')

        # ⚠️ A FEED VALUE ON THE CELSIUS GRID IS A RANGE, NOT A POINT.
        # V6.1 reported it as exact and told the user "BROKEN — max 89.6 is
        # already above 89" for Atlanta on 2026-09-09. But 89.6F is exactly
        # 32C, so the true peak was anywhere in 88.7-90.5F — about 44% of which
        # settles 89. The market was 51/49 and correctly split. Calling that
        # BROKEN was a false read produced by treating a quantized value as a
        # measurement.
        st.markdown('<div class="sub">Bracket check — enter the ceiling you '
                    'care about</div>', unsafe_allow_html=True)
        b1, b2 = st.columns([1, 4])
        with b1:
            ceiling = st.number_input('Ceiling °F', min_value=0, max_value=130,
                                      value=int(feed_max) if feed_max else 80,
                                      step=1, label_visibility='collapsed')
        with b2:
            st.write('')
            if feed_max is not None:
                lo, hi, c_round, on_grid = quantization_band(feed_max)
                # CLI rounds to the nearest whole degree F
                settle_lo = int(lo + 0.5)
                settle_hi = int(hi + 0.5 - 1e-9)

                if settle_lo > ceiling:
                    st.error(f'BROKEN — the peak settles {settle_lo} at best, '
                             f'above your {ceiling} ceiling.')
                elif settle_hi <= ceiling:
                    st.success(f'SAFE so far — the peak settles {settle_hi} at '
                               f'worst, at or below {ceiling}. '
                               f'(Still climbing? next possible '
                               f'{nxt:.1f}F.)' if nxt else
                               f'SAFE so far — settles {settle_hi} at worst.')
                else:
                    share = ((ceiling + 0.5 - lo) / (hi - lo)) if hi > lo else 0.5
                    share = max(0.0, min(1.0, share))
                    st.warning(
                        f'UNRESOLVED — feed max {feed_max:.1f}F is {c_round}°C, '
                        f'so the true peak was **{lo:.1f}–{hi:.1f}°F**. Roughly '
                        f'**{share*100:.0f}%** of that range settles {ceiling} '
                        f'or below. The station cannot tell you more until the '
                        f'next :51 METAR.')

                if on_grid:
                    st.caption(f'⚠️ {feed_max:.1f}F is exactly {c_round}°C — a '
                               f'quantized transmission, not a measurement. The '
                               f'real value is somewhere in a 1.8°F window. '
                               f'Treating it as exact is how you get a false '
                               f'BROKEN.')
                elif met_now is not None:
                    st.caption(f'This station reports native Fahrenheit tenths — '
                               f'the max is exact, no quantization window.')

    with st.expander('All cities', expanded=False):
        tbl = []
        for r in sorted(obs_rows,
                        key=lambda x: (x.get('day_max_f') is None,
                                       -(x.get('day_max_f') or 0))):
            a, m = r.get('obs_age_min'), r.get('metar_age_min')
            fx, px = r.get('day_max_f'), r.get('precise_max_f')
            tbl.append({
                'City': r.get('city', '—'),
                'Stn': r.get('station', '—'),
                'DAY MAX': f"{fx:.1f}" if fx is not None else '—',
                'Now': f"{r['temp_f']:.1f}" if r.get('temp_f') is not None else '—',
                'Age': (f"{a:.0f}m" + (' ⚠️' if a and a > 20 else '')) if a is not None else '—',
                'Next': f"{r['next_step_f']:.1f}" if r.get('next_step_f') is not None else '—',
                'METAR max': f"{px:.1f}" if px is not None else '—',
                'gap': f"{fx - px:+.1f}" if (fx is not None and px is not None) else '—',
                'obs/met': f"{r.get('n_obs_today','—')}/{r.get('n_metars_today','—')}",
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        st.caption('DAY MAX is the 5-minute feed — 200+ samples, catches the peak, '
                   'quantized to whole °C on most stations. METAR max is exact but '
                   'samples 9–14 times a day and routinely misses the peak; the '
                   '**gap** column is how much it is missing by. '
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
        fmax = o.get('day_max_f')
        togo = round(c - fmax, 1) if (c is not None and fmax is not None) else None
        tbl.append({
            'City': city,
            'Consensus': f'{c:.1f}' if c is not None else '—',
            'Day Max': f'{fmax:.1f}' if fmax is not None else '—',
            'To Go': f'{togo:+.1f}' if togo is not None else '—',
            'NWS': f"{r['forecast']:.1f}" if r.get('forecast') is not None else '—',
            'GFS': f"{r['ensemble_mean']:.1f}" if r.get('ensemble_mean') is not None else '—',
            'Bias': f"{r['bias_correction']:+.2f}" if r.get('bias_correction') is not None else '—',
            '⚠️': '⚠️' if r.get('high_uncertainty') else '',
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.caption('**To Go** = consensus minus the day\'s feed max. Negative means '
               'the station has already passed the forecast. ⚠️ = NWS and GFS '
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
st.caption('V6.2 — feed max leads, quantization made explicit. No gates, no trust '
           'scores, no bet selection. FAV V1 places the bets; this reads the tables.')
