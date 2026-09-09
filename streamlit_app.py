"""
app.py — MPH Weather, V6.0

WHAT THIS REPLACES
==================
streamlit_app.py was 3,356 lines. Its main output was a three-gate bet
selector whose picks lost 205 straight paper bets across four tags. The gates,
the trust tables, the NBM ladder, the Kelly sizing, the ladder editor, and the
per-city quality scores are all gone — none of them decided anything that made
money, and the one thing the app was actually used for every day was reading
the current observation, which it did badly.

This file does three things:

  1. LIVE OBS — the station reading Kalshi settles on, both the 5-minute feed
     and the hourly METAR T-group, with the true age of each.
  2. TODAY'S CONSENSUS — the one number fetch_weather.py V6 writes.
  3. RESULTS — FAV V1 by window and by agreement tag; consensus accuracy.

It places no bets, picks no brackets, and computes no probabilities. It is a
window onto tables that other files write.

WHY THE OBS PANEL LEADS
=======================
On 2026-09-08 the old app showed New York at 79.0F. The station had transmitted
25.6C, which is 78.1F. A full degree, sitting directly on a bracket boundary,
and it changed a live decision. The 79.0 came from the 5-minute ASOS feed,
which transmits WHOLE DEGREES CELSIUS — so a displayed 79.0F really means
"somewhere in 25.5C to 26.4C", or 77.9F to 79.5F.

The hourly METAR carries the precise value in its T-group:

    KNYC 081851Z AUTO 28006KT 10SM CLR 26/13 A3023 RMK AO2 SLP229 T02560128
                                                                  ^^^^^^
`26/13` is the rounded pair everyone displays. `T0256` is 25.6C to a tenth.
Every ASOS METAR has it. Every consumer source throws it away.

fetch_obs_live.py V2 now stores both. This panel shows both, side by side,
with the gap flagged.

⚠️ NEITHER MAX IS STRICTLY BETTER.
  day_max_f     — every 5 min, but quantized to 1.8F steps. Can read up to
                  0.9F LOW.
  precise_max_f — exact to a tenth, but only 12-14 samples a day. A peak that
                  falls between :51 reports is invisible to it, so it is a
                  FLOOR, not the answer.
Kalshi settles on CLI, which is built from the precise record.

⚠️ STATIONS DIFFER ENORMOUSLY. KNYC (Central Park) produced 14 observations by
2:38pm on 2026-09-08 while airport ASOS sites had 244+. It also delayed its
19:51 METAR past 4:03pm — confirmed against two independent NWS paths, so the
station, not a cache. n_obs / n_metars makes that visible per city.

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
                           label_visibility='collapsed',
                           placeholder='Password')
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
    font-family:'JetBrains Mono',monospace !important; font-size:20px !important; }
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


@st.cache_data(ttl=60)
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


@st.cache_data(ttl=120)
def fetch_favorites():
    return sb_get('favorites_bets', {'order': 'date.desc', 'limit': '1000'})


# ── Header ───────────────────────────────────────────────────────────────────
now_et = datetime.now(ET)
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744,#0d1b2a);
            border:1px solid #1e3a5f;border-radius:12px;padding:18px 26px;
            margin-bottom:16px;">
  <div style="font-size:24px;font-weight:700;color:#fff;">🌡️ MPH Weather
    <span style="font-size:11px;color:#00ff88;border:1px solid #00ff8840;
                 background:#00ff8820;padding:2px 9px;border-radius:20px;
                 margin-left:8px;vertical-align:middle;
                 font-family:'JetBrains Mono',monospace;">V6.0</span></div>
  <div style="font-size:12px;color:#64748b;font-family:'JetBrains Mono',monospace;">
    {now_et:%Y-%m-%d %I:%M %p ET} · settlement source: Iowa State CLI</div>
</div>
""", unsafe_allow_html=True)

if st.button('🔄 Refresh'):
    st.cache_data.clear()
    st.rerun()


# ── 1. LIVE OBS ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📡 Live Obs — Settlement Station</div>',
            unsafe_allow_html=True)

obs_rows = fetch_obs_live()

if not obs_rows:
    st.caption('No obs_live rows today. The poller runs every 5 min, 9am–9pm ET '
               'via cron-job.org → obs_live.yml. If empty during those hours, '
               'check the Obs Live workflow.')
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

        def age_color(a, warn, bad):
            if a is None:
                return '#64748b'
            return '#00ff88' if a <= warn else '#f59e0b' if a <= bad else '#ef4444'

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('5-min Feed', f'{feed_now:.1f} F' if feed_now is not None else '—')
            st.markdown(
                f'<div style="color:{age_color(feed_age,10,20)};font-size:11px;'
                f'font-family:\'JetBrains Mono\',monospace;">'
                f'{f"obs {feed_age:.0f}m ago" if feed_age is not None else "—"}'
                f'{f" · {n_obs} today" if n_obs else ""}</div>',
                unsafe_allow_html=True)
        with c2:
            st.metric('METAR (precise)', f'{met_now:.1f} F' if met_now is not None else '—')
            st.markdown(
                f'<div style="color:{age_color(met_age,70,90)};font-size:11px;'
                f'font-family:\'JetBrains Mono\',monospace;">'
                f'{f"{met_age:.0f}m ago" if met_age is not None else "no METAR yet"}'
                f'{f" · {n_met} today" if n_met else ""}</div>',
                unsafe_allow_html=True)
        with c3:
            st.metric('Feed Max', f'{feed_max:.1f} F' if feed_max is not None else '—')
            st.caption('5-min, ±0.9F (whole °C)')
        with c4:
            st.metric('Precise Max', f'{prec_max:.1f} F' if prec_max is not None else '—')
            st.caption('hourly T-group, a floor')

        # the 2026-09-08 gap, made visible
        if feed_max is not None and prec_max is not None:
            gap = round(feed_max - prec_max, 1)
            if abs(gap) >= 0.8:
                st.warning(
                    f'⚠️ Feed max reads {feed_max:.1f}F, precise max reads '
                    f'{prec_max:.1f}F ({gap:+.1f}F). The 5-minute feed is '
                    f'quantized to whole degrees Celsius; the T-group is exact. '
                    f'CLI settles from the precise record.')

        if nxt is not None and feed_max is not None:
            st.caption(f'Next value the feed can transmit: **{nxt:.1f}F** '
                       f'(steps 1.8F). Nothing exists between {feed_max:.1f} and {nxt:.1f}.')

    with st.expander('All cities', expanded=False):
        tbl = []
        for r in sorted(obs_rows,
                        key=lambda x: (x.get('precise_max_f') is None,
                                       -(x.get('precise_max_f') or 0))):
            a, m = r.get('obs_age_min'), r.get('metar_age_min')
            tbl.append({
                'City': r.get('city', '—'),
                'Stn': r.get('station', '—'),
                'Feed': f"{r['temp_f']:.1f}" if r.get('temp_f') is not None else '—',
                'Age': (f"{a:.0f}m" + (' ⚠️' if a and a > 20 else '')) if a is not None else '—',
                'METAR': f"{r['metar_temp_f']:.1f}" if r.get('metar_temp_f') is not None else '—',
                'Age ': (f"{m:.0f}m" + (' ⚠️' if m and m > 75 else '')) if m is not None else '—',
                'Feed Max': f"{r['day_max_f']:.1f}" if r.get('day_max_f') is not None else '—',
                'Prec Max': f"{r['precise_max_f']:.1f}" if r.get('precise_max_f') is not None else '—',
                'Next': f"{r['next_step_f']:.1f}" if r.get('next_step_f') is not None else '—',
                'n/m': f"{r.get('n_obs_today','—')}/{r.get('n_metars_today','—')}",
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        st.caption('⚠️ Preliminary, pre-QC. Kalshi settles on official CLI, not this '
                   'feed. Use it to SEE the day, never to score it.')


# ── 2. TODAY'S CONSENSUS ─────────────────────────────────────────────────────
st.markdown('<div class="sec">🎯 Today\'s Consensus</div>', unsafe_allow_html=True)

cons_rows = fetch_today_consensus()

if not cons_rows:
    st.caption('No consensus rows today. fetch_weather.py V6 writes these at '
               '14:00 UTC. FAV V1.1 reads them at 14:30 for its agreement tag — '
               'if this is empty during the day, morning bets log with a NULL tag.')
else:
    obs_by_city = {r.get('city'): r for r in obs_rows} if obs_rows else {}
    tbl = []
    for r in sorted(cons_rows, key=lambda x: x.get('city') or ''):
        city = r.get('city')
        c = r.get('consensus')
        o = obs_by_city.get(city, {})
        prec = o.get('precise_max_f')
        # how far above the day's precise max does consensus sit?
        togo = round(c - prec, 1) if (c is not None and prec is not None) else None
        tbl.append({
            'City': city,
            'Consensus': f'{c:.1f}' if c is not None else '—',
            'NWS': f"{r['forecast']:.1f}" if r.get('forecast') is not None else '—',
            'GFS': f"{r['ensemble_mean']:.1f}" if r.get('ensemble_mean') is not None else '—',
            'Bias': f"{r['bias_correction']:+.2f}" if r.get('bias_correction') is not None else '—',
            'Obs High': f"{r['obs_high']:.1f}" if r.get('obs_high') is not None else '—',
            'Prec Max': f'{prec:.1f}' if prec is not None else '—',
            'To Go': f'{togo:+.1f}' if togo is not None else '—',
            '⚠️': '⚠️' if r.get('high_uncertainty') else '',
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.caption('**To Go** = consensus minus the day\'s precise max so far. '
               'Negative means the station has already passed the forecast. '
               '⚠️ = NWS and GFS disagree by more than 5F.')


# ── 3. RESULTS ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Results</div>', unsafe_allow_html=True)

tab_fav, tab_agree, tab_acc = st.tabs(
    ['FAV V1 by window', 'Agreement tag', 'Consensus accuracy'])

fav = fetch_favorites()
settled_fav = [b for b in fav if b.get('result') in ('Won', 'Lost')]

with tab_fav:
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
            rows.append({
                'Window': w, 'n': n, 'Wins': wins,
                'Win %': f'{100.0*wins/n:.1f}',
                'Avg Ask': f'{avg_ask:.1f}c',
                'Break-even': f'{avg_ask+3.6:.1f}%',
                'Net': f'${net:+.2f}',
            })
        tot_n = len(settled_fav)
        tot_w = sum(1 for b in settled_fav if b['result'] == 'Won')
        tot_net = sum(float(b.get('net_profit') or 0) for b in settled_fav)
        rows.append({'Window': 'TOTAL', 'n': tot_n, 'Wins': tot_w,
                     'Win %': f'{100.0*tot_w/tot_n:.1f}', 'Avg Ask': '',
                     'Break-even': '', 'Net': f'${tot_net:+.2f}'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption('Kill line: **below 66% win rate at n=50 per window, that '
                   'window retires.** Break-even is avg ask + 3.6c fee — the fee '
                   'rests on ONE observed ticket and is not confirmed.')

with tab_agree:
    tagged = [b for b in settled_fav if b.get('agrees_with_consensus') is not None]
    if not tagged:
        st.caption('No settled bets carry an agreement tag yet. Tagging started '
                   '2026-09-07; the question needs ~50 settled per group.')
    else:
        rows = []
        for w in ('MORNING', 'MIDDAY', 'AFTERNOON'):
            for agree in (True, False):
                g = [b for b in tagged
                     if b.get('window_label') == w
                     and b.get('agrees_with_consensus') is agree]
                if not g:
                    continue
                n = len(g)
                wins = sum(1 for b in g if b['result'] == 'Won')
                net = sum(float(b.get('net_profit') or 0) for b in g)
                rows.append({
                    'Window': w,
                    'Consensus': 'AGREE' if agree else 'DISAGREE',
                    'n': n, 'Wins': wins,
                    'Win %': f'{100.0*wins/n:.1f}',
                    'Net': f'${net:+.2f}',
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption('Backtest (33 days, reconstructed ladders) said AGREE beats '
                   'all-bets by 4.9 points in the morning, and turns midday '
                   '(−1.02 → +13.56) and afternoon (−0.73 → +5.32) positive. '
                   '**Gate line: AGREE must beat DISAGREE by 5+ points at n=50 '
                   'each before switching from tag to filter.** Every bet is '
                   'still taken meanwhile.')

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
            st.caption('Sign convention: error = actual − consensus. **POSITIVE '
                       'means settlement came in WARMER than predicted — the '
                       'model runs COLD.** Misreading this caused a wrong call '
                       'on 2026-08-24.')

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
        st.caption('Consensus MAE runs ~0.73F on a city\'s clearest third of days '
                   'and ~1.10F on its cloudiest third (368 city-days, terciled '
                   'within each city). Uncertainty is predictable even where the '
                   'central estimate is not biased.')

st.markdown('---')
st.caption('V6.0 — consensus writer + station obs. No gates, no trust scores, '
           'no bet selection. FAV V1 places the bets; this reads the tables.')
