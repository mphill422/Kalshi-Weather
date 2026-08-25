"""
fetch_lows_snapshots.py — V5.30 lows Kalshi ladder snapshot collector
======================================================================
Writes to a NEW table (`lows_snapshots`). Touches NOTHING in the highs model.
Does not import from or modify fetch_weather.py, streamlit_app.py, or
fetch_lows.py. Safe to run during the highs validation freeze.

WHY THIS FILE EXISTS
--------------------
On 2026-08-25 the morning price probe returned the first real live lows ladder.
Reading it exposed the actual blocker:

    Houston      stdev 1.46F   top bracket 77c
    Los Angeles  stdev 1.06F   top bracket 82c
    Dallas       stdev 2.35F   top bracket 49c, 3 live brackets

There are now 60 days of settled lows VALUES in `lows_settlements`, but ZERO
days of historical lows PRICES. So "is 77c cheap for Houston?" is unanswerable.
Answering it requires knowing how often the market's top bracket at 77c actually
settles — which needs price history joined to settlement.

The highs side already has exactly this in `kalshi_snapshots` (15,027 rows, 72
days), and it is what produced the two most important findings of the project:
the model losing to the market 762-171 on disagreements, and the price-band P&L
table. The lows side has no equivalent. This file creates one.

DESIGN NOTES — deliberately different from kalshi_snapshots
------------------------------------------------------------
1. `kalshi_snapshots` captures exactly TWO points per city-day (EDGE and
   CONVICTION windows). That turned out to be a real limitation: it cannot
   answer anything about intraday price evolution, and the afternoon
   near-certainty strategy is unmeasurable because it happens after the last
   capture. This collector is designed to run on a frequent cron instead, so
   the lows side does not inherit that blind spot.

2. `utc_hour` and per-city `local_hour` are both stored. The 31-day timing work
   found median minimum times of 05:36-07:30 local but with FAT TAILS — only
   3/18 cities have tight timing, and Miami puts its minimum outside the
   03:00-09:00 dawn window 48% of the time. So "how do prices behave relative to
   the minimum" needs local hour, not UTC hour.

3. `local_date` is the key, not ET date. Kalshi settles lows on the city's own
   calendar date. A 5am PT minimum belongs to the Pacific date while ET has
   already rolled over. Join `lows_snapshots.local_date` to
   `lows_settlements.date` — both are city-local.

4. `bracket_rank` is the MARKET's price ranking (rank 1 = most expensive),
   mirroring kalshi_snapshots so the same analysis patterns transfer. Note the
   highs table has contamination at ranks 7-12 (~40 rows where rank 7 priced
   ABOVE rank 6, which is structurally impossible in a price-sorted ladder).
   This collector stores `n_brackets` per snapshot so that class of parsing
   artifact is detectable rather than silent.

5. Both yes and no prices are stored. The highs analysis only ever used yes,
   but NO-side edge is an open question on the highs punchlist and there is no
   reason to discard the data.

CREATE THE TABLE ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.lows_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    local_date TEXT NOT NULL,          -- city-LOCAL date; join key to lows_settlements
    et_date TEXT,                      -- ET date, for cross-referencing highs tables
    city TEXT NOT NULL,
    station TEXT,
    series TEXT,
    bracket_label TEXT NOT NULL,
    bracket_rank INTEGER,              -- 1 = most expensive (MARKET's ranking)
    yes_price_cents INTEGER,
    no_price_cents INTEGER,
    n_brackets INTEGER,                -- ladder size this snapshot; artifact check
    utc_hour INTEGER,
    local_hour INTEGER                 -- city-local hour; use THIS for timing work
  );
  ALTER TABLE public.lows_snapshots ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.lows_snapshots
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_lows_snapshots_city_date
    ON public.lows_snapshots (city, local_date);
  CREATE INDEX IF NOT EXISTS idx_lows_snapshots_time
    ON public.lows_snapshots (snapshot_time);

CRON GUIDANCE
-------------
Every 30-60 min is plenty and keeps the table manageable (18 cities x 6 brackets
x 48 runs/day ~ 5,200 rows/day at 30-min cadence). Unlike the intraday
atmospherics collector there is no reason to run this at 20-min resolution.

Do NOT carve a narrow window around dawn. The timing data says minimum timing is
unreliable in 15/18 cities, so a dawn-only window would presuppose the answer to
the question this table exists to answer.

WHAT THIS ENABLES (once ~3-4 weeks have accumulated)
-----------------------------------------------------
  - Top-bracket hit rate by price band, joined to lows_settlements. The single
    question that decides whether lows are bettable.
  - Whether the tight-dispersion cities (LA 1.06F, San Antonio 1.29F, Houston
    1.46F, New Orleans 1.62F) are priced efficiently or not.
  - How lows ladders evolve through the night and whether there is a cheap entry
    window — the lows analogue of the highs EDGE->CONVICTION drift finding.
  - Whether the ~2F Kalshi bracket width is well matched to actual dispersion.

Secrets needed: SUPABASE_URL, SUPABASE_KEY. (No WETHR key — Kalshi is public.)
ALWAYS exits 0 — a collection hiccup must never spam failure emails.
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta

import pytz
import requests

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
HEADERS = {'User-Agent': 'kalshi-lows-snapshots/1.0', 'Accept': 'application/json'}

REQUEST_SPACING_SECONDS = 0.4

# 18 cities. Series tickers verified live on Kalshi 2026-08-23 (all 20 resolved,
# including the previously-unverified SF/SEA guesses; those two are excluded here
# because they are out of the highs model and not currently tracked).
CITIES = {
    'Phoenix':       {'series': 'KXLOWTPHX',  'tz': 'America/Phoenix',     'station': 'KPHX'},
    'Las Vegas':     {'series': 'KXLOWTLV',   'tz': 'America/Los_Angeles', 'station': 'KLAS'},
    'Los Angeles':   {'series': 'KXLOWTLAX',  'tz': 'America/Los_Angeles', 'station': 'KLAX'},
    'Dallas':        {'series': 'KXLOWTDAL',  'tz': 'America/Chicago',     'station': 'KDFW'},
    'Austin':        {'series': 'KXLOWTAUS',  'tz': 'America/Chicago',     'station': 'KAUS'},
    'Houston':       {'series': 'KXLOWTHOU',  'tz': 'America/Chicago',     'station': 'KHOU'},
    'Atlanta':       {'series': 'KXLOWTATL',  'tz': 'America/New_York',    'station': 'KATL'},
    'Miami':         {'series': 'KXLOWTMIA',  'tz': 'America/New_York',    'station': 'KMIA'},
    'New York':      {'series': 'KXLOWTNYC',  'tz': 'America/New_York',    'station': 'KNYC'},
    'San Antonio':   {'series': 'KXLOWTSATX', 'tz': 'America/Chicago',     'station': 'KSAT'},
    'New Orleans':   {'series': 'KXLOWTNOLA', 'tz': 'America/Chicago',     'station': 'KMSY'},
    'Philadelphia':  {'series': 'KXLOWTPHIL', 'tz': 'America/New_York',    'station': 'KPHL'},
    'Boston':        {'series': 'KXLOWTBOS',  'tz': 'America/New_York',    'station': 'KBOS'},
    'Denver':        {'series': 'KXLOWTDEN',  'tz': 'America/Denver',      'station': 'KDEN'},
    'Oklahoma City': {'series': 'KXLOWTOKC',  'tz': 'America/Chicago',     'station': 'KOKC'},
    'Minneapolis':   {'series': 'KXLOWTMIN',  'tz': 'America/Chicago',     'station': 'KMSP'},
    'Washington DC': {'series': 'KXLOWTDC',   'tz': 'America/New_York',    'station': 'KDCA'},
    'Chicago':       {'series': 'KXLOWTCHI',  'tz': 'America/Chicago',     'station': 'KMDW'},
}

KALSHI_URL = 'https://api.elections.kalshi.com/trade-api/v2/markets'


def get_eastern_datetime():
    return datetime.now(pytz.timezone('America/New_York'))


def city_local_now(tz_name):
    return datetime.now(pytz.timezone(tz_name))


# ── Label parsing (mirrors fetch_lows.py verbatim) ───────────────────────────
def normalize_label(label):
    if not label:
        return ''
    label = label.strip()
    label = re.sub(r'(\d+)\s+to\s+(\d+)',
                   lambda m: m.group(1) + '-' + m.group(2), label, flags=re.I)
    label = re.sub(r'(\d+)\s*[\-\u2013\u2014]\s*(\d+)',
                   lambda m: m.group(1) + '-' + m.group(2), label)
    label = re.sub(r'\s+or\s+below', ' or below', label, flags=re.I)
    label = re.sub(r'\s+or\s+above', ' or above', label, flags=re.I)
    return label.replace('\u00b0', '').replace('deg', '').replace('+', ' or above').strip()


def parse_market_label(m):
    for field in ['subtitle', 'yes_sub_title', 'no_sub_title']:
        s = normalize_label((m.get(field) or '').replace('\u00b0', '').strip())
        if s:
            below = re.match(r'^(\d+)\s*or\s*below$', s, re.I)
            above = re.match(r'^(\d+)\s*or\s*above$', s, re.I)
            rng = re.match(r'^(\d+)-(\d+)$', s)
            if below:
                return below.group(1) + ' or below'
            if above:
                return above.group(1) + ' or above'
            if rng:
                return rng.group(1) + '-' + rng.group(2)
    cap, floor_s = m.get('cap_strike'), m.get('floor_strike')
    if cap is not None and floor_s is not None:
        try:
            return f'{int(float(floor_s))}-{int(float(cap))}'
        except Exception:
            pass
    if cap is not None:
        try:
            return f'{int(float(cap))} or below'
        except Exception:
            pass
    return None


def get_price_cents(m):
    yes_c = no_c = None
    for f in ['yes_ask_dollars', 'yes_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                yes_c = round(float(v) * 100)
                break
            except Exception:
                pass
    for f in ['no_ask_dollars', 'no_bid_dollars']:
        v = m.get(f)
        if v:
            try:
                no_c = round(float(v) * 100)
                break
            except Exception:
                pass
    if yes_c is None:
        raw = m.get('yes_ask') or m.get('yes_bid')
        if raw is not None:
            try:
                yes_c = int(raw)
            except Exception:
                pass
    if no_c is None:
        raw = m.get('no_ask') or m.get('no_bid')
        if raw is not None:
            try:
                no_c = int(raw)
            except Exception:
                pass
    return yes_c, no_c


def fetch_ladder(series):
    """Returns [(label, yes_cents, no_cents)] sorted by yes price DESC, or []."""
    et = get_eastern_datetime()
    event_ticker = series + '-' + et.strftime('%y%b%d').upper()
    today_fmt = et.strftime('%y%b%d').upper()
    tomorrow_utc = (et + timedelta(days=1)).strftime('%Y-%m-%d')

    def _try(params):
        try:
            r = requests.get(KALSHI_URL, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    data = _try({'event_ticker': event_ticker, 'limit': 30})
    if not (data and data.get('markets')):
        data = _try({'series_ticker': series, 'status': 'open', 'limit': 30})
    if not (data and data.get('markets')):
        data = _try({'series_ticker': series, 'limit': 30})
    if not (data and data.get('markets')):
        return []

    all_markets = data['markets']
    markets = [m for m in all_markets
               if today_fmt in (m.get('event_ticker') or '').upper()]
    if not markets:
        markets = [m for m in all_markets
                   if (m.get('close_time') or '').startswith(tomorrow_utc)]
    if not markets:
        markets = all_markets

    out, seen = [], set()
    for m in markets:
        label = parse_market_label(m)
        if not label:
            continue
        norm = normalize_label(label)
        if norm in seen:
            continue
        seen.add(norm)
        yes_c, no_c = get_price_cents(m)
        if yes_c is None:
            continue
        out.append((label, yes_c, no_c))
    out.sort(key=lambda x: -x[1])
    return out


def sb_insert(rows):
    if not rows:
        return True
    try:
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': 'Bearer ' + SUPABASE_KEY,
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal',
        }
        r = requests.post(SUPABASE_URL + '/rest/v1/lows_snapshots',
                          headers=headers, json=rows, timeout=25)
        if r.status_code not in (200, 201, 204):
            print(f'    ❌ insert HTTP {r.status_code}: {r.text[:200]}')
            return False
        return True
    except Exception as e:
        print(f'    ❌ insert EXC: {type(e).__name__}: {str(e)[:120]}')
        return False


def main():
    et = get_eastern_datetime()
    utc_hour = datetime.utcnow().hour
    print(f'=== lows ladder snapshot | {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC '
          f'| {len(CITIES)} cities ===')

    if not SUPABASE_URL or not SUPABASE_KEY:
        print('SUPABASE creds missing — nothing written (exit 0).')
        sys.exit(0)

    logged_cities = 0
    total_rows = 0
    thin = []

    for n, (city, cfg) in enumerate(CITIES.items()):
        if n:
            time.sleep(REQUEST_SPACING_SECONDS)

        ladder = fetch_ladder(cfg['series'])
        if not ladder:
            print(f'  [{city}] no ladder — skipped')
            continue

        local_now = city_local_now(cfg['tz'])
        payload = []
        for rank, (label, yes_c, no_c) in enumerate(ladder, start=1):
            payload.append({
                'local_date': local_now.strftime('%Y-%m-%d'),
                'et_date': et.strftime('%Y-%m-%d'),
                'city': city,
                'station': cfg['station'],
                'series': cfg['series'],
                'bracket_label': label,
                'bracket_rank': rank,
                'yes_price_cents': yes_c,
                'no_price_cents': no_c,
                'n_brackets': len(ladder),
                'utc_hour': utc_hour,
                'local_hour': local_now.hour,
            })

        # A ladder that has collapsed to one live bracket is a settled or
        # near-settled market. Worth noting, not worth discarding — the settled
        # state is itself informative about when a city locks.
        live = sum(1 for _, y, _ in ladder if y >= 10)
        if live <= 1:
            thin.append(city)

        if sb_insert(payload):
            logged_cities += 1
            total_rows += len(payload)
            top_label, top_yes, _ = ladder[0]
            print(f'  [{city}] lh={local_now.hour} {len(ladder)} brackets, '
                  f'top {top_label!r} @ {top_yes}c, {live} live ✅')
        else:
            print(f'  [{city}] fetched {len(payload)} but DB write FAILED')

    print(f'\nLogged {logged_cities}/{len(CITIES)} cities, {total_rows} rows.')
    if thin:
        print(f'Near-settled ladders (<=1 live bracket): {", ".join(thin)}')
    sys.exit(0)


if __name__ == '__main__':
    main()
