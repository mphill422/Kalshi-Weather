"""
kalshi_lows_backfill.py — minute-resolution KXLOWT candle history → kalshi_lows_candles

Companion to kalshi_backfill.py (which does KXHIGH). Same auth, same shape,
three differences that matter:

  1. SERIES FILTER: KXLOWT instead of KXHIGH.

  2. CANDLE WINDOW: 04:00–16:00 UTC on the event date, not 12:00–10:00.
     Derived from data, not guessed. Across 1,077 settled CLI-sourced rows in
     lows_settlements, low_time_utc distributes:
         hours 09-13 UTC → 811 rows (75%)
         hours 04-08 UTC → 163 rows (15%)
         everything else → 103 rows (10%)
     So 04:00-16:00 captures ~95% of actual minimum times. The shape makes
     sense: 09-13 UTC is 5-9am ET / 4-8am CT / 2-6am PT — dawn across the
     country. The thin 18:00-19:00 UTC tail is the convective cases (Miami
     collapses in the afternoon and sets its daily min in the evening 48% of
     the time).

  3. TABLE: kalshi_lows_candles, kept separate from kalshi_candles. Highs and
     lows are different markets with different mechanics and should not share
     a table — pooling them would repeat the mistake of pooling strategy tags.

WHY THIS EXISTS
---------------
The highs analysis off kalshi_candles produced a concrete, actionable result:
the market's top bracket in the 60-69c band, logged between 10:30am and 12:00pm
ET, returned roughly +8 to +17c per contract across every 15-minute slice
tested, positive in 12 of 17 cities. That analysis was only possible because
25 days of minute-level price history existed.

No equivalent exists for lows. lows_snapshots started collecting 2026-08-25 at
30-minute resolution and needs weeks to accumulate. This backfill produces the
same 25 days immediately, so the identical hour x price-band analysis can run
tonight instead of in September.

MARKET MECHANICS (confirmed by direct observation 2026-08-25/26)
----------------------------------------------------------------
A lows market is NOT tradeable the day before. At 12:46am on Aug 26, Kalshi
still showed Aug 25 markets and Aug 26 was not listed. The real sequence is:
prior day settles ~1-3am local, then the next day's market opens, then the
minimum occurs ~5-7am local. The bettable window is a few hours, overnight.

That is why the candle window starts at 04:00 UTC — it needs to cover the
period after settlement and before the minimum, which is where any entry
decision would actually be made.

CREATE THE TABLE ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.kalshi_lows_candles (
    id BIGSERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_ticker TEXT,
    market_ticker TEXT NOT NULL,
    bracket_label TEXT,
    ts_utc TIMESTAMPTZ NOT NULL,
    yes_bid NUMERIC,
    yes_ask NUMERIC,
    volume NUMERIC,
    open_interest NUMERIC,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (market_ticker, ts_utc)
  );
  ALTER TABLE public.kalshi_lows_candles ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.kalshi_lows_candles
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_lows_candles_city_date
    ON public.kalshi_lows_candles (city, event_date);
  CREATE INDEX IF NOT EXISTS idx_lows_candles_ts
    ON public.kalshi_lows_candles (ts_utc);

The UNIQUE (market_ticker, ts_utc) constraint plus merge-duplicates makes this
safe to re-run on overlapping date ranges. kalshi_settlements has no such
constraint — it was checked and happened to be clean (3,354 rows, 3,354
distinct tickers) but that was luck, not design.

PRICE UNITS: yes_bid / yes_ask are stored in DOLLARS (0.62), not cents. Same
as kalshi_candles. Multiply by 100 for the cent figures used in analysis.

Secrets: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY.
NOTE the last one — kalshi_backfill.py reads SUPABASE_SERVICE_KEY while the
weather scripts read SUPABASE_KEY. The repo secret list shows SUPABASE_KEY and
SUPABASE_URL. If this errors on import with a KeyError, map it in the workflow
env block rather than editing this file.
"""

import os, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
BASE = "https://api.elections.kalshi.com"

DAYS_BACK = int(os.environ.get("DAYS_BACK", "25"))

# Candle window, UTC hours on the event date. See header for the derivation.
WINDOW_START_HOUR = 4
WINDOW_HOURS = 12   # 04:00 -> 16:00 UTC

pk = serialization.load_pem_private_key(PEM.encode(), password=None)


def sign(ts, method, path):
    msg = f"{ts}{method}{path}".encode()
    return base64.b64encode(pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )).decode()


def kget(path, params=None):
    for attempt in range(4):
        ts = str(int(time.time() * 1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID,
             "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path)}
        r = requests.get(BASE + path, headers=h, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        print("ERR", r.status_code, path, r.text[:200])
        return None
    return None


def discover_series():
    """All KXLOWT series in the Climate and Weather category."""
    out, cursor = [], None
    while True:
        p = {"limit": 200, "category": "Climate and Weather"}
        if cursor:
            p["cursor"] = cursor
        d = kget("/trade-api/v2/series", p)
        if not d:
            break
        for s in d.get("series", []):
            t = s.get("ticker", "")
            if t.startswith("KXLOWT"):
                out.append(t)
        cursor = d.get("cursor")
        if not cursor:
            break
    return sorted(set(out))


def push(rows):
    if not rows:
        return
    r = requests.post(
        f"{SB_URL}/rest/v1/kalshi_lows_candles?on_conflict=market_ticker,ts_utc",
        headers={"apikey": SB_KEY,
                 "Authorization": f"Bearer {SB_KEY}",
                 "Content-Type": "application/json",
                 "Prefer": "return=minimal,resolution=merge-duplicates"},
        json=rows, timeout=60)
    if r.status_code >= 300:
        print("SB ERR", r.status_code, r.text[:300])


def num(block, *keys):
    if not block:
        return None
    for k in keys:
        v = block.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


series_list = discover_series()
print(f"KXLOWT series found: {len(series_list)}")
print(series_list)
print(f"window: {WINDOW_START_HOUR:02d}:00 -> "
      f"{(WINDOW_START_HOUR + WINDOW_HOURS) % 24:02d}:00 UTC "
      f"({WINDOW_HOURS}h) | days back: {DAYS_BACK}\n", flush=True)

cutoff = dt.date.today() - dt.timedelta(days=DAYS_BACK)
total = 0
cities_done = 0

for series in series_list:
    city = series.replace("KXLOWT", "")
    d = kget("/trade-api/v2/events",
             {"series_ticker": series, "limit": 200, "status": "settled"})
    if not d:
        print(f"{city}: no events")
        continue

    city_rows = 0
    for ev in d.get("events", []):
        et = ev.get("event_ticker", "")
        sd = ev.get("strike_date", "")[:10]
        if not sd:
            continue
        edate = dt.date.fromisoformat(sd) - dt.timedelta(days=1)
        if edate < cutoff:
            continue

        start = int(dt.datetime.combine(
            edate, dt.time(WINDOW_START_HOUR, 0)).timestamp())
        end = start + WINDOW_HOURS * 3600

        c = kget(f"/trade-api/v2/series/{series}/events/{et}/candlesticks",
                 {"period_interval": 1, "start_ts": start, "end_ts": end})
        if not c:
            continue

        rows = []
        for mt, candles in zip(c.get("market_tickers", []),
                               c.get("market_candlesticks", [])):
            label = mt.split("-")[-1]
            for k in candles or []:
                bid = num(k.get("yes_bid"), "close_dollars", "close")
                ask = num(k.get("yes_ask"), "close_dollars", "close")
                if bid is None and ask is None:
                    continue
                rows.append({
                    "city": city,
                    "event_date": edate.isoformat(),
                    "event_ticker": et,
                    "market_ticker": mt,
                    "bracket_label": label,
                    "ts_utc": dt.datetime.utcfromtimestamp(
                        k["end_period_ts"]).isoformat() + "+00:00",
                    "yes_bid": bid,
                    "yes_ask": ask,
                    "volume": num(k, "volume_fp", "volume"),
                    "open_interest": num(k, "open_interest_fp", "open_interest"),
                })

        for i in range(0, len(rows), 500):
            push(rows[i:i + 500])
        total += len(rows)
        city_rows += len(rows)
        time.sleep(0.3)

    cities_done += 1
    print(f"{city}: {city_rows} rows", flush=True)

print(f"\nTOTAL ROWS: {total} across {cities_done} cities")
print("\nNext: run kalshi_lows_settlements.py so there is something to score")
print("these prices against, then the hour x price-band analysis can run.")
