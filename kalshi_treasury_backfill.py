"""
kalshi_treasury_backfill.py — minute-resolution UST daily-yield candles
                              → kalshi_treasury_candles

WHAT THIS IS FOR — and how it differs from everything before it
----------------------------------------------------------------
Every model in this project so far tried to PREDICT better than the market:
tennis, MLB, and the weather bracket model. All three failed the same way —
when the model disagreed with the market, the market was right 75-82% of the
time.

This is not that. The five UST daily-yield series price five points on the SAME
yield curve:

    KXUST2AD    2 Year Daily     ~$18,024 vol
    KXUST5AD    5 Year Daily     ~$12,265 vol
    KXUST7AD    7 Year Daily     ~$18,085 vol
    KXUST10AD   10 Year Daily    ~$28,571 vol
    KXUST30AD   30 Year Daily    ~$24,771 vol
    (volumes observed 2026-08-31; ticker names are a GUESS — see below)

Those five instruments are mechanically linked. They cannot move independently.
So if the 5Y ladder implies one thing about today's rate move and the 7Y ladder
implies something incompatible, ONE OF THEM IS WRONG — and identifying which
requires no view on rates at all. The market contradicts itself; you take the
side of its own consensus.

That is a structurally different bet from forecasting, and it is the reason to
try this rather than USD/JPY.

⚠️ THE HONEST CASE AGAINST
--------------------------
Treasuries are the single most likely Kalshi category to have professionals in
it, and curve consistency is the FIRST thing a rates desk checks. The
inconsistency may simply not exist. If so this backfill answers that in one day,
which is the point — it is cheap to falsify.

⚠️ DO NOT IMPORT THE WEATHER BAND
----------------------------------
The 58-79c band came from where the weather market's favorite happens to be
mispriced across 25 summer days. It is NOT a law about prediction markets and
there is no reason to expect it here. What transfers is the METHOD — pull
minute history, join to settlement, cut by price and hour, find cells where win
rate exceeds price — not the numbers. The answer here might be the NO side, or
the 40s, or nothing at all.

⚠️ THESE ARE "OR ABOVE" LADDERS, NOT EXCLUSIVE RANGES
------------------------------------------------------
Observed on the exchange: "4.47% or above" at 60%, "4.49% or above" at 14%.
That is CUMULATIVE, unlike the weather brackets where exactly one of six settles
yes. Consequences:
  - MULTIPLE markets settle yes per event. The `n_yes != 1` sanity check from
    kalshi_lows_settlements.py DOES NOT APPLY and would fire constantly.
  - Probabilities must be MONOTONE DECREASING as the strike rises. A higher
    strike cannot be more likely than a lower one.
  - **That monotonicity is itself a free coherence test, WITHIN a single
    ladder, before any cross-series work.** Any violation is a guaranteed
    mispricing. Check it first — it is one query and needs no curve model.

TICKER DISCOVERY
----------------
kalshi_survey.py found KXUST10AD and KXUST5AD by scanning the Financials
category. The 2Y/7Y/30Y tickers are inferred by pattern and NOT verified. This
script does not hardcode them — it scans Financials for anything matching
KXUST*AD, prints what it finds, and backfills that. If the printed list does not
show five series, the pattern guess is wrong; read the printed tickers and
adjust SERIES_PATTERN.

WINDOW
------
These settle at a fixed clock time (the exchange showed a 10am EDT settlement
for the FX equivalents; UST settlement time is UNVERIFIED — confirm from the
market rules before trusting any hour-of-day analysis). Set wide at 12:00-22:00
UTC to cover a US session regardless. Narrow it once settlement time is known.

Note the consequence for analysis: a fixed-clock settlement compresses the
"cost of certainty" curve into a few hours rather than spreading it over a
physical event. The last hour before settlement is nearly deterministic.

CREATE THE TABLES ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.kalshi_treasury_candles (
    id BIGSERIAL PRIMARY KEY,
    tenor TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_ticker TEXT,
    market_ticker TEXT NOT NULL,
    strike_label TEXT,
    ts_utc TIMESTAMPTZ NOT NULL,
    yes_bid NUMERIC,
    yes_ask NUMERIC,
    volume NUMERIC,
    open_interest NUMERIC,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (market_ticker, ts_utc)
  );
  ALTER TABLE public.kalshi_treasury_candles ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.kalshi_treasury_candles
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_ust_candles_tenor_date
    ON public.kalshi_treasury_candles (tenor, event_date);
  CREATE INDEX IF NOT EXISTS idx_ust_candles_ts
    ON public.kalshi_treasury_candles (ts_utc);

  CREATE TABLE IF NOT EXISTS public.kalshi_treasury_settlements (
    id BIGSERIAL PRIMARY KEY,
    tenor TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_ticker TEXT,
    market_ticker TEXT NOT NULL,
    strike_label TEXT,
    strike_value NUMERIC,
    result TEXT,
    settled_ts TIMESTAMPTZ,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (market_ticker)
  );
  ALTER TABLE public.kalshi_treasury_settlements ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.kalshi_treasury_settlements
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_ust_settle_tenor_date
    ON public.kalshi_treasury_settlements (tenor, event_date);

PRICE UNITS: yes_bid / yes_ask stored in DOLLARS (0.62), same as every other
candle table here. Multiply by 100 for cents.

Secrets: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY, SUPABASE_URL, and either
SUPABASE_SERVICE_KEY or SUPABASE_KEY.
"""

import os, re, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
BASE = "https://api.elections.kalshi.com"

DAYS_BACK = int(os.environ.get("DAYS_BACK", "30"))
CATEGORY = os.environ.get("CATEGORY", "Financials")

# Matches KXUST2AD, KXUST5AD, KXUST7AD, KXUST10AD, KXUST30AD.
# If the run prints fewer than 5 series, this pattern is wrong — read the
# printed "all UST-ish series" list and widen it.
SERIES_PATTERN = re.compile(r"^KXUST\d+AD$")

WINDOW_START_HOUR = 12
WINDOW_HOURS = 10   # 12:00 -> 22:00 UTC

pk = serialization.load_pem_private_key(PEM.encode(), password=None)


def sign(ts, method, path):
    return base64.b64encode(pk.sign(
        f"{ts}{method}{path}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()


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


def push(table, rows, conflict):
    if not rows:
        return
    r = requests.post(
        f"{SB_URL}/rest/v1/{table}?on_conflict={conflict}",
        headers={"apikey": SB_KEY,
                 "Authorization": f"Bearer {SB_KEY}",
                 "Content-Type": "application/json",
                 "Prefer": "return=minimal,resolution=merge-duplicates"},
        json=rows, timeout=60)
    if r.status_code >= 300:
        print(f"  SB ERR {table}", r.status_code, r.text[:250])


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


def strike_from_label(label):
    """Pull the numeric strike out of a label like '4.47% or above'."""
    if not label:
        return None
    m = re.search(r"(\d+\.?\d*)", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def tenor_from_series(ticker):
    m = re.match(r"^KXUST(\d+)AD$", ticker)
    return (m.group(1) + "Y") if m else ticker


# ── Discover series ──────────────────────────────────────────────────────────
all_ust, matched, cursor = [], [], None
while True:
    p = {"limit": 200, "category": CATEGORY}
    if cursor:
        p["cursor"] = cursor
    d = kget("/trade-api/v2/series", p)
    if not d:
        break
    for s in d.get("series", []):
        t = s.get("ticker", "")
        if "UST" in t.upper():
            all_ust.append((t, s.get("title", "")))
        if SERIES_PATTERN.match(t):
            matched.append(t)
    cursor = d.get("cursor")
    if not cursor:
        break

matched = sorted(set(matched))

print(f"category={CATEGORY}")
print(f"\nall UST-ish series found ({len(all_ust)}):")
for t, title in sorted(set(all_ust)):
    print(f"   {t:<16} {title}")
print(f"\nmatched by pattern ({len(matched)}): {matched}")
if len(matched) < 5:
    print("\n⚠️ Fewer than 5 matched. The ticker pattern guess may be wrong —")
    print("   read the list above and adjust SERIES_PATTERN.")
print(f"\nwindow {WINDOW_START_HOUR:02d}:00 -> "
      f"{(WINDOW_START_HOUR + WINDOW_HOURS) % 24:02d}:00 UTC | "
      f"days back {DAYS_BACK}\n", flush=True)

if not matched:
    raise SystemExit("No series matched — nothing to backfill.")

cutoff = dt.date.today() - dt.timedelta(days=DAYS_BACK)
total_candles = 0
total_settle = 0
sample_shown = False

for series in matched:
    tenor = tenor_from_series(series)
    d = kget("/trade-api/v2/events",
             {"series_ticker": series, "limit": 200, "status": "settled"})
    if not d:
        print(f"{tenor}: no events")
        continue

    c_rows_total = 0
    s_rows_total = 0

    for ev in d.get("events", []):
        et = ev.get("event_ticker", "")
        sd = (ev.get("strike_date") or "")[:10]
        if not sd:
            continue
        edate = dt.date.fromisoformat(sd)
        if edate < cutoff:
            continue

        # ---- settlements (which strikes resolved yes) ----
        m = kget("/trade-api/v2/markets", {"event_ticker": et, "limit": 200})
        if m:
            srows = []
            for mk in m.get("markets", []):
                label = (mk.get("yes_sub_title") or mk.get("subtitle")
                         or mk.get("title") or "")
                if not sample_shown:
                    print(f"  SAMPLE {tenor}: ticker={mk.get('ticker')} "
                          f"label={label!r} result={mk.get('result')!r}")
                    sample_shown = True
                srows.append({
                    "tenor": tenor,
                    "event_date": edate.isoformat(),
                    "event_ticker": et,
                    "market_ticker": mk.get("ticker"),
                    "strike_label": label,
                    "strike_value": strike_from_label(label),
                    "result": mk.get("result"),
                    "settled_ts": mk.get("settlement_time") or mk.get("close_time"),
                })
            push("kalshi_treasury_settlements", srows, "market_ticker")
            s_rows_total += len(srows)
            total_settle += len(srows)

        # ---- candles ----
        start = int(dt.datetime.combine(
            edate, dt.time(WINDOW_START_HOUR, 0)).timestamp())
        end = start + WINDOW_HOURS * 3600

        c = kget(f"/trade-api/v2/series/{series}/events/{et}/candlesticks",
                 {"period_interval": 1, "start_ts": start, "end_ts": end})
        if not c:
            time.sleep(0.3)
            continue

        crows = []
        for mt, candles in zip(c.get("market_tickers", []),
                               c.get("market_candlesticks", [])):
            for k in candles or []:
                bid = num(k.get("yes_bid"), "close_dollars", "close")
                ask = num(k.get("yes_ask"), "close_dollars", "close")
                if bid is None and ask is None:
                    continue
                crows.append({
                    "tenor": tenor,
                    "event_date": edate.isoformat(),
                    "event_ticker": et,
                    "market_ticker": mt,
                    "strike_label": mt.split("-")[-1],
                    "ts_utc": dt.datetime.utcfromtimestamp(
                        k["end_period_ts"]).isoformat() + "+00:00",
                    "yes_bid": bid,
                    "yes_ask": ask,
                    "volume": num(k, "volume_fp", "volume"),
                    "open_interest": num(k, "open_interest_fp", "open_interest"),
                })

        for i in range(0, len(crows), 500):
            push("kalshi_treasury_candles", crows[i:i + 500],
                 "market_ticker,ts_utc")
        c_rows_total += len(crows)
        total_candles += len(crows)
        time.sleep(0.3)

    print(f"{tenor}: {c_rows_total} candle rows, {s_rows_total} settlement rows",
          flush=True)

print(f"\nTOTAL: {total_candles} candles, {total_settle} settlement rows")
print("""
NEXT — run the monotonicity check FIRST. It needs no curve model and no
forecast, and any violation is a guaranteed mispricing:

  -- Within one ladder at one instant, a HIGHER strike must never be
  -- priced MORE likely than a LOWER strike. Any row here is incoherent.
  select a.tenor, a.event_date, a.ts_utc,
         a.strike_label as lo_strike, a.yes_ask as lo_ask,
         b.strike_label as hi_strike, b.yes_ask as hi_ask,
         round((b.yes_ask - a.yes_ask) * 100, 1) as gap_cents
  from kalshi_treasury_candles a
  join kalshi_treasury_candles b
    on b.event_ticker = a.event_ticker
   and b.ts_utc = a.ts_utc
  join kalshi_treasury_settlements sa on sa.market_ticker = a.market_ticker
  join kalshi_treasury_settlements sb on sb.market_ticker = b.market_ticker
  where sb.strike_value > sa.strike_value
    and b.yes_ask > a.yes_ask + 0.02
    and a.yes_ask between 0.03 and 0.97
  order by gap_cents desc
  limit 100;

If that returns nothing, the ladders are internally coherent and the next
question is CROSS-tenor: do 2Y/5Y/7Y/10Y/30Y imply mutually consistent moves on
the same day? If it returns a lot, stop there — you have found a mispricing
that needs no rates view whatsoever.
""")
