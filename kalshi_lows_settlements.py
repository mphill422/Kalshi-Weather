"""
kalshi_lows_settlements.py — KXLOWT per-bracket settlement results
                              → kalshi_lows_settlements

Companion to kalshi_settlements.py (which does KXHIGH). This is what
kalshi_lows_candles gets scored against — prices are meaningless without
knowing which bracket actually won.

WHY A SEPARATE TABLE FROM lows_settlements
-------------------------------------------
`lows_settlements` (built 2026-08-24) holds the settled MINIMUM TEMPERATURE
per city-day, sourced from wethr's daily_extremes_api and cross-checked
against Iowa State CLI. It answers "what was the low."

This table holds the per-MARKET result from Kalshi itself — which bracket
ticker resolved yes. It answers "which contract paid." Those are different
questions and the second is what you need to score a price against, because
it removes any risk of a bracket-boundary or rounding disagreement between
our arithmetic and Kalshi's settlement.

Keep both. The temperature table is the independent check on the outcome
table, the same way Iowa CLI is the independent check on wethr.

KNOWN ISSUE CARRIED OVER FROM THE HIGHS VERSION
------------------------------------------------
bracket_label is derived as ticker.split("-")[-1], which yields Kalshi's
strike code (e.g. "B85.5" or "T89"), NOT a range label like "85-86". Those do
not join to lows_snapshots.bracket_label or to the model's bracket format
without a mapping.

The mapping was worked out on the highs side on 2026-08-27 and holds here:
    B{n}.5  ->  the range {n}-{n+1}      e.g. B85.5 = "85-86"
    T{n}    ->  a tail market            e.g. T85 = "84 or below", T92 = "93 or above"
Confirmed by lining up a full NY ladder against kalshi_snapshots on the same
event: B85.5 settled yes at 53c while snapshots showed "85-86". Despite the
"B" naming these resolve as EXCLUSIVE ranges — verified by counting yes per
event, which came back exactly 1 of 6 markets on every event checked.

That mapping is NOT applied here. This file stores the raw Kalshi label so
nothing is lost in translation; do the conversion in analysis where it can be
checked. Storing a derived value would bury an assumption in the data.

ALSO NOTE: this script writes with resolution=merge-duplicates but the table
below has no UNIQUE constraint by default — add one, or re-running will insert
duplicate rows and any join will fan out, silently multiplying every P&L
number. The highs table happened to be clean (3,354 rows, 3,354 distinct
tickers) but that was luck.

CREATE THE TABLE ONCE (Supabase SQL editor):

  CREATE TABLE IF NOT EXISTS public.kalshi_lows_settlements (
    id BIGSERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    event_date DATE NOT NULL,
    market_ticker TEXT NOT NULL,
    bracket_label TEXT,
    result TEXT,
    settled_ts TIMESTAMPTZ,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (market_ticker)
  );
  ALTER TABLE public.kalshi_lows_settlements ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.kalshi_lows_settlements
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_lows_settle_city_date
    ON public.kalshi_lows_settlements (city, event_date);

Secrets: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY, SUPABASE_URL, and either
SUPABASE_SERVICE_KEY or SUPABASE_KEY.
"""

import os, time, base64, requests, datetime as dt, json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
BASE = "https://api.elections.kalshi.com"

DAYS_BACK = int(os.environ.get("DAYS_BACK", "30"))

pk = serialization.load_pem_private_key(PEM.encode(), password=None)


def sign(ts, method, path):
    return base64.b64encode(pk.sign(
        f"{ts}{method}{path}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()


def kget(path, params=None):
    for a in range(4):
        ts = str(int(time.time() * 1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID,
             "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path)}
        r = requests.get(BASE + path, headers=h, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (a + 1))
            continue
        print("ERR", r.status_code, path, r.text[:200])
        return None
    return None


def push(rows):
    if not rows:
        return
    r = requests.post(
        f"{SB_URL}/rest/v1/kalshi_lows_settlements?on_conflict=market_ticker",
        headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}",
                 "Content-Type": "application/json",
                 "Prefer": "return=minimal,resolution=merge-duplicates"},
        json=rows, timeout=60)
    if r.status_code >= 300:
        print("SB ERR", r.status_code, r.text[:200])


series = []
cursor = None
while True:
    p = {"limit": 200, "category": "Climate and Weather"}
    if cursor:
        p["cursor"] = cursor
    d = kget("/trade-api/v2/series", p)
    if not d:
        break
    series += [s["ticker"] for s in d.get("series", [])
               if s.get("ticker", "").startswith("KXLOWT")]
    cursor = d.get("cursor")
    if not cursor:
        break
series = sorted(set(series))
print("KXLOWT series:", len(series), flush=True)

cutoff = dt.date.today() - dt.timedelta(days=DAYS_BACK)
sample_printed = False
total = 0
yes_total = 0

for s in series:
    city = s.replace("KXLOWT", "")
    d = kget("/trade-api/v2/events",
             {"series_ticker": s, "limit": 200, "status": "settled"})
    if not d:
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

        m = kget("/trade-api/v2/markets",
                 {"event_ticker": et, "limit": 200})
        if not m:
            continue

        rows = []
        n_yes = 0
        for mk in m.get("markets", []):
            if not sample_printed:
                print("SAMPLE MARKET:", json.dumps(mk, indent=2)[:1500])
                sample_printed = True
            res = mk.get("result")
            if res == "yes":
                n_yes += 1
            rows.append({
                "city": city,
                "event_date": edate.isoformat(),
                "market_ticker": mk.get("ticker"),
                "bracket_label": (mk.get("ticker") or "").split("-")[-1],
                "result": res,
                "settled_ts": mk.get("settlement_time") or mk.get("close_time"),
            })

        # Sanity check: an exclusive ladder settles exactly one market yes.
        # More than one, or zero on a settled event, means either the market
        # structure is not what we think or the event did not actually resolve.
        # Print it rather than swallowing it — this is the class of thing that
        # silently poisons every downstream number.
        if rows and n_yes != 1:
            print(f"    ⚠️ {city} {edate}: {n_yes} markets settled 'yes' "
                  f"out of {len(rows)} (expected exactly 1)")

        push(rows)
        total += len(rows)
        city_rows += len(rows)
        yes_total += n_yes
        time.sleep(0.2)

    print(f"{city}: {city_rows} rows", flush=True)

print(f"\nTOTAL: {total} market rows, {yes_total} settled yes")
if total:
    print(f"(expect roughly {total // 6} yes rows if every ladder has 6 brackets)")
print("\nBefore analyzing, verify no duplicate fan-out:")
print("  select count(*), count(distinct market_ticker) from kalshi_lows_settlements;")
print("Those two numbers must match.")
