import os, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BASE = "https://api.elections.kalshi.com"

DAYS_BACK = int(os.environ.get("DAYS_BACK", "25"))

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
            time.sleep(2 * (attempt + 1)); continue
        print("ERR", r.status_code, path, r.text[:200])
        return None
    return None

def discover_series():
    out, cursor = [], None
    while True:
        p = {"limit": 200, "category": "Climate and Weather"}
        if cursor: p["cursor"] = cursor
        d = kget("/trade-api/v2/series", p)
        if not d: break
        for s in d.get("series", []):
            t = s.get("ticker", "")
            if t.startswith("KXHIGH"):
                out.append(t)
        cursor = d.get("cursor")
        if not cursor: break
    return sorted(set(out))

def push(rows):
    if not rows: return
    r = requests.post(
        f"{SB_URL}/rest/v1/kalshi_candles",
        headers={"apikey": SB_KEY,
                 "Authorization": f"Bearer {SB_KEY}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates"},
        json=rows, timeout=60)
    if r.status_code >= 300:
        print("SB ERR", r.status_code, r.text[:300])

def num(block, *keys):
    if not block: return None
    for k in keys:
        v = block.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    return None

series_list = discover_series()
print("series found:", len(series_list), series_list)

cutoff = dt.date.today() - dt.timedelta(days=DAYS_BACK)
total = 0

for series in series_list:
    city = series.replace("KXHIGH", "")
    d = kget("/trade-api/v2/events",
             {"series_ticker": series, "limit": 200, "status": "settled"})
    if not d: continue

    for ev in d.get("events", []):
        et = ev.get("event_ticker", "")
        sd = ev.get("strike_date", "")[:10]
        if not sd: continue
        edate = dt.date.fromisoformat(sd) - dt.timedelta(days=1)
        if edate < cutoff: continue

        start = int(dt.datetime.combine(edate, dt.time(12, 0)).timestamp())
        end = start + 22 * 3600

        c = kget(f"/trade-api/v2/series/{series}/events/{et}/candlesticks",
                 {"period_interval": 1, "start_ts": start, "end_ts": end})
        if not c: continue

        rows = []
        for mt, candles in zip(c.get("market_tickers", []),
                               c.get("market_candlesticks", [])):
            label = mt.split("-")[-1]
            for k in candles or []:
                bid = num(k.get("yes_bid"), "close_dollars", "close")
                ask = num(k.get("yes_ask"), "close_dollars", "close")
                if bid is None and ask is None: continue
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
            push(rows[i:i+500])
        total += len(rows)
        print(f"{city} {edate} -> {len(rows)} rows")
        time.sleep(0.3)

print("TOTAL ROWS:", total)
