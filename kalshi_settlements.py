import os, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_SERVICE_KEY"]
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
        ts = str(int(time.time()*1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID,
             "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path)}
        r = requests.get(BASE+path, headers=h, params=params, timeout=30)
        if r.status_code == 200: return r.json()
        if r.status_code in (429,500,502,503):
            time.sleep(2*(a+1)); continue
        print("ERR", r.status_code, path, r.text[:200]); return None
    return None

def push(rows):
    if not rows: return
    r = requests.post(f"{SB_URL}/rest/v1/kalshi_settlements",
        headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}",
                 "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates"},
        json=rows, timeout=60)
    if r.status_code >= 300:
        print("SB ERR", r.status_code, r.text[:200])

series = []
cursor = None
while True:
    p = {"limit": 200, "category": "Climate and Weather"}
    if cursor: p["cursor"] = cursor
    d = kget("/trade-api/v2/series", p)
    if not d: break
    series += [s["ticker"] for s in d.get("series", [])
               if s.get("ticker","").startswith("KXHIGH")]
    cursor = d.get("cursor")
    if not cursor: break
series = sorted(set(series))
print("series:", len(series))

cutoff = dt.date.today() - dt.timedelta(days=DAYS_BACK)
sample_printed = False
total = 0

for s in series:
    city = s.replace("KXHIGH", "")
    d = kget("/trade-api/v2/events",
             {"series_ticker": s, "limit": 200, "status": "settled"})
    if not d: continue

    for ev in d.get("events", []):
        et = ev.get("event_ticker","")
        sd = ev.get("strike_date","")[:10]
        if not sd: continue
        edate = dt.date.fromisoformat(sd) - dt.timedelta(days=1)
        if edate < cutoff: continue

        m = kget("/trade-api/v2/markets",
                 {"event_ticker": et, "limit": 200})
        if not m: continue

        rows = []
        for mk in m.get("markets", []):
            if not sample_printed:
                import json
                print("SAMPLE MARKET:", json.dumps(mk, indent=2)[:1500])
                sample_printed = True
            rows.append({
                "city": city,
                "event_date": edate.isoformat(),
                "market_ticker": mk.get("ticker"),
                "bracket_label": (mk.get("ticker") or "").split("-")[-1],
                "result": mk.get("result"),
                "settled_ts": mk.get("settlement_time") or mk.get("close_time"),
            })
        push(rows)
        total += len(rows)
        time.sleep(0.2)
    print(f"{city} done")

print("TOTAL:", total)
