import os, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
BASE = "https://api.elections.kalshi.com"
CATEGORY = os.environ.get("CATEGORY", "Economics")
MAX_SERIES = int(os.environ.get("MAX_SERIES", "400"))

pk = serialization.load_pem_private_key(PEM.encode(), password=None)

def sign(ts, m, p):
    return base64.b64encode(pk.sign(f"{ts}{m}{p}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()

def kget(path, params=None):
    for a in range(3):
        ts = str(int(time.time()*1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID,
             "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path)}
        r = requests.get(BASE+path, headers=h, params=params, timeout=20)
        if r.status_code == 200: return r.json()
        time.sleep(1)
    return None

series, cursor = [], None
while True:
    p = {"limit": 200, "category": CATEGORY}
    if cursor: p["cursor"] = cursor
    d = kget("/trade-api/v2/series", p)
    if not d: break
    series += d.get("series", [])
    cursor = d.get("cursor")
    if not cursor or len(series) >= MAX_SERIES: break

series = series[:MAX_SERIES]
print(f"category: {CATEGORY}", flush=True)
print(f"scanning {len(series)} series\n", flush=True)
print(f"{'evts/30d':>9}  {'ticker':<22} title", flush=True)

cutoff = dt.date.today() - dt.timedelta(days=30)
found = 0

for i, s in enumerate(series):
    tk = s.get("ticker", "")
    d = kget("/trade-api/v2/events",
             {"series_ticker": tk, "limit": 200, "status": "settled"})
    if d:
        recent = 0
        for ev in d.get("events", []):
            sd = ev.get("strike_date", "")[:10]
            if sd:
                try:
                    if dt.date.fromisoformat(sd) >= cutoff:
                        recent += 1
                except ValueError:
                    pass
        if recent >= 5:
            found += 1
            print(f"{recent:>9}  {tk:<22} {s.get('title','')[:70]}", flush=True)
    if (i+1) % 25 == 0:
        print(f"   ... {i+1}/{len(series)} scanned, {found} hits", flush=True)
    time.sleep(0.1)

print(f"\ndone. {found} qualifying series.", flush=True)
