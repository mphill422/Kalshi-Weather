import os, time, base64, requests, datetime as dt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
BASE = "https://api.elections.kalshi.com"
pk = serialization.load_pem_private_key(PEM.encode(), password=None)

def sign(ts, m, p):
    return base64.b64encode(pk.sign(f"{ts}{m}{p}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()

def kget(path, params=None):
    for a in range(3):
        ts = str(int(time.time()*1000))
        h = {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
             "KALSHI-ACCESS-SIGNATURE": sign(ts,"GET",path)}
        r = requests.get(BASE+path, headers=h, params=params, timeout=30)
        if r.status_code == 200: return r.json()
        time.sleep(2)
    return None

series, cursor = [], None
while True:
    p = {"limit": 200}
    if cursor: p["cursor"] = cursor
    d = kget("/trade-api/v2/series", p)
    if not d: break
    series += d.get("series", [])
    cursor = d.get("cursor")
    if not cursor: break

print(f"total series: {len(series)}\n")
cutoff = dt.date.today() - dt.timedelta(days=30)
out = []

for s in series:
    tk = s.get("ticker","")
    d = kget("/trade-api/v2/events",
             {"series_ticker": tk, "limit": 200, "status": "settled"})
    if not d: continue
    recent = 0
    for ev in d.get("events", []):
        sd = ev.get("strike_date","")[:10]
        if sd and dt.date.fromisoformat(sd) >= cutoff:
            recent += 1
    if recent >= 15:
        out.append((recent, tk, s.get("category",""), s.get("title","")[:60]))
    time.sleep(0.15)

out.sort(reverse=True)
print(f"{'evts/30d':>9}  {'ticker':<18} {'category':<22} title")
for r, tk, cat, ti in out:
    print(f"{r:>9}  {tk:<18} {cat:<22} {ti}")
