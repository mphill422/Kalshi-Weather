import os, time, base64, requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KEY_ID = os.environ["KALSHI_API_KEY_ID"]
PEM = os.environ["KALSHI_PRIVATE_KEY"]
BASE = "https://api.elections.kalshi.com"

pk = serialization.load_pem_private_key(PEM.encode(), password=None)

def sign(ts, method, path):
    msg = f"{ts}{method}{path}".encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()

def get(path, params=None):
    ts = str(int(time.time() * 1000))
    h = {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sign(ts, "GET", path),
    }
    r = requests.get(BASE + path, headers=h, params=params)
    print("STATUS:", r.status_code)
    return r

START = 1787666400
END = 1787688000

r = get("/trade-api/v2/series/KXHIGHNY/events/KXHIGHNY-26AUG25/candlesticks",
        {"period_interval": 60, "start_ts": START, "end_ts": END})

d = r.json()

# dump one raw candle so we see every field name
first = d["market_candlesticks"][0]
if first:
    import json
    print("\nRAW SAMPLE:")
    print(json.dumps(first[0], indent=2))

for tick, candles in zip(d["market_tickers"], d["market_candlesticks"]):
    print(f"\n{tick}")
    for c in candles:
        b = c.get("yes_bid", {}) or {}
        a = c.get("yes_ask", {}) or {}
        print("  ts", c.get("end_period_ts"),
              "bid", b.get("close_dollars") or b.get("close"),
              "ask", a.get("close_dollars") or a.get("close"))
