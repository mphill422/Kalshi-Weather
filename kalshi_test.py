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

r = get("/trade-api/v2/events",
        {"series_ticker": "KXHIGHNY", "limit": 10, "status": "settled"})
print(r.text[:3000])
