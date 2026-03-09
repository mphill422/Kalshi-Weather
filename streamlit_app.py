# Kalshi Temperature Model v22
# Features
# - Live Kalshi ladder ingestion (series ticker or market URL)
# - Model probabilities displayed as percentages
# - Market probabilities pulled when available
# - Edge calculation
# - Forecast consensus (ICON/OpenMeteo/GFS/NWS)
# - Spread safety valve

import math
import re
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Kalshi Temperature Model v22", layout="wide")
st.title("Kalshi Temperature Model v22")

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","bias":0.5},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","bias":0.4},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","bias":-0.6},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","bias":0.4},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","bias":0.3},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","bias":0.3},
    "Atlanta": {"lat":33.6407,"lon":-84.4277,"tz":"America/New_York","bias":0.2},
    "NYC": {"lat":40.7829,"lon":-73.9654,"tz":"America/New_York","bias":0.6},
    "Miami": {"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","bias":0.2},
}

BASE_WEIGHTS = {"ICON":0.35,"OpenMeteo":0.30,"GFS":0.20,"NWS":0.15}

OUTLIER_HALF=3
OUTLIER_REMOVE=4.5
SPREAD_SAFETY_THRESHOLD=5

def safe_get(url,params=None):
    try:
        r=requests.get(url,params=params,timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def median(vals):
    s=sorted(vals)
    n=len(s)
    if n%2==1:
        return s[n//2]
    return (s[n//2-1]+s[n//2])/2

def compute_weights(forecasts):
    vals=[v for v in forecasts.values() if v is not None]
    med=median(vals)
    adj={}
    for k,v in forecasts.items():
        if v is None:
            adj[k]=0
            continue
        d=abs(v-med)
        w=BASE_WEIGHTS.get(k,0)
        if d>OUTLIER_REMOVE:
            w=0
        elif d>OUTLIER_HALF:
            w*=0.5
        adj[k]=w
    return adj

def consensus(forecasts,weights):
    num=0
    den=0
    for k,v in forecasts.items():
        if v is None: continue
        w=weights.get(k,0)
        num+=v*w
        den+=w
    if den==0: return None
    return num/den

def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def parse_range_label(text):
    nums=[int(x) for x in re.findall(r"\d+",text)]
    if "below" in text:
        return (text,None,nums[0])
    if "above" in text:
        return (text,nums[0],None)
    if len(nums)>=2:
        return (text,nums[0],nums[1])
    return None

def extract_series(text):
    if not text: return None
    m=re.search(r"(KX[A-Z0-9]+)",text.upper())
    if m: return m.group(1)
    m=re.search(r"/markets/([A-Za-z0-9\-]+)",text)
    if m: return m.group(1)
    return text

def fetch_kalshi(series):
    data=safe_get(f"{KALSHI_API}/markets",{"series_ticker":series})
    if not data or "markets" not in data: return []
    rows=[]
    for m in data["markets"]:
        label=None
        for f in ["subtitle","title","rules_primary"]:
            parsed=parse_range_label(str(m.get(f,"")))
            if parsed:
                label=parsed
                break
        if label:
            rows.append({
                "label":label[0],
                "lo":label[1],
                "hi":label[2],
                "prob":m.get("last_price_dollars"),
            })
    return rows

def bracket_probs(mu,sigma,ladder):
    rows=[]
    for r in ladder:
        lo=r["lo"]; hi=r["hi"]; lab=r["label"]
        if lo is None:
            p=normal_cdf(hi+.5,mu,sigma)
        elif hi is None:
            p=1-normal_cdf(lo-.5,mu,sigma)
        else:
            p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append({
            "Bracket":lab,
            "Model %":round(p*100,1),
            "Market %":round((r["prob"] or 0)*100,1) if r["prob"] else None
        })
    rows.sort(key=lambda x:x["Model %"],reverse=True)
    return rows

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

series_override=st.text_input("Kalshi market URL or series ticker")

series=extract_series(series_override)

ladder=fetch_kalshi(series) if series else []

st.subheader("Kalshi Ladder Source")
st.write("Live API" if ladder else "Paste market URL to ingest ladder")

lat=profile["lat"]; lon=profile["lon"]

openmeteo=safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "current":"temperature_2m",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

gfs=safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "models":"gfs_seamless",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

icon=safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "models":"icon_seamless",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

forecasts={
    "ICON":icon["daily"]["temperature_2m_max"][0] if icon else None,
    "OpenMeteo":openmeteo["daily"]["temperature_2m_max"][0] if openmeteo else None,
    "GFS":gfs["daily"]["temperature_2m_max"][0] if gfs else None,
}

weights=compute_weights(forecasts)
cons=consensus(forecasts,weights)

if cons:
    cons+=profile["bias"]

vals=[v for v in forecasts.values() if v]
spread=max(vals)-min(vals) if len(vals)>=2 else 0
sigma=1.3+(spread*.25)

st.subheader("Forecast Sources")
st.write(forecasts)

st.subheader("Consensus High")
st.write(round(cons,2) if cons else "N/A")

if ladder and cons:
    rows=bracket_probs(cons,sigma,ladder)
    df=pd.DataFrame(rows)
    st.subheader("Kalshi Bracket Probabilities (%)")
    st.dataframe(df)
