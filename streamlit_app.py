# Kalshi Temperature Model v21
# Improvements
# - Editable Kalshi ladder input (paste directly from market)
# - Market odds input
# - Automatic implied probability conversion
# - Expected Value (EV) comparison vs model
# - Spread safety valve retained
# - Forecast consensus unchanged

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v21", layout="wide")
st.title("Kalshi Temperature Model v21")

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

BASE_WEIGHTS = {
    "ICON":0.35,
    "OpenMeteo":0.30,
    "GFS":0.20,
    "NWS":0.15
}

OUTLIER_HALF = 3
OUTLIER_REMOVE = 4.5
SPREAD_SAFETY_THRESHOLD = 5
SPREAD_SAFETY_MULTIPLIER = 0.4

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
    if den==0:
        return None
    return num/den

def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def parse_ladder(text):
    brackets=[]
    parts=[p.strip() for p in text.split("|")]
    for p in parts:
        nums=[int(x) for x in re.findall(r"\d+",p)]
        if "below" in p:
            brackets.append((p,None,nums[0]))
        elif "above" in p:
            brackets.append((p,nums[0],None))
        else:
            brackets.append((p,nums[0],nums[1]))
    return brackets

def bracket_probs(mu,sigma,brackets):
    rows=[]
    for lab,lo,hi in brackets:
        if lo is None:
            p=normal_cdf(hi+.5,mu,sigma)
        elif hi is None:
            p=1-normal_cdf(lo-.5,mu,sigma)
        else:
            p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True)
    return rows

def american_to_prob(odds):
    if odds is None:
        return None
    if odds<0:
        return abs(odds)/(abs(odds)+100)
    return 100/(odds+100)

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]
tz=profile["tz"]

ladder_text=st.text_input(
    "Paste Kalshi ladder",
    "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above"
)

brackets=parse_ladder(ladder_text)

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

nws=safe_get(f"https://api.weather.gov/points/{lat},{lon}")

nws_high=None
if nws:
    fc=safe_get(nws["properties"]["forecast"])
    if fc:
        for p in fc["properties"]["periods"]:
            if p["isDaytime"]:
                nws_high=p["temperature"]
                break

forecasts={
    "ICON":icon["daily"]["temperature_2m_max"][0] if icon else None,
    "OpenMeteo":openmeteo["daily"]["temperature_2m_max"][0] if openmeteo else None,
    "GFS":gfs["daily"]["temperature_2m_max"][0] if gfs else None,
    "NWS":nws_high
}

weights=compute_weights(forecasts)
cons=consensus(forecasts,weights)

if cons:
    cons+=profile["bias"]

vals=[v for v in forecasts.values() if v is not None]
spread=(max(vals)-min(vals)) if len(vals)>=2 else 0
sigma=1.3+(spread*.25)

st.subheader("Forecast Sources")
st.write(forecasts)

st.subheader("Consensus High")
st.write(round(cons,2) if cons else "N/A")

rows=bracket_probs(cons,sigma,brackets) if cons else []

df=pd.DataFrame(rows,columns=["Bracket","Model Probability"])

st.subheader("Model Probabilities")
st.dataframe(df)

st.subheader("Market Odds Input")
odds={}
for lab,_ in zip(df["Bracket"],df["Model Probability"]):
    odds[lab]=st.number_input(f"{lab} odds",value=0)

market_probs=[american_to_prob(o) if o!=0 else None for o in odds.values()]

df["Market Probability"]=market_probs

if market_probs:
    df["Edge"]=df["Model Probability"]-df["Market Probability"]

st.subheader("Edge vs Market")
st.dataframe(df)

st.caption("Model v21 â ladder paste + EV calculation")
