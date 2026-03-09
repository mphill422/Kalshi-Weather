# Kalshi Temperature Model v15
# Outlier filtering + solar heating adjustment + weighted forecast consensus

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v15", layout="wide")
st.title("Kalshi Temperature Model v15")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","bias":0.5},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","bias":0.3},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","bias":-0.6},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","bias":0.4},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","bias":0.2},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","bias":0.3},
    "Miami": {"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","bias":0.1},
}

BASE_WEIGHTS = {
    "OpenMeteo":0.4,
    "GFS":0.3,
    "NWS":0.3
}

OUTLIER_HALF = 3.0
OUTLIER_REMOVE = 4.5

def safe_get(url,params=None):
    try:
        r=requests.get(url,params=params,timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def median(vals):
    s=sorted(vals)
    return s[len(s)//2]

def compute_weights(forecasts):
    vals=[v for v in forecasts.values() if v is not None]
    med=median(vals)
    adj={}
    for k,v in forecasts.items():
        if v is None:
            adj[k]=0
            continue
        d=abs(v-med)
        w=BASE_WEIGHTS[k]
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
        w=weights[k]
        num+=v*w
        den+=w
    if den==0: return None
    return num/den

def solar_factor(cloud):
    if cloud is None: return 1
    if cloud < 10: return 1.2
    if cloud < 30: return 1.1
    if cloud < 50: return 1.0
    return 0.85

def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def bracket_probs(mu):
    c=round(mu)
    labels=[
        f"{c-5} or below",
        f"{c-4} to {c-3}",
        f"{c-2} to {c-1}",
        f"{c} to {c+1}",
        f"{c+2} to {c+3}",
        f"{c+4} or above"
    ]
    sigma=1.3
    rows=[]
    for lab in labels:
        nums=[int(x) for x in re.findall(r"\d+",lab)]
        if "below" in lab:
            p=normal_cdf(nums[0]+.5,mu,sigma)
        elif "above" in lab:
            p=1-normal_cdf(nums[0]-.5,mu,sigma)
        else:
            lo,hi=nums
            p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True)
    return rows

city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]

openmeteo = safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "current":"cloud_cover",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

gfs = safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "models":"gfs_seamless",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

nws = safe_get(f"https://api.weather.gov/points/{lat},{lon}")

nws_high=None
if nws:
    fc=safe_get(nws["properties"]["forecast"])
    if fc:
        for p in fc["properties"]["periods"]:
            if p["isDaytime"]:
                nws_high=p["temperature"]
                break

forecasts={
    "OpenMeteo": openmeteo["daily"]["temperature_2m_max"][0] if openmeteo else None,
    "GFS": gfs["daily"]["temperature_2m_max"][0] if gfs else None,
    "NWS": nws_high
}

weights=compute_weights(forecasts)
cons=consensus(forecasts,weights)

cloud=openmeteo["current"]["cloud_cover"] if openmeteo else None

if cons:
    cons+=CITIES[city]["bias"]
    cons*=solar_factor(cloud)

st.subheader("Forecasts")
st.write(forecasts)

st.subheader("Weights")
st.write(weights)

st.subheader("Consensus Temperature")
st.write(round(cons,2) if cons else "N/A")

if cons:
    rows=bracket_probs(cons)
    st.subheader("Kalshi Bracket Probabilities")
    df=pd.DataFrame(rows,columns=["Bracket","Win Probability"])
    df["Win Probability"]=df["Win Probability"].apply(lambda x:f"{x*100:.1f}%")
    st.dataframe(df)
