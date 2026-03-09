# Kalshi Temperature Model v17
# Improvements
# - HRRR temperature integration
# - Correct median calculation
# - Additive solar heating adjustment
# - Dynamic probability spread
# - Improved outlier filtering
# - Afternoon airport warming adjustment

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v17", layout="wide")
st.title("Kalshi Temperature Model v17")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","station":"KPHX","bias":0.5},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","station":"KLAS","bias":0.4},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","station":"KLAX","bias":-0.6},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","station":"KDFW","bias":0.4},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","station":"KAUS","bias":0.3},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","station":"KIAH","bias":0.3},
}

BASE_WEIGHTS = {
    "HRRR":0.35,
    "OpenMeteo":0.30,
    "GFS":0.20,
    "NWS":0.15
}

OUTLIER_HALF = 3
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
    n=len(s)
    if n%2:
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

        if v is None:
            continue

        w=weights.get(k,0)

        num+=v*w
        den+=w

    if den==0:
        return None

    return num/den

def solar_adjust(cloud,hour):

    if hour<9 or hour>17:
        return 0

    if cloud is None:
        return 0

    if cloud<10:
        return 1.0
    if cloud<30:
        return 0.6
    if cloud<50:
        return 0.2

    return -0.5

def normal_cdf(x,mu,sigma):

    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def bracket_probs(mu,sigma):

    c=round(mu)

    labels=[
        f"{c-5} or below",
        f"{c-4} to {c-3}",
        f"{c-2} to {c-1}",
        f"{c} to {c+1}",
        f"{c+2} to {c+3}",
        f"{c+4} or above"
    ]

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

city=st.selectbox("City", list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]
tz=profile["tz"]

local_hour=datetime.now(ZoneInfo(tz)).hour

# OpenMeteo
openmeteo=safe_get(
"https://api.open-meteo.com/v1/forecast",
{
"latitude":lat,
"longitude":lon,
"daily":"temperature_2m_max",
"current":"temperature_2m,cloud_cover",
"temperature_unit":"fahrenheit",
"timezone":"auto"
})

# GFS
gfs=safe_get(
"https://api.open-meteo.com/v1/forecast",
{
"latitude":lat,
"longitude":lon,
"daily":"temperature_2m_max",
"models":"gfs_seamless",
"temperature_unit":"fahrenheit",
"timezone":"auto"
})

# HRRR
hrrr=safe_get(
"https://api.open-meteo.com/v1/forecast",
{
"latitude":lat,
"longitude":lon,
"daily":"temperature_2m_max",
"models":"hrrr",
"temperature_unit":"fahrenheit",
"timezone":"auto"
})

# NWS
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

"HRRR":hrrr["daily"]["temperature_2m_max"][0] if hrrr else None,
"OpenMeteo":openmeteo["daily"]["temperature_2m_max"][0] if openmeteo else None,
"GFS":gfs["daily"]["temperature_2m_max"][0] if gfs else None,
"NWS":nws_high
}

weights=compute_weights(forecasts)

cons=consensus(forecasts,weights)

cloud=None
current_temp=None

if openmeteo:

    cloud=openmeteo["current"].get("cloud_cover")
    current_temp=openmeteo["current"].get("temperature_2m")

if cons:

    cons+=profile["bias"]

    cons+=solar_adjust(cloud,local_hour)

    if local_hour>=13:
        cons+=0.3

vals=[v for v in forecasts.values() if v is not None]

spread=max(vals)-min(vals)

sigma=1.3+(spread*0.25)

st.subheader("Forecast Sources")
st.write(forecasts)

st.subheader("Weights")
st.write(weights)

st.subheader("Consensus High")
st.write(round(cons,2) if cons else "N/A")

st.subheader("Current Station Temp")
st.write(current_temp)

if cons:

    rows=bracket_probs(cons,sigma)

    st.subheader("Kalshi Bracket Probabilities")

    df=pd.DataFrame(rows,columns=["Bracket","Win Probability"])

    df["Win Probability"]=df["Win Probability"].apply(lambda x:f"{x*100:.1f}%")

    st.dataframe(df)

st.caption("Model v17 — HRRR + dynamic forecast confidence + station-of-record monitoring")
