# Kalshi Temperature Model v13.1 (FULL FIXED VERSION)
# Fix: base_sigma -> sigma
# Includes:
# - All cities
# - NWS forecast + NWS hourly
# - Open-Meteo + GFS
# - MET Norway
# - Weighted consensus
# - Spread calculation
# - Sigma calculation
# - Source diagnostics table

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v13.1", layout="wide")
st.title("Kalshi Temperature Model v13.1")

# -----------------------------
# Cities
# -----------------------------

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","station":"KPHX","sigma":1.10,"bias":0.50},
    "Las Vegas":{"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","station":"KLAS","sigma":1.00,"bias":0.30},
    "Los Angeles":{"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","station":"KLAX","sigma":1.15,"bias":-0.60},
    "Dallas":{"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","station":"KDFW","sigma":1.00,"bias":0.40},
    "Austin":{"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","station":"KAUS","sigma":1.10,"bias":0.20},
    "Houston":{"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","station":"KIAH","sigma":1.50,"bias":0.30},
    "Miami":{"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","station":"KMIA","sigma":1.40,"bias":0.10},
    "New Orleans":{"lat":29.9934,"lon":-90.2580,"tz":"America/Chicago","station":"KMSY","sigma":1.40,"bias":0.20},
    "Atlanta":{"lat":33.6407,"lon":-84.4277,"tz":"America/New_York","station":"KATL","sigma":1.30,"bias":0.20},
    "New York":{"lat":40.6413,"lon":-73.7781,"tz":"America/New_York","station":"KJFK","sigma":1.20,"bias":-0.10},
}

# -----------------------------
# Source weights
# -----------------------------

SOURCE_WEIGHTS = {
    "NWS forecast":0.35,
    "NWS hourly":0.25,
    "MET Norway":0.20,
    "Open-Meteo":0.10,
    "GFS":0.10,
}

# -----------------------------
# Utility
# -----------------------------

def safe_get_json(url,params=None,headers=None):
    try:
        h={"User-Agent":"kalshi-weather-model"}
        if headers:
            h.update(headers)
        r=requests.get(url,params=params,headers=h,timeout=12)
        r.raise_for_status()
        return r.json()
    except:
        return None

# -----------------------------
# Forecast sources
# -----------------------------

def fetch_nws(lat,lon):

    points=safe_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not points:
        return None,None

    forecast_url=points["properties"]["forecast"]
    hourly_url=points["properties"]["forecastHourly"]

    forecast=safe_get_json(forecast_url)
    hourly=safe_get_json(hourly_url)

    daily=None
    hourly_high=None

    if forecast:
        try:
            daily=forecast["properties"]["periods"][0]["temperature"]
        except:
            pass

    if hourly:
        try:
            temps=[p["temperature"] for p in hourly["properties"]["periods"][:24]]
            hourly_high=max(temps)
        except:
            pass

    return daily,hourly_high


def fetch_open_meteo(lat,lon,model=None):

    params={
        "latitude":lat,
        "longitude":lon,
        "daily":"temperature_2m_max",
        "temperature_unit":"fahrenheit",
        "timezone":"auto"
    }

    if model:
        params["models"]=model

    data=safe_get_json("https://api.open-meteo.com/v1/forecast",params)

    if not data:
        return None

    try:
        return float(data["daily"]["temperature_2m_max"][0])
    except:
        return None


def fetch_metno(lat,lon):

    data=safe_get_json(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        {"lat":lat,"lon":lon},
        {"User-Agent":"kalshi-weather-model"}
    )

    if not data:
        return None

    try:
        vals=[]
        for t in data["properties"]["timeseries"][:24]:
            c=t["data"]["instant"]["details"]["air_temperature"]
            f=c*9/5+32
            vals.append(f)

        return max(vals)

    except:
        return None


# -----------------------------
# Weighted consensus
# -----------------------------

def weighted_consensus(source_values,bias):

    numer=0
    denom=0

    for name,val in source_values.items():
        if val is None:
            continue

        w=SOURCE_WEIGHTS.get(name,0)

        numer+=w*val
        denom+=w

    if denom==0:
        return None

    return numer/denom + bias


# -----------------------------
# UI
# -----------------------------

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]

# -----------------------------
# Fetch data
# -----------------------------

nws_forecast,nws_hourly=fetch_nws(lat,lon)
open_meteo=fetch_open_meteo(lat,lon)
gfs=fetch_open_meteo(lat,lon,"gfs_seamless")
metno=fetch_metno(lat,lon)

sources={
"NWS forecast":nws_forecast,
"NWS hourly":nws_hourly,
"Open-Meteo":open_meteo,
"GFS":gfs,
"MET Norway":metno,
}

# -----------------------------
# Diagnostics table
# -----------------------------

rows=[]

for s,v in sources.items():
    rows.append({"Source":s,"Forecast High":v})

df=pd.DataFrame(rows)

st.subheader("Forecast Source Diagnostics")
st.dataframe(df,use_container_width=True)

# -----------------------------
# Model output
# -----------------------------

usable=[v for v in sources.values() if v is not None]

if not usable:
    st.warning("No weather sources returned data.")
    st.stop()

spread=max(usable)-min(usable)

consensus=weighted_consensus(sources,profile["bias"])

sigma=max(0.85,float(profile["sigma"])+spread*0.22)

# -----------------------------
# Display results
# -----------------------------

st.subheader("Model Output")

c1,c2,c3=st.columns(3)

c1.metric("Consensus High",round(consensus,1))
c2.metric("Forecast Spread",round(spread,1))
c3.metric("Sigma",round(sigma,2))

