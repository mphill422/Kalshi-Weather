# Kalshi Temperature Model v14
# Complete Streamlit version

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v14", layout="wide")
st.title("Kalshi Temperature Model v14")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","sigma":1.10,"bias":0.50},
    "Las Vegas":{"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","sigma":1.00,"bias":0.30},
    "Los Angeles":{"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","sigma":1.15,"bias":-0.60},
    "Dallas":{"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","sigma":1.00,"bias":0.40},
    "Austin":{"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","sigma":1.10,"bias":0.20},
    "Houston":{"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","sigma":1.50,"bias":0.30},
    "Miami":{"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","sigma":1.40,"bias":0.10},
    "New Orleans":{"lat":29.9934,"lon":-90.2580,"tz":"America/Chicago","sigma":1.40,"bias":0.20},
    "Atlanta":{"lat":33.6407,"lon":-84.4277,"tz":"America/New_York","sigma":1.30,"bias":0.20},
    "New York":{"lat":40.6413,"lon":-73.7781,"tz":"America/New_York","sigma":1.20,"bias":-0.10},
}

SOURCE_WEIGHTS = {
"NWS forecast":0.35,
"NWS hourly":0.25,
"MET Norway":0.20,
"Open-Meteo":0.10,
"GFS":0.10,
}

def safe_json(url,params=None,headers=None):
    try:
        h={"User-Agent":"kalshi-temp-model"}
        if headers:
            h.update(headers)
        r=requests.get(url,params=params,headers=h,timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def nws(lat,lon):

    p=safe_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not p:
        return None,None

    f_url=p["properties"]["forecast"]
    h_url=p["properties"]["forecastHourly"]

    forecast=safe_json(f_url)
    hourly=safe_json(h_url)

    daily=None
    hourly_high=None

    if forecast:
        try:
            daily=forecast["properties"]["periods"][0]["temperature"]
        except:
            pass

    if hourly:
        try:
            temps=[x["temperature"] for x in hourly["properties"]["periods"][:24]]
            hourly_high=max(temps)
        except:
            pass

    return daily,hourly_high


def open_meteo(lat,lon,model=None):

    params={
        "latitude":lat,
        "longitude":lon,
        "daily":"temperature_2m_max",
        "temperature_unit":"fahrenheit",
        "timezone":"auto"
    }

    if model:
        params["models"]=model

    j=safe_json("https://api.open-meteo.com/v1/forecast",params)

    if not j:
        return None

    try:
        return float(j["daily"]["temperature_2m_max"][0])
    except:
        return None


def metno(lat,lon):

    j=safe_json(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        {"lat":lat,"lon":lon},
        {"User-Agent":"kalshi-model"}
    )

    if not j:
        return None

    try:
        vals=[]
        for t in j["properties"]["timeseries"][:24]:
            c=t["data"]["instant"]["details"]["air_temperature"]
            f=c*9/5+32
            vals.append(f)
        return max(vals)
    except:
        return None


def consensus(values,bias):

    num=0
    den=0

    for name,val in values.items():

        if val is None:
            continue

        w=SOURCE_WEIGHTS.get(name,0)

        num+=w*val
        den+=w

    if den==0:
        return None

    return num/den + bias


def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))


def bracket_probs(mu,sigma):

    brackets=[
        ("82 or below",-999,82),
        ("83 to 84",83,84),
        ("85 to 86",85,86),
        ("87 to 88",87,88),
        ("89 to 90",89,90),
        ("91 or above",91,999)
    ]

    rows=[]

    for name,lo,hi in brackets:

        if lo==-999:
            p=normal_cdf(hi+0.5,mu,sigma)
        elif hi==999:
            p=1-normal_cdf(lo-0.5,mu,sigma)
        else:
            p=max(0,normal_cdf(hi+0.5,mu,sigma)-normal_cdf(lo-0.5,mu,sigma))

        rows.append({"Bracket":name,"Probability":round(p*100,1)})

    return pd.DataFrame(rows)


city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]

nws_f,nws_h=nws(lat,lon)
om=open_meteo(lat,lon)
gfs=open_meteo(lat,lon,"gfs_seamless")
met=metno(lat,lon)

sources={
"NWS forecast":nws_f,
"NWS hourly":nws_h,
"Open-Meteo":om,
"GFS":gfs,
"MET Norway":met,
}

rows=[]
for s,v in sources.items():
    rows.append({"Source":s,"Forecast":v})

df=pd.DataFrame(rows)

st.subheader("Forecast Sources")
st.dataframe(df)

usable=[v for v in sources.values() if v is not None]

if not usable:
    st.stop()

spread=max(usable)-min(usable)

mu=consensus(sources,profile["bias"])

sigma=max(0.85,float(profile["sigma"])+spread*0.22)

st.subheader("Model Output")

c1,c2,c3=st.columns(3)

c1.metric("Consensus High",round(mu,1))
c2.metric("Spread",round(spread,1))
c3.metric("Sigma",round(sigma,2))

st.subheader("Bracket Probabilities")

bp=bracket_probs(mu,sigma)
st.dataframe(bp)
