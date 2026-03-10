# Kalshi High Temperature Model - Expanded Cities Version
# Same logic as previous stable model. Only change: additional cities added.

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","bias":0.4},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","bias":0.3},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","bias":-0.5},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","bias":0.3},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","bias":0.2},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","bias":0.1},
    "Atlanta": {"lat":33.6407,"lon":-84.4277,"tz":"America/New_York","bias":0.2},
    "Miami": {"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","bias":0.1},
    "New York": {"lat":40.6413,"lon":-73.7781,"tz":"America/New_York","bias":0.1},
    "San Antonio": {"lat":29.5337,"lon":-98.4698,"tz":"America/Chicago","bias":0.2},
    "New Orleans": {"lat":29.9934,"lon":-90.2580,"tz":"America/Chicago","bias":0.1},
    "Philadelphia": {"lat":39.8744,"lon":-75.2424,"tz":"America/New_York","bias":0.1},
    "Boston": {"lat":42.3656,"lon":-71.0096,"tz":"America/New_York","bias":0.1},
    "Denver": {"lat":39.8561,"lon":-104.6737,"tz":"America/Denver","bias":0.2},
    "Oklahoma City": {"lat":35.3931,"lon":-97.6007,"tz":"America/Chicago","bias":0.2},
    "Minneapolis": {"lat":44.8848,"lon":-93.2223,"tz":"America/Chicago","bias":0.1},
    "Washington DC": {"lat":38.8512,"lon":-77.0402,"tz":"America/New_York","bias":0.1}
}

DEFAULT_LADDERS = {
    "Phoenix":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Las Vegas":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Los Angeles":"66 or below | 67-68 | 69-70 | 71-72 | 73-74 | 75 or above",
    "Dallas":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston":"79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
    "Atlanta":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Miami":"80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "New York":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "San Antonio":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "New Orleans":"80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "Philadelphia":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "Boston":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Denver":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Oklahoma City":"75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
    "Minneapolis":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Washington DC":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above"
}

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
    if n==0:return None
    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2

def compute_weights(forecasts):
    vals=[v for v in forecasts.values() if v is not None]
    med=median(vals)
    out={}
    for k,v in forecasts.items():
        if v is None:
            out[k]=0
            continue
        d=abs(v-med)
        w=1
        if d>4.5:w=0
        elif d>3:w*=0.5
        out[k]=w
    return out

def consensus(forecasts,weights):
    num=0
    den=0
    for k,v in forecasts.items():
        if v is None:continue
        w=weights.get(k,0)
        num+=v*w
        den+=w
    if den==0:return None
    return num/den

def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def parse_ladder(text):
    out=[]
    for p in text.split("|"):
        nums=[int(x) for x in re.findall(r"\d+",p)]
        if "below" in p:out.append((p,None,nums[0]))
        elif "above" in p:out.append((p,nums[0],None))
        else:out.append((p,nums[0],nums[1]))
    return out

def bracket_probs(mu):
    sigma=1.4
    brackets=parse_ladder(ladder_text)
    rows=[]
    for lab,lo,hi in brackets:
        if lo is None:p=normal_cdf(hi+.5,mu,sigma)
        elif hi is None:p=1-normal_cdf(lo-.5,mu,sigma)
        else:p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True)
    return rows

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]
tz=profile["tz"]

ladder_text=st.text_input("Kalshi Ladder",DEFAULT_LADDERS[city])

openmeteo=safe_get("https://api.open-meteo.com/v1/forecast",{
"latitude":lat,
"longitude":lon,
"daily":"temperature_2m_max",
"current":"temperature_2m",
"temperature_unit":"fahrenheit",
"timezone":"auto"
})

if openmeteo:
    current_temp=openmeteo["current"]["temperature_2m"]
    forecast=openmeteo["daily"]["temperature_2m_max"][0]

    cons=max(forecast,current_temp)

    st.subheader("Forecast High")
    st.write(forecast)

    st.subheader("Current Temp")
    st.write(current_temp)

    st.subheader("Consensus High")
    st.write(cons)

    rows=bracket_probs(cons)
    df=pd.DataFrame(rows,columns=["Bracket","Probability"])
    df["Probability"]=df["Probability"].apply(lambda x:f"{x*100:.1f}%")

    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df)

else:
    st.error("Weather data unavailable")
