# Kalshi High Temperature Model – Stable Station Version
# Focus: better estimate of the settlement station temperature

import math
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model – Stable Station Version")

SAVE_FILE = Path("saved_ladders.json")

STATIONS = {
    "Los Angeles": "CLILAX",
    "New York": "KNYC",
    "Atlanta": "CLIATL",
    "Houston": "CLIHOU",
    "Miami": "CLIMIA",
    "Philadelphia": "CLIPHL",
    "Boston": "CLIBOS",
    "Denver": "CLIDEN",
    "Minneapolis": "CLIMSP",
    "Washington DC": "CLIDCA",
    "Oklahoma City": "CLIOKC",
    "New Orleans": "CLIMSY",
    "San Antonio": "CLISAT",
}

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085},
    "Dallas": {"lat": 32.8998, "lon": -97.0403},
    "Austin": {"lat": 30.1945, "lon": -97.6699},
    "Houston": {"lat": 29.9902, "lon": -95.3368},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277},
    "Miami": {"lat": 25.7959, "lon": -80.2870},
    "New York": {"lat": 40.7812, "lon": -73.9665},
    "San Antonio": {"lat": 29.5337, "lon": -98.4698},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580},
    "Philadelphia": {"lat": 39.8744, "lon": -75.2424},
    "Boston": {"lat": 42.3656, "lon": -71.0096},
    "Denver": {"lat": 39.8561, "lon": -104.6737},
    "Oklahoma City": {"lat": 35.3931, "lon": -97.6007},
    "Minneapolis": {"lat": 44.8848, "lon": -93.2223},
    "Washington DC": {"lat": 38.8512, "lon": -77.0402},
}

CITY_SIGMA = {
    "New York": 1.5,
    "Philadelphia": 1.5,
    "Washington DC": 1.6,
    "Boston": 1.6,
    "Los Angeles": 1.4,
    "Denver": 1.6,
    "Miami": 1.7,
    "Minneapolis": 1.7,
    "New Orleans": 1.8,
    "Phoenix": 1.9,
    "Las Vegas": 1.9,
    "Atlanta": 2.0,
    "Dallas": 2.0,
    "Austin": 2.0,
    "Houston": 2.0,
    "San Antonio": 2.0,
    "Oklahoma City": 2.1,
}

DEFAULT_LADDERS = {
    "Phoenix": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Las Vegas": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Los Angeles": "66 or below | 67-68 | 69-70 | 71-72 | 73-74 | 75 or above",
    "Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
    "Atlanta": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Miami": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "New York": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "San Antonio": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
}

def load_saved():
    if SAVE_FILE.exists():
        return json.loads(SAVE_FILE.read_text())
    return {}

def save_saved(data):
    SAVE_FILE.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def normal_cdf(x, mu, sigma):
    z = (x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def parse_ladder(text):
    out=[]
    for p in text.split("|"):
        p=p.strip()
        nums=[int(x) for x in re.findall(r"\d+",p)]
        if "below" in p.lower():
            out.append((p,None,nums[0]))
        elif "above" in p.lower():
            out.append((p,nums[0],None))
        else:
            out.append((p,nums[0],nums[1]))
    return out

def bracket_probs(mu, ladder_text, city):
    sigma=CITY_SIGMA.get(city,1.8)
    ladder=parse_ladder(ladder_text)
    rows=[]
    for lab,lo,hi in ladder:
        if lo is None:
            p=normal_cdf(hi+.5,mu,sigma)
        elif hi is None:
            p=1-normal_cdf(lo-.5,mu,sigma)
        else:
            p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True)
    return rows

saved=load_saved()

city=st.selectbox("City",list(CITIES.keys()))
lat=CITIES[city]["lat"]
lon=CITIES[city]["lon"]

st.write("Kalshi Settlement Station:",STATIONS.get(city,"N/A"))

if city not in saved:
    saved[city]=DEFAULT_LADDERS.get(city,"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above")

ladder_text=st.text_input("Kalshi Ladder",saved[city])

if st.button("Save Ladder"):
    saved[city]=ladder_text
    save_saved(saved)
    st.success("Saved")

weather=safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude":lat,
        "longitude":lon,
        "daily":"temperature_2m_max",
        "current":"temperature_2m",
        "temperature_unit":"fahrenheit",
        "timezone":"auto"
    }
)

if weather:

    current=float(weather["current"]["temperature_2m"])
    forecast=float(weather["daily"]["temperature_2m_max"][0])

    # Station-centered estimate
    consensus=(forecast*0.7)+(current*0.3)

    # guardrail against unrealistic drift
    if abs(consensus-forecast)>2:
        consensus=forecast-1

    st.subheader("Forecast High")
    st.write(round(forecast,1))

    st.subheader("Model Consensus High")
    st.write(round(consensus,1))

    rows=bracket_probs(consensus,ladder_text,city)

    df=pd.DataFrame(rows,columns=["Bracket","Probability"])
    df["Probability"]=df["Probability"].apply(lambda x:f"{x*100:.1f}%")

    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df)

else:
    st.error("Weather data unavailable")
