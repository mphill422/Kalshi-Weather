# Kalshi High Temperature Model – v17 Lean Replicator
# Preserves:
# - Per-city ladder saving
# - City list and workflow
# - Open-Meteo weather input
# Improves:
# - Monte Carlo temperature distribution
# - Intraday sigma tightening

import math
import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model")

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

CITY_SIGMA = {
    "New York": 1.5,
    "Los Angeles": 1.5,
    "Miami": 1.7,
    "Phoenix": 1.9,
    "Las Vegas": 1.9,
    "Atlanta": 2.0,
    "Dallas": 1.9,
    "Austin": 1.9,
    "Houston": 1.9,
    "San Antonio": 1.9,
}

def load_saved():
    if SAVE_FILE.exists():
        try:
            return json.loads(SAVE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_saved(data):
    SAVE_FILE.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def parse_ladder(text):
    out = []
    for p in text.split("|"):
        p = p.strip()
        nums = [int(x) for x in re.findall(r"\d+", p)]
        if not nums:
            continue
        if "below" in p.lower():
            out.append((p, None, nums[0]))
        elif "above" in p.lower():
            out.append((p, nums[0], None))
        else:
            if len(nums) >= 2:
                out.append((p, nums[0], nums[1]))
    return out

def choose_sigma(city):
    hour = datetime.now().hour
    base = CITY_SIGMA.get(city, 1.8)
    if hour < 11:
        factor = 1.0
    elif hour < 14:
        factor = 0.9
    elif hour < 16:
        factor = 0.85
    else:
        factor = 0.75
    return base * factor

def bracket_probs(mu, ladder_text, city):
    sigma = choose_sigma(city)
    ladder = parse_ladder(ladder_text)

    sims = np.random.normal(mu, sigma, 12000)
    sims = np.rint(sims).astype(int)

    rows = []
    for lab, lo, hi in ladder:
        if lo is None:
            p = np.mean(sims <= hi)
        elif hi is None:
            p = np.mean(sims >= lo)
        else:
            p = np.mean((sims >= lo) & (sims <= hi))
        rows.append((lab, p))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

saved = load_saved()

city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]

st.write("Kalshi Settlement Station:", STATIONS.get(city, "N/A"))

if city not in saved:
    saved[city] = DEFAULT_LADDERS[city]

ladder_text = st.text_input("Kalshi Ladder", saved[city])

if st.button("Save Ladder"):
    saved[city] = ladder_text
    save_saved(saved)
    st.success("Saved for " + city)

weather = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "forecast_days": 1,
    },
)

if not weather:
    st.error("Weather data unavailable")
    st.stop()

current = float(weather["current"]["temperature_2m"])
forecast_high = float(weather["daily"]["temperature_2m_max"][0])

st.subheader("Forecast High")
st.write(f"{forecast_high:.1f}")

rows = bracket_probs(forecast_high, ladder_text, city)

df = pd.DataFrame(rows, columns=["Bracket", "Probability"])
df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.1f}%")

st.subheader("Kalshi Bracket Probabilities")
st.dataframe(df, use_container_width=True)
