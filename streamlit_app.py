# Kalshi High Temperature Model – Revised Ladder Save Version
# Changes:
# - Permanent per-city ladder saving (local JSON file)
# - Auto-sorts ladder text into numeric order on save/load
# - Adaptive sigma by city (less overconfidence in volatile cities)
# - Keeps same workflow and same weather source
# - Leaves unmapped settlement stations as N/A instead of guessing

import math
import re
import json
from pathlib import Path

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
    "New Orleans": {"lat": 29.9934, "lon": -90.2580},
    "Philadelphia": {"lat": 39.8744, "lon": -75.2424},
    "Boston": {"lat": 42.3656, "lon": -71.0096},
    "Denver": {"lat": 39.8561, "lon": -104.6737},
    "Oklahoma City": {"lat": 35.3931, "lon": -97.6007},
    "Minneapolis": {"lat": 44.8848, "lon": -93.2223},
    "Washington DC": {"lat": 38.8512, "lon": -77.0402},
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
    "New York": "70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "San Antonio": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "New Orleans": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "Philadelphia": "70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "Boston": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Denver": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Oklahoma City": "75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
    "Minneapolis": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Washington DC": "70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
}

CITY_SIGMA = {
    "New York": 1.5,
    "Philadelphia": 1.5,
    "Washington DC": 1.6,
    "Boston": 1.6,
    "Los Angeles": 1.5,
    "Denver": 1.6,
    "Miami": 1.7,
    "Minneapolis": 1.7,
    "New Orleans": 1.8,
    "Atlanta": 2.1,
    "Dallas": 2.0,
    "Austin": 2.0,
    "Houston": 2.0,
    "San Antonio": 2.0,
    "Oklahoma City": 2.2,
    "Phoenix": 1.9,
    "Las Vegas": 1.9,
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

def safe_get(url, params):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def parse_ladder(text):
    out = []
    parts = [p.strip() for p in text.split("|") if p.strip()]
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        lower = p.lower()

        if "below" in lower:
            if not nums:
                continue
            hi = nums[0]
            out.append((p, None, hi, hi - 1000))
        elif "above" in lower:
            if not nums:
                continue
            lo = nums[0]
            out.append((p, lo, None, lo + 1000))
        else:
            if len(nums) < 2:
                continue
            lo, hi = nums[0], nums[1]
            out.append((p, lo, hi, lo))
    return out

def sort_and_format_ladder(text):
    parsed = parse_ladder(text)
    if not parsed:
        return text

    parsed.sort(key=lambda x: x[3])
    labels = []
    for _, lo, hi, _ in parsed:
        if lo is None:
            labels.append(f"{hi} or below")
        elif hi is None:
            labels.append(f"{lo} or above")
        else:
            labels.append(f"{lo}-{hi}")
    return " | ".join(labels)

def bracket_probs(mu, ladder_text, city):
    sigma = CITY_SIGMA.get(city, 1.8)
    ladder = parse_ladder(ladder_text)
    rows = []

    for lab, lo, hi, _ in ladder:
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)

        rows.append((lab, max(0.0, min(1.0, p))))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, sigma

saved = load_saved()

city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]

st.write("Kalshi Settlement Station:", STATIONS.get(city, "N/A"))

if city not in saved:
    saved[city] = DEFAULT_LADDERS[city]

saved[city] = sort_and_format_ladder(saved[city])
ladder_text = st.text_input("Kalshi Ladder", saved[city])

if st.button("Save Ladder"):
    cleaned = sort_and_format_ladder(ladder_text)
    saved[city] = cleaned
    save_saved(saved)
    st.success("Saved for " + city)

weather = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    },
)

if weather:
    current = float(weather["current"]["temperature_2m"])
    forecast = float(weather["daily"]["temperature_2m_max"][0])
    consensus = max(current, forecast)

    st.subheader("Forecast High")
    st.write(f"{forecast:.1f}")

    st.subheader("Model Consensus High")
    st.write(f"{consensus:.1f}")

    rows, sigma = bracket_probs(consensus, ladder_text, city)

    df = pd.DataFrame(rows, columns=["Bracket", "Probability"])
    df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.1f}%")

    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df, use_container_width=True)
else:
    st.error("Weather data unavailable")
