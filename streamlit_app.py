# Kalshi High Temperature Model – Targeted Revision
# Changes added:
# - Permanent per-city ladder saving (local JSON file)
# - Auto-sorts ladder text into numeric order on save/load
# - Adaptive sigma by city (less overconfidence in volatile cities)
# - Adds hourly "peak ramp" logic using the same Open-Meteo source
# - Consensus now blends daily high + remaining hourly peak instead of max(current, forecast)
# - Same workflow / no extra weather sources added

import math
import re
import json
from pathlib import Path
from datetime import datetime

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
    "Phoenix": 1.9,
    "Las Vegas": 1.9,
    "Atlanta": 2.1,
    "Dallas": 2.0,
    "Austin": 2.0,
    "Houston": 2.0,
    "San Antonio": 2.0,
    "Oklahoma City": 2.2,
}

CITY_RAMP_WEIGHT = {
    # More stable cities rely more on the official daily forecast
    "New York": 0.30,
    "Philadelphia": 0.30,
    "Washington DC": 0.35,
    "Boston": 0.35,
    "Los Angeles": 0.30,
    "Denver": 0.35,
    # More volatile cities rely more on remaining hourly peak
    "Miami": 0.40,
    "New Orleans": 0.45,
    "Phoenix": 0.40,
    "Las Vegas": 0.40,
    "Atlanta": 0.50,
    "Dallas": 0.55,
    "Austin": 0.55,
    "Houston": 0.55,
    "San Antonio": 0.55,
    "Oklahoma City": 0.60,
    "Minneapolis": 0.40,
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
        r = requests.get(url, params=params, timeout=12)
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

def compute_consensus(city, current_temp, daily_high, hourly_times, hourly_temps):
    """
    Targeted fix:
    blend daily max with remaining hourly peak from the same source.
    This keeps the model closer to actual intraday ramp behavior without
    changing your source or cluttering the UI.
    """
    remaining_peak = daily_high
    local_hour = None

    if hourly_times and hourly_temps:
        try:
            # Open-Meteo timezone=auto returns local timestamps like 2026-03-11T15:00
            hourly_dt = [datetime.fromisoformat(t) for t in hourly_times]
            now_idx = 0
            # best effort: use current hour from the first hourly slot >= current local clock hour
            now_local = datetime.now()
            future_temps = [temp for dt, temp in zip(hourly_dt, hourly_temps) if dt >= now_local]
            if future_temps:
                remaining_peak = max(future_temps)
            else:
                remaining_peak = max(hourly_temps)
            local_hour = now_local.hour
        except Exception:
            remaining_peak = max(hourly_temps)

    ramp_weight = CITY_RAMP_WEIGHT.get(city, 0.45)

    # Baseline blend
    blended = (1 - ramp_weight) * daily_high + ramp_weight * remaining_peak

    # Late-day gentle dampener: after 3 PM local, if current is still far below daily forecast,
    # reduce consensus slightly toward remaining peak instead of clinging to the full daily max.
    if local_hour is not None and local_hour >= 15 and current_temp < daily_high - 2:
        blended = 0.35 * daily_high + 0.65 * remaining_peak

    # Never below current, never wildly above the stronger source
    return max(current_temp, blended)

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
        "hourly": "temperature_2m",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "forecast_days": 1,
    },
)

if weather:
    current = float(weather["current"]["temperature_2m"])
    forecast = float(weather["daily"]["temperature_2m_max"][0])
    hourly_times = weather.get("hourly", {}).get("time", [])
    hourly_temps = weather.get("hourly", {}).get("temperature_2m", [])

    consensus = compute_consensus(city, current, forecast, hourly_times, hourly_temps)

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
