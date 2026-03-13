# Kalshi High Temperature Model – V4.5 Targeted Fix
# Minimal changes from V4.4:
# 1) Late-day forecast anchor so consensus stops running too cold
# 2) Stronger sigma tightening after 3 PM so probabilities are less flat

import math
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model – V4.5 Targeted Fix")

SAVE_FILE = Path("saved_ladders.json")

HEADERS = {
    "User-Agent": "kalshi-temp-model/1.0",
    "Accept": "application/geo+json, application/json",
}

STATIONS = {
    "Phoenix": "CLIPHX",
    "Las Vegas": "CLILAS",
    "Los Angeles": "CLILAX",
    "Dallas": "CLIDFW",
    "Austin": "CLIAUS",
    "Houston": "CLIHOU",
    "Atlanta": "CLIATL",
    "Miami": "CLIMIA",
    "New York": "KNYC",
    "San Antonio": "CLISAT",
    "New Orleans": "CLIMSY",
    "Philadelphia": "CLIPHL",
    "Boston": "CLIBOS",
    "Denver": "CLIDEN",
    "Oklahoma City": "CLIOKC",
    "Minneapolis": "CLIMSP",
    "Washington DC": "CLIDCA",
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
    "New York": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "San Antonio": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "New Orleans": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "Philadelphia": "73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
    "Boston": "48 or below | 49-50 | 51-52 | 53-54 | 55-56 | 57 or above",
    "Denver": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Oklahoma City": "75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
    "Minneapolis": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Washington DC": "76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
}

BASE_SIGMA = {
    "New York": 2.4,
    "Philadelphia": 2.3,
    "Washington DC": 2.3,
    "Boston": 2.2,
    "Los Angeles": 1.8,
    "Denver": 2.2,
    "Miami": 2.6,
    "Minneapolis": 2.3,
    "New Orleans": 2.6,
    "Phoenix": 2.1,
    "Las Vegas": 2.1,
    "Atlanta": 2.6,
    "Dallas": 2.7,
    "Austin": 2.7,
    "Houston": 2.8,
    "San Antonio": 2.8,
    "Oklahoma City": 2.7,
}

DESERT_CITIES = {"Phoenix", "Las Vegas"}

def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def safe_get(url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def parse_ladder(text):
    out = []
    for p in text.split("|"):
        p = p.strip()
        nums = [int(x) for x in re.findall(r"\d+", p)]
        if not nums:
            continue
        low = p.lower()
        if "below" in low:
            out.append((p, None, nums[0]))
        elif "above" in low:
            out.append((p, nums[0], None))
        elif len(nums) >= 2:
            out.append((p, nums[0], nums[1]))
    return out

def ladder_to_boxes(text):
    parts = [p.strip() for p in text.split("|")]
    while len(parts) < 6:
        parts.append("")
    return parts[:6]

def normalize_box(text, index):
    t = text.strip()
    if not t:
        return ""
    nums = [int(x) for x in re.findall(r"\d+", t)]
    low = t.lower()
    if "below" in low or "above" in low or "-" in t:
        return t
    if len(nums) == 1:
        n = nums[0]
        if index == 0:
            return f"{n} or below"
        if index == 5:
            return f"{n} or above"
        return str(n)
    return t

def boxes_to_ladder(parts):
    normalized = [normalize_box(p, i) for i, p in enumerate(parts)]
    cleaned = [p for p in normalized if p.strip()]
    return " | ".join(cleaned)

def choose_sigma(city):
    sigma = BASE_SIGMA.get(city, 2.4)
    hour = datetime.now().hour
    if hour < 11:
        factor = 1.05
    elif hour < 14:
        factor = 0.95
    elif hour < 16:
        factor = 0.78
    else:
        factor = 0.62
    sigma *= factor
    if city in DESERT_CITIES:
        sigma *= 0.95
    return max(1.15, min(3.0, sigma))

def bracket_probs(mu, ladder_text, city):
    sigma = choose_sigma(city)
    ladder = parse_ladder(ladder_text)
    rows = []
    for lab, lo, hi in ladder:
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((lab, max(0.0, min(1.0, p))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, sigma

def central_two_degree_call(mu, ladder_text):
    ladder = parse_ladder(ladder_text)
    interior = []
    for lab, lo, hi in ladder:
        if lo is not None and hi is not None:
            center = (lo + hi) / 2
            interior.append((lab, center))
    if not interior:
        return None
    best = min(interior, key=lambda x: abs(x[1] - mu))
    return best[0]

def c_to_f(c):
    return (c * 9 / 5) + 32

def fetch_noaa_latest_observation(lat, lon, exact_station_id=None):
    if exact_station_id:
        obs = safe_get(f"https://api.weather.gov/stations/{exact_station_id}/observations/latest", headers=HEADERS)
        if obs:
            temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
            if temp_c is not None:
                return exact_station_id, float(c_to_f(temp_c))
    points = safe_get(f"https://api.weather.gov/points/{lat},{lon}", headers=HEADERS)
    if not points:
        return exact_station_id, None
    stations_url = points.get("properties", {}).get("observationStations")
    if not stations_url:
        return exact_station_id, None
    stations = safe_get(stations_url, headers=HEADERS)
    if not stations or not stations.get("observationStations"):
        return exact_station_id, None
    station_url = stations["observationStations"][0]
    station_id = station_url.rstrip("/").split("/")[-1]
    obs = safe_get(f"{station_url}/observations/latest", headers=HEADERS)
    if not obs:
        return station_id, None
    temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
    if temp_c is None:
        return station_id, None
    return station_id, float(c_to_f(temp_c))

def remaining_heat_floor(hour, forecast, current_obs):
    gap = max(0.0, forecast - current_obs)
    if hour < 12:
        frac = 0.45
    elif hour < 14:
        frac = 0.62
    elif hour < 16:
        frac = 0.78
    else:
        frac = 0.90
    return current_obs + frac * gap

def build_consensus(forecast, current_openmeteo, noaa_obs, hour):
    if noaa_obs is not None:
        base = (forecast * 0.60) + (current_openmeteo * 0.12) + (noaa_obs * 0.28)
        floor = remaining_heat_floor(hour, forecast, noaa_obs)
    else:
        base = (forecast * 0.72) + (current_openmeteo * 0.28)
        floor = remaining_heat_floor(hour, forecast, current_openmeteo)
    consensus = max(base, floor)
    if consensus > forecast + 0.6:
        consensus = forecast + 0.6
    return consensus

saved_ladders = load_json(SAVE_FILE)

city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]
exact_station = STATIONS.get(city)

st.write("Kalshi Settlement Station:", exact_station)

if city not in saved_ladders:
    saved_ladders[city] = DEFAULT_LADDERS.get(city, "70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above")

box_values = ladder_to_boxes(saved_ladders[city])

st.subheader("Kalshi Ladder")
b1 = st.text_input("Box 1", value=box_values[0], key=f"{city}_b1")
b2 = st.text_input("Box 2", value=box_values[1], key=f"{city}_b2")
b3 = st.text_input("Box 3", value=box_values[2], key=f"{city}_b3")
b4 = st.text_input("Box 4", value=box_values[3], key=f"{city}_b4")
b5 = st.text_input("Box 5", value=box_values[4], key=f"{city}_b5")
b6 = st.text_input("Box 6", value=box_values[5], key=f"{city}_b6")

ladder_text = boxes_to_ladder([b1, b2, b3, b4, b5, b6])

if st.button("Save Ladder"):
    saved_ladders[city] = ladder_text
    save_json(SAVE_FILE, saved_ladders)
    st.success("Saved")

weather = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto"
    }
)

if weather:
    current = float(weather["current"]["temperature_2m"])
    forecast = float(weather["daily"]["temperature_2m_max"][0])
    noaa_station_id, noaa_obs_temp = fetch_noaa_latest_observation(lat, lon, exact_station_id=exact_station)
    hour = datetime.now().hour

    st.subheader("Forecast High")
    st.write(round(forecast, 1))

    st.subheader("Current Temperature (Open-Meteo)")
    st.write(round(current, 1))

    st.subheader("Latest NOAA / NWS Station Observation")
    if noaa_obs_temp is not None:
        st.write(f"{round(noaa_obs_temp, 1)} °F")
        st.caption(f"Observation station used: {noaa_station_id}")
    else:
        st.write("Unavailable")
        if noaa_station_id:
            st.caption(f"Observation station attempted: {noaa_station_id}")

    anchor_obs = noaa_obs_temp if noaa_obs_temp is not None else current
    st.caption(f"Late-day forecast anchor floor: {remaining_heat_floor(hour, forecast, anchor_obs):.1f}")

    consensus = build_consensus(forecast, current, noaa_obs_temp, hour)

    st.subheader("Model Consensus High")
    st.write(round(consensus, 1))

    call = central_two_degree_call(consensus, ladder_text)
    if call:
        st.subheader("Model 2-Degree Call")
        st.write(call)

    rows, sigma = bracket_probs(consensus, ladder_text, city)
    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability %"] = df["Model Probability"].apply(lambda x: f"{x*100:.1f}%")
    st.caption(f"Model sigma used: {sigma:.2f}")
    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df[["Bracket", "Model Probability %"]], use_container_width=True)
else:
    st.error("Weather data unavailable")
