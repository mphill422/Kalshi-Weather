# Kalshi High Temperature Model – V4.2 Edge View
# Added:
# - Manual market odds inputs for each of the 6 ladder boxes
# - Market implied probability
# - Model probability
# - Edge %
# Keeps:
# - exact settlement stations
# - NOAA observation feed
# - 6 separate ladder boxes
# - permanent per-city ladder saving
# - time-of-day sigma tightening
# - desert-city tightening

import math
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model – V4.2 Edge View")

SAVE_FILE = Path("saved_ladders.json")
ODDS_FILE = Path("saved_market_odds.json")

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
    sigma = BASE_SIGMA.get(city, 1.8)
    hour = datetime.now().hour
    if hour < 11:
        factor = 1.00
    elif hour < 14:
        factor = 0.92
    elif hour < 16:
        factor = 0.86
    else:
        factor = 0.80
    sigma *= factor
    if city in DESERT_CITIES:
        sigma *= 0.90
    return max(1.10, min(2.4, sigma))

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

def american_to_implied_prob(odds_text):
    t = str(odds_text).strip().replace("−", "-")
    if t == "":
        return None
    t = t.replace("+", "")
    try:
        odds = int(t)
    except Exception:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return None

saved_ladders = load_json(SAVE_FILE)
saved_odds = load_json(ODDS_FILE)

city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]
exact_station = STATIONS.get(city)

st.write("Kalshi Settlement Station:", exact_station)

if city not in saved_ladders:
    saved_ladders[city] = DEFAULT_LADDERS.get(city, "70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above")
if city not in saved_odds:
    saved_odds[city] = ["", "", "", "", "", ""]

box_values = ladder_to_boxes(saved_ladders[city])
odds_values = saved_odds[city]
while len(odds_values) < 6:
    odds_values.append("")

st.subheader("Kalshi Ladder")
b1 = st.text_input("Box 1", value=box_values[0], key=f"{city}_b1")
b2 = st.text_input("Box 2", value=box_values[1], key=f"{city}_b2")
b3 = st.text_input("Box 3", value=box_values[2], key=f"{city}_b3")
b4 = st.text_input("Box 4", value=box_values[3], key=f"{city}_b4")
b5 = st.text_input("Box 5", value=box_values[4], key=f"{city}_b5")
b6 = st.text_input("Box 6", value=box_values[5], key=f"{city}_b6")
ladder_text = boxes_to_ladder([b1, b2, b3, b4, b5, b6])

st.subheader("Kalshi YES Odds (manual entry)")
o1 = st.text_input("Odds 1", value=odds_values[0], key=f"{city}_o1", placeholder="-150 or +212")
o2 = st.text_input("Odds 2", value=odds_values[1], key=f"{city}_o2", placeholder="-150 or +212")
o3 = st.text_input("Odds 3", value=odds_values[2], key=f"{city}_o3", placeholder="-150 or +212")
o4 = st.text_input("Odds 4", value=odds_values[3], key=f"{city}_o4", placeholder="-150 or +212")
o5 = st.text_input("Odds 5", value=odds_values[4], key=f"{city}_o5", placeholder="-150 or +212")
o6 = st.text_input("Odds 6", value=odds_values[5], key=f"{city}_o6", placeholder="-150 or +212")

if st.button("Save Ladder + Odds"):
    saved_ladders[city] = ladder_text
    saved_odds[city] = [o1, o2, o3, o4, o5, o6]
    save_json(SAVE_FILE, saved_ladders)
    save_json(ODDS_FILE, saved_odds)
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

    if noaa_obs_temp is not None:
        consensus = (forecast * 0.55) + (current * 0.20) + (noaa_obs_temp * 0.25)
    else:
        consensus = (forecast * 0.70) + (current * 0.30)

    if abs(consensus - forecast) > 2:
        consensus = forecast - 1 if consensus < forecast else forecast + 1

    st.subheader("Model Consensus High")
    st.write(round(consensus, 1))

    rows, sigma = bracket_probs(consensus, ladder_text, city)

    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability %"] = df["Model Probability"].apply(lambda x: round(x * 100, 1))

    st.subheader("Kalshi Bracket Probabilities")
    show_df = df[["Bracket", "Model Probability %"]].copy()
    show_df["Model Probability %"] = show_df["Model Probability %"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(show_df, use_container_width=True)

    ladder_labels_in_order = [normalize_box(x, i) for i, x in enumerate([b1, b2, b3, b4, b5, b6])]
    model_map = {lab: prob for lab, prob in rows}

    edge_rows = []
    for idx, (lab, odd) in enumerate(zip(ladder_labels_in_order, [o1, o2, o3, o4, o5, o6]), start=1):
        model_prob = model_map.get(lab)
        market_prob = american_to_implied_prob(odd)
        edge = None
        if model_prob is not None and market_prob is not None:
            edge = (model_prob - market_prob) * 100
        edge_rows.append({
            "Box": idx,
            "Bracket": lab,
            "YES Odds": odd,
            "Market Implied %": None if market_prob is None else round(market_prob * 100, 1),
            "Model %": None if model_prob is None else round(model_prob * 100, 1),
            "Edge %": None if edge is None else round(edge, 1),
        })

    edge_df = pd.DataFrame(edge_rows)
    for col in ["Market Implied %", "Model %", "Edge %"]:
        edge_df[col] = edge_df[col].apply(lambda x: "" if x is None else f"{x:.1f}%")

    st.subheader("Market vs Model Edge")
    st.dataframe(edge_df, use_container_width=True)
else:
    st.error("Weather data unavailable")
