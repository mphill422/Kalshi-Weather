# Kalshi High Temperature Model – NWS + Open-Meteo Fixed Version

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
    "New York": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "San Antonio": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "New Orleans": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "Philadelphia": "73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
    "Boston": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Denver": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Oklahoma City": "75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
    "Minneapolis": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Washington DC": "76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
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
    "Atlanta": 2.0,
    "Dallas": 1.9,
    "Austin": 1.9,
    "Houston": 1.9,
    "San Antonio": 1.9,
    "Oklahoma City": 2.1,
}

CITY_RAMP_WEIGHT = {
    "New York": 0.20,
    "Philadelphia": 0.20,
    "Boston": 0.20,
    "Washington DC": 0.25,
    "Los Angeles": 0.20,
    "Denver": 0.25,
    "Miami": 0.30,
    "Minneapolis": 0.25,
    "New Orleans": 0.30,
    "Phoenix": 0.35,
    "Las Vegas": 0.35,
    "Atlanta": 0.35,
    "Dallas": 0.40,
    "Austin": 0.40,
    "Houston": 0.40,
    "San Antonio": 0.40,
    "Oklahoma City": 0.45,
}

HEADERS = {
    "User-Agent": "kalshi-temp-model/1.0 (personal use)",
    "Accept": "application/geo+json, application/json",
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
            out.append((p, None, nums[0], nums[0] - 1000))
        elif "above" in low:
            out.append((p, nums[0], None, nums[0] + 1000))
        else:
            if len(nums) >= 2:
                out.append((p, nums[0], nums[1], nums[0]))
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
    return rows

def fetch_open_meteo(lat, lon):
    return safe_get(
        "https://api.open-meteo.com/v1/forecast",
        params={
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

def fetch_nws(lat, lon):
    pts = safe_get(f"https://api.weather.gov/points/{lat},{lon}", headers=HEADERS)
    if not pts:
        return None
    props = pts.get("properties", {})
    hourly_url = props.get("forecastHourly")
    forecast_url = props.get("forecast")
    if not hourly_url and not forecast_url:
        return None

    hourly = safe_get(hourly_url, headers=HEADERS) if hourly_url else None
    forecast = safe_get(forecast_url, headers=HEADERS) if forecast_url else None

    out = {"daily_high": None, "hourly_future_peak": None}
    now = datetime.now().astimezone()

    if hourly and hourly.get("properties", {}).get("periods"):
        periods = hourly["properties"]["periods"]
        future_vals = []
        for p in periods:
            temp = p.get("temperature")
            start = p.get("startTime")
            if temp is None or start is None:
                continue
            try:
                dt = datetime.fromisoformat(start)
            except Exception:
                continue
            if dt >= now:
                future_vals.append(float(temp))
        if future_vals:
            out["hourly_future_peak"] = max(future_vals)

    if forecast and forecast.get("properties", {}).get("periods"):
        periods = forecast["properties"]["periods"]
        highs = []
        for p in periods:
            temp = p.get("temperature")
            is_day = p.get("isDaytime")
            start = p.get("startTime")
            if temp is None or not is_day or start is None:
                continue
            try:
                dt = datetime.fromisoformat(start)
            except Exception:
                continue
            if dt.date() == now.date():
                highs.append(float(temp))
        if highs:
            out["daily_high"] = max(highs)

    return out

def compute_consensus(city, current_temp, om_daily, om_future_peak, nws_daily, nws_future_peak):
    daily_candidates = [x for x in [nws_daily, om_daily] if x is not None]
    hourly_candidates = [x for x in [nws_future_peak, om_future_peak] if x is not None]

    if not daily_candidates and not hourly_candidates:
        return current_temp, current_temp

    blended_daily = sum(daily_candidates) / len(daily_candidates) if daily_candidates else current_temp
    future_peak = sum(hourly_candidates) / len(hourly_candidates) if hourly_candidates else blended_daily

    ramp_weight = CITY_RAMP_WEIGHT.get(city, 0.30)
    raw = (1 - ramp_weight) * blended_daily + ramp_weight * future_peak

    all_candidates = daily_candidates + hourly_candidates
    upper_source = max(all_candidates) if all_candidates else raw
    lower_source = min(all_candidates) if all_candidates else raw

    # guards against bad spikes
    hard_cap = current_temp + 8.0
    soft_cap = upper_source + 2.0

    consensus = min(raw, hard_cap, soft_cap)
    consensus = max(consensus, current_temp, lower_source - 1.0)

    return consensus, blended_daily

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

om = fetch_open_meteo(lat, lon)
if not om:
    st.error("Weather data unavailable")
    st.stop()

current = float(om["current"]["temperature_2m"])
om_daily = float(om["daily"]["temperature_2m_max"][0])

now_local = datetime.now()
om_future_vals = []
for t, temp in zip(om.get("hourly", {}).get("time", []), om.get("hourly", {}).get("temperature_2m", [])):
    try:
        dt = datetime.fromisoformat(t)
    except Exception:
        continue
    if dt >= now_local:
        om_future_vals.append(float(temp))
om_future_peak = max(om_future_vals) if om_future_vals else om_daily

nws = fetch_nws(lat, lon)
nws_daily = nws.get("daily_high") if nws else None
nws_future_peak = nws.get("hourly_future_peak") if nws else None

consensus, blended_daily = compute_consensus(
    city=city,
    current_temp=current,
    om_daily=om_daily,
    om_future_peak=om_future_peak,
    nws_daily=nws_daily,
    nws_future_peak=nws_future_peak,
)

st.subheader("Forecast High")
st.write(f"{blended_daily:.1f}")

st.subheader("Model Consensus High")
st.write(f"{consensus:.1f}")

rows = bracket_probs(consensus, ladder_text, city)

df = pd.DataFrame(rows, columns=["Bracket", "Probability"])
df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.1f}%")

st.subheader("Kalshi Bracket Probabilities")
st.dataframe(df, use_container_width=True)
