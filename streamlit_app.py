# Kalshi Temperature Model v21.1
# Stable hotfix version based on v21
# - Reverts to the working v21 style
# - No live Kalshi API dependency
# - No crash on empty ladders
# - Uses built-in city ladders
# - Probabilities shown as percentages
# - Keeps spread safety valve and forecast model

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v21.1", layout="wide")
st.title("Kalshi Temperature Model v21.1")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","bias":0.5,"afternoon_bump":0.3},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","bias":0.4,"afternoon_bump":0.3},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","bias":-0.6,"afternoon_bump":0.0},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","bias":0.4,"afternoon_bump":0.3},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","bias":0.3,"afternoon_bump":0.2},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","bias":0.3,"afternoon_bump":0.2},
    "Atlanta": {"lat":33.6407,"lon":-84.4277,"tz":"America/New_York","bias":0.2,"afternoon_bump":0.2},
    "NYC": {"lat":40.7829,"lon":-73.9654,"tz":"America/New_York","bias":0.6,"afternoon_bump":0.1},
    "Miami": {"lat":25.7959,"lon":-80.2870,"tz":"America/New_York","bias":0.2,"afternoon_bump":0.1},
}

DEFAULT_LADDERS = {
    "Phoenix": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Las Vegas": "73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
    "Los Angeles": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin": "76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
    "Houston": "76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
    "Atlanta": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "NYC": "62 or below | 63-64 | 65-66 | 67-68 | 69-70 | 71 or above",
    "Miami": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
}

BASE_WEIGHTS = {"ICON":0.35,"OpenMeteo":0.30,"GFS":0.20,"NWS":0.15}
OUTLIER_HALF = 3.0
OUTLIER_REMOVE = 4.5
SPREAD_SAFETY_THRESHOLD = 5.0
SPREAD_SAFETY_MULTIPLIER = 0.4


def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n//2]
    return (s[n//2-1] + s[n//2]) / 2


def compute_weights(forecasts):
    vals = [v for v in forecasts.values() if v is not None]
    med = median(vals)
    if med is None:
        return {k: 0 for k in forecasts}
    adj = {}
    for k, v in forecasts.items():
        if v is None:
            adj[k] = 0
            continue
        d = abs(v - med)
        w = BASE_WEIGHTS.get(k, 0)
        if d > OUTLIER_REMOVE:
            w = 0
        elif d > OUTLIER_HALF:
            w *= 0.5
        adj[k] = w
    return adj


def consensus(forecasts, weights):
    num = 0.0
    den = 0.0
    for k, v in forecasts.items():
        if v is None:
            continue
        w = weights.get(k, 0)
        num += v * w
        den += w
    if den == 0:
        return None
    return num / den


def solar_adjust(cloud, hour, city_name):
    if hour < 9 or hour > 17:
        return 0.0
    if cloud is None:
        return 0.0
    if cloud < 10:
        adj = 1.0
    elif cloud < 30:
        adj = 0.6
    elif cloud < 50:
        adj = 0.2
    else:
        adj = -0.5
    if city_name in {"Phoenix", "Las Vegas"}:
        if cloud < 20:
            adj += 0.4
        elif cloud < 40:
            adj += 0.2
    return adj


def expected_curve(hour):
    curve = {
        6:0.55, 7:0.60, 8:0.65, 9:0.70, 10:0.75, 11:0.80,
        12:0.85, 13:0.90, 14:0.95, 15:0.98, 16:1.00, 17:1.00
    }
    return curve.get(hour, 1.0 if hour > 17 else 0.50)


def raw_trajectory_adjust(current_temp, forecast_high, hour):
    if current_temp is None or forecast_high is None:
        return 0.0
    if hour < 8 or hour > 16:
        return 0.0
    pct = expected_curve(hour)
    expected_temp = forecast_high * pct
    diff = current_temp - expected_temp
    adj = diff * 0.35
    return max(min(adj, 2.0), -2.0)


def apply_spread_safety_valve(traj, spread):
    if spread > SPREAD_SAFETY_THRESHOLD:
        return traj * SPREAD_SAFETY_MULTIPLIER
    return traj


def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def parse_ladder(text):
    brackets = []
    parts = [p.strip() for p in text.split("|") if p.strip()]
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        lower = p.lower()
        if "below" in lower:
            brackets.append((p, None, nums[0]))
        elif "above" in lower:
            brackets.append((p, nums[0], None))
        else:
            brackets.append((p, nums[0], nums[1]))
    return brackets


def bracket_probs(mu, sigma, brackets):
    rows = []
    for lab, lo, hi in brackets:
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((lab, p))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
lat = profile["lat"]
lon = profile["lon"]
tz = profile["tz"]
local_hour = datetime.now(ZoneInfo(tz)).hour

ladder_text = st.text_input("Kalshi ladder", DEFAULT_LADDERS[city])

openmeteo = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat,
    "longitude": lon,
    "daily": "temperature_2m_max",
    "current": "temperature_2m,cloud_cover",
    "temperature_unit": "fahrenheit",
    "timezone": "auto"
})

gfs = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat,
    "longitude": lon,
    "daily": "temperature_2m_max",
    "models": "gfs_seamless",
    "temperature_unit": "fahrenheit",
    "timezone": "auto"
})

icon = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat,
    "longitude": lon,
    "daily": "temperature_2m_max",
    "models": "icon_seamless",
    "temperature_unit": "fahrenheit",
    "timezone": "auto"
})

nws = safe_get(f"https://api.weather.gov/points/{lat},{lon}")
nws_high = None
if nws and "properties" in nws and nws["properties"].get("forecast"):
    fc = safe_get(nws["properties"]["forecast"])
    if fc and "properties" in fc and "periods" in fc["properties"]:
        for p in fc["properties"]["periods"]:
            if p.get("isDaytime"):
                nws_high = p.get("temperature")
                break

forecasts = {
    "ICON": icon["daily"]["temperature_2m_max"][0] if icon and "daily" in icon else None,
    "OpenMeteo": openmeteo["daily"]["temperature_2m_max"][0] if openmeteo and "daily" in openmeteo else None,
    "GFS": gfs["daily"]["temperature_2m_max"][0] if gfs and "daily" in gfs else None,
    "NWS": nws_high
}

weights = compute_weights(forecasts)
cons = consensus(forecasts, weights)

cloud = None
current_temp = None
if openmeteo and "current" in openmeteo:
    cloud = openmeteo["current"].get("cloud_cover")
    current_temp = openmeteo["current"].get("temperature_2m")

vals = [v for v in forecasts.values() if v is not None]
spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
sigma = 1.3 + (spread * 0.25)

raw_traj = 0.0
traj = 0.0
if cons is not None:
    cons += profile["bias"]
    cons += solar_adjust(cloud, local_hour, city)
    raw_traj = raw_trajectory_adjust(current_temp, cons, local_hour)
    traj = apply_spread_safety_valve(raw_traj, spread)
    cons += traj
    if local_hour >= 13:
        cons += profile.get("afternoon_bump", 0.0)

st.subheader("Forecast Sources")
st.write(forecasts)

st.subheader("Consensus High")
st.write(round(cons, 2) if cons is not None else "N/A")

st.subheader("Forecast Spread")
st.write(round(spread, 2))

st.subheader("Sigma")
st.write(round(sigma, 2))

st.subheader("Spread Safety Valve")
st.write("ON" if spread > SPREAD_SAFETY_THRESHOLD else "OFF")

if cons is not None:
    brackets = parse_ladder(ladder_text)
    rows = bracket_probs(cons, sigma, brackets)
    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability"] = df["Model Probability"].apply(lambda x: f"{x*100:.1f}%")
    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df, use_container_width=True)

st.caption("Model v21.1 â stable hotfix based on v21")
