
# Kalshi Temperature Model - Stable Adaptive
# Tweaks inspired by Daily Dew Point:
# 1) source-specific rolling bias correction by city
# 2) current-temperature floor rule
# 3) late-day remaining-heating cap
# 4) dewpoint / humidity suppression
# 5) no-bet filter
#
# Notes:
# - Bias correction gets better as history accumulates.
# - Snapshots are auto-logged locally.
# - You can enter the final actual high later to improve future corrections.

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model - Stable Adaptive", layout="wide")
st.title("Kalshi Temperature Model - Stable Adaptive")

DATA_DIR = Path("model_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_PATH = DATA_DIR / "forecast_history.csv"

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "bias": 0.5, "afternoon_bump": 0.3, "type": "desert"},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537, "tz": "America/Los_Angeles", "bias": 0.4, "afternoon_bump": 0.3, "type": "desert"},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles", "bias": -0.6, "afternoon_bump": 0.0, "type": "marine"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "bias": 0.4, "afternoon_bump": 0.3, "type": "texas"},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "bias": 0.3, "afternoon_bump": 0.2, "type": "texas"},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "bias": 0.3, "afternoon_bump": 0.2, "type": "gulf"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "bias": 0.2, "afternoon_bump": 0.2, "type": "urban_east"},
    "NYC": {"lat": 40.7829, "lon": -73.9654, "tz": "America/New_York", "bias": 0.6, "afternoon_bump": 0.1, "type": "urban_east"},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "bias": 0.2, "afternoon_bump": 0.1, "type": "gulf"},
}

DEFAULT_LADDERS = {
    "Phoenix": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Las Vegas": "73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
    "Los Angeles": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
    "Atlanta": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "NYC": "62 or below | 63-64 | 65-66 | 67-68 | 69-70 | 71 or above",
    "Miami": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
}

BASE_WEIGHTS = {"ICON": 0.35, "OpenMeteo": 0.30, "GFS": 0.20, "NWS": 0.15}

OUTLIER_HALF = 3.0
OUTLIER_REMOVE = 4.5
SPREAD_SAFETY_THRESHOLD = 5.0
SPREAD_SAFETY_MULTIPLIER = 0.4
SIGMA_MIN = 1.20
SIGMA_MAX = 2.00


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
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def load_history():
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "city", "date", "timestamp", "ICON", "OpenMeteo", "GFS", "NWS",
        "actual_high"
    ])


def save_history(df):
    df.to_csv(HISTORY_PATH, index=False)


def auto_log_snapshot(df, city, local_date, forecasts):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    row = {
        "city": city,
        "date": local_date,
        "timestamp": timestamp,
        "ICON": forecasts.get("ICON"),
        "OpenMeteo": forecasts.get("OpenMeteo"),
        "GFS": forecasts.get("GFS"),
        "NWS": forecasts.get("NWS"),
        "actual_high": None,
    }

    # Avoid exact duplicate rows
    mask = (
        (df["city"] == city) &
        (df["date"] == local_date) &
        (df["timestamp"] == timestamp)
    )
    if not mask.any():
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def rolling_source_biases(df, city, window=30):
    # Average (forecast - actual) using rows that have actuals
    city_df = df[(df["city"] == city) & (df["actual_high"].notna())].copy()
    if city_df.empty:
        return {k: 0.0 for k in BASE_WEIGHTS}, 0

    city_df = city_df.tail(window)
    biases = {}
    for src in BASE_WEIGHTS:
        vals = city_df[[src, "actual_high"]].dropna()
        if len(vals) >= 3:
            biases[src] = float((vals[src] - vals["actual_high"]).mean())
        else:
            biases[src] = 0.0
    return biases, len(city_df)


def apply_bias_correction(forecasts, biases):
    corrected = {}
    for src, val in forecasts.items():
        if val is None:
            corrected[src] = None
        else:
            corrected[src] = val - biases.get(src, 0.0)
    return corrected


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
    num = den = 0.0
    for k, v in forecasts.items():
        if v is None:
            continue
        w = weights.get(k, 0)
        num += v * w
        den += w
    return None if den == 0 else num / den


def solar_adjust(cloud, hour, city_name):
    if hour < 9 or hour > 17 or cloud is None:
        return 0.0
    adj = 1.0 if cloud < 10 else 0.6 if cloud < 30 else 0.2 if cloud < 50 else -0.5
    if city_name in {"Phoenix", "Las Vegas"}:
        if cloud < 20:
            adj += 0.4
        elif cloud < 40:
            adj += 0.2
    return adj


def humidity_suppression(city_type, dewpoint, cloud, hour):
    if dewpoint is None or hour < 12:
        return 0.0
    suppress = 0.0
    if city_type in {"gulf", "texas"}:
        if dewpoint >= 68:
            suppress -= 0.8
        elif dewpoint >= 64:
            suppress -= 0.4
    if cloud is not None and cloud >= 70 and hour >= 13:
        suppress -= 0.6
    return suppress


def expected_curve(hour):
    curve = {6:0.55,7:0.60,8:0.65,9:0.70,10:0.75,11:0.80,12:0.85,13:0.90,14:0.95,15:0.98,16:1.00,17:1.00}
    return curve.get(hour, 1.0 if hour > 17 else 0.50)


def raw_trajectory_adjust(current_temp, forecast_high, hour):
    if current_temp is None or forecast_high is None or hour < 8 or hour > 16:
        return 0.0
    diff = current_temp - forecast_high * expected_curve(hour)
    return max(min(diff * 0.35, 2.0), -2.0)


def apply_spread_safety_valve(traj, spread):
    return traj * SPREAD_SAFETY_MULTIPLIER if spread > SPREAD_SAFETY_THRESHOLD else traj


def late_day_cap(city_type, cloud, hour, current_temp, projected_high):
    if current_temp is None or projected_high is None:
        return projected_high, 0.0
    max_remaining = None
    if hour >= 13:
        if city_type == "desert":
            max_remaining = 6.0 if (cloud is None or cloud < 40) else 5.0
        elif city_type == "marine":
            max_remaining = 4.0 if (cloud is None or cloud < 50) else 3.0
        elif city_type in {"texas", "gulf"}:
            max_remaining = 4.0 if (cloud is None or cloud < 60) else 3.0
        else:
            max_remaining = 5.0 if (cloud is None or cloud < 50) else 4.0
    if max_remaining is None:
        return projected_high, 0.0
    capped = min(projected_high, current_temp + max_remaining)
    return capped, projected_high - capped


def enforce_current_temp_floor(projected_high, current_temp):
    if projected_high is None or current_temp is None:
        return projected_high, 0.0
    floored = max(projected_high, current_temp)
    return floored, floored - projected_high


def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def parse_ladder(text):
    out = []
    for p in [q.strip() for q in text.split("|") if q.strip()]:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        lower = p.lower()
        if "below" in lower and nums:
            out.append((p, None, nums[0]))
        elif "above" in lower and nums:
            out.append((p, nums[0], None))
        elif len(nums) >= 2:
            out.append((p, nums[0], nums[1]))
    return out


def bracket_probs(mu, sigma, brackets, current_temp=None):
    rows = []
    for lab, lo, hi in brackets:
        if current_temp is not None and hi is not None and current_temp > hi:
            rows.append((lab, 0.0))
            continue
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((lab, max(p, 0.0)))
    total = sum(p for _, p in rows)
    if total > 0:
        rows = [(lab, p / total) for lab, p in rows]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def no_bet_label(spread, sigma, cloud, current_temp, cons, city_type):
    if cons is None or current_temp is None:
        return "Pass"
    remaining = cons - current_temp
    if spread > 6 or sigma >= 2.0:
        return "Pass"
    if city_type in {"gulf", "texas"} and cloud is not None and cloud >= 70 and remaining > 5:
        return "Pass"
    if spread > 4:
        return "Playable"
    return "Strong"


city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
lat = profile["lat"]
lon = profile["lon"]
tz = profile["tz"]
local_now = datetime.now(ZoneInfo(tz))
local_hour = local_now.hour
local_date = local_now.strftime("%Y-%m-%d")

ladder_text = st.text_input("Kalshi ladder", DEFAULT_LADDERS[city])

# Weather pulls
openmeteo = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
    "hourly": "dew_point_2m", "current": "temperature_2m,cloud_cover,dew_point_2m",
    "temperature_unit": "fahrenheit", "timezone": "auto"
})
gfs = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
    "models": "gfs_seamless", "temperature_unit": "fahrenheit", "timezone": "auto"
})
icon = safe_get("https://api.open-meteo.com/v1/forecast", {
    "latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
    "models": "icon_seamless", "temperature_unit": "fahrenheit", "timezone": "auto"
})

nws = safe_get(f"https://api.weather.gov/points/{lat},{lon}")
nws_high = None
if nws and "properties" in nws and nws["properties"].get("forecast"):
    fc = safe_get(nws["properties"]["forecast"])
    if fc and "properties" in fc and "periods" in fc["properties"]:
        for per in fc["properties"]["periods"]:
            if per.get("isDaytime"):
                nws_high = per.get("temperature")
                break

raw_forecasts = {
    "ICON": icon["daily"]["temperature_2m_max"][0] if icon and "daily" in icon else None,
    "OpenMeteo": openmeteo["daily"]["temperature_2m_max"][0] if openmeteo and "daily" in openmeteo else None,
    "GFS": gfs["daily"]["temperature_2m_max"][0] if gfs and "daily" in gfs else None,
    "NWS": nws_high,
}

cloud = current_temp = dewpoint = None
if openmeteo and "current" in openmeteo:
    cloud = openmeteo["current"].get("cloud_cover")
    current_temp = openmeteo["current"].get("temperature_2m")
    dewpoint = openmeteo["current"].get("dew_point_2m")

# Load / update history
hist = load_history()
hist = auto_log_snapshot(hist, city, local_date, raw_forecasts)
save_history(hist)

biases, history_count = rolling_source_biases(hist, city, window=30)
corrected_forecasts = apply_bias_correction(raw_forecasts, biases)

weights = compute_weights(corrected_forecasts)
cons = consensus(corrected_forecasts, weights)

vals = [v for v in corrected_forecasts.values() if v is not None]
spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
sigma = min(max(1.3 + spread * 0.25, SIGMA_MIN), SIGMA_MAX)

raw_traj = traj = late_cap_reduction = floor_raise = 0.0
humidity_adj = 0.0

if cons is not None:
    cons += profile["bias"]
    cons += solar_adjust(cloud, local_hour, city)
    humidity_adj = humidity_suppression(profile["type"], dewpoint, cloud, local_hour)
    cons += humidity_adj
    raw_traj = raw_trajectory_adjust(current_temp, cons, local_hour)
    traj = apply_spread_safety_valve(raw_traj, spread)
    cons += traj
    if local_hour >= 13:
        cons += profile.get("afternoon_bump", 0.0)
    cons, late_cap_reduction = late_day_cap(profile["type"], cloud, local_hour, current_temp, cons)
    cons, floor_raise = enforce_current_temp_floor(cons, current_temp)

grade = no_bet_label(spread, sigma, cloud, current_temp, cons, profile["type"])

# UI
st.subheader("Raw Forecast Sources")
st.write(raw_forecasts)
st.subheader("Rolling Source Biases")
st.write({k: round(v, 2) for k, v in biases.items()})
st.subheader("Bias-Corrected Sources")
st.write(corrected_forecasts)
st.subheader("History Rows With Actual")
st.write(history_count)

st.subheader("Consensus High")
st.write(round(cons, 2) if cons is not None else "N/A")

st.subheader("Current Station Temp")
st.write(current_temp)
st.subheader("Cloud Cover")
st.write(cloud)
st.subheader("Dew Point")
st.write(dewpoint)

st.subheader("Humidity Suppression")
st.write(round(humidity_adj, 2) if cons is not None else "N/A")
st.subheader("Raw Trajectory Adjustment")
st.write(round(raw_traj, 2) if cons is not None else "N/A")
st.subheader("Trajectory Adjustment")
st.write(round(traj, 2) if cons is not None else "N/A")
st.subheader("Late-Day Cap Reduction")
st.write(round(late_cap_reduction, 2) if cons is not None else "N/A")
st.subheader("Current Temp Floor Raise")
st.write(round(floor_raise, 2) if cons is not None else "N/A")
st.subheader("Forecast Spread")
st.write(round(spread, 2))
st.subheader("Sigma")
st.write(round(sigma, 2))
st.subheader("Trade Grade")
st.write(grade)

if cons is not None:
    brackets = parse_ladder(ladder_text)
    rows = bracket_probs(cons, sigma, brackets, current_temp=current_temp)
    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability"] = df["Model Probability"].apply(lambda x: f"{x*100:.1f}%")
    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("Enter Final Actual High For Bias Learning")
with st.form("actual_high_form"):
    date_to_update = st.text_input("Date (YYYY-MM-DD)", value=local_date)
    actual_high_input = st.number_input("Actual High", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    submitted = st.form_submit_button("Save Actual High")
    if submitted:
        mask = (hist["city"] == city) & (hist["date"] == date_to_update)
        if mask.any():
            hist.loc[mask, "actual_high"] = actual_high_input
        else:
            new_row = {
                "city": city, "date": date_to_update, "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                "ICON": None, "OpenMeteo": None, "GFS": None, "NWS": None,
                "actual_high": actual_high_input
            }
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        save_history(hist)
        st.success(f"Saved actual high {actual_high_input} for {city} on {date_to_update}")

st.caption("Stable Adaptive â bias correction + temp floor + late-day cap + humidity suppression + no-bet filter")
