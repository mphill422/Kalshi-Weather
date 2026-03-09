# Kalshi High Temperature Model - Final Stable
# Focused on 7 cities for daily high-temperature betting

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model - Final Stable", layout="wide")
st.title("Kalshi High Temperature Model - Final Stable")

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "bias": 0.4, "city_type": "desert"},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537, "tz": "America/Los_Angeles", "bias": 0.3, "city_type": "desert"},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles", "bias": -0.5, "city_type": "marine"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "bias": 0.3, "city_type": "texas"},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "bias": 0.2, "city_type": "texas"},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "bias": 0.1, "city_type": "gulf"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "bias": 0.2, "city_type": "urban_east"},
}

DEFAULT_LADDERS = {
    "Phoenix": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Las Vegas": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Los Angeles": "63 or below | 64-65 | 66-67 | 68-69 | 70-71 | 72 or above",
    "Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
    "Atlanta": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
}

BASE_WEIGHTS = {"OpenMeteo": 0.35, "GFS": 0.25, "ICON": 0.25, "NWS": 0.15}
OUTLIER_HALF = 3.0
OUTLIER_REMOVE = 4.5
SIGMA_MIN = 1.15
SIGMA_MAX = 1.95


def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=12)
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


def compute_weights(forecasts):
    vals = [v for v in forecasts.values() if v is not None]
    med = median(vals)
    if med is None:
        return {k: 0.0 for k in forecasts}

    out = {}
    for k, v in forecasts.items():
        if v is None:
            out[k] = 0.0
            continue
        d = abs(v - med)
        w = BASE_WEIGHTS.get(k, 0.0)
        if d > OUTLIER_REMOVE:
            w = 0.0
        elif d > OUTLIER_HALF:
            w *= 0.5
        out[k] = w
    return out


def consensus(forecasts, weights):
    num = 0.0
    den = 0.0
    for k, v in forecasts.items():
        if v is None:
            continue
        w = weights.get(k, 0.0)
        num += v * w
        den += w
    return None if den == 0 else num / den


def solar_adjust(cloud, hour, city_type):
    if cloud is None or hour < 9 or hour > 17:
        return 0.0
    if city_type == "desert":
        if cloud < 10:
            return 1.2
        if cloud < 25:
            return 0.8
        if cloud < 50:
            return 0.3
        return -0.5
    if city_type == "marine":
        if cloud < 20:
            return 0.4
        if cloud < 50:
            return 0.1
        return -0.6
    if city_type in {"texas", "gulf"}:
        if cloud < 15:
            return 0.7
        if cloud < 40:
            return 0.3
        if cloud < 70:
            return -0.1
        return -0.7
    if cloud < 20:
        return 0.6
    if cloud < 50:
        return 0.2
    return -0.4


def humidity_suppression(city_type, dewpoint, hour):
    if dewpoint is None or hour < 12:
        return 0.0
    if city_type == "gulf":
        if dewpoint >= 70:
            return -0.5
        if dewpoint >= 66:
            return -0.25
    if city_type == "texas":
        if dewpoint >= 68:
            return -0.3
        if dewpoint >= 64:
            return -0.15
    return 0.0


def max_remaining_rise(city_type, cloud, dewpoint, hour):
    if hour >= 16:
        base = 2.0
    elif hour >= 15:
        base = 3.0
    elif hour >= 14:
        base = 4.0
    elif hour >= 13:
        base = 5.0
    else:
        return None

    if city_type == "desert":
        base += 2.0
    elif city_type == "marine":
        base -= 1.0
    elif city_type == "gulf":
        base -= 0.5

    if cloud is not None:
        if cloud >= 80:
            base -= 2.0
        elif cloud >= 60:
            base -= 1.0

    if dewpoint is not None and city_type in {"texas", "gulf"} and dewpoint >= 68:
        base -= 0.5

    return max(base, 1.0)


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


def implied_prob_from_american(odds):
    try:
        odds = int(odds)
    except Exception:
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def trade_grade(spread, sigma, current_temp, projected_high, cloud, city_type):
    if projected_high is None or current_temp is None:
        return "Pass"
    remaining = projected_high - current_temp
    if spread >= 6.0 or sigma >= 1.95:
        return "Pass"
    if city_type in {"texas", "gulf"} and cloud is not None and cloud >= 75 and remaining >= 6:
        return "Pass"
    if remaining <= 0.5:
        return "Playable"
    if spread <= 3.5 and sigma <= 1.6:
        return "Strong"
    return "Playable"


city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
lat = profile["lat"]
lon = profile["lon"]
tz = profile["tz"]
city_type = profile["city_type"]
local_hour = datetime.now(ZoneInfo(tz)).hour

ladder_text = st.text_input("Kalshi Ladder", DEFAULT_LADDERS[city])

weather = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,cloud_cover,dew_point_2m",
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    },
)

gfs = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "models": "gfs_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    },
)

icon = safe_get(
    "https://api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "models": "icon_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    },
)

nws_point = safe_get(f"https://api.weather.gov/points/{lat},{lon}")
nws_high = None
if nws_point and "properties" in nws_point and nws_point["properties"].get("forecast"):
    fc = safe_get(nws_point["properties"]["forecast"])
    if fc and "properties" in fc and "periods" in fc["properties"]:
        for period in fc["properties"]["periods"]:
            if period.get("isDaytime"):
                nws_high = period.get("temperature")
                break

if weather:
    current_temp = weather["current"].get("temperature_2m")
    cloud = weather["current"].get("cloud_cover")
    dew = weather["current"].get("dew_point_2m")

    forecasts = {
        "OpenMeteo": weather["daily"]["temperature_2m_max"][0] if "daily" in weather else None,
        "GFS": gfs["daily"]["temperature_2m_max"][0] if gfs and "daily" in gfs else None,
        "ICON": icon["daily"]["temperature_2m_max"][0] if icon and "daily" in icon else None,
        "NWS": nws_high,
    }

    weights = compute_weights(forecasts)
    cons = consensus(forecasts, weights)

    spread_vals = [v for v in forecasts.values() if v is not None]
    spread = (max(spread_vals) - min(spread_vals)) if len(spread_vals) >= 2 else 0.0
    sigma = min(max(1.25 + spread * 0.20, SIGMA_MIN), SIGMA_MAX)

    projected_high = cons
    if projected_high is not None:
        projected_high += profile["bias"]
        projected_high += solar_adjust(cloud, local_hour, city_type)
        projected_high += humidity_suppression(city_type, dew, local_hour)

        rem = max_remaining_rise(city_type, cloud, dew, local_hour)
        if rem is not None and current_temp is not None:
            projected_high = min(projected_high, current_temp + rem)

        if current_temp is not None:
            projected_high = max(projected_high, current_temp)

    grade = trade_grade(spread, sigma, current_temp, projected_high, cloud, city_type)

    st.subheader("Forecast Sources")
    st.write(forecasts)

    st.subheader("Weights")
    st.write(weights)

    st.subheader("Projected High")
    st.write(round(projected_high, 2) if projected_high is not None else "N/A")

    st.subheader("Current Temp")
    st.write(current_temp)

    st.subheader("Cloud Cover")
    st.write(cloud)

    st.subheader("Dew Point")
    st.write(dew)

    st.subheader("Forecast Spread")
    st.write(round(spread, 2))

    st.subheader("Sigma")
    st.write(round(sigma, 2))

    st.subheader("Trade Grade")
    st.write(grade)

    brackets = parse_ladder(ladder_text)
    rows = bracket_probs(projected_high, sigma, brackets, current_temp=current_temp)
    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability"] = df["Model Probability"].apply(lambda x: f"{x*100:.1f}%")
    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Market Comparison")
    st.caption("Optional: paste Kalshi YES odds to compare model vs market.")
    market_rows = []
    for bracket, prob in rows:
        odds = st.text_input(f"{bracket} YES odds", value="", key=f"odds_{bracket}")
        imp = implied_prob_from_american(odds) if odds.strip() else None
        market_rows.append({
            "Bracket": bracket,
            "Model %": round(prob * 100, 1),
            "Market %": round(imp * 100, 1) if imp is not None else None,
            "Edge %": round((prob - imp) * 100, 1) if imp is not None else None,
        })
    mdf = pd.DataFrame(market_rows)
    st.dataframe(mdf, use_container_width=True)

else:
    st.error("Weather data unavailable")
