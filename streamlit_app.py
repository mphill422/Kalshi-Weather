# Kalshi Temperature Model - Stable Adaptive v2
# Complete version with corrected Kalshi ladders (Vegas fix included)

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model - Stable Adaptive v2", layout="wide")
st.title("Kalshi Temperature Model - Stable Adaptive v2")

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "bias": 0.5},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537, "tz": "America/Los_Angeles", "bias": 0.4},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles", "bias": -0.6},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "bias": 0.4},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "bias": 0.3},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "bias": 0.3},
}

DEFAULT_LADDERS = {
    "Phoenix": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Las Vegas": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Los Angeles": "63 or below | 64-65 | 66-67 | 68-69 | 70-71 | 72 or above",
    "Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
}

SIGMA_MIN = 1.2
SIGMA_MAX = 2.0

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def parse_ladder(text):
    out = []
    for p in [q.strip() for q in text.split("|") if q.strip()]:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        lower = p.lower()
        if "below" in lower:
            out.append((p, None, nums[0]))
        elif "above" in lower:
            out.append((p, nums[0], None))
        else:
            out.append((p, nums[0], nums[1]))
    return out

def bracket_probs(mu, sigma, brackets):
    rows = []
    for lab, lo, hi in brackets:
        if lo is None:
            p = normal_cdf(hi + 0.5, mu, sigma)
        elif hi is None:
            p = 1 - normal_cdf(lo - 0.5, mu, sigma)
        else:
            p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
        rows.append((lab, max(p, 0)))
    total = sum(p for _, p in rows)
    rows = [(l, p / total) for l, p in rows]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]

lat = profile["lat"]
lon = profile["lon"]
tz = profile["tz"]

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

if weather:

    current_temp = weather["current"]["temperature_2m"]
    cloud = weather["current"]["cloud_cover"]
    dew = weather["current"]["dew_point_2m"]

    forecast_high = weather["daily"]["temperature_2m_max"][0]

    consensus = forecast_high + profile["bias"]

    if dew > 65:
        consensus -= 0.6

    hour = datetime.now(ZoneInfo(tz)).hour

    if hour >= 13:
        consensus = min(consensus, current_temp + 6)

    consensus = max(consensus, current_temp)

    spread = 3
    sigma = min(max(1.3 + spread * 0.25, SIGMA_MIN), SIGMA_MAX)

    st.subheader("Projected High")
    st.write(round(consensus, 2))

    st.subheader("Current Temp")
    st.write(current_temp)

    st.subheader("Cloud Cover")
    st.write(cloud)

    st.subheader("Dew Point")
    st.write(dew)

    brackets = parse_ladder(ladder_text)
    rows = bracket_probs(consensus, sigma, brackets)

    df = pd.DataFrame(rows, columns=["Bracket", "Model Probability"])
    df["Model Probability"] = df["Model Probability"].apply(lambda x: f"{x*100:.1f}%")

    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df, use_container_width=True)

else:
    st.error("Weather data unavailable")
