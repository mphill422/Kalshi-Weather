# Kalshi Temperature Model v10.4
# Calibrated version for temperature markets

import streamlit as st
import requests
import pandas as pd
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Kalshi Temperature Model v10.4", layout="wide")

st.title("Kalshi Temperature Model v10.4")

# -----------------------------
# City configuration
# -----------------------------

CITIES = {
    "Phoenix": {
        "lat": 33.4342,
        "lon": -112.0116,
        "station": "KPHX",
        "sigma": 1.10,
        "bias": 0.60,
        "prob_filter": 0.55
    },
    "Las Vegas": {
        "lat": 36.0801,
        "lon": -115.1522,
        "station": "KLAS",
        "sigma": 1.00,
        "bias": 0.00,
        "prob_filter": 0.55
    },
    "Los Angeles": {
        "lat": 33.9425,
        "lon": -118.4081,
        "station": "KLAX",
        "sigma": 1.10,
        "bias": -0.40,
        "prob_filter": 0.58
    },
    "Dallas": {
        "lat": 32.8998,
        "lon": -97.0403,
        "station": "KDFW",
        "sigma": 1.00,
        "bias": 0.15,
        "prob_filter": 0.58
    },
    "Austin": {
        "lat": 30.1945,
        "lon": -97.6699,
        "station": "KAUS",
        "sigma": 1.10,
        "bias": 0.20,
        "prob_filter": 0.58
    },
    "Houston": {
        "lat": 29.9902,
        "lon": -95.3368,
        "station": "KIAH",
        "sigma": 1.50,
        "bias": 0.30,
        "prob_filter": 0.62
    },
}

# -----------------------------
# Probability functions
# -----------------------------

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def bracket_probability(low, high, mu, sigma):
    return normal_cdf(high + 0.5, mu, sigma) - normal_cdf(low - 0.5, mu, sigma)

# -----------------------------
# Weather API
# -----------------------------

def get_nws_forecast(lat, lon):

    url = f"https://api.weather.gov/points/{lat},{lon}"
    r = requests.get(url).json()

    forecast_url = r["properties"]["forecast"]
    hourly_url = r["properties"]["forecastHourly"]

    forecast = requests.get(forecast_url).json()
    hourly = requests.get(hourly_url).json()

    daily_high = None

    for p in forecast["properties"]["periods"]:
        if p["isDaytime"]:
            daily_high = p["temperature"]
            break

    hourly_high = max(
        p["temperature"] for p in hourly["properties"]["periods"][:24]
    )

    return daily_high, hourly_high

# -----------------------------
# Model logic
# -----------------------------

city = st.selectbox("City", list(CITIES.keys()))

profile = CITIES[city]

daily_high, hourly_high = get_nws_forecast(
    profile["lat"],
    profile["lon"]
)

sources = []

if daily_high:
    sources.append(daily_high)

if hourly_high:
    sources.append(hourly_high)

consensus = sum(sources) / len(sources)

# Apply bias
consensus += profile["bias"]

spread = max(sources) - min(sources) if len(sources) > 1 else 0

sigma = profile["sigma"] + spread / 3

st.subheader("Forecast Inputs")

df = pd.DataFrame({
    "Source": ["NWS Forecast", "NWS Hourly"],
    "High": [daily_high, hourly_high]
})

st.dataframe(df)

# -----------------------------
# Model metrics
# -----------------------------

c1, c2, c3 = st.columns(3)

c1.metric("Consensus High", round(consensus,1))
c2.metric("Spread", spread)
c3.metric("Sigma", round(sigma,2))

# -----------------------------
# Generate brackets
# -----------------------------

center = round(consensus)

brackets = [
    (center-3, center-2),
    (center-1, center),
    (center+1, center+2),
]

rows = []

for lo, hi in brackets:

    p = bracket_probability(lo, hi, consensus, sigma)

    rows.append({
        "Bracket": f"{lo}-{hi}",
        "Win Probability": round(p*100,1)
    })

table = pd.DataFrame(rows).sort_values("Win Probability", ascending=False)

st.subheader("Model Bracket Probabilities")

st.dataframe(table)

top = table.iloc[0]

if top["Win Probability"]/100 >= profile["prob_filter"]:
    st.success(f"BET SIGNAL: {top['Bracket']}")
else:
    st.error("PASS")
