import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import math

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

NWS_HEADERS = {
    "User-Agent": "KalshiWeatherApp contact@example.com"
}

CITIES = {
    "Austin, TX": (30.2672, -97.7431),
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Las Vegas, NV": (36.1699, -115.1398),
    "San Antonio, TX": (29.4241, -98.4936),
    "Los Angeles, CA": (34.0522, -118.2437),
    "New York City, NY": (40.7128, -74.0060),
    "Atlanta, GA": (33.7490, -84.3880),
    "Miami, FL": (25.7617, -80.1918),
    "New Orleans, LA": (29.9511, -90.0715),
}

def open_meteo(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
        "forecast_days": 2
    }

    r = requests.get(url, params=params)
    data = r.json()

    hourly = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp": data["hourly"]["temperature_2m"]
    })

    daily_high = data["daily"]["temperature_2m_max"][0]

    return hourly, daily_high


def nws_forecast(lat, lon):

    point_url = f"https://api.weather.gov/points/{lat},{lon}"
    r = requests.get(point_url, headers=NWS_HEADERS)
    data = r.json()

    hourly_url = data["properties"]["forecastHourly"]

    r2 = requests.get(hourly_url, headers=NWS_HEADERS)
    hourly = r2.json()

    temps = []
    times = []

    for p in hourly["properties"]["periods"][:48]:
        temps.append(p["temperature"])
        times.append(pd.to_datetime(p["startTime"]))

    df = pd.DataFrame({"time": times, "temp": temps})

    today = datetime.now().date()

    df_today = df[df["time"].dt.date == today]

    if df_today.empty:
        return None, None

    peak_row = df_today.loc[df_today["temp"].idxmax()]

    return df, peak_row["temp"]


def kalshi_range(temp, bracket):

    base = math.floor(temp / bracket) * bracket
    low = int(base)
    high = int(base + bracket - 1)

    return low, high


st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox("Select City", list(CITIES.keys()))

bracket = st.selectbox("Kalshi bracket size (°F)", [1,2,5], index=1)

grace = st.slider("Grace Minutes Around Peak",0,120,30)

lat, lon = CITIES[city]

st.caption("Sources: Open-Meteo + National Weather Service")

hourly, om_high = open_meteo(lat,lon)

nws_hourly, nws_high = nws_forecast(lat,lon)

if nws_high:
    predicted_high = (om_high + nws_high) / 2
else:
    predicted_high = om_high

today = datetime.now().date()

today_hours = hourly[hourly["time"].dt.date == today]

peak_row = today_hours.loc[today_hours["temp"].idxmax()]

peak_time = peak_row["time"]

peak_temp = peak_row["temp"]

window_start = peak_time - timedelta(minutes=grace)
window_end = peak_time + timedelta(minutes=grace)

low, high = kalshi_range(predicted_high, bracket)

st.header(city)

col1,col2 = st.columns(2)

col1.metric("Predicted Daily High", f"{predicted_high:.1f}°F")
col2.metric("Estimated Peak Time", peak_time.strftime("%I:%M %p"))

st.write(f"Peak window: {window_start.strftime('%I:%M %p')} – {window_end.strftime('%I:%M %p')}")

st.divider()

st.subheader("Suggested Kalshi Range")

if bracket == 1:
    st.write(f"**{low}°F**")
else:
    st.write(f"**{low}–{high}°F**")

st.write("Nearby ranges to watch:")

if bracket == 1:

    st.write(f"- {low-1}")
    st.write(f"- {low} (current)")
    st.write(f"- {low+1}")

else:

    st.write(f"- {low-bracket}–{high-bracket}")
    st.write(f"- {low}–{high} (current)")
    st.write(f"- {low+bracket}–{high+bracket}")
