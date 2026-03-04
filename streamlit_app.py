import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

st.title("Kalshi Weather Trading Dashboard")

# Cities you care about (add/remove freely)
CITIES = {
    "Austin": (30.2672, -97.7431),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Las Vegas": (36.1699, -115.1398),
    "Los Angeles": (34.0522, -118.2437),
}

city = st.selectbox("Select City", list(CITIES.keys()))
lat, lon = CITIES[city]

@st.cache_data(ttl=300)
def fetch_open_meteo(lat: float, lon: float) -> dict:
    # Open-Meteo: free, no API key, usually works well on Streamlit Cloud
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "forecast_days": 3,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

st.caption("Source: Open-Meteo (no key)")

try:
    data = fetch_open_meteo(lat, lon)

    tz_name = data.get("timezone", "UTC")
    tz = ZoneInfo(tz_name)

    current = data["current"]
    temp = float(current["temperature_2m"])
    feels = float(current["apparent_temperature"])
    hum = float(current["relative_humidity_2m"])
    wind = float(current["wind_speed_10m"])

    c1, c2 = st.columns(2)
    c1.metric("Current Temperature (°F)", f"{temp:.1f}")
    c2.metric("Feels Like (°F)", f"{feels:.1f}")

    c3, c4 = st.columns(2)
    c3.metric("Humidity (%)", f"{hum:.0f}")
    c4.metric("Wind (mph)", f"{wind:.1f}")

    # Hourly table
    hourly = data["hourly"]
    times = pd.to_datetime(hourly["time"]).dt.tz_localize(None)
    # Open-Meteo returns temps in °C by default unless configured; however many users see °F
    # We'll detect by typical range and convert if needed.
    temps = pd.Series(hourly["temperature_2m"], dtype="float")

    # Heuristic: if temps look like Celsius (e.g., 0-45), convert to Fahrenheit
    if temps.max() < 60:
        temps_f = temps * 9 / 5 + 32
    else:
        temps_f = temps

    df = pd.DataFrame({
        "time_local": times,
        "temp_f": temps_f,
        "humidity": pd.Series(hourly["relative_humidity_2m"], dtype="float"),
        "feels_like_f": pd.Series(hourly["apparent_temperature"], dtype="float"),
        "wind_mph": pd.Series(hourly["wind_speed_10m"], dtype="float"),
    })

    # If apparent_temperature is also Celsius, convert similarly
    if df["feels_like_f"].max() < 60:
        df["feels_like_f"] = df["feels_like_f"] * 9 / 5 + 32

    # Use local "today" based on timezone
    now_local = datetime.now(tz).replace(tzinfo=None)
    today = now_local.date()

    # Filter next 48 hours for chart/table
    df_next = df[df["time_local"] >= now_local].head(48).copy()

    # Estimate today's peak temp + peak time using hourly temps for "today"
    df_today = df[df["time_local"].dt.date == today].copy()
    if not df_today.empty:
        peak_row = df_today.loc[df_today["temp_f"].idxmax()]
        peak_temp = float(peak_row["temp_f"])
        peak_time = peak_row["time_local"]
    else:
        peak_temp = None
        peak_time = None

    # Daily forecast highs/lows (Open-Meteo daily)
    d = data["daily"]
    daily_dates = pd.to_datetime(d["time"]).dt.date
    daily_max = pd.Series(d["temperature_2m_max"], dtype="float")
    daily_min = pd.Series(d["temperature_2m_min"], dtype="float")

    # Convert daily if Celsius
    if daily_max.max() < 60:
        daily_max = daily_max * 9 / 5 + 32
        daily_min = daily_min * 9 / 5 + 32

    daily_df = pd.DataFrame({
        "date": daily_dates,
        "forecast_high_f": daily_max.round(1),
        "forecast_low_f": daily_min.round(1),
    })

    st.divider()

    st.subheader("Trading helpers (estimate)")
    if peak_temp is not None:
        st.write(f"**Estimated peak temp today:** **{peak_temp:.1f}°F** around **{peak_time.strftime('%I:%M %p')}** ({tz_name})")
    else:
        st.write("Could not compute today's peak from hourly data.")

    st.write("**3-day forecast (high/low):**")
    st.dataframe(daily_df, use_container_width=True, hide_index=True)

    st.subheader("Next 48 hours (hourly)")
    st.line_chart(df_next.set_index("time_local")[["temp_f"]])

    with st.expander("Show hourly table"):
        show = df_next.copy()
        show["time_local"] = show["time_local"].dt.strftime("%a %I:%M %p")
        show = show.rename(columns={
            "time_local": "Time",
            "temp_f": "Temp (°F)",
            "feels_like_f": "Feels (°F)",
            "humidity": "Humidity (%)",
            "wind_mph": "Wind (mph)",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Could not fetch weather data right now.")
    st.caption("Tip: Click **Manage app** → **Logs** to see full details.")
    with st.expander("Technical details"):
        st.write(repr(e))
