import math
import datetime as dt
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

# --- Cities (lat/lon) ---
CITY_COORDS = {
    "Austin": (30.2672, -97.7431),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Las Vegas": (36.1699, -115.1398),
}

st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox("Select City", list(CITY_COORDS.keys()), index=0)

# --- Kalshi helper inputs ---
st.subheader("Kalshi range helper (enter the contract range)")
colA, colB = st.columns(2)
with colA:
    low = st.number_input("Range low (°F)", value=80, step=1)
with colB:
    high = st.number_input("Range high (°F)", value=84, step=1)

buffer = st.slider("Safety buffer (°F)", 0.0, 5.0, 1.0, 0.5)

# --- Data fetch (Open-Meteo: reliable + no API key) ---
@st.cache_data(ttl=300)
def fetch_open_meteo(lat: float, lon: float):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&temperature_unit=fahrenheit"
        "&timezone=auto"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

lat, lon = CITY_COORDS[city]

try:
    data = fetch_open_meteo(lat, lon)

    # Hourly temps
    hourly_times = pd.to_datetime(data["hourly"]["time"])
    hourly_temps = pd.Series(data["hourly"]["temperature_2m"], index=hourly_times)

    # Daily highs/lows
    daily_dates = pd.to_datetime(data["daily"]["time"]).dt.date
    daily_highs = pd.Series(data["daily"]["temperature_2m_max"], index=daily_dates)
    daily_lows = pd.Series(data["daily"]["temperature_2m_min"], index=daily_dates)

    today = dt.date.today()

    # Today's hourly slice (local timezone already applied by API)
    today_mask = hourly_temps.index.date == today
    today_hourly = hourly_temps[today_mask]

    st.subheader(f"{city} — Forecast snapshot")

    c1, c2, c3 = st.columns(3)

    # Today's forecast high (daily)
    today_high = float(daily_highs.get(today, float("nan")))
    today_low = float(daily_lows.get(today, float("nan")))

    if not math.isnan(today_high):
        c1.metric("Today forecast high (°F)", f"{today_high:.1f}")
    else:
        c1.metric("Today forecast high (°F)", "N/A")

    if not math.isnan(today_low):
        c2.metric("Today forecast low (°F)", f"{today_low:.1f}")
    else:
        c2.metric("Today forecast low (°F)", "N/A")

    # Peak hour estimate from hourly temps (today)
    if len(today_hourly) > 0:
        peak_time = today_hourly.idxmax()
        peak_temp = float(today_hourly.max())
        c3.metric("Peak hour est.", f"{peak_time.strftime('%-I:%M %p')}", f"{peak_temp:.1f}°F")
    else:
        c3.metric("Peak hour est.", "N/A")

    # Plot hourly temps today
    if len(today_hourly) > 0:
        st.caption("Hourly temperature (today)")
        st.line_chart(today_hourly.rename("Temp (°F)"))
    else:
        st.info("No hourly data available for today yet.")

    # --- Range decision helper ---
    st.subheader("Range check")
    if not math.isnan(today_high):
        in_range = (low <= today_high <= high)
        if in_range:
            st.success(f"Forecast high {today_high:.1f}°F is INSIDE [{low}, {high}].")
        else:
            st.error(f"Forecast high {today_high:.1f}°F is OUTSIDE [{low}, {high}].")

        # "Safer" version using buffer (avoid 0.1°F heartbreak)
        safe_low = low + buffer
        safe_high = high - buffer
        st.caption(f"With buffer {buffer:.1f}°F, the ‘safer’ interior is [{safe_low:.1f}, {safe_high:.1f}].")

        if safe_low <= today_high <= safe_high:
            st.success("✅ Inside the safer interior (less sweat).")
        else:
            st.warning("⚠️ Near an edge (more risk of a late-model swing).")

    # --- Quick checklist for betting timing ---
    st.subheader("Quick timing checklist (simple)")
    st.write(
        "- Check this page **early morning**, then again **2–4 hours before the expected peak hour**.\n"
        "- If today’s forecast high is sitting **within ~1°F of a boundary**, treat it as a coin flip.\n"
        "- Peak-hour estimate helps you know when the day will ‘show its hand’."
    )

    st.caption("Data source: Open-Meteo (forecast).")

except Exception as e:
    st.error("Could not fetch weather data right now.")
    st.caption(f"Error: {type(e).__name__}")
