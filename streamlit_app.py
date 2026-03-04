import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

CITY_COORDS = {
    "Austin":  (30.2672, -97.7431),
    "Dallas":  (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
}

st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox("Select City", list(CITY_COORDS.keys()), index=0)
lat, lon = CITY_COORDS[city]

st.caption("Data source: Open-Meteo (no API key).")

@st.cache_data(ttl=300)
def fetch_openmeteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

try:
    data = fetch_openmeteo(lat, lon)

    # Current conditions
    cur = data.get("current", {})
    temp_now = cur.get("temperature_2m")
    hum_now = cur.get("relative_humidity_2m")
    wind_now = cur.get("wind_speed_10m")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Temp (°F)", f"{temp_now:.1f}" if temp_now is not None else "—")
    c2.metric("Humidity (%)", f"{hum_now:.0f}" if hum_now is not None else "—")
    c3.metric("Wind (mph)", f"{wind_now:.1f}" if wind_now is not None else "—")

    # Hourly data → today only
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    hums = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "temp_f": temps,
        "humidity": hums,
        "wind_mph": winds,
    })

    today = date.today()
    df_today = df[df["time"].dt.date == today].copy()

    if df_today.empty:
        st.warning("No hourly data returned for today yet.")
    else:
        # Expected high/low + peak time
        idx_max = df_today["temp_f"].idxmax()
        idx_min = df_today["temp_f"].idxmin()

        high = float(df_today.loc[idx_max, "temp_f"])
        low = float(df_today.loc[idx_min, "temp_f"])
        high_time = df_today.loc[idx_max, "time"].strftime("%-I:%M %p")
        low_time = df_today.loc[idx_min, "time"].strftime("%-I:%M %p")

        st.subheader("Today (from hourly forecast)")
        a1, a2 = st.columns(2)
        a1.metric("Expected High (°F)", f"{high:.1f}", help=f"Peak around {high_time}")
        a2.metric("Expected Low (°F)", f"{low:.1f}", help=f"Low around {low_time}")

        st.caption(f"Peak temp time estimate: **{high_time}** • Low time estimate: **{low_time}**")

        # Table + quick chart
        show = df_today[["time", "temp_f", "humidity", "wind_mph"]].copy()
        show["time"] = show["time"].dt.strftime("%-I:%M %p")
        show = show.rename(columns={
            "time": "Time",
            "temp_f": "Temp (°F)",
            "humidity": "Humidity (%)",
            "wind_mph": "Wind (mph)",
        })

        st.write("Hourly (today)")
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.line_chart(df_today.set_index("time")["temp_f"])

except requests.exceptions.RequestException:
    st.error("Could not fetch weather data right now. Try again or reboot the app (Manage app → Reboot).")
except Exception as e:
    st.error(f"App error: {e}")
