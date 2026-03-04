import requests
import streamlit as st
from datetime import datetime
from math import floor

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

# --- Cities (add/remove here) ---
CITIES = {
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740},
    "Las Vegas, NV": {"lat": 36.1699, "lon": -115.1398},
}

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

def f_to_display(x):
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"

def safe_get(url, params):
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

def suggested_bracket(temp_f, bracket_size):
    """
    Example:
      temp_f=83.2, bracket_size=2 -> 82–83 or 84–85 depending on convention.
    We'll use inclusive ranges like:
      82–83, 84–85, etc. (size 2)
    """
    # Convert to integer-ish for bracketing
    t = float(temp_f)

    if bracket_size == 1:
        lo = floor(t)
        hi = lo
    else:
        lo = int(floor(t / bracket_size) * bracket_size)
        hi = lo + (bracket_size - 1)

    return lo, hi

def get_forecast(lat, lon):
    # Daily high (temperature_2m_max) + hourly temps to estimate peak hour
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": 2,
    }
    return safe_get(OPEN_METEO_BASE, params=params)

def today_peak_hour(hourly_times, hourly_temps, today_str):
    # Find max temp for entries whose date matches today_str (YYYY-MM-DD)
    best_temp = None
    best_time = None
    for t, temp in zip(hourly_times, hourly_temps):
        # t format: "2026-03-04T14:00"
        if t.startswith(today_str):
            if best_temp is None or temp > best_temp:
                best_temp = temp
                best_time = t
    return best_time, best_temp

st.title("Kalshi Weather Trading Dashboard")

city_name = st.selectbox("Select City", list(CITIES.keys()))
bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 5], index=1)

lat = CITIES[city_name]["lat"]
lon = CITIES[city_name]["lon"]

st.caption("Source: Open-Meteo (no key)")

with st.spinner(f"Fetching forecast for {city_name}..."):
    try:
        payload = get_forecast(lat, lon)

        # Daily high for "today" is the first element of daily arrays
        daily = payload.get("daily", {})
        daily_times = daily.get("time", [])
        daily_max = daily.get("temperature_2m_max", [])

        if not daily_times or not daily_max:
            st.error("Could not read daily forecast from Open-Meteo.")
            st.stop()

        today_str = daily_times[0]  # YYYY-MM-DD
        predicted_high = float(daily_max[0])

        # Hourly peak time estimate (today)
        hourly = payload.get("hourly", {})
        hourly_times = hourly.get("time", [])
        hourly_temps = hourly.get("temperature_2m", [])

        peak_time_raw, peak_temp = today_peak_hour(hourly_times, hourly_temps, today_str)

        # Format peak time nicely
        peak_time_display = "—"
        if peak_time_raw:
            # "YYYY-MM-DDTHH:MM"
            dt = datetime.fromisoformat(peak_time_raw)
            peak_time_display = dt.strftime("%-I:%M %p") if hasattr(dt, "strftime") else str(dt)

        # Suggested Kalshi bracket
        lo, hi = suggested_bracket(predicted_high, bracket_size)

        st.subheader(city_name)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Daily High (°F)", f_to_display(predicted_high))
        col2.metric("Estimated Peak Time", peak_time_display)

        st.markdown("---")

        st.subheader("Suggested Kalshi Range (Daily High)")
        if bracket_size == 1:
            st.write(f"**{lo}°F**")
        else:
            st.write(f"**{lo}–{hi}°F**")

        # Add a small “buffer” idea without pretending certainty
        st.caption(
            "Tip: If the market offers adjacent brackets, consider watching the neighboring range(s) "
            "because forecasts can drift."
        )

        # Optional: show “neighbor” brackets for quick decision-making
        st.markdown("**Nearby ranges to watch:**")
        if bracket_size == 1:
            st.write(f"- {lo-1}°F")
            st.write(f"- {lo}°F (current)")
            st.write(f"- {lo+1}°F")
        else:
            prev_lo, prev_hi = lo - bracket_size, (lo - 1)
            next_lo, next_hi = hi + 1, hi + bracket_size
            st.write(f"- {prev_lo}–{prev_hi}")
            st.write(f"- {lo}–{hi} (current)")
            st.write(f"- {next_lo}–{next_hi}")

        st.markdown("---")
        with st.expander("See raw forecast numbers"):
            st.write(f"Forecast date (today): {today_str}")
            st.write(f"Daily max (today): {predicted_high}")
            if peak_temp is not None:
                st.write(f"Peak hour temp (today): {peak_temp} at {peak_time_raw}")

    except Exception as e:
        st.error("Could not fetch weather data right now.")
        st.caption("If this keeps happening: Manage app → Logs to see details.")
        st.caption(f"Debug info: {type(e).__name__}")
