import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

USER_AGENT = "kalshi-weather-dashboard/1.0 (contact: example@example.com)"  # NWS requires a UA string

CITIES = {
    # Existing core cities
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Las Vegas, NV": {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles"},

    # New requested cities
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta, GA": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "San Antonio, TX": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}


# -----------------------------
# Helpers
# -----------------------------
def safe_get(url: str, headers=None, params=None, timeout=12):
    """Simple requests.get with basic error handling."""
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        return e


def normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bracket_for_temp(temp_f: float, bracket_size: int):
    """
    Returns (low, high) integers for a bracket.
    bracket_size=2 -> 56–57, 58–59...
    bracket_size=5 -> 55–59, 60–64...
    bracket_size=1 -> 57–57
    """
    if bracket_size <= 1:
        low = int(math.floor(temp_f))
        return low, low

    low = int(math.floor(temp_f / bracket_size) * bracket_size)
    high = low + (bracket_size - 1)
    return low, high


def fmt_range(lo: int, hi: int) -> str:
    return f"{lo}–{hi}°F" if lo != hi else f"{lo}°F"


def parse_iso(dt_str: str) -> datetime:
    # NWS provides ISO strings with offset like 2026-03-04T15:00:00-06:00
    return datetime.fromisoformat(dt_str)


def today_in_tz(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def peak_window_str(peak_dt: datetime, grace_minutes: int) -> str:
    start = peak_dt - pd.Timedelta(minutes=grace_minutes)
    end = peak_dt + pd.Timedelta(minutes=grace_minutes)
    return f"{start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')}"


# -----------------------------
# Data fetchers
# -----------------------------
@st.cache_data(ttl=180)  # cache 3 minutes
def fetch_open_meteo(lat: float, lon: float, tz_name: str, model: str | None = None):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz_name,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
    }
    # Open-Meteo supports model selection via "models" on some endpoints.
    if model:
        params["models"] = model

    r = safe_get(base, params=params)
    if isinstance(r, Exception):
        raise r
    return r.json()


def open_meteo_today_stats(payload: dict, tz_name: str):
    """
    Uses hourly data to compute today's max temp, peak time, and current snapshot.
    """
    tz = ZoneInfo(tz_name)
    today = today_in_tz(tz_name)

    times = payload.get("hourly", {}).get("time", [])
    temps = payload.get("hourly", {}).get("temperature_2m", [])
    hums = payload.get("hourly", {}).get("relative_humidity_2m", [])
    winds = payload.get("hourly", {}).get("windspeed_10m", [])

    if not times or not temps:
        raise ValueError("Open-Meteo hourly data missing.")

    # Current values = last timepoint at/near now (simple approach: use last element)
    current_temp = float(temps[-1]) if temps else None
    current_hum = float(hums[-1]) if hums else None
    current_wind = float(winds[-1]) if winds else None

    best_max = None
    best_dt = None

    for t_str, tf in zip(times, temps):
        dt_local = datetime.fromisoformat(t_str).replace(tzinfo=tz) if "T" in t_str else datetime.fromisoformat(t_str).replace(tzinfo=tz)
        if dt_local.date() != today:
            continue
        tf = float(tf)
        if (best_max is None) or (tf > best_max):
            best_max = tf
            best_dt = dt_local

    if best_max is None or best_dt is None:
        raise ValueError("Open-Meteo did not include today's hours (timezone mismatch?).")

    return {
        "today_high_f": float(best_max),
        "peak_dt": best_dt,
        "current_temp_f": current_temp,
        "humidity_pct": current_hum,
        "wind_mph": current_wind,
    }


@st.cache_data(ttl=300)  # cache 5 minutes
def fetch_nws_hourly(lat: float, lon: float):
    """
    NWS: points -> forecastHourly URL -> periods
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/ld+json, application/json"}
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    r1 = safe_get(points_url, headers=headers)
    if isinstance(r1, Exception):
        raise r1
    points = r1.json()

    hourly_url = points.get("properties", {}).get("forecastHourly")
    if not hourly_url:
        raise ValueError("NWS points response missing forecastHourly URL.")

    r2 = safe_get(hourly_url, headers=headers)
    if isinstance(r2, Exception):
        raise r2
    return r2.json()


def nws_today_stats(payload: dict, tz_name: str):
    tz = ZoneInfo(tz_name)
    today = today_in_tz(tz_name)

    periods = payload.get("properties", {}).get("periods", [])
    if not periods:
        raise ValueError("NWS hourly periods missing.")

    best_max = None
    best_dt = None

    for p in periods:
        start = p.get("startTime")
        temp = p.get("temperature")
        unit = p.get("temperatureUnit", "F")

        if not start or temp is None:
            continue

        dt = parse_iso(start).astimezone(tz)
        if dt.date() != today:
            continue

        tf = float(temp)
        if unit.upper() == "C":
            tf = tf * 9.0 / 5.0 + 32.0

        if (best_max is None) or (tf > best_max):
            best_max = tf
            best_dt = dt

    if best_max is None or best_dt is None:
        raise ValueError("NWS did not include today's hours.")

    return {"today_high_f": float(best_max), "peak_dt": best_dt}


# -----------------------------
# UI
# -----------------------------
st.title("Kalshi Weather Trading Dashboard")

city_name = st.selectbox("Select City", list(CITIES.keys()), index=0)
bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 5], index=1)
grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

city = CITIES[city_name]
lat, lon, tz_name = city["lat"], city["lon"], city["tz"]

st.caption("Sources: Open-Meteo + National Weather Service (NWS) + Open-Meteo GFS")

# -----------------------------
# Fetch + compute
# -----------------------------
with st.spinner("Fetching forecasts…"):
    errors = []
    om_best = om_gfs = nws = None

    # Open-Meteo (best)
    try:
        om_payload = fetch_open_meteo(lat, lon, tz_name, model=None)
        om_best = open_meteo_today_stats(om_payload, tz_name)
    except Exception as e:
        errors.append(f"Open-Meteo (best) failed: {e}")

    # Open-Meteo (GFS)
    try:
        # If 'models=gfs' isn't supported for your plan/endpoint, this may fail—handled gracefully.
        gfs_payload = fetch_open_meteo(lat, lon, tz_name, model="gfs")
        om_gfs = open_meteo_today_stats(gfs_payload, tz_name)
    except Exception as e:
        errors.append(f"Open-Meteo (GFS) failed: {e}")

    # NWS hourly
    try:
        nws_payload = fetch_nws_hourly(lat, lon)
        nws = nws_today_stats(nws_payload, tz_name)
    except Exception as e:
        errors.append(f"NWS failed: {e}")

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for msg in errors:
            st.write("•", msg)

# Collect source highs that exist
sources = []
if om_best:
    sources.append(("Open-Meteo (best)", om_best["today_high_f"], om_best["peak_dt"]))
if om_gfs:
    sources.append(("Open-Meteo (GFS)", om_gfs["today_high_f"], om_gfs["peak_dt"]))
if nws:
    sources.append(("NWS", nws["today_high_f"], nws["peak_dt"]))

if not sources:
    st.error("No forecast sources available right now. Try again in a minute.")
    st.stop()

highs = [s[1] for s in sources]
peak_dts = [s[2] for s in sources]

# Ensemble
ensemble_mean = float(sum(highs) / len(highs))
spread = float(max(highs) - min(highs))

# Confidence buckets
if spread <= 1.0:
    confidence = "High"
elif spread <= 2.0:
    confidence = "Medium"
else:
    confidence = "Low"

# Pick a peak time from the source that has the highest high (ties -> first)
peak_source = max(sources, key=lambda x: x[1])
peak_dt = peak_source[2]

# Brackets
lo, hi = bracket_for_temp(ensemble_mean, int(bracket_size))
prev_lo, prev_hi = bracket_for_temp(lo - 0.1, int(bracket_size))  # previous bracket
next_lo, next_hi = bracket_for_temp(hi + 0.1, int(bracket_size))  # next bracket

# Peak window based on grace minutes
peak_window = peak_window_str(peak_dt, grace_minutes)

# Probability model for value check:
# mean = ensemble_mean
# sigma = max(1.25, spread/2)  (spread is a proxy for uncertainty)
sigma = max(1.25, spread / 2.0)

# Probability that true high falls in [lo, hi] (continuous approx):
# Using 0.5-degree continuity correction:
lower = lo - 0.5
upper = hi + 0.5
p_range = normal_cdf((upper - ensemble_mean) / sigma) - normal_cdf((lower - ensemble_mean) / sigma)

# -----------------------------
# Display
# -----------------------------
st.header(city_name)

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Daily High (°F)", f"{ensemble_mean:.1f}")
with col2:
    st.metric("Confidence", f"{confidence} (spread {spread:.1f}°)")

st.metric("Estimated Peak Time", peak_dt.strftime("%I:%M %p"))
st.write(f"Peak window: **{peak_window}**")

# Current conditions from Open-Meteo best if available
if om_best and om_best.get("current_temp_f") is not None:
    st.subheader("Current Conditions (Open-Meteo)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temp (°F)", f"{om_best['current_temp_f']:.1f}")
    if om_best.get("humidity_pct") is not None:
        c2.metric("Humidity (%)", f"{om_best['humidity_pct']:.0f}")
    if om_best.get("wind_mph") is not None:
        c3.metric("Wind (mph)", f"{om_best['wind_mph']:.1f}")

st.divider()

st.subheader("Suggested Kalshi Range (Daily High)")
st.write(f"**{fmt_range(lo, hi)}**")
st.caption("Tip: watch adjacent brackets because forecasts drift.")

st.write("**Nearby ranges to watch:**")
st.write(f"• {fmt_range(prev_lo, prev_hi)}")
st.write(f"• {fmt_range(lo, hi)} (current)")
st.write(f"• {fmt_range(next_lo, next_hi)}")

st.divider()

st.subheader("Model Agreement (Source Highs)")
df = pd.DataFrame(
    [
        {
            "Source": name,
            "Daily High (°F)": round(high, 1),
            "Peak Time": dt.strftime("%I:%M %p"),
        }
        for (name, high, dt) in sources
    ]
).sort_values("Daily High (°F)", ascending=False)
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Value Bet Check (you enter the Kalshi price)")
st.write(
    f"Model probability for **{fmt_range(lo, hi)}** ≈ **{p_range*100:.1f}%** "
    f"(mean {ensemble_mean:.1f}, σ {sigma:.2f})"
)

price_cents = st.number_input(
    "Kalshi price for the suggested bracket (cents, 0–100)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

implied = price_cents / 100.0
edge = p_range - implied

cA, cB, cC = st.columns(3)
cA.metric("Implied Prob", f"{implied*100:.1f}%")
cB.metric("Model Prob", f"{p_range*100:.1f}%")
cC.metric("Edge", f"{edge*100:+.1f}%")

# Simple value tag
if edge >= 0.08:
    st.success("✅ VALUE BET signal (model probability meaningfully above price).")
elif edge <= -0.08:
    st.error("❌ OVERPRICED (market price above model probability).")
else:
    st.info("Neutral / small edge. Consider watching adjacent brackets or waiting for updates.")

with st.expander("See raw numbers"):
    st.write(f"Forecast date (today): {today_in_tz(tz_name)}")
    st.write(f"Ensemble mean high: {ensemble_mean:.2f}")
    st.write(f"Spread (max-min): {spread:.2f}")
    st.write(f"Sigma used: {sigma:.2f}")
    st.write(f"Suggested bracket: {lo}–{hi}")
    st.write(f"P(bracket): {p_range:.4f}")
