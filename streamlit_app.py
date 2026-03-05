# streamlit_app.py
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

USER_AGENT = "kalshi-weather-dashboard/1.0 (contact: example@example.com)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    tz: str


# Hardcoded cities (NO geocoding = ultra-stable)
CITIES: List[City] = [
    City("Austin, TX", 30.2672, -97.7431, "America/Chicago"),
    City("Dallas, TX", 32.7767, -96.7970, "America/Chicago"),
    City("Houston, TX", 29.7604, -95.3698, "America/Chicago"),
    City("San Antonio, TX", 29.4241, -98.4936, "America/Chicago"),
    City("Phoenix, AZ", 33.4484, -112.0740, "America/Phoenix"),
    City("Los Angeles, CA", 34.0522, -118.2437, "America/Los_Angeles"),
    City("New York City, NY", 40.7128, -74.0060, "America/New_York"),
    City("Atlanta, GA", 33.7490, -84.3880, "America/New_York"),
    City("Miami, FL", 25.7617, -80.1918, "America/New_York"),
    City("New Orleans, LA", 29.9511, -90.0715, "America/Chicago"),
    # Optional extras (won't hurt anything)
    City("Chicago, IL", 41.8781, -87.6298, "America/Chicago"),
]


# =========================
# Helpers (never crash)
# =========================
def now_local(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))


def today_local(tz: str) -> date:
    return now_local(tz).date()


def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 14) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_timeseries_df(times: List[str], values: List[float], tz: str) -> pd.DataFrame:
    """
    Returns df with columns: dt (timezone-aware), value
    """
    if not times or not values or len(times) != len(values):
        return pd.DataFrame(columns=["dt", "value"])
    dt = pd.to_datetime(times, errors="coerce")
    # Open-Meteo returns local times if timezone param set; treat as local-naive -> localize
    # If dt already has tz, convert; else localize.
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(ZoneInfo(tz), ambiguous="NaT", nonexistent="NaT")
    else:
        dt = dt.dt.tz_convert(ZoneInfo(tz))
    df = pd.DataFrame({"dt": dt, "value": pd.to_numeric(values, errors="coerce")})
    df = df.dropna(subset=["dt", "value"])
    return df


def filter_to_local_today(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df.empty:
        return df
    td = today_local(tz)
    return df[df["dt"].dt.date == td].copy()


def fmt_time(dt: Optional[pd.Timestamp], tz: str) -> str:
    if dt is None or pd.isna(dt):
        return "—"
    try:
        return dt.tz_convert(ZoneInfo(tz)).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "—"


def bracket_label(lo: int, size: int) -> str:
    hi = lo + size - 1
    return f"{lo}–{hi}"


def compute_bracket_even_start(x: float, size: int) -> Tuple[int, int]:
    """
    Even-start alignment example for size=2: 78–79, 80–81, 82–83...
    i.e. brackets start at even numbers.
    """
    lo = int(math.floor(x / size) * size)
    if size == 2 and lo % 2 != 0:
        lo -= 1
    return lo, lo + size - 1


def compute_bracket_odd_start(x: float, size: int) -> Tuple[int, int]:
    """
    Odd-start alignment example for size=2: 79–80, 81–82, 83–84...
    i.e. brackets start at odd numbers.
    """
    lo = int(math.floor(x / size) * size)
    if size == 2 and lo % 2 == 0:
        lo -= 1
    return lo, lo + size - 1


def confidence_from_spread(spread: float) -> str:
    if spread <= 1.0:
        return f"High (spread {spread:.1f}°)"
    if spread <= 2.0:
        return f"Medium (spread {spread:.1f}°)"
    return f"Low (spread {spread:.1f}°)"


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def prob_between(lo: float, hi: float, mu: float, sigma: float) -> float:
    # inclusive bracket; approximate using half-degree padding
    a = lo - 0.5
    b = hi + 0.5
    return max(0.0, min(1.0, normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)))


# =========================
# Data sources
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_meteo_best(city: City) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float], Optional[str]]:
    """
    Open-Meteo "best" (forecast endpoint).
    Returns: (hourly_today_df, daily_max_today, current_temp, error)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "timezone": city.tz,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
        "current_weather": "true",
    }
    payload, err = safe_get_json(url, params=params, timeout=14)
    if payload is None:
        return None, None, None, f"Open-Meteo (best) failed: {err}"

    hourly = payload.get("hourly") or {}
    h_times = hourly.get("time") or []
    h_temps = hourly.get("temperature_2m") or []
    hourly_df = parse_timeseries_df(h_times, h_temps, city.tz).rename(columns={"value": "temp_f"})
    hourly_df = filter_to_local_today(hourly_df, city.tz)

    daily = payload.get("daily") or {}
    d_times = daily.get("time") or []
    d_max = daily.get("temperature_2m_max") or []
    daily_max_today = None
    if d_times and d_max and len(d_times) == len(d_max):
        ddf = parse_timeseries_df(d_times, d_max, city.tz).rename(columns={"value": "daily_max_f"})
        ddf = filter_to_local_today(ddf, city.tz)
        if not ddf.empty:
            daily_max_today = float(ddf.iloc[0]["daily_max_f"])

    cw = payload.get("current_weather") or {}
    current_temp = None
    if isinstance(cw, dict) and "temperature" in cw:
        try:
            current_temp = float(cw["temperature"])
        except Exception:
            current_temp = None

    return hourly_df, daily_max_today, current_temp, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_meteo_gfs(city: City) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    """
    Correct Open-Meteo GFS endpoint (/v1/gfs).
    Never raises — returns (hourly_today_df, daily_max_today, error_str).
    """
    url = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "timezone": city.tz,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
        # IMPORTANT: do NOT include current_weather here
    }

    payload, err = safe_get_json(url, params=params, timeout=14)
    if payload is None:
        return None, None, f"Open-Meteo (GFS) failed: {err}"

    hourly = payload.get("hourly") or {}
    h_times = hourly.get("time") or []
    h_temps = hourly.get("temperature_2m") or []
    hourly_df = parse_timeseries_df(h_times, h_temps, city.tz).rename(columns={"value": "temp_f"})
    hourly_df = filter_to_local_today(hourly_df, city.tz)

    daily = payload.get("daily") or {}
    d_times = daily.get("time") or []
    d_max = daily.get("temperature_2m_max") or []
    daily_max_today = None
    if d_times and d_max and len(d_times) == len(d_max):
        ddf = parse_timeseries_df(d_times, d_max, city.tz).rename(columns={"value": "daily_max_f"})
        ddf = filter_to_local_today(ddf, city.tz)
        if not ddf.empty:
            daily_max_today = float(ddf.iloc[0]["daily_max_f"])

    return hourly_df, daily_max_today, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nws_hourly(city: City) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
    """
    National Weather Service hourly forecast via api.weather.gov.
    Returns: (hourly_today_df, daily_max_today, error)
    """
    points_url = f"https://api.weather.gov/points/{city.lat:.4f},{city.lon:.4f}"
    points, err = safe_get_json(points_url, params=None, timeout=14)
    if points is None:
        return None, None, f"NWS points failed: {err}"

    props = (points.get("properties") or {})
    hourly_url = props.get("forecastHourly")
    if not hourly_url:
        return None, None, "NWS failed: points response missing forecastHourly URL."

    hourly_payload, err2 = safe_get_json(hourly_url, params=None, timeout=14)
    if hourly_payload is None:
        return None, None, f"NWS hourly failed: {err2}"

    periods = (hourly_payload.get("properties") or {}).get("periods") or []
    if not isinstance(periods, list) or len(periods) == 0:
        return None, None, "NWS hourly failed: no periods returned."

    times = []
    temps = []
    for p in periods:
        try:
            t = p.get("startTime")
            temp = p.get("temperature")
            if t is None or temp is None:
                continue
            times.append(t)
            temps.append(float(temp))
        except Exception:
            continue

    df = parse_timeseries_df(times, temps, city.tz).rename(columns={"value": "temp_f"})
    df = filter_to_local_today(df, city.tz)
    if df.empty:
        return df, None, "NWS hourly: no rows for today."

    daily_max = float(df["temp_f"].max())
    return df, daily_max, None


# =========================
# UI
# =========================
st.title("Kalshi Weather Trading Dashboard")

city_names = [c.name for c in CITIES]
default_city = "Austin, TX"
if default_city not in city_names:
    default_city = city_names[0]

selected_name = st.selectbox("Select City", city_names, index=city_names.index(default_city))
city = next(c for c in CITIES if c.name == selected_name)

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4], index=1)
grace = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard **ignores it automatically** (so it can't break cities).")

# =========================
# Fetch data
# =========================
errors: List[str] = []
sources_used: List[str] = []

om_hourly, om_daily_max, om_current, e1 = fetch_open_meteo_best(city)
if e1:
    errors.append(e1)
else:
    sources_used.append("Open-Meteo")

nws_hourly, nws_daily_max, e2 = fetch_nws_hourly(city)
if e2:
    errors.append(e2)
else:
    sources_used.append("National Weather Service (NWS)")

gfs_hourly, gfs_daily_max = None, None
if use_gfs:
    gfs_hourly, gfs_daily_max, e3 = fetch_open_meteo_gfs(city)
    if e3:
        errors.append(e3)
    else:
        sources_used.append("Open-Meteo GFS")

if sources_used:
    st.caption("Sources: " + " + ".join(sources_used))
else:
    st.error("No sources available right now.")
    st.stop()

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for msg in errors:
            st.write(f"• {msg}")

# =========================
# Choose "best" model
# =========================
best_label = None
best_high = None
best_hourly = None

# Prefer Open-Meteo daily max if available
if om_daily_max is not None:
    best_label = "Open-Meteo (best)"
    best_high = float(om_daily_max)
    best_hourly = om_hourly
elif nws_daily_max is not None:
    best_label = "NWS"
    best_high = float(nws_daily_max)
    best_hourly = nws_hourly
elif gfs_daily_max is not None:
    best_label = "Open-Meteo GFS"
    best_high = float(gfs_daily_max)
    best_hourly = gfs_hourly

if best_high is None:
    st.error("Could not compute a predicted daily high from available sources.")
    st.stop()

# Peak time from best hourly (if we have it)
peak_time = None
if best_hourly is not None and not best_hourly.empty:
    i = best_hourly["temp_f"].idxmax()
    try:
        peak_time = pd.Timestamp(best_hourly.loc[i, "dt"])
    except Exception:
        peak_time = None

# Peak window
peak_window = "—"
if peak_time is not None and not pd.isna(peak_time):
    start = peak_time - pd.Timedelta(minutes=grace)
    end = peak_time + pd.Timedelta(minutes=grace)
    peak_window = f"{fmt_time(start, city.tz)} – {fmt_time(end, city.tz)}"

# Spread for confidence + probability model
source_highs: Dict[str, float] = {}
if om_daily_max is not None:
    source_highs["Open-Meteo (best)"] = float(om_daily_max)
if nws_daily_max is not None:
    source_highs["NWS"] = float(nws_daily_max)
if use_gfs and gfs_daily_max is not None:
    source_highs["Open-Meteo (GFS)"] = float(gfs_daily_max)

if len(source_highs) >= 2:
    spread = float(max(source_highs.values()) - min(source_highs.values()))
else:
    # If only one source, assume some uncertainty
    spread = 0.8

confidence = confidence_from_spread(spread)
sigma = max(0.7, spread / 2.0)  # stable, simple approximation

# =========================
# Display header metrics
# =========================
st.header(city.name)

st.subheader("Predicted Daily High (°F)")
st.markdown(f"## {best_high:.1f}")

st.subheader("Confidence")
st.markdown(f"## {confidence}")

st.subheader("Estimated Peak Time")
st.markdown(f"## {fmt_time(peak_time, city.tz)}")
st.write(f"Peak window: **{peak_window}**")

# Current conditions (Open-Meteo current_weather if available)
st.subheader("Current Conditions")
if om_current is not None:
    st.write(f"Current Temp: **{om_current:.1f}°F**")
else:
    st.write("Current Temp: —")

# =========================
# Kalshi bracket suggestion
# =========================
st.divider()
st.header("Suggested Kalshi Range (Daily High)")

even_lo, even_hi = compute_bracket_even_start(best_high, bracket_size)
odd_lo, odd_hi = compute_bracket_odd_start(best_high, bracket_size)

# We’ll treat the "main" bracket as even-start by default (user can still see both)
main_lo, main_hi = even_lo, even_hi
main_label = f"{main_lo}-{main_hi}°F"

st.markdown(f"## {main_label}")
st.caption(f"Bracket interpretation: {bracket_size}° even-start (example for 2°: 78–79)")

# Show alternate alignment for 2-degree markets (very common in Kalshi)
if bracket_size == 2:
    st.write("Also check this alternate 2° alignment (Kalshi sometimes uses this):")
    st.write(f"• **{odd_lo}-{odd_hi}°F** — 2° odd-start (example: 79–80)")

# Nearby ranges
st.write("**Nearby ranges to watch:**")
near = []
for k in [-1, 0, 1]:
    lo = main_lo + k * bracket_size
    hi = lo + bracket_size - 1
    label = f"{lo}–{hi}"
    near.append(label + (" (current)" if k == 0 else ""))
for item in near:
    st.write(f"• {item}")

with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {today_local(city.tz)}")
    st.write(f"Best model used: **{best_label}**")
    for k, v in source_highs.items():
        st.write(f"{k}: **{v:.1f}°F**")

# =========================
# Probability ladder (based on simple normal approximation)
# =========================
st.divider()
st.header("Kalshi Probability Ladder")

# Build ladder around the main bracket
ladder_center = main_lo
steps_each_side = 6
rows = []
for i in range(-steps_each_side, steps_each_side + 1):
    lo = ladder_center + i * bracket_size
    hi = lo + bracket_size - 1
    p = prob_between(lo, hi, best_high, sigma) * 100.0
    rows.append({"Bracket": f"{lo}-{hi}", "Probability %": round(p, 2)})

ladder_df = pd.DataFrame(rows)
# Sort by bracket numeric low
ladder_df["__lo"] = ladder_df["Bracket"].str.split("-").str[0].astype(int)
ladder_df = ladder_df.sort_values("__lo").drop(columns="__lo").reset_index(drop=True)

st.dataframe(ladder_df, use_container_width=True, hide_index=True)

# =========================
# Value bet check
# =========================
st.divider()
st.header("Value Bet Check (you enter the Kalshi price)")

price_cents = st.number_input(
    "Enter Kalshi YES price for main bracket (cents, 0–100)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0,
)

model_p = prob_between(main_lo, main_hi, best_high, sigma)
implied_p = float(price_cents) / 100.0
edge = model_p - implied_p

st.write(f"Model probability for **{main_lo}-{main_hi}°F** ≈ **{model_p*100:.1f}%** (σ≈{sigma:.2f})")
st.write(f"Kalshi implied probability at **{price_cents:.0f}¢** ≈ **{implied_p*100:.1f}%**")

if edge > 0.02:
    st.success(f"Model edge: **+{edge*100:.1f}%** (model > market)")
elif edge < -0.02:
    st.error(f"Model edge: **{edge*100:.1f}%** (market > model)")
else:
    st.info(f"Model edge: **{edge*100:.1f}%** (roughly neutral)")

# =========================
# Model agreement table
# =========================
st.divider()
st.header("Model Agreement (Source Highs)")

if source_highs:
    rows = []
    for k, v in source_highs.items():
        # peak time per source if hourly available
        pt = None
        if k.startswith("Open-Meteo") and om_hourly is not None and not om_hourly.empty:
            ii = om_hourly["temp_f"].idxmax()
            pt = pd.Timestamp(om_hourly.loc[ii, "dt"])
        elif k == "NWS" and nws_hourly is not None and not nws_hourly.empty:
            ii = nws_hourly["temp_f"].idxmax()
            pt = pd.Timestamp(nws_hourly.loc[ii, "dt"])
        elif k.startswith("Open-Meteo (GFS)") and gfs_hourly is not None and not gfs_hourly.empty:
            ii = gfs_hourly["temp_f"].idxmax()
            pt = pd.Timestamp(gfs_hourly.loc[ii, "dt"])

        rows.append(
            {
                "Source": k,
                "Daily High (°F)": round(v, 1),
                "Peak Time": fmt_time(pt, city.tz),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =========================
# Peak-time heat spike detector (simple)
# =========================
st.divider()
st.header("Peak-time Heat Spike Detector")

if best_hourly is None or best_hourly.empty:
    st.write("Not enough hourly data to evaluate spike risk.")
else:
    # Simple spike detection: steepest 1-hour increase in the 4 hours leading to peak
    df = best_hourly.sort_values("dt").copy()
    df["delta"] = df["temp_f"].diff()
    if peak_time is not None:
        start_window = peak_time - pd.Timedelta(hours=4)
        end_window = peak_time
        w = df[(df["dt"] >= start_window) & (df["dt"] <= end_window)]
    else:
        w = df.tail(6)

    if w.empty or w["delta"].dropna().empty:
        st.write("No spike signal available.")
    else:
        max_jump = float(w["delta"].max())
        if max_jump >= 2.0:
            st.warning(f"Spike risk: max 1-hr jump **{max_jump:.1f}°F** near peak window.")
        elif max_jump >= 1.0:
            st.info(f"Moderate spike: max 1-hr jump **{max_jump:.1f}°F** near peak window.")
        else:
            st.success(f"Stable curve: max 1-hr jump **{max_jump:.1f}°F** near peak window.")

st.caption("Ultra-stable mode: this app never crashes from data-source errors. It only warns and continues.")
