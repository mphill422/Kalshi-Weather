# streamlit_app.py
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

UA_HEADERS = {
    "User-Agent": "kalshi-weather-dashboard/1.0 (contact: none)",
    "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(UA_HEADERS)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"

# ----------------------------
# City definitions (hard-coded = stable)
# Add/adjust anytime.
# ----------------------------
@dataclass(frozen=True)
class City:
    label: str
    lat: float
    lon: float
    timezone: str

CITIES: Dict[str, City] = {
    # Your core cities
    "Austin, TX": City("Austin, TX", 30.2672, -97.7431, "America/Chicago"),
    "Dallas, TX": City("Dallas, TX", 32.7767, -96.7970, "America/Chicago"),
    "Houston, TX": City("Houston, TX", 29.7604, -95.3698, "America/Chicago"),
    "Phoenix, AZ": City("Phoenix, AZ", 33.4484, -112.0740, "America/Phoenix"),

    # Added cities you requested
    "New York City, NY": City("New York City, NY", 40.7128, -74.0060, "America/New_York"),
    "Atlanta, GA": City("Atlanta, GA", 33.7490, -84.3880, "America/New_York"),
    "Miami, FL": City("Miami, FL", 25.7617, -80.1918, "America/New_York"),
    "New Orleans, LA": City("New Orleans, LA", 29.9511, -90.0715, "America/Chicago"),
    "San Antonio, TX": City("San Antonio, TX", 29.4241, -98.4936, "America/Chicago"),
    "Los Angeles, CA": City("Los Angeles, CA", 34.0522, -118.2437, "America/Los_Angeles"),
}


# ----------------------------
# Helpers: safe requests + parsing
# ----------------------------
def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 12) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def f_to_str(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.1f}"


def dt_local_str(iso_str: str, tz_label: str) -> str:
    # Open-Meteo returns local time strings when timezone param is used.
    # Keep it simple: parse as naive.
    try:
        d = pd.to_datetime(iso_str, errors="coerce")
        if pd.isna(d):
            return "—"
        return d.strftime("%I:%M %p")
    except Exception:
        return "—"


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def prob_between(a: float, b: float, mu: float, sigma: float) -> float:
    # P(a <= X < b)
    return max(0.0, min(1.0, normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)))


# ----------------------------
# Data fetchers
# ----------------------------
@st.cache_data(ttl=90, show_spinner=False)
def fetch_open_meteo_best(city: City) -> Tuple[Optional[dict], Optional[str]]:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "timezone": city.timezone,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        # Important: always ask for current_weather to avoid None
        "current_weather": "true",
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
    }
    return safe_get_json(OPEN_METEO_URL, params=params)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_open_meteo_gfs(city: City) -> Tuple[Optional[dict], Optional[str]]:
    # GFS can fail sometimes; we isolate it so it never breaks the app.
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "timezone": city.timezone,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "current_weather": "true",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": "gfs",
    }
    return safe_get_json(OPEN_METEO_URL, params=params)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_nws(city: City) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    # Returns (points_json, hourly_forecast_json, error)
    points_url = NWS_POINTS_URL.format(lat=city.lat, lon=city.lon)
    points, err = safe_get_json(points_url, timeout=12)
    if points is None:
        return None, None, f"NWS points failed: {err}"

    props = (points or {}).get("properties") or {}
    hourly_url = props.get("forecastHourly")
    # Some points responses may not include hourly; handle it gracefully.
    if not hourly_url:
        return points, None, "NWS points response missing forecastHourly URL."

    hourly, err2 = safe_get_json(hourly_url, timeout=12)
    if hourly is None:
        return points, None, f"NWS hourly failed: {err2}"

    return points, hourly, None


# ----------------------------
# Parsers (safe)
# ----------------------------
def parse_open_meteo(payload: dict) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float], Optional[float], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns:
      daily_high, peak_time_iso, current_temp, humidity, wind,
      daily_df, hourly_df
    """
    if not payload:
        return None, None, None, None, None, None, None

    # Daily
    daily = payload.get("daily") or {}
    daily_times = daily.get("time") or []
    daily_max = daily.get("temperature_2m_max") or []
    daily_min = daily.get("temperature_2m_min") or []

    daily_df = None
    if daily_times:
        daily_df = pd.DataFrame({
            "date": pd.to_datetime(daily_times, errors="coerce"),
            "tmax": pd.to_numeric(daily_max, errors="coerce"),
            "tmin": pd.to_numeric(daily_min, errors="coerce"),
        }).dropna(subset=["date"]).reset_index(drop=True)

    daily_high = None
    if daily_df is not None and not daily_df.empty:
        daily_high = float(daily_df.loc[0, "tmax"]) if pd.notna(daily_df.loc[0, "tmax"]) else None

    # Hourly
    hourly = payload.get("hourly") or {}
    h_times = hourly.get("time") or []
    h_temp = hourly.get("temperature_2m") or []
    h_rh = hourly.get("relative_humidity_2m") or []
    h_wind = hourly.get("windspeed_10m") or []

    hourly_df = None
    if h_times and h_temp:
        hourly_df = pd.DataFrame({
            "time": pd.to_datetime(h_times, errors="coerce"),
            "temp": pd.to_numeric(h_temp, errors="coerce"),
            "rh": pd.to_numeric(h_rh, errors="coerce") if h_rh else pd.NA,
            "wind": pd.to_numeric(h_wind, errors="coerce") if h_wind else pd.NA,
        }).dropna(subset=["time", "temp"]).reset_index(drop=True)

    peak_time_iso = None
    if hourly_df is not None and not hourly_df.empty:
        idx = hourly_df["temp"].astype(float).idxmax()
        peak_time_iso = hourly_df.loc[idx, "time"].isoformat()

    # Current
    cw = payload.get("current_weather") or {}
    current_temp = cw.get("temperature")
    # Some Open-Meteo responses may not provide RH in current_weather; we pull from nearest hourly row.
    humidity = None
    wind = cw.get("windspeed")

    if hourly_df is not None and not hourly_df.empty:
        # nearest hourly row to "now" if current_weather time exists
        now_iso = cw.get("time")
        if now_iso:
            now_dt = pd.to_datetime(now_iso, errors="coerce")
            if pd.notna(now_dt):
                diffs = (hourly_df["time"] - now_dt).abs()
                j = diffs.idxmin()
                if "rh" in hourly_df.columns:
                    v = hourly_df.loc[j, "rh"]
                    humidity = float(v) if pd.notna(v) else None
                if wind is None and "wind" in hourly_df.columns:
                    w = hourly_df.loc[j, "wind"]
                    wind = float(w) if pd.notna(w) else None

    return (
        float(current_temp) if current_temp is not None else None,
        peak_time_iso,
        float(current_temp) if current_temp is not None else None,
        float(humidity) if humidity is not None else None,
        float(wind) if wind is not None else None,
        daily_df,
        hourly_df,
    )


def parse_nws_hourly(payload: dict) -> Tuple[Optional[float], Optional[str]]:
    """
    Approx daily high and peak time from NWS hourly forecast periods.
    """
    if not payload:
        return None, None

    props = (payload.get("properties") or {})
    periods = props.get("periods") or []
    if not periods:
        return None, None

    best_temp = None
    best_time = None

    # NWS hourly has "startTime" and "temperature"
    for p in periods[:36]:  # enough for today
        t = p.get("temperature")
        ts = p.get("startTime")
        if t is None or ts is None:
            continue
        try:
            t = float(t)
        except Exception:
            continue
        if best_temp is None or t > best_temp:
            best_temp = t
            best_time = ts

    return best_temp, best_time


# ----------------------------
# Kalshi bracket helpers (alignment fix)
# ----------------------------
def bracket_candidates(temp: float, size: int) -> List[Tuple[int, int, str]]:
    """
    Returns possible bracket interpretations so we match Kalshi menus.
    For size=2, Kalshi often uses either:
      even-start: 78–79, 80–81 ...
      odd-start:  77–78, 79–80 ...
    We'll return both.
    """
    if size <= 0:
        size = 2

    if size == 1:
        lo = int(math.floor(temp))
        return [(lo, lo, "1° bracket")]

    # size == 2 (or more): produce a couple alignment options
    # Option A: even-start  (… 78–79)
    lo_even = int(2 * math.floor(temp / 2))
    hi_even = lo_even + 1

    # Option B: odd-start (… 79–80) is represented by lo_odd=2k+1
    lo_odd = int(2 * math.floor((temp - 1) / 2) + 1)
    hi_odd = lo_odd + 1

    out = []
    out.append((lo_even, hi_even, "2° even-start (e.g., 78–79)"))
    if lo_odd != lo_even:
        out.append((lo_odd, hi_odd, "2° odd-start (e.g., 79–80)"))

    # If size > 2, we can generalize (rare in Kalshi). Keep it simple:
    if size > 2:
        lo = int(size * math.floor(temp / size))
        hi = lo + (size - 1)
        out = [(lo, hi, f"{size}° bracket")]

    return out


def fmt_range(lo: int, hi: int) -> str:
    return f"{lo}–{hi}°F"


# ----------------------------
# UI
# ----------------------------
st.title("Kalshi Weather Trading Dashboard")

city_label = st.selectbox("Select City", list(CITIES.keys()), index=0)
city = CITIES[city_label]

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=1)
grace_min = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard ignores it automatically. (This prevents GFS errors from breaking cities.)")

st.markdown(f"Sources: Open-Meteo + National Weather Service (NWS){' + Open-Meteo GFS' if use_gfs else ''}")

# ----------------------------
# Fetch all sources (but never crash)
# ----------------------------
errors = []

om_best, om_err = fetch_open_meteo_best(city)
if om_best is None:
    errors.append(f"Open-Meteo (best) failed: {om_err}")

nws_points, nws_hourly, nws_err = fetch_nws(city)
if nws_err:
    errors.append(f"NWS: {nws_err}")

gfs_payload = None
if use_gfs:
    gfs_payload, gfs_err = fetch_open_meteo_gfs(city)
    if gfs_payload is None:
        errors.append(f"Open-Meteo (GFS) failed: {gfs_err}")

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write("•", e)

# ----------------------------
# Parse sources
# ----------------------------
# Open-Meteo best (primary)
om_current, om_peak_iso, om_current2, om_humidity, om_wind, om_daily_df, om_hourly_df = (None, None, None, None, None, None, None)
om_daily_high = None
if om_best:
    om_current, om_peak_iso, om_current2, om_humidity, om_wind, om_daily_df, om_hourly_df = parse_open_meteo(om_best)
    # daily high is in daily df tmax[0] but we re-derive safely:
    if om_daily_df is not None and not om_daily_df.empty and pd.notna(om_daily_df.loc[0, "tmax"]):
        om_daily_high = float(om_daily_df.loc[0, "tmax"])

# NWS
nws_daily_high, nws_peak_iso = (None, None)
if nws_hourly:
    nws_daily_high, nws_peak_iso = parse_nws_hourly(nws_hourly)

# GFS
gfs_daily_high = None
gfs_peak_iso = None
if gfs_payload:
    _, gfs_peak_iso, _, _, _, gfs_daily_df, _ = parse_open_meteo(gfs_payload)
    if gfs_daily_df is not None and not gfs_daily_df.empty and pd.notna(gfs_daily_df.loc[0, "tmax"]):
        gfs_daily_high = float(gfs_daily_df.loc[0, "tmax"])

# ----------------------------
# Choose best daily high & peak time
# ----------------------------
source_rows = []
if om_daily_high is not None:
    source_rows.append(("Open-Meteo (best)", om_daily_high, om_peak_iso))
if nws_daily_high is not None:
    source_rows.append(("NWS", nws_daily_high, nws_peak_iso))
if use_gfs and gfs_daily_high is not None:
    source_rows.append(("Open-Meteo (GFS)", gfs_daily_high, gfs_peak_iso))

st.header(city.label)

if not source_rows:
    st.error("Could not fetch enough forecast data to compute a daily high right now.")
    st.stop()

# Best model used = Open-Meteo best if available, else NWS, else GFS
best_name, best_high, best_peak_iso = source_rows[0]
if om_daily_high is None and nws_daily_high is not None:
    best_name, best_high, best_peak_iso = ("NWS", nws_daily_high, nws_peak_iso)
elif om_daily_high is not None:
    best_name, best_high, best_peak_iso = ("Open-Meteo (best)", om_daily_high, om_peak_iso)
elif use_gfs and gfs_daily_high is not None:
    best_name, best_high, best_peak_iso = ("Open-Meteo (GFS)", gfs_daily_high, gfs_peak_iso)

# Confidence / spread
vals = [r[1] for r in source_rows if r[1] is not None]
spread = float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0

if spread <= 1.0:
    conf = f"High (spread {spread:.1f}°)"
elif spread <= 2.0:
    conf = f"Medium (spread {spread:.1f}°)"
else:
    conf = f"Low (spread {spread:.1f}°)"

# Peak window (grace minutes)
peak_time_disp = "—"
window_disp = "—"
if best_peak_iso:
    peak_dt = pd.to_datetime(best_peak_iso, errors="coerce")
    if pd.notna(peak_dt):
        peak_time_disp = peak_dt.strftime("%I:%M %p")
        start = peak_dt - pd.Timedelta(minutes=grace_min)
        end = peak_dt + pd.Timedelta(minutes=grace_min)
        window_disp = f"{start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')}"

st.subheader("Predicted Daily High (°F)")
st.markdown(f"<div style='font-size:54px; font-weight:700'>{best_high:.1f}</div>", unsafe_allow_html=True)

st.write(f"**Confidence:** {conf}")
st.write(f"**Estimated Peak Time:** {peak_time_disp}")
st.write(f"**Peak window:** {window_disp}")
st.caption("Peak window = the hottest part of the day *approx* (peak hour ± your grace minutes).")

# ----------------------------
# Current conditions (Open-Meteo best only, if available)
# ----------------------------
st.subheader("Current Conditions (Open-Meteo)")
col1, col2, col3 = st.columns(3)
col1.metric("Temp (°F)", f_to_str(om_current))
col2.metric("Humidity (%)", f_to_str(om_humidity))
col3.metric("Wind (mph)", f_to_str(om_wind))

# ----------------------------
# Suggested Kalshi range (alignment-aware)
# ----------------------------
st.subheader("Suggested Kalshi Range (Daily High)")

cand = bracket_candidates(best_high, int(bracket_size))
# show first as "primary"
primary_lo, primary_hi, primary_label = cand[0]
st.markdown(f"### {fmt_range(primary_lo, primary_hi)}")
st.caption(f"Bracket interpretation: {primary_label}")

if len(cand) > 1:
    st.write("Also check this alternate 2° alignment (Kalshi sometimes uses this):")
    for lo, hi, lbl in cand[1:]:
        st.write(f"• **{fmt_range(lo, hi)}** — {lbl}")

# nearby ranges to watch around primary bracket
if bracket_size == 2:
    st.write("**Nearby ranges to watch:**")
    st.write(f"• {fmt_range(primary_lo - 2, primary_hi - 2)}")
    st.write(f"• {fmt_range(primary_lo, primary_hi)} (current)")
    st.write(f"• {fmt_range(primary_lo + 2, primary_hi + 2)}")

# raw numbers expander
with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {dt.date.today().isoformat()}")
    st.write(f"Best model used: {best_name}")
    for name, high, peak_iso in source_rows:
        peak_disp = dt_local_str(peak_iso, city.timezone) if peak_iso else "—"
        st.write(f"- {name}: **{high:.1f}°F** | peak ~ {peak_disp}")

# ----------------------------
# Model agreement table
# ----------------------------
st.subheader("Model Agreement (Source Highs)")
tbl = pd.DataFrame(
    [{
        "Source": name,
        "Daily High (°F)": round(float(high), 1),
        "Peak Time": dt_local_str(peak_iso, city.timezone) if peak_iso else "—"
    } for (name, high, peak_iso) in source_rows]
)
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ----------------------------
# Probability ladder (simple normal approximation)
# ----------------------------
st.subheader("Kalshi Probability Ladder")

# Use mean of sources; sigma based on spread (never 0)
mu = float(sum(vals) / len(vals))
sigma = max(0.75, spread / 2.0)  # keep stable even when spread is 0

# Build ladder around primary bracket ± 4°F
if bracket_size == 1:
    ladder_los = list(range(primary_lo - 4, primary_lo + 5))
    ladder = []
    for lo in ladder_los:
        hi = lo
        # treat as [lo, lo+1)
        p = prob_between(lo, lo + 1, mu, sigma)
        ladder.append((f"{lo}–{hi}", p * 100))
else:
    ladder_los = list(range(primary_lo - 6, primary_lo + 7, 2))
    ladder = []
    for lo in ladder_los:
        hi = lo + 1
        # treat bracket as [lo, hi+1) because inclusive integers
        p = prob_between(lo, hi + 1, mu, sigma)
        ladder.append((f"{lo}–{hi}", p * 100))

ladder_df = pd.DataFrame(ladder, columns=["Bracket", "Probability %"])
ladder_df["Probability %"] = ladder_df["Probability %"].round(1)
st.dataframe(ladder_df, use_container_width=True, hide_index=True)

# ----------------------------
# Value bet check
# ----------------------------
st.subheader("Value Bet Check (you enter the Kalshi price)")

if bracket_size == 2:
    main_prob = prob_between(primary_lo, primary_hi + 1, mu, sigma)
else:
    main_prob = prob_between(primary_lo, primary_lo + 1, mu, sigma)

price_cents = st.number_input("Enter Kalshi YES price for main bracket (cents)", min_value=0, max_value=100, value=50, step=1)
implied = price_cents / 100.0
edge = main_prob - implied

st.write(f"Model probability for **{fmt_range(primary_lo, primary_hi)}** ≈ **{main_prob*100:.1f}%**")
st.write(f"Implied probability from price ≈ **{implied*100:.1f}%**")
st.write(f"Model edge ≈ **{edge*100:.1f}%**")

if edge >= 0.05:
    st.success("Positive edge (by this simple model). Still watch nearby brackets + price movement.")
elif edge <= -0.05:
    st.error("Negative edge (by this simple model).")
else:
    st.info("Close to fair (by this simple model).")

st.caption("This is a rough statistical approximation. Weather + market structure can move quickly.")
