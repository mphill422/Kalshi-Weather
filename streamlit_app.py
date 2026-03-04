import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ----------------------------
# Config
# ----------------------------
APP_TITLE = "Kalshi Weather Trading Dashboard"

# City list (requested)
CITIES = {
    "Austin, TX": {"query": "Austin, TX"},
    "Dallas, TX": {"query": "Dallas, TX"},
    "Houston, TX": {"query": "Houston, TX"},
    "San Antonio, TX": {"query": "San Antonio, TX"},
    "Phoenix, AZ": {"query": "Phoenix, AZ"},
    "New York City, NY": {"query": "New York City, NY"},
    "Atlanta, GA": {"query": "Atlanta, GA"},
    "Miami, FL": {"query": "Miami, FL"},
    "New Orleans, LA": {"query": "New Orleans, LA"},
    "Los Angeles, CA": {"query": "Los Angeles, CA"},
}

UA = {
    "User-Agent": "kalshi-weather-dashboard/1.0 (contact: none)",
    "Accept": "application/json",
}

REQ_TIMEOUT = 10  # seconds


# ----------------------------
# Helpers
# ----------------------------
def safe_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    """Return (json, error_string). Never throws."""
    try:
        h = dict(UA)
        if headers:
            h.update(headers)
        r = requests.get(url, params=params, headers=h, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fmt_time_local(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "—"
    # Keep simple display
    return dt.strftime("%I:%M %p").lstrip("0")


def today_date_in_tz(tz_name: str) -> pd.Timestamp:
    # Use pandas to handle tz
    now = pd.Timestamp.now(tz=tz_name)
    return now.normalize()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def bracket_label(lo: int, hi: int) -> str:
    return f"{lo}\u2013{hi}"


# ----------------------------
# Geocoding (Open-Meteo)
# ----------------------------
@st.cache_data(ttl=6 * 60 * 60)
def geocode_city(query: str) -> Tuple[Optional[dict], Optional[str]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    data, err = safe_get_json(url, params=params)
    if err:
        return None, err
    results = (data or {}).get("results") or []
    if not results:
        return None, "No geocoding results"
    r0 = results[0]
    return {
        "name": r0.get("name"),
        "admin1": r0.get("admin1"),
        "country": r0.get("country"),
        "latitude": r0.get("latitude"),
        "longitude": r0.get("longitude"),
        "timezone": r0.get("timezone") or "auto",
    }, None


# ----------------------------
# Open-Meteo Forecast
# ----------------------------
def open_meteo_params(lat: float, lon: float, tz: str, model: Optional[str] = None) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz if tz != "auto" else "auto",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "current_weather": True,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
    }
    if model:
        params["models"] = model  # e.g., "gfs"
    return params


def parse_open_meteo_hourly(payload: dict) -> pd.DataFrame:
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    rh = hourly.get("relative_humidity_2m") or []
    wind = hourly.get("windspeed_10m") or []

    # Guard against empty/mismatched lengths
    n = min(len(times), len(temps), len(rh), len(wind))
    if n == 0:
        return pd.DataFrame(columns=["time", "temp_f", "rh", "wind_mph"])

    df = pd.DataFrame({
        "time": pd.to_datetime(times[:n], errors="coerce"),
        "temp_f": pd.to_numeric(temps[:n], errors="coerce"),
        "rh": pd.to_numeric(rh[:n], errors="coerce"),
        "wind_mph": pd.to_numeric(wind[:n], errors="coerce"),
    }).dropna(subset=["time", "temp_f"])
    return df


def parse_open_meteo_daily(payload: dict) -> pd.DataFrame:
    daily = (payload or {}).get("daily") or {}
    times = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []

    n = min(len(times), len(tmax), len(tmin))
    if n == 0:
        return pd.DataFrame(columns=["date", "tmax_f", "tmin_f"])

    df = pd.DataFrame({
        "date": pd.to_datetime(times[:n], errors="coerce").dt.normalize(),
        "tmax_f": pd.to_numeric(tmax[:n], errors="coerce"),
        "tmin_f": pd.to_numeric(tmin[:n], errors="coerce"),
    }).dropna(subset=["date", "tmax_f"])
    return df


@st.cache_data(ttl=10 * 60)
def fetch_open_meteo(lat: float, lon: float, tz: str, model: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = open_meteo_params(lat, lon, tz, model=model)
    return safe_get_json(url, params=params)


def open_meteo_summary(payload: dict, tz: str) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float], pd.DataFrame, pd.DataFrame]:
    """Returns (today_high, peak_time, current_temp, daily_df, hourly_df)"""
    hourly_df = parse_open_meteo_hourly(payload)
    daily_df = parse_open_meteo_daily(payload)

    today = today_date_in_tz(tz)

    today_high = None
    if not daily_df.empty:
        row = daily_df[daily_df["date"] == today]
        if not row.empty:
            today_high = float(row.iloc[0]["tmax_f"])

    peak_time = None
    if not hourly_df.empty:
        # filter to today's local day
        start = today
        end = today + pd.Timedelta(days=1)
        day_hours = hourly_df[(hourly_df["time"] >= start) & (hourly_df["time"] < end)]
        if not day_hours.empty:
            idx = day_hours["temp_f"].astype(float).idxmax()
            peak_time = day_hours.loc[idx, "time"]

    current_temp = None
    cw = (payload or {}).get("current_weather") or {}
    if "temperature" in cw:
        try:
            current_temp = float(cw["temperature"])
        except Exception:
            current_temp = None

    return today_high, peak_time, current_temp, daily_df, hourly_df


# ----------------------------
# NWS (api.weather.gov)
# ----------------------------
@st.cache_data(ttl=10 * 60)
def fetch_nws_point(lat: float, lon: float) -> Tuple[Optional[dict], Optional[str]]:
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    return safe_get_json(url)


@st.cache_data(ttl=10 * 60)
def fetch_nws_forecast(url: str) -> Tuple[Optional[dict], Optional[str]]:
    return safe_get_json(url)


def nws_summary(lat: float, lon: float, tz: str) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[str]]:
    """
    Returns (today_high, approx_peak_time, error)
    Reliability fix: uses forecastHourly if present; otherwise uses forecast (periods).
    """
    point, err = fetch_nws_point(lat, lon)
    if err or not point:
        return None, None, f"NWS points failed: {err or 'unknown'}"

    props = (point.get("properties") or {})
    forecast_hourly_url = props.get("forecastHourly")
    forecast_url = props.get("forecast")

    # If hourly exists, try it first
    if forecast_hourly_url:
        fc, err2 = fetch_nws_forecast(forecast_hourly_url)
        if not err2 and fc:
            periods = (fc.get("properties") or {}).get("periods") or []
            if periods:
                today = today_date_in_tz(tz)
                # Convert to DataFrame of hourly temperatures
                rows = []
                for p in periods:
                    t = p.get("startTime")
                    temp = p.get("temperature")
                    if t is None or temp is None:
                        continue
                    rows.append((pd.to_datetime(t, errors="coerce"), temp))
                if rows:
                    df = pd.DataFrame(rows, columns=["time", "temp"]).dropna()
                    df["time"] = df["time"].dt.tz_convert(tz) if df["time"].dt.tz is not None else df["time"].dt.tz_localize(tz)
                    start = today
                    end = today + pd.Timedelta(days=1)
                    day = df[(df["time"] >= start) & (df["time"] < end)]
                    if not day.empty:
                        hi = float(day["temp"].max())
                        peak_time = day.loc[day["temp"].idxmax(), "time"]
                        return hi, peak_time, None

    # Fallback: use "forecast" (day/night periods)
    if forecast_url:
        fc, err3 = fetch_nws_forecast(forecast_url)
        if err3 or not fc:
            return None, None, f"NWS forecast failed: {err3 or 'unknown'}"
        periods = (fc.get("properties") or {}).get("periods") or []
        if not periods:
            return None, None, "NWS forecast missing periods"

        today = today_date_in_tz(tz)

        # Find today's daytime period
        best = None
        for p in periods:
            if not p.get("isDaytime"):
                continue
            stt = pd.to_datetime(p.get("startTime"), errors="coerce")
            ett = pd.to_datetime(p.get("endTime"), errors="coerce")
            if pd.isna(stt) or pd.isna(ett):
                continue
            # normalize to tz
            if stt.tzinfo is not None:
                stt = stt.tz_convert(tz)
            else:
                stt = stt.tz_localize(tz)
            if ett.tzinfo is not None:
                ett = ett.tz_convert(tz)
            else:
                ett = ett.tz_localize(tz)

            if stt.normalize() == today:
                best = (p, stt, ett)
                break

        if best:
            p, stt, ett = best
            temp = p.get("temperature")
            if temp is None:
                return None, None, "NWS daytime period missing temperature"
            hi = float(temp)
            # Approx peak time = midpoint of daytime window
            peak_time = stt + (ett - stt) / 2
            return hi, peak_time, None

        return None, None, "NWS couldn't find today's daytime period"

    return None, None, "NWS points response missing forecast URLs"


# ----------------------------
# Model combination + Kalshi math
# ----------------------------
def suggested_bracket(mean_high: float, bracket_size: int) -> Tuple[int, int]:
    """
    Returns (lo, hi) for the bracket that contains mean_high.
    Example: mean 79.4, size 2 => 78-79 (lo=78, hi=79)
    """
    size = max(1, int(bracket_size))
    lo = int(math.floor(mean_high / size) * size)
    hi = lo + size - 1
    return lo, hi


def nearby_brackets(lo: int, hi: int, size: int) -> List[Tuple[int, int]]:
    return [(lo - size, hi - size), (lo, hi), (lo + size, hi + size)]


def compute_sigma_from_spread(spread: float) -> float:
    """
    Heuristic: convert source disagreement into a usable sigma (°F).
    Keep a floor so ladder doesn't become degenerate.
    """
    # If spread is 0, still allow a little uncertainty.
    return max(0.8, float(spread) / 2.0)


def ladder_probs(mu: float, sigma: float, center_lo: int, size: int, steps_each_side: int = 3) -> pd.DataFrame:
    """
    Build probabilities for brackets around the suggested bracket.
    """
    brackets = []
    for k in range(-steps_each_side, steps_each_side + 1):
        lo = center_lo + k * size
        hi = lo + size - 1
        # Use half-open interval [lo, hi+1) for continuous approx
        p = normal_cdf(hi + 1, mu, sigma) - normal_cdf(lo, mu, sigma)
        brackets.append((bracket_label(lo, hi), max(0.0, p)))
    df = pd.DataFrame(brackets, columns=["Bracket", "Probability %"])
    df["Probability %"] = (df["Probability %"] * 100).round(1)
    # sort by bracket numeric lo
    def parse_lo(lbl: str) -> int:
        return int(lbl.split("–")[0])
    df["_lo"] = df["Bracket"].apply(parse_lo)
    df = df.sort_values("_lo").drop(columns=["_lo"]).reset_index(drop=True)
    return df


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

city_name = st.selectbox("Select City", list(CITIES.keys()), index=0)
bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4, 5], index=1)

grace = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard ignores it automatically. (This prevents the old “GFS error” from breaking cities.)")

geo, geo_err = geocode_city(CITIES[city_name]["query"])
if geo_err or not geo:
    st.error(f"Geocoding failed: {geo_err}")
    st.stop()

lat = float(geo["latitude"])
lon = float(geo["longitude"])
tz_name = geo.get("timezone") or "auto"
if tz_name == "auto":
    # Open-Meteo supports timezone=auto, but for consistent display we’ll still show "auto"
    tz_display = "auto"
else:
    tz_display = tz_name

sources_used = []
errors = []

# --- Open-Meteo (best)
om_payload, om_err = fetch_open_meteo(lat, lon, tz_name, model=None)
om_high = om_peak = om_current = None
om_daily_df = pd.DataFrame()
om_hourly_df = pd.DataFrame()
if om_err or not om_payload:
    errors.append(f"Open-Meteo (best) failed: {om_err}")
else:
    om_high, om_peak, om_current, om_daily_df, om_hourly_df = open_meteo_summary(om_payload, tz_name)
    if om_high is not None:
        sources_used.append(("Open-Meteo (best)", om_high, om_peak))

# --- NWS
nws_high = nws_peak = None
nws_err = None
try:
    nws_high, nws_peak, nws_err = nws_summary(lat, lon, tz_name)
    if nws_err:
        errors.append(nws_err)
    elif nws_high is not None:
        sources_used.append(("National Weather Service", nws_high, nws_peak))
except Exception as e:
    errors.append(f"NWS unexpected error: {type(e).__name__}: {e}")

# --- Optional GFS
gfs_high = gfs_peak = None
if use_gfs:
    gfs_payload, gfs_err = fetch_open_meteo(lat, lon, tz_name, model="gfs")
    if gfs_err or not gfs_payload:
        errors.append(f"Open-Meteo (GFS) failed: {gfs_err}")
    else:
        try:
            gfs_high, gfs_peak, _, _, gfs_hourly = open_meteo_summary(gfs_payload, tz_name)
            if gfs_high is not None:
                # prefer peak from GFS hourly if available
                sources_used.append(("Open-Meteo (GFS)", gfs_high, gfs_peak))
        except Exception as e:
            errors.append(f"Open-Meteo (GFS) parse error: {type(e).__name__}: {e}")

# Show sources banner
src_names = [s[0] for s in sources_used]
if src_names:
    st.caption("Sources: " + " + ".join(src_names))
else:
    st.caption("Sources: (none available right now)")

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write(f"• {e}")

st.header(city_name)

# If no highs, stop gracefully
if not sources_used:
    st.error("Could not fetch usable forecast data right now.")
    st.stop()

# Combine highs
highs = [float(s[1]) for s in sources_used if s[1] is not None]
mu = float(np.mean(highs))
spread = float(np.max(highs) - np.min(highs)) if len(highs) >= 2 else 0.0
sigma = compute_sigma_from_spread(spread)

# Peak time: prefer Open-Meteo best peak if present, else first available
peak = om_peak
if peak is None:
    for _, _, p in sources_used:
        if p is not None:
            peak = p
            break

# Peak window
if peak is not None and not pd.isna(peak):
    peak_window_start = peak - pd.Timedelta(minutes=int(grace))
    peak_window_end = peak + pd.Timedelta(minutes=int(grace))
else:
    peak_window_start = peak_window_end = None

# Display main metrics
st.subheader("Predicted Daily High (°F)")
st.metric(label="", value=f"{mu:.1f}")

conf_label = "High" if spread <= 1.0 else ("Medium" if spread <= 2.0 else "Low")
st.subheader("Confidence")
st.metric(label="", value=f"{conf_label} (spread {spread:.1f}°)")

st.subheader("Estimated Peak Time")
st.metric(label="", value=fmt_time_local(peak) if peak is not None else "—")

if peak_window_start is not None:
    st.write(f"Peak window: {fmt_time_local(peak_window_start)} – {fmt_time_local(peak_window_end)}")

# Suggested bracket + nearby
lo, hi = suggested_bracket(mu, int(bracket_size))
st.header("Suggested Kalshi Range")
st.write(f"**{lo}\u2013{hi}°F**")

st.write("Nearby ranges to watch:")
for (nlo, nhi) in nearby_brackets(lo, hi, int(bracket_size)):
    tag = " (current)" if (nlo == lo and nhi == hi) else ""
    st.write(f"• {nlo}\u2013{nhi}°F{tag}")

# Ladder
st.header("Kalshi Probability Ladder")
ladder = ladder_probs(mu, sigma, lo, int(bracket_size), steps_each_side=3)
st.dataframe(ladder, use_container_width=True, hide_index=True)

# Model agreement table
st.header("Model Agreement (Source Highs)")
rows = []
for name, h, p in sources_used:
    rows.append({
        "Source": name,
        "Daily High (°F)": round(float(h), 1),
        "Peak Time": fmt_time_local(p) if p is not None else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Value bet check (simple)
st.header("Value Bet Check (you enter the Kalshi price)")
st.caption("Enter the YES price (in cents) for the suggested bracket. This estimates implied edge vs model probability.")
yes_cents = st.number_input("YES price (cents)", min_value=1, max_value=99, value=50, step=1)

# Find probability of suggested bracket in ladder
p_row = ladder[ladder["Bracket"] == bracket_label(lo, hi)]
model_p = float(p_row["Probability %"].iloc[0]) / 100.0 if not p_row.empty else None
if model_p is not None:
    market_p = float(yes_cents) / 100.0
    edge = model_p - market_p
    st.write(f"Model probability for **{lo}\u2013{hi}°F** ≈ **{model_p*100:.1f}%**")
    st.write(f"Market implied probability at **{yes_cents}¢** ≈ **{market_p*100:.1f}%**")
    st.write(f"Estimated edge ≈ **{edge*100:.1f}%** (positive favors YES, negative favors NO)")
else:
    st.write("Model probability unavailable (not enough data).")

# Current conditions (Open-Meteo best)
st.header("Current Conditions (Open-Meteo)")
if om_current is not None:
    st.metric("Current Temp", f"{om_current:.1f}")
    # Show additional current from hourly if available
    if not om_hourly_df.empty:
        # nearest hour to now
        try:
            now = pd.Timestamp.now(tz=tz_name if tz_name != "auto" else None)
            if now.tzinfo is None and tz_name != "auto":
                now = now.tz_localize(tz_name)
            df = om_hourly_df.copy()
            if tz_name != "auto" and df["time"].dt.tz is None:
                df["time"] = df["time"].dt.tz_localize(tz_name)
            idx = (df["time"] - now).abs().idxmin()
            row = df.loc[idx]
            st.metric("Humidity (%)", f"{row['rh']:.0f}")
            st.metric("Wind (mph)", f"{row['wind_mph']:.1f}")
        except Exception:
            pass
else:
    st.write("Current conditions unavailable.")
