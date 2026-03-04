# streamlit_app.py
# Ultra-stable Kalshi Weather Trading Dashboard (no geocoding; resilient sources)

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None


# -----------------------------
# Hard-coded cities (NO geocoding)
# -----------------------------
CITIES: Dict[str, Dict[str, object]] = {
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "San Antonio, TX": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta, GA": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

DEFAULT_CITY = "Austin, TX"


# -----------------------------
# HTTP helpers
# -----------------------------
UA = "Kalshi-Weather-Streamlit/1.0 (contact: local)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})


def _safe_get_json(url: str, params: dict, timeout: int = 12) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# -----------------------------
# Open-Meteo (Best baseline; no key)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_meteo(lat: float, lon: float, tz_name: str, model: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    # model=None uses Open-Meteo "best available"
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz_name,  # IMPORTANT: makes dates/times local (fixes NYC weirdness)
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "current_weather": True,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
    }
    if model:
        # Open-Meteo expects models as comma list (e.g., "gfs")
        params["models"] = model
    return _safe_get_json(base, params)


def parse_open_meteo(payload: dict, tz_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float], Optional[float]]:
    """
    Returns: daily_df, hourly_df, current_temp_f, current_wind_mph
    All time columns are timezone-aware if ZoneInfo available; otherwise naive local.
    """
    # Daily
    daily = payload.get("daily") or {}
    d_times = daily.get("time") or []
    d_max = daily.get("temperature_2m_max") or []
    d_min = daily.get("temperature_2m_min") or []

    daily_df = pd.DataFrame({"date": pd.to_datetime(d_times), "tmax_f": d_max, "tmin_f": d_min})
    # Hourly
    hourly = payload.get("hourly") or {}
    h_times = hourly.get("time") or []
    h_temp = hourly.get("temperature_2m") or []
    h_rh = hourly.get("relative_humidity_2m") or []
    h_wind = hourly.get("windspeed_10m") or []

    hourly_df = pd.DataFrame(
        {
            "time": pd.to_datetime(h_times),
            "temp_f": h_temp,
            "rh_pct": h_rh,
            "wind_mph": h_wind,
        }
    )

    if ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
        # Open-Meteo returns local timestamps without offset; localize them
        daily_df["date"] = daily_df["date"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        hourly_df["time"] = hourly_df["time"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    cw = payload.get("current_weather") or {}
    current_temp = cw.get("temperature")
    current_wind = cw.get("windspeed")
    return daily_df, hourly_df, current_temp, current_wind


# -----------------------------
# NWS (no key; can be flaky -> handle safely)
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_nws_points(lat: float, lon: float) -> Tuple[Optional[dict], Optional[str]]:
    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    return _safe_get_json(url, params={}, timeout=12)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_nws_forecast(url: str) -> Tuple[Optional[dict], Optional[str]]:
    return _safe_get_json(url, params={}, timeout=12)


def parse_nws_high(points_payload: dict, tz_name: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Attempts to extract a *today* high from NWS.
    Falls back gracefully if fields are missing.
    """
    try:
        props = (points_payload or {}).get("properties") or {}
        forecast_url = props.get("forecast")
        if not forecast_url:
            return None, "NWS points missing forecast URL"

        fc, err = fetch_nws_forecast(forecast_url)
        if err or not fc:
            return None, f"NWS forecast failed: {err}"

        periods = (fc.get("properties") or {}).get("periods") or []
        if not periods:
            return None, "NWS forecast periods missing"

        # Pick first daytime period that matches local date "today"
        tz = ZoneInfo(tz_name) if ZoneInfo is not None else None
        now_local = datetime.now(tz) if tz else datetime.now()
        today_local = now_local.date()

        for p in periods[:6]:
            # Example: startTime "2026-03-04T06:00:00-05:00"
            start = p.get("startTime")
            is_day = p.get("isDaytime")
            temp = p.get("temperature")
            unit = p.get("temperatureUnit")

            if start is None or temp is None:
                continue

            dt = pd.to_datetime(start)
            if tz and dt.tzinfo is None:
                dt = dt.tz_localize(tz)
            # if dt has its own tz offset, convert to local tz for correct date compare
            if tz and dt.tzinfo is not None:
                dt = dt.tz_convert(tz)

            if is_day and dt.date() == today_local:
                # Convert to F if necessary
                if unit == "C":
                    temp_f = temp * 9 / 5 + 32
                else:
                    temp_f = float(temp)
                return temp_f, None

        return None, "NWS did not provide a usable daytime period for today"
    except Exception as e:
        return None, f"NWS parse error: {type(e).__name__}: {e}"


# -----------------------------
# Trading logic helpers
# -----------------------------
def local_today(tz_name: str) -> date:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(tz_name)).date()
    return datetime.now().date()


def pick_today_daily_high(daily_df: pd.DataFrame, tz_name: str) -> Optional[float]:
    if daily_df.empty:
        return None
    today = local_today(tz_name)
    # daily_df["date"] is tz-aware; compare by date
    d = daily_df.copy()
    d["d"] = d["date"].dt.date
    row = d.loc[d["d"] == today]
    if row.empty:
        # fallback to first row (often "today" anyway)
        try:
            return float(daily_df.iloc[0]["tmax_f"])
        except Exception:
            return None
    try:
        return float(row.iloc[0]["tmax_f"])
    except Exception:
        return None


def peak_from_hourly(hourly_df: pd.DataFrame, tz_name: str) -> Tuple[Optional[float], Optional[datetime]]:
    if hourly_df.empty:
        return None, None
    today = local_today(tz_name)
    h = hourly_df.copy()
    h["d"] = h["time"].dt.date if "time" in h.columns else None
    today_rows = h.loc[h["d"] == today] if "d" in h.columns else h
    if today_rows.empty:
        today_rows = h
    try:
        idx = today_rows["temp_f"].astype(float).idxmax()
        peak_temp = float(today_rows.loc[idx, "temp_f"])
        peak_time = today_rows.loc[idx, "time"].to_pydatetime()
        return peak_temp, peak_time
    except Exception:
        return None, None


def bracket_for_temp(temp_f: float, bracket_size: int) -> Tuple[int, int]:
    """
    Returns inclusive integer bracket low/high for a given predicted high.
    Example: temp 79.4, size=2 => 78-79
    """
    if bracket_size <= 0:
        bracket_size = 2
    # Kalshi brackets are typically integer ranges
    t = float(temp_f)
    low = int(math.floor(t / bracket_size) * bracket_size)
    high = low + bracket_size - 1
    return low, high


def confidence_label(spread: float) -> str:
    if spread <= 1.0:
        return f"High (spread {spread:.1f}°)"
    if spread <= 2.0:
        return f"Medium (spread {spread:.1f}°)"
    return f"Low (spread {spread:.1f}°)"


def normal_bracket_probs(mu: float, sigma: float, bracket_size: int, center_low: int, n_brackets_each_side: int = 6) -> pd.DataFrame:
    """
    Builds probability table across nearby brackets under Normal(mu, sigma).
    """
    sigma = max(float(sigma), 0.6)  # keep sane / avoids divide-by-zero
    lows = [center_low + (i * bracket_size) for i in range(-n_brackets_each_side, n_brackets_each_side + 1)]
    rows = []
    for lo in lows:
        hi = lo + bracket_size - 1
        # integrate normal approx over [lo, hi+1) for integer bins
        a = (lo - mu) / sigma
        b = ((hi + 1) - mu) / sigma
        p = 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
        rows.append({"Bracket": f"{lo}-{hi}", "Probability %": max(0.0, p * 100.0), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    # Normalize (numerical stability)
    total = df["Probability %"].sum()
    if total > 0:
        df["Probability %"] = df["Probability %"] * (100.0 / total)
    return df[["Bracket", "Probability %", "lo", "hi"]]


def implied_prob_from_yes_price_cents(yes_price_cents: float) -> float:
    return max(0.0, min(1.0, float(yes_price_cents) / 100.0))


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")
st.title("Kalshi Weather Trading Dashboard")

# Controls
city_name = st.selectbox("Select City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY))
meta = CITIES[city_name]
lat = float(meta["lat"])
lon = float(meta["lon"])
tz_name = str(meta["tz"])

bracket_size = st.selectbox("Kalshi bracket size (°F)", options=[1, 2, 3, 4, 5], index=1)

grace = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption(
    "If this ever fails, the dashboard ignores it automatically. "
    "(This prevents the old “GFS error” from breaking cities.)"
)

# Fetch sources
source_errors: List[str] = []
sources_used: List[str] = []

om_payload, om_err = fetch_open_meteo(lat, lon, tz_name, model=None)
if om_err or not om_payload:
    source_errors.append(f"Open-Meteo (best) failed: {om_err}")
    om_daily = pd.DataFrame()
    om_hourly = pd.DataFrame()
    om_current = None
    om_wind = None
else:
    sources_used.append("Open-Meteo")
    om_daily, om_hourly, om_current, om_wind = parse_open_meteo(om_payload, tz_name)

# Optional GFS
gfs_daily = pd.DataFrame()
gfs_hourly = pd.DataFrame()
gfs_current = None
gfs_wind = None
if use_gfs:
    gfs_payload, gfs_err = fetch_open_meteo(lat, lon, tz_name, model="gfs")
    if gfs_err or not gfs_payload:
        source_errors.append(f"Open-Meteo (GFS) failed: {gfs_err}")
    else:
        sources_used.append("Open-Meteo GFS")
        try:
            gfs_daily, gfs_hourly, gfs_current, gfs_wind = parse_open_meteo(gfs_payload, tz_name)
        except Exception as e:
            source_errors.append(f"Open-Meteo (GFS) parse failed: {type(e).__name__}: {e}")

# NWS
nws_points, nws_points_err = fetch_nws_points(lat, lon)
nws_high = None
if nws_points_err or not nws_points:
    source_errors.append(f"NWS points failed: {nws_points_err}")
else:
    nws_high, nws_err = parse_nws_high(nws_points, tz_name)
    if nws_err:
        source_errors.append(f"NWS failed: {nws_err}")
    else:
        sources_used.append("National Weather Service (NWS)")

# Show source status
st.caption(f"Sources: " + (" + ".join(sources_used) if sources_used else "None"))
if source_errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in source_errors:
            st.write("• " + e)

# Compute today's predicted high from available sources
today_om_high = pick_today_daily_high(om_daily, tz_name) if not om_daily.empty else None
today_gfs_high = pick_today_daily_high(gfs_daily, tz_name) if (use_gfs and not gfs_daily.empty) else None

candidates = []
if today_om_high is not None:
    candidates.append(("Open-Meteo (best)", float(today_om_high)))
if nws_high is not None:
    candidates.append(("NWS", float(nws_high)))
if today_gfs_high is not None:
    candidates.append(("Open-Meteo (GFS)", float(today_gfs_high)))

# If nothing, stop gracefully
if not candidates:
    st.error("Could not fetch any usable forecast data right now.")
    st.stop()

# "Best" = Open-Meteo (best) if available; else first available
best_name, best_high = candidates[0]
for n, v in candidates:
    if n.startswith("Open-Meteo"):
        best_name, best_high = n, v
        break

# Spread/confidence based on agreement of sources
vals = [v for _, v in candidates]
if len(vals) >= 2:
    spread = float(statistics.pstdev(vals))
else:
    spread = 0.8  # default when only one source

# Peak time/window based on Open-Meteo hourly (most reliable availability)
peak_temp, peak_time = peak_from_hourly(om_hourly, tz_name) if not om_hourly.empty else (None, None)
if peak_time is None and use_gfs and not gfs_hourly.empty:
    peak_temp, peak_time = peak_from_hourly(gfs_hourly, tz_name)

# Current conditions (prefer Open-Meteo)
current_temp = om_current if om_current is not None else gfs_current
current_wind = om_wind if om_wind is not None else gfs_wind

# Render header metrics
st.markdown(f"## {city_name}")
st.metric("Predicted Daily High (°F)", f"{best_high:.1f}")
st.metric("Confidence", confidence_label(spread))

if peak_time is not None:
    st.metric("Estimated Peak Time", peak_time.strftime("%I:%M %p"))
    window_start = peak_time - timedelta(minutes=int(grace))
    window_end = peak_time + timedelta(minutes=int(grace))
    st.write(f"Peak window: {window_start.strftime('%I:%M %p')} – {window_end.strftime('%I:%M %p')}")
else:
    st.write("Estimated Peak Time: (not available)")

# Suggested Kalshi Range
low, high = bracket_for_temp(best_high, int(bracket_size))
st.markdown("## Suggested Kalshi Range")
st.markdown(f"### {low}–{high}°F")

nearby = [
    (low - bracket_size, low - 1),
    (low, high),
    (high + 1, high + bracket_size),
]
st.write("Nearby ranges to watch:")
for i, (lo, hi) in enumerate(nearby):
    tag = " (current)" if i == 1 else ""
    st.write(f"• {lo}–{hi}{tag}")

with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {local_today(tz_name).isoformat()}")
    st.write(f"Best model used: {best_name}")
    for n, v in candidates:
        st.write(f"{n}: {v:.1f}°F")
    if peak_time is not None and peak_temp is not None:
        st.write(f"Peak hour temp (today): {peak_temp:.1f} at {peak_time.isoformat()}")

# Current conditions section
st.markdown("## Current Conditions (Open-Meteo)")
if current_temp is not None:
    st.metric("Current Temp", f"{float(current_temp):.1f}")
else:
    st.write("Current Temp: (not available)")
if current_wind is not None:
    st.metric("Wind (mph)", f"{float(current_wind):.1f}")

# Model Agreement Table
st.markdown("## Model Agreement (Source Highs)")
agree_df = pd.DataFrame(
    [{"Source": n, "Daily High (°F)": f"{v:.1f}"} for n, v in candidates]
)
st.dataframe(agree_df, use_container_width=True, hide_index=True)

# Probability ladder
st.markdown("## Kalshi Probability Ladder")
# Use mu = best_high; sigma based on spread but keep floor
sigma = max(spread, 0.8)
center_low = (low // bracket_size) * bracket_size
ladder_df = normal_bracket_probs(best_high, sigma, int(bracket_size), int(center_low), n_brackets_each_side=6)

# Display with nice rounding
display_df = ladder_df.copy()
display_df["Probability %"] = display_df["Probability %"].map(lambda x: round(float(x), 1))
st.dataframe(display_df[["Bracket", "Probability %"]], use_container_width=True, hide_index=True)

# Value bet check
st.markdown("## Value Bet Check (you enter the Kalshi price)")
yes_price = st.number_input("Enter Kalshi YES price for main bracket (cents)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)

# Find model prob for main bracket
main_row = ladder_df.loc[(ladder_df["lo"] == low) & (ladder_df["hi"] == high)]
model_prob = float(main_row["Probability %"].iloc[0]) / 100.0 if not main_row.empty else None
implied = implied_prob_from_yes_price_cents(yes_price)

if model_prob is not None:
    edge = model_prob - implied
    st.write(f"Model probability for {low}–{high}°F ≈ **{model_prob*100:.1f}%**")
    st.write(f"Kalshi implied probability from YES price {yes_price:.1f}¢ ≈ **{implied*100:.1f}%**")
    if edge > 0.03:
        st.success(f"Model edge: **+{edge*100:.1f}%** (looks favorable vs price)")
    elif edge < -0.03:
        st.error(f"Model edge: **{edge*100:.1f}%** (price looks worse than model)")
    else:
        st.info(f"Model edge: **{edge*100:.1f}%** (roughly fair / close)")
else:
    st.write("Could not compute model probability for the main bracket.")

# Peak-time heat spike detector
st.markdown("## Peak-time heat spike detector")
st.write("Flags situations where **current temp is tracking ABOVE the expected curve** heading into the peak window.")
if current_temp is not None and peak_time is not None and not om_hourly.empty:
    try:
        tz = ZoneInfo(tz_name) if ZoneInfo is not None else None
        now_local = datetime.now(tz) if tz else datetime.now()
        # Build today's curve
        h = om_hourly.copy()
        h["d"] = h["time"].dt.date
        today = local_today(tz_name)
        h = h.loc[h["d"] == today].sort_values("time")
        if not h.empty:
            # expected temp at "now": nearest previous hourly point
            prior = h.loc[h["time"] <= (pd.Timestamp(now_local) if ZoneInfo is None else pd.Timestamp(now_local))]
            if prior.empty:
                expected_now = float(h.iloc[0]["temp_f"])
            else:
                expected_now = float(prior.iloc[-1]["temp_f"])

            curr = float(current_temp)
            delta = curr - expected_now

            # Simple thresholds
            if delta >= 2.0:
                st.warning(f"🔥 Heat spike: current is **{delta:+.1f}°F** above expected at this time.")
            elif delta <= -2.0:
                st.info(f"Cooling: current is **{delta:+.1f}°F** vs expected (below curve).")
            else:
                st.success(f"Normal tracking: current is **{delta:+.1f}°F** vs expected.")
    except Exception as e:
        st.write(f"(Spike detector unavailable: {type(e).__name__}: {e})")
else:
    st.write("(Need current temp + hourly curve + peak time to run spike detector.)")
