# streamlit_app.py
# Ultra-stable Kalshi Weather Trading Dashboard
# - Open-Meteo (best) + NWS hourly + optional Open-Meteo GFS (best-effort)
# - Never crashes when a source fails
# - Hardcoded city coordinates (no geocoding dependency)
# - Kalshi bracket ladder + value bet check + peak window w/ grace minutes

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------

APP_TITLE = "Kalshi Weather Trading Dashboard"

DEFAULT_CITY = "Austin, TX"
DEFAULT_BRACKET_SIZE = 2
DEFAULT_GRACE_MINUTES = 30

REQUEST_TIMEOUT = 10
RETRIES = 2
BACKOFF_SEC = 0.6

OM_BASE = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"

# IMPORTANT: Hardcode coords to avoid any geocoding failures
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    # Texas core
    "Austin, TX": (30.2672, -97.7431),
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "San Antonio, TX": (29.4241, -98.4936),

    # Requested additions
    "New York City, NY": (40.7128, -74.0060),
    "Atlanta, GA": (33.7490, -84.3880),
    "Miami, FL": (25.7617, -80.1918),
    "New Orleans, LA": (29.9511, -90.0715),
    "Los Angeles, CA": (34.0522, -118.2437),

    # Common Kalshi cities you’ve used
    "Phoenix, AZ": (33.4484, -112.0740),
    "Las Vegas, NV": (36.1699, -115.1398),
    "Chicago, IL": (41.8781, -87.6298),
}

# -----------------------------
# Helpers
# -----------------------------

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "kalshi-weather-dashboard/1.0 (contact: none)",
        "Accept": "application/json",
    }
)


def safe_get_json(url: str, params: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    """HTTP GET with retries. Returns (json, error_str). Never throws."""
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                # include a short hint; avoid huge dumps
                return None, f"HTTP {r.status_code} for url: {r.url}"
            return r.json(), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
    return None, last_err


def to_dt_series(s: pd.Series) -> pd.Series:
    """Parse datetimes safely."""
    dt = pd.to_datetime(s, errors="coerce")
    return dt


def today_local_from_series(dt: pd.Series) -> Optional[pd.Timestamp]:
    """Get 'today' date based on a datetime series. Fallback to local today."""
    try:
        dt_clean = dt.dropna()
        if len(dt_clean) > 0:
            return dt_clean.iloc[0].normalize()
    except Exception:
        pass
    # fallback to local date (server date)
    return pd.Timestamp.today().normalize()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_temp(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def fmt_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p")


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def bracket_label(lo: int, size: int) -> str:
    hi = lo + (size - 1)
    return f"{lo}–{hi}"


def nearest_bracket_floor(temp: float, size: int, start_parity: int) -> int:
    """
    Return the bracket starting temperature (integer) for a bracket of width 'size'.
    start_parity: 0 => even-start (e.g. size=2 uses 78–79)
                 1 => odd-start  (e.g. size=2 uses 79–80)
    """
    # Find nearest "grid" point with given parity
    # For size=2: starts are ..., 78,80,82 (even) or 79,81,83 (odd)
    # For general size: we keep parity meaning on the start integer.
    t = int(math.floor(temp))
    if (t % 2) != start_parity:
        t -= 1
    # Now slide to the closest bracket start on this parity grid
    # Using step=size? Kalshi always increments by 2 when size=2. For size=1 increments by 1.
    step = 2 if size == 2 else size
    # Align to step while preserving parity for size=2
    if size == 2:
        # already parity-aligned; now move in 2s
        lo = t
    else:
        lo = (t // step) * step
    # Center around the predicted temp
    # Compute candidate starts around lo and pick the one whose bracket mid is closest to temp
    candidates = [lo - step, lo, lo + step]
    best = candidates[0]
    best_dist = 1e9
    for c in candidates:
        mid = c + (size - 1) / 2
        d = abs(mid - temp)
        if d < best_dist:
            best_dist = d
            best = c
    return best


def build_probability_ladder(mu: float, sigma: float, size: int, main_lo: int, steps: int = 2) -> pd.DataFrame:
    """
    Build a 5-row ladder centered around main_lo.
    steps: how far to jump between adjacent brackets (2 for size=2, else size)
    """
    step = 2 if size == 2 else size
    los = [main_lo - 2 * step, main_lo - step, main_lo, main_lo + step, main_lo + 2 * step]

    rows = []
    for lo in los:
        hi = lo + (size - 1)
        # Probability temp in [lo, hi+1) using CDF on integer boundaries
        p = normal_cdf(hi + 1, mu, sigma) - normal_cdf(lo, mu, sigma)
        rows.append({"Bracket": bracket_label(lo, size), "Probability %": 100 * p})
    df = pd.DataFrame(rows)
    # clean presentation
    df["Probability %"] = df["Probability %"].clip(lower=0.0, upper=100.0).round(1)
    return df


# -----------------------------
# Data fetchers
# -----------------------------

@dataclass
class SourceResult:
    name: str
    daily_high: Optional[float]
    peak_time: Optional[pd.Timestamp]
    current_temp: Optional[float]
    hourly_df: Optional[pd.DataFrame]
    error: Optional[str]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_meteo(lat: float, lon: float) -> SourceResult:
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "timezone": "auto",  # more stable than forcing a tz string
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "current_weather": "true",
    }
    payload, err = safe_get_json(OM_BASE, params=params)
    if payload is None:
        return SourceResult("Open-Meteo (best)", None, None, None, None, err)

    try:
        # hourly
        h = payload.get("hourly", {}) or {}
        times = h.get("time", []) or []
        temps = h.get("temperature_2m", []) or []
        hourly_df = pd.DataFrame({"dt": times, "temp": temps})
        hourly_df["dt"] = to_dt_series(hourly_df["dt"])
        hourly_df = hourly_df.dropna(subset=["dt"]).sort_values("dt")

        # daily
        d = payload.get("daily", {}) or {}
        d_times = d.get("time", []) or []
        d_max = d.get("temperature_2m_max", []) or []
        daily_df = pd.DataFrame({"date": d_times, "tmax": d_max})
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce").dt.date
        daily_df = daily_df.dropna(subset=["date"])

        # pick "today" based on hourly data if possible
        today = today_local_from_series(hourly_df["dt"])
        today_date = today.date() if today is not None else pd.Timestamp.today().date()

        daily_high = None
        if len(daily_df) > 0:
            row = daily_df[daily_df["date"] == today_date]
            if len(row) > 0:
                daily_high = float(row["tmax"].iloc[0])

        # current
        cw = payload.get("current_weather", {}) or {}
        current_temp = cw.get("temperature", None)
        current_temp = float(current_temp) if current_temp is not None else None

        # peak time from today's hourly max
        peak_time = None
        if hourly_df is not None and len(hourly_df) > 0 and today is not None:
            start = today
            end = today + pd.Timedelta(days=1)
            day_df = hourly_df[(hourly_df["dt"] >= start) & (hourly_df["dt"] < end)]
            if len(day_df) > 0:
                idx = day_df["temp"].astype(float).idxmax()
                peak_time = pd.to_datetime(day_df.loc[idx, "dt"], errors="coerce")

        return SourceResult("Open-Meteo (best)", daily_high, peak_time, current_temp, hourly_df, None)
    except Exception as e:
        return SourceResult("Open-Meteo (best)", None, None, None, None, f"{type(e).__name__}: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_open_meteo_gfs(lat: float, lon: float) -> SourceResult:
    """
    Best-effort GFS. If Open-Meteo rejects models=gfs (HTTP 400), we return an error,
    but the UI must never crash.
    """
    # Some Open-Meteo deployments accept "models=gfs", others may require different model ids.
    # We try it; if it fails, we gracefully degrade.
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "current_weather": "true",
        "models": "gfs",
    }
    payload, err = safe_get_json(OM_BASE, params=params)
    if payload is None:
        return SourceResult("Open-Meteo (GFS)", None, None, None, None, err)

    try:
        h = payload.get("hourly", {}) or {}
        times = h.get("time", []) or []
        temps = h.get("temperature_2m", []) or []
        hourly_df = pd.DataFrame({"dt": times, "temp": temps})
        hourly_df["dt"] = to_dt_series(hourly_df["dt"])
        hourly_df = hourly_df.dropna(subset=["dt"]).sort_values("dt")

        d = payload.get("daily", {}) or {}
        d_times = d.get("time", []) or []
        d_max = d.get("temperature_2m_max", []) or []
        daily_df = pd.DataFrame({"date": d_times, "tmax": d_max})
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce").dt.date
        daily_df = daily_df.dropna(subset=["date"])

        today = today_local_from_series(hourly_df["dt"])
        today_date = today.date() if today is not None else pd.Timestamp.today().date()

        daily_high = None
        row = daily_df[daily_df["date"] == today_date]
        if len(row) > 0:
            daily_high = float(row["tmax"].iloc[0])

        cw = payload.get("current_weather", {}) or {}
        current_temp = cw.get("temperature", None)
        current_temp = float(current_temp) if current_temp is not None else None

        peak_time = None
        if len(hourly_df) > 0 and today is not None:
            start = today
            end = today + pd.Timedelta(days=1)
            day_df = hourly_df[(hourly_df["dt"] >= start) & (hourly_df["dt"] < end)]
            if len(day_df) > 0:
                idx = day_df["temp"].astype(float).idxmax()
                peak_time = pd.to_datetime(day_df.loc[idx, "dt"], errors="coerce")

        return SourceResult("Open-Meteo (GFS)", daily_high, peak_time, current_temp, hourly_df, None)
    except Exception as e:
        return SourceResult("Open-Meteo (GFS)", None, None, None, None, f"{type(e).__name__}: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nws_hourly(lat: float, lon: float) -> SourceResult:
    points_url = NWS_POINTS.format(lat=lat, lon=lon)
    p, err = safe_get_json(points_url)
    if p is None:
        return SourceResult("NWS", None, None, None, None, err)

    try:
        props = (p.get("properties", {}) or {})
        hourly_url = props.get("forecastHourly", None)

        if not hourly_url:
            # Some locations occasionally omit forecastHourly — fail gracefully
            return SourceResult("NWS", None, None, None, None, "NWS points response missing forecastHourly URL.")

        hx, err2 = safe_get_json(hourly_url)
        if hx is None:
            return SourceResult("NWS", None, None, None, None, err2)

        periods = ((hx.get("properties", {}) or {}).get("periods", []) or [])
        if not periods:
            return SourceResult("NWS", None, None, None, None, "NWS hourly had no periods.")

        df = pd.DataFrame(
            {
                "dt": [x.get("startTime", None) for x in periods],
                "temp": [x.get("temperature", None) for x in periods],
            }
        )
        df["dt"] = to_dt_series(df["dt"])
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df = df.dropna(subset=["dt", "temp"]).sort_values("dt")

        today = today_local_from_series(df["dt"])
        today_date = today.date() if today is not None else pd.Timestamp.today().date()

        # daily high = max temp among today's forecast periods
        daily_high = None
        peak_time = None
        if len(df) > 0 and today is not None:
            start = today
            end = today + pd.Timedelta(days=1)
            day_df = df[(df["dt"] >= start) & (df["dt"] < end)]
            if len(day_df) > 0:
                daily_high = float(day_df["temp"].max())
                idx = day_df["temp"].idxmax()
                peak_time = pd.to_datetime(day_df.loc[idx, "dt"], errors="coerce")

        # current temp: use earliest period as proxy (NWS doesn’t always provide "current" here)
        current_temp = float(df["temp"].iloc[0]) if len(df) > 0 else None

        return SourceResult("NWS", daily_high, peak_time, current_temp, df, None)
    except Exception as e:
        return SourceResult("NWS", None, None, None, None, f"{type(e).__name__}: {e}")


# -----------------------------
# Model logic
# -----------------------------

def combine_sources(sources: List[SourceResult]) -> Tuple[Optional[float], Optional[pd.Timestamp], float, str]:
    """
    Combine daily highs into predicted high.
    Returns: predicted_high, peak_time (from best available hourly), spread, confidence_label
    """
    highs = [s.daily_high for s in sources if s.daily_high is not None and not math.isnan(s.daily_high)]
    if not highs:
        return None, None, float("nan"), "Unknown"

    lo = float(min(highs))
    hi = float(max(highs))
    spread = hi - lo

    # Simple weighting: Open-Meteo (best) and NWS are primary; GFS is secondary.
    # If only one available, use it.
    weighted = []
    for s in sources:
        if s.daily_high is None or (isinstance(s.daily_high, float) and math.isnan(s.daily_high)):
            continue
        w = 1.0
        if "Open-Meteo (best)" in s.name:
            w = 1.2
        elif s.name == "NWS":
            w = 1.2
        elif "GFS" in s.name:
            w = 0.7
        weighted.append((float(s.daily_high), w))
    if not weighted:
        pred = float(np.mean(highs))
    else:
        num = sum(v * w for v, w in weighted)
        den = sum(w for _, w in weighted)
        pred = num / den if den > 0 else float(np.mean(highs))

    # Choose peak time from Open-Meteo(best) if present else NWS else GFS
    peak = None
    for preferred in ["Open-Meteo (best)", "NWS", "Open-Meteo (GFS)"]:
        for s in sources:
            if s.name == preferred and s.peak_time is not None and not pd.isna(s.peak_time):
                peak = s.peak_time
                break
        if peak is not None:
            break

    # Confidence label
    if spread <= 1.0:
        conf = f"High (spread {spread:.1f}°)"
    elif spread <= 2.0:
        conf = f"Medium (spread {spread:.1f}°)"
    else:
        conf = f"Low (spread {spread:.1f}°)"

    return float(pred), peak, float(spread), conf


def peak_window(peak: Optional[pd.Timestamp], grace_minutes: int) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if peak is None or pd.isna(peak):
        return None, None
    delta = pd.Timedelta(minutes=int(grace_minutes))
    return peak - delta, peak + delta


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

# Controls
city = st.selectbox("Select City", list(CITY_COORDS.keys()), index=list(CITY_COORDS.keys()).index(DEFAULT_CITY) if DEFAULT_CITY in CITY_COORDS else 0)
bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3], index=[1, 2, 3].index(DEFAULT_BRACKET_SIZE))
grace = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=DEFAULT_GRACE_MINUTES, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If GFS ever fails, the dashboard ignores it automatically so it won’t break cities.")

lat, lon = CITY_COORDS[city]

# Fetch sources
om = fetch_open_meteo(lat, lon)
nws = fetch_nws_hourly(lat, lon)

sources: List[SourceResult] = [om, nws]
gfs = None
if use_gfs:
    gfs = fetch_open_meteo_gfs(lat, lon)
    sources.append(gfs)

# Sources line
src_names = [s.name for s in sources]
st.caption("Sources: " + " + ".join(src_names))

# Any failures (non-fatal)
errors = []
for s in sources:
    if s.error:
        errors.append(f"• {s.name} failed: {s.error}")

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write(e)

pred_high, peak_t, spread, conf = combine_sources(sources)
lo_t, hi_t = peak_window(peak_t, grace)

# Header section
st.header(city)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicted Daily High (°F)")
    st.markdown(f"<h1 style='margin-top:-10px'>{fmt_temp(pred_high)}</h1>", unsafe_allow_html=True)

with col2:
    st.subheader("Confidence")
    st.markdown(f"<h1 style='margin-top:-10px'>{conf}</h1>", unsafe_allow_html=True)

st.subheader("Estimated Peak Time")
st.markdown(f"<h1 style='margin-top:-10px'>{fmt_time(peak_t)}</h1>", unsafe_allow_html=True)

if lo_t is not None and hi_t is not None:
    st.write(f"Peak window: {fmt_time(lo_t)} – {fmt_time(hi_t)}")

# Current conditions (prefer Open-Meteo current)
current_temp = om.current_temp if om.current_temp is not None else (nws.current_temp if nws.current_temp is not None else None)
st.subheader("Current Conditions")
st.write(f"Current Temp: **{fmt_temp(current_temp)}°F**")

# Suggested Kalshi Range
st.divider()
st.header("Suggested Kalshi Range (Daily High)")

if pred_high is None:
    st.write("Not enough data to suggest a range right now.")
else:
    # Two bracket alignments (Kalshi sometimes lists 2° bins as 77–78, 79–80, etc.)
    # "Even-start": 78–79 style (start parity 0)
    # "Odd-start":  79–80 style (start parity 1)
    even_lo = nearest_bracket_floor(pred_high, bracket_size, start_parity=0)
    odd_lo = nearest_bracket_floor(pred_high, bracket_size, start_parity=1)

    # Main suggestion: show both if they differ (this fixes “not aligned with Kalshi list” confusion)
    even_lbl = bracket_label(even_lo, bracket_size)
    odd_lbl = bracket_label(odd_lo, bracket_size)

    st.write(f"**Primary suggestion:** {even_lbl}°F")
    st.caption(f"Bracket interpretation: {bracket_size}° even-start (example for 2°: 78–79)")

    if odd_lbl != even_lbl:
        st.write("Also check this alternate alignment (Kalshi sometimes uses this):")
        st.write(f"• **{odd_lbl}°F** — {bracket_size}° odd-start (example for 2°: 79–80)")

    st.write("**Nearby ranges to watch:**")
    step = 2 if bracket_size == 2 else bracket_size
    near = [even_lo - step, even_lo, even_lo + step]
    for lo in near:
        label = bracket_label(lo, bracket_size)
        suffix = " (current)" if lo == even_lo else ""
        st.write(f"• {label}{suffix}")

    # Raw numbers
    with st.expander("See raw forecast numbers"):
        st.write(f"Forecast date (today): {pd.Timestamp.today().date()}")
        st.write(f"Best model used: {om.name if om.daily_high is not None else '—'}")
        if om.daily_high is not None:
            st.write(f"{om.name}: {om.daily_high:.1f}°F")
        if nws.daily_high is not None:
            st.write(f"NWS: {nws.daily_high:.1f}°F")
        if gfs and gfs.daily_high is not None:
            st.write(f"{gfs.name}: {gfs.daily_high:.1f}°F")

# Model agreement table
st.divider()
st.header("Model Agreement (Source Highs)")

rows = []
for s in sources:
    rows.append(
        {
            "Source": s.name,
            "Daily High (°F)": None if s.daily_high is None else round(float(s.daily_high), 1),
            "Peak Time": fmt_time(s.peak_time),
        }
    )
agree_df = pd.DataFrame(rows)
st.dataframe(agree_df, use_container_width=True, hide_index=True)

# Probability ladder
st.divider()
st.header("Kalshi Probability Ladder")

if pred_high is None:
    st.write("Not enough data to build a ladder.")
else:
    # Convert spread into sigma. If spread small, keep a minimum sigma so ladder isn’t degenerate.
    # Heuristic: sigma ~ max(0.8, spread/2). (You can tune later.)
    sigma = max(0.8, (0 if (spread is None or math.isnan(spread)) else spread / 2.0))
    ladder_df = build_probability_ladder(mu=pred_high, sigma=sigma, size=bracket_size, main_lo=even_lo)
    st.dataframe(ladder_df, use_container_width=True, hide_index=True)
    st.caption(f"Model uses μ={pred_high:.1f}, σ≈{sigma:.2f}. Ladder is approximate (but useful for edge checks).")

# Value bet check
st.divider()
st.header("Value Bet Check (you enter the Kalshi price)")

if pred_high is None:
    st.write("Not enough data.")
else:
    yes_price = st.number_input("Enter Kalshi YES price for main bracket (cents, 0–100)", min_value=0, max_value=100, value=50, step=1)
    implied = yes_price / 100.0

    # Model probability for the main (even-start) bracket:
    main_hi = even_lo + (bracket_size - 1)
    sigma = max(0.8, (0 if (spread is None or math.isnan(spread)) else spread / 2.0))
    model_p = (normal_cdf(main_hi + 1, pred_high, sigma) - normal_cdf(even_lo, pred_high, sigma))
    model_p = clamp(model_p, 0.0, 1.0)

    edge = model_p - implied

    st.write(f"Model probability for **{bracket_label(even_lo, bracket_size)}°F** ≈ **{model_p*100:.1f}%**")
    st.write(f"Implied probability from {yes_price}¢ ≈ **{implied*100:.1f}%**")

    if edge > 0.02:
        st.success(f"Positive edge ≈ **{edge*100:.1f}%** (model > market)")
    elif edge < -0.02:
        st.error(f"Negative edge ≈ **{edge*100:.1f}%** (market > model)")
    else:
        st.warning(f"Near fair value (edge ≈ {edge*100:.1f}%).")

# Notes
st.divider()
with st.expander("Quick notes (so you don’t get tripped up)"):
    st.write("• **Peak window** = the hottest part of the day *around the estimated peak time*, expanded by your grace minutes.")
    st.write("• If **spread > ~2°**, that means sources disagree — not always a skip, but it’s riskier and the ladder will flatten.")
    st.write("• If the Kalshi market bins look like **77–78, 79–80**, use the **odd-start** alignment shown above.")
    st.write("• The GFS toggle is strictly optional; if it fails, the dashboard ignores it and continues.")
