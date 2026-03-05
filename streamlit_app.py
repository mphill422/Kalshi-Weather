# streamlit_app.py
# Ultra-stable Kalshi Weather Dashboard (Open-Meteo + NWS + optional Open-Meteo GFS)
# - No geocoding (fixed lat/lon) => removes “No geocoding results”
# - NWS reliability hardening (fallbacks, never crashes app)
# - GFS toggle hardened (uses Open-Meteo 'current='; failures are non-fatal and won’t break cities)
# - Kalshi bracket sizing + both 2° alignments (even-start + odd-start) so you can match Kalshi listings
# - Grace minutes slider, probability ladder, value check, simple heat-spike detector

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# Configuration / Cities
# =========================

APP_TITLE = "Kalshi Weather Trading Dashboard"
DEFAULT_CITY = "Austin, TX"

# Fixed coordinates (no geocoding)
# tz must be IANA tz name
CITIES: Dict[str, Dict[str, object]] = {
    # Texas core
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "San Antonio, TX": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},

    # Other cities requested
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta, GA": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
}

BRACKET_SIZES = [1, 2, 3, 4]


# =========================
# HTTP helpers (stable)
# =========================

SESSION = requests.Session()
SESSION.headers.update(
    {
        # NWS asks for a UA string with some contact; keep it simple
        "User-Agent": "KalshiWeatherDashboard/1.0 (contact: user)",
        "Accept": "application/geo+json,application/json;q=0.9,*/*;q=0.8",
    }
)

def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 12, retries: int = 2) -> Tuple[Optional[dict], Optional[str]]:
    """
    Defensive JSON fetch. Never raises.
    Returns (json, error_str).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                # Keep it short; Streamlit shows it in UI
                return None, f"{r.status_code} {r.reason}: {r.text[:200]}"
            return r.json(), None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.25 * (attempt + 1))
    return None, last_err or "Unknown error"


# =========================
# Open-Meteo
# =========================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_open_meteo(lat: float, lon: float, tz: str, model: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float], Optional[str]]:
    """
    Returns:
      hourly_df: columns [dt,temp_f,rh,wind_mph]
      daily_df:  columns [date,tmax_f,tmin_f]
      current_temp_f
      error_str
    """
    # KEY FIX for the toggle:
    # Use Open-Meteo's newer "current=" fields (NOT current_weather=true),
    # which avoids many 400 Bad Request cases when adding models=gfs.
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    }
    if model:
        params["models"] = model  # e.g., "gfs"

    payload, err = safe_get_json(OPEN_METEO_URL, params=params)
    if err or not payload:
        return None, None, None, err or "Open-Meteo returned no data."

    try:
        # Hourly
        h = payload.get("hourly", {}) or {}
        ht = h.get("time", []) or []
        temps = h.get("temperature_2m", []) or []
        rh = h.get("relative_humidity_2m", []) or []
        wind = h.get("wind_speed_10m", []) or []

        hourly_df = pd.DataFrame(
            {
                "dt": pd.to_datetime(ht, errors="coerce"),
                "temp_f": pd.to_numeric(temps, errors="coerce"),
                "rh": pd.to_numeric(rh, errors="coerce"),
                "wind_mph": pd.to_numeric(wind, errors="coerce"),
            }
        ).dropna(subset=["dt", "temp_f"])

        # Daily
        d = payload.get("daily", {}) or {}
        dt = d.get("time", []) or []
        tmax = d.get("temperature_2m_max", []) or []
        tmin = d.get("temperature_2m_min", []) or []

        daily_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dt, errors="coerce").dt.date,
                "tmax_f": pd.to_numeric(tmax, errors="coerce"),
                "tmin_f": pd.to_numeric(tmin, errors="coerce"),
            }
        ).dropna(subset=["date", "tmax_f"])

        # Current
        cur = payload.get("current", {}) or {}
        current_temp = cur.get("temperature_2m", None)
        try:
            current_temp = float(current_temp) if current_temp is not None else None
        except Exception:
            current_temp = None

        return hourly_df, daily_df, current_temp, None
    except Exception as e:
        return None, None, None, f"Parse error: {e}"


# =========================
# NWS (stable + fallbacks)
# =========================

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"

def _periods_to_hourly_df(periods: list) -> pd.DataFrame:
    rows = []
    for p in periods or []:
        stime = p.get("startTime")
        temp = p.get("temperature")
        if stime is None or temp is None:
            continue
        rows.append({"dt": pd.to_datetime(stime, errors="coerce"), "temp_f": pd.to_numeric(temp, errors="coerce")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["dt", "temp_f"]).sort_values("dt")

def fetch_nws_hourly(lat: float, lon: float) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Returns hourly_df [dt,temp_f] or None + error string.
    Never crashes the app.
    """
    points_payload, err = safe_get_json(NWS_POINTS_URL.format(lat=lat, lon=lon))
    if err or not points_payload:
        return None, f"NWS points failed: {err or 'no data'}"

    props = (points_payload.get("properties") or {})
    hourly_url = props.get("forecastHourly")
    forecast_url = props.get("forecast")  # fallback if hourly missing

    use_url = hourly_url or forecast_url
    if not use_url:
        return None, "NWS points response missing forecastHourly AND forecast URL."

    forecast_payload, err2 = safe_get_json(use_url)
    if err2 or not forecast_payload:
        return None, f"NWS forecast fetch failed: {err2 or 'no data'}"

    periods = (forecast_payload.get("properties") or {}).get("periods") or []
    df = _periods_to_hourly_df(periods)
    if df.empty:
        return None, "NWS returned no usable periods."
    return df, None


# =========================
# Math helpers
# =========================

def fmt_temp(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        if math.isnan(x) or math.isinf(x):
            return "—"
    except Exception:
        pass
    return f"{x:.1f}"

def to_12h(dt: pd.Timestamp) -> str:
    try:
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "—"

def bracket_for_temp(temp: float, size: int, start: int) -> Tuple[int, int]:
    """
    size=2:
      start=0 => 84–85, 86–87...
      start=1 => 83–84, 85–86...
    """
    lower = int(math.floor(float(temp)))
    while (lower - start) % size != 0:
        lower -= 1
    upper = lower + size - 1
    return lower, upper

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bracket_probabilities(mu: float, sigma: float, brackets: List[Tuple[int, int]]) -> List[float]:
    probs = []
    for lo, hi in brackets:
        # continuity correction for integer-inclusive bracket
        a = (lo - 0.5 - mu) / sigma
        b = (hi + 0.5 - mu) / sigma
        p = max(0.0, normal_cdf(b) - normal_cdf(a))
        probs.append(p)
    s = sum(probs)
    return [p / s for p in probs] if s > 0 else [0.0 for _ in probs]

def confidence_from_spread(spread: float) -> str:
    if spread <= 1.0:
        return f"High (spread {spread:.1f}°)"
    if spread <= 2.0:
        return f"Medium (spread {spread:.1f}°)"
    return f"Low (spread {spread:.1f}°)"


# =========================
# Core computations
# =========================

def todays_high_from_hourly(df: Optional[pd.DataFrame], today_date) -> Optional[float]:
    if df is None or df.empty:
        return None
    dff = df.copy()
    dff["dt"] = pd.to_datetime(dff["dt"], errors="coerce")
    dff = dff.dropna(subset=["dt"])
    dff = dff[dff["dt"].dt.date == today_date]
    if dff.empty:
        return None
    return float(dff["temp_f"].max())

def peak_from_hourly(df: Optional[pd.DataFrame], today_date) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    dff = df.copy()
    dff["dt"] = pd.to_datetime(dff["dt"], errors="coerce")
    dff = dff.dropna(subset=["dt"])
    dff = dff[dff["dt"].dt.date == today_date]
    if dff.empty:
        return None
    idx = dff["temp_f"].astype(float).idxmax()
    return pd.to_datetime(dff.loc[idx, "dt"], errors="coerce")


# =========================
# UI
# =========================

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

city_names = list(CITIES.keys())
default_idx = city_names.index(DEFAULT_CITY) if DEFAULT_CITY in city_names else 0

city = st.selectbox("Select City", city_names, index=default_idx)
bracket_size = st.selectbox("Kalshi bracket size (°F)", BRACKET_SIZES, index=BRACKET_SIZES.index(2))
grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

include_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If GFS fails, the dashboard ignores it automatically so it won’t break cities.")

meta = CITIES[city]
lat = float(meta["lat"])
lon = float(meta["lon"])
tz = str(meta["tz"])

@st.cache_data(ttl=300)
def load_all(lat: float, lon: float, tz: str, include_gfs: bool):
    om_hourly, om_daily, om_current, om_err = fetch_open_meteo(lat, lon, tz, model=None)
    nws_hourly, nws_err = fetch_nws_hourly(lat, lon)

    gfs_hourly = gfs_daily = None
    gfs_current = None
    gfs_err = None
    if include_gfs:
        gfs_hourly, gfs_daily, gfs_current, gfs_err = fetch_open_meteo(lat, lon, tz, model="gfs")

    return {
        "om": (om_hourly, om_daily, om_current, om_err),
        "nws": (nws_hourly, nws_err),
        "gfs": (gfs_hourly, gfs_daily, gfs_current, gfs_err),
    }

data = load_all(lat, lon, tz, include_gfs)

om_hourly, om_daily, om_current, om_err = data["om"]
nws_hourly, nws_err = data["nws"]
gfs_hourly, gfs_daily, gfs_current, gfs_err = data["gfs"]

sources_used = ["Open-Meteo (best)", "National Weather Service (NWS)"]
if include_gfs:
    sources_used.append("Open-Meteo (GFS)")
st.caption("Sources: " + " + ".join(sources_used))

errs = []
if om_err:
    errs.append(f"Open-Meteo (best) failed: {om_err}")
if nws_err:
    errs.append(f"NWS failed: {nws_err}")
if include_gfs and gfs_err:
    errs.append(f"Open-Meteo (GFS) failed: {gfs_err}")

if errs:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errs:
            st.write("• " + e)

st.header(city)

# Determine "today" based on Open-Meteo timestamps if available; else system date
if om_hourly is not None and not om_hourly.empty:
    today = om_hourly["dt"].iloc[0].date()
else:
    today = pd.Timestamp.now().date()

# Compute highs per source
source_highs: List[Tuple[str, float]] = []

# Open-Meteo best daily max preferred
om_high = None
if om_daily is not None and not om_daily.empty:
    row = om_daily[om_daily["date"] == today]
    if not row.empty:
        om_high = float(row["tmax_f"].iloc[0])
if om_high is None:
    om_high = todays_high_from_hourly(om_hourly, today)
if om_high is not None:
    source_highs.append(("Open-Meteo (best)", om_high))

# NWS high from hourly/periods
nws_high = todays_high_from_hourly(nws_hourly, today)
if nws_high is not None:
    source_highs.append(("NWS", nws_high))

# GFS high
gfs_high = None
if include_gfs:
    if gfs_daily is not None and not gfs_daily.empty:
        row = gfs_daily[gfs_daily["date"] == today]
        if not row.empty:
            gfs_high = float(row["tmax_f"].iloc[0])
    if gfs_high is None:
        gfs_high = todays_high_from_hourly(gfs_hourly, today)
    if gfs_high is not None:
        source_highs.append(("Open-Meteo (GFS)", gfs_high))

# Predicted high (anchor Open-Meteo best if present, else median of available)
pred_high = None
if om_high is not None:
    pred_high = om_high
elif source_highs:
    pred_high = float(np.median([h for _, h in source_highs]))

# Spread/confidence
spread = None
if len(source_highs) >= 2:
    vals = [h for _, h in source_highs]
    spread = float(max(vals) - min(vals))
elif len(source_highs) == 1:
    spread = 0.8  # small default uncertainty when only one source

# Peak time + window (prefer Open-Meteo hourly, then NWS, then GFS)
peak_time = peak_from_hourly(om_hourly, today) or peak_from_hourly(nws_hourly, today) or peak_from_hourly(gfs_hourly, today)
peak_window = None
if peak_time is not None:
    peak_window = (
        peak_time - pd.Timedelta(minutes=int(grace_minutes)),
        peak_time + pd.Timedelta(minutes=int(grace_minutes)),
    )

# Main display
st.subheader("Predicted Daily High (°F)")
st.markdown(f"<div style='font-size:56px; font-weight:700; line-height:1'>{fmt_temp(pred_high)}</div>", unsafe_allow_html=True)

if spread is not None:
    st.write(f"**Confidence:** {confidence_from_spread(spread)}")

if peak_time is not None:
    st.write(f"**Estimated Peak Time:** {to_12h(peak_time)}")
    if peak_window:
        st.write(f"**Peak window:** {to_12h(peak_window[0])} – {to_12h(peak_window[1])}")

# Suggested range + alignments
st.header("Suggested Kalshi Range (Daily High)")
if pred_high is None:
    st.info("Not enough forecast data available to generate a suggested range.")
else:
    size = int(bracket_size)

    # Show both alignments so it matches Kalshi listings (important for your “not aligned” issue)
    lo_even, hi_even = bracket_for_temp(pred_high, size=size, start=0)
    lo_odd, hi_odd = bracket_for_temp(pred_high, size=size, start=1)

    st.markdown(f"<div style='font-size:40px; font-weight:700'>{lo_even}–{hi_even}°F</div>", unsafe_allow_html=True)
    st.caption(f"Bracket interpretation: {size}° even-start (e.g., 78–79, 80–81, 82–83...)")

    st.write("Also check this alternate alignment (Kalshi sometimes uses this):")
    st.write(f"• **{lo_odd}–{hi_odd}°F** — {size}° odd-start (e.g., 79–80, 81–82, 83–84...)")

    st.subheader("Nearby ranges to watch (even-start)")
    for k in [-1, 0, 1]:
        wlo = lo_even + k * size
        whi = wlo + size - 1
        tag = " (current)" if k == 0 else ""
        st.write(f"• {wlo}–{whi}{tag}°F")

    with st.expander("See raw forecast numbers"):
        st.write(f"Forecast date (today): **{today}**")
        if source_highs:
            st.write("**Source highs:**")
            for name, h in source_highs:
                st.write(f"- {name}: **{h:.1f}°F**")
        else:
            st.write("No usable source highs.")

    # Probability ladder (normal approx)
    st.header("Kalshi Probability Ladder")

    sigma = max(0.8, (spread / 2.0) if spread is not None else 1.2)  # safe floor
    center_lo = lo_even

    brackets = []
    for i in range(-2, 3):
        blo = center_lo + i * size
        bhi = blo + size - 1
        brackets.append((blo, bhi))

    probs = bracket_probabilities(pred_high, sigma, brackets)
    ladder_df = pd.DataFrame(
        {
            "Bracket": [f"{a}–{b}" for a, b in brackets],
            "Probability %": [round(p * 100.0, 1) for p in probs],
        }
    )
    st.dataframe(ladder_df, use_container_width=True, hide_index=True)

    st.header("Value Bet Check (you enter the Kalshi price)")
    st.caption("Enter the YES price for the **main even-start bracket** in cents (e.g., 62 for $0.62).")
    price_cents = st.number_input("Kalshi YES price (cents)", min_value=0, max_value=100, value=50, step=1)

    model_prob = probs[2]  # middle bracket
    implied = float(price_cents) / 100.0
    edge = model_prob - implied

    st.write(f"Model probability for **{lo_even}–{hi_even}°F** ≈ **{model_prob*100:.1f}%** (σ≈{sigma:.1f}°)")
    st.write(f"Market implied probability ≈ **{implied*100:.1f}%**")
    st.write(f"Model edge ≈ **{edge*100:.1f}%**")
    if edge > 0.05:
        st.success("Positive edge (by this simple model).")
    elif edge < -0.05:
        st.error("Negative edge (market richer than model).")
    else:
        st.info("Close to fair (small edge).")

    # Simple spike detector
    st.header("Peak-time heat spike detector")
    st.caption("Flags when current temp is running hot vs the forecast curve heading into peak window.")
    if om_current is not None and om_hourly is not None and not om_hourly.empty and peak_time is not None:
        now = pd.Timestamp.now(tz=None)
        dff = om_hourly.copy()
        dff["dt"] = pd.to_datetime(dff["dt"], errors="coerce")
        dff = dff.dropna(subset=["dt"]).sort_values("dt")
        diffs = (dff["dt"] - now).abs()
        if not diffs.empty:
            nearest_idx = diffs.idxmin()
            forecast_now = float(dff.loc[nearest_idx, "temp_f"])
            delta = float(om_current - forecast_now)
            st.write(f"Current temp: **{om_current:.1f}°F** vs forecast-now: **{forecast_now:.1f}°F** (Δ {delta:+.1f}°)")
            if delta >= 1.0:
                st.success("Heat-spike signal: current is running hot vs forecast.")
            elif delta <= -1.0:
                st.warning("Cooler-than-forecast signal: current is running below forecast.")
            else:
                st.info("Tracking close to forecast.")
    else:
        st.info("Spike detector needs Open-Meteo current + hourly data.")
