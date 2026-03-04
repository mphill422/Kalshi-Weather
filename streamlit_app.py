import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

# =========================
# Ultra-stable Kalshi Weather Dashboard
# - No geocoding (hardcoded lat/lon)
# - Robust Open-Meteo + optional Open-Meteo GFS
# - Robust NWS with fallbacks + proper User-Agent
# - Kalshi bracket alignment helper (even-start vs odd-start)
# - Probability ladder (normal approx based on model spread)
# - Peak-time heat spike detector (current vs forecast curve)
# =========================

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

# ----------- City list (no geocoding) -----------
CITIES = {
    # Core
    "Austin, TX": (30.2672, -97.7431),
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    # Added
    "New York City, NY": (40.7128, -74.0060),
    "Atlanta, GA": (33.7490, -84.3880),
    "Miami, FL": (25.7617, -80.1918),
    "New Orleans, LA": (29.9511, -90.0715),
    "San Antonio, TX": (29.4241, -98.4936),
    "Los Angeles, CA": (34.0522, -118.2437),
}

DEFAULT_CITY = "Austin, TX"

# ----------- HTTP helpers -----------
SESSION = requests.Session()

def safe_get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 12):
    """
    Ultra-defensive GET JSON. Returns (data, error_str).
    """
    try:
        r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def parse_iso(dt_str: str):
    # Open-Meteo uses ISO strings. Some include timezone offset.
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

# ----------- Normal CDF (no scipy) -----------
def norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def prob_between(a: float, b: float, mu: float, sigma: float) -> float:
    # Inclusive-ish bounds
    return max(0.0, min(1.0, norm_cdf(b, mu, sigma) - norm_cdf(a, mu, sigma)))

# ----------- Kalshi bracket helpers -----------
def bracket_even_start(x: float, size: int) -> tuple[int, int]:
    """
    Even-start buckets for size=2 means ... 78-79, 80-81, 82-83, 84-85 ...
    Generally: start = floor(x/size)*size, but force even-start for size=2.
    """
    if size <= 0:
        size = 2
    start = int(math.floor(x / size) * size)
    end = start + size - 1
    return start, end

def bracket_odd_start(x: float, size: int) -> tuple[int, int]:
    """
    Odd-start buckets for size=2 means ... 79-80, 81-82, 83-84, 85-86 ...
    This is the common Kalshi menu you showed (83-84, 85-86, 87-88).
    """
    if size <= 0:
        size = 2
    # Shift by 1 then use even-start on shifted coordinate, then shift back
    start, end = bracket_even_start(x - 1, size)
    return start + 1, end + 1

def neighbors_for_bucket(lo: int, hi: int, size: int) -> list[tuple[int, int]]:
    return [(lo - size, hi - size), (lo, hi), (lo + size, hi + size)]

# ----------- Open-Meteo fetch + parse -----------
def fetch_open_meteo(lat: float, lon: float, use_gfs: bool = False):
    """
    Returns dict with:
      - source_name
      - tz_name
      - current_temp
      - hourly_df: columns [time, temp_f]
      - daily_high
      - peak_time (datetime)
      - errors list
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
        "current_weather": "true",
    }
    source_name = "Open-Meteo (best)"
    if use_gfs:
        params["models"] = "gfs"
        source_name = "Open-Meteo (GFS)"

    payload, err = safe_get_json(base_url, params=params, timeout=14)
    if err or not isinstance(payload, dict):
        return None, f"{source_name} failed: {err or 'Bad payload'}"

    tz_name = payload.get("timezone", "auto")

    # Hourly
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    if not times or not temps or len(times) != len(temps):
        return None, f"{source_name} failed: hourly data missing/invalid"

    hourly_df = pd.DataFrame({"time": times, "temp_f": temps})
    # Parse time with timezone if possible
    hourly_df["dt"] = hourly_df["time"].apply(parse_iso)
    hourly_df = hourly_df.dropna(subset=["dt"]).reset_index(drop=True)
    if hourly_df.empty:
        return None, f"{source_name} failed: could not parse hourly timestamps"

    # Current weather (optional)
    current_temp = None
    cw = payload.get("current_weather") or {}
    if isinstance(cw, dict):
        current_temp = cw.get("temperature")

    # Daily high "today" from daily list (first entry)
    daily = payload.get("daily") or {}
    daily_max = daily.get("temperature_2m_max") or []
    daily_time = daily.get("time") or []
    daily_high = None
    if daily_max:
        daily_high = float(daily_max[0])

    # Peak time today from hourly max within today's date
    # Determine today's date in the returned timezone by using the first hourly dt date
    first_dt = hourly_df["dt"].iloc[0]
    today_date = first_dt.date()
    today_df = hourly_df[hourly_df["dt"].dt.date == today_date]
    if today_df.empty:
        today_df = hourly_df.copy()

    peak_row = today_df.loc[today_df["temp_f"].astype(float).idxmax()]
    peak_time = peak_row["dt"]
    peak_temp = float(peak_row["temp_f"])

    # If daily_high missing, fallback to peak_temp
    if daily_high is None:
        daily_high = peak_temp

    return {
        "source": source_name,
        "tz": tz_name,
        "current_temp": float(current_temp) if current_temp is not None else None,
        "hourly_df": today_df[["dt", "temp_f"]].copy(),
        "daily_high": float(daily_high),
        "peak_time": peak_time,
        "peak_temp": float(peak_temp),
        "raw": payload,
    }, None

# ----------- NWS fetch + parse (robust) -----------
def fetch_nws_hourly(lat: float, lon: float):
    """
    NWS flow:
      1) /points/{lat},{lon} -> forecastHourly URL (sometimes missing)
      2) if missing, try forecastGridData (has time series)
      3) if missing, try forecast (periods) as last resort
    Returns dict in same format as Open-Meteo (hourly_df may be approximate if grid/period).
    """
    # NWS requires a descriptive User-Agent with contact per their guidance.
    # Put something stable; you can edit the email if you want.
    headers = {
        "User-Agent": "KalshiWeatherDashboard/1.0 (contact: mphill422@users.noreply.github.com)",
        "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
    }

    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    points, err = safe_get_json(points_url, headers=headers, timeout=14)
    if err or not isinstance(points, dict):
        return None, f"NWS failed: points error: {err or 'Bad payload'}"

    props = points.get("properties") or {}
    forecast_hourly_url = props.get("forecastHourly")
    grid_url = props.get("forecastGridData")
    forecast_url = props.get("forecast")

    # Helper: build hourly df from NWS periods (hourly)
    def hourly_df_from_periods(periods):
        rows = []
        for p in periods:
            t = p.get("startTime")
            temp = p.get("temperature")
            if t is None or temp is None:
                continue
            dt = parse_iso(t)
            if dt is None:
                continue
            # NWS temps are in Fahrenheit by default
            rows.append({"dt": dt, "temp_f": float(temp)})
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        # Keep today's date based on first row local date
        today_date = df["dt"].iloc[0].date()
        df = df[df["dt"].dt.date == today_date].reset_index(drop=True)
        if df.empty:
            return None
        return df

    # 1) Try forecastHourly
    if forecast_hourly_url:
        hourly_payload, herr = safe_get_json(forecast_hourly_url, headers=headers, timeout=14)
        if not herr and isinstance(hourly_payload, dict):
            periods = (hourly_payload.get("properties") or {}).get("periods") or []
            df = hourly_df_from_periods(periods)
            if df is not None and not df.empty:
                peak_row = df.loc[df["temp_f"].astype(float).idxmax()]
                return {
                    "source": "National Weather Service (NWS)",
                    "tz": "NWS",
                    "current_temp": None,  # NWS doesn't provide true "current" here reliably
                    "hourly_df": df.copy(),
                    "daily_high": float(peak_row["temp_f"]),
                    "peak_time": peak_row["dt"],
                    "peak_temp": float(peak_row["temp_f"]),
                    "raw": hourly_payload,
                }, None

    # 2) Try forecastGridData (contains time series, but not always hourly)
    # We'll attempt to read temperature.values and map validTime windows to midpoints.
    if grid_url:
        grid_payload, gerr = safe_get_json(grid_url, headers=headers, timeout=14)
        if not gerr and isinstance(grid_payload, dict):
            gprops = grid_payload.get("properties") or {}
            temp_obj = gprops.get("temperature") or {}
            values = temp_obj.get("values") or []
            rows = []
            for v in values:
                vt = v.get("validTime")
                val = v.get("value")
                if vt is None or val is None:
                    continue
                # validTime like "2026-03-04T15:00:00+00:00/PT1H"
                try:
                    start_str = vt.split("/")[0]
                    dt = parse_iso(start_str)
                except Exception:
                    dt = None
                if dt is None:
                    continue
                # Grid is usually Celsius; convert if unit is degC
                # Try to detect unit
                unit = temp_obj.get("uom", "")
                temp_f = None
                if "degC" in unit:
                    temp_f = (float(val) * 9 / 5) + 32
                else:
                    temp_f = float(val)
                rows.append({"dt": dt, "temp_f": float(temp_f)})

            df = pd.DataFrame(rows)
            if not df.empty:
                today_date = df["dt"].iloc[0].date()
                df = df[df["dt"].dt.date == today_date].reset_index(drop=True)
                if not df.empty:
                    peak_row = df.loc[df["temp_f"].astype(float).idxmax()]
                    return {
                        "source": "National Weather Service (NWS)",
                        "tz": "NWS",
                        "current_temp": None,
                        "hourly_df": df.copy(),
                        "daily_high": float(peak_row["temp_f"]),
                        "peak_time": peak_row["dt"],
                        "peak_temp": float(peak_row["temp_f"]),
                        "raw": grid_payload,
                    }, None

    # 3) Last resort: forecast (periods) - not hourly; use period temperatures
    if forecast_url:
        f_payload, ferr = safe_get_json(forecast_url, headers=headers, timeout=14)
        if not ferr and isinstance(f_payload, dict):
            periods = (f_payload.get("properties") or {}).get("periods") or []
            # Use any "isDaytime" period for today as a proxy
            rows = []
            for p in periods:
                t = p.get("startTime")
                temp = p.get("temperature")
                if t is None or temp is None:
                    continue
                dt = parse_iso(t)
                if dt is None:
                    continue
                rows.append({"dt": dt, "temp_f": float(temp)})
            df = pd.DataFrame(rows)
            if not df.empty:
                today_date = df["dt"].iloc[0].date()
                df = df[df["dt"].dt.date == today_date].reset_index(drop=True)
                if not df.empty:
                    peak_row = df.loc[df["temp_f"].astype(float).idxmax()]
                    return {
                        "source": "National Weather Service (NWS)",
                        "tz": "NWS",
                        "current_temp": None,
                        "hourly_df": df.copy(),
                        "daily_high": float(peak_row["temp_f"]),
                        "peak_time": peak_row["dt"],
                        "peak_temp": float(peak_row["temp_f"]),
                        "raw": f_payload,
                    }, None

    return None, "NWS failed: points response missing usable forecast URLs."

# ----------- Compute confidence from spread -----------
def confidence_label(spread: float) -> str:
    # Your rule of thumb: spread > 2 = be cautious
    if spread <= 1.0:
        return "High"
    if spread <= 2.0:
        return "Medium"
    return "Low"

# ----------- Main UI -----------
st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox("Select City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY))
lat, lon = CITIES[city]

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4, 5], index=1)

grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption(
    "If this ever fails, the dashboard ignores it automatically. "
    "(This prevents the old “GFS error” from breaking cities.)"
)

kalshi_alignment = st.radio(
    "Kalshi bracket alignment (menu style)",
    ["Auto (show both)", "Even-start (…82–83, 84–85…)", "Odd-start (…83–84, 85–86…)"],
    index=0,
)
st.caption(
    "Kalshi sometimes lists 2° ranges in an **odd-start** menu (83–84, 85–86…) like your screenshot. "
    "If your app range doesn’t match Kalshi’s list, switch alignment."
)

# ----------- Fetch sources -----------
errors = []

om_best, e1 = fetch_open_meteo(lat, lon, use_gfs=False)
if e1:
    errors.append(e1)

om_gfs = None
if use_gfs:
    om_gfs, e2 = fetch_open_meteo(lat, lon, use_gfs=True)
    if e2:
        errors.append(e2)

nws, e3 = fetch_nws_hourly(lat, lon)
if e3:
    errors.append(e3)

available = [x for x in [om_best, nws, om_gfs] if x is not None]

if not available:
    st.error("No sources available right now for this city. Try again in a minute.")
    with st.expander("See errors"):
        for e in errors:
            st.write("• " + e)
    st.stop()

sources_line = " + ".join([a["source"] for a in available])
st.caption(f"Sources: {sources_line}")

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write("• " + e)

# ----------- Choose "best" model for primary display -----------
# Prefer Open-Meteo (best), else NWS, else GFS.
best = om_best or nws or om_gfs

# ----------- Compute peak window -----------
peak_dt = best["peak_time"]
window_start = peak_dt - timedelta(minutes=grace_minutes)
window_end = peak_dt + timedelta(minutes=grace_minutes)

# ----------- Show headline -----------
st.header(city)

pred_high = float(best["daily_high"])
st.subheader("Predicted Daily High (°F)")
st.markdown(f"<div style='font-size:64px; font-weight:700; line-height:1;'>{pred_high:.1f}</div>", unsafe_allow_html=True)

# Agreement + spread
highs = [float(a["daily_high"]) for a in available]
spread = max(highs) - min(highs) if len(highs) >= 2 else 0.0
conf = confidence_label(spread)
st.subheader("Confidence")
st.markdown(f"<div style='font-size:56px; font-weight:700; line-height:1;'>{conf} (spread {spread:.1f}°)</div>", unsafe_allow_html=True)

# Peak time
st.subheader("Estimated Peak Time")
try:
    peak_str = peak_dt.strftime("%I:%M %p")
except Exception:
    peak_str = str(peak_dt)

st.markdown(f"<div style='font-size:56px; font-weight:700; line-height:1;'>{peak_str}</div>", unsafe_allow_html=True)
st.write(f"Peak window: {window_start.strftime('%I:%M %p')} – {window_end.strftime('%I:%M %p')}")

# ----------- Current conditions (from Open-Meteo best if present) -----------
if om_best and om_best.get("current_temp") is not None:
    st.subheader("Current Conditions (Open-Meteo)")
    st.metric("Current Temp (°F)", f"{om_best['current_temp']:.1f}")

# ----------- Peak-time heat spike detector -----------
st.subheader("Peak-time Heat Spike Detector")
detector_msg = "No spike detected."
detector_level = "info"

try:
    if om_best and not om_best["hourly_df"].empty and om_best.get("current_temp") is not None:
        # Find nearest hourly forecast to 'now' using Open-Meteo current_weather time if available
        cw_time = (om_best.get("raw") or {}).get("current_weather", {}).get("time")
        now_dt = parse_iso(cw_time) if cw_time else datetime.now(timezone.utc)
        dfh = om_best["hourly_df"].copy()
        dfh["diff"] = (dfh["dt"] - now_dt).abs()
        nearest = dfh.loc[dfh["diff"].idxmin()]
        forecast_now = float(nearest["temp_f"])
        current_now = float(om_best["current_temp"])
        delta = current_now - forecast_now

        # “Running hot” threshold
        if delta >= 2.0:
            detector_msg = f"⚠️ Running HOT: current is {delta:.1f}°F above forecast curve right now."
            detector_level = "warning"
        elif delta <= -2.0:
            detector_msg = f"⬇️ Running COOL: current is {abs(delta):.1f}°F below forecast curve right now."
            detector_level = "info"
        else:
            detector_msg = f"Normal: current is {delta:.1f}°F vs forecast curve."
            detector_level = "success"

        st.caption(f"Forecast-at-now uses nearest hourly point: {forecast_now:.1f}°F.")
    else:
        st.caption("Needs Open-Meteo current temp + hourly forecast to run the spike detector.")
except Exception as _:
    st.caption("Spike detector skipped due to an internal parsing issue (does not affect the rest of the dashboard).")

if detector_level == "warning":
    st.warning(detector_msg)
elif detector_level == "success":
    st.success(detector_msg)
else:
    st.info(detector_msg)

# ----------- Kalshi suggested range -----------
st.header("Suggested Kalshi Range (Daily High)")

# Determine suggestions for both alignments
lo_even, hi_even = bracket_even_start(pred_high, bracket_size)
lo_odd, hi_odd = bracket_odd_start(pred_high, bracket_size)

def fmt_range(lo, hi): return f"{lo}–{hi}°F"

# Pick display based on radio
if kalshi_alignment.startswith("Even-start"):
    main_lo, main_hi = lo_even, hi_even
    alt_lo, alt_hi = lo_odd, hi_odd
    main_label = "Even-start"
    alt_label = "Odd-start"
elif kalshi_alignment.startswith("Odd-start"):
    main_lo, main_hi = lo_odd, hi_odd
    alt_lo, alt_hi = lo_even, hi_even
    main_label = "Odd-start"
    alt_label = "Even-start"
else:
    # Auto: show both and let user decide from Kalshi menu
    main_lo, main_hi = lo_odd, hi_odd  # default to odd-start because that's what your screenshot showed
    alt_lo, alt_hi = lo_even, hi_even
    main_label = "Odd-start (common Kalshi menu)"
    alt_label = "Even-start"

st.markdown(f"<div style='font-size:52px; font-weight:800; line-height:1;'>{fmt_range(main_lo, main_hi)}</div>", unsafe_allow_html=True)
st.caption(f"Bracket interpretation: {bracket_size}° {main_label}")

# Always show alternate alignment so you can match the Kalshi screen quickly
st.write(f"Also check alternate alignment ({alt_label}): **{fmt_range(alt_lo, alt_hi)}**")

st.write("Nearby ranges to watch:")
near = neighbors_for_bucket(main_lo, main_hi, bracket_size)
for (lo, hi) in near:
    if lo == main_lo and hi == main_hi:
        st.write(f"• **{lo}–{hi} (current)**")
    else:
        st.write(f"• {lo}–{hi}")

# ----------- Raw forecast numbers expander -----------
with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {datetime.now().date()}")
    st.write(f"Best model used: {best['source']}")
    for a in available:
        try:
            st.write(f"{a['source']}: {float(a['daily_high']):.1f}°F | peak {a['peak_time'].strftime('%I:%M %p')}")
        except Exception:
            st.write(f"{a['source']}: {float(a['daily_high']):.1f}°F")

# ----------- Model agreement table -----------
st.header("Model Agreement (Source Highs)")
rows = []
for a in available:
    try:
        rows.append({
            "Source": a["source"],
            "Daily High (°F)": round(float(a["daily_high"]), 1),
            "Peak Time": a["peak_time"].strftime("%I:%M %p") if isinstance(a["peak_time"], datetime) else str(a["peak_time"]),
        })
    except Exception:
        rows.append({"Source": a["source"], "Daily High (°F)": a.get("daily_high"), "Peak Time": str(a.get("peak_time"))})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ----------- Probability ladder -----------
st.header("Kalshi Probability Ladder")

# Build a normal approx distribution from:
# mean = pred_high (best)
# sigma = max(0.8, spread/2) with a small floor
sigma = max(0.8, spread / 2.0)

# Ladder centered around the suggested main range
ladder_center = (main_lo + main_hi) / 2.0
# Show 5 buckets around the suggested one
ladder = []
for k in range(-2, 3):
    lo = main_lo + k * bracket_size
    hi = main_hi + k * bracket_size
    p = prob_between(lo - 0.5, hi + 0.5, mu=pred_high, sigma=sigma)  # continuity-ish
    ladder.append({"Bracket": f"{lo}–{hi}", "Probability %": round(100 * p, 1)})

ladder_df = pd.DataFrame(ladder)
st.dataframe(ladder_df, use_container_width=True, hide_index=True)
st.caption(f"Probability model: Normal(mean={pred_high:.1f}, sigma={sigma:.2f}) derived from model spread.")

# ----------- Value bet check -----------
st.header("Value Bet Check (you enter the Kalshi price)")
st.caption("Enter the YES price (in cents) for the bracket you’re considering. We compare it to the model probability.")

yes_price_cents = st.number_input("Enter Kalshi YES price (cents)", min_value=1, max_value=99, value=50, step=1)
model_p_main = float(ladder_df.loc[ladder_df["Bracket"] == f"{main_lo}–{main_hi}", "Probability %"].iloc[0]) / 100.0
implied_p = yes_price_cents / 100.0

edge = model_p_main - implied_p
st.write(f"Model probability for **{main_lo}–{main_hi}** ≈ **{model_p_main*100:.1f}%**")
st.write(f"Market implied probability at **{yes_price_cents}¢** ≈ **{implied_p*100:.1f}%**")

if edge > 0.03:
    st.success(f"✅ Positive edge: about **+{edge*100:.1f}%** vs market.")
elif edge > 0:
    st.info(f"Small edge: about **+{edge*100:.1f}%** vs market (thin).")
else:
    st.warning(f"❌ Negative edge: about **{edge*100:.1f}%** vs market.")

# ----------- Practical note on spread threshold -----------
with st.expander("Quick trading rule-of-thumb (spread)"):
    st.write(
        "If **spread > 2°F**, treat it as **low confidence**. "
        "That usually means forecasts disagree enough that late updates / station effects can swing the bracket."
    )
