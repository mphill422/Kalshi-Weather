import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

USER_AGENT = "kalshi-weather-dashboard/1.0 (contact: you@example.com)"  # NWS likes having a UA
REQ_TIMEOUT = 12


@dataclass(frozen=True)
class City:
    label: str
    latitude: float
    longitude: float
    timezone: str


CITIES: Dict[str, City] = {
    # Core cities you were using
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


def safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def now_in_tz(tz_name: str) -> dt.datetime:
    # Avoid pytz dependency; rely on Open-Meteo time strings and compare by date
    # Streamlit Cloud can be UTC; we only need the local "today" date
    return dt.datetime.utcnow()


def parse_open_meteo(payload: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """
    Returns:
      daily_df: columns [date, tmax_f]
      hourly_df: columns [time, temp_f]
      current_temp_f: float or None
    """
    if not payload:
        return pd.DataFrame(), pd.DataFrame(), None

    # Daily
    daily = payload.get("daily", {}) or {}
    daily_times = daily.get("time", []) or []
    daily_tmax = daily.get("temperature_2m_max", []) or []
    daily_df = pd.DataFrame()
    if len(daily_times) and len(daily_tmax) and len(daily_times) == len(daily_tmax):
        daily_df = pd.DataFrame({
            "date": pd.to_datetime(daily_times, errors="coerce").dt.date,
            "tmax_f": pd.to_numeric(daily_tmax, errors="coerce")
        }).dropna()

    # Hourly
    hourly = payload.get("hourly", {}) or {}
    hourly_times = hourly.get("time", []) or []
    hourly_temp = hourly.get("temperature_2m", []) or []
    hourly_df = pd.DataFrame()
    if len(hourly_times) and len(hourly_temp) and len(hourly_times) == len(hourly_temp):
        hourly_df = pd.DataFrame({
            "time": pd.to_datetime(hourly_times, errors="coerce"),
            "temp_f": pd.to_numeric(hourly_temp, errors="coerce")
        }).dropna()

    # Current
    current_temp_f = None
    cw = payload.get("current_weather") or {}
    if "temperature" in cw:
        try:
            current_temp_f = float(cw["temperature"])
        except Exception:
            current_temp_f = None

    return daily_df, hourly_df, current_temp_f


def fetch_open_meteo_best(city: City) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float], Optional[str], pd.DataFrame]:
    """
    Returns:
      todays_high_f, peak_time, current_temp_f, error, hourly_df
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.timezone,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "current_weather": "true",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
    }
    payload, err = safe_get(url, params=params)
    if err:
        return None, None, None, err, pd.DataFrame()

    daily_df, hourly_df, current_temp_f = parse_open_meteo(payload)
    if daily_df.empty:
        return None, None, current_temp_f, "Open-Meteo (best): missing daily data", hourly_df

    # Determine "today" based on city timezone by using Open-Meteo timezone output
    today_date = dt.date.today()
    # daily_df dates already in city timezone date via Open-Meteo response
    row = daily_df[daily_df["date"] == today_date]
    if row.empty:
        # fallback: use first row
        todays_high_f = float(daily_df.iloc[0]["tmax_f"])
    else:
        todays_high_f = float(row.iloc[0]["tmax_f"])

    peak_time = None
    if not hourly_df.empty:
        hd = hourly_df.copy()
        hd["date"] = hd["time"].dt.date
        hrow = hd[hd["date"] == today_date]
        if not hrow.empty:
            max_temp = hrow["temp_f"].max()
            peak_row = hrow[hrow["temp_f"] == max_temp].iloc[0]
            peak_time = pd.Timestamp(peak_row["time"])

    return todays_high_f, peak_time, current_temp_f, None, hourly_df


def fetch_open_meteo_gfs(city: City) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[str]]:
    """
    Optional extra model. Failures should never break the app.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.timezone,
        "temperature_unit": "fahrenheit",
        "current_weather": "false",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max",
        "models": "gfs",
    }
    payload, err = safe_get(url, params=params)
    if err:
        return None, None, f"Open-Meteo (GFS) failed: {err}"

    daily_df, hourly_df, _ = parse_open_meteo(payload)
    if daily_df.empty:
        return None, None, "Open-Meteo (GFS): missing daily data"

    today_date = dt.date.today()
    row = daily_df[daily_df["date"] == today_date]
    if row.empty:
        todays_high_f = float(daily_df.iloc[0]["tmax_f"])
    else:
        todays_high_f = float(row.iloc[0]["tmax_f"])

    peak_time = None
    if not hourly_df.empty:
        hd = hourly_df.copy()
        hd["date"] = hd["time"].dt.date
        hrow = hd[hd["date"] == today_date]
        if not hrow.empty:
            max_temp = hrow["temp_f"].max()
            peak_row = hrow[hrow["temp_f"] == max_temp].iloc[0]
            peak_time = pd.Timestamp(peak_row["time"])

    return todays_high_f, peak_time, None


def fetch_nws_hourly(city: City) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[str]]:
    """
    NWS: points -> forecastHourly. If missing, fallback to 'forecast' (3–12 hr) max temperature.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}

    points_url = f"https://api.weather.gov/points/{city.latitude},{city.longitude}"
    points_payload, err = safe_get(points_url, headers=headers)
    if err or not points_payload:
        return None, None, f"NWS points failed: {err}"

    props = (points_payload.get("properties") or {})
    hourly_url = props.get("forecastHourly")
    forecast_url = props.get("forecast")

    # Try hourly first
    if hourly_url:
        hourly_payload, err2 = safe_get(hourly_url, headers=headers)
        if not err2 and hourly_payload:
            periods = (hourly_payload.get("properties") or {}).get("periods") or []
            if periods:
                # Use today's local date by parsing the ISO startTime (includes timezone offset)
                temps = []
                times = []
                today_date = dt.date.today()
                for p in periods:
                    t = p.get("temperature")
                    stime = p.get("startTime")
                    if t is None or stime is None:
                        continue
                    ts = pd.to_datetime(stime, errors="coerce")
                    if pd.isna(ts):
                        continue
                    if ts.date() != today_date:
                        continue
                    temps.append(float(t))
                    times.append(ts)
                if temps:
                    max_temp = max(temps)
                    peak_idx = temps.index(max_temp)
                    return max_temp, pd.Timestamp(times[peak_idx]), None

    # Fallback: use "forecast" (period-based)
    if forecast_url:
        f_payload, err3 = safe_get(forecast_url, headers=headers)
        if not err3 and f_payload:
            periods = (f_payload.get("properties") or {}).get("periods") or []
            if periods:
                today_date = dt.date.today()
                temps = []
                times = []
                for p in periods:
                    t = p.get("temperature")
                    stime = p.get("startTime")
                    if t is None or stime is None:
                        continue
                    ts = pd.to_datetime(stime, errors="coerce")
                    if pd.isna(ts):
                        continue
                    if ts.date() != today_date:
                        continue
                    temps.append(float(t))
                    times.append(ts)
                if temps:
                    max_temp = max(temps)
                    peak_idx = temps.index(max_temp)
                    return max_temp, pd.Timestamp(times[peak_idx]), None

    return None, None, "NWS failed: missing forecastHourly and forecast data"


def confidence_label(spread: float) -> str:
    if spread <= 1.5:
        return f"High (spread {spread:.1f}°)"
    if spread <= 2.5:
        return f"Medium (spread {spread:.1f}°)"
    return f"Low (spread {spread:.1f}°)"


def bracket_for_value(value_f: float, size: int, offset: int = 0) -> Tuple[int, int]:
    """
    size=2 → ranges like 78–79, 80–81, ...
    offset=1 would produce 79–80, 81–82, ...
    """
    base = math.floor((value_f - offset) / size) * size + offset
    lo = int(base)
    hi = int(base + size - 1)
    return lo, hi


def gaussian_mass(lo: float, hi: float, mu: float, sigma: float) -> float:
    # Approximate probability mass using CDF with error function
    if sigma <= 0:
        return 0.0
    def cdf(x):
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))
    return max(0.0, cdf(hi) - cdf(lo))


def build_probability_ladder(mu: float, sigma: float, size: int, offset: int) -> pd.DataFrame:
    # show 5 buckets centered around the bucket containing mu
    lo0, hi0 = bracket_for_value(mu, size=size, offset=offset)
    center = lo0
    buckets = []
    for k in range(-2, 3):
        lo = center + k * size
        hi = lo + size - 1
        p = gaussian_mass(lo - 0.5, hi + 0.5, mu=mu, sigma=sigma)  # continuity correction
        buckets.append((f"{lo}-{hi}", p * 100))
    df = pd.DataFrame(buckets, columns=["Bracket", "Probability %"])
    df["Probability %"] = df["Probability %"].round(2)
    # normalize (optional) so these 5 buckets sum to 100
    total = df["Probability %"].sum()
    if total > 0:
        df["Probability %"] = (df["Probability %"] * (100.0 / total)).round(2)
    return df


def format_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p")


# -----------------------------
# UI
# -----------------------------
st.title("Kalshi Weather Trading Dashboard")

city_key = st.selectbox("Select City", list(CITIES.keys()), index=0)
city = CITIES[city_key]

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4], index=1)
grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30)

# Many Kalshi markets have buckets that start at odd numbers (e.g., 79–80)
# This lets you align the suggested range to the market you see.
bucket_offset = st.selectbox("Bracket alignment (if your market shows 79–80 instead of 78–79)", [0, 1], index=0)
st.caption("Tip: If your Kalshi choices are 77–78 and 79–80, pick alignment = 1.")

try_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard ignores it automatically. (This removes the old “GFS error” problem.)")

# Fetch sources
errors = []

om_high, om_peak, om_current, om_err, om_hourly = fetch_open_meteo_best(city)
if om_err:
    errors.append(om_err)

nws_high, nws_peak, nws_err = fetch_nws_hourly(city)
if nws_err:
    errors.append(nws_err)

gfs_high, gfs_peak, gfs_err = (None, None, None)
if try_gfs:
    gfs_high, gfs_peak, gfs_err = fetch_open_meteo_gfs(city)
    if gfs_err:
        errors.append(gfs_err)

sources_used = []
if om_high is not None:
    sources_used.append("Open-Meteo (best)")
if nws_high is not None:
    sources_used.append("National Weather Service (NWS)")
if try_gfs and gfs_high is not None:
    sources_used.append("Open-Meteo GFS")

st.write(f"**Sources:** " + (" + ".join(sources_used) if sources_used else "None (errors below)"))

if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write(f"• {e}")

# If no valid high at all, stop
valid_highs = [(k, v) for (k, v) in [
    ("Open-Meteo (best)", om_high),
    ("NWS", nws_high),
    ("Open-Meteo (GFS)", gfs_high if try_gfs else None),
] if v is not None]

if not valid_highs:
    st.error("Could not fetch usable forecast data right now.")
    st.stop()

# Model agreement stats
high_values = [v for _, v in valid_highs]
spread = float(max(high_values) - min(high_values)) if len(high_values) >= 2 else 0.0
conf = confidence_label(spread)

# Base prediction = average of available highs (simple + robust)
pred_high = float(sum(high_values) / len(high_values))

# Peak time = prefer Open-Meteo hourly peak; else NWS; else GFS
peak_time = om_peak or nws_peak or gfs_peak

# Peak window display using grace minutes
peak_window = "—"
if peak_time is not None and not pd.isna(peak_time):
    start = (peak_time - pd.Timedelta(minutes=grace_minutes))
    end = (peak_time + pd.Timedelta(minutes=grace_minutes))
    peak_window = f"{format_time(start)} – {format_time(end)}"

# Heat spike detector (simple)
heat_note = None
adjusted_pred = pred_high
if om_current is not None and not om_hourly.empty:
    # compare current temp to nearest hourly forecast time
    now = pd.Timestamp.utcnow()
    # if open-meteo times are tz-local, we'll approximate by nearest row
    # (good enough for a "running hot/cool" alert)
    idx = (om_hourly["time"] - now).abs().idxmin() if len(om_hourly) else None
    if idx is not None and idx in om_hourly.index:
        forecast_now = float(om_hourly.loc[idx, "temp_f"])
        delta = float(om_current - forecast_now)
        if abs(delta) >= 2.0:
            direction = "hotter" if delta > 0 else "cooler"
            heat_note = f"Heat spike check: current is ~{abs(delta):.1f}° {direction} than the Open-Meteo hourly curve."
            # conservative adjustment
            adjusted_pred = pred_high + (0.4 * delta)

# Use adjusted prediction only to *inform*, but keep base pred for transparency
display_pred = adjusted_pred

# Suggested Kalshi range
lo, hi = bracket_for_value(display_pred, size=bracket_size, offset=bucket_offset)

# Nearby ranges
nearby = [
    bracket_for_value(display_pred - bracket_size, size=bracket_size, offset=bucket_offset),
    (lo, hi),
    bracket_for_value(display_pred + bracket_size, size=bracket_size, offset=bucket_offset),
]

# Sigma choice for probability ladder
# If models disagree, widen sigma.
sigma = max(1.0, spread / 1.8)  # heuristic

ladder_df = build_probability_ladder(mu=display_pred, sigma=sigma, size=bracket_size, offset=bucket_offset)

# -----------------------------
# Display
# -----------------------------
st.header(city.label)

st.subheader("Predicted Daily High (°F)")
st.metric(label="", value=f"{display_pred:.1f}")

st.subheader("Confidence")
st.write(conf)

st.subheader("Estimated Peak Time")
st.write(format_time(peak_time))
st.write(f"Peak window: {peak_window}")

if heat_note:
    st.info(heat_note)

st.divider()
st.subheader("Suggested Kalshi Range")
st.write(f"**{lo}–{hi}°F**")

st.write("Nearby ranges to watch:")
for a, b in nearby:
    tag = " (current)" if (a, b) == (lo, hi) else ""
    st.write(f"• {a}–{b}{tag}")

st.divider()
st.subheader("Kalshi Probability Ladder")
st.dataframe(ladder_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Model Agreement (Source Highs)")
agree_df = pd.DataFrame(
    [(name, float(val), format_time(om_peak if name.startswith("Open-Meteo (best)") else (nws_peak if name == "NWS" else gfs_peak)))
     for name, val in valid_highs],
    columns=["Source", "Daily High (°F)", "Peak Time"]
)
st.dataframe(agree_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Value Bet Check (you enter the Kalshi price)")
price_cents = st.number_input("Enter Kalshi YES price for main bracket (cents)", min_value=0, max_value=100, value=50, step=1)
model_prob = float(ladder_df[ladder_df["Bracket"] == f"{lo}-{hi}"]["Probability %"].iloc[0]) / 100.0 if not ladder_df.empty else None
if model_prob is not None:
    implied = price_cents / 100.0
    edge = (model_prob - implied)
    st.write(f"Model probability for **{lo}–{hi}°F** ≈ **{model_prob*100:.1f}%**")
    st.write(f"Implied probability at {price_cents}¢ ≈ **{implied*100:.1f}%**")
    st.write(f"Model edge ≈ **{edge*100:+.1f}%** (positive means model > market)")

st.divider()
with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {dt.date.today()}")
    if om_high is not None:
        st.write(f"Open-Meteo daily max (today): {om_high:.1f}")
    if nws_high is not None:
        st.write(f"NWS daily max (today): {nws_high:.1f}")
    if try_gfs and gfs_high is not None:
        st.write(f"Open-Meteo GFS daily max (today): {gfs_high:.1f}")
    if om_current is not None:
        st.write(f"Open-Meteo current temp: {om_current:.1f}")
