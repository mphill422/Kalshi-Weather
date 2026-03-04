import math
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# ----------------------------
# App config / UI
# ----------------------------
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

st.title("Kalshi Weather Trading Dashboard")


# ----------------------------
# City config
# ----------------------------
@dataclass(frozen=True)
class City:
    label: str
    latitude: float
    longitude: float
    tz: str  # IANA timezone


CITIES: Dict[str, City] = {
    "Austin, TX": City("Austin, TX", 30.2672, -97.7431, "America/Chicago"),
    "Dallas, TX": City("Dallas, TX", 32.7767, -96.7970, "America/Chicago"),
    "Houston, TX": City("Houston, TX", 29.7604, -95.3698, "America/Chicago"),
    "Phoenix, AZ": City("Phoenix, AZ", 33.4484, -112.0740, "America/Phoenix"),
    "New York City, NY": City("New York City, NY", 40.7128, -74.0060, "America/New_York"),
    "Atlanta, GA": City("Atlanta, GA", 33.7490, -84.3880, "America/New_York"),
    "Miami, FL": City("Miami, FL", 25.7617, -80.1918, "America/New_York"),
    "New Orleans, LA": City("New Orleans, LA", 29.9511, -90.0715, "America/Chicago"),
    "San Antonio, TX": City("San Antonio, TX", 29.4241, -98.4936, "America/Chicago"),
    "Los Angeles, CA": City("Los Angeles, CA", 34.0522, -118.2437, "America/Los_Angeles"),
}


# ----------------------------
# Helpers
# ----------------------------
def safe_get(url: str, headers: Optional[dict] = None, params: Optional[dict] = None, timeout: int = 20) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def now_in_tz(tz: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz)


def today_in_tz(tz: str) -> date:
    return now_in_tz(tz).date()


def parse_open_meteo_hourly(payload: dict, tz: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    # Open-Meteo returns local times if timezone param is set; treat as that tz
    df["time"] = df["time"].dt.tz_localize(tz)

    # Add supported fields if present
    for k in [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "apparent_temperature",
    ]:
        if k in hourly:
            df[k] = pd.to_numeric(hourly[k], errors="coerce")

    return df.dropna(subset=["time"])


def parse_open_meteo_daily(payload: dict, tz: str) -> pd.DataFrame:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"date": pd.to_datetime(times).dt.date})
    for k in [
        "temperature_2m_max",
        "temperature_2m_min",
    ]:
        if k in daily:
            df[k] = pd.to_numeric(daily[k], errors="coerce")

    return df


def open_meteo_fetch(city: City, models: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": city.tz,  # critical to avoid NYC-style day mismatch bugs
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timeformat": "iso8601",
        # 72 hours gives enough runway for “today” even if local time is late/early
        "forecast_days": 3,
        "hourly": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "wind_speed_10m",
            ]
        ),
        "daily": ",".join(["temperature_2m_max", "temperature_2m_min"]),
        # current= is supported by Open-Meteo; if it fails they still have hourly
        "current": ",".join(["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]),
    }
    if models:
        params["models"] = models
    return safe_get(url, params=params)


def nws_fetch_hourly(city: City) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    # NWS requires a valid User-Agent with contact info (policy). Put anything you want here.
    headers = {"User-Agent": "KalshiWeatherDashboard/1.0 (contact: you@example.com)", "Accept": "application/geo+json"}

    points_url = f"https://api.weather.gov/points/{city.latitude:.4f},{city.longitude:.4f}"
    points, err = safe_get(points_url, headers=headers, timeout=20)
    if err or not points:
        return None, f"NWS points failed: {err}"

    props = points.get("properties", {}) or {}
    # Reliability fix: forecastHourly is sometimes missing. Use gridpoints fallback.
    forecast_hourly_url = props.get("forecastHourly")
    grid_id = props.get("gridId")
    grid_x = props.get("gridX")
    grid_y = props.get("gridY")

    if not forecast_hourly_url and grid_id and grid_x is not None and grid_y is not None:
        forecast_hourly_url = f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly"

    if not forecast_hourly_url:
        return None, "NWS failed: no forecastHourly and no gridpoints fallback available."

    hourly_payload, err2 = safe_get(forecast_hourly_url, headers=headers, timeout=25)
    if err2 or not hourly_payload:
        return None, f"NWS hourly failed: {err2}"

    periods = (hourly_payload.get("properties", {}) or {}).get("periods", []) or []
    if not periods:
        return None, "NWS hourly returned no periods."

    rows = []
    for p in periods:
        # NWS period startTime is ISO with offset
        t = pd.to_datetime(p.get("startTime"), errors="coerce")
        if pd.isna(t):
            continue
        temp_f = p.get("temperature")
        # NWS might be in F already; if not, skip conversion complexity here.
        rows.append(
            {
                "time": t.tz_convert(city.tz) if t.tzinfo is not None else t.tz_localize(city.tz),
                "temperature_f": float(temp_f) if temp_f is not None else None,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["time", "temperature_f"])
    if df.empty:
        return None, "NWS hourly parsed empty."
    return df, None


def daily_high_from_hourly(df_hourly: pd.DataFrame, tz: str, target_date: date) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    if df_hourly is None or df_hourly.empty:
        return None, None
    if "time" not in df_hourly:
        return None, None

    # Identify temp column
    temp_col = None
    for c in ["temperature_2m", "temperature_f"]:
        if c in df_hourly.columns:
            temp_col = c
            break
    if not temp_col:
        return None, None

    d = df_hourly.copy()
    # ensure tz
    if d["time"].dt.tz is None:
        d["time"] = d["time"].dt.tz_localize(tz)
    else:
        d["time"] = d["time"].dt.tz_convert(tz)

    d["local_date"] = d["time"].dt.date
    d_today = d[d["local_date"] == target_date].dropna(subset=[temp_col])
    if d_today.empty:
        return None, None

    idx = d_today[temp_col].astype(float).idxmax()
    peak_temp = float(d_today.loc[idx, temp_col])
    peak_time = d_today.loc[idx, "time"]
    return peak_temp, peak_time


def kalshi_bin_for_temp(temp_f: float, size: int, offset: int) -> Tuple[int, int]:
    """
    size: bin width in degrees (1,2,3,...)
    offset: starting alignment (0 = even bins for size=2 like 78–79, 1 = odd bins like 77–78)
    Returns inclusive integer bounds (low, high).
    """
    # bins start at: offset + k*size
    low = offset + size * math.floor((temp_f - offset) / size)
    high = low + (size - 1)
    return int(low), int(high)


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def prob_temp_in_bin(mu: float, sigma: float, low: float, high: float) -> float:
    # treat bin as [low, high+1) in continuous space
    return max(0.0, min(1.0, normal_cdf(high + 1.0, mu, sigma) - normal_cdf(low, mu, sigma)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Controls
# ----------------------------
city_label = st.selectbox("Select City", list(CITIES.keys()), index=0)
city = CITIES[city_label]

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4], index=1)
st.caption("This controls how wide each Kalshi temperature bucket is (e.g., 2°F → 78–79, 80–81, etc.).")

# IMPORTANT: this fixes Miami-style “77–78 vs 78–79” mismatch
kalshi_offset = st.selectbox(
    "Kalshi bracket alignment (offset)",
    [0, 1],
    index=0,
    format_func=lambda x: "0 (even-start bins like 78–79)" if x == 0 else "1 (odd-start bins like 77–78)",
)

grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=1)

use_open_meteo_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard will ignore it automatically. (This removes the old “GFS error” problem.)")


# ----------------------------
# Fetch sources
# ----------------------------
sources_used = []
errors: List[str] = []

# Open-Meteo (default)
om_payload, om_err = open_meteo_fetch(city, models=None)
om_hourly = None
om_daily = None
om_current = None
if om_payload:
    om_hourly = parse_open_meteo_hourly(om_payload, city.tz)
    om_daily = parse_open_meteo_daily(om_payload, city.tz)
    om_current = (om_payload.get("current") or {}) if isinstance(om_payload.get("current"), dict) else None
    sources_used.append("Open-Meteo (best)")
else:
    errors.append(f"Open-Meteo failed: {om_err}")

# Open-Meteo (GFS)
gfs_payload = None
gfs_hourly = None
if use_open_meteo_gfs:
    gfs_payload, gfs_err = open_meteo_fetch(city, models="gfs")
    if gfs_payload:
        gfs_hourly = parse_open_meteo_hourly(gfs_payload, city.tz)
        sources_used.append("Open-Meteo GFS")
    else:
        errors.append(f"Open-Meteo (GFS) failed: {gfs_err}")

# NWS hourly
nws_df, nws_err = nws_fetch_hourly(city)
if nws_df is not None and not nws_df.empty:
    sources_used.append("National Weather Service (NWS)")
else:
    if nws_err:
        errors.append(nws_err)

st.write(f"Sources: " + (" + ".join(sources_used) if sources_used else "None"))


# ----------------------------
# Build “today” prediction
# ----------------------------
target_date = today_in_tz(city.tz)

# Compute daily highs from each source that can produce it
source_highs: List[Tuple[str, float, Optional[pd.Timestamp]]] = []

# Open-Meteo daily max if available for today
if isinstance(om_daily, pd.DataFrame) and not om_daily.empty and "temperature_2m_max" in om_daily.columns:
    row = om_daily[om_daily["date"] == target_date]
    if not row.empty:
        val = row["temperature_2m_max"].iloc[0]
        if pd.notna(val):
            # Peak time from hourly curve (better for trading)
            peak_temp, peak_time = daily_high_from_hourly(om_hourly, city.tz, target_date) if isinstance(om_hourly, pd.DataFrame) else (None, None)
            source_highs.append(("Open-Meteo (daily)", float(val), peak_time))

# Open-Meteo hourly-derived max (today)
if isinstance(om_hourly, pd.DataFrame) and not om_hourly.empty:
    peak_temp, peak_time = daily_high_from_hourly(om_hourly, city.tz, target_date)
    if peak_temp is not None:
        source_highs.append(("Open-Meteo (hourly)", float(peak_temp), peak_time))

# NWS hourly-derived max (today)
if isinstance(nws_df, pd.DataFrame) and not nws_df.empty:
    peak_temp, peak_time = daily_high_from_hourly(nws_df.rename(columns={"temperature_f": "temperature_f"}), city.tz, target_date)
    if peak_temp is not None:
        source_highs.append(("NWS (hourly)", float(peak_temp), peak_time))

# GFS hourly-derived max (today)
if isinstance(gfs_hourly, pd.DataFrame) and not gfs_hourly.empty:
    peak_temp, peak_time = daily_high_from_hourly(gfs_hourly, city.tz, target_date)
    if peak_temp is not None:
        source_highs.append(("Open-Meteo GFS (hourly)", float(peak_temp), peak_time))

if not source_highs:
    st.error("Could not compute a daily high from available sources.")
    if errors:
        with st.expander("See errors"):
            for e in errors:
                st.write("• " + e)
    st.stop()

# Choose “best” high:
# Prefer Open-Meteo hourly (most consistent), then daily, then NWS, then GFS.
priority = {
    "Open-Meteo (hourly)": 0,
    "Open-Meteo (daily)": 1,
    "NWS (hourly)": 2,
    "Open-Meteo GFS (hourly)": 3,
}
source_highs_sorted = sorted(source_highs, key=lambda x: priority.get(x[0], 99))
best_name, best_high, best_peak_time = source_highs_sorted[0]

# Spread/confidence (using all available highs)
high_vals = [h for _, h, _ in source_highs]
spread = (max(high_vals) - min(high_vals)) if len(high_vals) >= 2 else 0.0

if spread <= 1.0:
    conf = "High"
elif spread <= 2.0:
    conf = "Medium"
else:
    conf = "Low"

# Estimate peak time from best_peak_time; if missing, use open-meteo hourly peak
if best_peak_time is None and isinstance(om_hourly, pd.DataFrame):
    _, best_peak_time = daily_high_from_hourly(om_hourly, city.tz, target_date)

# Peak window
if best_peak_time is not None:
    window_start = (best_peak_time - pd.Timedelta(minutes=grace_minutes)).strftime("%I:%M %p")
    window_end = (best_peak_time + pd.Timedelta(minutes=grace_minutes)).strftime("%I:%M %p")
    peak_str = best_peak_time.strftime("%I:%M %p")
else:
    peak_str = "Unknown"
    window_start = "—"
    window_end = "—"

# Display headline
st.header(city.label)
st.subheader("Predicted Daily High (°F)")
st.write(f"**{best_high:.1f}**")
st.caption(f"Confidence: **{conf}** (spread {spread:.1f}°) • Best source: {best_name}")

st.subheader("Estimated Peak Time")
st.write(f"**{peak_str}**")
st.write(f"Peak window: **{window_start} — {window_end}**")


# ----------------------------
# Current conditions (Open-Meteo)
# ----------------------------
st.subheader("Current Conditions (Open-Meteo)")

current_temp = None
current_hum = None
current_wind = None

if isinstance(om_current, dict) and om_current:
    # Open-Meteo current object uses same variable names
    current_temp = om_current.get("temperature_2m")
    current_hum = om_current.get("relative_humidity_2m")
    current_wind = om_current.get("wind_speed_10m")

# fallback: latest hourly
if (current_temp is None or pd.isna(current_temp)) and isinstance(om_hourly, pd.DataFrame) and not om_hourly.empty:
    latest = om_hourly.sort_values("time").iloc[-1]
    current_temp = latest.get("temperature_2m")
    current_hum = latest.get("relative_humidity_2m")
    current_wind = latest.get("wind_speed_10m")

c1, c2, c3 = st.columns(3)
c1.metric("Current Temp (°F)", f"{float(current_temp):.1f}" if current_temp is not None and not pd.isna(current_temp) else "—")
c2.metric("Humidity (%)", f"{float(current_hum):.0f}" if current_hum is not None and not pd.isna(current_hum) else "—")
c3.metric("Wind (mph)", f"{float(current_wind):.1f}" if current_wind is not None and not pd.isna(current_wind) else "—")


# ----------------------------
# NYC / “impossible” guardrail
# ----------------------------
if current_temp is not None and not pd.isna(current_temp):
    if float(current_temp) > best_high + 1.0:
        st.warning(
            "⚠️ Data mismatch detected: current temperature is already above the predicted daily high. "
            "This usually means a day/timezone alignment issue from one source. "
            "Consider changing bracket alignment or relying on Open-Meteo hourly today."
        )


# ----------------------------
# Suggested Kalshi range (Daily High)
# ----------------------------
st.header("Suggested Kalshi Range (Daily High)")

bin_low, bin_high = kalshi_bin_for_temp(best_high, bracket_size, kalshi_offset)
st.write(f"**{bin_low}–{bin_high}°F**")

# Nearby bins to watch
near_bins = []
for k in [-1, 0, 1]:
    low = bin_low + k * bracket_size
    high = low + (bracket_size - 1)
    near_bins.append((low, high))

st.write("Nearby ranges to watch:")
for low, high in near_bins:
    label = f"{low}–{high}"
    if low == bin_low and high == bin_high:
        label += " (current)"
    st.write(f"• {label}")


# ----------------------------
# Model Agreement table (Source highs)
# ----------------------------
st.header("Model Agreement (Source Highs)")

rows = []
for name, h, peak_t in sorted(source_highs, key=lambda x: x[0]):
    rows.append(
        {
            "Source": name,
            "Daily High (°F)": round(h, 1),
            "Peak Time": peak_t.strftime("%I:%M %p") if isinstance(peak_t, pd.Timestamp) else "—",
        }
    )

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------
# Probability ladder
# ----------------------------
st.header("Kalshi Probability Ladder")

# Convert spread to sigma for a simple forecast distribution.
# Lower bound keeps ladder from becoming “all or nothing” too often.
sigma = max(0.8, spread / 2.5) if spread > 0 else 0.8

# Build bins around the best_high bin
ladder_bins = []
# show 5 bins by default
center_low = bin_low
for i in range(-2, 3):
    low = center_low + i * bracket_size
    high = low + (bracket_size - 1)
    p = prob_temp_in_bin(best_high, sigma, low, high) * 100.0
    ladder_bins.append({"Bracket": f"{low}–{high}", "Probability %": round(p, 1)})

st.dataframe(pd.DataFrame(ladder_bins), use_container_width=True, hide_index=True)

# Extra: if user wants raw numbers
with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): {target_date.isoformat()}")
    st.write(f"Best high (used): {best_high:.1f}°F")
    st.write(f"Spread across sources: {spread:.1f}°F")
    st.write(f"Sigma used for ladder: {sigma:.2f}")


# ----------------------------
# Value Bet Check (user enters Kalshi price)
# ----------------------------
st.header("Value Bet Check (you enter the Kalshi price)")

st.caption("This does **not** place a bet. It just compares the model probability vs the market implied probability.")

price_cents = st.number_input(
    f"Enter Kalshi YES price for main bracket ({bin_low}–{bin_high}) in cents (0–100)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

model_p_main = prob_temp_in_bin(best_high, sigma, bin_low, bin_high)
market_p = price_cents / 100.0
edge = (model_p_main - market_p) * 100.0

st.write(f"Model probability for **{bin_low}–{bin_high}** ≈ **{model_p_main*100:.1f}%**")
st.write(f"Market implied probability ≈ **{market_p*100:.1f}%**")
st.write(f"Model edge ≈ **{edge:+.1f} percentage points**")

if edge > 5:
    st.success("Model is meaningfully higher than the market implied probability (potential value).")
elif edge < -5:
    st.error("Market implied probability is meaningfully higher than the model (likely overpriced).")
else:
    st.info("Close call (small edge either way).")


# ----------------------------
# Peak-time heat spike detector
# ----------------------------
st.header("Peak-time Heat Spike Detector")

spike_threshold = st.slider("Spike alert threshold (°F above forecast curve)", 1.0, 6.0, 2.0, 0.5)

spike_msg = "Not enough data."
if isinstance(om_hourly, pd.DataFrame) and not om_hourly.empty and current_temp is not None and not pd.isna(current_temp):
    # Forecast curve value near “now”
    now_ts = now_in_tz(city.tz)
    d = om_hourly.copy()
    d["time"] = d["time"].dt.tz_convert(city.tz)
    d = d.sort_values("time")

    # nearest timestamp
    idx = (d["time"] - now_ts).abs().idxmin()
    forecast_now = float(d.loc[idx, "temperature_2m"]) if "temperature_2m" in d.columns and pd.notna(d.loc[idx, "temperature_2m"]) else None

    if forecast_now is not None:
        diff = float(current_temp) - forecast_now
        if diff >= spike_threshold:
            spike_msg = f"🔥 Spike detected: current is **{diff:.1f}°F** above forecast curve near now."
        else:
            spike_msg = f"No spike: current is **{diff:.1f}°F** vs forecast curve near now."
    else:
        spike_msg = "Could not compute forecast curve near now."

st.write(spike_msg)


# ----------------------------
# Errors panel (non-fatal)
# ----------------------------
if errors:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errors:
            st.write("• " + e)
