import math
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Kalshi Weather MVP", layout="wide")

APP_TITLE = "Kalshi Weather MVP – Daily High (Multi-Source)"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-mvp/1.0 (contact: none)"
REQ_TIMEOUT = 12

# Cities you requested + the core ones we’ve used
CITIES = {
    "Miami": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New York City": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "New Orleans": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "Austin": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "San Antonio": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Las Vegas": {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles"},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

# ----------------------------
# Helpers
# ----------------------------
def http_get_json(url: str) -> dict | None:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQ_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__url__": url}


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def today_local(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def parse_iso_to_local_dt(ts: str, tz_name: str) -> datetime | None:
    try:
        # Open-Meteo returns ISO without offset when timezone parameter is used.
        # Example: "2026-03-05T13:00"
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def round_bracket_label(low: int, size: int) -> str:
    return f"{low}–{low + size}"


def normal_cdf(x: float) -> float:
    # Standard normal CDF approximation via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bracket_probability(mean: float, sigma: float, low: float, high: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean < high) else 0.0
    z1 = (low - mean) / sigma
    z2 = (high - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))


def compute_sigma(source_highs: list[float]) -> float:
    """
    We want something stable, not overconfident:
    - base sigma ~ 1.6°F
    - plus 0.55 * (spread across sources)
    """
    if not source_highs:
        return 2.5
    spread = max(source_highs) - min(source_highs) if len(source_highs) >= 2 else 0.0
    return max(1.4, 1.6 + 0.55 * spread)


# ----------------------------
# Data Sources
# ----------------------------
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None) -> dict:
    # temperature_unit=fahrenheit makes all temps °F.
    base = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&temperature_unit=fahrenheit"
        f"&hourly=temperature_2m"
        f"&daily=temperature_2m_max"
        f"&timezone={tz}"
        f"&forecast_days=2"
    )
    if model:
        # Open-Meteo supports a 'models=' parameter for some backends.
        # If it fails, we catch it gracefully.
        base += f"&models={model}"

    j = http_get_json(base)
    return j


def extract_open_meteo_today(j: dict, tz: str) -> tuple[float | None, pd.DataFrame | None, str | None]:
    if not j or "__error__" in j:
        return None, None, (j.get("__error__") if isinstance(j, dict) else "Unknown error")

    try:
        daily = j.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        daily_time = daily.get("time", [])
        if not highs or not daily_time:
            return None, None, "Open-Meteo missing daily high fields"

        t0 = today_local(tz).isoformat()

        # Find today's index
        idx = 0
        if t0 in daily_time:
            idx = daily_time.index(t0)

        daily_high = safe_float(highs[idx])

        hourly = j.get("hourly", {})
        ht = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                dt = parse_iso_to_local_dt(ts, tz)
                if dt is None:
                    continue
                rows.append({"dt": dt, "temp_f": safe_float(temp)})
            df = pd.DataFrame(rows).dropna()
        else:
            df = None

        # Keep only today's hours
        if df is not None and not df.empty:
            tday = today_local(tz)
            df = df[df["dt"].dt.date == tday].copy()
            if df.empty:
                df = None

        return daily_high, df, None
    except Exception as e:
        return None, None, f"Open-Meteo parse error: {e}"


def fetch_nws_hourly_high(lat: float, lon: float, tz: str) -> tuple[float | None, pd.DataFrame | None, str | None]:
    """
    NWS: points -> forecastHourly (periods)
    We'll compute today's max temperature from the hourly forecast periods (°F).
    """
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        p = http_get_json(points_url)
        if not p or "__error__" in p:
            return None, None, f"NWS points error: {p.get('__error__')}"

        props = p.get("properties", {})
        hourly_url = props.get("forecastHourly")
        if not hourly_url:
            return None, None, "NWS missing forecastHourly URL"

        h = http_get_json(hourly_url)
        if not h or "__error__" in h:
            return None, None, f"NWS hourly error: {h.get('__error__')}"

        periods = (h.get("properties") or {}).get("periods") or []
        if not periods:
            return None, None, "NWS hourly periods missing/empty"

        rows = []
        for per in periods:
            start = per.get("startTime")
            temp = per.get("temperature")
            unit = per.get("temperatureUnit")

            if unit and unit.upper() != "F":
                # We only handle F reliably; skip otherwise
                continue

            try:
                # NWS is offset-aware ISO
                dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
            except Exception:
                continue

            rows.append({"dt": dt, "temp_f": safe_float(temp)})

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return None, None, "NWS hourly parse produced no rows"

        # Today's max
        tday = today_local(tz)
        df_today = df[df["dt"].dt.date == tday].copy()
        if df_today.empty:
            return None, None, "NWS hourly has no rows for today (maybe too late/early)"

        daily_high = float(df_today["temp_f"].max())
        return daily_high, df_today, None

    except Exception as e:
        return None, None, f"NWS error: {e}"


# ----------------------------
# UI Controls
# ----------------------------
left, right = st.columns([1, 1])
with left:
    city = st.selectbox("City", list(CITIES.keys()))
with right:
    st.caption("Tip: If a source errors, the app still runs — it just excludes that source from consensus.")

info = CITIES[city]
tz = info["tz"]

controls = st.expander("Settings", expanded=True)
with controls:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_gfs = st.toggle("Include Open-Meteo GFS model (extra check)", value=False)
    with col2:
        use_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    with col3:
        bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=0)
    with col4:
        grace_minutes = st.slider("Grace minutes after 10:30 local", min_value=0, max_value=120, value=45, step=5)

    lock_time_local = time(10, 30)

# ----------------------------
# Fetch + Compute
# ----------------------------
with st.spinner("Fetching forecasts…"):
    # Open-Meteo default
    om = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    om_high, om_hourly, om_err = extract_open_meteo_today(om, tz)

    # Open-Meteo GFS (optional)
    gfs_high = None
    gfs_hourly = None
    gfs_err = None
    if use_gfs:
        # model name can vary; gfs_seamless is commonly supported by Open-Meteo
        om_gfs = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        gfs_high, gfs_hourly, gfs_err = extract_open_meteo_today(om_gfs, tz)

    # NWS (optional)
    nws_high = None
    nws_hourly = None
    nws_err = None
    if use_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)

# ----------------------------
# Display source status
# ----------------------------
status_rows = []
if om_high is not None:
    status_rows.append(("Open-Meteo", f"{om_high:.1f}°F", "OK"))
else:
    status_rows.append(("Open-Meteo", "—", om_err or "Error"))

if use_gfs:
    if gfs_high is not None:
        status_rows.append(("Open-Meteo (GFS)", f"{gfs_high:.1f}°F", "OK"))
    else:
        status_rows.append(("Open-Meteo (GFS)", "—", gfs_err or "Error"))

if use_nws:
    if nws_high is not None:
        status_rows.append(("NWS (api.weather.gov)", f"{nws_high:.1f}°F", "OK"))
    else:
        status_rows.append(("NWS (api.weather.gov)", "—", nws_err or "Error"))

st.subheader(f"{city} – Today’s High Forecasts (°F)")
st.table(pd.DataFrame(status_rows, columns=["Source", "Today High", "Status"]))

# ----------------------------
# Consensus + Suggested bracket
# ----------------------------
source_highs = [x for x in [om_high, gfs_high, nws_high] if isinstance(x, (int, float)) and x is not None]
if not source_highs:
    st.error("No valid sources returned a high for today. Try again, or toggle off a failing source.")
    st.stop()

consensus = float(sum(source_highs) / len(source_highs))
sigma = compute_sigma(source_highs)

spread = max(source_highs) - min(source_highs) if len(source_highs) > 1 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")

# Build candidate brackets around consensus
# We test +/- 6°F around the rounded consensus to find the max-prob bracket.
center = int(round(consensus))
candidates = []
for low in range(center - 6, center + 7):
    high = low + bracket_size
    p = bracket_probability(consensus, sigma, low, high)
    candidates.append((low, high, p))

best = max(candidates, key=lambda x: x[2])
best_low, best_high, best_p = best

st.success(
    f"Suggested Kalshi bracket: **{round_bracket_label(best_low, bracket_size)}**  "
    f"(model probability ≈ **{best_p*100:.0f}%**)"
)

# Show top 5 brackets for context (so you can see hedges if needed)
top5 = sorted(candidates, key=lambda x: x[2], reverse=True)[:5]
top_df = pd.DataFrame(
    [{"Bracket": f"{lo}–{hi}", "Model Prob %": round(p*100, 1)} for lo, hi, p in top5]
)
st.caption("Top bracket candidates (model-based):")
st.dataframe(top_df, use_container_width=True, hide_index=True)

# ----------------------------
# Timing / Decision window
# ----------------------------
now_local = datetime.now(ZoneInfo(tz))
lock_dt = datetime.combine(now_local.date(), lock_time_local, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.subheader("Decision Window")
st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}**  "
    f"| Target lock: **10:30 AM**  "
    f"| With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)

if now_local <= deadline_dt:
    st.info("You are inside the preferred betting window (or within grace).")
else:
    st.warning("You are past the preferred window. You can still bet, but edge typically shrinks as the day progresses.")

# ----------------------------
# Hourly chart + peak window
# ----------------------------
st.subheader("Hourly temperature curve (today)")
hourly_source_name = None
hourly_df = None

# Prefer Open-Meteo hourly; fall back to NWS hourly
if om_hourly is not None and not om_hourly.empty:
    hourly_df = om_hourly.sort_values("dt")
    hourly_source_name = "Open-Meteo"
elif nws_hourly is not None and not nws_hourly.empty:
    hourly_df = nws_hourly.sort_values("dt")
    hourly_source_name = "NWS"

if hourly_df is not None and not hourly_df.empty:
    plot_df = hourly_df.copy()
    plot_df["time"] = plot_df["dt"].dt.strftime("%I:%M %p")
    st.caption(f"Hourly source used: {hourly_source_name}")
    st.line_chart(plot_df.set_index("dt")["temp_f"])

    # Peak window = hottest 90-minute span approximation (max of rolling 2 hours)
    # If sparse, just show the max hour.
    peak_row = plot_df.loc[plot_df["temp_f"].idxmax()]
    peak_time = peak_row["dt"]
    peak_temp = float(peak_row["temp_f"])

    st.write(f"Peak hour (from {hourly_source_name}): **{peak_time.strftime('%I:%M %p')}** at **{peak_temp:.1f}°F**")
else:
    st.caption("Hourly curve unavailable (sources returned daily high but not usable hourly).")

st.divider()
st.caption(
    "Note: This tool estimates probability from cross-source agreement. It can’t see Kalshi’s exact station/reporting rules, "
    "so always sanity-check the bracket options Kalshi offers before placing size."
)
