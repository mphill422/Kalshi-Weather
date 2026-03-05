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

APP_TITLE = "Kalshi Weather MVP – Daily High (Multi-Source) [Auto Weights]"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-mvp/1.2 (contact: none)"
REQ_TIMEOUT = 12

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

# Daytime window to avoid odd “late peak” artifacts in hourly feeds
DAY_START_HOUR = 10
DAY_END_HOUR = 18

# Default weights (good baseline)
DEFAULT_W_OPEN = 0.55
DEFAULT_W_NWS = 0.30
DEFAULT_W_GFS = 0.15

# ----------------------------
# Helpers
# ----------------------------
def http_get_json(url: str) -> dict | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__url__": url}


def safe_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def today_local(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def parse_iso_to_local_dt(ts: str, tz_name: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bracket_prob_inclusive(mean: float, sigma: float, low: float, high_inclusive: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean <= high_inclusive) else 0.0
    z1 = ((low - 0.5) - mean) / sigma
    z2 = ((high_inclusive + 0.5) - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))


def compute_sigma_from_spread(source_highs: list[float]) -> float:
    fallback = 2.2
    if not source_highs or len(source_highs) < 2:
        return fallback
    spread = max(source_highs) - min(source_highs)
    sigma = max(fallback, spread / 3.0)
    return max(1.0, min(6.0, sigma))


def daytime_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[(df["dt"].dt.hour >= DAY_START_HOUR) & (df["dt"].dt.hour <= DAY_END_HOUR)].copy()


def normalize_weights(w_open: float, w_nws: float, w_gfs: float, use_gfs: bool) -> tuple[float, float, float]:
    # If GFS not used, force weight to 0 and renormalize.
    if not use_gfs:
        w_gfs = 0.0
    total = w_open + w_nws + w_gfs
    if total <= 0:
        # fallback
        w_open, w_nws, w_gfs = (DEFAULT_W_OPEN, DEFAULT_W_NWS, DEFAULT_W_GFS if use_gfs else 0.0)
        total = w_open + w_nws + w_gfs
    return w_open / total, w_nws / total, w_gfs / total


def auto_weights(om_high: float | None, nws_high: float | None, gfs_high: float | None, use_gfs: bool) -> tuple[float, float, float]:
    """
    AUTO logic:
    - Start with defaults
    - If NWS deviates strongly (cold bias day), downweight NWS a bit
    - If GFS is extreme vs others, downweight GFS
    """
    w_open, w_nws, w_gfs = DEFAULT_W_OPEN, DEFAULT_W_NWS, DEFAULT_W_GFS

    vals = [v for v in [om_high, nws_high, gfs_high if use_gfs else None] if v is not None]
    if len(vals) >= 2:
        med = sorted(vals)[len(vals)//2]

        # If NWS exists and is far from median, reduce it
        if nws_high is not None and abs(nws_high - med) >= 3.0:
            w_nws *= 0.6
            w_open *= 1.1

        # If GFS exists and is far from median, reduce it
        if use_gfs and gfs_high is not None and abs(gfs_high - med) >= 3.5:
            w_gfs *= 0.5
            w_open *= 1.1

    return normalize_weights(w_open, w_nws, w_gfs, use_gfs)


def weighted_mean(vals: list[float], wts: list[float]) -> float:
    sw = sum(wts)
    if sw <= 0:
        return sum(vals) / len(vals)
    return sum(v * w for v, w in zip(vals, wts)) / sw


# ----------------------------
# Data Sources
# ----------------------------
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None) -> dict:
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
        base += f"&models={model}"
    return http_get_json(base)


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
        idx = daily_time.index(t0) if t0 in daily_time else 0
        daily_high = safe_float(highs[idx])

        hourly = j.get("hourly", {})
        ht = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        df = None
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                dt = parse_iso_to_local_dt(ts, tz)
                if dt is None:
                    continue
                rows.append({"dt": dt, "temp_f": safe_float(temp)})
            df = pd.DataFrame(rows).dropna()

        if df is not None and not df.empty:
            tday = today_local(tz)
            df = df[df["dt"].dt.date == tday].copy()
            if df.empty:
                df = None

        return daily_high, df, None
    except Exception as e:
        return None, None, f"Open-Meteo parse error: {e}"


def fetch_nws_hourly_high(lat: float, lon: float, tz: str) -> tuple[float | None, pd.DataFrame | None, str | None]:
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        p = http_get_json(points_url)
        if not p or "__error__" in p:
            return None, None, f"NWS points error: {p.get('__error__')}"

        hourly_url = (p.get("properties", {}) or {}).get("forecastHourly")
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
                continue
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
            except Exception:
                continue
            rows.append({"dt": dt, "temp_f": safe_float(temp)})

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return None, None, "NWS hourly parse produced no rows"

        tday = today_local(tz)
        df_today = df[df["dt"].dt.date == tday].copy()
        if df_today.empty:
            return None, None, "NWS hourly has no rows for today"

        df_day = daytime_filter(df_today)
        if df_day is not None and not df_day.empty:
            daily_high = float(df_day["temp_f"].max())
            return daily_high, df_today, None

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

    use_nowcast = st.toggle("Use nowcast (heating-rate adjustment)", value=True)

    # NEW: Auto weights toggle
    use_auto_weights = st.toggle("Auto weights (recommended)", value=True)

    show_advanced = st.toggle("Show advanced weight sliders", value=False)
    if show_advanced and not use_auto_weights:
        w_open = st.slider("Weight Open-Meteo", 0.0, 1.0, DEFAULT_W_OPEN, 0.05)
        w_nws = st.slider("Weight NWS", 0.0, 1.0, DEFAULT_W_NWS, 0.05)
        w_gfs = st.slider("Weight GFS", 0.0, 1.0, DEFAULT_W_GFS, 0.05)
    else:
        w_open, w_nws, w_gfs = DEFAULT_W_OPEN, DEFAULT_W_NWS, DEFAULT_W_GFS

    lock_time_local = time(10, 30)

# ----------------------------
# Fetch + Compute
# ----------------------------
with st.spinner("Fetching forecasts…"):
    om = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    om_high, om_hourly, om_err = extract_open_meteo_today(om, tz)

    gfs_high = None
    gfs_hourly = None
    gfs_err = None
    if use_gfs:
        om_gfs = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        gfs_high, gfs_hourly, gfs_err = extract_open_meteo_today(om_gfs, tz)

    nws_high = None
    nws_hourly = None
    nws_err = None
    if use_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)

# ----------------------------
# Status table
# ----------------------------
status_rows = []
status_rows.append(("Open-Meteo", f"{om_high:.1f}°F" if om_high is not None else "—", "OK" if om_high is not None else (om_err or "Error")))
if use_gfs:
    status_rows.append(("Open-Meteo (GFS)", f"{gfs_high:.1f}°F" if gfs_high is not None else "—", "OK" if gfs_high is not None else (gfs_err or "Error")))
if use_nws:
    status_rows.append(("NWS (api.weather.gov)", f"{nws_high:.1f}°F" if nws_high is not None else "—", "OK" if nws_high is not None else (nws_err or "Error")))

st.subheader(f"{city} – Today’s High Forecasts (°F)")
st.table(pd.DataFrame(status_rows, columns=["Source", "Today High", "Status"]))

# ----------------------------
# Consensus + Sigma
# ----------------------------
vals = []
labs = []
if om_high is not None:
    vals.append(float(om_high)); labs.append("Open-Meteo")
if use_nws and nws_high is not None:
    vals.append(float(nws_high)); labs.append("NWS")
if use_gfs and gfs_high is not None:
    vals.append(float(gfs_high)); labs.append("GFS")

if not vals:
    st.error("No valid sources returned a high for today.")
    st.stop()

# Decide weights automatically (recommended) or manual defaults
if use_auto_weights:
    w_open, w_nws, w_gfs = auto_weights(om_high, nws_high if use_nws else None, gfs_high if use_gfs else None, use_gfs)
else:
    w_open, w_nws, w_gfs = normalize_weights(w_open, w_nws, w_gfs, use_gfs)

# Build weighted mean using available sources
vals_w = []
wts_w = []
for v, lab in zip(vals, labs):
    if lab == "Open-Meteo":
        vals_w.append(v); wts_w.append(w_open)
    elif lab == "NWS":
        vals_w.append(v); wts_w.append(w_nws)
    elif lab == "GFS":
        vals_w.append(v); wts_w.append(w_gfs)

consensus = float(weighted_mean(vals_w, wts_w))

spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
sigma = compute_sigma_from_spread(vals)

c1, c2, c3 = st.columns(3)
c1.metric("Consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")

st.caption(f"Effective weights used → Open-Meteo: {w_open:.2f} | NWS: {w_nws:.2f} | GFS: {w_gfs:.2f}")

# ----------------------------
# Suggested bracket + Top candidates
# ----------------------------
mu = consensus

center = int(round(mu))
candidates = []
for low in range(center - 6, center + 7):
    high = low + bracket_size - 1
    p = bracket_prob_inclusive(mu, sigma, low, high)
    candidates.append((low, high, p))

best_low, best_high, best_p = max(candidates, key=lambda x: x[2])

st.success(f"Suggested Kalshi bracket: **{best_low}–{best_high}** (model probability ≈ **{best_p*100:.0f}%**)")

top5 = sorted(candidates, key=lambda x: x[2], reverse=True)[:5]
top_df = pd.DataFrame([{"Bracket": f"{lo}–{hi}", "Model Prob %": round(p*100, 1)} for lo, hi, p in top5])
st.caption("Top bracket candidates (model-based):")
st.dataframe(top_df, use_container_width=True, hide_index=True)

# ----------------------------
# Decision Window
# ----------------------------
st.subheader("Decision Window")
now_local = datetime.now(ZoneInfo(tz))
lock_dt = datetime.combine(now_local.date(), lock_time_local, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}** | Target lock: **10:30 AM** | With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)
if now_local <= deadline_dt:
    st.info("You are inside the preferred betting window (or within grace).")
else:
    st.warning("You are past the preferred window. Edge typically shrinks as the day progresses.")

# ----------------------------
# Hourly curve + peak (kept)
# ----------------------------
st.subheader("Hourly temperature curve (today)")
hourly_source_name = None
hourly_df = None

if om_hourly is not None and not om_hourly.empty:
    hourly_df = om_hourly.sort_values("dt")
    hourly_source_name = "Open-Meteo"
elif nws_hourly is not None and not nws_hourly.empty:
    hourly_df = nws_hourly.sort_values("dt")
    hourly_source_name = "NWS"

if hourly_df is not None and not hourly_df.empty:
    st.caption(f"Hourly source used: {hourly_source_name}")

    # Fix weird axis bugs in Streamlit charts (timezone objects can cause weirdness)
    hourly_df = hourly_df.copy()
    hourly_df["dt"] = hourly_df["dt"].dt.tz_localize(None)
    hourly_df["temp_f"] = pd.to_numeric(hourly_df["temp_f"], errors="coerce")

    st.line_chart(hourly_df.set_index("dt")["temp_f"])

    df_day = hourly_df[(hourly_df["dt"].dt.hour >= DAY_START_HOUR) & (hourly_df["dt"].dt.hour <= DAY_END_HOUR)].copy()
    df_peak = df_day if not df_day.empty else hourly_df
    peak_row = df_peak.loc[df_peak["temp_f"].idxmax()]
    st.write(f"Peak hour (daytime-filtered): **{peak_row['dt'].strftime('%I:%M %p')}** at **{float(peak_row['temp_f']):.1f}°F**")
else:
    st.caption("Hourly curve unavailable (sources returned daily high but not usable hourly).")
