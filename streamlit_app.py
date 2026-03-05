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

APP_TITLE = "Kalshi Weather MVP – Daily High (Multi-Source) [Upgraded]"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-mvp/1.1 (contact: none)"
REQ_TIMEOUT = 12

# Cities
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
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def round_bracket_label(low: int, size: int) -> str:
    # Inclusive bracket label (size=2 -> low–low+1)
    return f"{low}–{low + size - 1}"


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bracket_probability_inclusive(mean: float, sigma: float, low: float, high_inclusive: float) -> float:
    """
    P(low <= X <= high_inclusive) for Normal(mean, sigma).
    For integer brackets, use CDF(high+0.5)-CDF(low-0.5) to match discrete degree rounding.
    """
    if sigma <= 1e-9:
        return 1.0 if (low <= mean <= high_inclusive) else 0.0

    z1 = ((low - 0.5) - mean) / sigma
    z2 = ((high_inclusive + 0.5) - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))


def compute_sigma_from_spread(source_highs: list[float]) -> float:
    """
    Better behaved sigma:
      sigma = max(fallback, spread/3) with clamps
    """
    fallback = 2.2
    if not source_highs:
        return fallback
    if len(source_highs) < 2:
        return fallback
    spread = max(source_highs) - min(source_highs)
    sigma = max(fallback, spread / 3.0)
    return max(1.0, min(6.0, sigma))


def daytime_filter(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Keep only rows in local daytime hours."""
    if df is None or df.empty:
        return df
    return df[(df["dt"].dt.hour >= start_hour) & (df["dt"].dt.hour <= end_hour)].copy()


def heating_rate_f_per_hr(hourly_df: pd.DataFrame, now_local: datetime, lookback_minutes: int = 90) -> float | None:
    if hourly_df is None or hourly_df.empty:
        return None
    cutoff = now_local - timedelta(minutes=lookback_minutes)
    df = hourly_df[(hourly_df["dt"] >= cutoff) & (hourly_df["dt"] <= now_local)].copy()
    if df.shape[0] < 2:
        return None
    df = df.sort_values("dt")
    t0, t1 = df.iloc[0], df.iloc[-1]
    dt_hours = (t1["dt"] - t0["dt"]).total_seconds() / 3600.0
    if dt_hours <= 0:
        return None
    return float((t1["temp_f"] - t0["temp_f"]) / dt_hours)


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
    """
    NWS: points -> forecastHourly (periods).
    IMPORTANT FIX: compute today's HIGH using *daytime hours only* to avoid weird late selections.
    """
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

        # Daytime-only max (fix)
        df_day = daytime_filter(df_today, 10, 18)
        if df_day is not None and not df_day.empty:
            daily_high = float(df_day["temp_f"].max())
            return daily_high, df_today, None

        # Fallback: if daytime missing, use full day
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
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        use_gfs = st.toggle("Include Open-Meteo GFS model (extra check)", value=False)
    with col2:
        use_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    with col3:
        bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=0)
    with col4:
        grace_minutes = st.slider("Grace minutes after 10:30 local", min_value=0, max_value=120, value=45, step=5)
    with col5:
        use_nowcast = st.toggle("Use nowcast (heating-rate adjustment)", value=True)

    # weights (sane defaults)
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        w_open = st.slider("Weight Open-Meteo", 0.0, 1.0, 0.55, 0.05)
    with wcol2:
        w_nws = st.slider("Weight NWS", 0.0, 1.0, 0.30, 0.05)
    with wcol3:
        w_gfs = st.slider("Weight GFS", 0.0, 1.0, 0.15, 0.05)

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
# Weighted consensus + sigma
# ----------------------------
vals = []
wts = []
if om_high is not None:
    vals.append(float(om_high)); wts.append(w_open)
if gfs_high is not None and use_gfs:
    vals.append(float(gfs_high)); wts.append(w_gfs)
if nws_high is not None and use_nws:
    vals.append(float(nws_high)); wts.append(w_nws)

if not vals:
    st.error("No valid sources returned a high for today. Try again, or toggle off a failing source.")
    st.stop()

wsum = sum(wts) if sum(wts) > 0 else len(wts)
consensus = float(sum(v * w for v, w in zip(vals, wts)) / wsum)

source_highs = vals[:]  # for spread/sigma
spread = (max(source_highs) - min(source_highs)) if len(source_highs) > 1 else 0.0
sigma = compute_sigma_from_spread(source_highs)

c1, c2, c3 = st.columns(3)
c1.metric("Weighted consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")

# ----------------------------
# Hourly curve + peak + nowcast
# ----------------------------
now_local = datetime.now(ZoneInfo(tz))

hourly_source_name = None
hourly_df = None
if om_hourly is not None and not om_hourly.empty:
    hourly_df = om_hourly.sort_values("dt")
    hourly_source_name = "Open-Meteo"
elif nws_hourly is not None and not nws_hourly.empty:
    hourly_df = nws_hourly.sort_values("dt")
    hourly_source_name = "NWS"

peak_time = None
peak_temp = None
max_so_far = None
now_temp = None
rate = None

if hourly_df is not None and not hourly_df.empty:
    st.subheader("Hourly temperature curve (today)")
    st.caption(f"Hourly source used: {hourly_source_name}")
    st.line_chart(hourly_df.set_index("dt")["temp_f"])

    # Daytime peak (fix): restrict to 10am–6pm local for peak time
    df_day = daytime_filter(hourly_df, 10, 18)
    df_peak = df_day if df_day is not None and not df_day.empty else hourly_df
    idx = df_peak["temp_f"].idxmax()
    peak_row = df_peak.loc[idx]
    peak_time = peak_row["dt"]
    peak_temp = float(peak_row["temp_f"])

    # current-ish temp from nearest hour <= now
    df_past = hourly_df[hourly_df["dt"] <= now_local]
    if not df_past.empty:
        now_temp = float(df_past.sort_values("dt").iloc[-1]["temp_f"])
        max_so_far = float(df_past["temp_f"].max())

    rate = heating_rate_f_per_hr(hourly_df, now_local, lookback_minutes=90)

    st.write(
        f"Peak hour (daytime-filtered): **{peak_time.strftime('%I:%M %p')}** at **{peak_temp:.1f}°F**"
    )

# Apply nowcast shift to consensus mean
mu = consensus
nowcast_peak = None
if use_nowcast and (now_temp is not None) and (rate is not None) and (peak_time is not None):
    # hours until expected peak, clipped
    hours_to_peak = (peak_time - now_local).total_seconds() / 3600.0
    if hours_to_peak > 0:
        # cap heating rate to avoid crazy spikes
        rate_capped = max(-2.0, min(6.0, rate))
        nowcast_peak = now_temp + rate_capped * hours_to_peak
        if max_so_far is not None:
            nowcast_peak = max(nowcast_peak, max_so_far)

        # blend (30% nowcast)
        mu = 0.70 * consensus + 0.30 * nowcast_peak

if nowcast_peak is not None:
    st.info(f"Nowcast peak estimate: **{nowcast_peak:.1f}°F** → blended mean used: **{mu:.1f}°F**")

# ----------------------------
# Suggested bracket (FIXED)
# ----------------------------
# Build candidate brackets around mu (mean)
center = int(round(mu))
candidates = []
for low in range(center - 6, center + 7):
    high = low + bracket_size - 1  # inclusive high
    p = bracket_probability_inclusive(mu, sigma, low, high)
    candidates.append((low, high, p))

best_low, best_high, best_p = max(candidates, key=lambda x: x[2])

# Boundary warning
boundary_warn = (abs(mu - best_low) < 0.7) or (abs(mu - best_high) < 0.7)

msg = (
    f"Suggested Kalshi bracket: **{best_low}–{best_high}**  "
    f"(model probability ≈ **{best_p*100:.0f}%**)"
)
if boundary_warn:
    st.warning(msg + "  ⚠ **Near bracket boundary** (high variance) — size down or wait for more data.")
else:
    st.success(msg)

top5 = sorted(candidates, key=lambda x: x[2], reverse=True)[:5]
top_df = pd.DataFrame(
    [{"Bracket": f"{lo}–{hi}", "Model Prob %": round(p*100, 1)} for lo, hi, p in top5]
)
st.caption("Top bracket candidates (model-based):")
st.dataframe(top_df, use_container_width=True, hide_index=True)

# ----------------------------
# Timing / Decision window
# ----------------------------
st.subheader("Decision Window")
lock_dt = datetime.combine(now_local.date(), lock_time_local, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}**  "
    f"| Target lock: **10:30 AM**  "
    f"| With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)

if now_local <= deadline_dt:
    st.info("You are inside the preferred betting window (or within grace).")
else:
    st.warning("You are past the preferred window. You can still bet, but edge typically shrinks as the day progresses.")

st.divider()
st.caption(
    "Upgrades included: fixed bracket math, daytime NWS peak filtering, weighted ensemble, sigma from spread, "
    "optional nowcast heating-rate adjustment, and boundary risk warnings."
)
