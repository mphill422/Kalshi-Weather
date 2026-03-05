import math
import re
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Kalshi Weather MVP – Daily High (Multi-Source) [Upgraded v2.2]
# - Open-Meteo + NWS (+ optional Open-Meteo GFS)
# - Live airport temp via METAR (NOAA AviationWeather)
# - Heating rate since sunrise + projected "trend high" (nowcast)
# - Correct Kalshi 2° bracket alignment (even-start like 84–85)
# ============================================================

st.set_page_config(page_title="Kalshi Weather MVP – Daily High", layout="wide")
st.title("Kalshi Weather MVP – Daily High (Multi-Source) [Upgraded v2.2]")

USER_AGENT = "kalshi-weather-mvp/2.2"
REQ_TIMEOUT = 12

# Cities (lat/lon/tz + main airport ICAO for "airport temp")
CITIES = {
    "Miami": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "icao": "KMIA"},
    "New York City": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "icao": "KJFK"},
    "Atlanta": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York", "icao": "KATL"},
    "New Orleans": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago", "icao": "KMSY"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago", "icao": "KIAH"},
    "Austin": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago", "icao": "KAUS"},
    "Dallas": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago", "icao": "KDFW"},
    "San Antonio": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago", "icao": "KSAT"},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix", "icao": "KPHX"},
    "Las Vegas": {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles", "icao": "KLAS"},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "icao": "KLAX"},
}

def http_get_text(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, str(e)

def http_get_json(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def safe_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def today_local(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()

def parse_iso_to_local_dt(ts: str, tz_name: str):
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None

def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bracket_probability(mean: float, sigma: float, low: float, high_exclusive: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean < high_exclusive) else 0.0
    z1 = (low - mean) / sigma
    z2 = (high_exclusive - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))

def compute_sigma(source_highs):
    # Conservative uncertainty: base + spread penalty
    if not source_highs:
        return 2.8
    spread = (max(source_highs) - min(source_highs)) if len(source_highs) >= 2 else 0.0
    return max(1.6, 1.8 + 0.55 * spread)

def fmt_temp(x):
    return "—" if x is None else f"{x:.1f}°F"

# Kalshi bracket mapping
def kalshi_bin_low(temp_f: float, size: int, alignment: str) -> int:
    if size == 1:
        return int(math.floor(temp_f))
    if alignment == "odd":
        return int(math.floor((temp_f - 1.0) / 2.0) * 2 + 1)
    return int(math.floor(temp_f / 2.0) * 2)

def kalshi_label(low: int, size: int) -> str:
    if size == 1:
        return f"{low}"
    return f"{low}–{low + size - 1}"

# Sources
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&temperature_unit=fahrenheit"
        f"&hourly=temperature_2m"
        f"&daily=temperature_2m_max,sunrise"
        f"&timezone={tz}"
        f"&forecast_days=2"
    )
    if model:
        url += f"&models={model}"
    return http_get_json(url)

def extract_open_meteo_today(j: dict, tz: str):
    try:
        daily = j.get("daily", {}) if isinstance(j, dict) else {}
        highs = daily.get("temperature_2m_max", []) or []
        daily_time = daily.get("time", []) or []
        sunrises = daily.get("sunrise", []) or []

        if not highs or not daily_time:
            return None, None, None, "Open-Meteo missing daily fields"

        t0 = today_local(tz).isoformat()
        idx = daily_time.index(t0) if t0 in daily_time else 0
        daily_high = safe_float(highs[idx])

        sunrise_dt = None
        if sunrises and idx < len(sunrises) and sunrises[idx]:
            sunrise_dt = parse_iso_to_local_dt(sunrises[idx], tz)

        hourly = j.get("hourly", {}) if isinstance(j, dict) else {}
        ht = hourly.get("time", []) or []
        temps = hourly.get("temperature_2m", []) or []

        df = None
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                dt = parse_iso_to_local_dt(ts, tz)
                tf = safe_float(temp)
                if dt is None or tf is None:
                    continue
                rows.append({"dt": dt, "temp_f": tf})
            df = pd.DataFrame(rows)

        if df is not None and not df.empty:
            tday = today_local(tz)
            df = df[df["dt"].dt.date == tday].sort_values("dt").reset_index(drop=True)
            if df.empty:
                df = None

        return daily_high, df, sunrise_dt, None
    except Exception as e:
        return None, None, None, f"Open-Meteo parse error: {e}"

def fetch_nws_hourly_high(lat: float, lon: float, tz: str):
    try:
        p, err = http_get_json(f"https://api.weather.gov/points/{lat},{lon}")
        if err or not p:
            return None, None, f"NWS points error: {err or 'unknown'}"
        hourly_url = (p.get("properties") or {}).get("forecastHourly")
        if not hourly_url:
            return None, None, "NWS missing forecastHourly URL"
        h, err2 = http_get_json(hourly_url)
        if err2 or not h:
            return None, None, f"NWS hourly error: {err2 or 'unknown'}"
        periods = (h.get("properties") or {}).get("periods") or []
        rows = []
        for per in periods:
            start = per.get("startTime")
            temp = per.get("temperature")
            unit = per.get("temperatureUnit")
            if not start or temp is None or (unit and unit.upper() != "F"):
                continue
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
            except Exception:
                continue
            tf = safe_float(temp)
            if tf is None:
                continue
            rows.append({"dt": dt, "temp_f": tf})
        df = pd.DataFrame(rows)
        if df.empty:
            return None, None, "NWS produced no hourly rows"
        tday = today_local(tz)
        df_today = df[df["dt"].dt.date == tday].sort_values("dt").reset_index(drop=True)
        if df_today.empty:
            return None, None, "NWS hourly has no rows for today"
        return float(df_today["temp_f"].max()), df_today, None
    except Exception as e:
        return None, None, f"NWS error: {e}"

def fetch_metar_temp_f(icao: str, tz: str):
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw&hours=2"
    txt, err = http_get_text(url)
    if err or not txt:
        return None, None, None, f"METAR fetch error: {err or 'unknown'}"
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return None, None, None, "METAR empty response"
    raw = lines[-1]

    m = re.search(r"\s(M?\d{2})/(M?\d{2})\s", f" {raw} ")
    if not m:
        m = re.search(r"(M?\d{2})/(M?\d{2})", raw)
    if not m:
        return None, None, raw, "METAR temp group not found"

    def parse_metar_c(s: str) -> int:
        neg = s.startswith("M")
        v = int(s[1:]) if neg else int(s)
        return -v if neg else v

    temp_f = c_to_f(parse_metar_c(m.group(1)))

    tmatch = re.search(r"\s(\d{2})(\d{2})(\d{2})Z\s", f" {raw} ")
    obs_local = None
    if tmatch:
        dd, hh, mm = int(tmatch.group(1)), int(tmatch.group(2)), int(tmatch.group(3))
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        guess = datetime(now_utc.year, now_utc.month, dd, hh, mm, tzinfo=ZoneInfo("UTC"))
        if guess - now_utc > timedelta(days=2):
            prev_month = (now_utc.replace(day=1) - timedelta(days=1))
            guess = datetime(prev_month.year, prev_month.month, dd, hh, mm, tzinfo=ZoneInfo("UTC"))
        obs_local = guess.astimezone(ZoneInfo(tz))

    return float(temp_f), obs_local, raw, None

# Nowcast
def nearest_temp_at_or_after(df: pd.DataFrame, target: datetime):
    if df is None or df.empty:
        return None
    after = df[df["dt"] >= target]
    if not after.empty:
        return float(after.iloc[0]["temp_f"])
    before = df[df["dt"] <= target]
    if not before.empty:
        return float(before.iloc[-1]["temp_f"])
    return None

def compute_heating_rate_since_sunrise(sunrise_dt, sunrise_temp_f, current_temp_f, now_local):
    if sunrise_dt is None or sunrise_temp_f is None or current_temp_f is None:
        return None, None
    if now_local <= sunrise_dt:
        return None, None
    hours = (now_local - sunrise_dt).total_seconds() / 3600.0
    if hours <= 0.2:
        return None, hours
    return (current_temp_f - sunrise_temp_f) / hours, hours

def projected_high_from_trend(current_temp_f, rate_f_per_hr, now_local, peak_time, clamp_min, clamp_max):
    if current_temp_f is None or rate_f_per_hr is None:
        return None
    if peak_time is None or peak_time <= now_local:
        hours = 3.0
    else:
        hours = max(0.0, (peak_time - now_local).total_seconds() / 3600.0)
    proj = current_temp_f + rate_f_per_hr * hours
    if clamp_min is not None and proj < clamp_min - 6:
        proj = clamp_min - 6
    if clamp_max is not None and proj > clamp_max + 6:
        proj = clamp_max + 6
    return float(proj)

# ========================= UI =========================
st.caption("Bookmark tip: it updates when you refresh the page (pull down / reload).")

city = st.selectbox("City", list(CITIES.keys()))
info = CITIES[city]
tz = info["tz"]
icao = info["icao"]
now_local = datetime.now(ZoneInfo(tz))

with st.expander("Settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    use_gfs = c1.toggle("Include Open-Meteo GFS model (extra check)", value=True)
    use_nws = c2.toggle("Include NWS (api.weather.gov)", value=True)
    bracket_size = c3.selectbox("Kalshi bracket size (°F)", [2, 1], index=0)

    c4, c5, c6 = st.columns(3)
    grace_minutes = c4.slider("Grace minutes after 10:30 local", 0, 180, 45, 5)
    use_nowcast = c5.toggle("Use nowcast (heating-rate adjustment)", value=True)
    kalshi_alignment = c6.selectbox("Kalshi 2° alignment", ["even", "odd"], index=0)

    st.caption("Weights")
    w1, w2 = st.columns(2)
    auto_weights = w1.toggle("Auto weights (recommended)", value=True)
    show_advanced = w2.toggle("Show advanced weight sliders", value=False)

    w_om, w_nws, w_gfs = 0.65, 0.35, 0.20
    if show_advanced:
        a1, a2, a3 = st.columns(3)
        w_om = a1.slider("Weight Open-Meteo", 0.0, 1.0, float(w_om), 0.05)
        w_nws = a2.slider("Weight NWS", 0.0, 1.0, float(w_nws), 0.05)
        w_gfs = a3.slider("Weight GFS", 0.0, 1.0, float(w_gfs), 0.05)

with st.spinner("Fetching forecasts & live data…"):
    om_j, om_fetch_err = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    if om_fetch_err or not om_j:
        om_high, om_hourly, om_sunrise, om_err = None, None, None, f"Open-Meteo fetch error: {om_fetch_err or 'unknown'}"
    else:
        om_high, om_hourly, om_sunrise, om_err = extract_open_meteo_today(om_j, tz)

    gfs_high = gfs_hourly = gfs_sunrise = None
    gfs_err = None
    if use_gfs:
        gfs_j, gfs_fetch_err = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        if gfs_fetch_err or not gfs_j:
            gfs_err = f"GFS fetch error: {gfs_fetch_err or 'unknown'}"
        else:
            gfs_high, gfs_hourly, gfs_sunrise, gfs_err = extract_open_meteo_today(gfs_j, tz)

    nws_high = nws_hourly = None
    nws_err = None
    if use_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)

    metar_temp, metar_time, metar_raw, metar_err = fetch_metar_temp_f(icao, tz)

# Forecast table
st.subheader(f"{city} — Today’s High Forecasts (°F)")
rows = []
rows.append(("Open-Meteo", fmt_temp(om_high), "OK" if om_high is not None else (om_err or "Error")))
if use_gfs:
    rows.append(("Open-Meteo (GFS)", fmt_temp(gfs_high), "OK" if gfs_high is not None else (gfs_err or "Error")))
if use_nws:
    rows.append(("NWS (api.weather.gov)", fmt_temp(nws_high), "OK" if nws_high is not None else (nws_err or "Error")))
st.table(pd.DataFrame(rows, columns=["Source", "Today High", "Status"]))

# Consensus
sources = []
if om_high is not None:
    sources.append(("om", om_high))
if gfs_high is not None:
    sources.append(("gfs", gfs_high))
if nws_high is not None:
    sources.append(("nws", nws_high))

if not sources:
    st.error("No valid sources returned a high for today.")
    st.stop()

vals = [v for _, v in sources]
spread = max(vals) - min(vals) if len(vals) > 1 else 0.0

if auto_weights:
    med = sorted(vals)[len(vals) // 2]
    base = {"om": 0.55, "nws": 0.35, "gfs": 0.20}
    adj = {}
    for k, v in sources:
        dist = abs(v - med)
        adj[k] = base.get(k, 0.2) * (1.0 / (1.0 + 0.35 * dist))
    total = sum(adj.values()) or 1.0
    weights = {k: adj.get(k, 0.0) / total for k, _ in sources}
else:
    base = {"om": float(w_om), "nws": float(w_nws), "gfs": float(w_gfs)}
    total = sum(base.get(k, 0.0) for k, _ in sources) or 1.0
    weights = {k: base.get(k, 0.0) / total for k, _ in sources}

consensus = sum(weights[k] * v for k, v in sources)
sigma = compute_sigma(vals)

c1, c2, c3 = st.columns(3)
c1.metric("Model consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")
st.caption("Effective weights → " + " | ".join([f"{k.upper()}: {weights[k]:.2f}" for k, _ in sources]))

# Live / trend
st.subheader("Live / Trend (Airport Nowcast)")
lc1, lc2, lc3 = st.columns(3)

with lc1:
    if metar_temp is None:
        st.metric(f"Current airport temp ({icao})", "—")
        st.caption(metar_err or "METAR unavailable")
    else:
        st.metric(f"Current airport temp ({icao})", f"{metar_temp:.1f}°F")
        if metar_time:
            st.caption(f"Obs time: {metar_time.strftime('%I:%M %p %Z')}")
        if metar_raw:
            with st.expander("Raw METAR"):
                st.code(metar_raw)

sunrise_dt = om_sunrise or gfs_sunrise
sunrise_temp = None
if sunrise_dt is not None and om_hourly is not None and not om_hourly.empty:
    sunrise_temp = nearest_temp_at_or_after(om_hourly, sunrise_dt)

peak_dt = None
hourly_for_peak = om_hourly if (om_hourly is not None and not om_hourly.empty) else nws_hourly
if hourly_for_peak is not None and not hourly_for_peak.empty:
    idx = hourly_for_peak["temp_f"].idxmax()
    peak_dt = hourly_for_peak.loc[idx, "dt"]

rate, hrs = compute_heating_rate_since_sunrise(sunrise_dt, sunrise_temp, metar_temp, now_local)

with lc2:
    if sunrise_dt is None:
        st.metric("Heating rate since sunrise", "—")
        st.caption("Sunrise unavailable")
    elif sunrise_temp is None:
        st.metric("Heating rate since sunrise", "—")
        st.caption("Sunrise temp unavailable")
    elif rate is None:
        st.metric("Heating rate since sunrise", "—")
        st.caption("Too early or missing airport temp")
    else:
        st.metric("Heating rate since sunrise", f"{rate:+.2f} °F/hr")
        st.caption(f"Sunrise: {sunrise_dt.strftime('%I:%M %p')} | Temp@sunrise≈ {sunrise_temp:.1f}°F")

trend_high = projected_high_from_trend(metar_temp, rate, now_local, peak_dt, min(vals), max(vals))

with lc3:
    if not use_nowcast:
        st.metric("Projected high (trend)", "—")
        st.caption("Nowcast OFF")
    elif trend_high is None:
        st.metric("Projected high (trend)", "—")
        st.caption("Need airport temp + heating rate")
    else:
        st.metric("Projected high (trend)", f"{trend_high:.1f}°F")
        if peak_dt is not None:
            st.caption(f"Peak window target: {peak_dt.strftime('%I:%M %p')}")

if use_nowcast and trend_high is not None:
    hour = now_local.hour + now_local.minute / 60.0
    blend = min(0.45, max(0.15, (hour - 9.0) / 10.0))
    final_mean = (1.0 - blend) * consensus + blend * trend_high
    st.caption(f"Nowcast blend: {blend:.2f} → Final mean: {final_mean:.1f}°F")
else:
    final_mean = consensus

# Brackets
st.subheader("Suggested Kalshi Range (Daily High)")
center_low = kalshi_bin_low(final_mean, bracket_size, kalshi_alignment)

candidates = []
if bracket_size == 1:
    lows = range(int(math.floor(final_mean)) - 6, int(math.floor(final_mean)) + 7)
else:
    lows = range(center_low - 12, center_low + 14, 2)

for low in lows:
    p = bracket_probability(final_mean, sigma, low, low + bracket_size)
    candidates.append((low, p))

best_low, best_p = max(candidates, key=lambda x: x[1])
st.success(f"Primary suggestion: **{kalshi_label(best_low, bracket_size)}°F** (model probability ≈ **{best_p*100:.0f}%**)")

top = sorted(candidates, key=lambda x: x[1], reverse=True)[:6]
top_df = pd.DataFrame([{"Bracket": kalshi_label(lo, bracket_size), "Model Prob %": round(p * 100, 1)} for lo, p in top])
st.caption("Top bracket candidates (model-based):")
st.dataframe(top_df, use_container_width=True, hide_index=True)

# Decision window
st.subheader("Decision Window")
lock_dt = datetime.combine(now_local.date(), time(10, 30), tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))
st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}** | "
    f"Target lock: **10:30 AM** | With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)
if now_local <= deadline_dt:
    st.info("You are inside the preferred betting window (or within grace).")
else:
    st.warning("You are past the preferred window. Edge often shrinks as the day progresses.")

# Hourly curve
st.subheader("Hourly temperature curve (today)")
hourly_df = om_hourly if (om_hourly is not None and not om_hourly.empty) else nws_hourly
source_name = "Open-Meteo" if (om_hourly is not None and not om_hourly.empty) else ("NWS" if (nws_hourly is not None and not nws_hourly.empty) else None)

if hourly_df is not None and not hourly_df.empty:
    st.caption(f"Hourly source used: {source_name}")
    st.line_chart(hourly_df.set_index("dt")["temp_f"])
    idx = hourly_df["temp_f"].idxmax()
    pk_dt = hourly_df.loc[idx, "dt"]
    pk_temp = float(hourly_df.loc[idx, "temp_f"])
    st.write(f"Peak hour: **{pk_dt.strftime('%I:%M %p')}** at **{pk_temp:.1f}°F**")
else:
    st.caption("Hourly curve unavailable.")

# Value check
st.subheader("Value Bet Check (you enter the Kalshi price)")
price = st.number_input("Enter Kalshi YES price for the suggested bracket (cents, 0–100)", 0, 100, 50, 1)
implied = float(price) / 100.0
edge = best_p - implied
st.write(f"Model probability for **{kalshi_label(best_low, bracket_size)}** ≈ **{best_p*100:.1f}%**")
st.write(f"Implied probability from **{price}¢** ≈ **{implied*100:.1f}%**")
if edge >= 0.02:
    st.success(f"Positive edge ≈ **{edge*100:.1f}%** (model > market)")
elif edge <= -0.02:
    st.error(f"Negative edge ≈ **{edge*100:.1f}%** (market > model)")
else:
    st.warning(f"Close to fair ≈ **{edge*100:.1f}%**")

st.caption("If Kalshi bins don’t match the suggested 2° labels, switch 'Kalshi 2° alignment' to ODD for that city.")
