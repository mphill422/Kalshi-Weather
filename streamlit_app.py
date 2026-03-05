import math
import re
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Kalshi Weather Model – Full Update (v3)
# - Open-Meteo + NWS + optional Open-Meteo GFS
# - LIVE temp from NWS Station Observations (primary) + METAR fallback
# - Sunrise + heating-rate + nowcast projected high
# - Discrete (integer) bracket probability w/ continuity correction
# - Auto-weights based on agreement with live observation
# ============================================================

st.set_page_config(page_title="Kalshi Weather Model v3", layout="wide")
st.title("Kalshi Weather Model – Daily High (Multi-Source + Live Nowcast)")

USER_AGENT = "kalshi-weather-model-v3 (contact: none)"
REQ_TIMEOUT = 12

# NOTE: Station IDs below are NWS station identifiers that often match the airport ICAO.
# If you discover Kalshi settles off a different station for a city, update `station` here.
CITIES = {
    "Miami, FL":         {"lat": 25.7617, "lon": -80.1918,  "tz": "America/New_York",    "icao": "KMIA", "station": "KMIA"},
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060,  "tz": "America/New_York",    "icao": "KJFK", "station": "KJFK"},
    "Atlanta, GA":       {"lat": 33.7490, "lon": -84.3880,  "tz": "America/New_York",    "icao": "KATL", "station": "KATL"},
    "New Orleans, LA":   {"lat": 29.9511, "lon": -90.0715,  "tz": "America/Chicago",     "icao": "KMSY", "station": "KMSY"},
    "Houston, TX":       {"lat": 29.7604, "lon": -95.3698,  "tz": "America/Chicago",     "icao": "KIAH", "station": "KIAH"},
    "Austin, TX":        {"lat": 30.2672, "lon": -97.7431,  "tz": "America/Chicago",     "icao": "KAUS", "station": "KAUS"},
    "Dallas, TX":        {"lat": 32.7767, "lon": -96.7970,  "tz": "America/Chicago",     "icao": "KDFW", "station": "KDFW"},
    "San Antonio, TX":   {"lat": 29.4241, "lon": -98.4936,  "tz": "America/Chicago",     "icao": "KSAT", "station": "KSAT"},
    "Phoenix, AZ":       {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix",     "icao": "KPHX", "station": "KPHX"},
    "Las Vegas, NV":     {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles", "icao": "KLAS", "station": "KLAS"},
    "Los Angeles, CA":   {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "icao": "KLAX", "station": "KLAX"},
}

# ----------------------------
# HTTP helpers
# ----------------------------
def http_get_json(url: str) -> dict | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def http_get_text(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0

def now_local(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))

def today_local(tz_name: str) -> date:
    return now_local(tz_name).date()

def parse_iso_to_local_dt(ts: str, tz_name: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None

# ----------------------------
# Probability math
# ----------------------------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bracket_probability_continuous(mean: float, sigma: float, low: float, high: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean < high) else 0.0
    z1 = (low - mean) / sigma
    z2 = (high - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))

def bracket_probability_integer(mean: float, sigma: float, low_int: int, high_int: int) -> float:
    # Continuity correction for integer-reported daily highs.
    return bracket_probability_continuous(mean, sigma, low_int - 0.5, high_int + 0.5)

def compute_sigma(source_highs: list[float]) -> float:
    if not source_highs:
        return 2.8
    spread = max(source_highs) - min(source_highs) if len(source_highs) >= 2 else 0.0
    return max(1.6, 1.7 + 0.55 * spread)

def align_low_even(temp: float) -> int:
    # 2° bins like 84–85, 86–87, ...
    return int(math.floor(temp / 2.0) * 2)

def align_low_odd(temp: float) -> int:
    # 2° bins like 83–84, 85–86, ...
    return int(math.floor((temp - 1) / 2.0) * 2 + 1)

# ----------------------------
# Data sources
# ----------------------------
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None) -> dict | None:
    base = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&temperature_unit=fahrenheit"
        f"&hourly=temperature_2m"
        f"&daily=temperature_2m_max,sunrise,sunset"
        f"&timezone={tz}"
        f"&forecast_days=2"
    )
    if model:
        base += f"&models={model}"
    return http_get_json(base)

def extract_open_meteo_today(j: dict, tz: str) -> tuple[float | None, pd.DataFrame | None, datetime | None, str | None]:
    if not j:
        return None, None, None, "Open-Meteo request failed"
    try:
        daily = j.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        times = daily.get("time", [])
        sunr = daily.get("sunrise", [])
        if not highs or not times:
            return None, None, None, "Open-Meteo missing daily fields"

        t0 = today_local(tz).isoformat()
        idx = times.index(t0) if t0 in times else 0
        high = safe_float(highs[idx])

        sunrise_dt = None
        if sunr and len(sunr) > idx:
            sunrise_dt = parse_iso_to_local_dt(sunr[idx], tz)

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
            df = df[df["dt"].dt.date == tday].sort_values("dt").copy()
            if df.empty:
                df = None

        return high, df, sunrise_dt, None
    except Exception as e:
        return None, None, None, f"Open-Meteo parse error: {e}"

def fetch_nws_hourly_high(lat: float, lon: float, tz: str) -> tuple[float | None, pd.DataFrame | None, str | None]:
    try:
        p = http_get_json(f"https://api.weather.gov/points/{lat},{lon}")
        if not p:
            return None, None, "NWS points request failed"

        hourly_url = (p.get("properties") or {}).get("forecastHourly")
        if not hourly_url:
            return None, None, "NWS missing forecastHourly"

        h = http_get_json(hourly_url)
        if not h:
            return None, None, "NWS hourly request failed"

        periods = ((h.get("properties") or {}).get("periods")) or []
        if not periods:
            return None, None, "NWS hourly periods empty"

        rows = []
        for per in periods:
            start = per.get("startTime")
            temp = per.get("temperature")
            unit = (per.get("temperatureUnit") or "").upper()
            if unit and unit != "F":
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

        high = float(df_today["temp_f"].max())
        return high, df_today.sort_values("dt"), None
    except Exception as e:
        return None, None, f"NWS error: {e}"

def fetch_nws_station_observation(station_id: str) -> tuple[float | None, datetime | None, str | None]:
    try:
        j = http_get_json(f"https://api.weather.gov/stations/{station_id}/observations/latest")
        if not j:
            return None, None, "NWS station obs request failed"
        props = j.get("properties") or {}
        temp_c = (props.get("temperature") or {}).get("value")  # Celsius
        ts = props.get("timestamp")
        if temp_c is None:
            return None, None, "NWS station obs missing temperature"
        temp_f = c_to_f(float(temp_c))
        obs_dt = None
        if ts:
            try:
                obs_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                obs_dt = None
        return temp_f, obs_dt, None
    except Exception as e:
        return None, None, f"NWS station obs error: {e}"

def fetch_metar_temp(icao: str) -> tuple[float | None, str | None]:
    txt = http_get_text(f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw&hours=2")
    if not txt:
        return None, "METAR request failed"
    lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    if not lines:
        return None, "METAR empty response"
    metar = lines[-1]
    m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", metar)
    if not m:
        return None, "METAR temp group not found"
    t = m.group(1)
    neg = t.startswith("M")
    v = int(t[1:] if neg else t)
    if neg:
        v = -v
    return c_to_f(v), None

# ----------------------------
# Nowcast logic
# ----------------------------
def hourly_at(df: pd.DataFrame, t: datetime) -> float | None:
    if df is None or df.empty:
        return None
    df = df.sort_values("dt")
    if t <= df["dt"].iloc[0]:
        return float(df["temp_f"].iloc[0])
    if t >= df["dt"].iloc[-1]:
        return float(df["temp_f"].iloc[-1])

    after = df[df["dt"] >= t].iloc[0]
    before = df[df["dt"] <= t].iloc[-1]
    t0, y0 = before["dt"], float(before["temp_f"])
    t1, y1 = after["dt"], float(after["temp_f"])
    if t1 == t0:
        return y1
    w = (t - t0).total_seconds() / (t1 - t0).total_seconds()
    return y0 + w * (y1 - y0)

def peak_from_hourly(df: pd.DataFrame) -> tuple[datetime | None, float | None]:
    if df is None or df.empty:
        return None, None
    df = df.copy().sort_values("dt")
    df_day = df[(df["dt"].dt.hour >= 9) & (df["dt"].dt.hour <= 19)].copy()
    if df_day.empty:
        df_day = df
    idx = df_day["temp_f"].idxmax()
    row = df_day.loc[idx]
    return row["dt"], float(row["temp_f"])

def heating_rate_since_sunrise(live_temp: float | None, sunrise: datetime | None, om_hourly: pd.DataFrame | None, tz: str) -> float | None:
    if live_temp is None or sunrise is None or om_hourly is None or om_hourly.empty:
        return None
    nowt = now_local(tz)
    if nowt <= sunrise:
        return None
    t0 = hourly_at(om_hourly, sunrise)
    if t0 is None:
        return None
    hours = (nowt - sunrise).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return (live_temp - t0) / hours

def project_high_from_trend(live_temp: float | None, sunrise: datetime | None, om_hourly: pd.DataFrame | None, tz: str) -> float | None:
    if live_temp is None or sunrise is None or om_hourly is None or om_hourly.empty:
        return None
    rate = heating_rate_since_sunrise(live_temp, sunrise, om_hourly, tz)
    if rate is None:
        return None
    peak_t, _ = peak_from_hourly(om_hourly)
    if peak_t is None:
        return None
    nowt = now_local(tz)
    hrs_to_peak = max(0.0, (peak_t - nowt).total_seconds() / 3600.0)
    return live_temp + rate * hrs_to_peak

def nowcast_adjusted_high(forecast_high: float | None, live_temp: float | None, forecast_now: float | None, hours_to_peak: float | None) -> float | None:
    if forecast_high is None or live_temp is None or forecast_now is None:
        return None
    delta = live_temp - forecast_now
    if hours_to_peak is None:
        alpha = 0.6
    else:
        alpha = 0.75 if hours_to_peak >= 4 else (0.55 if hours_to_peak >= 2 else 0.35)
    return forecast_high + alpha * delta

def auto_weights_against_live(live_temp: float | None, nowt: datetime, om_hourly: pd.DataFrame | None, gfs_hourly: pd.DataFrame | None, nws_hourly: pd.DataFrame | None) -> dict:
    base = {"Open-Meteo": 0.0, "GFS": 0.0, "NWS": 0.0}
    if live_temp is None:
        base.update({"Open-Meteo": 0.65, "NWS": 0.35, "GFS": 0.00})
        s = sum(base.values())
        return {k: v / s for k, v in base.items()}

    def score(df):
        if df is None or df.empty:
            return 0.0
        fn = hourly_at(df, nowt)
        if fn is None:
            return 0.0
        err = abs(live_temp - fn)
        return math.exp(-err / 2.0)

    base["Open-Meteo"] = score(om_hourly)
    base["GFS"] = score(gfs_hourly)
    base["NWS"] = score(nws_hourly)
    s = sum(base.values())
    if s <= 1e-9:
        base.update({"Open-Meteo": 0.65, "NWS": 0.35, "GFS": 0.00})
        s = sum(base.values())
    return {k: v / s for k, v in base.items()}

# ----------------------------
# UI
# ----------------------------
colA, colB = st.columns([1, 1])
with colA:
    city = st.selectbox("City", list(CITIES.keys()))
with colB:
    st.caption("Tip: If a source errors, the app still runs — it just excludes that source.")

info = CITIES[city]
tz = info["tz"]

with st.expander("Settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_gfs = st.toggle("Include Open-Meteo GFS (extra check)", value=True)
    with c2:
        use_nws = st.toggle("Include NWS forecast (api.weather.gov)", value=True)
    with c3:
        bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=1)
    with c4:
        grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)

    st.divider()
    use_nowcast = st.toggle("Use nowcast (live-trend adjustment)", value=True)
    auto_w = st.toggle("Auto weights (recommended)", value=True)
    show_adv = st.toggle("Show advanced weight sliders", value=False)

    w_om_default, w_nws_default, w_gfs_default = 0.65, 0.35, 0.00
    w_om, w_nws, w_gfs = w_om_default, w_nws_default, w_gfs_default
    if show_adv and not auto_w:
        s1, s2, s3 = st.columns(3)
        with s1:
            w_om = st.slider("Weight Open-Meteo", 0.0, 1.0, float(w_om_default), 0.05)
        with s2:
            w_nws = st.slider("Weight NWS", 0.0, 1.0, float(w_nws_default), 0.05)
        with s3:
            w_gfs = st.slider("Weight GFS", 0.0, 1.0, float(w_gfs_default), 0.05)

# ----------------------------
# Fetch
# ----------------------------
with st.spinner("Fetching forecasts + live observations…"):
    om = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    om_high, om_hourly, sunrise_dt, om_err = extract_open_meteo_today(om or {}, tz)

    gfs_high = gfs_hourly = None
    gfs_err = None
    if use_gfs:
        om_gfs = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        gfs_high, gfs_hourly, _, gfs_err = extract_open_meteo_today(om_gfs or {}, tz)

    nws_high = nws_hourly = None
    nws_err = None
    if use_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)

    live_temp, live_dt_utc, live_err = fetch_nws_station_observation(info["station"])
    live_source = f"NWS station {info['station']}"
    if live_temp is None:
        metar_temp, metar_err = fetch_metar_temp(info["icao"])
        if metar_temp is not None:
            live_temp = metar_temp
            live_source = f"METAR {info['icao']} (fallback)"
            live_err = None
        else:
            live_err = live_err or metar_err or "No live observation source available"

# ----------------------------
# Display sources
# ----------------------------
st.subheader(f"{city} – Source Highs (°F)")
rows = []
rows.append(("Open-Meteo", f"{om_high:.1f}°F" if om_high is not None else "—", "OK" if om_high is not None else (om_err or "Error")))
if use_gfs:
    rows.append(("Open-Meteo (GFS)", f"{gfs_high:.1f}°F" if gfs_high is not None else "—", "OK" if gfs_high is not None else (gfs_err or "Error")))
if use_nws:
    rows.append(("NWS (forecastHourly)", f"{nws_high:.1f}°F" if nws_high is not None else "—", "OK" if nws_high is not None else (nws_err or "Error")))
st.table(pd.DataFrame(rows, columns=["Source", "Today High", "Status"]))

# ----------------------------
# Live observation
# ----------------------------
st.subheader("Live Observation (airport/station reality check)")
lc1, lc2, lc3 = st.columns(3)
if live_temp is not None:
    lc1.metric("Current temp", f"{live_temp:.1f}°F")
    lc2.metric("Source", live_source)
    if live_dt_utc is not None:
        try:
            local_dt = live_dt_utc.astimezone(ZoneInfo(tz))
            lc3.metric("Obs time (local)", local_dt.strftime("%I:%M %p"))
        except Exception:
            lc3.metric("Obs time", "—")
else:
    st.warning(f"Live observation unavailable: {live_err}")

# ----------------------------
# Consensus + nowcast
# ----------------------------
valid_highs = []
source_map = {}
if om_high is not None:
    valid_highs.append(om_high); source_map["Open-Meteo"] = om_high
if gfs_high is not None:
    valid_highs.append(gfs_high); source_map["GFS"] = gfs_high
if nws_high is not None:
    valid_highs.append(nws_high); source_map["NWS"] = nws_high

if not valid_highs:
    st.error("No valid forecast highs returned. Toggle off failing sources and try again.")
    st.stop()

sigma = compute_sigma(valid_highs)
spread = (max(valid_highs) - min(valid_highs)) if len(valid_highs) > 1 else 0.0

nowt = now_local(tz)
if auto_w:
    w = auto_weights_against_live(live_temp, nowt, om_hourly, gfs_hourly, nws_hourly)
    w_om_eff, w_gfs_eff, w_nws_eff = w["Open-Meteo"], w["GFS"], w["NWS"]
else:
    s = max(1e-9, (w_om + w_nws + w_gfs))
    w_om_eff, w_nws_eff, w_gfs_eff = w_om/s, w_nws/s, w_gfs/s

# Weighted consensus across available sources (renormalized)
w_sum = 0.0
cons = 0.0
for name, val in source_map.items():
    wt = w_om_eff if name == "Open-Meteo" else (w_gfs_eff if name == "GFS" else w_nws_eff)
    cons += wt * float(val)
    w_sum += wt
consensus = cons / w_sum if w_sum > 1e-9 else float(sum(valid_highs)/len(valid_highs))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")
c4.metric("Weights", f"OM {w_om_eff:.2f} | NWS {w_nws_eff:.2f} | GFS {w_gfs_eff:.2f}")

forecast_now = hourly_at(om_hourly, nowt) if om_hourly is not None else None
peak_t, _ = peak_from_hourly(om_hourly) if om_hourly is not None else (None, None)
hrs_to_peak = (peak_t - nowt).total_seconds()/3600.0 if peak_t else None

trend_rate = heating_rate_since_sunrise(live_temp, sunrise_dt, om_hourly, tz)
trend_high = project_high_from_trend(live_temp, sunrise_dt, om_hourly, tz)
bias_high = nowcast_adjusted_high(om_high, live_temp, forecast_now, hrs_to_peak)

nowcast_high = None
if use_nowcast:
    candidates = [x for x in [bias_high, trend_high] if isinstance(x, (int, float)) and x is not None]
    if candidates:
        nowcast_high = (0.65 * float(bias_high) + 0.35 * float(trend_high)) if (bias_high is not None and trend_high is not None) else float(candidates[0])

st.subheader("Nowcast (live-trend edge)")
nc1, nc2, nc3, nc4 = st.columns(4)
nc1.metric("Sunrise (local)", sunrise_dt.strftime("%I:%M %p") if sunrise_dt else "—")
nc2.metric("Heating rate since sunrise", f"{trend_rate:.2f}°F/hr" if trend_rate is not None else "—")
nc3.metric("Projected high (trend→peak)", f"{trend_high:.1f}°F" if trend_high is not None else "—")
nc4.metric("Nowcast blended high", f"{nowcast_high:.1f}°F" if nowcast_high is not None else "—")

mean_for_bins = nowcast_high if (use_nowcast and nowcast_high is not None) else consensus

# ----------------------------
# Kalshi bracket suggestion
# ----------------------------
st.subheader("Suggested Kalshi bracket (integer-aware)")
center = int(round(mean_for_bins))

cands = []
if bracket_size == 1:
    for low in range(center - 6, center + 7):
        hi = low
        p = bracket_probability_integer(mean_for_bins, sigma, low, hi)
        cands.append((low, hi, p, "1°"))
else:
    base_even = align_low_even(mean_for_bins)
    base_odd = align_low_odd(mean_for_bins)
    for base, tag in [(base_even, "even-start"), (base_odd, "odd-start")]:
        for low in range(base - 12, base + 14, 2):
            hi = low + 1
            p = bracket_probability_integer(mean_for_bins, sigma, low, hi)
            cands.append((low, hi, p, tag))

best_low, best_hi, best_p, best_tag = max(cands, key=lambda x: x[2])
st.success(f"Primary suggestion: **{best_low} to {best_hi}** (prob ≈ **{best_p*100:.0f}%**, {best_tag})")

top = sorted(cands, key=lambda x: x[2], reverse=True)[:10]
top_df = pd.DataFrame([{"Bracket": f"{lo}–{hi}", "Model Prob %": round(p*100, 1), "Alignment": tag} for lo, hi, p, tag in top])
st.dataframe(top_df, use_container_width=True, hide_index=True)

# ----------------------------
# Value check
# ----------------------------
st.subheader("Value Bet Check (enter Kalshi price)")
price = st.number_input("Kalshi YES price (cents, 0–100)", 0, 100, 50, 1)
implied = price / 100.0
edge = best_p - implied
st.write(f"Model probability for {best_low}–{best_hi}: **{best_p*100:.1f}%**")
st.write(f"Implied probability from {price}¢: **{implied*100:.1f}%**")
st.success(f"Edge ≈ **{edge*100:.1f}%** (positive means model > market)") if edge >= 0 else st.error(f"Edge ≈ **{edge*100:.1f}%**")

# ----------------------------
# Decision window
# ----------------------------
lock_time_local = time(10, 30)
lock_dt = datetime.combine(nowt.date(), lock_time_local, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.subheader("Decision Window")
st.write(
    f"Local time now: **{nowt.strftime('%a %b %d, %I:%M %p')}** | "
    f"Target lock: **10:30 AM** | With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)
st.info("Inside preferred window (or within grace).") if nowt <= deadline_dt else st.warning("Past preferred window. Edge usually shrinks.")

# ----------------------------
# Hourly curve
# ----------------------------
st.subheader("Hourly temperature curve (today)")
hourly_df, hourly_source = None, None
if om_hourly is not None and not om_hourly.empty:
    hourly_df, hourly_source = om_hourly.sort_values("dt"), "Open-Meteo"
elif nws_hourly is not None and not nws_hourly.empty:
    hourly_df, hourly_source = nws_hourly.sort_values("dt"), "NWS"

if hourly_df is not None and not hourly_df.empty:
    st.caption(f"Hourly source used: {hourly_source}")
    st.line_chart(hourly_df.set_index("dt")["temp_f"])
    pt, pv = peak_from_hourly(hourly_df)
    if pt is not None and pv is not None:
        st.write(f"Peak hour (daytime-filtered): **{pt.strftime('%I:%M %p')}** at **{pv:.1f}°F**")
else:
    st.caption("Hourly curve unavailable.")

st.divider()
st.caption(
    "Important: Kalshi settlement uses a specific official observation station. "
    "This app pulls LIVE temperature from NWS station observations (per city `station`) and falls back to METAR. "
    "If a city's settlement station differs, update that city's `station` value in the CITIES dict."
)
