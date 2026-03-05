import math
import re
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Kalshi Weather Model v4 (single-file)
# - Live temperature from NWS observation station (closest working)
# - Sunrise heating-rate + nowcast projected high
# - Better 2° bracket alignment (checks even/odd starts)
# - Kalshi auto-import (best-effort) from URL (__NEXT_DATA__) or pasted table
# - Fixes Streamlit formatting (no DeltaGenerator objects printed)
# ============================================================

st.set_page_config(page_title="Kalshi Weather Model v4", layout="wide")
APP_TITLE = "Kalshi Weather Model v4 – Daily High (Forecast + Live Station + Kalshi Import)"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-model-v4"
REQ_TIMEOUT = 15

# --- Cities (edit / add as needed) ---
CITIES = {
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta, GA": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "San Antonio, TX": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Las Vegas, NV": {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles"},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

# ICAO fallback (METAR)
ICAO_FALLBACK = {
    "Miami, FL": "KMIA",
    "New York City, NY": "KJFK",
    "Atlanta, GA": "KATL",
    "New Orleans, LA": "KMSY",
    "Houston, TX": "KIAH",
    "Austin, TX": "KAUS",
    "Dallas, TX": "KDFW",
    "San Antonio, TX": "KSAT",
    "Phoenix, AZ": "KPHX",
    "Las Vegas, NV": "KLAS",
    "Los Angeles, CA": "KLAX",
}

def _headers():
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}

@st.cache_data(ttl=120, show_spinner=False)
def http_get_json(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=_headers(), timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def http_get_text(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def safe_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def now_local(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))

def today_local(tz: str) -> date:
    return now_local(tz).date()

# ---------------- Probability helpers ----------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bracket_probability(mean: float, sigma: float, low: float, high: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean < high) else 0.0
    z1 = (low - mean) / sigma
    z2 = (high - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))

def bracket_probability_integer_cc(mean: float, sigma: float, low: float, high: float) -> float:
    # continuity correction for integer-settlement
    return bracket_probability(mean, sigma, low - 0.5, high + 0.5)

def compute_sigma(source_highs: list[float], live_conf_boost: float = 0.0) -> float:
    if not source_highs:
        return 3.0
    spread = (max(source_highs) - min(source_highs)) if len(source_highs) >= 2 else 0.0
    sig = 1.8 + 0.55 * spread + live_conf_boost
    return max(1.5, min(6.0, sig))

def two_degree_candidates_for_temp(x: float) -> list[int]:
    f = int(math.floor(x))
    even_low = (f // 2) * 2
    odd_low = even_low + 1
    return sorted({even_low, even_low-2, even_low+2, odd_low, odd_low-2, odd_low+2})

# ---------------- Forecast sources ----------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None) -> dict | None:
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

def extract_open_meteo_today(j: dict, tz: str) -> tuple[float | None, pd.DataFrame | None, datetime | None, str | None]:
    if not j:
        return None, None, None, "Open‑Meteo fetch failed"
    try:
        daily = j.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        dtime = daily.get("time", [])
        sunr = daily.get("sunrise", [])
        if not highs or not dtime:
            return None, None, None, "Open‑Meteo missing daily data"

        t0 = today_local(tz).isoformat()
        idx = dtime.index(t0) if t0 in dtime else 0
        daily_high = safe_float(highs[idx])

        sunrise_dt = None
        if sunr and idx < len(sunr):
            sunrise_dt = datetime.fromisoformat(sunr[idx]).replace(tzinfo=ZoneInfo(tz))

        hourly = j.get("hourly", {})
        ht = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        df = None
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                try:
                    dt = datetime.fromisoformat(ts).replace(tzinfo=ZoneInfo(tz))
                except Exception:
                    continue
                rows.append({"dt": dt, "temp_f": safe_float(temp)})
            df = pd.DataFrame(rows).dropna()
            df = df[df["dt"].dt.date == today_local(tz)].copy()
            if df.empty:
                df = None
        return daily_high, df, sunrise_dt, None
    except Exception as e:
        return None, None, None, f"Open‑Meteo parse error: {e}"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_nws_hourly(lat: float, lon: float, tz: str) -> tuple[pd.DataFrame | None, str | None]:
    p = http_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not p:
        return None, "NWS points fetch failed"
    hourly_url = (p.get("properties") or {}).get("forecastHourly")
    if not hourly_url:
        return None, "NWS missing forecastHourly"
    h = http_get_json(hourly_url)
    if not h:
        return None, "NWS hourly fetch failed"
    periods = (h.get("properties") or {}).get("periods") or []
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
        return None, "NWS hourly returned no usable rows"
    df = df[df["dt"].dt.date == today_local(tz)].copy()
    if df.empty:
        return None, "NWS hourly has no rows for today"
    return df, None

# ---------------- Biggest improvement: Live station selection ----------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_nws_observation_station(lat: float, lon: float) -> tuple[dict | None, str | None]:
    p = http_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not p:
        return None, "NWS points fetch failed"
    obs_url = (p.get("properties") or {}).get("observationStations")
    if not obs_url:
        return None, "NWS missing observationStations URL"
    s = http_get_json(obs_url)
    if not s:
        return None, "NWS station list fetch failed"
    feats = s.get("features") or []
    for feat in feats[:8]:
        props = feat.get("properties") or {}
        station_id = props.get("stationIdentifier") or props.get("id") or ""
        name = props.get("name") or station_id
        if not station_id:
            continue
        j = http_get_json(f"https://api.weather.gov/stations/{station_id}/observations/latest")
        if not j:
            continue
        if (j.get("properties") or {}).get("temperature") is None:
            continue
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        return {"station_id": station_id, "name": name, "lon": coords[0], "lat": coords[1]}, None
    return None, "No working station with latest observation found."

@st.cache_data(ttl=120, show_spinner=False)
def fetch_nws_latest_temp_f(station_id: str) -> tuple[float | None, datetime | None, str | None]:
    j = http_get_json(f"https://api.weather.gov/stations/{station_id}/observations/latest")
    if not j:
        return None, None, "NWS latest observation fetch failed"
    props = j.get("properties") or {}
    t = (props.get("temperature") or {}).get("value")  # Celsius
    if t is None:
        return None, None, "NWS station has no temperature value"
    temp_f = (float(t) * 9.0 / 5.0) + 32.0
    ts = props.get("timestamp")
    obs_dt = None
    if ts:
        try:
            obs_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            obs_dt = None
    return temp_f, obs_dt, None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_metar_temp_f(icao: str) -> tuple[float | None, datetime | None, str | None]:
    txt = http_get_text(f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw&hours=2")
    if not txt:
        return None, None, "METAR fetch failed"
    line = txt.strip().splitlines()[-1].strip()
    m = re.search(r"\\b(M?\\d{2})/(M?\\d{2})\\b", line)
    if not m:
        return None, None, "METAR parse failed"
    t = m.group(1)
    neg = t.startswith("M")
    c = int(t[1:]) if neg else int(t)
    if neg:
        c = -c
    temp_f = (c * 9.0 / 5.0) + 32.0
    t2 = re.search(r"\\b(\\d{2})(\\d{2})(\\d{2})Z\\b", line)
    obs_dt = None
    if t2:
        dd, hh, mm = map(int, t2.groups())
        nowu = datetime.utcnow()
        try:
            obs_dt = datetime(nowu.year, nowu.month, dd, hh, mm)
        except Exception:
            obs_dt = None
    return temp_f, obs_dt, None

# ---------------- Kalshi import ----------------

def _american_odds_to_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    a = abs(odds)
    return a / (a + 100.0)

def _price_to_prob(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace("¢", "").replace("%", "").replace(" ", "")
    if not s:
        return None
    if re.fullmatch(r"[+-]\\d{2,5}", s):
        return _american_odds_to_prob(float(s))
    try:
        v = float(s)
    except Exception:
        return None
    if 0 <= v <= 1:
        return v
    if 0 <= v <= 100:
        return v / 100.0
    if abs(v) >= 100:
        return _american_odds_to_prob(v)
    return None

def parse_kalshi_from_next_data(html: str) -> pd.DataFrame | None:
    if not html:
        return None
    m = re.search(r'id="__NEXT_DATA__"\\s*type="application/json"\\s*>\\s*(\\{.*?\\})\\s*</script>', html, re.S)
    if not m:
        return None
    raw = m.group(1)
    try:
        import json
        data = json.loads(raw)
    except Exception:
        return None
    rows = []

    def walk(obj):
        if isinstance(obj, dict):
            keys = set(obj.keys())
            if any(k in keys for k in ["yesBid", "yesAsk", "yes_bid", "yes_ask", "yesPrice", "yes_price"]):
                label = obj.get("title") or obj.get("name") or obj.get("label") or obj.get("ticker") or ""
                yes_bid = obj.get("yesBid", obj.get("yes_bid"))
                yes_ask = obj.get("yesAsk", obj.get("yes_ask"))
                yes_px = obj.get("yesPrice", obj.get("yes_price"))
                px = yes_px if yes_px is not None else (yes_ask if yes_ask is not None else yes_bid)
                if label and px is not None:
                    rows.append({"Outcome": str(label), "YesRaw": str(px)})
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(data)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates()
    df["MarketProb"] = df["YesRaw"].apply(_price_to_prob)
    df = df.dropna(subset=["MarketProb"])
    return None if df.empty else df[["Outcome", "YesRaw", "MarketProb"]].reset_index(drop=True)

def parse_kalshi_from_pasted_table(txt: str) -> pd.DataFrame | None:
    if not txt:
        return None
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return None
    rows = []
    for ln in lines:
        br = None
        m = re.search(r"(\\d{2,3})\\s*(?:°)?\\s*(?:to|–|-)\\s*(\\d{2,3})", ln, re.I)
        if m:
            lo = int(m.group(1)); hi = int(m.group(2))
            br = f"{lo}–{hi}"
        tokens = re.findall(r"[+-]\\d{2,5}|\\b\\d{1,3}\\b", ln)
        if br and tokens:
            picked = None
            for t in tokens:
                if re.fullmatch(r"\\d{2,3}", t) and (t == m.group(1) or t == m.group(2)):
                    continue
                picked = t
                break
            if picked is None:
                picked = tokens[0]
            pr = _price_to_prob(picked)
            if pr is not None:
                rows.append({"Outcome": br, "YesRaw": picked, "MarketProb": pr})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates()

# ---------------- UI ----------------

left, right = st.columns([1, 1])
with left:
    city = st.selectbox("City", list(CITIES.keys()))
with right:
    st.caption("Use ~10:00–11:15 local if possible. If uncertainty is high, pass.")

info = CITIES[city]
tz = info["tz"]; lat = info["lat"]; lon = info["lon"]

with st.expander("Settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_nws = st.toggle("Include NWS forecast (api.weather.gov)", value=True)
    with c2:
        use_gfs = st.toggle("Include Open‑Meteo GFS (extra check)", value=False)
    with c3:
        bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=1)
    with c4:
        grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)

lock_time_local = datetime.combine(today_local(tz), datetime.min.time().replace(hour=10, minute=30), tzinfo=ZoneInfo(tz))
deadline_dt = lock_time_local + timedelta(minutes=int(grace_minutes))

with st.spinner("Fetching forecasts + live station…"):
    om = fetch_open_meteo(lat, lon, tz, model=None)
    om_high, om_hourly, sunrise_dt, om_err = extract_open_meteo_today(om or {}, tz)

    gfs_high = None; gfs_hourly = None; gfs_err = None
    if use_gfs:
        om_gfs = fetch_open_meteo(lat, lon, tz, model="gfs_seamless")
        gfs_high, gfs_hourly, _sun2, gfs_err = extract_open_meteo_today(om_gfs or {}, tz)

    nws_hourly = None; nws_err = None; nws_high = None
    if use_nws:
        nws_hourly, nws_err = fetch_nws_hourly(lat, lon, tz)
        if nws_hourly is not None and not nws_hourly.empty:
            nws_high = float(nws_hourly["temp_f"].max())

    station_meta, station_err = fetch_nws_observation_station(lat, lon)
    live_temp = None; live_obs_dt = None; live_source = None; live_err = None
    if station_meta:
        live_temp, live_obs_dt, live_err = fetch_nws_latest_temp_f(station_meta["station_id"])
        if live_temp is not None:
            live_source = f"NWS station {station_meta['station_id']}"
    if live_temp is None:
        icao = ICAO_FALLBACK.get(city)
        if icao:
            t, dtu, e = fetch_metar_temp_f(icao)
            if t is not None:
                live_temp, live_obs_dt, live_source = t, dtu, f"METAR {icao}"
            else:
                live_err = live_err or e

# Source table
st.subheader(f"{city} – Forecast Sources (Daily High °F)")
rows = []
def add_row(name, val, err):
    rows.append((name, f"{val:.1f}°F" if isinstance(val, (int, float)) and val is not None else "—",
                 "OK" if err is None and val is not None else (err or "Error")))
add_row("Open‑Meteo", om_high, om_err)
if use_gfs:
    add_row("Open‑Meteo (GFS)", gfs_high, gfs_err)
if use_nws:
    add_row("NWS hourly (max today)", nws_high, nws_err)
st.table(pd.DataFrame(rows, columns=["Source", "Today High", "Status"]))

# Live / station
st.subheader("Live Temperature (station shown)")
cA, cB, cC = st.columns([1.2, 1.2, 2.0])
cA.metric("Live temp (°F)", f"{live_temp:.1f}" if live_temp is not None else "—")
cB.metric("Observed (UTC)", live_obs_dt.strftime("%H:%M") if live_obs_dt else "—")
if station_meta:
    cC.caption(f"Using: {live_source} — {station_meta['name']}")
else:
    cC.caption(f"Using: {live_source or '—'}")
if station_err:
    st.caption(f"Station note: {station_err}")
if live_err and live_temp is None:
    st.caption(f"Live temp error: {live_err}")

# Heating rate since sunrise + nowcast
heating_rate = None; sunrise_temp = None
if sunrise_dt and live_temp is not None and om_hourly is not None and not om_hourly.empty:
    dfh = om_hourly.sort_values("dt").copy()
    df_after = dfh[dfh["dt"] >= sunrise_dt]
    if not df_after.empty:
        sunrise_temp = float(df_after.iloc[0]["temp_f"])
        hours_since = max(0.01, (now_local(tz) - sunrise_dt).total_seconds() / 3600.0)
        heating_rate = (live_temp - sunrise_temp) / hours_since

peak_time = None; peak_temp = None
if om_hourly is not None and not om_hourly.empty:
    dfh = om_hourly.sort_values("dt").copy()
    peak_idx = dfh["temp_f"].astype(float).idxmax()
    peak_time = dfh.loc[peak_idx, "dt"]
    peak_temp = float(dfh.loc[peak_idx, "temp_f"])

nowcast_high = None
if live_temp is not None and heating_rate is not None and peak_time is not None:
    hrs_to_peak = max(0.0, min(8.0, (peak_time - now_local(tz)).total_seconds() / 3600.0))
    taper = 0.65 if hrs_to_peak >= 3 else (0.45 if hrs_to_peak >= 1.5 else 0.25)
    nowcast_high = max(live_temp, live_temp + max(0.0, heating_rate) * hrs_to_peak * taper)

# Consensus
source_highs = [x for x in [om_high, gfs_high, nws_high] if isinstance(x, (int, float)) and x is not None]
if not source_highs:
    st.error("No valid forecast sources returned a high. Try again later.")
    st.stop()

weights = {"om": 0.55, "nws": 0.30, "gfs": 0.15}
if not use_gfs:
    weights["gfs"] = 0.0
if not use_nws:
    weights["nws"] = 0.0

w_om = weights["om"] if om_high is not None else 0.0
w_nws = weights["nws"] if nws_high is not None else 0.0
w_gfs = weights["gfs"] if gfs_high is not None else 0.0
w_sum = w_om + w_nws + w_gfs
consensus = float(sum(source_highs) / len(source_highs)) if w_sum <= 1e-9 else float(
    (w_om*(om_high or 0.0) + w_nws*(nws_high or 0.0) + w_gfs*(gfs_high or 0.0)) / w_sum
)

spread = (max(source_highs) - min(source_highs)) if len(source_highs) > 1 else 0.0

live_conf_boost = 0.0
if live_temp is not None and om_hourly is not None and not om_hourly.empty:
    dfh = om_hourly.sort_values("dt").copy()
    nowdt = now_local(tz)
    idx = (dfh["dt"] - nowdt).abs().idxmin()
    f_now = float(dfh.loc[idx, "temp_f"])
    mismatch = abs(live_temp - f_now)
    live_conf_boost = max(0.0, min(1.2, (mismatch - 2.0) / 3.0))

sigma = compute_sigma(source_highs, live_conf_boost=live_conf_boost)

mean_final = consensus
if nowcast_high is not None:
    local_hour = now_local(tz).hour + now_local(tz).minute / 60.0
    w_now = 0.25
    if local_hour >= 9:
        w_now = min(0.75, 0.25 + (local_hour - 9) * (0.50/6.0))
    mean_final = (1 - w_now) * consensus + w_now * nowcast_high

st.subheader("Key Numbers")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model consensus high", f"{consensus:.1f}°F")
m2.metric("Cross-source spread", f"{spread:.1f}°F")
m3.metric("Uncertainty σ", f"{sigma:.2f}°F")
m4.metric("Nowcast projected high", f"{nowcast_high:.1f}°F" if nowcast_high is not None else "—")
if heating_rate is not None and sunrise_temp is not None:
    st.caption(f"Heating rate since sunrise: {heating_rate:+.2f} °F/hr (sunrise temp {sunrise_temp:.1f}°F).")

# Suggested bracket
st.subheader("Suggested Kalshi Bracket")
cands = []
if bracket_size == 1:
    center = int(round(mean_final))
    for low in range(center - 6, center + 7):
        p = bracket_probability_integer_cc(mean_final, sigma, low, low)
        cands.append((low, low, p))
else:
    for low in two_degree_candidates_for_temp(mean_final):
        high = low + 1
        p = bracket_probability_integer_cc(mean_final, sigma, low, high)
        cands.append((low, high, p))
best_low, best_high, best_p = max(cands, key=lambda x: x[2])
st.success(f"Suggested bracket: **{best_low}–{best_high}** (model ≈ **{best_p*100:.0f}%**)" if bracket_size == 2
           else f"Suggested bracket: **{best_low}** (model ≈ **{best_p*100:.0f}%**)")

top_df = pd.DataFrame(
    [{"Bracket": f"{lo}–{hi}" if bracket_size == 2 else f"{lo}", "Model Prob %": round(p*100, 1)}
     for lo, hi, p in sorted(cands, key=lambda x: x[2], reverse=True)[:8]]
)
st.dataframe(top_df, use_container_width=True, hide_index=True)

# Decision window
st.subheader("Decision Window")
nowdt = now_local(tz)
st.write(
    f"Local time now: **{nowdt.strftime('%a %b %d, %I:%M %p')}** | Target check: **10:30 AM** | With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)
st.info("Inside the preferred window (or within grace)." if nowdt <= deadline_dt else
        "Past the preferred window. Often the market has already adjusted.")

# Hourly curve
st.subheader("Hourly Temperature Curve (Forecast)")
hourly_df = None; hourly_source = None
if om_hourly is not None and not om_hourly.empty:
    hourly_df = om_hourly.sort_values("dt").copy()
    hourly_source = "Open‑Meteo"
elif nws_hourly is not None and not nws_hourly.empty:
    hourly_df = nws_hourly.sort_values("dt").copy()
    hourly_source = "NWS"
if hourly_df is not None and not hourly_df.empty:
    st.caption(f"Hourly source used: {hourly_source}")
    st.line_chart(hourly_df.set_index("dt")["temp_f"])
    peak_idx = hourly_df["temp_f"].astype(float).idxmax()
    pk_dt = hourly_df.loc[peak_idx, "dt"]
    pk_temp = float(hourly_df.loc[peak_idx, "temp_f"])
    st.write(f"Forecast peak hour: **{pk_dt.strftime('%I:%M %p')}** at **{pk_temp:.1f}°F**")
else:
    st.caption("Hourly curve unavailable.")

st.divider()

# Kalshi auto import + edge
st.subheader("Kalshi Auto‑Import (best‑effort)")
st.caption("If URL import fails (Kalshi may require login), paste the market rows instead.")
col1, col2 = st.columns([1.5, 1.0])
with col1:
    kalshi_url = st.text_input("Kalshi market URL (optional)", value="", placeholder="Paste Kalshi market page URL")
with col2:
    edge_threshold = st.slider("Min edge to bet", 0.0, 0.25, 0.05, 0.01)
pasted = st.text_area("Or paste Kalshi table lines (odds or cents)", height=120)

market_df = None
note = None
if kalshi_url:
    html = http_get_text(kalshi_url)
    market_df = parse_kalshi_from_next_data(html or "")
    if market_df is None:
        note = "URL import failed (page might require login). Paste the table lines instead."
elif pasted.strip():
    market_df = parse_kalshi_from_pasted_table(pasted.strip())
    if market_df is None:
        note = "Could not parse pasted text. Include lines like: 86° to 87°   +100"
if note:
    st.warning(note)

if market_df is not None and not market_df.empty:
    model_rows = []
    for _, r in market_df.iterrows():
        out = str(r["Outcome"])
        nums = re.findall(r"\\d{2,3}", out)
        if bracket_size == 2 and len(nums) >= 2:
            lo, hi = int(nums[0]), int(nums[1])
            if hi < lo:
                lo, hi = hi, lo
            p = bracket_probability_integer_cc(mean_final, sigma, lo, hi)
            model_rows.append({"Outcome": out, "ModelProb": p})
        elif bracket_size == 1 and len(nums) >= 1:
            lo = int(nums[0])
            p = bracket_probability_integer_cc(mean_final, sigma, lo, lo)
            model_rows.append({"Outcome": out, "ModelProb": p})

    if model_rows:
        mdf = pd.DataFrame(model_rows).drop_duplicates(subset=["Outcome"])
        df = market_df.merge(mdf, on="Outcome", how="inner")
        if not df.empty:
            df["Edge"] = df["ModelProb"] - df["MarketProb"]
            df = df.sort_values("Edge", ascending=False).reset_index(drop=True)

            show = df.copy()
            show["ModelProb %"] = (show["ModelProb"] * 100).round(1)
            show["MarketProb %"] = (show["MarketProb"] * 100).round(1)
            show["Edge %"] = (show["Edge"] * 100).round(1)
            show = show[["Outcome", "YesRaw", "ModelProb %", "MarketProb %", "Edge %"]]
            st.dataframe(show, use_container_width=True, hide_index=True)

            best = df.iloc[0]
            if best["Edge"] >= edge_threshold:
                st.success(f"Best edge: **{best['Outcome']}** | edge ≈ **{best['Edge']*100:.1f}%**")
            else:
                st.error(f"No edges ≥ {edge_threshold*100:.0f}%. Best found ≈ {best['Edge']*100:.1f}%")
        else:
            st.warning("Imported outcomes didn’t match bracket format. Paste lines like “86° to 87°  +100”.")
    else:
        st.warning("Couldn’t map market outcomes to brackets. Paste explicit lines like “86° to 87°  +100”.")
else:
    st.caption("No market imported yet.")

st.caption(
    "Note: Kalshi settlement uses a specific station/ruleset. This version shows the NWS observation station used for live temp "
    "so you can compare it to the market’s rules."
)
