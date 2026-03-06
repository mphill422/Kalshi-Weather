# streamlit_app.py
# Kalshi Weather Model – Ladder-Aligned (Daily High) [v6]
# (single-file Streamlit app)

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Weather Model – Ladder-Aligned (Daily High)", layout="centered")
st.title("Kalshi Weather Model – Ladder-Aligned (Daily High)")

st.caption(
    "Auto-builds a Kalshi-style ladder with win % per bracket, live airport temp (NWS station), "
    "heating rate since sunrise + projected high, optional mispricing/edge, and a LOW / HIGH / PASS indicator."
)

# -----------------------------
# City / Station mapping
# -----------------------------
CITIES: Dict[str, Dict[str, str | float]] = {
    "Miami": {"lat": 25.7933, "lon": -80.2906, "station": "KMIA", "tz": "America/New_York"},
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "station": "KPHX", "tz": "America/Phoenix"},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "station": "KIAH", "tz": "America/Chicago"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "station": "KDFW", "tz": "America/Chicago"},
    "Austin": {"lat": 30.1975, "lon": -97.6664, "station": "KAUS", "tz": "America/Chicago"},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "station": "KMSY", "tz": "America/Chicago"},
    "San Antonio": {"lat": 29.5337, "lon": -98.4698, "station": "KSAT", "tz": "America/Chicago"},
    "Las Vegas": {"lat": 36.0801, "lon": -115.1522, "station": "KLAS", "tz": "America/Los_Angeles"},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "station": "KLAX", "tz": "America/Los_Angeles"},
}
DEFAULT_CITY = "Miami"

UA = {"User-Agent": "kalshi-weather-model/6 (streamlit)"}

def safe_get_json(url: str, timeout: int = 12) -> Optional[dict]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def prob_between(mu: float, sigma: float, a: float, b: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (a <= mu <= b) else 0.0
    return max(0.0, min(1.0, norm_cdf((b - mu) / sigma) - norm_cdf((a - mu) / sigma)))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def parse_iso(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def american_odds_to_prob(odds: float) -> Optional[float]:
    try:
        if odds == 0:
            return None
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return (-odds) / ((-odds) + 100.0)
    except Exception:
        return None

def cents_to_prob(cents: float) -> Optional[float]:
    if cents < 0 or cents > 100:
        return None
    return cents / 100.0

@dataclass
class SourcePoint:
    name: str
    today_high_f: float
    ok: bool
    note: str = ""

def fetch_open_meteo(lat: float, lon: float, tz: str, model: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[dict], str]:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "timezone": tz,
        "hourly": "temperature_2m",
        "daily": "sunrise,sunset,temperature_2m_max",
        "forecast_days": 2,
    }
    if model:
        params["models"] = model  # best-effort (skips if unsupported)
    try:
        r = requests.get(base, params=params, headers=UA, timeout=12)
        if r.status_code != 200:
            return None, None, None, f"HTTP {r.status_code}"
        js = r.json()
        hourly = js.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        if not times or not temps or len(times) != len(temps):
            return None, None, js, "Missing hourly temps"
        df = pd.DataFrame({"time": times, "temp_f": temps})
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize(ZoneInfo(tz), ambiguous="infer", nonexistent="shift_forward")
        daily = js.get("daily", {})
        tmax = daily.get("temperature_2m_max", [])
        today_high = float(tmax[0]) if tmax else float(df["temp_f"].max())
        return df, today_high, js, ""
    except Exception as e:
        return None, None, None, str(e)

def fetch_nws_hourly_forecast(lat: float, lon: float) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    js = safe_get_json(points_url)
    if not js:
        return None, None, "NWS points lookup failed"
    fh_url = js.get("properties", {}).get("forecastHourly")
    if not fh_url:
        return None, None, "NWS forecastHourly URL missing"
    fh = safe_get_json(fh_url)
    if not fh:
        return None, None, "NWS forecastHourly fetch failed"
    periods = fh.get("properties", {}).get("periods", [])
    if not periods:
        return None, None, "NWS hourly periods missing"
    rows = []
    for p in periods:
        t = parse_iso(p.get("startTime", ""))
        temp = p.get("temperature", None)
        if t is None or temp is None:
            continue
        rows.append({"time": t, "temp_f": float(temp)})
    if not rows:
        return None, None, "NWS hourly parse failed"
    df = pd.DataFrame(rows).sort_values("time")
    first_date = df["time"].iloc[0].date()
    today_mask = df["time"].apply(lambda x: x.date() == first_date)
    today_high = float(df.loc[today_mask, "temp_f"].max()) if today_mask.any() else float(df["temp_f"].max())
    return df, today_high, ""

def fetch_nws_station_obs(station: str) -> Tuple[Optional[float], Optional[datetime], str]:
    url = f"https://api.weather.gov/stations/{station}/observations/latest"
    js = safe_get_json(url)
    if not js:
        return None, None, "Station obs fetch failed"
    props = js.get("properties", {})
    ts = parse_iso(props.get("timestamp", ""))
    temp_c = props.get("temperature", {}).get("value", None)
    if ts is None or temp_c is None:
        return None, None, "Station obs missing temp/timestamp"
    return c_to_f(float(temp_c)), ts, ""

@dataclass
class LadderRow:
    label: str
    lo: Optional[float]
    hi: Optional[float]
    model_prob: float
    market_prob: Optional[float] = None
    edge: Optional[float] = None

def build_kalshi_ladder(mu: float, sigma: float) -> List[LadderRow]:
    base = int(round(mu))
    if base % 2 == 1:
        base -= 1
    bins = [
        (None, base - 3, f"{base-3}° or below"),
        (base - 2, base - 1, f"{base-2}° to {base-1}°"),
        (base, base + 1, f"{base}° to {base+1}°"),
        (base + 2, base + 3, f"{base+2}° to {base+3}°"),
        (base + 4, base + 5, f"{base+4}° to {base+5}°"),
        (base + 6, None, f"{base+6}° or above"),
    ]
    rows: List[LadderRow] = []
    for lo, hi, label in bins:
        if lo is None:
            prob = norm_cdf((hi - mu) / sigma) if sigma > 1e-9 else (1.0 if mu <= hi else 0.0)
        elif hi is None:
            prob = 1.0 - norm_cdf((lo - mu) / sigma) if sigma > 1e-9 else (1.0 if mu >= lo else 0.0)
        else:
            prob = prob_between(mu, sigma, lo, hi + 1e-9)
        rows.append(LadderRow(label=label, lo=lo, hi=hi, model_prob=prob))
    s = sum(r.model_prob for r in rows)
    if s > 0:
        for r in rows:
            r.model_prob /= s
    return rows

def parse_kalshi_lines_to_market_probs(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        odds_match = re.search(r"([+-]\d{2,5})", line)
        cents_match = re.search(r"(\d{1,3})\s*(?:c|¢)\b", line, flags=re.IGNORECASE)

        label = line
        prob = None
        if odds_match:
            odds = float(odds_match.group(1))
            prob = american_odds_to_prob(odds)
            label = line[: odds_match.start()].strip()
        elif cents_match:
            cents = float(cents_match.group(1))
            prob = cents_to_prob(cents)
            label = line[: cents_match.start()].strip()

        if prob is None:
            continue

        label = label.replace("º", "°")
        label = re.sub(r"\s+", " ", label).strip()
        if label:
            out[label] = float(clamp(prob, 0.0, 1.0))
    return out

def classify_risk(spread_f: float, sigma_f: float, top_prob: float) -> Tuple[str, str]:
    if spread_f >= 4.0 or sigma_f >= 3.5 or top_prob < 0.20:
        return "PASS", "Uncertainty is too high (spread/σ) or the top bracket isn't strong enough."
    if spread_f >= 2.5 or sigma_f >= 3.0 or top_prob < 0.25:
        return "HIGH RISK", "Playable only with a clear edge vs market odds; otherwise skip."
    return "LOW RISK", "Models are relatively aligned and top bracket is reasonably strong."

def decision_window_box(local_now: datetime, grace_minutes: int) -> Tuple[str, str]:
    target = local_now.replace(hour=10, minute=30, second=0, microsecond=0)
    if local_now < target:
        mins = int((target - local_now).total_seconds() / 60)
        return "EARLY", f"Early. Target check is 10:30 AM local (in ~{mins} min)."
    grace_end = target + timedelta(minutes=grace_minutes)
    if local_now <= grace_end:
        return "WINDOW", "Inside preferred window (or grace)."
    return "LATE", "Past preferred window. Market is often sharper later in the day."

def estimate_temp_at_time_from_hourly(df: Optional[pd.DataFrame], target: datetime) -> Optional[float]:
    if df is None or df.empty:
        return None
    idx = (df["time"] - target).abs().idxmin()
    try:
        return float(df.loc[idx, "temp_f"])
    except Exception:
        return None

# -----------------------------
# UI controls
# -----------------------------
city = st.selectbox("City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY))
cfg = CITIES[city]
lat = float(cfg["lat"])
lon = float(cfg["lon"])
station = str(cfg["station"])
tz = str(cfg["tz"])
tzinfo = ZoneInfo(tz)
local_now = datetime.now(tzinfo)

with st.expander("Settings", expanded=True):
    include_gfs = st.toggle("Include Open-Meteo GFS (extra check)", value=True)
    include_nws = st.toggle("Include NWS (api.weather.gov) hourly forecast", value=True)
    include_hrrr = st.toggle("Include HRRR (best-effort)", value=True)
    show_hourly_chart = st.toggle("Show hourly chart", value=True)
    grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)

    st.divider()
    st.subheader("Market / Edge (optional)")
    use_market = st.toggle("Paste Kalshi market YES odds/price (to compute edge)", value=False)
    min_edge = st.slider("Min edge to flag (model - market)", 0.0, 0.25, 0.05, 0.01)
    pasted_lines = ""
    if use_market:
        pasted_lines = st.text_area(
            "Paste Kalshi contract lines (one per line) with YES odds/price",
            height=160,
            placeholder="Example:\n80° to 81°  -144\n82° to 83°  +156\n77° or below  6c\n86° or above  8c",
        )

# -----------------------------
# Fetch forecasts
# -----------------------------
sources: List[SourcePoint] = []
hourly_for_chart: Optional[pd.DataFrame] = None
sunrise_local: Optional[datetime] = None

om_df, om_high, om_js, om_err = fetch_open_meteo(lat, lon, tz, model=None)
if om_high is not None:
    sources.append(SourcePoint("Open-Meteo", float(om_high), True))
    hourly_for_chart = om_df
    try:
        sunrise_str = (om_js or {}).get("daily", {}).get("sunrise", [None])[0]
        if sunrise_str:
            sunrise_local = pd.to_datetime(sunrise_str).tz_localize(tzinfo, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        sunrise_local = None
else:
    sources.append(SourcePoint("Open-Meteo", float("nan"), False, om_err))

if include_gfs:
    gfs_df, gfs_high, _, gfs_err = fetch_open_meteo(lat, lon, tz, model="gfs")
    sources.append(SourcePoint("Open-Meteo (GFS)", float(gfs_high) if gfs_high is not None else float("nan"), gfs_high is not None, gfs_err))

if include_nws:
    nws_df, nws_high, nws_err = fetch_nws_hourly_forecast(lat, lon)
    sources.append(SourcePoint("NWS (forecastHourly)", float(nws_high) if nws_high is not None else float("nan"), nws_high is not None, nws_err))

if include_hrrr:
    hrrr_df, hrrr_high, _, hrrr_err = fetch_open_meteo(lat, lon, tz, model="hrrr")
    sources.append(SourcePoint("HRRR (best-effort)", float(hrrr_high) if hrrr_high is not None else float("nan"), hrrr_high is not None, hrrr_err))

obs_temp_f, obs_time, obs_err = fetch_nws_station_obs(station)

ok_vals = [s.today_high_f for s in sources if s.ok and not math.isnan(s.today_high_f)]
if not ok_vals:
    st.error("No forecast sources returned successfully.")
    st.stop()

consensus = float(sum(ok_vals) / len(ok_vals))
spread = float(max(ok_vals) - min(ok_vals)) if len(ok_vals) > 1 else 0.0
sigma = float(max(1.2, 0.9 + 0.55 * spread))

# -----------------------------
# Display forecast table
# -----------------------------
st.subheader(f"{city} – Today’s High Forecasts (°F)")
df_sources = pd.DataFrame([{
    "Source": s.name,
    "Today High": (f"{s.today_high_f:.1f}°F" if s.ok and not math.isnan(s.today_high_f) else "—"),
    "Status": ("OK" if s.ok else "ERR"),
    "Note": s.note
} for s in sources])
st.dataframe(df_sources, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("Consensus high", f"{consensus:.1f}°F")
    st.metric("Cross-source spread", f"{spread:.1f}°F")
with c2:
    st.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")
    if obs_temp_f is not None and obs_time is not None:
        st.metric(f"Current airport temp ({station})", f"{obs_temp_f:.1f}°F")
        st.caption(f"Obs time: {obs_time.astimezone(tzinfo).strftime('%a %b %d, %I:%M %p')} local")
    else:
        st.metric(f"Current airport temp ({station})", "—")
        st.caption(f"Obs error: {obs_err}")

# -----------------------------
# Live trend / nowcast
# -----------------------------
st.divider()
st.subheader("Live trend / nowcast")

heating_rate = None
trend_proj_high = None
peak_hour = None

if obs_temp_f is not None:
    sunrise_for_calc = sunrise_local or local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    sunrise_temp = estimate_temp_at_time_from_hourly(hourly_for_chart, sunrise_for_calc)
    if sunrise_temp is not None:
        hours = max(0.25, (local_now - sunrise_for_calc).total_seconds() / 3600.0)
        heating_rate = (obs_temp_f - sunrise_temp) / hours

    if hourly_for_chart is not None and not hourly_for_chart.empty:
        df_day = hourly_for_chart[hourly_for_chart["time"].dt.date == local_now.date()].copy()
        df_day = df_day[(df_day["time"].dt.hour >= 8) & (df_day["time"].dt.hour <= 20)]
        if not df_day.empty:
            peak_row = df_day.loc[df_day["temp_f"].idxmax()]
            peak_hour = peak_row["time"]
            if heating_rate is not None:
                hours_to_peak = (peak_hour - local_now).total_seconds() / 3600.0
                trend_proj_high = obs_temp_f + heating_rate * max(0.0, hours_to_peak)

a, b, c = st.columns(3)
with a:
    st.metric("Heating rate since sunrise", (f"{heating_rate:+.2f} °F/hr" if heating_rate is not None else "—"))
with b:
    st.metric("Forecast peak hour", (peak_hour.strftime("%I:%M %p") if isinstance(peak_hour, datetime) else "—"))
with c:
    st.metric("Projected high (trend-based)", (f"{trend_proj_high:.1f}°F" if trend_proj_high is not None else "—"))

# -----------------------------
# Suggested bracket + win %
# -----------------------------
st.divider()
st.subheader("Suggested Kalshi Bracket (auto ladder)")

ladder = build_kalshi_ladder(consensus, sigma)
top = max(ladder, key=lambda r: r.model_prob)
risk_level, risk_msg = classify_risk(spread, sigma, top.model_prob)

if risk_level == "LOW RISK":
    st.success(f"🟢 {risk_level}: {risk_msg}")
elif risk_level == "HIGH RISK":
    st.warning(f"🟡 {risk_level}: {risk_msg}")
else:
    st.error(f"🔴 {risk_level}: {risk_msg}")

st.success(f"Suggested bracket: **{top.label}** (model ≈ **{top.model_prob*100:.0f}%**)")

df_ladder = pd.DataFrame([{"Bracket": r.label, "Win %": f"{r.model_prob*100:.1f}%"} for r in sorted(ladder, key=lambda r: r.model_prob, reverse=True)])
st.dataframe(df_ladder, use_container_width=True, hide_index=True)

# -----------------------------
# Decision window
# -----------------------------
st.divider()
st.subheader("Decision Window")
status, note = decision_window_box(local_now, grace_minutes=grace_minutes)
st.caption(f"Local time now: **{local_now.strftime('%a %b %d, %I:%M %p')}**  | Target check: **10:30 AM** | Grace: **{grace_minutes} min**")
if status == "WINDOW":
    st.info("Inside preferred window (or within grace).")
else:
    st.warning(note)

# -----------------------------
# Mispricing / edge (optional)
# -----------------------------
if use_market:
    market_probs = parse_kalshi_lines_to_market_probs(pasted_lines)
    if market_probs:
        for r in ladder:
            if r.label in market_probs:
                r.market_prob = market_probs[r.label]
                r.edge = r.model_prob - r.market_prob

        df_edge = pd.DataFrame([{
            "Bracket": r.label,
            "Model %": r.model_prob * 100,
            "Market %": (r.market_prob * 100 if r.market_prob is not None else None),
            "Edge %": (r.edge * 100 if r.edge is not None else None),
        } for r in ladder]).sort_values(by="Edge %", ascending=False, na_position="last")

        st.subheader("Mispricing / Edge (model vs market)")
        st.dataframe(df_edge, use_container_width=True, hide_index=True)

        best = None
        for r in ladder:
            if r.edge is not None and r.edge >= min_edge:
                if best is None or r.edge > best.edge:
                    best = r
        if best is not None:
            st.success(f"💰 Mispriced bracket: **{best.label}** | Edge **{best.edge*100:+.1f}%**")
        else:
            st.info("No bracket exceeds your edge threshold yet (or labels didn't match).")
    else:
        st.info("Paste market lines to compute edge.")

# -----------------------------
# Hourly chart
# -----------------------------
if show_hourly_chart and hourly_for_chart is not None and not hourly_for_chart.empty:
    st.divider()
    st.subheader("Hourly temperature curve (today)")
    df_plot = hourly_for_chart[hourly_for_chart["time"].dt.date == local_now.date()].copy()
    if not df_plot.empty:
        st.line_chart(df_plot.set_index("time")["temp_f"])
        peak_t = df_plot.loc[df_plot["temp_f"].idxmax(), "time"]
        peak_v = float(df_plot["temp_f"].max())
        st.caption(f"Peak hour (forecast): {peak_t.strftime('%I:%M %p')} at {peak_v:.1f}°F")
    else:
        st.caption("No hourly points for today available.")

st.divider()
st.caption(
    "Kalshi settlement uses a specific station/ruleset. This app shows the NWS airport observation station so you can compare it to the market Rules Summary."
)
