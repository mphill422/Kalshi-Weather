# streamlit_app.py
# Kalshi Weather Model – Daily High [v7 clean rebuild]

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Weather Model – Daily High", layout="centered")
st.title("Kalshi Weather Model – Daily High")
st.caption("Fresh rebuild: corrected ladder logic, LOW / HIGH RISK / PASS restored, win % shown next to each bracket.")

UA = {"User-Agent": "kalshi-weather-model/7"}

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

def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 12):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
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
        return 1.0 if a <= mu <= b else 0.0
    return max(0.0, min(1.0, norm_cdf((b - mu) / sigma) - norm_cdf((a - mu) / sigma)))

def parse_iso(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def fetch_open_meteo(lat: float, lon: float, tz: str, model: Optional[str] = None):
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
        params["models"] = model
    js = safe_get_json("https://api.open-meteo.com/v1/forecast", params=params)
    if not js:
        return None, None, None, "fetch failed"
    try:
        times = js["hourly"]["time"]
        temps = js["hourly"]["temperature_2m"]
        df = pd.DataFrame({"time": pd.to_datetime(times), "temp_f": temps})
        df["time"] = df["time"].dt.tz_localize(ZoneInfo(tz), ambiguous="infer", nonexistent="shift_forward")
        tmax = float(js["daily"]["temperature_2m_max"][0])
        sunrise = pd.to_datetime(js["daily"]["sunrise"][0]).tz_localize(ZoneInfo(tz), ambiguous="infer", nonexistent="shift_forward")
        return df, tmax, sunrise, ""
    except Exception as e:
        return None, None, None, str(e)

def fetch_nws_hourly(lat: float, lon: float):
    pts = safe_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not pts:
        return None, None, "points failed"
    url = pts.get("properties", {}).get("forecastHourly")
    if not url:
        return None, None, "forecastHourly missing"
    fh = safe_get_json(url)
    if not fh:
        return None, None, "forecastHourly failed"
    periods = fh.get("properties", {}).get("periods", [])
    rows = []
    for p in periods:
        t = parse_iso(p.get("startTime", ""))
        temp = p.get("temperature")
        if t is not None and temp is not None:
            rows.append({"time": t, "temp_f": float(temp)})
    if not rows:
        return None, None, "parse failed"
    df = pd.DataFrame(rows).sort_values("time")
    d0 = df["time"].iloc[0].date()
    today_high = float(df[df["time"].dt.date == d0]["temp_f"].max())
    return df, today_high, ""

def fetch_station_obs(station: str):
    js = safe_get_json(f"https://api.weather.gov/stations/{station}/observations/latest")
    if not js:
        return None, None, "station failed"
    props = js.get("properties", {})
    ts = parse_iso(props.get("timestamp", ""))
    temp_c = props.get("temperature", {}).get("value")
    if ts is None or temp_c is None:
        return None, None, "obs missing"
    return c_to_f(float(temp_c)), ts, ""

def build_ladder(consensus: float, sigma: float, mode: str):
    even_start = int(2 * round(consensus / 2.0))
    odd_start = even_start - 1

    def one(start_low: int):
        bins = [
            (None, start_low - 1, f"{start_low-1}° or below"),
            (start_low, start_low + 1, f"{start_low}° to {start_low+1}°"),
            (start_low + 2, start_low + 3, f"{start_low+2}° to {start_low+3}°"),
            (start_low + 4, start_low + 5, f"{start_low+4}° to {start_low+5}°"),
            (start_low + 6, start_low + 7, f"{start_low+6}° to {start_low+7}°"),
            (start_low + 8, None, f"{start_low+8}° or above"),
        ]
        rows = []
        for lo, hi, label in bins:
            if lo is None:
                p = norm_cdf((hi - consensus) / sigma)
            elif hi is None:
                p = 1.0 - norm_cdf((lo - consensus) / sigma)
            else:
                p = prob_between(consensus, sigma, lo, hi + 1e-9)
            rows.append({"Bracket": label, "Win %": p})
        s = sum(x["Win %"] for x in rows)
        for x in rows:
            x["Win %"] = x["Win %"] / s if s > 0 else 0.0
        return rows

    ladders = {"even": one(even_start), "odd": one(odd_start)}
    if mode in ("even", "odd"):
        chosen = mode
    else:
        chosen = max(ladders.keys(), key=lambda k: max(x["Win %"] for x in ladders[k]))
    return ladders[chosen], chosen

def classify_risk(spread: float, sigma: float, top_prob: float):
    if spread >= 4.0 or sigma >= 3.5 or top_prob < 0.20:
        return "PASS", "🔴", "Uncertainty is too high (spread/σ) or the top bracket is not strong enough."
    if spread >= 2.5 or sigma >= 3.0 or top_prob < 0.25:
        return "HIGH RISK", "🟡", "Playable only with a clear edge vs market odds; otherwise skip."
    return "LOW RISK", "🟢", "Models are relatively aligned and the top bracket is reasonably strong."

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
    include_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    include_hrrr = st.toggle("Include HRRR (best-effort)", value=True)
    show_hourly_chart = st.toggle("Show hourly chart", value=True)
    grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["auto", "even", "odd"], index=0)

sources = []
chart_df = None
sunrise_local = None

om_df, om_high, om_sunrise, om_err = fetch_open_meteo(lat, lon, tz)
sources.append(("Open-Meteo", om_high, "OK" if om_high is not None else "ERR", om_err))
if om_high is not None:
    chart_df = om_df
    sunrise_local = om_sunrise

if include_gfs:
    _, gfs_high, _, gfs_err = fetch_open_meteo(lat, lon, tz, model="gfs")
    sources.append(("Open-Meteo (GFS)", gfs_high, "OK" if gfs_high is not None else "ERR", gfs_err))

if include_nws:
    _, nws_high, nws_err = fetch_nws_hourly(lat, lon)
    sources.append(("NWS (forecastHourly)", nws_high, "OK" if nws_high is not None else "ERR", nws_err))

if include_hrrr:
    _, hrrr_high, _, hrrr_err = fetch_open_meteo(lat, lon, tz, model="hrrr")
    sources.append(("HRRR (best-effort)", hrrr_high, "OK" if hrrr_high is not None else "ERR", hrrr_err))

obs_temp_f, obs_time, obs_err = fetch_station_obs(station)

vals = [x[1] for x in sources if x[1] is not None]
if not vals:
    st.error("No forecast sources returned successfully.")
    st.stop()

consensus = float(sum(vals) / len(vals))
spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
sigma = float(max(1.2, 0.9 + 0.55 * spread))

st.subheader(f"{city} – Today’s High Forecasts (°F)")
df_sources = pd.DataFrame([{
    "Source": s[0],
    "Today High": ("—" if s[1] is None else f"{s[1]:.1f}°F"),
    "Status": s[2],
    "Note": s[3],
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

st.divider()
st.subheader("Live trend / nowcast")

heating_rate = None
peak_hour = None
trend_proj_high = None

if obs_temp_f is not None and chart_df is not None and not chart_df.empty:
    sunrise_for_calc = sunrise_local or local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    idx = (chart_df["time"] - sunrise_for_calc).abs().idxmin()
    sunrise_temp = float(chart_df.loc[idx, "temp_f"])
    hrs = max(0.25, (local_now - sunrise_for_calc).total_seconds() / 3600.0)
    heating_rate = (obs_temp_f - sunrise_temp) / hrs

    day_df = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    day_df = day_df[(day_df["time"].dt.hour >= 8) & (day_df["time"].dt.hour <= 20)]
    if not day_df.empty:
        peak_row = day_df.loc[day_df["temp_f"].idxmax()]
        peak_hour = peak_row["time"]
        trend_proj_high = obs_temp_f + max(0.0, (peak_hour - local_now).total_seconds() / 3600.0) * (heating_rate or 0.0)

a, b, c = st.columns(3)
a.metric("Heating rate since sunrise", "—" if heating_rate is None else f"{heating_rate:+.2f} °F/hr")
b.metric("Forecast peak hour", "—" if peak_hour is None else peak_hour.strftime("%I:%M %p"))
c.metric("Projected high (trend-based)", "—" if trend_proj_high is None else f"{trend_proj_high:.1f}°F")

st.divider()
st.subheader("Suggested Kalshi Bracket (auto ladder)")
ladder, chosen_mode = build_ladder(consensus, sigma, ladder_mode)
top = max(ladder, key=lambda x: x["Win %"])
risk_level, risk_icon, risk_msg = classify_risk(spread, sigma, top["Win %"])

st.caption(f"Ladder alignment used: **{chosen_mode}**")
if risk_level == "LOW RISK":
    st.success(f"{risk_icon} {risk_level}: {risk_msg}")
elif risk_level == "HIGH RISK":
    st.warning(f"{risk_icon} {risk_level}: {risk_msg}")
else:
    st.error(f"{risk_icon} {risk_level}: {risk_msg}")

st.success(f"Suggested bracket: **{top['Bracket']}** (model ≈ **{top['Win %']*100:.0f}%**)")

df_ladder = pd.DataFrame([{"Bracket": x["Bracket"], "Win %": f"{x['Win %']*100:.1f}%"} for x in sorted(ladder, key=lambda r: r["Win %"], reverse=True)])
st.dataframe(df_ladder, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Decision Window")
target = local_now.replace(hour=10, minute=30, second=0, microsecond=0)
grace_end = target + timedelta(minutes=grace_minutes)
st.caption(f"Local time now: **{local_now.strftime('%a %b %d, %I:%M %p')}** | Target check: **10:30 AM** | Grace: **{grace_minutes} min**")
if local_now < target:
    st.warning("Early. Target check is 10:30 AM local.")
elif local_now <= grace_end:
    st.info("Inside preferred window (or within grace).")
else:
    st.warning("Past preferred window. Market is often sharper later in the day.")

if show_hourly_chart and chart_df is not None and not chart_df.empty:
    st.divider()
    st.subheader("Hourly temperature curve (today)")
    df_plot = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    if not df_plot.empty:
        st.line_chart(df_plot.set_index("time")["temp_f"])
        peak_t = df_plot.loc[df_plot["temp_f"].idxmax(), "time"]
        peak_v = float(df_plot["temp_f"].max())
        st.caption(f"Peak hour (forecast): {peak_t.strftime('%I:%M %p')} at {peak_v:.1f}°F")

st.divider()
st.caption("Fresh rebuild v7: no pasted-contract workflow, corrected ladder logic, LOW/HIGH/PASS restored.")
