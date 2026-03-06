# streamlit_app.py
# Kalshi Weather Model – Daily High [v7.2]
# Adds:
# - Best-effort Kalshi URL import
# - EV / edge calculator + bet signal
# - Safer Open-Meteo midnight timezone handling
# - Keeps ladder alignment, risk warning, win %, live airport temp, nowcast

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Weather Model – Daily High", layout="centered")
st.title("Kalshi Weather Model – Daily High")
st.caption(
    "v7.2: ladder alignment, risk warning, win %, live airport temp, nowcast, plus Kalshi odds import "
    "(best-effort) and EV / bet signal."
)

UA = {"User-Agent": "kalshi-weather-model/7.2"}

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

def safe_get_text(url: str, params: Optional[dict] = None, timeout: int = 12):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
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

def american_odds_to_prob(odds: float):
    if odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)

def prob_to_fair_cents(p: float) -> float:
    return max(0.0, min(100.0, p * 100.0))

def expected_value_per_1_dollar(p: float, yes_price_cents: float) -> float:
    price = yes_price_cents / 100.0
    return p * (1.0 - price) - (1.0 - p) * price

def yes_price_from_american_odds(odds: float) -> Optional[float]:
    p = american_odds_to_prob(odds)
    return None if p is None else p * 100.0

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
        df["time"] = df["time"].dt.tz_localize(ZoneInfo(tz), ambiguous="NaT", nonexistent="shift_forward")
        df = df.dropna(subset=["time"]).reset_index(drop=True)
        tmax = float(js["daily"]["temperature_2m_max"][0])
        sunrise = pd.to_datetime(js["daily"]["sunrise"][0]).tz_localize(ZoneInfo(tz), ambiguous="NaT", nonexistent="shift_forward")
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
    df["time"] = pd.to_datetime(df["time"], utc=True)
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
            rows.append({"Bracket": label, "WinProb": p})
        s = sum(x["WinProb"] for x in rows)
        for x in rows:
            x["WinProb"] = x["WinProb"] / s if s > 0 else 0.0
        return rows

    ladders = {"even": one(even_start), "odd": one(odd_start)}
    if mode in ("even", "odd"):
        chosen = mode
    else:
        chosen = max(ladders.keys(), key=lambda k: max(x["WinProb"] for x in ladders[k]))
    return ladders[chosen], chosen

def classify_risk(spread: float, sigma: float, top_prob: float):
    if spread >= 4.0 or sigma >= 3.5 or top_prob < 0.20:
        return "PASS", "🔴", "Uncertainty is too high (spread/σ) or the top bracket is not strong enough."
    if spread >= 2.5 or sigma >= 3.0 or top_prob < 0.25:
        return "HIGH RISK", "🟡", "Playable only with a clear edge vs market odds; otherwise skip."
    return "LOW RISK", "🟢", "Models are relatively aligned and the top bracket is reasonably strong."

def parse_market_lines(text: str):
    out = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        odds_match = re.search(r"([+-]\d{2,5})", line)
        cents_match = re.search(r"(\d{1,3})\s*(?:c|¢)\b", line, flags=re.IGNORECASE)
        label = line
        price_cents = None
        if odds_match:
            odds = float(odds_match.group(1))
            price_cents = yes_price_from_american_odds(odds)
            label = line[:odds_match.start()].strip()
        elif cents_match:
            price_cents = float(cents_match.group(1))
            label = line[:cents_match.start()].strip()
        label = label.replace("º", "°").strip()
        if label and price_cents is not None:
            out[label] = max(0.0, min(100.0, price_cents))
    return out

def best_effort_import_from_url(url: str):
    if not url.strip():
        return ""
    html = safe_get_text(url.strip(), timeout=10)
    if not html:
        return ""
    labels = re.findall(r'(\d{1,3}°\s*(?:to)\s*\d{1,3}°|\d{1,3}°\s*or\s*(?:below|above))', html, flags=re.IGNORECASE)
    odds = re.findall(r'([+-]\d{2,5})', html)
    labels = [re.sub(r"\s+", " ", x).replace("º", "°").strip() for x in labels]
    if not labels or not odds:
        return ""
    lines = []
    for i in range(min(len(labels), len(odds))):
        lines.append(f"{labels[i]} {odds[i]}")
    return "\n".join(lines)

# ---------- UI ----------
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

with st.expander("Kalshi Odds / EV (optional)", expanded=True):
    import_mode = st.selectbox("Odds input method", ["None", "Paste lines", "Best-effort URL import"], index=0)
    min_edge_pct = st.slider("Minimum edge to flag", 0.0, 25.0, 5.0, 0.5)
    market_url = ""
    market_text = ""
    if import_mode == "Paste lines":
        market_text = st.text_area(
            "Paste Kalshi lines with YES odds or cents",
            height=140,
            placeholder="81° to 82° +156\n83° to 84° 42c\n80° or below 9c",
        )
    elif import_mode == "Best-effort URL import":
        market_url = st.text_input("Kalshi market URL", value="", placeholder="Paste Kalshi market URL")

# ---------- fetch ----------
sources = []
chart_df = None
sunrise_local = None

om_df, om_high, om_sunrise, om_err = fetch_open_meteo(lat, lon, tz)
sources.append(("Open-Meteo", om_high, om_err))
if om_high is not None:
    chart_df = om_df
    sunrise_local = om_sunrise

if include_gfs:
    _, gfs_high, _, gfs_err = fetch_open_meteo(lat, lon, tz, model="gfs")
    sources.append(("Open-Meteo (GFS)", gfs_high, gfs_err))

if include_nws:
    _, nws_high, nws_err = fetch_nws_hourly(lat, lon)
    sources.append(("NWS (forecastHourly)", nws_high, nws_err))

if include_hrrr:
    _, hrrr_high, _, hrrr_err = fetch_open_meteo(lat, lon, tz, model="hrrr")
    sources.append(("HRRR (best-effort)", hrrr_high, hrrr_err))

obs_temp_f, obs_time, obs_err = fetch_station_obs(station)

vals = [x[1] for x in sources if x[1] is not None]
if not vals:
    st.error("No forecast sources returned successfully.")
    st.stop()

consensus = float(sum(vals) / len(vals))
spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
sigma = float(max(1.2, 0.9 + 0.55 * spread))

# ---------- table ----------
st.subheader(f"{city} – Today’s High Forecasts (°F)")
df_sources = pd.DataFrame([{
    "Source": s[0],
    "Today High": ("—" if s[1] is None else f"{s[1]:.1f}°F"),
    "Note": s[2],
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

# ---------- nowcast ----------
st.divider()
st.subheader("Live trend / nowcast")
heating_rate = None
peak_hour = None
trend_proj_high = None

if obs_temp_f is not None and chart_df is not None and not chart_df.empty:
    sunrise_for_calc = sunrise_local if sunrise_local is not None else local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    idx = (chart_df["time"] - sunrise_for_calc).abs().idxmin()
    sunrise_temp = float(chart_df.loc[idx, "temp_f"])
    hrs = max(0.25, (local_now - sunrise_for_calc).total_seconds() / 3600.0)

    if local_now > sunrise_for_calc:
        heating_rate = (obs_temp_f - sunrise_temp) / hrs

    day_df = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    day_df = day_df[(day_df["time"].dt.hour >= 8) & (day_df["time"].dt.hour <= 20)]
    if not day_df.empty:
        peak_row = day_df.loc[day_df["temp_f"].idxmax()]
        peak_hour = peak_row["time"]
        if heating_rate is not None:
            trend_proj_high = obs_temp_f + max(0.0, (peak_hour - local_now).total_seconds() / 3600.0) * heating_rate

a, b, c = st.columns(3)
a.metric("Heating rate since sunrise", "—" if heating_rate is None else f"{heating_rate:+.2f} °F/hr")
b.metric("Forecast peak hour", "—" if peak_hour is None else peak_hour.strftime("%I:%M %p"))
c.metric("Projected high (trend-based)", "—" if trend_proj_high is None else f"{trend_proj_high:.1f}°F")

# ---------- ladder ----------
st.divider()
st.subheader("Suggested Kalshi Bracket")
ladder, chosen_mode = build_ladder(consensus, sigma, ladder_mode)
top = max(ladder, key=lambda x: x["WinProb"])
risk_level, risk_icon, risk_msg = classify_risk(spread, sigma, top["WinProb"])

st.caption(f"Ladder alignment used: **{chosen_mode}**")
if risk_level == "LOW RISK":
    st.success(f"{risk_icon} {risk_level}: {risk_msg}")
elif risk_level == "HIGH RISK":
    st.warning(f"{risk_icon} {risk_level}: {risk_msg}")
else:
    st.error(f"{risk_icon} {risk_level}: {risk_msg}")

st.success(f"Suggested bracket: **{top['Bracket']}** (model ≈ **{top['WinProb']*100:.0f}%**)")

market_prices = {}
if import_mode == "Paste lines" and market_text.strip():
    market_prices = parse_market_lines(market_text)
elif import_mode == "Best-effort URL import" and market_url.strip():
    imported = best_effort_import_from_url(market_url)
    if imported:
        st.caption("Best-effort URL import found these lines:")
        st.code(imported)
        market_prices = parse_market_lines(imported)
    else:
        st.warning("URL import failed. Kalshi may block access or require login. Paste lines is more reliable.")

rows = []
best_bet = None
for x in sorted(ladder, key=lambda r: r["WinProb"], reverse=True):
    row = {
        "Bracket": x["Bracket"],
        "Win %": f"{x['WinProb']*100:.1f}%",
        "Fair YES": f"{prob_to_fair_cents(x['WinProb']):.1f}¢",
    }
    if x["Bracket"] in market_prices:
        yes_cents = market_prices[x["Bracket"]]
        market_prob = yes_cents / 100.0
        edge = (x["WinProb"] - market_prob) * 100.0
        ev = expected_value_per_1_dollar(x["WinProb"], yes_cents)
        row["Market YES"] = f"{yes_cents:.1f}¢"
        row["Edge %"] = f"{edge:+.1f}%"
        row["EV / $1"] = f"{ev:+.3f}"
        if edge >= min_edge_pct and ev > 0:
            if best_bet is None or edge > best_bet["edge"]:
                best_bet = {"bracket": x["Bracket"], "edge": edge, "ev": ev, "price": yes_cents}
    rows.append(row)

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if market_prices:
    st.subheader("Bet Signal")
    if best_bet is not None:
        st.success(
            f"💰 BET SIGNAL: **{best_bet['bracket']}** | "
            f"Market YES **{best_bet['price']:.1f}¢** | "
            f"Edge **{best_bet['edge']:+.1f}%** | EV per $1 **{best_bet['ev']:+.3f}**"
        )
    else:
        st.warning("No bracket currently clears your minimum edge and positive EV filter.")

# ---------- decision window ----------
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

# ---------- hourly ----------
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
st.caption(
    "Night behavior is normal: Open-Meteo/GFS/HRRR may fail or be delayed around midnight. "
    "Best-effort Kalshi URL import may fail if Kalshi blocks access; paste lines is more reliable."
)
