import math
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Kalshi Temperature Model v10", layout="wide")
st.title("Kalshi Temperature Model v10")
st.caption(
    "Stable full version: automatic Open-Meteo + NWS pulls, city-specific filters, "
    "Kalshi ladder parsing, bracket probabilities, and BET / PASS output."
)

# -----------------------------
# City profiles
# -----------------------------
CITIES = {
    "Phoenix": {
        "lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix",
        "station": "KPHX", "station_label": "Phoenix Sky Harbor",
        "sigma": 0.82, "bias": 0.0, "prob_filter": 0.55,
    },
    "Las Vegas": {
        "lat": 36.0801, "lon": -115.1522, "tz": "America/Los_Angeles",
        "station": "KLAS", "station_label": "Harry Reid Intl",
        "sigma": 0.85, "bias": 0.0, "prob_filter": 0.55,
    },
    "Los Angeles": {
        "lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles",
        "station": "KLAX", "station_label": "Los Angeles Intl",
        "sigma": 0.90, "bias": -0.5, "prob_filter": 0.58,
    },
    "Dallas": {
        "lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago",
        "station": "KDFW", "station_label": "Dallas/Fort Worth Intl",
        "sigma": 0.96, "bias": 0.0, "prob_filter": 0.58,
    },
    "Austin": {
        "lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago",
        "station": "KAUS", "station_label": "Austin Bergstrom",
        "sigma": 1.08, "bias": 0.0, "prob_filter": 0.58,
    },
    "Houston": {
        "lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago",
        "station": "KIAH", "station_label": "George Bush Intercontinental",
        "sigma": 1.18, "bias": 0.0, "prob_filter": 0.62,
    },
    "Miami": {
        "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York",
        "station": "KMIA", "station_label": "Miami Intl",
        "sigma": 1.15, "bias": 0.0, "prob_filter": 0.58,
    },
    "New York": {
        "lat": 40.6413, "lon": -73.7781, "tz": "America/New_York",
        "station": "KJFK", "station_label": "John F. Kennedy Intl",
        "sigma": 1.00, "bias": 0.0, "prob_filter": 0.58,
    },
    "Atlanta": {
        "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York",
        "station": "KATL", "station_label": "Hartsfield-Jackson Atlanta Intl",
        "sigma": 0.98, "bias": 0.0, "prob_filter": 0.58,
    },
    "San Antonio": {
        "lat": 29.5337, "lon": -98.4698, "tz": "America/Chicago",
        "station": "KSAT", "station_label": "San Antonio Intl",
        "sigma": 1.05, "bias": 0.0, "prob_filter": 0.58,
    },
    "New Orleans": {
        "lat": 29.9934, "lon": -90.2580, "tz": "America/Chicago",
        "station": "KMSY", "station_label": "Louis Armstrong New Orleans Intl",
        "sigma": 1.16, "bias": 0.0, "prob_filter": 0.58,
    },
}

MIN_TOP_TWO_GAP = 0.12
STRONG_EDGE = 0.10
SMALL_EDGE = 0.03
MOMENTUM_WEIGHT = 0.35
NO_BET_LAG = 1.5

# -----------------------------
# Helpers
# -----------------------------
session = requests.Session()
session.headers.update({"User-Agent": "kalshi-weather-model-v10"})

def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def safe_get_json(url: str, params=None):
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def parse_label_numbers(label: str):
    nums = [int(x) for x in re.findall(r"\d+", label)]
    if "below" in label.lower():
        return ("below", nums[0] if nums else None)
    if "above" in label.lower():
        return ("above", nums[0] if nums else None)
    if len(nums) >= 2:
        return ("range", nums[0], nums[1])
    return None

def default_ladder(mu: float, mode: str = "auto"):
    center = round(mu)
    if mode == "auto":
        mode = "even" if center % 2 == 0 else "odd"
    if mode == "even":
        low_tail = f"{center-5}Â° or below"
        bins = [
            low_tail,
            f"{center-4}Â° to {center-3}Â°",
            f"{center-2}Â° to {center-1}Â°",
            f"{center}Â° to {center+1}Â°",
            f"{center+2}Â° to {center+3}Â°",
            f"{center+4}Â° or above",
        ]
    else:
        low_tail = f"{center-4}Â° or below"
        bins = [
            low_tail,
            f"{center-3}Â° to {center-2}Â°",
            f"{center-1}Â° to {center}Â°",
            f"{center+1}Â° to {center+2}Â°",
            f"{center+3}Â° to {center+4}Â°",
            f"{center+5}Â° or above",
        ]
    return bins, mode

def detect_market_ladder(labels):
    first_starts = []
    for lab in labels:
        nums = [int(x) for x in re.findall(r"\d+", lab)]
        if nums:
            first_starts.append(nums[0])
    if not first_starts:
        return None
    odd = sum(n % 2 == 1 for n in first_starts)
    even = sum(n % 2 == 0 for n in first_starts)
    return "odd" if odd > even else "even"

def parse_market_lines(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {}
    for ln in lines:
        m = re.match(r"(.+?)\s+([+-]?\d+(?:\.\d+)?c?)$", ln)
        if not m:
            continue
        label = m.group(1).strip()
        price = m.group(2).strip().lower()
        out[label] = price
    return out

def price_to_prob(price: str):
    price = price.strip().lower()
    if price.endswith("c"):
        cents = float(price[:-1])
        return cents / 100.0
    val = float(price)
    if val > 0:
        return 100.0 / (val + 100.0)
    return abs(val) / (abs(val) + 100.0)

def label_prob(label: str, mu: float, sigma: float):
    parsed = parse_label_numbers(label)
    if parsed is None:
        return 0.0
    kind = parsed[0]
    if kind == "below":
        upper = parsed[1] + 0.5
        return normal_cdf(upper, mu, sigma)
    if kind == "above":
        lower = parsed[1] - 0.5
        return 1.0 - normal_cdf(lower, mu, sigma)
    _, lo, hi = parsed
    return max(0.0, normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma))

def build_probability_table(labels, mu: float, sigma: float):
    rows = []
    for lab in labels:
        p = label_prob(lab, mu, sigma)
        rows.append({"Bracket": lab, "WinProb": p, "Fair YES": p})
    rows.sort(key=lambda x: x["WinProb"], reverse=True)
    return rows

def fetch_open_meteo(lat, lon, tz, model=None):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "timezone": tz,
        "forecast_days": 1,
    }
    if model:
        params["models"] = model
    data, err = safe_get_json("https://api.open-meteo.com/v1/forecast", params=params)
    if err or not data:
        return None, err or "fetch failed"
    try:
        c = data["daily"]["temperature_2m_max"][0]
        return c_to_f(float(c)), None
    except Exception as e:
        return None, str(e)

def fetch_nws_hourly(lat, lon):
    points, err = safe_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if err or not points:
        return None, None, err or "points fetch failed"
    hourly_url = points["properties"].get("forecastHourly")
    hourly, err2 = safe_get_json(hourly_url)
    if err2 or not hourly:
        return None, None, err2 or "hourly fetch failed"
    try:
        periods = hourly["properties"]["periods"]
        vals = []
        for p in periods[:24]:
            temp_f = p.get("temperature")
            if temp_f is not None:
                vals.append(float(temp_f))
        return (max(vals) if vals else None), periods, None
    except Exception as e:
        return None, None, str(e)

def fetch_latest_observation(station: str):
    data, err = safe_get_json(f"https://api.weather.gov/stations/{station}/observations/latest")
    if err or not data:
        return None, None, err or "obs fetch failed"
    try:
        c = data["properties"]["temperature"]["value"]
        obs_time = data["properties"]["timestamp"]
        if c is None:
            return None, obs_time, "obs unavailable"
        return c_to_f(float(c)), obs_time, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------
# UI
# -----------------------------
city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
tz = ZoneInfo(profile["tz"])
local_now = datetime.now(tz)

colA, colB = st.columns([1, 1])

with colA:
    include_open_default = st.checkbox("Use Open-Meteo default", value=True)
    include_open_gfs = st.checkbox("Use Open-Meteo GFS", value=True)
    include_nws = st.checkbox("Use NWS hourly", value=True)

with colB:
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["auto", "even", "odd", "market_auto"])
    no_bet_after_hour = st.slider("No new bets after local hour", 9, 15, 12)
    no_bet_after_minute = st.slider("No new bets after minute", 0, 59, 35, step=5)

st.markdown(
    f"""
**Locked settings for {city}**
- Probability filter: **{profile['prob_filter']:.2f}**
- Minimum top-two gap: **{MIN_TOP_TWO_GAP*100:.0f}%**
- Momentum weight: **{MOMENTUM_WEIGHT:.2f}**
- No-bet lag threshold: **{NO_BET_LAG:.1f}Â°F**
- City sigma: **{profile['sigma']:.2f}**
- Settlement station: **{profile['station']} â {profile['station_label']}**
"""
)

# Mobile-friendly Kalshi entry
with st.expander("Kalshi Odds / EV (optional)", expanded=False):
    market_text = st.text_area(
        "Paste Kalshi lines with YES odds or cents",
        height=120,
        placeholder="79-80 +100\n78 or below +177\n81-82 +143",
    )
    use_quick = st.toggle("Use quick manual Kalshi entry", value=False)
    quick_lines = []
    if use_quick:
        for i in range(6):
            c1, c2 = st.columns([3, 2])
            with c1:
                b = st.text_input(f"Bracket {i+1}", key=f"b{i}")
            with c2:
                p = st.text_input(f"YES odds/cents {i+1}", key=f"p{i}")
            if b.strip() and p.strip():
                quick_lines.append(f"{b.strip()} {p.strip()}")
    if quick_lines:
        market_text = "\n".join(quick_lines)
        st.code(market_text, language="text")

# -----------------------------
# Pull weather sources
# -----------------------------
source_rows = []

if include_open_default:
    val, err = fetch_open_meteo(profile["lat"], profile["lon"], profile["tz"], model=None)
    source_rows.append({"Source": "Open-Meteo (best match)", "Today High": val, "Note": "OK" if err is None else err})

if include_open_gfs:
    val, err = fetch_open_meteo(profile["lat"], profile["lon"], profile["tz"], model="gfs_seamless")
    source_rows.append({"Source": "Open-Meteo GFS", "Today High": val, "Note": "OK" if err is None else err})

nws_high = None
hourly_periods = None
if include_nws:
    nws_high, hourly_periods, err = fetch_nws_hourly(profile["lat"], profile["lon"])
    source_rows.append({"Source": "NWS (forecastHourly)", "Today High": nws_high, "Note": "OK" if err is None else err})

obs_temp, obs_time, obs_err = fetch_latest_observation(profile["station"])

source_df = pd.DataFrame(source_rows)
show_df = source_df.copy()
show_df["Today High"] = show_df["Today High"].apply(lambda x: f"{x:.1f}Â°F" if isinstance(x, (int, float)) else "â")
st.subheader(f"{city} â Todayâs High Forecasts (Â°F)")
st.dataframe(show_df, use_container_width=True, hide_index=True)

usable = [r["Today High"] for r in source_rows if isinstance(r["Today High"], (int, float))]
if not usable:
    st.error("No forecast sources are available right now.")
    st.stop()

consensus = sum(usable) / len(usable) + float(profile["bias"])
spread = (max(usable) - min(usable)) if len(usable) >= 2 else 0.0
sigma = max(0.75, float(profile["sigma"]) + spread / 2.5)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Consensus high", f"{consensus:.1f}Â°F")
c2.metric("Cross-source spread", f"{spread:.1f}Â°F")
c3.metric("Model uncertainty (Ï)", f"{sigma:.2f}Â°F")
c4.metric(f"Current settlement temp ({profile['station']})", f"{obs_temp:.1f}Â°F" if obs_temp is not None else "â")

if obs_time:
    st.caption(f"Obs time: {obs_time}")

# -----------------------------
# Live trend / nowcast
# -----------------------------
momentum_delta = None
expected_now = None
if hourly_periods and obs_temp is not None:
    try:
        now_local = local_now
        best = None
        best_dt = None
        for p in hourly_periods[:18]:
            start = datetime.fromisoformat(p["startTime"])
            if start.tzinfo is None:
                continue
            local_start = start.astimezone(tz)
            delta = abs((local_start - now_local).total_seconds())
            if best_dt is None or delta < best_dt:
                best_dt = delta
                best = p
        if best:
            expected_now = float(best["temperature"])
            momentum_delta = obs_temp - expected_now
    except Exception:
        momentum_delta = None

st.subheader("Live trend / nowcast")
cc1, cc2 = st.columns(2)
cc1.metric("Expected now (from NWS hourly)", f"{expected_now:.1f}Â°F" if expected_now is not None else "â")
cc2.metric("Live momentum vs forecast", f"{momentum_delta:+.1f}Â°F" if momentum_delta is not None else "â")

mu = consensus
if momentum_delta is not None:
    adjust = MOMENTUM_WEIGHT * momentum_delta
    mu += adjust
    st.caption(f"Momentum-adjusted consensus = {consensus:.1f}Â°F + {adjust:+.1f} = **{mu:.1f}Â°F**")

# -----------------------------
# Ladder and probabilities
# -----------------------------
market_prices = parse_market_lines(market_text) if market_text.strip() else {}
market_labels = list(market_prices.keys())

effective_ladder_mode = ladder_mode
if ladder_mode == "market_auto" and market_labels:
    detected = detect_market_ladder(market_labels)
    effective_ladder_mode = detected if detected else "auto"

labels, chosen_mode = default_ladder(mu, "auto" if effective_ladder_mode == "market_auto" else effective_ladder_mode)
if market_labels:
    labels = market_labels

rows = build_probability_table(labels, mu, sigma)
top = rows[0]
second = rows[1] if len(rows) > 1 else {"WinProb": 0.0}
top_gap = top["WinProb"] - second["WinProb"]

st.subheader("Suggested Kalshi Bracket")
st.caption(f"Ladder alignment used: **{chosen_mode}**")
if market_labels:
    st.caption("Using brackets from pasted / manual Kalshi ladder.")

# Decision
reasons = []
cutoff = local_now.replace(hour=no_bet_after_hour, minute=no_bet_after_minute, second=0, microsecond=0)
if top["WinProb"] < float(profile["prob_filter"]):
    reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {profile['prob_filter']*100:.0f}%)")
if top_gap < MIN_TOP_TWO_GAP:
    reasons.append(f"Top-two gap too small ({top_gap*100:.1f}% < {MIN_TOP_TWO_GAP*100:.0f}%)")
if local_now > cutoff:
    reasons.append(f"Past cutoff ({cutoff.strftime('%I:%M %p')} local)")
if momentum_delta is not None and momentum_delta <= -NO_BET_LAG and local_now.hour >= 11:
    reasons.append(f"Live temp is {abs(momentum_delta):.1f}Â°F behind forecast track")

if reasons:
    st.error("TRADE FILTER: DO NOT BET â " + " | ".join(reasons))
else:
    st.success("TRADE FILTER: BET ALLOWED â confidence passed your rules.")

st.success(f"Suggested bracket: **{top['Bracket']}** (model â {top['WinProb']*100:.0f}%)")
st.metric("Top-two bracket gap", f"{top_gap*100:.1f}%")

# Table
table_rows = []
for r in rows:
    row = {
        "Bracket": r["Bracket"],
        "Win %": f"{r['WinProb']*100:.1f}%",
        "Fair YES": f"{r['Fair YES']*100:.1f}Â¢",
    }
    if r["Bracket"] in market_prices:
        mprob = price_to_prob(market_prices[r["Bracket"]])
        edge = r["WinProb"] - mprob
        row["Market YES"] = market_prices[r["Bracket"]]
        row["Edge"] = f"{edge*100:+.1f}%"
    table_rows.append(row)

st.subheader("Kalshi Edge Table")
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
