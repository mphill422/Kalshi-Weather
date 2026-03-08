
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v10.3", layout="wide")
st.title("Kalshi Temperature Model v10.3")
st.caption(
    "Clean ASCII build: NWS forecast + NWS hourly + station temp, automatic ladder detection, "
    "city filters, and BET/PASS output."
)

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "station": "KPHX", "station_label": "Phoenix Sky Harbor", "sigma": 0.82, "bias": 0.0, "prob_filter": 0.55},
    "Las Vegas": {"lat": 36.0801, "lon": -115.1522, "tz": "America/Los_Angeles", "station": "KLAS", "station_label": "Harry Reid Intl", "sigma": 0.85, "bias": 0.0, "prob_filter": 0.55},
    "Los Angeles": {"lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles", "station": "KLAX", "station_label": "Los Angeles Intl", "sigma": 0.90, "bias": -0.5, "prob_filter": 0.58},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "station": "KDFW", "station_label": "Dallas/Fort Worth Intl", "sigma": 0.96, "bias": 0.0, "prob_filter": 0.58},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "station": "KAUS", "station_label": "Austin Bergstrom", "sigma": 1.08, "bias": 0.0, "prob_filter": 0.58},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "station": "KIAH", "station_label": "George Bush Intercontinental", "sigma": 1.18, "bias": 0.0, "prob_filter": 0.62},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "station": "KMIA", "station_label": "Miami Intl", "sigma": 1.15, "bias": 0.0, "prob_filter": 0.58},
    "New York": {"lat": 40.6413, "lon": -73.7781, "tz": "America/New_York", "station": "KJFK", "station_label": "John F. Kennedy Intl", "sigma": 1.00, "bias": 0.0, "prob_filter": 0.58},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "station": "KATL", "station_label": "Hartsfield-Jackson Atlanta Intl", "sigma": 0.98, "bias": 0.0, "prob_filter": 0.58},
    "San Antonio": {"lat": 29.5337, "lon": -98.4698, "tz": "America/Chicago", "station": "KSAT", "station_label": "San Antonio Intl", "sigma": 1.05, "bias": 0.0, "prob_filter": 0.58},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "tz": "America/Chicago", "station": "KMSY", "station_label": "Louis Armstrong New Orleans Intl", "sigma": 1.16, "bias": 0.0, "prob_filter": 0.58},
}

MIN_TOP_TWO_GAP = 0.12
MOMENTUM_WEIGHT = 0.35
NO_BET_LAG = 1.5

session = requests.Session()
session.headers.update({"User-Agent": "kalshi-temp-model-v10-3"})

def safe_get_json(url: str, params=None):
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def parse_label_numbers(label: str):
    nums = [int(x) for x in re.findall(r"\d+", label)]
    low = label.lower()
    if "below" in low:
        return ("below", nums[0] if nums else None)
    if "above" in low:
        return ("above", nums[0] if nums else None)
    if len(nums) >= 2:
        return ("range", nums[0], nums[1])
    return None

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

def default_ladder(mu: float, mode: str = "auto"):
    center = round(mu)
    if mode == "auto":
        mode = "even" if center % 2 == 0 else "odd"
    if mode == "even":
        labels = [
            f"{center-5} or below",
            f"{center-4} to {center-3}",
            f"{center-2} to {center-1}",
            f"{center} to {center+1}",
            f"{center+2} to {center+3}",
            f"{center+4} or above",
        ]
    else:
        labels = [
            f"{center-4} or below",
            f"{center-3} to {center-2}",
            f"{center-1} to {center}",
            f"{center+1} to {center+2}",
            f"{center+3} to {center+4}",
            f"{center+5} or above",
        ]
    return labels, mode

def detect_market_ladder(labels):
    starts = []
    for lab in labels:
        nums = [int(x) for x in re.findall(r"\d+", lab)]
        if nums:
            starts.append(nums[0])
    if not starts:
        return None
    odd = sum(x % 2 == 1 for x in starts)
    even = sum(x % 2 == 0 for x in starts)
    return "odd" if odd > even else "even"

def parse_market_lines(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {}
    for ln in lines:
        m = re.match(r"(.+?)\s+([+-]?\d+(?:\.\d+)?c?)$", ln)
        if not m:
            continue
        out[m.group(1).strip()] = m.group(2).strip().lower()
    return out

def price_to_prob(price: str):
    if price.endswith("c"):
        return float(price[:-1]) / 100.0
    val = float(price)
    if val > 0:
        return 100.0 / (val + 100.0)
    return abs(val) / (abs(val) + 100.0)

def fetch_nws_products(lat: float, lon: float, tzname: str):
    points, err = safe_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if err or not points:
        return None, None, None, err or "points fetch failed"

    forecast_url = points["properties"].get("forecast")
    hourly_url = points["properties"].get("forecastHourly")

    daily_high = None
    daily_note = "Unavailable"
    if forecast_url:
        forecast_data, ferr = safe_get_json(forecast_url)
        if ferr or not forecast_data:
            daily_note = ferr or "forecast fetch failed"
        else:
            try:
                periods = forecast_data["properties"]["periods"]
                tz = ZoneInfo(tzname)
                now_local = datetime.now(tz)
                best = None
                for p in periods:
                    start = datetime.fromisoformat(p["startTime"]).astimezone(tz)
                    end = datetime.fromisoformat(p["endTime"]).astimezone(tz)
                    is_day = bool(p.get("isDaytime"))
                    if is_day and start.date() <= now_local.date() <= end.date():
                        best = p
                        break
                if best is None:
                    for p in periods:
                        if bool(p.get("isDaytime")):
                            best = p
                            break
                if best:
                    daily_high = float(best["temperature"])
                    daily_note = "OK"
            except Exception as e:
                daily_note = str(e)

    hourly_high = None
    hourly_periods = None
    hourly_note = "Unavailable"
    if hourly_url:
        hourly_data, herr = safe_get_json(hourly_url)
        if herr or not hourly_data:
            hourly_note = herr or "hourly fetch failed"
        else:
            try:
                hourly_periods = hourly_data["properties"]["periods"]
                vals = []
                tz = ZoneInfo(tzname)
                now_local = datetime.now(tz)
                for p in hourly_periods[:24]:
                    start = datetime.fromisoformat(p["startTime"]).astimezone(tz)
                    if start.date() == now_local.date():
                        temp = p.get("temperature")
                        if temp is not None:
                            vals.append(float(temp))
                if vals:
                    hourly_high = max(vals)
                    hourly_note = "OK"
                else:
                    hourly_note = "No hourly values"
            except Exception as e:
                hourly_note = str(e)

    return daily_high, hourly_high, hourly_periods, {"forecast_note": daily_note, "hourly_note": hourly_note}

def fetch_latest_observation(station: str):
    data, err = safe_get_json(f"https://api.weather.gov/stations/{station}/observations/latest")
    if err or not data:
        return None, None, err or "obs fetch failed"
    try:
        c = data["properties"]["temperature"]["value"]
        ts = data["properties"]["timestamp"]
        if c is None:
            return None, ts, "obs unavailable"
        return c_to_f(float(c)), ts, None
    except Exception as e:
        return None, None, str(e)

city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
tz = ZoneInfo(profile["tz"])
local_now = datetime.now(tz)

left, right = st.columns(2)
with left:
    use_nws_forecast = st.checkbox("Use NWS forecast", value=True)
    use_nws_hourly = st.checkbox("Use NWS hourly", value=True)
with right:
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["auto", "market_auto", "even", "odd"], index=1)
    no_bet_after_hour = st.slider("No new bets after local hour", 9, 15, 12)
    no_bet_after_minute = st.slider("No new bets after minute", 0, 59, 35, step=5)

st.markdown(
    f"""
**Locked settings for {city}**
- Probability filter: **{profile['prob_filter']:.2f}**
- Minimum top-two gap: **{MIN_TOP_TWO_GAP*100:.0f}%**
- Momentum weight: **{MOMENTUM_WEIGHT:.2f}**
- No-bet lag threshold: **{NO_BET_LAG:.1f} deg F**
- City sigma: **{profile['sigma']:.2f}**
- Settlement station: **{profile['station']} - {profile['station_label']}**
"""
)

with st.expander("Kalshi Odds / EV (optional)", expanded=False):
    market_text = st.text_area("Paste Kalshi lines with YES odds or cents", height=120, placeholder="79-80 +100\n78 or below +177\n81-82 +143")
    use_quick = st.toggle("Use quick manual Kalshi entry", value=False)
    quick_lines = []
    if use_quick:
        for i in range(6):
            a, b = st.columns([3, 2])
            with a:
                bracket = st.text_input(f"Bracket {i+1}", key=f"br_{i}")
            with b:
                price = st.text_input(f"YES odds/cents {i+1}", key=f"pr_{i}")
            if bracket.strip() and price.strip():
                quick_lines.append(f"{bracket.strip()} {price.strip()}")
    if quick_lines:
        market_text = "\n".join(quick_lines)
        st.code(market_text, language="text")

daily_high, hourly_high, hourly_periods, notes = fetch_nws_products(profile["lat"], profile["lon"], profile["tz"])
obs_temp, obs_time, obs_err = fetch_latest_observation(profile["station"])

source_rows = []
if use_nws_forecast:
    source_rows.append({"Source": "NWS forecast", "Today High": daily_high, "Note": notes["forecast_note"]})
if use_nws_hourly:
    source_rows.append({"Source": "NWS hourly", "Today High": hourly_high, "Note": notes["hourly_note"]})

show_df = pd.DataFrame(source_rows)

def fmt_temp(x):
    try:
        if x is None:
            return "-"
        xf = float(x)
        if math.isnan(xf):
            return "-"
        return f"{xf:.1f} deg F"
    except Exception:
        return "-"

if show_df.empty:
    st.error("No forecast sources available.")
    st.stop()

show_df["Today High"] = show_df["Today High"].apply(fmt_temp)
st.subheader(f"{city} - Today's High Forecasts (deg F)")
st.dataframe(show_df, use_container_width=True, hide_index=True)

usable = []
for row in source_rows:
    val = row["Today High"]
    if isinstance(val, (int, float)) and not math.isnan(val):
        usable.append(val)

if not usable:
    st.error("No usable forecast values available.")
    st.stop()

consensus = sum(usable) / len(usable) + float(profile["bias"])
spread = max(usable) - min(usable) if len(usable) >= 2 else 0.0
sigma = max(0.75, float(profile["sigma"]) + spread / 2.5)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Consensus high", f"{consensus:.1f} deg F")
m2.metric("Cross-source spread", f"{spread:.1f} deg F")
m3.metric("Model uncertainty (sigma)", f"{sigma:.2f} deg F")
m4.metric(f"Current settlement temp ({profile['station']})", f"{obs_temp:.1f} deg F" if obs_temp is not None else "-")
if obs_time:
    st.caption(f"Obs time: {obs_time}")

expected_now = None
momentum_delta = None
if obs_temp is not None and hourly_periods:
    try:
        best = None
        best_delta = None
        for p in hourly_periods[:18]:
            dt = datetime.fromisoformat(p["startTime"]).astimezone(tz)
            delta = abs((dt - local_now).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = p
        if best and best.get("temperature") is not None:
            expected_now = float(best["temperature"])
            momentum_delta = obs_temp - expected_now
    except Exception:
        pass

st.subheader("Live trend / nowcast")
q1, q2 = st.columns(2)
q1.metric("Expected now (from NWS hourly)", f"{expected_now:.1f} deg F" if expected_now is not None else "-")
q2.metric("Live momentum vs forecast", f"{momentum_delta:+.1f} deg F" if momentum_delta is not None else "-")

mu = consensus
if momentum_delta is not None:
    adjust = MOMENTUM_WEIGHT * momentum_delta
    mu += adjust
    st.caption(f"Momentum-adjusted consensus = {consensus:.1f} deg F + {adjust:+.1f} = {mu:.1f} deg F")

market_prices = parse_market_lines(market_text) if market_text.strip() else {}
market_labels = list(market_prices.keys())
detected = detect_market_ladder(market_labels) if market_labels else None
effective_mode = detected if ladder_mode == "market_auto" and detected else ladder_mode
labels, chosen_mode = default_ladder(mu, "auto" if effective_mode == "market_auto" else effective_mode)
if market_labels:
    labels = market_labels

rows = []
for label in labels:
    p = label_prob(label, mu, sigma)
    rows.append({"Bracket": label, "WinProb": p, "Fair YES": p})
rows.sort(key=lambda x: x["WinProb"], reverse=True)

top = rows[0]
second = rows[1] if len(rows) > 1 else {"WinProb": 0.0}
top_gap = top["WinProb"] - second["WinProb"]

st.subheader("Suggested Kalshi Bracket")
if market_labels and detected:
    st.caption(f"Kalshi ladder auto-detected: {detected}")
elif market_labels:
    st.caption("Kalshi ladder entered manually.")
else:
    st.caption(f"Model ladder alignment used: {chosen_mode}")

cutoff = local_now.replace(hour=no_bet_after_hour, minute=no_bet_after_minute, second=0, microsecond=0)
reasons = []
if top["WinProb"] < float(profile["prob_filter"]):
    reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {profile['prob_filter']*100:.0f}%)")
if top_gap < MIN_TOP_TWO_GAP:
    reasons.append(f"Top-two gap too small ({top_gap*100:.1f}% < {MIN_TOP_TWO_GAP*100:.0f}%)")
if local_now > cutoff:
    reasons.append(f"Past cutoff ({cutoff.strftime('%I:%M %p')} local)")
if momentum_delta is not None and momentum_delta <= -NO_BET_LAG and local_now.hour >= 11:
    reasons.append(f"Live temp is {abs(momentum_delta):.1f} deg F behind forecast track")

if reasons:
    st.error("TRADE FILTER: DO NOT BET - " + " | ".join(reasons))
else:
    st.success("TRADE FILTER: BET ALLOWED - confidence passed your rules.")

st.success(f"Suggested bracket: {top['Bracket']} (model ~ {top['WinProb']*100:.0f}%)")
st.metric("Top-two bracket gap", f"{top_gap*100:.1f}%")

table_rows = []
for r in rows:
    item = {"Bracket": r["Bracket"], "Win %": f"{r['WinProb']*100:.1f}%", "Fair YES": f"{r['Fair YES']*100:.1f}c"}
    if r["Bracket"] in market_prices:
        mprob = price_to_prob(market_prices[r["Bracket"]])
        edge = r["WinProb"] - mprob
        item["Market YES"] = market_prices[r["Bracket"]]
        item["Edge"] = f"{edge*100:+.1f}%"
    table_rows.append(item)

st.subheader("Kalshi Edge Table")
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
