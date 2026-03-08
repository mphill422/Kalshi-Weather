import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v11.4", layout="wide")
st.title("Kalshi Temperature Model v11.4")
st.caption(
    "Expanded city list plus extra weather sources with safer fallbacks. "
    "If outside APIs rate-limit, the app falls back to NWS-only."
)

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "station": "KPHX", "sigma": 1.10, "bias": 0.60, "prob_filter": 0.55},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537, "tz": "America/Los_Angeles", "station": "KLAS", "sigma": 1.00, "bias": 0.00, "prob_filter": 0.55},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles", "station": "KLAX", "sigma": 1.10, "bias": -0.40, "prob_filter": 0.58},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "station": "KDFW", "sigma": 1.00, "bias": 0.15, "prob_filter": 0.58},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "station": "KAUS", "sigma": 1.10, "bias": 0.20, "prob_filter": 0.58},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "station": "KIAH", "sigma": 1.50, "bias": 0.30, "prob_filter": 0.62},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "tz": "America/Chicago", "station": "KMSY", "sigma": 1.40, "bias": 0.20, "prob_filter": 0.62},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "station": "KMIA", "sigma": 1.40, "bias": 0.10, "prob_filter": 0.62},
    "Washington DC": {"lat": 38.8521, "lon": -77.0377, "tz": "America/New_York", "station": "KDCA", "sigma": 1.15, "bias": 0.00, "prob_filter": 0.58},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "station": "KATL", "sigma": 1.15, "bias": 0.10, "prob_filter": 0.58},
}

MIN_TOP_TWO_GAP = 0.12
MOMENTUM_WEIGHT = 0.35
NO_BET_LAG = 1.5

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url, params=None):
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "kalshi-temp-v11-4"},
            timeout=12,
        )
        r.raise_for_status()
        return r.json(), "OK"
    except Exception as e:
        return None, str(e)

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def parse_label_numbers(label):
    nums = [int(x) for x in re.findall(r"\d+", label)]
    low = label.lower()
    if "below" in low:
        return ("below", nums[0] if nums else None)
    if "above" in low:
        return ("above", nums[0] if nums else None)
    if len(nums) >= 2:
        return ("range", nums[0], nums[1])
    return None

def label_prob(label, mu, sigma):
    parsed = parse_label_numbers(label)
    if parsed is None:
        return 0.0
    kind = parsed[0]
    if kind == "below":
        return normal_cdf(parsed[1] + 0.5, mu, sigma)
    if kind == "above":
        return 1.0 - normal_cdf(parsed[1] - 0.5, mu, sigma)
    _, lo, hi = parsed
    return max(0.0, normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma))

def default_ladder(mu, mode="auto"):
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
    return labels

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

def parse_market_lines(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = {}
    for ln in lines:
        m = re.match(r"(.+?)\s+([+-]?\d+(?:\.\d+)?c?)$", ln)
        if m:
            out[m.group(1).strip()] = m.group(2).strip().lower()
    return out

def price_to_prob(price):
    if price.endswith("c"):
        return float(price[:-1]) / 100.0
    val = float(price)
    if val > 0:
        return 100.0 / (val + 100.0)
    return abs(val) / (abs(val) + 100.0)

def fmt_num(x):
    try:
        if x is None:
            return "-"
        xf = float(x)
        if math.isnan(xf):
            return "-"
        return f"{xf:.1f}"
    except Exception:
        return "-"

def fetch_nws_all(lat, lon, tzname, station):
    daily_high = None
    hourly_high = None
    hourly_periods = None
    obs_temp = None
    obs_ts = None
    notes = {"NWS forecast": "FAILED", "NWS hourly": "FAILED", "Station obs": "FAILED"}

    points, status = safe_get_json(f"https://api.weather.gov/points/{lat},{lon}")
    if not points:
        return daily_high, hourly_high, hourly_periods, obs_temp, obs_ts, {
            "NWS forecast": status, "NWS hourly": status, "Station obs": "FAILED"
        }

    forecast_url = points["properties"].get("forecast")
    hourly_url = points["properties"].get("forecastHourly")

    if forecast_url:
        data, s = safe_get_json(forecast_url)
        if data:
            try:
                periods = data["properties"]["periods"]
                tz = ZoneInfo(tzname)
                now_local = datetime.now(tz)
                best = None
                for p in periods:
                    start = datetime.fromisoformat(p["startTime"]).astimezone(tz)
                    end = datetime.fromisoformat(p["endTime"]).astimezone(tz)
                    if bool(p.get("isDaytime")) and start.date() <= now_local.date() <= end.date():
                        best = p
                        break
                if best is None:
                    for p in periods:
                        if bool(p.get("isDaytime")):
                            best = p
                            break
                if best:
                    daily_high = float(best["temperature"])
                    notes["NWS forecast"] = "OK"
            except Exception as e:
                notes["NWS forecast"] = str(e)
        else:
            notes["NWS forecast"] = s

    if hourly_url:
        data, s = safe_get_json(hourly_url)
        if data:
            try:
                hourly_periods = data["properties"]["periods"]
                vals = []
                tz = ZoneInfo(tzname)
                now_local = datetime.now(tz)
                for p in hourly_periods[:24]:
                    dt = datetime.fromisoformat(p["startTime"]).astimezone(tz)
                    if dt.date() == now_local.date():
                        temp = p.get("temperature")
                        if temp is not None:
                            vals.append(float(temp))
                if vals:
                    hourly_high = max(vals)
                    notes["NWS hourly"] = "OK"
                else:
                    notes["NWS hourly"] = "No values"
            except Exception as e:
                notes["NWS hourly"] = str(e)
        else:
            notes["NWS hourly"] = s

    obs, s = safe_get_json(f"https://api.weather.gov/stations/{station}/observations/latest")
    if obs:
        try:
            c = obs["properties"]["temperature"]["value"]
            obs_ts = obs["properties"]["timestamp"]
            if c is not None:
                obs_temp = float(c) * 9.0 / 5.0 + 32.0
                notes["Station obs"] = "OK"
            else:
                notes["Station obs"] = "Unavailable"
        except Exception as e:
            notes["Station obs"] = str(e)
    else:
        notes["Station obs"] = s

    return daily_high, hourly_high, hourly_periods, obs_temp, obs_ts, notes

def fetch_open_meteo(lat, lon, model=None):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "hourly": "temperature_2m",
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "forecast_days": 1,
    }
    if model:
        params["models"] = model
    data, status = safe_get_json("https://api.open-meteo.com/v1/forecast", params=params)
    if not data:
        return None, None, None, status
    try:
        daily_high = float(data["daily"]["temperature_2m_max"][0])
    except Exception:
        daily_high = None
    try:
        current_temp = float(data["current"]["temperature_2m"])
    except Exception:
        current_temp = None
    try:
        hourly = data["hourly"]["temperature_2m"]
        hourly_high = max(hourly[:24]) if hourly else None
    except Exception:
        hourly_high = None
    return daily_high, hourly_high, current_temp, "OK"

def fetch_metno(lat, lon):
    params = {"lat": lat, "lon": lon}
    try:
        r = requests.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params=params,
            headers={"User-Agent": "kalshi-temp-v11-4"},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, None, None, str(e)

    try:
        ts = data["properties"]["timeseries"][:24]
        vals = []
        current_temp = None
        for i, item in enumerate(ts):
            det = item["data"]["instant"]["details"]
            t = det.get("air_temperature")
            if t is not None:
                tf = float(t) * 9.0 / 5.0 + 32.0
                vals.append(tf)
                if i == 0:
                    current_temp = tf
        daily_high = max(vals) if vals else None
        hourly_high = daily_high
        return daily_high, hourly_high, current_temp, "OK"
    except Exception as e:
        return None, None, None, str(e)

city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
local_now = datetime.now(ZoneInfo(profile["tz"]))

c1, c2 = st.columns(2)
with c1:
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["market_auto", "auto", "even", "odd"], index=0)
with c2:
    no_bet_after_hour = st.slider("No new bets after local hour", 9, 18, 12)
    no_bet_after_minute = st.slider("No new bets after minute", 0, 59, 35, step=5)

with st.expander("Paste Kalshi ladder (optional but recommended)", expanded=False):
    market_text = st.text_area(
        "Paste lines like: 72 to 73 +100",
        height=120,
        placeholder="72 to 73 +100\n70 to 71 +316\n74 to 75 +376"
    )

daily_nws, hourly_nws, hourly_periods, obs_temp, obs_ts, nws_notes = fetch_nws_all(
    profile["lat"], profile["lon"], profile["tz"], profile["station"]
)
daily_om, hourly_om, current_om, om_note = fetch_open_meteo(profile["lat"], profile["lon"], None)
daily_gfs, hourly_gfs, current_gfs, gfs_note = fetch_open_meteo(profile["lat"], profile["lon"], "gfs_seamless")
daily_metno, hourly_metno, current_metno, metno_note = fetch_metno(profile["lat"], profile["lon"])

source_rows = [
    {"Source": "NWS forecast", "Forecast High": daily_nws, "Status": nws_notes["NWS forecast"]},
    {"Source": "NWS hourly", "Forecast High": hourly_nws, "Status": nws_notes["NWS hourly"]},
    {"Source": "Open-Meteo", "Forecast High": daily_om, "Status": om_note},
    {"Source": "GFS", "Forecast High": daily_gfs, "Status": gfs_note},
    {"Source": "MET Norway", "Forecast High": daily_metno, "Status": metno_note},
]

df_show = pd.DataFrame(source_rows)
df_show["Forecast High"] = df_show["Forecast High"].apply(fmt_num)

st.subheader("Forecast Source Diagnostics")
st.dataframe(df_show, use_container_width=True, hide_index=True)

usable = []
for x in [daily_nws, hourly_nws, daily_om, daily_gfs, daily_metno]:
    if isinstance(x, (int, float)):
        xf = float(x)
        if not math.isnan(xf):
            usable.append(xf)

if not usable:
    st.error("No usable forecast sources available right now.")
    st.stop()

sorted_vals = sorted(usable)
core = sorted_vals[1:-1] if len(sorted_vals) >= 4 else sorted_vals
consensus = sum(core) / len(core) + float(profile["bias"])
spread = max(usable) - min(usable) if len(usable) >= 2 else 0.0
sigma = max(0.85, float(profile["sigma"]) + spread * 0.22)

current_candidates = []
for x in [obs_temp, current_om, current_gfs, current_metno]:
    if isinstance(x, (int, float)):
        xf = float(x)
        if not math.isnan(xf):
            current_candidates.append(xf)
current_temp = sum(current_candidates) / len(current_candidates) if current_candidates else None

expected_now = None
momentum_delta = None
heating_needed = None
if current_temp is not None:
    heating_needed = consensus - current_temp

if hourly_periods and current_temp is not None:
    try:
        best = None
        best_delta = None
        tz = ZoneInfo(profile["tz"])
        for p in hourly_periods[:18]:
            dt = datetime.fromisoformat(p["startTime"]).astimezone(tz)
            delta = abs((dt - local_now).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = p
        if best and best.get("temperature") is not None:
            expected_now = float(best["temperature"])
            momentum_delta = current_temp - expected_now
    except Exception:
        pass

st.subheader("Model Inputs")
m1, m2, m3 = st.columns(3)
m1.metric("Consensus High", f"{consensus:.1f}")
m2.metric("Forecast Spread", f"{spread:.1f}")
m3.metric("Sigma", f"{sigma:.2f}")

m4, m5, m6 = st.columns(3)
m4.metric("Current Temperature", f"{current_temp:.1f}" if current_temp is not None else "-")
m5.metric("Expected Now", f"{expected_now:.1f}" if expected_now is not None else "-")
m6.metric("Live Momentum", f"{momentum_delta:+.1f}" if momentum_delta is not None else "-")

if heating_needed is not None:
    st.write("Heating still needed to reach consensus:", round(heating_needed, 1))
if obs_ts:
    st.caption(f"Station observation time: {obs_ts}")

mu = consensus
if momentum_delta is not None:
    mu = mu + MOMENTUM_WEIGHT * momentum_delta
    st.caption(f"Momentum-adjusted consensus: {mu:.1f}")

market_prices = parse_market_lines(market_text) if market_text.strip() else {}
market_labels = list(market_prices.keys())
detected = detect_market_ladder(market_labels) if market_labels else None

effective_mode = ladder_mode
if ladder_mode == "market_auto" and detected:
    effective_mode = detected
elif ladder_mode == "market_auto":
    effective_mode = "auto"

labels = market_labels if market_labels else default_ladder(mu, effective_mode)

rows = []
for lab in labels:
    p = label_prob(lab, mu, sigma)
    rows.append({"Bracket": lab, "WinProb": p, "FairYES": p})
rows.sort(key=lambda x: x["WinProb"], reverse=True)

top = rows[0]
second = rows[1] if len(rows) > 1 else {"WinProb": 0.0}
top_gap = top["WinProb"] - second["WinProb"]

table_rows = []
for r in rows:
    item = {
        "Bracket": r["Bracket"],
        "Fair YES Price": f"{r['FairYES']*100:.1f}c",
        "Win Probability": f"{r['WinProb']*100:.1f}%",
    }
    if r["Bracket"] in market_prices:
        mprob = price_to_prob(market_prices[r["Bracket"]])
        item["Market YES"] = market_prices[r["Bracket"]]
        item["Edge"] = f"{(r['WinProb'] - mprob)*100:+.1f}%"
    table_rows.append(item)

st.subheader("Kalshi Probability Table")
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
st.metric("Top-two bracket gap", f"{top_gap*100:.1f}%")

cutoff = local_now.replace(hour=no_bet_after_hour, minute=no_bet_after_minute, second=0, microsecond=0)
reasons = []
if top["WinProb"] < float(profile["prob_filter"]):
    reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {profile['prob_filter']*100:.0f}%)")
if top_gap < MIN_TOP_TWO_GAP:
    reasons.append(f"Top-two gap too small ({top_gap*100:.1f}% < {MIN_TOP_TWO_GAP*100:.0f}%)")
if local_now > cutoff:
    reasons.append(f"Past cutoff ({cutoff.strftime('%I:%M %p')} local)")
if momentum_delta is not None and momentum_delta <= -NO_BET_LAG and local_now.hour >= 11:
    reasons.append(f"Live temp is {abs(momentum_delta):.1f} behind forecast track")

if reasons:
    st.error("PASS - " + " | ".join(reasons))
else:
    st.success(f"BET SIGNAL: {top['Bracket']} ({top['WinProb']*100:.1f}%)")

st.caption("v11.4 adds New Orleans, Miami, Washington DC, Atlanta, and extra fallback sources (MET Norway + trimmed-average consensus).")
