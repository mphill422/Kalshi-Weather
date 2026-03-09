# Kalshi Temperature Model v13.0
# Full Streamlit model with:
# - all requested cities restored
# - NWS forecast / NWS hourly / airport observation
# - Open-Meteo / GFS / MET Norway
# - weighted consensus
# - airport observation trend
# - cloud suppression
# - peak-time logic
# - market ladder detection
# - confidence score
# - BET / PASS / HOLD / CASHOUT notes

import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Temperature Model v13", layout="wide")
st.title("Kalshi Temperature Model v13.0")
st.caption(
    "Full model: all cities, all weather sources, weighted consensus, airport trend, "
    "cloud penalty, peak-time logic, ladder detection, confidence scoring, and BET / PASS / HOLD / CASHOUT notes."
)

CITIES = {
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "tz": "America/Phoenix", "station": "KPHX", "sigma": 1.10, "bias": 0.50, "prob_filter": 0.55, "peak_hour": 17, "peak_minute": 30},
    "Las Vegas": {"lat": 36.0840, "lon": -115.1537, "tz": "America/Los_Angeles", "station": "KLAS", "sigma": 1.00, "bias": 0.30, "prob_filter": 0.55, "peak_hour": 17, "peak_minute": 45},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles", "station": "KLAX", "sigma": 1.15, "bias": -0.60, "prob_filter": 0.60, "peak_hour": 15, "peak_minute": 30},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "station": "KDFW", "sigma": 1.00, "bias": 0.40, "prob_filter": 0.60, "peak_hour": 16, "peak_minute": 30},
    "Austin": {"lat": 30.1945, "lon": -97.6699, "tz": "America/Chicago", "station": "KAUS", "sigma": 1.10, "bias": 0.20, "prob_filter": 0.60, "peak_hour": 16, "peak_minute": 15},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "station": "KIAH", "sigma": 1.50, "bias": 0.30, "prob_filter": 0.62, "peak_hour": 16, "peak_minute": 15},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "tz": "America/Chicago", "station": "KMSY", "sigma": 1.40, "bias": 0.20, "prob_filter": 0.62, "peak_hour": 16, "peak_minute": 0},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "station": "KMIA", "sigma": 1.40, "bias": 0.10, "prob_filter": 0.62, "peak_hour": 15, "peak_minute": 45},
    "Washington DC": {"lat": 38.8521, "lon": -77.0377, "tz": "America/New_York", "station": "KDCA", "sigma": 1.15, "bias": 0.00, "prob_filter": 0.58, "peak_hour": 15, "peak_minute": 30},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "station": "KATL", "sigma": 1.15, "bias": 0.10, "prob_filter": 0.58, "peak_hour": 15, "peak_minute": 45},
}

SOURCE_WEIGHTS = {
    "NWS forecast": 0.35,
    "NWS hourly": 0.25,
    "MET Norway": 0.20,
    "Open-Meteo": 0.10,
    "GFS": 0.10,
}

MAX_SPREAD_FOR_TRADE = 3.0
MIN_TOP_TWO_GAP = 0.12
MOMENTUM_WEIGHT = 0.35
LATE_WINDOW_MINUTES = 90
HEATING_STALL_THRESHOLD_30M = 0.5
DEFAULT_CUTOFF_HOUR = 12
DEFAULT_CUTOFF_MINUTE = 35
CONFIDENCE_MIN = 70

@st.cache_data(ttl=600, show_spinner=False)
def safe_get_json(url, params=None, headers=None):
    try:
        final_headers = {"User-Agent": "kalshi-temp-v13"}
        if headers:
            final_headers.update(headers)
        r = requests.get(url, params=params, headers=final_headers, timeout=12)
        r.raise_for_status()
        return r.json(), "OK"
    except Exception as e:
        return None, str(e)

def fmt_num(x, digits=1):
    try:
        if x is None:
            return "-"
        xf = float(x)
        if math.isnan(xf):
            return "-"
        return f"{xf:.{digits}f}"
    except Exception:
        return "-"

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
        return [
            f"{center-5} or below",
            f"{center-4} to {center-3}",
            f"{center-2} to {center-1}",
            f"{center} to {center+1}",
            f"{center+2} to {center+3}",
            f"{center+4} or above",
        ]
    return [
        f"{center-4} or below",
        f"{center-3} to {center-2}",
        f"{center-1} to {center}",
        f"{center+1} to {center+2}",
        f"{center+3} to {center+4}",
        f"{center+5} or above",
    ]

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

def weighted_consensus(source_values, bias):
    numer = 0.0
    denom = 0.0
    for name, val in source_values.items():
        if val is None:
            continue
        try:
            vf = float(val)
            if math.isnan(vf):
                continue
            w = SOURCE_WEIGHTS.get(name, 0.0)
            numer += w * vf
            denom += w
        except Exception:
            continue
    if denom == 0:
        vals = [v for v in source_values.values() if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) + bias if vals else None
    return (numer / denom) + bias

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
            "NWS forecast": status, "NWS hourly": status, "Station obs": status,
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
        "hourly": "temperature_2m,cloud_cover",
        "current": "temperature_2m,cloud_cover",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "forecast_days": 1,
    }
    if model:
        params["models"] = model
    data, status = safe_get_json("https://api.open-meteo.com/v1/forecast", params=params)
    if not data:
        return {"daily_high": None, "hourly_high": None, "current_temp": None, "current_cloud": None, "status": status}

    out = {"daily_high": None, "hourly_high": None, "current_temp": None, "current_cloud": None, "status": "OK"}
    try:
        out["daily_high"] = float(data["daily"]["temperature_2m_max"][0])
    except Exception:
        pass
    try:
        out["current_temp"] = float(data["current"]["temperature_2m"])
    except Exception:
        pass
    try:
        out["current_cloud"] = float(data["current"]["cloud_cover"])
    except Exception:
        pass
    try:
        hourly = data["hourly"]["temperature_2m"]
        if hourly:
            out["hourly_high"] = max(hourly[:24])
    except Exception:
        pass
    return out

def fetch_metno(lat, lon):
    data, status = safe_get_json(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": lat, "lon": lon},
        headers={"User-Agent": "kalshi-temp-v13"},
    )
    if not data:
        return {"daily_high": None, "hourly_high": None, "current_temp": None, "status": status}

    out = {"daily_high": None, "hourly_high": None, "current_temp": None, "status": "OK"}
    try:
        ts = data["properties"]["timeseries"][:24]
        vals = []
        for i, item in enumerate(ts):
            det = item["data"]["instant"]["details"]
            t = det.get("air_temperature")
            if t is not None:
                tf = float(t) * 9.0 / 5.0 + 32.0
                vals.append(tf)
                if i == 0:
                    out["current_temp"] = tf
        if vals:
            out["daily_high"] = max(vals)
            out["hourly_high"] = max(vals)
    except Exception as e:
        out["status"] = str(e)
    return out

@st.cache_data(ttl=300, show_spinner=False)
def fetch_obs_history(station):
    data, status = safe_get_json(f"https://api.weather.gov/stations/{station}/observations")
    if not data:
        return pd.DataFrame(), status
    try:
        feats = data.get("features", [])
        rows = []
        for f in feats[:80]:
            props = f.get("properties", {})
            c = props.get("temperature", {}).get("value")
            ts = props.get("timestamp")
            if c is None or ts is None:
                continue
            tf = float(c) * 9.0 / 5.0 + 32.0
            rows.append({"timestamp": ts, "temp_f": tf})
        df = pd.DataFrame(rows)
        if df.empty:
            return df, "No rows"
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna().sort_values("timestamp")
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)

def nearest_prior_temp(df, target_ts):
    if df.empty:
        return None
    prior = df[df["timestamp"] <= target_ts]
    if prior.empty:
        return None
    return float(prior.iloc[-1]["temp_f"])

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
city = st.selectbox("City", list(CITIES.keys()))
profile = CITIES[city]
local_now = datetime.now(ZoneInfo(profile["tz"]))

c1, c2 = st.columns(2)
with c1:
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["market_auto", "auto", "even", "odd"], index=0)
with c2:
    no_bet_after_hour = st.slider("No new bets after local hour", 9, 18, DEFAULT_CUTOFF_HOUR)
    no_bet_after_minute = st.slider("No new bets after minute", 0, 59, DEFAULT_CUTOFF_MINUTE, step=5)

with st.expander("Paste Kalshi ladder (optional but recommended)", expanded=False):
    market_text = st.text_area(
        "Paste lines like: 72 to 73 +100",
        height=120,
        placeholder="72 to 73 +100\n70 to 71 +316\n74 to 75 +376"
    )

# ------------------------------------------------------------
# Fetch all sources
# ------------------------------------------------------------
daily_nws, hourly_nws, hourly_periods, obs_temp, obs_ts, nws_notes = fetch_nws_all(
    profile["lat"], profile["lon"], profile["tz"], profile["station"]
)
om = fetch_open_meteo(profile["lat"], profile["lon"], None)
gfs = fetch_open_meteo(profile["lat"], profile["lon"], "gfs_seamless")
metno = fetch_metno(profile["lat"], profile["lon"])
obs_hist, obs_hist_status = fetch_obs_history(profile["station"])

source_rows = [
    {"Source": "NWS forecast", "Forecast High": daily_nws, "Status": nws_notes["NWS forecast"]},
    {"Source": "NWS hourly", "Forecast High": hourly_nws, "Status": nws_notes["NWS hourly"]},
    {"Source": "Open-Meteo", "Forecast High": om["daily_high"], "Status": om["status"]},
    {"Source": "GFS", "Forecast High": gfs["daily_high"], "Status": gfs["status"]},
    {"Source": "MET Norway", "Forecast High": metno["daily_high"], "Status": metno["status"]},
]

df_sources = pd.DataFrame(source_rows)
df_sources["Forecast High"] = df_sources["Forecast High"].apply(fmt_num)

st.subheader("Forecast Source Diagnostics")
st.dataframe(df_sources, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Build model inputs
# ------------------------------------------------------------
source_values = {
    "NWS forecast": daily_nws,
    "NWS hourly": hourly_nws,
    "Open-Meteo": om["daily_high"],
    "GFS": gfs["daily_high"],
    "MET Norway": metno["daily_high"],
}

usable = [float(v) for v in source_values.values() if isinstance(v, (int, float)) and not math.isnan(float(v))]
if not usable:
    st.error("No usable forecast sources available right now.")
    st.stop()

consensus = weighted_consensus(source_values, float(profile["bias"]))
spread = max(usable) - min(usable) if len(usable) >= 2 else 0.0
sigma = max(0.85, float(profile["base_sigma"]) + spread * 0.22)

current_candidates = []
for x in [obs_temp, om["current_temp"], gfs["current_temp"], metno["current_temp"]]:
    if isinstance(x, (int, float)):
        current_candidates.append(float(x))
current_temp = sum(current_candidates) / len(current_candidates) if current_candidates else None

cloud_candidates = []
for x in [om.get("current_cloud"), gfs.get("current_cloud")]:
    if isinstance(x, (int, float)):
        cloud_candidates.append(float(x))
current_cloud = sum(cloud_candidates) / len(cloud_candidates) if cloud_candidates else None

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

# Airport trend
temp_30m_ago = None
temp_60m_ago = None
heating_rate_30m = None
heating_rate_60m = None

if not obs_hist.empty and current_temp is not None:
    current_ts_utc = pd.Timestamp(datetime.now(timezone.utc))
    temp_30m_ago = nearest_prior_temp(obs_hist, current_ts_utc - pd.Timedelta(minutes=30))
    temp_60m_ago = nearest_prior_temp(obs_hist, current_ts_utc - pd.Timedelta(minutes=60))
    if temp_30m_ago is not None:
        heating_rate_30m = current_temp - temp_30m_ago
    if temp_60m_ago is not None:
        heating_rate_60m = current_temp - temp_60m_ago

# Peak logic
peak_dt = local_now.replace(hour=profile["peak_hour"], minute=profile["peak_minute"], second=0, microsecond=0)
minutes_to_peak = int((peak_dt - local_now).total_seconds() / 60)

# Penalties / adjustments
if current_cloud is not None:
    if current_cloud > 40:
        sigma += 0.70
    elif current_cloud > 20:
        sigma += 0.25

late_peak_window = minutes_to_peak <= LATE_WINDOW_MINUTES
if late_peak_window:
    sigma += 0.50

if momentum_delta is not None and momentum_delta < -1.0:
    sigma += 0.45

mu = consensus
if momentum_delta is not None:
    mu = mu + MOMENTUM_WEIGHT * momentum_delta

# ------------------------------------------------------------
# Display inputs
# ------------------------------------------------------------
st.subheader("Model Inputs")
m1, m2, m3 = st.columns(3)
m1.metric("Consensus High", fmt_num(consensus))
m2.metric("Forecast Spread", fmt_num(spread))
m3.metric("Sigma", fmt_num(sigma, 2))

m4, m5, m6 = st.columns(3)
m4.metric("Current Temperature", fmt_num(current_temp))
m5.metric("Expected Now", fmt_num(expected_now))
m6.metric("Live Momentum", "-" if momentum_delta is None else f"{momentum_delta:+.1f}")

m7, m8, m9 = st.columns(3)
m7.metric("Cloud Cover", "-" if current_cloud is None else f"{current_cloud:.0f}%")
m8.metric("Minutes to Peak", f"{minutes_to_peak}")
m9.metric("Obs History Status", obs_hist_status)

st.subheader("Airport Trend")
a1, a2, a3 = st.columns(3)
a1.metric("Temp 30m Ago", fmt_num(temp_30m_ago))
a2.metric("Temp 60m Ago", fmt_num(temp_60m_ago))
a3.metric("Current Airport Temp", fmt_num(obs_temp))

a4, a5, a6 = st.columns(3)
a4.metric("Heating Last 30m", "-" if heating_rate_30m is None else f"{heating_rate_30m:+.1f}")
a5.metric("Heating Last 60m", "-" if heating_rate_60m is None else f"{heating_rate_60m:+.1f}")
still_forming = "Yes"
if late_peak_window and heating_rate_30m is not None and heating_rate_30m < HEATING_STALL_THRESHOLD_30M:
    still_forming = "Probably not"
a6.metric("High Still Forming?", still_forming)

if heating_needed is not None:
    st.write("Heating still needed to reach consensus:", round(heating_needed, 1))
if obs_ts:
    st.caption(f"Station observation time: {obs_ts}")
st.caption(f"Momentum-adjusted consensus: {mu:.1f}")

# Confidence score
source_agreement = max(0, 100 - spread * 18)
momentum_score = 50
if momentum_delta is not None:
    momentum_score = max(0, min(100, 50 + momentum_delta * 18))
cloud_score = 100 if current_cloud is None else max(0, 100 - current_cloud * 1.25)
trend_score = 50
if heating_rate_30m is not None:
    trend_score = max(0, min(100, 50 + heating_rate_30m * 25))
confidence = round(0.35 * source_agreement + 0.20 * momentum_score + 0.20 * cloud_score + 0.25 * trend_score, 1)
st.metric("Trade Confidence", f"{confidence}/100")

# ------------------------------------------------------------
# Ladder and probabilities
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Trade rules
# ------------------------------------------------------------
cutoff = local_now.replace(hour=no_bet_after_hour, minute=no_bet_after_minute, second=0, microsecond=0)
reasons = []

if spread > MAX_SPREAD_FOR_TRADE:
    reasons.append(f"Forecast spread too wide ({spread:.1f} > {MAX_SPREAD_FOR_TRADE:.1f})")
if top["WinProb"] < float(profile["prob_filter"]):
    reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {profile['prob_filter']*100:.0f}%)")
if top_gap < MIN_TOP_TWO_GAP:
    reasons.append(f"Top-two gap too small ({top_gap*100:.1f}% < {MIN_TOP_TWO_GAP*100:.1f}%)")
if local_now > cutoff:
    reasons.append(f"Past cutoff ({cutoff.strftime('%I:%M %p')} local)")
if momentum_delta is not None and momentum_delta < -0.5:
    reasons.append(f"Momentum weak ({momentum_delta:+.1f})")
if late_peak_window and heating_rate_30m is not None and heating_rate_30m < HEATING_STALL_THRESHOLD_30M and heating_needed is not None and heating_needed >= 2.0:
    reasons.append(f"Late-day stall: heating last 30m only {heating_rate_30m:+.1f}, still need {heating_needed:.1f}")
if confidence < CONFIDENCE_MIN:
    reasons.append(f"Confidence too low ({confidence}/100)")

hold_notes = []
if top["WinProb"] >= 0.70:
    hold_notes.append("Top bracket still dominant")
if heating_rate_30m is not None and heating_rate_30m > 0.8:
    hold_notes.append("Airport still heating")
if late_peak_window and heating_rate_30m is not None and heating_rate_30m < 0.0:
    hold_notes.append("Consider cashout: airport cooling")

if reasons:
    st.error("PASS - " + " | ".join(reasons))
else:
    st.success(f"BET SIGNAL: {top['Bracket']} ({top['WinProb']*100:.1f}%)")

if hold_notes:
    st.info("Hold / exit notes: " + " | ".join(hold_notes))

st.caption(
    "v13 restores all cities and weather sources, plus weighted consensus, airport trend, "
    "cloud filter, peak-time logic, ladder detection, confidence score, and late-day stall detection."
)
