# streamlit_app.py
# Kalshi Weather Model – Daily High [v9.5]
# Adds city-specific distribution shaping on top of v8.5:
# - city-specific sigma multiplier
# - slight city bias defaults
# - bracket probabilities better matched to each city's typical behavior
# - keeps settlement stations, trade day quality, top-two gap,
#   revision tracker, momentum checks, exact cutoff, EV table,
#   and market mispricing detector
#
# Note:
# This version aims to improve bracket probability quality.
# It does not guarantee wins.

import math
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Model – Daily High (v9.5)", layout="centered")
st.title("Model – Daily High (v9.5)")
st.caption("Adds locked core settings, mobile-friendly Kalshi entry, market ladder auto-detection, auto city shaping, and stronger live momentum handling.")

UA = {"User-Agent": "kalshi-weather-model/9.5"}

CITIES: Dict[str, Dict[str, str | float]] = {
    "Miami": {"lat": 25.7933, "lon": -80.2906, "station": "KMIA", "label": "Miami Intl Airport", "tz": "America/New_York"},
    "New York": {"lat": 40.7812, "lon": -73.9665, "station": "KNYC", "label": "Central Park", "tz": "America/New_York"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "station": "KATL", "label": "Hartsfield-Jackson Atlanta Intl", "tz": "America/New_York"},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "station": "KMSY", "label": "Louis Armstrong New Orleans Intl", "tz": "America/Chicago"},
    "Houston": {"lat": 29.6454, "lon": -95.2789, "station": "KHOU", "label": "Houston Hobby Airport", "tz": "America/Chicago"},
    "Austin": {"lat": 30.1975, "lon": -97.6664, "station": "KAUS", "label": "Austin Bergstrom Intl", "tz": "America/Chicago"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "station": "KDFW", "label": "Dallas/Fort Worth Intl", "tz": "America/Chicago"},
    "San Antonio": {"lat": 29.5337, "lon": -98.4698, "station": "KSAT", "label": "San Antonio Intl", "tz": "America/Chicago"},
    "Phoenix": {"lat": 33.4342, "lon": -112.0116, "station": "KPHX", "label": "Phoenix Sky Harbor Intl", "tz": "America/Phoenix"},
    "Las Vegas": {"lat": 36.0801, "lon": -115.1522, "station": "KLAS", "label": "Harry Reid Intl", "tz": "America/Los_Angeles"},
    "Los Angeles": {"lat": 33.9416, "lon": -118.4085, "station": "KLAX", "label": "Los Angeles Intl Airport", "tz": "America/Los_Angeles"},
}
DEFAULT_CITY = "Miami"

# City-specific distribution shaping
# lower sigma_mult = tighter outcome distribution
# higher sigma_mult = wider outcome distribution
CITY_PROFILE = {
    "Phoenix": {"sigma_mult": 0.82, "default_bias": 0.1},
    "Las Vegas": {"sigma_mult": 0.85, "default_bias": 0.1},
    "Los Angeles": {"sigma_mult": 0.90, "default_bias": -0.1},
    "Dallas": {"sigma_mult": 0.96, "default_bias": 0.0},
    "Atlanta": {"sigma_mult": 0.98, "default_bias": 0.0},
    "Austin": {"sigma_mult": 1.08, "default_bias": 0.1},
    "San Antonio": {"sigma_mult": 1.05, "default_bias": 0.1},
    "New York": {"sigma_mult": 1.00, "default_bias": 0.0},
    "Miami": {"sigma_mult": 1.15, "default_bias": 0.0},
    "Houston": {"sigma_mult": 1.18, "default_bias": 0.0},
    "New Orleans": {"sigma_mult": 1.16, "default_bias": 0.0},
}

CITY_PROB_FILTERS = {
    "Phoenix": 0.55,
    "Las Vegas": 0.55,
    "Houston": 0.62,
}
DEFAULT_PROB_FILTER = 0.58
MIN_TOP_TWO_GAP = 0.12


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def american_odds_to_prob(odds: float):
    if odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def yes_price_from_american_odds(odds: float):
    p = american_odds_to_prob(odds)
    return None if p is None else p * 100.0


def fair_cents_from_prob(p: float) -> float:
    return clamp(p * 100.0, 0.0, 100.0)


def expected_value_per_1_dollar(p: float, yes_price_cents: float) -> float:
    price = yes_price_cents / 100.0
    return p * (1.0 - price) - (1.0 - p) * price


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
            price_cents = yes_price_from_american_odds(float(odds_match.group(1)))
            label = line[:odds_match.start()].strip()
        elif cents_match:
            price_cents = float(cents_match.group(1))
            label = line[:cents_match.start()].strip()
        label = label.replace("º", "°").strip()
        if label and price_cents is not None:
            out[label] = clamp(price_cents, 0.0, 100.0)
    return out




def detect_market_ladder(market_labels):
    nums = []
    odd_hits = 0
    even_hits = 0
    for label in market_labels:
        found = re.findall(r"\d+", label)
        if found:
            first = int(found[0])
            if first % 2 == 0:
                even_hits += 1
            else:
                odd_hits += 1
            nums.extend(int(x) for x in found[:2])
    if not nums:
        return None
    return "odd" if odd_hits > even_hits else "even"


def remap_label_to_market(target_label, market_labels):
    if target_label in market_labels:
        return target_label
    target_nums = [int(x) for x in re.findall(r"\d+", target_label)]
    if not target_nums:
        return target_label

    target_center = sum(target_nums[:2]) / max(1, len(target_nums[:2]))
    best = None
    best_score = None
    for label in market_labels:
        nums = [int(x) for x in re.findall(r"\d+", label)]
        if not nums:
            continue
        center = sum(nums[:2]) / max(1, len(nums[:2]))
        score = abs(center - target_center)
        if best_score is None or score < best_score:
            best_score = score
            best = label
    return best or target_label

def parse_revision_lines(text: str):
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r'(\d{1,2}):(\d{2})\s*,?\s*(-?\d+(?:\.\d+)?)', line)
        if m:
            rows.append({"hh": int(m.group(1)), "mm": int(m.group(2)), "forecast_f": float(m.group(3)), "raw": line})
            continue
        m2 = re.search(r'(-?\d+(?:\.\d+)?)', line)
        if m2:
            rows.append({"hh": None, "mm": None, "forecast_f": float(m2.group(1)), "raw": line})
    return rows


def _parse_open_meteo_payload(js: dict, tz: str):
    try:
        times = js["hourly"]["time"]
        temps = js["hourly"]["temperature_2m"]
        df = pd.DataFrame({"time": pd.to_datetime(times), "temp_f": temps})
        df["time"] = df["time"].dt.tz_localize(ZoneInfo(tz), ambiguous="NaT", nonexistent="shift_forward")
        df = df.dropna(subset=["time"]).reset_index(drop=True)
        if df.empty:
            return None, None, None, "no valid hourly rows"
        today_high = float(js["daily"]["temperature_2m_max"][0])
        sunrise = pd.to_datetime(js["daily"]["sunrise"][0]).tz_localize(
            ZoneInfo(tz), ambiguous="NaT", nonexistent="shift_forward"
        )
        return df, today_high, sunrise, ""
    except Exception as e:
        return None, None, None, str(e)


def fetch_open_meteo_with_retries(base_url: str, lat: float, lon: float, tz: str):
    attempts = [
        {"timezone": tz, "forecast_days": 2},
        {"timezone": "auto", "forecast_days": 2},
        {"timezone": tz, "forecast_days": 1},
        {"timezone": "auto", "forecast_days": 1},
    ]
    last_err = "fetch failed"
    for attempt in attempts:
        params = {
            "latitude": lat,
            "longitude": lon,
            "temperature_unit": "fahrenheit",
            "timezone": attempt["timezone"],
            "hourly": "temperature_2m",
            "daily": "sunrise,sunset,temperature_2m_max",
            "forecast_days": attempt["forecast_days"],
        }
        js = safe_get_json(base_url, params=params, timeout=14)
        if not js:
            last_err = "fetch failed"
            continue
        df, hi, sunrise, err = _parse_open_meteo_payload(js, tz)
        if df is not None and hi is not None:
            return df, hi, sunrise, ""
        last_err = err or "parse failed"
    return None, None, None, last_err


def fetch_open_meteo_best(lat: float, lon: float, tz: str):
    return fetch_open_meteo_with_retries("https://api.open-meteo.com/v1/forecast", lat, lon, tz)


def fetch_open_meteo_noaa(lat: float, lon: float, tz: str):
    return fetch_open_meteo_with_retries("https://api.open-meteo.com/v1/gfs", lat, lon, tz)


def fetch_nws_hourly(lat: float, lon: float) -> Tuple[Optional[pd.DataFrame], Optional[float], str]:
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


def nearest_even(x: float) -> int:
    return int(2 * round(x / 2.0))


def nearest_odd(x: float) -> int:
    n = int(round(x))
    if n % 2 == 0:
        lower = n - 1
        upper = n + 1
        return lower if abs(x - lower) <= abs(x - upper) else upper
    return n


def build_ladder(mu: float, sigma: float, mode: str):
    even_start = nearest_even(mu) - 4
    odd_start = nearest_odd(mu) - 4

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
                p = norm_cdf((hi - mu) / sigma)
            elif hi is None:
                p = 1.0 - norm_cdf((lo - mu) / sigma)
            else:
                p = prob_between(mu, sigma, lo, hi + 1e-9)
            rows.append({"Bracket": label, "WinProb": p})
        total = sum(r["WinProb"] for r in rows)
        for r in rows:
            r["WinProb"] = r["WinProb"] / total if total > 0 else 0.0
        return rows

    ladders = {"even": one(even_start), "odd": one(odd_start)}
    chosen = mode if mode in ("even", "odd") else max(ladders.keys(), key=lambda k: max(x["WinProb"] for x in ladders[k]))
    return ladders[chosen], chosen


def classify_stability(spread: float, sigma: float, top_prob: float):
    if spread >= 4.0 or sigma >= 3.5 or top_prob < 0.20:
        return "PASS", "🔴", "Forecast stability is poor."
    if spread >= 2.5 or sigma >= 3.0 or top_prob < 0.25:
        return "HIGH RISK", "🟡", "Forecast stability is moderate."
    return "LOW RISK", "🟢", "Forecast stability is good."


def score_trade_day(source_count: int, spread: float, sigma: float, top_prob: float, top_gap: float, momentum_delta: Optional[float], revision_bias: float):
    score = 0.0
    if source_count >= 3:
        score += 2
    elif source_count == 2:
        score += 1
    if spread <= 1.5:
        score += 2
    elif spread <= 2.2:
        score += 1
    if sigma <= 1.7:
        score += 2
    elif sigma <= 2.3:
        score += 1
    if top_prob >= 0.62:
        score += 3
    elif top_prob >= 0.55:
        score += 2
    elif top_prob >= 0.50:
        score += 1
    if top_gap >= 0.12:
        score += 2
    elif top_gap >= 0.07:
        score += 1
    if momentum_delta is not None:
        if abs(momentum_delta) <= 1.0:
            score += 1
        elif momentum_delta >= 1.0:
            score += 1
    if revision_bias == 0:
        score += 1
    elif revision_bias > 0:
        score += 0.5

    if score >= 9:
        return "HIGH EDGE DAY", "🟢", f"Trade score {score:.1f}/13"
    if score >= 6:
        return "NORMAL DAY", "🟡", f"Trade score {score:.1f}/13"
    return "NO TRADE DAY", "🔴", f"Trade score {score:.1f}/13"


def classify_mispricing(edge_pct: float, ev: float):
    if edge_pct >= 12 and ev > 0:
        return "🔥 MISPRICED", "Large model edge"
    if edge_pct >= 6 and ev > 0:
        return "✅ UNDERPRICED", "Positive edge"
    if abs(edge_pct) <= 3:
        return "⚖️ FAIR", "Near model fair value"
    if edge_pct < -3:
        return "❌ OVERPRICED", "Market too expensive"
    return "—", ""


city = st.selectbox("City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY))
cfg = CITIES[city]
lat = float(cfg["lat"])
lon = float(cfg["lon"])
station = str(cfg["station"])
station_label = str(cfg["label"])
tz = str(cfg["tz"])
tzinfo = ZoneInfo(tz)
local_now = datetime.now(tzinfo)
profile = CITY_PROFILE.get(city, {"sigma_mult": 1.0, "default_bias": 0.0})

with st.expander("Settings", expanded=True):
    st.caption("Core settings are locked in. Only day-to-day items stay adjustable.")

    include_noaa = st.toggle("Include Open-Meteo NOAA GFS/HRRR", value=True)
    include_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    show_hourly_chart = st.toggle("Show hourly chart", value=True)

    grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 45, 5)
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["auto", "even", "odd", "market_auto"], index=0)
    no_bet_after_hour = st.slider("No new bets after local hour", 9, 15, 12, 1)
    no_bet_after_minute = st.slider("No new bets after minute", 0, 59, 35, 5)

    strong_edge_threshold = 10.0
    small_edge_threshold = 3.0
    settlement_bias = 0.0
    momentum_weight = 0.35
    noon_lag_threshold = 1.5
    do_not_bet_prob = CITY_PROB_FILTERS.get(city, DEFAULT_PROB_FILTER)

    st.markdown(
        f"""
**Permanent built-in settings**
- Probability filter for **{city}**: **{do_not_bet_prob:.2f}**
- Strong edge threshold: **10%**
- Small edge threshold: **3%**
- Settlement station bias: **0.0°F**
- Peak heating momentum weight: **0.35**
- No-bet lag threshold: **1.5°F**
- Minimum top-two gap: **{MIN_TOP_TWO_GAP*100:.0f}%**
        """
    )

with st.expander("Forecast revision tracker (paste recent forecast updates)", expanded=False):
    revision_text = st.text_area(
        "Paste lines like 01:40, 75 or 02:41, 74",
        height=120,
        placeholder="01:40, 75\n01:52, 72\n02:41, 74\n02:48, 75\n03:31, 72",
    )

with st.expander("Kalshi Odds / EV (recommended)", expanded=True):
    st.caption("Use paste mode or the mobile-friendly quick entry fields.")
    market_text = st.text_area(
        "Paste Kalshi lines with YES odds or cents",
        height=140,
        placeholder="82° to 83° +566\n84° to 85° -489\n86° or above +1011\n77° or below 1c",
    )

    st.markdown("**Mobile quick entry (use this if paste won't work)**")
    quick_mode = st.toggle("Use quick manual Kalshi entry", value=False)

    quick_lines = []
    if quick_mode:
        for i in range(6):
            c1, c2 = st.columns([3, 2])
            with c1:
                bracket = st.text_input(
                    f"Bracket {i+1}",
                    value="",
                    key=f"quick_bracket_{i}",
                    placeholder="e.g. 81° to 82° or 80° or below",
                )
            with c2:
                price = st.text_input(
                    f"YES odds/cents {i+1}",
                    value="",
                    key=f"quick_price_{i}",
                    placeholder="e.g. +334 or 41c",
                )
            if bracket.strip() and price.strip():
                quick_lines.append(f"{bracket.strip()} {price.strip()}")

    if quick_mode and quick_lines:
        market_text = "\n".join(quick_lines)
        st.code(market_text, language="text")

st.info(f"Settlement station for **{city}**: **{station}** — {station_label}")
st.caption("City profile loaded.")

sources = []
chart_df = None
sunrise_local = None
nws_df = None

best_df, best_high, best_sunrise, best_err = fetch_open_meteo_best(lat, lon, tz)
sources.append(("Open-Meteo (best match)", best_high, best_err or "OK"))
if best_high is not None:
    chart_df = best_df
    sunrise_local = best_sunrise

if include_noaa:
    noaa_df, noaa_high, noaa_sunrise, noaa_err = fetch_open_meteo_noaa(lat, lon, tz)
    sources.append(("Open-Meteo NOAA (GFS/HRRR)", noaa_high, noaa_err or "OK"))
    if chart_df is None and noaa_high is not None:
        chart_df = noaa_df
        sunrise_local = noaa_sunrise

if include_nws:
    nws_df, nws_high, nws_err = fetch_nws_hourly(lat, lon)
    sources.append(("NWS (forecastHourly)", nws_high, nws_err or "OK"))

if chart_df is None and nws_df is not None:
    chart_df = nws_df.copy()
    chart_df["time"] = chart_df["time"].dt.tz_convert(tzinfo)
    sunrise_local = local_now.replace(hour=6, minute=0, second=0, microsecond=0)

settle_temp_f, settle_obs_time, settle_obs_err = fetch_station_obs(station)

vals = [x[1] for x in sources if x[1] is not None]
source_count = len(vals)
if not vals:
    st.error("No forecast sources returned successfully.")
    st.stop()

consensus = float(sum(vals) / len(vals))
spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
base_sigma = float(max(1.1, 0.85 + 0.55 * spread))
sigma = clamp(base_sigma * sigma_shape, 0.8, 5.0)

st.subheader(f"{city} – Today’s High Forecasts (°F)")
df_sources = pd.DataFrame([{
    "Source": s[0],
    "Today High": ("—" if s[1] is None else f"{s[1]:.1f}°F"),
    "Status": s[2],
} for s in sources])
st.dataframe(df_sources, use_container_width=True, hide_index=True)

if source_count < 2:
    st.warning("Forecast confidence reduced: only one forecast source is currently available.")
elif source_count == 2:
    st.info("Forecast confidence moderate: two forecast sources are available.")
else:
    st.success("Forecast confidence stronger: three forecast sources are available.")

c1, c2 = st.columns(2)
with c1:
    st.metric("Consensus high", f"{consensus:.1f}°F")
    st.metric("Cross-source spread", f"{spread:.1f}°F")
with c2:
    st.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")
    if settle_temp_f is not None and settle_obs_time is not None:
        st.metric(f"Current settlement temp ({station})", f"{settle_temp_f:.1f}°F")
        st.caption(f"Obs time: {settle_obs_time.astimezone(tzinfo).strftime('%a %b %d, %I:%M %p')} local")
    else:
        st.metric(f"Current settlement temp ({station})", "—")
        st.caption(f"Obs error: {settle_obs_err}")

revision_rows = parse_revision_lines(revision_text)
revision_bias = 0.0
revision_msg = ""
if revision_rows:
    rev_vals = [r["forecast_f"] for r in revision_rows]
    latest_rev = rev_vals[-1]
    first_rev = rev_vals[0]
    net_change = latest_rev - first_rev
    recent_change = latest_rev - (rev_vals[-2] if len(rev_vals) >= 2 else first_rev)
    if net_change <= -1.0:
        revision_bias = -0.7
        revision_msg = f"Forecast revisions trending cooler ({net_change:+.1f}°F)."
    elif net_change >= 1.0:
        revision_bias = +0.7
        revision_msg = f"Forecast revisions trending hotter ({net_change:+.1f}°F)."
    elif recent_change <= -1.0:
        revision_bias = -0.4
        revision_msg = f"Latest revision turned cooler ({recent_change:+.1f}°F)."
    elif recent_change >= 1.0:
        revision_bias = +0.4
        revision_msg = f"Latest revision turned hotter ({recent_change:+.1f}°F)."
    else:
        revision_msg = "Forecast revisions mostly stable."

st.divider()
st.subheader("Live trend / nowcast")

heating_rate = None
peak_hour = None
trend_proj_high = None
momentum_delta = None
momentum_label = None
expected_now = None

if settle_temp_f is not None and chart_df is not None and not chart_df.empty:
    sunrise_for_calc = sunrise_local if sunrise_local is not None else local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    idx = (chart_df["time"] - sunrise_for_calc).abs().idxmin()
    sunrise_temp = float(chart_df.loc[idx, "temp_f"])
    hrs = max(0.25, (local_now - sunrise_for_calc).total_seconds() / 3600.0)
    if local_now > sunrise_for_calc:
        heating_rate = (settle_temp_f - sunrise_temp) / hrs

    day_df = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    day_df = day_df[(day_df["time"].dt.hour >= 6) & (day_df["time"].dt.hour <= 20)]
    if not day_df.empty:
        nearest_idx = (day_df["time"] - local_now).abs().idxmin()
        expected_now = float(day_df.loc[nearest_idx, "temp_f"])
        momentum_delta = settle_temp_f - expected_now
        if momentum_delta >= 1.0:
            momentum_label = "running hotter than forecast"
        elif momentum_delta <= -1.0:
            momentum_label = "running cooler than forecast"
        else:
            momentum_label = "tracking close to forecast"

        peak_row = day_df.loc[day_df["temp_f"].idxmax()]
        peak_hour = peak_row["time"]
        if heating_rate is not None:
            hours_to_peak = max(0.0, (peak_hour - local_now).total_seconds() / 3600.0)
            trend_proj_high = settle_temp_f + hours_to_peak * heating_rate

a, b, c = st.columns(3)
a.metric("Heating rate since sunrise", "—" if heating_rate is None else f"{heating_rate:+.2f} °F/hr")
b.metric("Forecast peak hour", "—" if peak_hour is None else peak_hour.strftime("%I:%M %p"))
c.metric("Projected high (trend-based)", "—" if trend_proj_high is None else f"{trend_proj_high:.1f}°F")

if momentum_delta is not None:
    st.info(f"Peak heating momentum: **{momentum_delta:+.1f}°F** — {momentum_label}.")
    if momentum_delta >= 1.5:
        st.success("Live momentum signal: station is running materially HOTTER than forecast track.")
    elif momentum_delta <= -1.5:
        st.error("Live momentum signal: station is running materially COOLER than forecast track.")
    elif abs(momentum_delta) <= 0.7:
        st.caption("Live momentum signal: station is tracking close to forecast.")
elif chart_df is None:
    st.warning("Trend engine unavailable because no hourly curve could be built.")
else:
    st.info("Trend engine is waiting for enough usable hourly data.")

if revision_msg:
    if revision_bias < 0:
        st.warning(f"Revision tracker: {revision_msg}")
    elif revision_bias > 0:
        st.info(f"Revision tracker: {revision_msg}")
    else:
        st.caption(f"Revision tracker: {revision_msg}")

mu = consensus
if trend_proj_high is not None:
    mu = 0.70 * consensus + 0.30 * trend_proj_high
    st.caption(f"Blended model high = 70% forecast consensus + 30% live trend = **{mu:.1f}°F**")

if momentum_delta is not None:
    momentum_adjust = momentum_weight * momentum_delta
    if momentum_delta <= -1.5:
        momentum_adjust -= 0.3
    elif momentum_delta >= 1.5:
        momentum_adjust += 0.2
    mu = mu + momentum_adjust
    st.caption(f"Momentum-adjusted high = blended high + live momentum = **{mu:.1f}°F**")

if revision_bias != 0:
    mu = mu + revision_bias
    st.caption(f"Revision-adjusted high = model high + revision bias ({revision_bias:+.1f}°F) = **{mu:.1f}°F**")

mu += settlement_bias
if settlement_bias != 0:
    st.caption(f"Settlement-adjusted high = model high + bias ({settlement_bias:+.1f}°F) = **{mu:.1f}°F**")

st.divider()
st.subheader("Suggested Kalshi Bracket")
market_prices = parse_market_lines(market_text) if market_text.strip() else {}
effective_ladder_mode = ladder_mode
if ladder_mode == "market_auto" and market_prices:
    detected_mode = detect_market_ladder(list(market_prices.keys()))
    if detected_mode in ("even", "odd"):
        effective_ladder_mode = detected_mode
ladder, chosen_mode = build_ladder(mu, sigma, effective_ladder_mode if effective_ladder_mode != "market_auto" else "auto")
ordered = sorted(ladder, key=lambda x: x["WinProb"], reverse=True)
top = ordered[0]
second = ordered[1]
top_gap = top["WinProb"] - second["WinProb"]

risk_level, risk_icon, risk_msg = classify_stability(spread, sigma, top["WinProb"])
day_quality, day_icon, day_msg = score_trade_day(source_count, spread, sigma, top["WinProb"], top_gap, momentum_delta, revision_bias)

st.caption(f"Ladder alignment used: **{chosen_mode}**")
if day_quality == "HIGH EDGE DAY":
    st.success(f"{day_icon} TRADE DAY QUALITY: {day_quality} — {day_msg}")
elif day_quality == "NORMAL DAY":
    st.warning(f"{day_icon} TRADE DAY QUALITY: {day_quality} — {day_msg}")
else:
    st.error(f"{day_icon} TRADE DAY QUALITY: {day_quality} — {day_msg}")

if risk_level == "LOW RISK":
    st.success(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")
elif risk_level == "HIGH RISK":
    st.warning(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")
else:
    st.error(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")

st.metric("Top-two bracket gap", f"{top_gap*100:.1f}%")

current_cutoff = local_now.replace(hour=no_bet_after_hour, minute=no_bet_after_minute, second=0, microsecond=0)

trade_filter_reasons = []
if top["WinProb"] < do_not_bet_prob:
    trade_filter_reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {do_not_bet_prob*100:.0f}%)")
if top_gap < MIN_TOP_TWO_GAP:
    trade_filter_reasons.append(f"Top-two gap too small ({top_gap*100:.1f}% < {MIN_TOP_TWO_GAP*100:.0f}%)")
if local_now > current_cutoff:
    trade_filter_reasons.append(f"Past cutoff ({current_cutoff.strftime('%I:%M %p')} local)")
if risk_level == "PASS":
    trade_filter_reasons.append("Forecast stability state is PASS")
if source_count < 2:
    trade_filter_reasons.append("Only one forecast source available")
if day_quality == "NO TRADE DAY":
    trade_filter_reasons.append("Trade day quality is NO TRADE")
if top_gap < 0.07:
    trade_filter_reasons.append(f"Top-two gap too small ({top_gap*100:.1f}%)")
if momentum_delta is not None and momentum_delta <= -noon_lag_threshold and local_now.hour >= 11:
    trade_filter_reasons.append(f"Live temp is {abs(momentum_delta):.1f}°F behind forecast track")
if revision_bias < 0 and local_now.hour >= 11:
    trade_filter_reasons.append("Forecast revisions are trending cooler")
if settle_temp_f is not None and expected_now is not None and local_now.hour >= 12:
    if settle_temp_f < expected_now - noon_lag_threshold:
        trade_filter_reasons.append("Settlement station is lagging too much for an upper-bracket bet")
if revision_bias < 0 and momentum_delta is not None and momentum_delta < 0:
    trade_filter_reasons.append("Momentum + revisions both lean lower")

if trade_filter_reasons:
    st.error("TRADE FILTER: DO NOT BET — " + " | ".join(trade_filter_reasons))
else:
    st.success("TRADE FILTER: BET ALLOWED — confidence passed your rules.")

st.success(f"Suggested bracket: **{top['Bracket']}** (model ≈ **{top['WinProb']*100:.0f}%**)")

rows = []
best_bet = None

for x in ordered:
    row = {
        "Bracket": x["Bracket"],
        "Win %": f"{x['WinProb']*100:.1f}%",
        "Fair YES": f"{fair_cents_from_prob(x['WinProb']):.1f}¢",
    }
    market_label = remap_label_to_market(x["Bracket"], list(market_prices.keys())) if market_prices else x["Bracket"]
    if market_label in market_prices:
        yes_cents = market_prices[market_label]
        market_prob = yes_cents / 100.0
        edge = (x["WinProb"] - market_prob) * 100.0
        ev = expected_value_per_1_dollar(x["WinProb"], yes_cents)
        mispricing, note = classify_mispricing(edge, ev)
        if trade_filter_reasons:
            signal = "NO BET"
        elif edge >= strong_edge_threshold and ev > 0:
            signal = "BET"
        elif edge >= small_edge_threshold and ev > 0:
            signal = "SMALL EDGE"
        else:
            signal = "NO BET"
        row["Market YES"] = f"{yes_cents:.1f}¢"
        row["Model vs Mkt"] = f"{edge:+.1f}%"
        row["EV / $1"] = f"{ev:+.3f}"
        row["Mispricing"] = mispricing
        row["Signal"] = signal
        if signal == "BET":
            if best_bet is None or edge > best_bet["edge"]:
                best_bet = {
                    "bracket": x["Bracket"],
                    "edge": edge,
                    "ev": ev,
                    "price": yes_cents,
                    "prob": x["WinProb"] * 100.0,
                    "mispricing": mispricing,
                    "note": note,
                }
    rows.append(row)

st.subheader("Kalshi Edge Table")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if market_prices:
    st.subheader("Market Mispricing Detector")
    flagged = [r for r in rows if r.get("Mispricing") in ("🔥 MISPRICED", "✅ UNDERPRICED")]
    if flagged:
        st.success("Market has at least one bracket priced below the model.")
        st.dataframe(pd.DataFrame(flagged), use_container_width=True, hide_index=True)
    else:
        st.info("No strong underpriced bracket detected right now.")

    st.subheader("Bet Signal")
    if best_bet is not None:
        st.success(
            f"{best_bet['mispricing']} | **{best_bet['bracket']}** | "
            f"Model **{best_bet['prob']:.1f}%** | Market YES **{best_bet['price']:.1f}¢** | "
            f"Gap **{best_bet['edge']:+.1f}%** | EV per $1 **{best_bet['ev']:+.3f}**"
        )
    else:
        st.warning("No live edge gate confirmation. Do not force a bet.")
else:
    st.info("Paste Kalshi prices to activate live edge / EV / mispricing detection.")

st.subheader("Decision Window")
target = local_now.replace(hour=10, minute=30, second=0, microsecond=0)
grace_end = target + timedelta(minutes=grace_minutes)
st.caption(
    f"Local time now: **{local_now.strftime('%a %b %d, %I:%M %p')}** | "
    f"Target check: **10:30 AM** | Grace: **{grace_minutes} min** | "
    f"Hard cutoff: **{current_cutoff.strftime('%I:%M %p')}**"
)
if local_now < target:
    st.warning("Early. Target check is 10:30 AM local.")
elif local_now <= grace_end:
    st.info("Inside preferred window (or within grace).")
else:
    st.warning("Past preferred window. Market is often sharper later in the day.")

if show_hourly_chart and chart_df is not None and not chart_df.empty:
    st.subheader("Hourly temperature curve (today)")
    df_plot = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    if not df_plot.empty:
        st.line_chart(df_plot.set_index("time")["temp_f"])
        peak_t = df_plot.loc[df_plot["temp_f"].idxmax(), "time"]
        peak_v = float(df_plot["temp_f"].max())
        st.caption(f"Peak hour (forecast): {peak_t.strftime('%I:%M %p')} at {peak_v:.1f}°F")

st.caption("v9.5 adds city-specific distribution shaping so bracket probabilities are tighter in cities like Phoenix and Vegas, and wider in cities like Miami and Houston.")
