# streamlit_app.py
# Kalshi Weather Model – Daily High [v8.0]
# Upgrades:
# - Correct Kalshi settlement station map
# - Settlement station validation shown in UI
# - Live obs from settlement station
# - Weather sources + nowcast + settlement bias
# - Kalshi ladder alignment
# - Edge detection / EV
# - Forecast stability and trade filter separated
# - Decision window / market correction warning
# - Peak-temperature tracking

import math
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi Weather Model – Daily High (v8.0)", layout="centered")
st.title("Kalshi Weather Model – Daily High (v8.0)")
st.caption("Kalshi-aligned settlement stations, live settlement obs, nowcast, ladder probabilities, EV and trade filters.")

UA = {"User-Agent": "kalshi-weather-model/8.0"}

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


def fetch_open_meteo_best(lat: float, lon: float, tz: str):
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
    js = safe_get_json(base, params=params)
    if not js:
        return None, None, None, "fetch failed"
    df, hi, sunrise, err = _parse_open_meteo_payload(js, tz)
    if df is not None:
        return df, hi, sunrise, ""
    params["timezone"] = "auto"
    js2 = safe_get_json(base, params=params)
    if not js2:
        return None, None, None, err or "fetch failed"
    return _parse_open_meteo_payload(js2, tz)


def fetch_open_meteo_noaa(lat: float, lon: float, tz: str):
    base = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "timezone": tz,
        "hourly": "temperature_2m",
        "daily": "sunrise,sunset,temperature_2m_max",
        "forecast_days": 2,
    }
    js = safe_get_json(base, params=params)
    if not js:
        return None, None, None, "fetch failed"
    df, hi, sunrise, err = _parse_open_meteo_payload(js, tz)
    if df is not None:
        return df, hi, sunrise, ""
    params["timezone"] = "auto"
    js2 = safe_get_json(base, params=params)
    if not js2:
        return None, None, None, err or "fetch failed"
    return _parse_open_meteo_payload(js2, tz)


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


city = st.selectbox("City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY))
cfg = CITIES[city]
lat = float(cfg["lat"])
lon = float(cfg["lon"])
station = str(cfg["station"])
station_label = str(cfg["label"])
tz = str(cfg["tz"])
tzinfo = ZoneInfo(tz)
local_now = datetime.now(tzinfo)

with st.expander("Settings", expanded=True):
    include_noaa = st.toggle("Include Open-Meteo NOAA GFS/HRRR", value=True)
    include_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    show_hourly_chart = st.toggle("Show hourly chart", value=True)
    grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)
    ladder_mode = st.selectbox("Kalshi ladder alignment", ["auto", "even", "odd"], index=0)
    do_not_bet_prob = st.slider("Trade filter: top probability must exceed", 0.40, 0.70, 0.55, 0.01)
    strong_edge_threshold = st.slider("Strong edge threshold (%)", 1.0, 30.0, 8.0, 0.5)
    small_edge_threshold = st.slider("Small edge threshold (%)", 0.5, 20.0, 3.0, 0.5)
    settlement_bias = st.slider("Settlement station bias correction (°F)", -1.5, 1.5, 0.0, 0.1)
    no_bet_after_hour = st.slider("No new bets after local hour", 10, 15, 11, 1)

with st.expander("Kalshi Odds / EV (recommended)", expanded=True):
    market_text = st.text_area(
        "Paste Kalshi lines with YES odds or cents",
        height=140,
        placeholder="82° to 83° +566\n84° to 85° -489\n86° or above +1011\n77° or below 1c",
    )

st.info(f"Settlement station for **{city}**: **{station}** — {station_label}")

sources = []
chart_df = None
sunrise_local = None

best_df, best_high, best_sunrise, best_err = fetch_open_meteo_best(lat, lon, tz)
sources.append(("Open-Meteo (best match)", best_high, best_err))
if best_high is not None:
    chart_df = best_df
    sunrise_local = best_sunrise

if include_noaa:
    noaa_df, noaa_high, noaa_sunrise, noaa_err = fetch_open_meteo_noaa(lat, lon, tz)
    sources.append(("Open-Meteo NOAA (GFS/HRRR)", noaa_high, noaa_err))
    if chart_df is None and noaa_high is not None:
        chart_df = noaa_df
        sunrise_local = noaa_sunrise

if include_nws:
    _, nws_high, nws_err = fetch_nws_hourly(lat, lon)
    sources.append(("NWS (forecastHourly)", nws_high, nws_err))

settle_temp_f, settle_obs_time, settle_obs_err = fetch_station_obs(station)

vals = [x[1] for x in sources if x[1] is not None]
if not vals:
    st.error("No forecast sources returned successfully.")
    st.stop()

consensus = float(sum(vals) / len(vals))
spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
sigma = float(max(1.1, 0.85 + 0.55 * spread))

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
    if settle_temp_f is not None and settle_obs_time is not None:
        st.metric(f"Current settlement temp ({station})", f"{settle_temp_f:.1f}°F")
        st.caption(f"Obs time: {settle_obs_time.astimezone(tzinfo).strftime('%a %b %d, %I:%M %p')} local")
    else:
        st.metric(f"Current settlement temp ({station})", "—")
        st.caption(f"Obs error: {settle_obs_err}")

st.divider()
st.subheader("Live trend / nowcast")

heating_rate = None
peak_hour = None
trend_proj_high = None

if settle_temp_f is not None and chart_df is not None and not chart_df.empty:
    sunrise_for_calc = sunrise_local if sunrise_local is not None else local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    idx = (chart_df["time"] - sunrise_for_calc).abs().idxmin()
    sunrise_temp = float(chart_df.loc[idx, "temp_f"])
    hrs = max(0.25, (local_now - sunrise_for_calc).total_seconds() / 3600.0)
    if local_now > sunrise_for_calc:
        heating_rate = (settle_temp_f - sunrise_temp) / hrs

    day_df = chart_df[chart_df["time"].dt.date == local_now.date()].copy()
    day_df = day_df[(day_df["time"].dt.hour >= 8) & (day_df["time"].dt.hour <= 20)]
    if not day_df.empty:
        peak_row = day_df.loc[day_df["temp_f"].idxmax()]
        peak_hour = peak_row["time"]
        if heating_rate is not None:
            hours_to_peak = max(0.0, (peak_hour - local_now).total_seconds() / 3600.0)
            trend_proj_high = settle_temp_f + hours_to_peak * heating_rate

a, b, c = st.columns(3)
a.metric("Heating rate since sunrise", "—" if heating_rate is None else f"{heating_rate:+.2f} °F/hr")
b.metric("Forecast peak hour", "—" if peak_hour is None else peak_hour.strftime("%I:%M %p"))
c.metric("Projected high (trend-based)", "—" if trend_proj_high is None else f"{trend_proj_high:.1f}°F")

mu = consensus
if trend_proj_high is not None:
    mu = 0.70 * consensus + 0.30 * trend_proj_high
    st.caption(f"Blended model high = 70% forecast consensus + 30% live trend = **{mu:.1f}°F**")

mu += settlement_bias
if settlement_bias != 0:
    st.caption(f"Settlement-adjusted high = blended high + bias ({settlement_bias:+.1f}°F) = **{mu:.1f}°F**")

st.divider()
st.subheader("Suggested Kalshi Bracket")
ladder, chosen_mode = build_ladder(mu, sigma, ladder_mode)
top = max(ladder, key=lambda x: x["WinProb"])
risk_level, risk_icon, risk_msg = classify_stability(spread, sigma, top["WinProb"])

st.caption(f"Ladder alignment used: **{chosen_mode}**")
if risk_level == "LOW RISK":
    st.success(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")
elif risk_level == "HIGH RISK":
    st.warning(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")
else:
    st.error(f"{risk_icon} FORECAST STABILITY: {risk_level} — {risk_msg}")

trade_filter_reasons = []
if top["WinProb"] < do_not_bet_prob:
    trade_filter_reasons.append(f"Top bracket only {top['WinProb']*100:.1f}% (< {do_not_bet_prob*100:.0f}%)")
if local_now.hour >= no_bet_after_hour and local_now.minute > 15:
    trade_filter_reasons.append(f"Past cutoff ({no_bet_after_hour}:15 local)")
if risk_level == "PASS":
    trade_filter_reasons.append("Forecast stability state is PASS")

if trade_filter_reasons:
    st.error("TRADE FILTER: DO NOT BET — " + " | ".join(trade_filter_reasons))
else:
    st.success("TRADE FILTER: BET ALLOWED — confidence passed your rules.")

st.success(f"Suggested bracket: **{top['Bracket']}** (model ≈ **{top['WinProb']*100:.0f}%**)")

market_prices = parse_market_lines(market_text) if market_text.strip() else {}
rows = []
best_bet = None

for x in sorted(ladder, key=lambda r: r["WinProb"], reverse=True):
    row = {
        "Bracket": x["Bracket"],
        "Win %": f"{x['WinProb']*100:.1f}%",
        "Fair YES": f"{fair_cents_from_prob(x['WinProb']):.1f}¢",
    }
    if x["Bracket"] in market_prices:
        yes_cents = market_prices[x["Bracket"]]
        market_prob = yes_cents / 100.0
        edge = (x["WinProb"] - market_prob) * 100.0
        ev = expected_value_per_1_dollar(x["WinProb"], yes_cents)
        if trade_filter_reasons:
            signal = "NO BET"
        elif edge >= strong_edge_threshold and ev > 0:
            signal = "BET"
        elif edge >= small_edge_threshold and ev > 0:
            signal = "SMALL EDGE"
        else:
            signal = "NO BET"
        row["Market YES"] = f"{yes_cents:.1f}¢"
        row["Edge %"] = f"{edge:+.1f}%"
        row["EV / $1"] = f"{ev:+.3f}"
        row["Signal"] = signal
        if signal == "BET":
            if best_bet is None or edge > best_bet["edge"]:
                best_bet = {"bracket": x["Bracket"], "edge": edge, "ev": ev, "price": yes_cents, "prob": x["WinProb"]*100.0}
    rows.append(row)

st.subheader("Kalshi Edge Table")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if market_prices:
    st.subheader("Bet Signal")
    if best_bet is not None:
        st.success(
            f"BET SIGNAL: **{best_bet['bracket']}** | Model **{best_bet['prob']:.1f}%** | "
            f"Market YES **{best_bet['price']:.1f}¢** | Edge **{best_bet['edge']:+.1f}%** | EV per $1 **{best_bet['ev']:+.3f}**"
        )
    else:
        st.warning("No live edge gate confirmation. Do not force a bet.")
else:
    st.info("Paste Kalshi prices to activate live edge / EV / bet signal.")

st.subheader("Decision Window")
target = local_now.replace(hour=10, minute=30, second=0, microsecond=0)
grace_end = target + timedelta(minutes=grace_minutes)
st.caption(
    f"Local time now: **{local_now.strftime('%a %b %d, %I:%M %p')}** | Target check: **10:30 AM** | Grace: **{grace_minutes} min**"
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

st.caption("v8.0 adds Kalshi settlement station mapping and uses the settlement station obs in the model.")
