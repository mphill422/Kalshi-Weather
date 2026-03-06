import math
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Kalshi Weather MVP – v5 (HRRR + Live Temp + Risk Badge + Ladder Alignment)
# ============================================================

st.set_page_config(page_title="Kalshi Weather MVP", layout="wide")

APP_TITLE = "Kalshi Weather MVP – Daily High (Multi-Source + Live + Ladder)"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-mvp/5.0 (contact: none)"
REQ_TIMEOUT = 12

# Cities
CITIES = {
    "Miami": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New York City": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "New Orleans": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "Austin": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "San Antonio": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    "Phoenix": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Las Vegas": {"lat": 36.1699, "lon": -115.1398, "tz": "America/Los_Angeles"},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
}

# ============================================================
# Helpers
# ============================================================

def http_get_json(url: str) -> dict:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__url__": url}


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def today_local(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def now_local_dt(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def parse_iso_local(ts: str, tz_name: str) -> Optional[datetime]:
    """Open-Meteo returns ISO strings without offset when timezone param is used."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def f_from_c(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bracket_probability(mean: float, sigma: float, low: float, high: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if (low <= mean < high) else 0.0
    z1 = (low - mean) / sigma
    z2 = (high - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))


def compute_sigma(source_highs: list[float]) -> float:
    """Stable uncertainty: base + spread penalty."""
    if not source_highs:
        return 2.8
    spread = max(source_highs) - min(source_highs) if len(source_highs) >= 2 else 0.0
    return max(1.5, 1.8 + 0.60 * spread)


def risk_badge(spread: float, sigma: float, top_prob: float, edge: Optional[float]) -> tuple[str, str]:
    """Return (label, color_kind)."""
    # Base risk from uncertainty
    if spread >= 4.0 or sigma >= 3.5 or top_prob < 0.15:
        base = "PASS"
        kind = "error"
    elif spread >= 2.5 or sigma >= 2.8 or top_prob < 0.20:
        base = "HIGH RISK"
        kind = "warning"
    else:
        base = "LOW RISK"
        kind = "success"

    # If market edge is known, incorporate it
    if edge is not None:
        if edge < 0:
            # Negative edge: downgrade one step
            if base == "LOW RISK":
                base, kind = "HIGH RISK", "warning"
            else:
                base, kind = "PASS", "error"
        elif edge >= 0.05 and base != "PASS":
            # Strong positive edge: keep base, but annotate
            base = f"{base} (EDGE)"

    return base, kind


def round_bracket_label(low: int, size: int) -> str:
    return f"{low}–{low + size}"


# ============================================================
# Open-Meteo (models)
# ============================================================

def fetch_open_meteo(lat: float, lon: float, tz: str, model: Optional[str] = None) -> dict:
    base = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&temperature_unit=fahrenheit"
        f"&hourly=temperature_2m"
        f"&daily=temperature_2m_max,sunrise"
        f"&timezone={tz}"
        f"&forecast_days=2"
    )
    if model:
        base += f"&models={model}"
    return http_get_json(base)


@dataclass
class ModelOut:
    name: str
    daily_high_f: Optional[float]
    hourly_df: Optional[pd.DataFrame]
    sunrise_local: Optional[datetime]
    err: Optional[str]


def extract_open_meteo_today(j: dict, tz: str, name: str) -> ModelOut:
    if not j or "__error__" in j:
        return ModelOut(name, None, None, None, (j.get("__error__") if isinstance(j, dict) else "Unknown error"))

    try:
        t0 = today_local(tz).isoformat()
        daily = j.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        days = daily.get("time", [])
        sunr = daily.get("sunrise", [])

        if not highs or not days:
            return ModelOut(name, None, None, None, "Open-Meteo missing daily fields")

        idx = days.index(t0) if t0 in days else 0
        daily_high = safe_float(highs[idx])

        sunrise_dt = None
        if sunr and len(sunr) > idx:
            sunrise_dt = parse_iso_local(sunr[idx], tz)

        hourly = j.get("hourly", {})
        ht = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])

        df = None
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                dt = parse_iso_local(ts, tz)
                if dt is None:
                    continue
                rows.append({"dt": dt, "temp_f": safe_float(temp)})
            df0 = pd.DataFrame(rows).dropna()
            if not df0.empty:
                tday = today_local(tz)
                df0 = df0[df0["dt"].dt.date == tday].copy()
                if not df0.empty:
                    df = df0

        return ModelOut(name, daily_high, df, sunrise_dt, None)
    except Exception as e:
        return ModelOut(name, None, None, None, f"Open-Meteo parse error: {e}")


# ============================================================
# NWS: hourly forecast + LIVE observation
# ============================================================

def nws_points(lat: float, lon: float) -> dict:
    return http_get_json(f"https://api.weather.gov/points/{lat},{lon}")


def fetch_nws_hourly_high(lat: float, lon: float, tz: str) -> tuple[Optional[float], Optional[pd.DataFrame], Optional[str]]:
    try:
        p = nws_points(lat, lon)
        if "__error__" in p:
            return None, None, f"NWS points error: {p.get('__error__')}"

        hourly_url = (p.get("properties") or {}).get("forecastHourly")
        if not hourly_url:
            return None, None, "NWS missing forecastHourly URL"

        h = http_get_json(hourly_url)
        if "__error__" in h:
            return None, None, f"NWS hourly error: {h.get('__error__')}"

        periods = ((h.get("properties") or {}).get("periods") or [])
        if not periods:
            return None, None, "NWS hourly periods missing/empty"

        rows = []
        for per in periods:
            start = per.get("startTime")
            temp = per.get("temperature")
            unit = per.get("temperatureUnit")
            if unit and unit.upper() != "F":
                continue
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
            except Exception:
                continue
            rows.append({"dt": dt, "temp_f": safe_float(temp)})

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return None, None, "NWS hourly parse produced no rows"

        tday = today_local(tz)
        df_today = df[df["dt"].dt.date == tday].copy()
        if df_today.empty:
            return None, None, "NWS hourly has no rows for today"

        daily_high = float(df_today["temp_f"].max())
        return daily_high, df_today, None

    except Exception as e:
        return None, None, f"NWS error: {e}"


def fetch_nws_live_observation(lat: float, lon: float, tz: str) -> tuple[Optional[float], Optional[datetime], Optional[str]]:
    """Best-effort: nearest station -> observations/latest."""
    try:
        p = nws_points(lat, lon)
        if "__error__" in p:
            return None, None, f"NWS points error: {p.get('__error__')}"

        stations_url = (p.get("properties") or {}).get("observationStations")
        if not stations_url:
            return None, None, "NWS missing observationStations URL"

        s = http_get_json(stations_url)
        if "__error__" in s:
            return None, None, f"NWS stations error: {s.get('__error__')}"

        features = s.get("features") or []
        if not features:
            return None, None, "NWS stations list empty"

        station_id = ((features[0].get("properties") or {}).get("stationIdentifier"))
        if not station_id:
            return None, None, "NWS stationIdentifier missing"

        latest = http_get_json(f"https://api.weather.gov/stations/{station_id}/observations/latest")
        if "__error__" in latest:
            return None, None, f"NWS latest obs error: {latest.get('__error__')}"

        props = latest.get("properties") or {}
        temp_c = safe_float(((props.get("temperature") or {}).get("value")))
        ts = props.get("timestamp")
        if temp_c is None or not ts:
            return None, None, "NWS latest obs missing temperature/timestamp"

        try:
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(ZoneInfo(tz))
        except Exception:
            dt_local = None

        return f_from_c(temp_c), dt_local, None

    except Exception as e:
        return None, None, f"NWS live obs error: {e}"


# ============================================================
# Kalshi ladder alignment (NO API)
# ============================================================

LADDER_RE = re.compile(
    r"(?P<a>-?\d+)\s*(?:°)?\s*(?:to|–|-)\s*(?P<b>-?\d+)|(?P<lo>-?\d+)\s*(?:°)?\s*or\s*below|(?P<hi>-?\d+)\s*(?:°)?\s*or\s*above",
    re.IGNORECASE,
)


@dataclass
class Ladder:
    bracket_size: int
    bins: list[tuple[Optional[int], Optional[int]]]
    # Each bin as (low_inclusive, high_inclusive) for display.


def parse_kalshi_contract_lines(text: str, bracket_size: int) -> Optional[Ladder]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None

    bins: list[tuple[Optional[int], Optional[int]]] = []
    for ln in lines:
        m = LADDER_RE.search(ln)
        if not m:
            continue
        if m.group("a") and m.group("b"):
            a = int(m.group("a"))
            b = int(m.group("b"))
            lo, hi = min(a, b), max(a, b)
            bins.append((lo, hi))
        elif m.group("lo"):
            bins.append((None, int(m.group("lo"))))
        elif m.group("hi"):
            bins.append((int(m.group("hi")), None))

    if not bins:
        return None

    # Keep order as pasted (Kalshi UI order)
    return Ladder(bracket_size=bracket_size, bins=bins)


def generate_estimated_kalshi_ladder(consensus: float, bracket_size: int) -> Ladder:
    """Fallback if you don't paste lines.

    This tries to mimic the *common* Kalshi ladder pattern:
    - width = bracket_size
    - 4 interior bins + two tails

    It will not always match Kalshi (they can change ladders), but it avoids nonsense.
    """
    if bracket_size == 2:
        mid_low = int(round(consensus)) - 1
        tail_lo = mid_low - 3
        tail_hi = mid_low + 6
        interior = [(mid_low - 2, mid_low - 1), (mid_low, mid_low + 1), (mid_low + 2, mid_low + 3), (mid_low + 4, mid_low + 5)]
        bins = [(None, tail_lo)] + interior + [(tail_hi, None)]
        return Ladder(bracket_size=2, bins=bins)

    # bracket_size == 1
    mid_low = int(round(consensus))
    tail_lo = mid_low - 3
    tail_hi = mid_low + 4
    interior = [(mid_low - 2, mid_low - 2), (mid_low - 1, mid_low - 1), (mid_low, mid_low), (mid_low + 1, mid_low + 1), (mid_low + 2, mid_low + 2), (mid_low + 3, mid_low + 3)]
    bins = [(None, tail_lo)] + interior + [(tail_hi, None)]
    return Ladder(bracket_size=1, bins=bins)


def ladder_bin_label(bin_: tuple[Optional[int], Optional[int]]) -> str:
    lo, hi = bin_
    if lo is None and hi is not None:
        return f"{hi}° or below"
    if hi is None and lo is not None:
        return f"{lo}° or above"
    if lo is not None and hi is not None:
        if lo == hi:
            return f"{lo}°"
        return f"{lo}° to {hi}°"
    return "(unknown)"


def prob_for_ladder_bin(mean: float, sigma: float, bin_: tuple[Optional[int], Optional[int]], bracket_size: int) -> float:
    lo, hi = bin_

    # Convert inclusive bounds to [low, high+1) for 1-degree, or [low, high+1) for 2-degree too.
    # Example: 81–82 corresponds to [81, 83) in degree-F integer space.
    if lo is None and hi is not None:
        return bracket_probability(mean, sigma, -1e9, hi + 1)
    if hi is None and lo is not None:
        return bracket_probability(mean, sigma, lo, 1e9)
    if lo is not None and hi is not None:
        return bracket_probability(mean, sigma, lo, hi + 1)

    return 0.0


# ============================================================
# UI
# ============================================================

left, right = st.columns([1, 1])
with left:
    city = st.selectbox("City", list(CITIES.keys()))
with right:
    st.caption("If a source errors, the app still runs — it just excludes that source from consensus.")

info = CITIES[city]
tz = info["tz"]

controls = st.expander("Settings", expanded=True)
with controls:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include_gfs = st.toggle("Include Open-Meteo GFS", value=True)
    with c2:
        include_hrrr = st.toggle("Include Open-Meteo HRRR (best-effort)", value=False)
    with c3:
        include_nws = st.toggle("Include NWS (forecastHourly + live obs)", value=True)
    with c4:
        bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2], index=1)

    lock_time_local = time(10, 30)
    grace_minutes = st.slider("Grace minutes after 10:30 local", min_value=0, max_value=180, value=95, step=5)

    st.divider()
    st.subheader("Kalshi ladder alignment")
    use_pasted_contracts = st.toggle("Use pasted Kalshi contract list (recommended)", value=True)
    contracts_text = ""
    if use_pasted_contracts:
        contracts_text = st.text_area(
            "Paste Kalshi market contracts (one per line). Example: `84° to 85°`",
            height=140,
            placeholder="84° to 85°\n78° to 79°\n80° to 81°\n82° to 83°\n77° or below\n86° or above",
        )

    st.divider()
    st.subheader("Market pricing (optional)")
    st.caption("If you paste prices/odds, the app can show mispricing (edge). Otherwise, you still get model probabilities.")
    pricing_text = st.text_area(
        "Paste pricing lines (optional). Examples: `81° to 82° 0.42` OR `81° to 82° +120` (Yes price or American odds)",
        height=120,
        placeholder="81° to 82° 0.42\n83° to 84° +354\n...",
    )


# ============================================================
# Pricing parsing
# ============================================================

PRICE_RE = re.compile(r"(?P<label>.+?)\s+(?P<val>[-+]?\d+(?:\.\d+)?)\s*$")


def implied_prob_from_american_odds(odds: float) -> Optional[float]:
    # odds like +120 or -150
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def parse_pricing_lines(text: str) -> dict[str, float]:
    """Return label -> implied probability.

    Accepted:
    - label 0.42   (assumed probability or 'Yes' price in [0,1])
    - label 42     (assumed cents => 0.42)
    - label +120 or -150 (American odds)
    """
    out: dict[str, float] = {}
    for ln in [x.strip() for x in (text or "").splitlines() if x.strip()]:
        m = PRICE_RE.match(ln)
        if not m:
            continue
        label = m.group("label").strip()
        val = float(m.group("val"))

        prob: Optional[float]
        if abs(val) >= 101 and ("+" in ln or "-" in ln):
            prob = implied_prob_from_american_odds(val)
        else:
            # price
            if val > 1.0:
                prob = val / 100.0
            else:
                prob = val
        if prob is None:
            continue
        prob = max(0.0, min(1.0, prob))
        out[label] = prob
    return out


# ============================================================
# Fetch + Compute
# ============================================================

with st.spinner("Fetching forecasts…"):
    # Open-Meteo default
    om_raw = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    om = extract_open_meteo_today(om_raw, tz, name="Open-Meteo")

    gfs = None
    if include_gfs:
        gfs_raw = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        gfs = extract_open_meteo_today(gfs_raw, tz, name="Open-Meteo (GFS)")

    hrrr = None
    if include_hrrr:
        # Open-Meteo model names can vary; this is best-effort and will fail gracefully.
        hrrr_raw = fetch_open_meteo(info["lat"], info["lon"], tz, model="hrrr")
        hrrr = extract_open_meteo_today(hrrr_raw, tz, name="Open-Meteo (HRRR)")

    nws_high = None
    nws_hourly = None
    nws_err = None
    live_temp_f = None
    live_dt = None
    live_err = None
    if include_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)
        live_temp_f, live_dt, live_err = fetch_nws_live_observation(info["lat"], info["lon"], tz)


# ============================================================
# Display source status
# ============================================================

st.subheader(f"{city} – Today’s High Forecasts (°F)")

status_rows = []

def add_row(name: str, high: Optional[float], err: Optional[str]):
    if high is not None:
        status_rows.append((name, f"{high:.1f}°F", "OK"))
    else:
        status_rows.append((name, "—", err or "Error"))

add_row(om.name, om.daily_high_f, om.err)
if gfs is not None:
    add_row(gfs.name, gfs.daily_high_f, gfs.err)
if hrrr is not None:
    add_row(hrrr.name, hrrr.daily_high_f, hrrr.err)
if include_nws:
    add_row("NWS (forecastHourly)", nws_high, nws_err)

st.table(pd.DataFrame(status_rows, columns=["Source", "Today High", "Status"]))

if include_nws:
    if live_temp_f is not None and live_dt is not None:
        st.info(f"**Current airport/nearby NWS station temp:** {live_temp_f:.1f}°F (as of {live_dt.strftime('%I:%M %p')})")
    else:
        st.caption(f"Live temperature unavailable: {live_err or 'Unknown issue'}")


# ============================================================
# Consensus + hourly curve
# ============================================================

source_highs: list[float] = []
for x in [om.daily_high_f, (gfs.daily_high_f if gfs else None), (hrrr.daily_high_f if hrrr else None), nws_high]:
    if isinstance(x, (int, float)) and x is not None:
        source_highs.append(float(x))

if not source_highs:
    st.error("No valid sources returned a high for today. Toggle off failing sources and try again.")
    st.stop()

consensus = float(sum(source_highs) / len(source_highs))
spread = max(source_highs) - min(source_highs) if len(source_highs) > 1 else 0.0
sigma = compute_sigma(source_highs)

# Choose an hourly series for chart/nowcast
hourly_source_name = None
hourly_df = None
sunrise_dt = None

# Prefer Open-Meteo hourly (it is consistent with the forecast highs)
for model_out in [om, (hrrr if hrrr else None), (gfs if gfs else None)]:
    if model_out is not None and model_out.hourly_df is not None and not model_out.hourly_df.empty:
        hourly_df = model_out.hourly_df.sort_values("dt").copy()
        hourly_source_name = model_out.name
        sunrise_dt = model_out.sunrise_local
        break

# Fallback to NWS hourly
if hourly_df is None and nws_hourly is not None and not nws_hourly.empty:
    hourly_df = nws_hourly.sort_values("dt").copy()
    hourly_source_name = "NWS (forecastHourly)"

c1, c2, c3 = st.columns(3)
c1.metric("Consensus high", f"{consensus:.1f}°F")
c2.metric("Cross-source spread", f"{spread:.1f}°F")
c3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")


# ============================================================
# Nowcast: heating rate since sunrise + projected high
# ============================================================

st.subheader("Live trend (nowcast)")

now_local = now_local_dt(tz)

heating_rate = None
projected_high = None
peak_time = None

if live_temp_f is not None and hourly_df is not None and not hourly_df.empty:
    # Peak time from hourly forecast curve
    peak_row = hourly_df.loc[hourly_df["temp_f"].idxmax()]
    peak_time = peak_row["dt"]

    if sunrise_dt is None:
        # If we couldn't get sunrise from OM, approximate as 6:30 AM local
        sunrise_dt = datetime.combine(now_local.date(), time(6, 30), tzinfo=ZoneInfo(tz))

    # Get forecast temp near sunrise to baseline the heating
    df2 = hourly_df.copy()
    df2["mins_from_sunrise"] = (df2["dt"] - sunrise_dt).dt.total_seconds() / 60.0
    df_after = df2[df2["mins_from_sunrise"].between(-90, 180)].copy()  # around sunrise
    sunrise_temp = None
    if not df_after.empty:
        sunrise_temp = float(df_after.iloc[(df_after["mins_from_sunrise"].abs().argsort())].iloc[0]["temp_f"])

    # Heating rate using observed current temp vs sunrise baseline
    hours_since_sunrise = max(0.01, (now_local - sunrise_dt).total_seconds() / 3600.0)
    if sunrise_temp is not None and hours_since_sunrise > 0.1:
        heating_rate = (live_temp_f - sunrise_temp) / hours_since_sunrise

    # Projected high: current + heating_rate * hours_until_peak, then blend with model consensus
    if heating_rate is not None and peak_time is not None:
        hours_to_peak = max(0.0, (peak_time - now_local).total_seconds() / 3600.0)
        raw_proj = live_temp_f + heating_rate * hours_to_peak
        # Blend 60% model, 40% nowcast early; shift toward nowcast later in morning
        w_now = min(0.65, max(0.25, hours_since_sunrise / 6.0))
        projected_high = (1 - w_now) * consensus + w_now * raw_proj

cols = st.columns(3)
if live_temp_f is not None:
    cols[0].metric("Current airport temp", f"{live_temp_f:.1f}°F", f"as of {live_dt.strftime('%I:%M %p') if live_dt else ''}")
else:
    cols[0].metric("Current airport temp", "—")

if heating_rate is not None:
    cols[1].metric("Heating rate since sunrise", f"{heating_rate:+.2f} °F/hr")
else:
    cols[1].metric("Heating rate since sunrise", "—")

if projected_high is not None:
    cols[2].metric("Projected high (nowcast)", f"{projected_high:.1f}°F")
else:
    cols[2].metric("Projected high (nowcast)", "—")


# ============================================================
# Ladder-aligned bracket suggestion + mispricing
# ============================================================

st.subheader("Suggested Kalshi Bracket")

# Ladder
ladder = parse_kalshi_contract_lines(contracts_text, bracket_size) if use_pasted_contracts else None
ladder_mode = "PASTED" if ladder else "ESTIMATED"
if ladder is None:
    ladder = generate_estimated_kalshi_ladder(consensus, bracket_size)

# Pricing (optional)
market_probs = parse_pricing_lines(pricing_text)

rows = []
for b in ladder.bins:
    label = ladder_bin_label(b)
    p_model = prob_for_ladder_bin(projected_high if projected_high is not None else consensus, sigma, b, bracket_size)

    p_mkt = None
    # Match using simple containment: if a pasted label starts with ours or vice versa
    for k, v in market_probs.items():
        kk = k.strip().lower().replace("°", "")
        ll = label.strip().lower().replace("°", "")
        if kk == ll or kk.startswith(ll) or ll.startswith(kk):
            p_mkt = v
            break

    edge = (p_model - p_mkt) if (p_mkt is not None) else None

    rows.append(
        {
            "Kalshi contract": label,
            "Model win %": round(p_model * 100.0, 1),
            "Market implied %": (round(p_mkt * 100.0, 1) if p_mkt is not None else None),
            "Edge %": (round(edge * 100.0, 1) if edge is not None else None),
        }
    )

df_ladder = pd.DataFrame(rows)

# Pick best model bin (or best edge if market provided)
if "Edge %" in df_ladder.columns and df_ladder["Edge %"].notna().any():
    best_idx = df_ladder["Edge %"].astype(float).idxmax()
else:
    best_idx = df_ladder["Model win %"].astype(float).idxmax()

best_row = df_ladder.loc[best_idx]

# Summaries
p_top = float(best_row["Model win %"]) / 100.0
edge_val = None
if best_row.get("Edge %") is not None and pd.notna(best_row.get("Edge %")):
    edge_val = float(best_row["Edge %"]) / 100.0

badge, kind = risk_badge(spread=spread, sigma=sigma, top_prob=p_top, edge=edge_val)

if kind == "success":
    st.success(f"**{badge}**")
elif kind == "warning":
    st.warning(f"**{badge}**")
else:
    st.error(f"**{badge}**")

st.info(
    f"Ladder mode: **{ladder_mode}**.  Best contract: **{best_row['Kalshi contract']}**  "
    f"(model ≈ **{best_row['Model win %']:.1f}%**"
    + (f", market ≈ **{best_row['Market implied %']:.1f}%**, edge ≈ **{best_row['Edge %']:.1f}%**" if pd.notna(best_row.get("Market implied %")) else "")
    + ")"
)

st.dataframe(df_ladder, use_container_width=True, hide_index=True)


# ============================================================
# Decision window
# ============================================================

lock_dt = datetime.combine(now_local.date(), lock_time_local, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.subheader("Decision Window")
st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}**  "
    f"| Target check: **10:30 AM**  "
    f"| With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)

if now_local <= deadline_dt:
    st.info("You are inside the preferred window (or within grace).")
else:
    st.warning("Past the preferred window. The market is often sharper later in the day.")


# ============================================================
# Hourly curve
# ============================================================

st.subheader("Hourly temperature curve (today)")
if hourly_df is not None and not hourly_df.empty:
    st.caption(f"Hourly source used: {hourly_source_name}")
    st.line_chart(hourly_df.set_index("dt")["temp_f"])

    peak_row = hourly_df.loc[hourly_df["temp_f"].idxmax()]
    peak_time = peak_row["dt"]
    peak_temp = float(peak_row["temp_f"])

    st.write(f"Peak hour (from {hourly_source_name}): **{peak_time.strftime('%I:%M %p')}** at **{peak_temp:.1f}°F**")
else:
    st.caption("Hourly curve unavailable (sources returned daily high but not usable hourly).")


st.divider()
st.caption(
    "Notes: "
    "(1) Kalshi ladders can change; pasting the contract list is the only way to guarantee alignment. "
    "(2) ‘HRRR’ is best-effort via Open-Meteo; if it errors, just toggle it off. "
    "(3) Market implied % and Edge % require you to paste prices/odds."
)
