import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# App config
# =========================
st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

DEFAULT_CITY = "Austin, TX"
REQUEST_TIMEOUT = 12

SESSION = requests.Session()
SESSION.headers.update(
    {
        # NWS asks for a User-Agent that identifies your application
        "User-Agent": "KalshiWeatherDashboard/1.0 (contact: streamlit-app)",
        "Accept": "application/geo+json, application/json",
    }
)

# =========================
# City coordinates (NO geocoding needed)
# =========================
CITIES: Dict[str, Dict[str, object]] = {
    # Texas core
    "Austin, TX": {"lat": 30.2672, "lon": -97.7431, "tz": "America/Chicago"},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970, "tz": "America/Chicago"},
    "Houston, TX": {"lat": 29.7604, "lon": -95.3698, "tz": "America/Chicago"},
    "San Antonio, TX": {"lat": 29.4241, "lon": -98.4936, "tz": "America/Chicago"},
    # Requested additions
    "New York City, NY": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Atlanta, GA": {"lat": 33.7490, "lon": -84.3880, "tz": "America/New_York"},
    "Miami, FL": {"lat": 25.7617, "lon": -80.1918, "tz": "America/New_York"},
    "New Orleans, LA": {"lat": 29.9511, "lon": -90.0715, "tz": "America/Chicago"},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
    # Useful extra
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
}

# =========================
# Utilities
# =========================
def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = REQUEST_TIMEOUT) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def now_date_in_tz(tz_name: str) -> pd.Timestamp:
    # Streamlit cloud supports tz database names
    return pd.Timestamp.now(tz=tz_name).normalize()


def to_local_dt(series, tz_name: str) -> pd.Series:
    # Accepts strings / timestamps; returns tz-aware local datetimes where possible
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert(tz_name)
    except Exception:
        # If conversion fails, return best-effort (maybe already local)
        return pd.to_datetime(series, errors="coerce")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def fmt_temp(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.1f}"


def fmt_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    try:
        return ts.strftime("%I:%M %p")
    except Exception:
        return "—"


def erf_cdf(x: float) -> float:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_prob_between(a: float, b: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    za = (a - mu) / sigma
    zb = (b - mu) / sigma
    return max(0.0, erf_cdf(zb) - erf_cdf(za))


# =========================
# Weather source fetchers
# =========================
@dataclass
class SourceResult:
    name: str
    daily_high_f: Optional[float]
    peak_time_local: Optional[pd.Timestamp]
    current_temp_f: Optional[float]
    hourly_df: Optional[pd.DataFrame]
    err: Optional[str]


def fetch_open_meteo(lat: float, lon: float, tz: str, model: str = "best") -> SourceResult:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "current_weather": "true",
    }
    if model and model != "best":
        params["models"] = model  # e.g., "gfs"
    payload, err = safe_get_json(url, params=params)

    if payload is None:
        return SourceResult(
            name=f"Open-Meteo ({model})",
            daily_high_f=None,
            peak_time_local=None,
            current_temp_f=None,
            hourly_df=None,
            err=err,
        )

    # Hourly
    hourly = payload.get("hourly") or {}
    ht = hourly.get("time")
    temps = hourly.get("temperature_2m")

    hourly_df = None
    daily_high = None
    peak_time = None

    try:
        if ht is not None and temps is not None and len(ht) == len(temps) and len(ht) > 0:
            df = pd.DataFrame({"time": ht, "temp_f": temps})
            df["dt"] = pd.to_datetime(df["time"], errors="coerce")
            # Open-Meteo returns already in local tz when timezone param set; keep naive ok
            # But we still filter by local date using tz name:
            # If df["dt"] is naive, compare to local today naive date.
            today_local = now_date_in_tz(tz).date()
            df["date_local"] = pd.to_datetime(df["dt"], errors="coerce").dt.date
            df = df.dropna(subset=["dt", "temp_f"])
            df_today = df[df["date_local"] == today_local].copy()

            if len(df_today) > 0:
                idx = df_today["temp_f"].astype(float).idxmax()
                daily_high = float(df_today.loc[idx, "temp_f"])
                peak_time = pd.to_datetime(df_today.loc[idx, "dt"], errors="coerce")
                hourly_df = df_today[["dt", "temp_f"]].copy()
    except Exception as e:
        # fail-soft
        err = (err or "") + f" | hourly parse failed: {type(e).__name__}: {e}"

    # Daily max (fallback)
    try:
        daily = payload.get("daily") or {}
        dtime = daily.get("time")
        dmax = daily.get("temperature_2m_max")
        if (daily_high is None) and dtime is not None and dmax is not None and len(dtime) == len(dmax) and len(dtime) > 0:
            ddf = pd.DataFrame({"date": dtime, "tmax": dmax})
            ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce").dt.date
            today_local = now_date_in_tz(tz).date()
            row = ddf[ddf["date"] == today_local]
            if len(row) > 0:
                daily_high = float(row.iloc[0]["tmax"])
    except Exception as e:
        err = (err or "") + f" | daily parse failed: {type(e).__name__}: {e}"

    # Current weather
    current_temp = None
    try:
        cw = payload.get("current_weather") or {}
        # Open-Meteo uses "temperature" in selected unit
        if "temperature" in cw:
            current_temp = float(cw["temperature"])
    except Exception:
        pass

    return SourceResult(
        name=f"Open-Meteo ({model})",
        daily_high_f=daily_high,
        peak_time_local=peak_time,
        current_temp_f=current_temp,
        hourly_df=hourly_df,
        err=err,
    )


def fetch_nws_hourly(lat: float, lon: float, tz: str) -> SourceResult:
    # Step 1: points endpoint
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    points, err = safe_get_json(points_url)

    if points is None:
        return SourceResult(
            name="NWS",
            daily_high_f=None,
            peak_time_local=None,
            current_temp_f=None,
            hourly_df=None,
            err=err,
        )

    props = points.get("properties") or {}
    hourly_url = props.get("forecastHourly")

    if not hourly_url:
        return SourceResult(
            name="NWS",
            daily_high_f=None,
            peak_time_local=None,
            current_temp_f=None,
            hourly_df=None,
            err="NWS points response missing forecastHourly URL.",
        )

    hourly_payload, err2 = safe_get_json(hourly_url)
    if hourly_payload is None:
        return SourceResult(
            name="NWS",
            daily_high_f=None,
            peak_time_local=None,
            current_temp_f=None,
            hourly_df=None,
            err=err2,
        )

    periods = (hourly_payload.get("properties") or {}).get("periods") or []
    if not periods:
        return SourceResult(
            name="NWS",
            daily_high_f=None,
            peak_time_local=None,
            current_temp_f=None,
            hourly_df=None,
            err="NWS hourly periods empty.",
        )

    # Build df
    try:
        df = pd.DataFrame(periods)
        # temperature is already in °F usually
        if "startTime" not in df.columns or "temperature" not in df.columns:
            return SourceResult("NWS", None, None, None, None, "NWS hourly missing startTime/temperature fields.")

        df["dt"] = to_local_dt(df["startTime"], tz)
        df["temp_f"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["dt", "temp_f"])

        today = now_date_in_tz(tz).date()
        df_today = df[df["dt"].dt.date == today].copy()

        if len(df_today) == 0:
            # If early morning UTC confusion etc, use next 24h as fallback
            df_today = df.iloc[:24].copy()

        idx = df_today["temp_f"].idxmax()
        daily_high = float(df_today.loc[idx, "temp_f"])
        peak_time = pd.to_datetime(df_today.loc[idx, "dt"], errors="coerce")

        # Current temp: first period should be "now-ish"
        current_temp = float(df.iloc[0]["temp_f"]) if len(df) else None

        hourly_df = df_today[["dt", "temp_f"]].copy()
        return SourceResult("NWS", daily_high, peak_time, current_temp, hourly_df, None)
    except Exception as e:
        return SourceResult("NWS", None, None, None, None, f"NWS parse failed: {type(e).__name__}: {e}")


# =========================
# Kalshi bracket utilities
# =========================
def bracket_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi}"


def make_brackets_around(value: float, size: int, odd_start: bool) -> Tuple[int, int]:
    """
    Returns the bracket [lo, hi] (inclusive bounds in integers) containing value
    using 1°,2°,3°,4° etc. For 2°:
      odd_start=True  -> 83–84, 85–86 ...
      odd_start=False -> 84–85, 86–87 ...
    """
    v = int(math.floor(value))
    if size <= 1:
        return v, v

    if size == 2:
        if odd_start:
            # starts at odd
            # find nearest odd <= v
            base = v if (v % 2 == 1) else v - 1
        else:
            # starts at even
            base = v if (v % 2 == 0) else v - 1
        return base, base + 1

    # Generic size: align to multiples of size
    base = (v // size) * size
    return base, base + (size - 1)


def ladder_rows(mu: float, sigma: float, size: int, odd_start: bool, n: int = 7) -> pd.DataFrame:
    lo, hi = make_brackets_around(mu, size, odd_start)
    # center bracket index
    center_lo = lo

    # build n brackets around center
    half = n // 2
    rows = []
    for k in range(-half, half + 1):
        b_lo = center_lo + k * size
        b_hi = b_lo + (size - 1)
        # use continuous bounds with inclusive integer interpretation
        p = normal_prob_between(b_lo - 0.5, b_hi + 0.5, mu, sigma) * 100.0
        rows.append({"Bracket": bracket_label(b_lo, b_hi), "Probability %": round(p, 1)})
    return pd.DataFrame(rows)


# =========================
# Peak spike detector
# =========================
def spike_detector(hourly_df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    """
    Simple check: if temps rise quickly within 2 hours before peak, flag.
    """
    if hourly_df is None or len(hourly_df) < 6:
        return False, "Not enough hourly data."

    df = hourly_df.sort_values("dt").copy()
    df["temp_f"] = pd.to_numeric(df["temp_f"], errors="coerce")
    df = df.dropna(subset=["dt", "temp_f"])
    if len(df) < 6:
        return False, "Not enough hourly data."

    peak_idx = df["temp_f"].idxmax()
    peak_dt = df.loc[peak_idx, "dt"]

    window = df[(df["dt"] >= peak_dt - pd.Timedelta(hours=2)) & (df["dt"] <= peak_dt)].copy()
    if len(window) < 3:
        return False, "No 2-hour window available."

    rise = float(window["temp_f"].iloc[-1] - window["temp_f"].iloc[0])
    if rise >= 4.0:
        return True, f"Fast rise of ~{rise:.1f}°F in the 2 hours before peak."
    return False, f"Normal ramp (~{rise:.1f}°F over 2 hours)."


# =========================
# UI
# =========================
st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox("Select City", list(CITIES.keys()), index=list(CITIES.keys()).index(DEFAULT_CITY) if DEFAULT_CITY in CITIES else 0)

bracket_size = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4], index=1)

grace_minutes = st.slider("Grace Minutes Around Peak", 0, 90, 30)

use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
st.caption("If this ever fails, the dashboard ignores it automatically. (This prevents GFS errors from breaking cities.)")

lat = float(CITIES[city]["lat"])
lon = float(CITIES[city]["lon"])
tz = str(CITIES[city]["tz"])
today_local = now_date_in_tz(tz).date()

# Fetch sources
sources: List[SourceResult] = []
errs: List[str] = []

om_best = fetch_open_meteo(lat, lon, tz, model="best")
sources.append(om_best)
if om_best.err:
    errs.append(f"Open-Meteo (best) failed: {om_best.err}")

nws = fetch_nws_hourly(lat, lon, tz)
sources.append(nws)
if nws.err:
    errs.append(f"NWS failed: {nws.err}")

om_gfs = None
if use_gfs:
    om_gfs = fetch_open_meteo(lat, lon, tz, model="gfs")
    sources.append(om_gfs)
    if om_gfs.err:
        errs.append(f"Open-Meteo (GFS) failed: {om_gfs.err}")

# Determine "best model used" for prediction:
# Preference: Open-Meteo(best) daily high if present, otherwise NWS, otherwise GFS, otherwise None.
best_source = None
for preferred in ["Open-Meteo (best)", "NWS", "Open-Meteo (gfs)"]:
    for s in sources:
        if s.name.lower().startswith(preferred.lower().split("(")[0].lower()) and s.daily_high_f is not None:
            # more exact match:
            if preferred == "Open-Meteo (best)" and s.name == "Open-Meteo (best)":
                best_source = s
                break
            if preferred == "NWS" and s.name == "NWS":
                best_source = s
                break
            if preferred == "Open-Meteo (gfs)" and s.name == "Open-Meteo (gfs)":
                best_source = s
                break
    if best_source is not None:
        break

if best_source is None:
    # fallback to any source with a daily high
    for s in sources:
        if s.daily_high_f is not None:
            best_source = s
            break

# Compute agreement / spread
highs = [s.daily_high_f for s in sources if s.daily_high_f is not None]
spread = (max(highs) - min(highs)) if len(highs) >= 2 else None

if spread is None:
    confidence = "Unknown"
else:
    # simple bucket
    if spread <= 1.0:
        confidence = f"High (spread {spread:.1f}°)"
    elif spread <= 2.0:
        confidence = f"Medium (spread {spread:.1f}°)"
    else:
        confidence = f"Low (spread {spread:.1f}°)"

# Summary header
st.caption(
    "Sources: "
    + " + ".join(
        [
            "Open-Meteo",
            "National Weather Service (NWS)",
            "Open-Meteo GFS" if use_gfs else "",
        ]
    ).replace(" + ", " + ").strip(" +")
)

if len(errs) > 0:
    st.warning("Some sources failed. The dashboard will use whatever data is available.")
    with st.expander("See errors"):
        for e in errs:
            st.write("• " + e)

# Display city section
st.header(city)

pred_high = best_source.daily_high_f if best_source else None
st.subheader("Predicted Daily High (°F)")
st.markdown(f"## {fmt_temp(pred_high)}")

st.subheader("Confidence")
st.markdown(f"## {confidence}")

# Peak time & window
peak_dt = best_source.peak_time_local if best_source else None
st.subheader("Estimated Peak Time")
st.markdown(f"## {fmt_time(peak_dt)}")

# Peak window based on grace minutes
if peak_dt is not None and not pd.isna(peak_dt):
    start = peak_dt - pd.Timedelta(minutes=int(grace_minutes))
    end = peak_dt + pd.Timedelta(minutes=int(grace_minutes))
    st.write(f"**Peak window:** {fmt_time(start)} – {fmt_time(end)}")

# Suggested Kalshi range (show BOTH alignments, default to odd-start for 2°)
if pred_high is not None:
    if bracket_size == 2:
        # Primary (odd-start) matches 83–84,85–86...
        lo1, hi1 = make_brackets_around(pred_high, 2, odd_start=True)
        lo2, hi2 = make_brackets_around(pred_high, 2, odd_start=False)

        st.subheader("Suggested Kalshi Range (Daily High)")
        st.markdown(f"## {lo1}–{hi1}°F")
        st.caption("Bracket interpretation: 2° odd-start (e.g., 83–84, 85–86) — matches most Kalshi cards.")

        st.write("Also check this alternate 2° alignment (Kalshi sometimes uses this):")
        st.write(f"• **{lo2}–{hi2}°F** — 2° even-start (e.g., 84–85, 86–87)")
    else:
        lo, hi = make_brackets_around(pred_high, bracket_size, odd_start=False)
        st.subheader("Suggested Kalshi Range (Daily High)")
        st.markdown(f"## {lo}–{hi}°F")

# Nearby ranges to watch
if pred_high is not None:
    st.write("**Nearby ranges to watch:**")
    if bracket_size == 2:
        lo, hi = make_brackets_around(pred_high, 2, odd_start=True)
        for k in [-1, 0, 1]:
            b_lo = lo + k * 2
            b_hi = b_lo + 1
            if k == 0:
                st.write(f"• {b_lo}–{b_hi} (current)")
            else:
                st.write(f"• {b_lo}–{b_hi}")
    else:
        lo, hi = make_brackets_around(pred_high, bracket_size, odd_start=False)
        for k in [-1, 0, 1]:
            b_lo = lo + k * bracket_size
            b_hi = b_lo + (bracket_size - 1)
            if k == 0:
                st.write(f"• {b_lo}–{b_hi} (current)")
            else:
                st.write(f"• {b_lo}–{b_hi}")

# Raw numbers expander
with st.expander("See raw forecast numbers"):
    st.write(f"Forecast date (today): **{today_local}**")
    st.write(f"Best model used: **{best_source.name if best_source else '—'}**")
    for s in sources:
        st.write(f"- **{s.name}**: high={fmt_temp(s.daily_high_f)}°F, peak={fmt_time(s.peak_time_local)}, current={fmt_temp(s.current_temp_f)}°F")

# Current conditions (Open-Meteo best if available)
st.subheader("Current Conditions (Open-Meteo)")
st.write(f"Temp (°F): **{fmt_temp(om_best.current_temp_f)}**")

# Model agreement table
st.subheader("Model Agreement (Source Highs)")
rows = []
for s in sources:
    if s.daily_high_f is not None:
        rows.append(
            {
                "Source": s.name,
                "Daily High (°F)": round(float(s.daily_high_f), 1),
                "Peak Time": fmt_time(s.peak_time_local),
            }
        )
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No source provided a usable daily high right now.")

# Probability ladder (uses mu = predicted high, sigma based on spread)
if pred_high is not None:
    st.subheader("Kalshi Probability Ladder")
    mu = float(pred_high)
    # sigma heuristic: spread/2, with a sensible floor so it’s never “0.0 spread” fake certainty
    if spread is None:
        sigma = 1.6
    else:
        sigma = max(1.2, float(spread) / 2.0)

    if bracket_size == 2:
        ladder = ladder_rows(mu, sigma, 2, odd_start=True, n=7)  # match Kalshi odd-start
    else:
        ladder = ladder_rows(mu, sigma, bracket_size, odd_start=False, n=7)

    st.dataframe(ladder, use_container_width=True)

# Value bet check (user inputs Kalshi price)
st.subheader("Value Bet Check (you enter the Kalshi price)")
st.caption("This is a rough, model-based check. Use it to compare the market price vs your estimated probability.")
price_cents = st.number_input("Enter Kalshi YES price for the main bracket (cents)", min_value=0, max_value=100, value=50, step=1)

if pred_high is not None:
    mu = float(pred_high)
    if spread is None:
        sigma = 1.6
    else:
        sigma = max(1.2, float(spread) / 2.0)

    # choose primary bracket the dashboard suggested
    if bracket_size == 2:
        lo, hi = make_brackets_around(mu, 2, odd_start=True)
    else:
        lo, hi = make_brackets_around(mu, bracket_size, odd_start=False)

    p = normal_prob_between(lo - 0.5, hi + 0.5, mu, sigma)
    model_yes = p
    market_yes = clamp(price_cents / 100.0, 0.0, 1.0)

    # Edge: expected value per $1 (very simplified, ignores fees/spread)
    # If YES costs market_yes and pays 1 if event occurs.
    ev = model_yes * 1.0 - market_yes

    st.write(f"Model probability for **{lo}–{hi}°F** ≈ **{model_yes*100:.1f}%** (sigma≈{sigma:.1f}°)")
    st.write(f"Market implied YES ≈ **{market_yes*100:.1f}%** (from {price_cents}¢)")
    st.write(f"Simple edge (model − market) ≈ **{ev*100:.1f}%**")
    if ev > 0.03:
        st.success("Model shows positive edge (roughly).")
    elif ev < -0.03:
        st.error("Model shows negative edge (roughly).")
    else:
        st.warning("Edge is small / unclear.")

# Peak-time heat spike detector
st.subheader("Peak-time heat spike detector")
best_hourly = None
if best_source and best_source.hourly_df is not None:
    best_hourly = best_source.hourly_df
elif om_best.hourly_df is not None:
    best_hourly = om_best.hourly_df
elif nws.hourly_df is not None:
    best_hourly = nws.hourly_df

flag, msg = spike_detector(best_hourly)
if flag:
    st.warning(f"Potential spike risk: {msg}")
else:
    st.write(msg)
