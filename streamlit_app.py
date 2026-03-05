# app.py
# Kalshi Weather Trading Dashboard v2 (all-in-one upgrade build)
# Streamlit app with Open-Meteo + NWS (+ optional GFS), station mapping, daytime peak fix,
# weighted ensemble, probabilistic bracket ladder, nowcast/trajectory, and value-bet edge checks.

import math
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List

import requests
import streamlit as st

# ----------------------------
# Helpers: math / stats
# ----------------------------

def norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal CDF using erf (no scipy dependency)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def bracket_prob(lower: float, upper: float, mu: float, sigma: float) -> float:
    """P(lower <= X <= upper) for Normal(mu, sigma). Inclusive-ish."""
    # Use CDF(upper+0.5*eps) - CDF(lower-0.5*eps) for stability with integers if desired.
    return max(0.0, min(1.0, norm_cdf(upper, mu, sigma) - norm_cdf(lower, mu, sigma)))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ----------------------------
# City / station config
# ----------------------------

@dataclass
class CityConfig:
    name: str
    state: str
    station: str              # ASOS/airport code typically used for settlement
    lat: float
    lon: float
    tz: str                   # IANA timezone string for Open-Meteo timezone param (e.g., "America/Chicago")
    peak_start_hour: int = 10 # local hour
    peak_end_hour: int = 18   # local hour

# NOTE: Verify Kalshi settlement stations for your specific markets.
# These are common airport stations. Adjust as needed.
CITIES: Dict[str, CityConfig] = {
    "Austin, TX": CityConfig("Austin", "TX", "KAUS", 30.1945, -97.6699, "America/Chicago"),
    "Dallas, TX": CityConfig("Dallas", "TX", "KDFW", 32.8998, -97.0403, "America/Chicago"),
    "Houston, TX": CityConfig("Houston", "TX", "KIAH", 29.9844, -95.3414, "America/Chicago"),
    "Phoenix, AZ": CityConfig("Phoenix", "AZ", "KPHX", 33.4342, -112.0116, "America/Phoenix"),
    "Las Vegas, NV": CityConfig("Las Vegas", "NV", "KLAS", 36.0801, -115.1522, "America/Los_Angeles"),
    "Los Angeles, CA": CityConfig("Los Angeles", "CA", "KLAX", 33.9425, -118.4081, "America/Los_Angeles"),
}

# ----------------------------
# Data fetchers
# ----------------------------

USER_AGENT = "kalshi-weather-dashboard/2.0 (streamlit)"

def safe_get(url: str, params: Optional[dict] = None, timeout: int = 12) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r
        return None
    except Exception:
        return None

def fetch_open_meteo_hourly(city: CityConfig, model: str = "best") -> Optional[dict]:
    """
    Open-Meteo hourly temps for today.
    model: "best" (their blend) or "gfs"
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m,precipitation",
        "forecast_days": 2,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": city.tz,
    }
    # Open-Meteo supports selecting models via "models" parameter.
    # If unavailable for your plan/endpoint, it will ignore.
    if model.lower() in ("gfs",):
        params["models"] = "gfs"
    else:
        # "best" is default; you can also request specific like "hrrr", "nam", etc. depending on availability.
        # We'll leave as default for broad compatibility.
        pass

    r = safe_get(base, params=params)
    return r.json() if r else None

def fetch_nws_hourly_forecast(city: CityConfig) -> Optional[dict]:
    """
    NWS requires:
    1) points endpoint -> forecastHourly URL
    2) fetch forecastHourly JSON
    """
    points_url = f"https://api.weather.gov/points/{city.lat},{city.lon}"
    r1 = safe_get(points_url)
    if not r1:
        return None
    j1 = r1.json()
    forecast_hourly_url = j1.get("properties", {}).get("forecastHourly")
    if not forecast_hourly_url:
        return None
    r2 = safe_get(forecast_hourly_url)
    return r2.json() if r2 else None

# ----------------------------
# Parsing / feature engineering
# ----------------------------

def parse_open_meteo_hourly(json_data: dict) -> Optional[List[dict]]:
    try:
        h = json_data["hourly"]
        times = h["time"]
        temps = h["temperature_2m"]
        clouds = h.get("cloud_cover", [None] * len(times))
        winds = h.get("wind_speed_10m", [None] * len(times))
        precs = h.get("precipitation", [None] * len(times))
        rows = []
        for i in range(len(times)):
            # times are already localized if timezone param is used; they come as ISO strings without TZ offset.
            t = datetime.fromisoformat(times[i])
            rows.append({
                "time": t,
                "temp_f": float(temps[i]),
                "cloud": None if clouds[i] is None else float(clouds[i]),
                "wind_mph": None if winds[i] is None else float(winds[i]),
                "precip_in": None if precs[i] is None else float(precs[i]),
                "source": "open_meteo",
            })
        return rows
    except Exception:
        return None

def parse_nws_hourly(json_data: dict) -> Optional[List[dict]]:
    try:
        periods = json_data["properties"]["periods"]
        rows = []
        for p in periods:
            # startTime includes offset like 2026-03-05T13:00:00-06:00
            t = datetime.fromisoformat(p["startTime"])
            # NWS temps are in F by default
            rows.append({
                "time": t.replace(tzinfo=None),  # we will treat as local wall time for comparisons
                "temp_f": float(p["temperature"]),
                "wind_mph": None,
                "cloud": None,
                "precip_in": None,
                "source": "nws",
            })
        return rows
    except Exception:
        return None

def filter_today(rows: List[dict], local_today: datetime) -> List[dict]:
    """Keep only rows matching local_today date."""
    d = local_today.date()
    return [r for r in rows if r["time"].date() == d]

def daytime_peak(rows: List[dict], start_hour: int, end_hour: int) -> Tuple[Optional[float], Optional[datetime]]:
    """Return (max_temp, time_of_max) within [start_hour, end_hour]. Fixes '11 PM peak' bug."""
    daytime = [r for r in rows if start_hour <= r["time"].hour <= end_hour]
    if not daytime:
        return None, None
    mx = max(daytime, key=lambda r: r["temp_f"])
    return mx["temp_f"], mx["time"]

def get_current_temp_from_rows(rows: List[dict], now_local: datetime) -> Optional[float]:
    """
    Approximate "current" temp from hourly list by finding the closest hour <= now.
    """
    past = [r for r in rows if r["time"] <= now_local]
    if not past:
        return None
    closest = max(past, key=lambda r: r["time"])
    return closest["temp_f"]

def heating_rate(rows: List[dict], now_local: datetime, lookback_minutes: int = 90) -> Optional[float]:
    """
    Heating rate (F/hr) from last lookback window. Uses closest points.
    """
    cutoff = now_local - timedelta(minutes=lookback_minutes)
    pts = [r for r in rows if cutoff <= r["time"] <= now_local]
    pts = sorted(pts, key=lambda r: r["time"])
    if len(pts) < 2:
        return None
    t0, t1 = pts[0], pts[-1]
    dt_hours = (t1["time"] - t0["time"]).total_seconds() / 3600.0
    if dt_hours <= 0:
        return None
    return (t1["temp_f"] - t0["temp_f"]) / dt_hours

# ----------------------------
# Ensemble + distribution logic
# ----------------------------

def weighted_ensemble(open_high: Optional[float], nws_high: Optional[float], gfs_high: Optional[float],
                      w_open: float = 0.55, w_nws: float = 0.30, w_gfs: float = 0.15) -> Optional[float]:
    vals = []
    wts = []
    if open_high is not None:
        vals.append(open_high); wts.append(w_open)
    if nws_high is not None:
        vals.append(nws_high); wts.append(w_nws)
    if gfs_high is not None:
        vals.append(gfs_high); wts.append(w_gfs)
    if not vals:
        return None
    sw = sum(wts)
    if sw <= 0:
        return sum(vals) / len(vals)
    return sum(v * w for v, w in zip(vals, wts)) / sw

def estimate_sigma(spread: Optional[float], fallback_sigma: float = 2.2) -> float:
    """
    Convert model disagreement spread into sigma.
    Heuristic: sigma ~ spread/3 (spread is roughly ~3σ in many cases).
    Clamp for sanity.
    """
    if spread is None:
        return fallback_sigma
    sig = max(fallback_sigma, spread / 3.0)
    return clamp(sig, 1.0, 6.0)

def compute_spread(*highs: Optional[float]) -> Optional[float]:
    vals = [h for h in highs if h is not None]
    if len(vals) < 2:
        return None
    return max(vals) - min(vals)

def nowcast_adjustment(mu_model: float,
                       current_temp: Optional[float],
                       max_so_far: Optional[float],
                       rate_f_per_hr: Optional[float],
                       now_local: datetime,
                       peak_time_local: Optional[datetime],
                       cap_rate: float = 6.0,
                       blend_nowcast: float = 0.30) -> Tuple[float, Optional[float]]:
    """
    Build a nowcast peak using heating rate and time-to-peak, then blend into mu.
    - cap_rate prevents crazy spikes
    - blend_nowcast controls how much nowcast shifts the mean
    """
    if current_temp is None or rate_f_per_hr is None or peak_time_local is None:
        return mu_model, None

    hours_to_peak = (peak_time_local - now_local).total_seconds() / 3600.0
    # If past peak time, no forward heating expected.
    if hours_to_peak <= 0:
        return mu_model, None

    rate = clamp(rate_f_per_hr, -2.0, cap_rate)
    projected = current_temp + rate * hours_to_peak

    # Respect observed max so far (daily high can't be below it)
    if max_so_far is not None:
        projected = max(projected, max_so_far)

    mu_final = (1 - blend_nowcast) * mu_model + blend_nowcast * projected
    return mu_final, projected

# ----------------------------
# Kalshi bracket logic
# ----------------------------

def make_brackets(center: int, step: int, n_each_side: int = 6) -> List[Tuple[int, int]]:
    """
    Create brackets like [center-step, center-step+1], ... depending on step size.
    For step=2, brackets are 2-degree bins: 76-77, 78-79, etc (inclusive endpoints for UI).
    """
    start = center - step * n_each_side
    br = []
    for i in range(2 * n_each_side + 1):
        lo = start + i * step
        hi = lo + (step - 1)
        br.append((lo, hi))
    return br

def primary_bracket(mu: float, step: int) -> Tuple[int, int]:
    """
    For step=2:
      "even-start" bins: 76-77, 78-79, ...
      "odd-start" bins: 75-76, 77-78, ...
    We'll compute both and let UI display alt alignment.
    """
    m = int(round(mu))
    # Even start: multiples of step (e.g., 2) align at even numbers.
    even_start = (m // step) * step
    return (even_start, even_start + step - 1)

def alternate_bracket(mu: float, step: int) -> Tuple[int, int]:
    m = int(round(mu))
    # Odd alignment shifts by 1
    base = (m // step) * step - 1
    return (base, base + step - 1)

def near_boundary_warning(mu: float, bracket_lo: int, bracket_hi: int, threshold: float = 0.7) -> bool:
    """
    Warn if mean is close to either bracket edge (within ~0.7°F).
    """
    return (abs(mu - bracket_lo) < threshold) or (abs(mu - bracket_hi) < threshold)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Kalshi Weather Trading Dashboard v2", layout="centered")
st.title("Kalshi Weather Trading Dashboard")

with st.sidebar:
    st.header("Controls")
    city_key = st.selectbox("Select City", list(CITIES.keys()), index=0)
    city = CITIES[city_key]

    step = st.selectbox("Kalshi bracket size (°F)", [1, 2, 3, 4], index=1)
    grace_minutes = st.slider("Grace Minutes Around Peak", min_value=0, max_value=90, value=30, step=5)

    use_gfs = st.toggle("Also try Open-Meteo GFS model (optional)", value=False)
    st.caption("If GFS fails, the dashboard ignores it automatically so it won’t break cities.")

    # Advanced knobs (optional)
    with st.expander("Advanced model settings"):
        w_open = st.slider("Weight: Open-Meteo", 0.0, 1.0, 0.55, 0.05)
        w_nws = st.slider("Weight: NWS", 0.0, 1.0, 0.30, 0.05)
