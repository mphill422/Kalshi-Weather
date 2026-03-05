import math
import re
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ============================
# Config
# ============================
st.set_page_config(page_title="Kalshi Weather Model (Ladder-Aligned)", layout="wide")

APP_TITLE = "Kalshi Weather Model – Ladder-Aligned (Daily High)"
st.title(APP_TITLE)

USER_AGENT = "kalshi-weather-model/4.2 (contact: none)"
REQ_TIMEOUT = 12

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

DEFAULT_LOCK_LOCAL = time(10, 30)

# ============================
# Helpers
# ============================
def http_get_json(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e), "__url__": url}

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def today_local(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()

def parse_iso_to_local_dt(ts: str, tz_name: str) -> datetime | None:
    # Open-Meteo returns ISO without offset when timezone= is set
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def prob_between(mean: float, sigma: float, lo: float, hi: float) -> float:
    """P(lo <= X < hi) for Normal(mean, sigma). lo/hi can be +/-inf."""
    if sigma <= 1e-12:
        return 1.0 if (lo <= mean < hi) else 0.0
    if lo == float("-inf"):
        z = (hi - mean) / sigma
        return max(0.0, min(1.0, normal_cdf(z)))
    if hi == float("inf"):
        z = (lo - mean) / sigma
        return max(0.0, min(1.0, 1.0 - normal_cdf(z)))
    z1 = (lo - mean) / sigma
    z2 = (hi - mean) / sigma
    return max(0.0, min(1.0, normal_cdf(z2) - normal_cdf(z1)))

def compute_sigma(source_highs: list[float]) -> float:
    """
    Conservative uncertainty:
    - base 1.7°F
    - + 0.6 * cross-source spread
    """
    if not source_highs:
        return 2.8
    spread = (max(source_highs) - min(source_highs)) if len(source_highs) >= 2 else 0.0
    return max(1.6, 1.7 + 0.60 * spread)

# ============================
# Forecast Sources
# ============================
def fetch_open_meteo(lat: float, lon: float, tz: str, model: str | None = None) -> dict:
    base = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&temperature_unit=fahrenheit"
        f"&hourly=temperature_2m"
        f"&daily=temperature_2m_max"
        f"&current_weather=true"
        f"&timezone={tz}"
        f"&forecast_days=2"
    )
    if model:
        base += f"&models={model}"
    return http_get_json(base)

def extract_open_meteo_today(j: dict, tz: str):
    if not j or "__error__" in j:
        return None, None, None, (j.get("__error__") if isinstance(j, dict) else "Unknown error")

    try:
        # Daily high
        daily = j.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        dts = daily.get("time", [])
        if not highs or not dts:
            return None, None, None, "Open-Meteo missing daily fields"
        t0 = today_local(tz).isoformat()
        idx = dts.index(t0) if t0 in dts else 0
        daily_high = safe_float(highs[idx])

        # Current
        cur = j.get("current_weather") or {}
        current_temp = safe_float(cur.get("temperature"))

        # Hourly for today
        hourly = j.get("hourly", {})
        ht = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        df = None
        if ht and temps and len(ht) == len(temps):
            rows = []
            for ts, temp in zip(ht, temps):
                dt = parse_iso_to_local_dt(ts, tz)
                if dt is None:
                    continue
                rows.append({"dt": dt, "temp_f": safe_float(temp)})
            df = pd.DataFrame(rows).dropna()
            if not df.empty:
                tday = today_local(tz)
                df = df[df["dt"].dt.date == tday].copy()
                if df.empty:
                    df = None

        return daily_high, df, current_temp, None
    except Exception as e:
        return None, None, None, f"Open-Meteo parse error: {e}"

def fetch_nws_hourly_high(lat: float, lon: float, tz: str):
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        p = http_get_json(points_url)
        if not p or "__error__" in p:
            return None, None, f"NWS points error: {p.get('__error__')}"
        hourly_url = (p.get("properties") or {}).get("forecastHourly")
        if not hourly_url:
            return None, None, "NWS missing forecastHourly"

        h = http_get_json(hourly_url)
        if not h or "__error__" in h:
            return None, None, f"NWS hourly error: {h.get('__error__')}"

        periods = ((h.get("properties") or {}).get("periods") or [])
        if not periods:
            return None, None, "NWS hourly periods empty"

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
            return None, None, "NWS parse produced no rows"

        tday = today_local(tz)
        df_today = df[df["dt"].dt.date == tday].copy()
        if df_today.empty:
            return None, None, "NWS hourly has no rows for today"

        daily_high = float(df_today["temp_f"].max())
        return daily_high, df_today, None
    except Exception as e:
        return None, None, f"NWS error: {e}"

# ============================
# Kalshi ladder parsing (KEY FIX)
# ============================
RANGE_RE = re.compile(r"(-?\d+)\s*(?:°)?\s*(?:to|\-)\s*(-?\d+)", re.IGNORECASE)
BELOW_RE = re.compile(r"(-?\d+)\s*(?:°)?\s*(?:or\s+below|and\s+below|≤)", re.IGNORECASE)
ABOVE_RE = re.compile(r"(-?\d+)\s*(?:°)?\s*(?:or\s+above|and\s+above|≥)", re.IGNORECASE)

def contract_to_interval(label: str):
    """
    Convert Kalshi label to a continuous interval [lo, hi)
    using integer-degree settlement logic.

    Examples:
      "84° to 85°"  -> [84, 86)
      "77 or below" -> (-inf, 78)
      "86 or above" -> [86, +inf)
    """
    s = label.strip().replace("º", "°")

    m = RANGE_RE.search(s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        lo = min(a, b)
        hi_inclusive = max(a, b)
        return float(lo), float(hi_inclusive + 1), "range"

    m = BELOW_RE.search(s)
    if m:
        x = int(m.group(1))
        return float("-inf"), float(x + 1), "below"

    m = ABOVE_RE.search(s)
    if m:
        x = int(m.group(1))
        return float(x), float("inf"), "above"

    return None, None, "unknown"

def parse_kalshi_contracts(text: str):
    """
    Accepts lines like:
      84° to 85°
      78° to 79°
      77° or below
      86° or above
    Also tolerates extra whitespace/characters.
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    contracts = []
    for ln in lines:
        lo, hi, kind = contract_to_interval(ln)
        if lo is None:
            continue
        contracts.append({"contract": ln, "lo": lo, "hi": hi, "kind": kind})

    # Remove duplicates preserving order
    seen = set()
    out = []
    for c in contracts:
        key = (c["lo"], c["hi"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    # Sort by lo for display
    out_sorted = sorted(out, key=lambda x: (x["lo"] if x["lo"] != float("-inf") else -1e9))
    return out_sorted

# ============================
# UI
# ============================
top_left, top_right = st.columns([1, 1])
with top_left:
    city = st.selectbox("City", list(CITIES.keys()))
with top_right:
    st.caption("This version aligns probabilities to the exact Kalshi contract ladder you paste (no mismatch).")

info = CITIES[city]
tz = info["tz"]

with st.expander("Settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_gfs = st.toggle("Include Open-Meteo GFS (extra check)", value=False)
    with c2:
        use_nws = st.toggle("Include NWS (api.weather.gov)", value=True)
    with c3:
        grace_minutes = st.slider("Grace minutes after 10:30 local", 0, 180, 80, 5)
    with c4:
        show_hourly = st.toggle("Show hourly chart", value=True)

    st.markdown("### Kalshi ladder mode (fixes bracket mismatch)")
    use_kalshi_ladder = st.toggle("Use pasted Kalshi contract list (recommended)", value=True)

    default_example = (
        "84° to 85°\n"
        "78° to 79°\n"
        "80° to 81°\n"
        "82° to 83°\n"
        "77° or below\n"
        "86° or above\n"
    )
    kalshi_text = st.text_area(
        "Paste Kalshi market contracts (one per line). Example format:",
        value="",
        height=140,
        placeholder=default_example,
    )

# ============================
# Fetch + Compute
# ============================
with st.spinner("Fetching forecasts…"):
    om = fetch_open_meteo(info["lat"], info["lon"], tz, model=None)
    om_high, om_hourly, om_current, om_err = extract_open_meteo_today(om, tz)

    gfs_high = None
    gfs_hourly = None
    gfs_current = None
    gfs_err = None
    if use_gfs:
        om_gfs = fetch_open_meteo(info["lat"], info["lon"], tz, model="gfs_seamless")
        gfs_high, gfs_hourly, gfs_current, gfs_err = extract_open_meteo_today(om_gfs, tz)

    nws_high = None
    nws_hourly = None
    nws_err = None
    if use_nws:
        nws_high, nws_hourly, nws_err = fetch_nws_hourly_high(info["lat"], info["lon"], tz)

# Source status table
status_rows = []
if om_high is not None:
    status_rows.append(("Open-Meteo", f"{om_high:.1f}°F", "OK"))
else:
    status_rows.append(("Open-Meteo", "—", om_err or "Error"))

if use_gfs:
    if gfs_high is not None:
        status_rows.append(("Open-Meteo (GFS)", f"{gfs_high:.1f}°F", "OK"))
    else:
        status_rows.append(("Open-Meteo (GFS)", "—", gfs_err or "Error"))

if use_nws:
    if nws_high is not None:
        status_rows.append(("NWS (forecastHourly)", f"{nws_high:.1f}°F", "OK"))
    else:
        status_rows.append(("NWS (forecastHourly)", "—", nws_err or "Error"))

st.subheader(f"{city} – Today’s High Forecasts (°F)")
st.table(pd.DataFrame(status_rows, columns=["Source", "Today High", "Status"]))

source_highs = [x for x in [om_high, gfs_high, nws_high] if isinstance(x, (int, float)) and x is not None]
if not source_highs:
    st.error("No valid sources returned a high for today. Toggle off a failing source or try again.")
    st.stop()

consensus = float(sum(source_highs) / len(source_highs))
sigma = compute_sigma(source_highs)
spread = (max(source_highs) - min(source_highs)) if len(source_highs) > 1 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Consensus high", f"{consensus:.1f}°F")
m2.metric("Cross-source spread", f"{spread:.1f}°F")
m3.metric("Model uncertainty (σ)", f"{sigma:.2f}°F")
if om_current is not None:
    m4.metric("Current temp (Open-Meteo)", f"{om_current:.1f}°F")
else:
    m4.metric("Current temp (Open-Meteo)", "—")

# ============================
# Kalshi ladder-aligned probabilities (MAIN FIX)
# ============================
st.subheader("Suggested Kalshi Bracket")

contracts = []
if use_kalshi_ladder:
    contracts = parse_kalshi_contracts(kalshi_text)

    if not contracts:
        st.warning("Paste the Kalshi contract lines to enable ladder-aligned suggestions (recommended).")
    else:
        # Compute exact contract probabilities
        rows = []
        for c in contracts:
            p = prob_between(consensus, sigma, c["lo"], c["hi"])
            rows.append({
                "Bracket": c["contract"],
                "Model Prob %": round(p * 100, 1),
                "lo": c["lo"],
                "hi": c["hi"]
            })

        dfp = pd.DataFrame(rows).sort_values("Model Prob %", ascending=False).reset_index(drop=True)
        best = dfp.iloc[0]
        st.success(f"Suggested bracket: **{best['Bracket']}**  (model ≈ **{best['Model Prob %']:.1f}%**)")

        st.caption("Model probabilities for the exact Kalshi contracts you pasted:")
        st.dataframe(dfp[["Bracket", "Model Prob %"]], use_container_width=True, hide_index=True)

        with st.expander("Debug: how the model interprets your Kalshi contracts"):
            dbg = pd.DataFrame([{
                "Contract": c["contract"],
                "Interval used": f"[{c['lo']}, {c['hi']})",
                "Type": c["kind"]
            } for c in contracts])
            st.dataframe(dbg, use_container_width=True, hide_index=True)

else:
    # Fallback: generic 2-degree bins around consensus
    bracket_size = 2
    center = int(round(consensus))
    candidates = []
    for low in range(center - 8, center + 9):
        high = low + bracket_size
        p = prob_between(consensus, sigma, float(low), float(high))
        candidates.append((f"{low}–{high-1}", p))
    candidates.sort(key=lambda x: x[1], reverse=True)
    st.success(f"Suggested bracket: **{candidates[0][0]}**  (model ≈ **{candidates[0][1]*100:.1f}%**)")
    st.dataframe(pd.DataFrame([{"Bracket": b, "Model Prob %": round(p*100,1)} for b,p in candidates[:8]]),
                 use_container_width=True, hide_index=True)

# ============================
# Decision window
# ============================
now_local = datetime.now(ZoneInfo(tz))
lock_dt = datetime.combine(now_local.date(), DEFAULT_LOCK_LOCAL, tzinfo=ZoneInfo(tz))
deadline_dt = lock_dt + timedelta(minutes=int(grace_minutes))

st.subheader("Decision Window")
st.write(
    f"Local time now: **{now_local.strftime('%a %b %d, %I:%M %p')}**  "
    f"| Target check: **10:30 AM**  "
    f"| With grace: **{deadline_dt.strftime('%I:%M %p')}**"
)
if now_local <= deadline_dt:
    st.info("You’re inside the preferred betting window (or within grace).")
else:
    st.warning("Past the preferred window. Market is often sharper later in the day.")

# ============================
# Hourly curve (optional)
# ============================
if show_hourly:
    st.subheader("Hourly temperature curve (today)")
    hourly_df = None
    src = None

    # Prefer Open-Meteo hourly, else NWS hourly
    if om_hourly is not None and not om_hourly.empty:
        hourly_df = om_hourly.sort_values("dt")
        src = "Open-Meteo"
    elif nws_hourly is not None and not nws_hourly.empty:
        hourly_df = nws_hourly.sort_values("dt")
        src = "NWS"

    if hourly_df is not None and not hourly_df.empty:
        st.caption(f"Hourly source used: {src}")
        st.line_chart(hourly_df.set_index("dt")["temp_f"])

        peak_row = hourly_df.loc[hourly_df["temp_f"].idxmax()]
        st.write(f"Peak hour: **{peak_row['dt'].strftime('%I:%M %p')}** at **{float(peak_row['temp_f']):.1f}°F**")
    else:
        st.caption("Hourly curve unavailable for today from the enabled sources.")

st.divider()
st.caption(
    "Key fix: the app now computes probabilities on the exact Kalshi contracts you paste, "
    "so it won’t invent a bracket like 81–82 if Kalshi doesn’t offer it."
)
