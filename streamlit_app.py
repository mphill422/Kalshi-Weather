
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Kalshi Weather Trading Dashboard", layout="centered")

st.title("Kalshi Weather Trading Dashboard")

# --- City list (you can add more later) ---
CITIES = {
    "Austin":  (30.2672, -97.7431),
    "Dallas":  (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
}

# Open-Meteo supports multiple forecast models. Using a small set gives you a simple "range".
# If a model isn't available for your location/time, we gracefully fall back.
MODELS = ["gfs_seamless", "ecmwf_ifs04", "gem_seamless"]  # keep as-is for now


def fetch_open_meteo(lat: float, lon: float, model: str | None = None) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        # hourly temps so we can find "peak time"
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        # daily max/min is handy for quick display
        "daily": "temperature_2m_max,temperature_2m_min",
        "forecast_days": 7,
    }
    if model:
        params["models"] = model

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def to_hourly_df(data: dict) -> pd.DataFrame:
    h = data.get("hourly", {})
    df = pd.DataFrame({
        "time": pd.to_datetime(h.get("time", [])),
        "temp_f": h.get("temperature_2m", []),
        "humidity": h.get("relative_humidity_2m", []),
        "wind_mph": h.get("wind_speed_10m", []),
    })
    if df.empty:
        return df
    df["date"] = df["time"].dt.date
    return df


def to_daily_df(data: dict) -> pd.DataFrame:
    d = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(d.get("time", [])).dt.date,
        "tmax_f": d.get("temperature_2m_max", []),
        "tmin_f": d.get("temperature_2m_min", []),
    })
    return df


def safe_fetch_models(lat: float, lon: float) -> dict:
    """Try multiple models; if some fail, keep the ones that work."""
    results = {}
    for m in MODELS:
        try:
            results[m] = fetch_open_meteo(lat, lon, model=m)
        except Exception:
            continue

    # Fallback: at least try default (no model specified)
    if not results:
        results["default"] = fetch_open_meteo(lat, lon, model=None)

    return results


# --- UI ---
city = st.selectbox("Select City", list(CITIES.keys()))
lat, lon = CITIES[city]

with st.spinner("Fetching forecast models (no API key)…"):
    model_payloads = safe_fetch_models(lat, lon)

# Build per-model daily highs from HOURLY data (better peak time + consistency)
per_model = {}
for model_name, payload in model_payloads.items():
    hourly_df = to_hourly_df(payload)
    if hourly_df.empty:
        continue
    # daily high from hourly temps
    daily_highs = (
        hourly_df.groupby("date")["temp_f"]
        .max()
        .rename("high_f")
        .reset_index()
    )
    per_model[model_name] = {
        "hourly": hourly_df,
        "daily_highs": daily_highs,
        "daily": to_daily_df(payload),
    }

if not per_model:
    st.error("Could not fetch forecast data right now.")
    st.stop()

# Choose available dates from the first model we have
first_model = next(iter(per_model.keys()))
available_dates = per_model[first_model]["daily_highs"]["date"].tolist()

selected_date = st.selectbox("Forecast day (local time)", available_dates)

# Compute daily-high range across models
highs = []
for model_name, data in per_model.items():
    row = data["daily_highs"].loc[data["daily_highs"]["date"] == selected_date]
    if not row.empty:
        highs.append(float(row["high_f"].iloc[0]))

if not highs:
    st.error("No forecast highs available for that day.")
    st.stop()

low_est = min(highs)
high_est = max(highs)
median_est = sorted(highs)[len(highs)//2]

# Peak time: use the first model’s hourly temps for the selected date
hdf = per_model[first_model]["hourly"]
day_rows = hdf[hdf["date"] == selected_date].copy()
peak_time_str = "N/A"
if not day_rows.empty:
    peak_idx = day_rows["temp_f"].astype(float).idxmax()
    peak_time = day_rows.loc[peak_idx, "time"]
    peak_time_str = peak_time.strftime("%-I:%M %p")

# "Suggested Kalshi range" — simple band you can use for bracket/strike thinking
# Round outward to whole degrees.
suggest_low = int(pd.Series([low_est]).apply(lambda x: int(x // 1)).iloc[0])
suggest_high = int(pd.Series([high_est]).apply(lambda x: int(x // 1 + (1 if x % 1 else 0))).iloc[0])

# --- Display ---
st.caption(f"Source: Open-Meteo (no key). Models used: {', '.join(per_model.keys())}")

st.subheader("Daily High Prediction")
st.metric("Predicted High (median of models)", f"{median_est:.1f} °F")
st.write(f"**Model range:** {low_est:.1f}–{high_est:.1f} °F")
st.write(f"**Estimated peak temperature time:** {peak_time_str} (local time)")

st.subheader("Suggested Kalshi range band (simple)")
st.write(
    f"Use this as a practical band for the day’s high: **{suggest_low}–{suggest_high} °F** "
    f"(rounded outward from the model range)."
)

st.caption(
    "Tip: If Kalshi uses threshold markets (e.g., ‘High ≥ X’), values well below the band’s low usually mean ‘lean YES’; "
    "values well above the band’s high usually mean ‘lean NO’. If X is inside the band, it’s closer/coin-flip territory."
)

# Optional: show the model highs table
st.subheader("Per-model highs (for transparency)")
rows = []
for model_name, data in per_model.items():
    row = data["daily_highs"].loc[data["daily_highs"]["date"] == selected_date]
    if not row.empty:
        rows.append({"model": model_name, "high_f": float(row["high_f"].iloc[0])})
if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("high_f"), use_container_width=True)
