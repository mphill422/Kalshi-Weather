# Kalshi High Temperature Model - V4.3
# - Lean sigma values (not inflated)
# - V4.3 consensus formula with late-day floor
# - 2-degree call
# - Auto-fetch: Open-Meteo + NOAA
# - Kalshi bracket auto-fetch
# - Settlement history tracking
import math
import re
import json
import requests
import streamlit as st
from pathlib import Path
from datetime import datetime
st.set_page_config(page_title="Kalshi High Temp V4.3", layout="wide")
st.title("Kalshi High Temperature Model - V4.3")
SAVE_FILE = Path("saved_ladders.json")
HISTORY_FILE = Path("settlement_history.json")
HEADERS = {
"User-Agent": "kalshi-temp-model/4.3",
"Accept": "application/geo+json, application/json",
}
SERIES = {
"Phoenix": "KXHIGHPHX", "Las Vegas": "KXHIGHLAS", "Los Angeles": "KXHIGHLAX",
"Dallas": "KXHIGHDAL", "Austin": "KXHIGHAUS", "Houston": "KXHIGHHOU",
"Atlanta": "KXHIGHATL", "Miami": "KXHIGHMIA", "New York": "KXHIGHNY",
"San Antonio": "KXHIGHSAT", "New Orleans": "KXHIGHMSY", "Philadelphia": "KXHIGHPHL",
"Boston": "KXHIGHBOS", "Denver": "KXHIGHDEN", "Oklahoma City": "KXHIGHOKC",
"Minneapolis": "KXHIGHMSP", "Washington DC": "KXHIGHDCA",
}
STATIONS = {
"Phoenix": "CLIPHX", "Las Vegas": "CLILAS", "Los Angeles": "CLILAX",
"Dallas": "CLIDFW", "Austin": "CLIAUS", "Houston": "CLIHOU",
"Atlanta": "CLIATL", "Miami": "CLIMIA", "New York": "KNYC",
"San Antonio": "CLISAT", "New Orleans": "CLIMSY", "Philadelphia": "CLIPHL",
"Boston": "CLIBOS", "Denver": "CLIDEN", "Oklahoma City": "CLIOKC",
"Minneapolis": "CLIMSP", "Washington DC": "CLIDCA",
}
SETTLEMENT_LOCATION = {
"Phoenix": "Phoenix Sky Harbor Airport",
"Las Vegas": "Las Vegas Harry Reid Airport",
"Los Angeles": "LA International Airport",
"Dallas": "Dallas/Fort Worth Airport",
"Austin": "Austin-Bergstrom Airport",
"Houston": "Houston Hobby Airport",
"Atlanta": "Atlanta Hartsfield Airport",
"Miami": "Miami International Airport",
"New York": "Central Park, Manhattan",
"San Antonio": "San Antonio International Airport",
"New Orleans": "New Orleans Armstrong Airport",
"Philadelphia": "Philadelphia International Airport",
"Boston": "Boston Logan Airport",
"Denver": "Denver International Airport",
"Oklahoma City": "Oklahoma City Will Rogers Airport",
"Minneapolis": "Minneapolis-St. Paul Airport",
"Washington DC": "Reagan National Airport",
}
CITIES = {
"Phoenix": {"lat": 33.4342, "lon": -112.0116},
"Las Vegas": {"lat": 36.0840, "lon": -115.1537},
"Los Angeles": {"lat": 33.9416, "lon": -118.4085},
"Dallas": {"lat": 32.8998, "lon": -97.0403},
"Austin": {"lat": 30.1945, "lon": -97.6699},
"Houston": {"lat": 29.9902, "lon": -95.3368},
"Atlanta": {"lat": 33.6407, "lon": -84.4277},
"Miami": {"lat": 25.7959, "lon": -80.2870},
"New York": {"lat": 40.7812, "lon": -73.9665},
"San Antonio": {"lat": 29.5337, "lon": -98.4698},
"New Orleans": {"lat": 29.9934, "lon": -90.2580},
"Philadelphia": {"lat": 39.8744, "lon": -75.2424},
"Boston": {"lat": 42.3656, "lon": -71.0096},
"Denver": {"lat": 39.8561, "lon": -104.6737},
"Oklahoma City": {"lat": 35.3931, "lon": -97.6007},
"Minneapolis": {"lat": 44.8848, "lon": -93.2223},
"Washington DC": {"lat": 38.8512, "lon": -77.0402},
}
DEFAULT_LADDERS = {
"Phoenix": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
"Las Vegas": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
"Los Angeles": "66 or below | 67-68 | 69-70 | 71-72 | 73-74 | 75 or above",
"Dallas": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
"Austin": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
"Houston": "79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
"Atlanta": "74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
"Miami": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
"New York": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
"San Antonio": "78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
"New Orleans": "80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
"Philadelphia": "73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
"Boston": "48 or below | 49-50 | 51-52 | 53-54 | 55-56 | 57 or above",
"Denver": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
"Oklahoma City": "75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
"Minneapolis": "65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
"Washington DC": "76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
}
BASE_SIGMA = {
"New York": 1.5, "Philadelphia": 1.5, "Washington DC": 1.6, "Boston": 1.6,
"Los Angeles": 1.4, "Denver": 1.6, "Miami": 1.7, "Minneapolis": 1.7,
"New Orleans": 1.8, "Phoenix": 1.9, "Las Vegas": 1.9, "Atlanta": 2.0,
"Dallas": 2.0, "Austin": 2.0, "Houston": 2.0, "San Antonio": 2.0,
"Oklahoma City": 2.1,
}
DESERT_CITIES = {"Phoenix", "Las Vegas"}
def normal_cdf(x, mu, sigma):
return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
def choose_sigma(city):
s = BASE_SIGMA.get(city, 1.8)
hour = datetime.now().hour
s *= 1.00 if hour < 11 else 0.92 if hour < 14 else 0.86 if hour < 16 else 0.80
if city in DESERT_CITIES:
s *= 0.90
return max(1.10, min(2.40, s))
def late_day_floor(fc, obs, hour):
gap = max(0.0, fc - obs)
frac = 0.45 if hour < 12 else 0.62 if hour < 14 else 0.78 if hour < 16 else 0.90
return obs + frac * gap
def compute_consensus(fc, cur, noaa, hour):
if noaa is not None:
base = fc * 0.55 + cur * 0.20 + noaa * 0.25
else:
base = fc * 0.70 + cur * 0.30
if abs(base - fc) > 2:
base = fc - 1 if base < fc else fc + 1
obs = noaa if noaa is not None else cur
floor = late_day_floor(fc, obs, hour)
consensus = max(base, floor)
if consensus > fc + 0.6:
consensus = fc + 0.6
return consensus
def parse_ladder(text):
out = []
for p in text.split("|"):
p = p.strip()
nums = [int(x) for x in re.findall(r"\d+", p)]
if not nums:
continue
low = p.lower()
if "below" in low:
out.append((p, None, nums[0]))
elif "above" in low:
out.append((p, nums[0], None))
elif len(nums) >= 2:
out.append((p, nums[0], nums[1]))
return out
def bracket_probs(mu, ladder_text, city):
sigma = choose_sigma(city)
rows = []
for label, lo, hi in parse_ladder(ladder_text):
if lo is None:
p = normal_cdf(hi + 0.5, mu, sigma)
elif hi is None:
p = 1 - normal_cdf(lo - 0.5, mu, sigma)
else:
p = normal_cdf(hi + 0.5, mu, sigma) - normal_cdf(lo - 0.5, mu, sigma)
rows.append((label, max(0.0, min(1.0, p))))
rows.sort(key=lambda x: x[1], reverse=True)
return rows, sigma
def two_degree_call(mu, ladder_text):
best_label, best_dist = None, float("inf")
for label, lo, hi in parse_ladder(ladder_text):
if lo is None or hi is None:
continue
center = (lo + hi) / 2
dist = abs(center - mu)
if dist < best_dist:
best_dist = dist
best_label = label
return best_label
def ladder_to_boxes(text):
parts = [p.strip() for p in text.split("|")]
while len(parts) < 6:
parts.append("")
return parts[:6]
def boxes_to_ladder(parts):
cleaned = []
for i, p in enumerate(parts):
t = p.strip()
if not t:
continue
nums = re.findall(r"\d+", t)
low = t.lower()
if "below" in low or "above" in low or "-" in t:
cleaned.append(t)
elif len(nums) == 1:
n = int(nums[0])
if i == 0:
cleaned.append(str(n) + " or below")
elif i == 5:
cleaned.append(str(n) + " or above")
else:
cleaned.append(str(n))
else:
cleaned.append(t)
return " | ".join(cleaned)
def load_json(path):
if path.exists():
try:
return json.loads(path.read_text())
except Exception:
return {}
return {}
def save_json(path, data):
path.write_text(json.dumps(data, indent=2))
def safe_get(url, params=None):
try:
r = requests.get(url, params=params, headers=HEADERS, timeout=12)
r.raise_for_status()
return r.json()
except Exception:
return None
def c_to_f(c):
return c * 9 / 5 + 32
def fetch_open_meteo(lat, lon):
data = safe_get("https://api.open-meteo.com/v1/forecast", {
"latitude": lat, "longitude": lon,
"daily": "temperature_2m_max",
"current": "temperature_2m",
"temperature_unit": "fahrenheit",
"timezone": "auto",
"forecast_days": 2,
})
if not data:
return None, None
today = datetime.now().strftime("%Y-%m-%d")
times = data.get("daily", {}).get("time", [])
idx = next((i for i, t in enumerate(times) if t.startswith(today)), 0)
fc = data.get("daily", {}).get("temperature_2m_max", [None])[idx]
cur = data.get("current", {}).get("temperature_2m")
return (float(fc) if fc is not None else None,
float(cur) if cur is not None else None)
def fetch_noaa(lat, lon, station_id):
if station_id:
if obs:
obs = safe_get("https://api.weather.gov/stations/" + station_id + "/observations/late
temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
if temp_c is not None:
return station_id, float(c_to_f(temp_c))
points = safe_get("https://api.weather.gov/points/" + str(lat) + "," + str(lon))
if not points:
return station_id, None
stations_url = points.get("properties", {}).get("observationStations")
if not stations_url:
return station_id, None
stations = safe_get(stations_url)
if not stations or not stations.get("observationStations"):
return station_id, None
first = stations["observationStations"][0]
sid = first.rstrip("/").split("/")[-1]
obs = safe_get(first + "/observations/latest")
if not obs:
return sid, None
temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
if temp_c is None:
return sid, None
return sid, float(c_to_f(temp_c))
def fetch_kalshi_brackets(series):
url = "https://api.elections.kalshi.com/trade-api/v2/markets"
params = {"series_ticker": series, "status": "open", "limit": 30}
data = safe_get(url, params)
if not data or not data.get("markets"):
return None
markets = data["markets"]
parsed = []
for m in markets:
s = (m.get("subtitle") or "").replace("deg", "").replace("\u00b0", "").strip()
below = re.match(r"^(\d+)\s*or\s*below$", s, re.I)
above = re.match(r"^(\d+)\s*or\s*above$", s, re.I)
rng = re.match(r"^(\d+)\s*to\s*(\d+)$", s, re.I)
if below:
label = below.group(1) + " or below"
key = int(below.group(1)) - 10000
elif above:
label = above.group(1) + " or above"
key = int(above.group(1)) + 10000
elif rng:
label = rng.group(1) + "-" + rng.group(2)
key = int(rng.group(1))
else:
continue
parsed.append((key, label, m.get("yes_ask"), m.get("no_ask")))
if len(parsed) < 2:
return None
parsed.sort(key=lambda x: x[0])
return [(label, yes_ask, no_ask) for _, label, yes_ask, no_ask in parsed]
# App
saved_ladders = load_json(SAVE_FILE)
history = load_json(HISTORY_FILE)
if not isinstance(history, list):
history = []
city = st.selectbox("City", list(CITIES.keys()))
lat = CITIES[city]["lat"]
lon = CITIES[city]["lon"]
station = STATIONS[city]
series = SERIES[city]
st.caption("Settlement: " + STATIONS[city] + " - " + SETTLEMENT_LOCATION[city] + " - Series:
st.subheader("Kalshi Ladder")
col_fetch, col_status = st.columns([2, 3])
with col_fetch:
fetch_brackets = st.button("Fetch Live Brackets from Kalshi")
kalshi_markets = None
if fetch_brackets:
with st.spinner("Fetching from Kalshi..."):
kalshi_markets = fetch_kalshi_brackets(series)
if kalshi_markets:
labels = [m[0] for m in kalshi_markets]
while len(labels) < 6:
labels.append("")
saved_ladders[city] = " | ".join(labels[:6])
save_json(SAVE_FILE, saved_ladders)
st.success("Loaded " + str(len(kalshi_markets)) + " brackets from Kalshi")
else:
st.warning("Could not fetch from Kalshi API. Edit brackets manually below.")
if city not in saved_ladders:
saved_ladders[city] = DEFAULT_LADDERS.get(city, "")
box_values = ladder_to_boxes(saved_ladders[city])
with st.expander("Edit Brackets", expanded=False):
cols = st.columns(6)
new_boxes = []
for i, col in enumerate(cols):
with col:
new_boxes.append(st.text_input("Box " + str(i + 1), value=box_values[i], key=city
if st.button("Save Ladder"):
ladder_text = boxes_to_ladder(new_boxes)
saved_ladders[city] = ladder_text
save_json(SAVE_FILE, saved_ladders)
st.success("Saved")
st.rerun()
ladder_text = saved_ladders[city]
st.caption("Current ladder: " + ladder_text)
st.subheader("Live Weather")
with st.spinner("Fetching weather..."):
forecast, current = fetch_open_meteo(lat, lon)
noaa_station, noaa_obs = fetch_noaa(lat, lon, station)
hour = datetime.now().hour
col1, col2, col3 = st.columns(3)
with col1:
st.metric("Forecast High", str(round(forecast, 1)) + " F" if forecast else "unavailable")
with col2:
st.metric("Current Temp", str(round(current, 1)) + " F" if current else "unavailable")
with col3:
if noaa_obs is not None:
st.metric("NOAA Obs", str(round(noaa_obs, 1)) + " F")
st.caption("Station: " + noaa_station)
else:
st.metric("NOAA Obs", "Unavailable")
if forecast is not None and current is not None:
consensus = compute_consensus(forecast, current, noaa_obs, hour)
rows, sigma = bracket_probs(consensus, ladder_text, city)
call = two_degree_call(consensus, ladder_text)
st.subheader("Model Output")
c1, c2, c3 = st.columns(3)
with c1:
st.metric("Consensus High", str(round(consensus, 1)) + " F")
with c2:
st.metric("2 Degree Call", call or "none")
with c3:
st.metric("Sigma", str(round(sigma, 2)) + " F")
st.caption("Time: " + str(hour) + ":00 local - Late-day floor active")
import pandas as pd
df_rows = []
for label, prob in rows:
fair = round(prob * 100)
yes_ask = no_ask = None
if kalshi_markets:
match = next((m for m in kalshi_markets if m[0] == label), None)
if match:
yes_ask = match[1]
no_ask = match[2]
edge = (fair - yes_ask) if yes_ask is not None else None
df_rows.append({
"Bracket": label,
"Model %": str(round(prob * 100, 1)) + "%",
"Fair": str(fair) + "c",
"YES ask": str(yes_ask) + "c" if yes_ask else "none",
"NO ask": str(no_ask) + "c" if no_ask else "none",
"Edge": ("+" + str(edge) + "c") if edge and edge > 0 else (str(edge) + "c" if edg
})
df = pd.DataFrame(df_rows)
st.dataframe(df, use_container_width=True, hide_index=True)
parsed = parse_ladder(ladder_text)
top_b = next((b for b in parsed if b[2] is None), None)
bot_b = next((b for b in parsed if b[1] is None), None)
if (top_b and consensus > top_b[1] + 5) or (bot_b and consensus < bot_b[2] - 5):
st.warning("Ladder does not cover consensus of " + str(round(consensus, 1)) + " F - u
else:
st.error("Weather data unavailable. Check your internet connection.")
st.subheader("Log Actual High (after settlement)")
with st.form("log_form"):
actual = st.number_input("Actual NWS High F", min_value=0.0, max_value=130.0, step=0.1)
submitted = st.form_submit_button("Log Settlement")
if submitted and forecast is not None:
entry = {
"date": datetime.now().strftime("%Y-%m-%d"),
"city": city,
"actual": actual,
"forecast": round(forecast, 1),
"consensus": round(consensus, 1),
"error": round(actual - consensus, 1),
}
history.append(entry)
save_json(HISTORY_FILE, history[-300:])
st.success("Logged - actual=" + str(actual) + "F consensus=" + str(round(consensus,
if history:
st.subheader("Settlement History")
import pandas as pd
df_h = pd.DataFrame(history[-50:][::-1])
wd = [h for h in history if h.get("consensus") and h.get("actual")]
if wd:
mae = sum(abs(h["actual"] - h["consensus"]) for h in wd) / len(wd)
w1 = sum(1 for h in wd if abs(h["actual"] - h["consensus"]) <= 1) / len(wd)
hc1, hc2, hc3 = st.columns(3)
with hc1:
st.metric("Records", len(history))
with hc2:
st.metric("Model MAE", str(round(mae, 2)) + " F")
with hc3:
st.metric("Within 1 degree", str(round(w1 * 100, 0)) + "%")
st.dataframe(df_h, use_container_width=True, hide_index=True)