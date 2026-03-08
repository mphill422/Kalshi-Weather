import streamlit as st
import numpy as np
import pandas as pd
import requests
import math
from datetime import datetime

st.set_page_config(page_title="Kalshi Temperature Model v11.2")

st.title("Kalshi Temperature Model v11.2")

cities = {
    "Phoenix": (33.4342,-112.0116),
    "Las Vegas": (36.0840,-115.1537),
    "Los Angeles": (33.9416,-118.4085),
    "Dallas": (32.8998,-97.0403),
    "Austin": (30.1945,-97.6699),
    "Houston": (29.9902,-95.3368)
}

city = st.selectbox("City", list(cities.keys()))
lat, lon = cities[city]

def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

sources = []
source_table = []

def nws_forecast():
    try:
        url=f"https://api.weather.gov/points/{lat},{lon}"
        r=requests.get(url,timeout=5).json()
        grid=r["properties"]["forecastHourly"]
        data=requests.get(grid,timeout=5).json()

        temps=[p["temperature"] for p in data["properties"]["periods"][:24]]
        return max(temps)
    except:
        return None

def openmeteo_forecast():
    try:
        url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&temperature_unit=fahrenheit"
        r=requests.get(url,timeout=5).json()

        temps=r["hourly"]["temperature_2m"][:24]
        return max(temps)
    except:
        return None

def openmeteo_current():
    try:
        url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"
        r=requests.get(url,timeout=5).json()
        return r["current_weather"]["temperature"]
    except:
        return None

nws = nws_forecast()
if nws:
    sources.append(nws)
    source_table.append(["NWS", nws, "OK"])
else:
    source_table.append(["NWS", "-", "FAILED"])

om = openmeteo_forecast()
if om:
    sources.append(om)
    source_table.append(["Open-Meteo", om, "OK"])
else:
    source_table.append(["Open-Meteo", "-", "FAILED"])

current_temp = openmeteo_current()

st.subheader("Forecast Sources")

df_sources = pd.DataFrame(source_table, columns=["Source","Forecast High","Status"])
st.dataframe(df_sources)

if len(sources) == 0:
    st.error("No weather sources available")
    st.stop()

consensus = np.mean(sources)
spread = max(sources) - min(sources) if len(sources) > 1 else 0

sigma = 1 + (spread * 0.4)

st.subheader("Model Inputs")

st.write("Consensus High:", round(consensus,2))
st.write("Forecast Spread:", round(spread,2))
st.write("Sigma:", round(sigma,2))

if current_temp:
    st.write("Current Temperature:", current_temp)

if current_temp:
    heating_rate = (consensus - current_temp) / 6
    st.write("Estimated Heating Rate:", round(heating_rate,2), "°F/hr")

brackets = [
("69 or below",-100,69),
("70-71",70,71),
("72-73",72,73),
("74-75",74,75),
("76-77",76,77),
("78 or above",78,200)
]

probs = []

for name, lo, hi in brackets:

    if lo == -100:
        p = normal_cdf(69, consensus, sigma)

    elif hi == 200:
        p = 1 - normal_cdf(78, consensus, sigma)

    else:
        p = normal_cdf(hi, consensus, sigma) - normal_cdf(lo, consensus, sigma)

    probs.append(p)

df = pd.DataFrame({
"Bracket":[b[0] for b in brackets],
"Win Probability":np.round(np.array(probs)*100,1),
"Fair YES Price":np.round(np.array(probs)*100,1)
})

st.subheader("Kalshi Probability Table")

st.dataframe(df)

best = df.iloc[df["Win Probability"].idxmax()]

st.success(f"BET SIGNAL: {best['Bracket']} ({best['Win Probability']}%)")
