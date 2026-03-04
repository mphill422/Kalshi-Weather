import streamlit as st
import requests
import pandas as pd

st.title("Kalshi Weather Trading Dashboard")

cities = {
    "Austin": (30.27, -97.74),
    "Dallas": (32.78, -96.80),
    "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07)
}

city = st.selectbox("Select City", list(cities.keys()))

lat, lon = cities[city]

url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&daily=temperature_2m_max&temperature_unit=fahrenheit&timezone=auto"

response = requests.get(url)
data = response.json()

current_temp = data["hourly"]["temperature_2m"][0]
today_high = data["daily"]["temperature_2m_max"][0]

st.metric("Current Temperature (°F)", current_temp)
st.metric("Today's Forecast High (°F)", today_high)

temps = pd.DataFrame({
    "time": data["hourly"]["time"],
    "temp": data["hourly"]["temperature_2m"]
})

st.line_chart(temps.set_index("time"))
