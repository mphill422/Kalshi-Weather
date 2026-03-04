import streamlit as st
import requests

st.title("Kalshi Weather Trading Dashboard")

cities = {
    "Austin": (30.27, -97.74),
    "Dallas": (32.78, -96.80),
    "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07)
}

city = st.selectbox("Select City", list(cities.keys()))

lat, lon = cities[city]

url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&temperature_unit=fahrenheit"

try:
    response = requests.get(url)
    data = response.json()

    temp = data["current"]["temperature_2m"]

    st.metric("Current Temperature (°F)", temp)

except:
    st.error("Weather data not available right now")
