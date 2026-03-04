import streamlit as st
import requests

st.title("Kalshi Weather Trading Dashboard")

city = st.selectbox(
    "Select City",
    ["Austin", "Dallas", "Houston", "Phoenix", "Las Vegas"]
)

st.write(f"Fetching weather data for {city}...")

url = f"https://wttr.in/{city}?format=j1"

try:
    response = requests.get(url, timeout=10)
    data = response.json()

    temp = data["current_condition"][0]["temp_F"]
    humidity = data["current_condition"][0]["humidity"]

    st.metric("Current Temperature (°F)", temp)
    st.metric("Humidity (%)", humidity)

    st.write("Weather data source: wttr.in")

except Exception as e:
    st.error("Could not fetch weather data right now.")
    st.write("This usually happens if the external weather API blocks the request.")
