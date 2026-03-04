import streamlit as st
import requests
import pandas as pd
import math
from datetime import datetime

st.set_page_config(page_title="Kalshi Weather Trading Dashboard")

st.title("Kalshi Weather Trading Dashboard")

# -------------------------
# Cities
# -------------------------

cities = {
"Austin, TX": (30.2672,-97.7431),
"Dallas, TX": (32.7767,-96.7970),
"Houston, TX": (29.7604,-95.3698),
"Phoenix, AZ": (33.4484,-112.0740),
"Las Vegas, NV": (36.1699,-115.1398),
"New York City, NY": (40.7128,-74.0060),
"Atlanta, GA": (33.7490,-84.3880),
"Miami, FL": (25.7617,-80.1918),
"New Orleans, LA": (29.9511,-90.0715),
"San Antonio, TX": (29.4241,-98.4936),
"Los Angeles, CA": (34.0522,-118.2437)
}

city = st.selectbox("Select City",cities)

lat,lon = cities[city]

bracket_size = st.selectbox("Kalshi bracket size (°F)",[1,2,5],index=1)

grace = st.slider("Grace Minutes Around Peak",0,90,30)

st.caption("Sources: Open-Meteo + National Weather Service")

# -------------------------
# Open Meteo
# -------------------------

url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&daily=temperature_2m_max&temperature_unit=fahrenheit&timezone=auto"

data = requests.get(url).json()

temps = data["hourly"]["temperature_2m"]
times = data["hourly"]["time"]

today_high = max(temps[:24])
peak_index = temps.index(today_high)

peak_time = times[peak_index]
peak_dt = datetime.fromisoformat(peak_time)

current_temp = temps[-1]

# -------------------------
# NWS (reliable version)
# -------------------------

try:

    purl=f"https://api.weather.gov/points/{lat},{lon}"
    p=requests.get(purl,headers={"User-Agent":"weather"}).json()

    hourly=p["properties"]["forecastHourly"]

    nws=requests.get(hourly).json()

    nws_high=max([p["temperature"] for p in nws["properties"]["periods"][:24]])

except:
    nws_high=None

# -------------------------
# Model agreement
# -------------------------

models=[today_high]

if nws_high:
    models.append(nws_high)

spread=max(models)-min(models)

if spread<=1:
    confidence="High"
elif spread<=2:
    confidence="Medium"
else:
    confidence="Low"

pred_high=sum(models)/len(models)

# -------------------------
# Peak window
# -------------------------

peak_start=peak_dt-pd.Timedelta(minutes=grace)
peak_end=peak_dt+pd.Timedelta(minutes=grace)

# -------------------------
# Bracket calculation
# -------------------------

def bracket(temp,size):

    low=int(math.floor(temp/size)*size)
    high=low+size-1

    return low,high

low,high=bracket(pred_high,bracket_size)

# -------------------------
# Probability model
# -------------------------

sigma=max(1.25,spread/2)

def normal(x):

    return 0.5*(1+math.erf(x/math.sqrt(2)))

def prob_range(low,high):

    a=(low-.5-pred_high)/sigma
    b=(high+.5-pred_high)/sigma

    return normal(b)-normal(a)

# -------------------------
# Display forecast
# -------------------------

st.header(city)

col1,col2=st.columns(2)

col1.metric("Predicted Daily High",round(pred_high,1))
col2.metric("Confidence",f"{confidence} (spread {round(spread,1)}°)")

st.metric("Estimated Peak Time",peak_dt.strftime("%I:%M %p"))

st.write(f"Peak window: {peak_start.strftime('%I:%M %p')} – {peak_end.strftime('%I:%M %p')}")

# -------------------------
# Current temp
# -------------------------

st.subheader("Current Conditions")

st.metric("Current Temp",round(current_temp,1))

# -------------------------
# Suggested bracket
# -------------------------

st.divider()

st.subheader("Suggested Kalshi Range")

st.write(f"### {low}-{high}°F")

# -------------------------
# Probability ladder
# -------------------------

st.divider()

st.subheader("Kalshi Probability Ladder")

ladder=[]

for i in range(-2,3):

    l=low+i*bracket_size
    h=l+bracket_size-1

    p=prob_range(l,h)

    ladder.append((f"{l}-{h}",round(p*100,1)))

df=pd.DataFrame(ladder,columns=["Bracket","Probability %"])

st.table(df)

# -------------------------
# Value calculator
# -------------------------

st.divider()

st.subheader("Value Bet Check")

price=st.number_input("Enter Kalshi price for main bracket (cents)",0,100,50)

prob=prob_range(low,high)

edge=prob-price/100

col1,col2,col3=st.columns(3)

col1.metric("Model Prob",f"{round(prob*100,1)}%")
col2.metric("Market Prob",f"{price}%")
col3.metric("Edge",f"{round(edge*100,1)}%")

if edge>0.08:
    st.success("VALUE BET")
elif edge<-0.08:
    st.error("OVERPRICED")
else:
    st.info("Neutral")

# -------------------------
# Heat spike detector
# -------------------------

st.divider()

st.subheader("Heat Spike Detector")

expected=pred_high-((peak_dt.hour-datetime.now().hour)*2)

if current_temp>expected:

    st.success("Temperature running HOT vs forecast — upside risk")

else:

    st.write("Temperature tracking forecast")
