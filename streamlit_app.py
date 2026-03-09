# Kalshi Temperature Model v12.1
import math
import requests
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Kalshi Temp Model v12.1", layout="wide")
st.title("Kalshi Temperature Model v12.1")

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix","bias":0.5},
    "Las Vegas":{"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles","bias":0.3},
    "Dallas":{"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","bias":0.4},
    "Austin":{"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago","bias":0.2},
    "Houston":{"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago","bias":0.3},
    "Los Angeles":{"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","bias":-0.6},
}

def normal_cdf(x,mu,sigma):
    return 0.5*(1+math.erf((x-mu)/(sigma*math.sqrt(2))))

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]

data=requests.get(
"https://api.open-meteo.com/v1/forecast",
params={
"latitude":lat,
"longitude":lon,
"hourly":"temperature_2m",
"current_weather":True,
"temperature_unit":"fahrenheit",
"forecast_days":1
}
).json()

current=data["current_weather"]["temperature"]
hourly=data["hourly"]["temperature_2m"]
forecast_high=max(hourly[:24])

consensus=forecast_high+profile["bias"]
spread=max(hourly[:12])-min(hourly[:12])
sigma=max(1.0,spread*0.3)

st.subheader("Forecast Data")

c1,c2,c3=st.columns(3)
c1.metric("Current Temp",f"{current:.1f}")
c2.metric("Forecast High",f"{forecast_high:.1f}")
c3.metric("Consensus High",f"{consensus:.1f}")

center=round(consensus)

brackets=[
f"{center-4} or below",
f"{center-3} to {center-2}",
f"{center-1} to {center}",
f"{center+1} to {center+2}",
f"{center+3} to {center+4}",
f"{center+5} or above"
]

rows=[]

for b in brackets:
    nums=[int(x) for x in b.split() if x.isdigit()]
    if "below" in b:
        p=normal_cdf(nums[0],consensus,sigma)
    elif "above" in b:
        p=1-normal_cdf(nums[0],consensus,sigma)
    else:
        lo,hi=nums
        p=normal_cdf(hi,consensus,sigma)-normal_cdf(lo,consensus,sigma)
    rows.append({"Bracket":b,"WinProb":p})

rows=sorted(rows,key=lambda x:x["WinProb"],reverse=True)

df=pd.DataFrame(rows)
df["WinProb"]=(df["WinProb"]*100).round(1)

st.subheader("Bracket Probabilities")
st.dataframe(df,use_container_width=True)

top=df.iloc[0]

if top["WinProb"]>60 and spread<4:
    st.success(f"BET SIGNAL: {top['Bracket']} ({top['WinProb']}%)")
else:
    st.error("PASS â Edge not strong enough")
