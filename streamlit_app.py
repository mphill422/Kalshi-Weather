import streamlit as st
import numpy as np
import pandas as pd
import requests
import math

st.set_page_config(page_title="Kalshi Temperature Model v11")

st.title("Kalshi Temperature Model v11")

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

def nws_high():
    url=f"https://api.weather.gov/points/{lat},{lon}"
    r=requests.get(url).json()
    grid=r["properties"]["forecastHourly"]
    data=requests.get(grid).json()
    temps=[p["temperature"] for p in data["properties"]["periods"][:24]]
    return max(temps)

def open_meteo_high():
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&temperature_unit=fahrenheit"
    r=requests.get(url).json()
    temps=r["hourly"]["temperature_2m"][:24]
    return max(temps)

def current_temp():
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"
    r=requests.get(url).json()
    return r["current_weather"]["temperature"]

sources=[]
labels=[]

try:
    nws=nws_high()
    sources.append(nws)
    labels.append(("NWS",nws))
except:
    pass

try:
    om=open_meteo_high()
    sources.append(om)
    labels.append(("OpenMeteo",om))
except:
    pass

consensus=np.mean(sources)

try:
    cur=current_temp()
except:
    cur=None

spread=max(sources)-min(sources) if len(sources)>1 else 0

sigma=1+spread*0.3

st.subheader("Forecast Sources")

df=pd.DataFrame(labels,columns=["Source","High"])
st.dataframe(df)

st.subheader("Model Inputs")

st.write("Consensus High:",round(consensus,2))
st.write("Forecast Spread:",round(spread,2))
st.write("Sigma:",round(sigma,2))

if cur:
    st.write("Current Temperature:",cur)

brackets=[
("69 or below",-100,69),
("70-71",70,71),
("72-73",72,73),
("74-75",74,75),
("76-77",76,77),
("78 or above",78,200)
]

probs=[]

for name,lo,hi in brackets:

    if lo==-100:
        p=normal_cdf(69,consensus,sigma)

    elif hi==200:
        p=1-normal_cdf(78,consensus,sigma)

    else:
        p=normal_cdf(hi,consensus,sigma)-normal_cdf(lo,consensus,sigma)

    probs.append(p)

df=pd.DataFrame({
"Bracket":[b[0] for b in brackets],
"Win Probability":np.round(np.array(probs)*100,1),
"Fair YES price":np.round(np.array(probs)*100,1)
})

st.subheader("Kalshi Probability Table")
st.dataframe(df)

best=df.iloc[df["Win Probability"].idxmax()]

st.success(f"BET SIGNAL: {best['Bracket']} ({best['Win Probability']}%)")
