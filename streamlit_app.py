import streamlit as st
import numpy as np
import pandas as pd
import requests
import math

st.set_page_config(page_title="Kalshi Temp Model v12")

st.title("Kalshi Temperature Model v12")

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


sources=[]
table=[]

# --- NWS ---
try:
    url=f"https://api.weather.gov/points/{lat},{lon}"
    r=requests.get(url,timeout=5).json()
    grid=r["properties"]["forecastHourly"]

    data=requests.get(grid,timeout=5).json()

    temps=[p["temperature"] for p in data["properties"]["periods"][:24]]

    nws=max(temps)

    sources.append(nws)
    table.append(["NWS",nws,"OK"])

except:
    table.append(["NWS","-","FAILED"])


# --- Open Meteo ---
try:
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&temperature_unit=fahrenheit"

    r=requests.get(url,timeout=5).json()

    temps=r["hourly"]["temperature_2m"][:24]

    om=max(temps)

    sources.append(om)
    table.append(["Open-Meteo",om,"OK"])

except:
    table.append(["Open-Meteo","-","FAILED"])


# --- GFS ---
try:
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&models=gfs&hourly=temperature_2m&temperature_unit=fahrenheit"

    r=requests.get(url,timeout=5).json()

    temps=r["hourly"]["temperature_2m"][:24]

    gfs=max(temps)

    sources.append(gfs)
    table.append(["GFS",gfs,"OK"])

except:
    table.append(["GFS","-","FAILED"])


# --- ECMWF ---
try:
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&models=ecmwf&hourly=temperature_2m&temperature_unit=fahrenheit"

    r=requests.get(url,timeout=5).json()

    temps=r["hourly"]["temperature_2m"][:24]

    ecm=max(temps)

    sources.append(ecm)
    table.append(["ECMWF",ecm,"OK"])

except:
    table.append(["ECMWF","-","FAILED"])



st.subheader("Forecast Sources")

df_sources=pd.DataFrame(table,columns=["Source","Forecast High","Status"])

st.dataframe(df_sources)



if len(sources)==0:
    st.error("No sources available")
    st.stop()


consensus=np.mean(sources)

spread=max(sources)-min(sources) if len(sources)>1 else 0

sigma=1 + spread*0.35


st.subheader("Model Inputs")

st.write("Consensus High:",round(consensus,2))

st.write("Forecast Spread:",round(spread,2))

st.write("Sigma:",round(sigma,2))


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

"Fair YES Price":np.round(np.array(probs)*100,1)

})


st.subheader("Kalshi Probability Table")

st.dataframe(df)


best=df.iloc[df["Win Probability"].idxmax()]

st.success(f"BET SIGNAL: {best['Bracket']} ({best['Win Probability']}%)")
