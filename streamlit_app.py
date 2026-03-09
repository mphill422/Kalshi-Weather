# Kalshi Temperature Model v23
# Fully automatic Kalshi ladder mapping by city (no copy/paste needed)
# Probabilities shown as percentages

import math
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kalshi Temp Model v23", layout="wide")

CITY_SERIES = {
    "Phoenix":"KXHIGHPHX",
    "Las Vegas":"KXHIGHLAS",
    "Los Angeles":"KXHIGHLA",
    "Dallas":"KXHIGHDAL",
    "Austin":"KXHIGHAUS",
    "Houston":"KXHIGHHOU",
    "Atlanta":"KXHIGHATL",
    "NYC":"KXHIGHNYC",
    "Miami":"KXHIGHMIA"
}

CITY_COORDS = {
    "Phoenix":(33.4342,-112.0116),
    "Las Vegas":(36.084,-115.1537),
    "Los Angeles":(33.9416,-118.4085),
    "Dallas":(32.8998,-97.0403),
    "Austin":(30.1945,-97.6699),
    "Houston":(29.9902,-95.3368),
    "Atlanta":(33.6407,-84.4277),
    "NYC":(40.7829,-73.9654),
    "Miami":(25.7959,-80.2870)
}

API="https://api.elections.kalshi.com/trade-api/v2/markets"

def weather(lat,lon):
    url="https://api.open-meteo.com/v1/forecast"
    r=requests.get(url,params={
        "latitude":lat,
        "longitude":lon,
        "daily":"temperature_2m_max",
        "temperature_unit":"fahrenheit",
        "timezone":"auto"
    }).json()
    return r["daily"]["temperature_2m_max"][0]

def ladder(series):
    r=requests.get(API,params={"series_ticker":series}).json()
    rows=[]
    for m in r["markets"]:
        title=m["title"]
        rows.append({
            "Bracket":title,
            "Market %":round((m.get("last_price_dollars") or 0)*100,1)
        })
    return rows

def normal_prob(lo,hi,mu,sigma):
    def cdf(x):
        return 0.5*(1+math.erf((x-mu)/(sigma*math.sqrt(2))))
    if lo is None:
        return cdf(hi)
    if hi is None:
        return 1-cdf(lo)
    return cdf(hi)-cdf(lo)

st.title("Kalshi Temperature Model v23")

city=st.selectbox("City",list(CITY_SERIES.keys()))

lat,lon=CITY_COORDS[city]

forecast=weather(lat,lon)
sigma=2

st.subheader("Forecast High")
st.write(round(forecast,2))

series=CITY_SERIES[city]

data=ladder(series)

rows=[]
for d in data:
    text=d["Bracket"]
    nums=[int(x) for x in __import__("re").findall(r"\d+",text)]
    if "below" in text:
        lo=None
        hi=nums[0]
    elif "above" in text:
        lo=nums[0]
        hi=None
    else:
        lo,hi=nums[0],nums[1]
    p=normal_prob(lo,hi,forecast,sigma)*100
    rows.append({
        "Bracket":text,
        "Model %":round(p,1),
        "Market %":d["Market %"]
    })

df=pd.DataFrame(rows).sort_values("Model %",ascending=False)

st.subheader("Model vs Market (%)")
st.dataframe(df)
