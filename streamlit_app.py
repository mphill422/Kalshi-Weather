# Kalshi Temperature Model v21.2
# Hotfix: sigma capped to prevent probability flattening

import math, re, requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kalshi Temp Model v21.2", layout="wide")
st.title("Kalshi Temperature Model v21.2")

CITIES={
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

DEFAULT_LADDERS={
"Phoenix":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
"Las Vegas":"73 or below | 74-75 | 76-77 | 78-79 | 80-81 | 82 or above",
"Los Angeles":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
"Dallas":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
"Austin":"76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
"Houston":"76 or below | 77-78 | 79-80 | 81-82 | 83-84 | 85 or above",
"Atlanta":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
"NYC":"62 or below | 63-64 | 65-66 | 67-68 | 69-70 | 71 or above",
"Miami":"79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above"
}

def weather(lat,lon):
    r=requests.get("https://api.open-meteo.com/v1/forecast",params={
        "latitude":lat,"longitude":lon,
        "daily":"temperature_2m_max",
        "temperature_unit":"fahrenheit",
        "timezone":"auto"
    }).json()
    return r["daily"]["temperature_2m_max"][0]

def parse_ladder(text):
    brackets=[]
    parts=[p.strip() for p in text.split("|")]
    for p in parts:
        nums=[int(x) for x in re.findall(r"\d+",p)]
        if "below" in p:
            brackets.append((p,None,nums[0]))
        elif "above" in p:
            brackets.append((p,nums[0],None))
        else:
            brackets.append((p,nums[0],nums[1]))
    return brackets

def normal_prob(lo,hi,mu,sigma):
    def cdf(x):
        return 0.5*(1+math.erf((x-mu)/(sigma*math.sqrt(2))))
    if lo is None:
        return cdf(hi+.5)
    if hi is None:
        return 1-cdf(lo-.5)
    return cdf(hi+.5)-cdf(lo-.5)

city=st.selectbox("City",list(CITIES.keys()))
lat,lon=CITIES[city]

forecast=weather(lat,lon)

sigma=1.6
sigma=min(max(sigma,1.2),2.0)

st.subheader("Forecast High")
st.write(round(forecast,2))

ladder_text=st.text_input("Kalshi ladder",DEFAULT_LADDERS[city])
brackets=parse_ladder(ladder_text)

rows=[]
for lab,lo,hi in brackets:
    p=normal_prob(lo,hi,forecast,sigma)*100
    rows.append({"Bracket":lab,"Model %":round(p,1)})

df=pd.DataFrame(rows).sort_values("Model %",ascending=False)

st.subheader("Kalshi Bracket Probabilities")
st.dataframe(df)
