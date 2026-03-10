# Kalshi High Temperature Model – Station Corrected Version
# Only requested changes:
# Houston -> CLIHOU
# New York -> KNYC
# Everything else unchanged logically.

import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kalshi High Temperature Model", layout="wide")
st.title("Kalshi High Temperature Model")

# Settlement stations aligned with Kalshi contracts
STATIONS = {
    "Los Angeles": "CLILAX",
    "New York": "KNYC",
    "Atlanta": "CLIATL",
    "Houston": "CLIHOU",
    "Miami": "CLIMIA",
    "Philadelphia": "CLIPHL",
    "Boston": "CLIBOS",
    "Denver": "CLIDEN",
    "Minneapolis": "CLIMSP",
    "Washington DC": "CLIDCA",
    "Oklahoma City": "CLIOKC",
    "New Orleans": "CLIMSY",
    "San Antonio": "CLISAT",
}

CITIES = {
    "Phoenix": {"lat":33.4342,"lon":-112.0116,"tz":"America/Phoenix"},
    "Las Vegas": {"lat":36.0840,"lon":-115.1537,"tz":"America/Los_Angeles"},
    "Los Angeles": {"lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles"},
    "Dallas": {"lat":32.8998,"lon":-97.0403,"tz":"America/Chicago"},
    "Austin": {"lat":30.1945,"lon":-97.6699,"tz":"America/Chicago"},
    "Houston": {"lat":29.9902,"lon":-95.3368,"tz":"America/Chicago"},
    "Atlanta": {"lat":33.6407,"lon":-84.4277,"tz":"America/New_York"},
    "Miami": {"lat":25.7959,"lon":-80.2870,"tz":"America/New_York"},
    "New York": {"lat":40.7812,"lon":-73.9665,"tz":"America/New_York"},
    "San Antonio": {"lat":29.5337,"lon":-98.4698,"tz":"America/Chicago"},
    "New Orleans": {"lat":29.9934,"lon":-90.2580,"tz":"America/Chicago"},
    "Philadelphia": {"lat":39.8744,"lon":-75.2424,"tz":"America/New_York"},
    "Boston": {"lat":42.3656,"lon":-71.0096,"tz":"America/New_York"},
    "Denver": {"lat":39.8561,"lon":-104.6737,"tz":"America/Denver"},
    "Oklahoma City": {"lat":35.3931,"lon":-97.6007,"tz":"America/Chicago"},
    "Minneapolis": {"lat":44.8848,"lon":-93.2223,"tz":"America/Chicago"},
    "Washington DC": {"lat":38.8512,"lon":-77.0402,"tz":"America/New_York"},
}

DEFAULT_LADDERS = {
    "Phoenix":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Las Vegas":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Los Angeles":"66 or below | 67-68 | 69-70 | 71-72 | 73-74 | 75 or above",
    "Dallas":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Austin":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "Houston":"79 or below | 80-81 | 82-83 | 84-85 | 86-87 | 88 or above",
    "Atlanta":"74 or below | 75-76 | 77-78 | 79-80 | 81-82 | 83 or above",
    "Miami":"80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "New York":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "San Antonio":"78 or below | 79-80 | 81-82 | 83-84 | 85-86 | 87 or above",
    "New Orleans":"80 or below | 81-82 | 83-84 | 85-86 | 87-88 | 89 or above",
    "Philadelphia":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
    "Boston":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Denver":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Oklahoma City":"75 or below | 76-77 | 78-79 | 80-81 | 82-83 | 84 or above",
    "Minneapolis":"65 or below | 66-67 | 68-69 | 70-71 | 72-73 | 74 or above",
    "Washington DC":"70 or below | 71-72 | 73-74 | 75-76 | 77-78 | 79 or above",
}

def safe_get(url,params=None):
    try:
        r=requests.get(url,params=params,timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def normal_cdf(x,mu,sigma):
    z=(x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1+math.erf(z))

def parse_ladder(text):
    out=[]
    for p in text.split("|"):
        nums=[int(x) for x in re.findall(r"\d+",p)]
        if "below" in p:out.append((p,None,nums[0]))
        elif "above" in p:out.append((p,nums[0],None))
        else:out.append((p,nums[0],nums[1]))
    return out

def bracket_probs(mu,ladder):
    sigma=1.4
    rows=[]
    for lab,lo,hi in ladder:
        if lo is None:p=normal_cdf(hi+.5,mu,sigma)
        elif hi is None:p=1-normal_cdf(lo-.5,mu,sigma)
        else:p=normal_cdf(hi+.5,mu,sigma)-normal_cdf(lo-.5,mu,sigma)
        rows.append((lab,p))
    rows.sort(key=lambda x:x[1],reverse=True)
    return rows

city=st.selectbox("City",list(CITIES.keys()))
profile=CITIES[city]

lat=profile["lat"]
lon=profile["lon"]

st.write("Kalshi Settlement Station:", STATIONS.get(city,"N/A"))

ladder_text=st.text_input("Kalshi Ladder",DEFAULT_LADDERS[city])
ladder=parse_ladder(ladder_text)

weather=safe_get("https://api.open-meteo.com/v1/forecast",{
    "latitude":lat,
    "longitude":lon,
    "daily":"temperature_2m_max",
    "current":"temperature_2m",
    "temperature_unit":"fahrenheit",
    "timezone":"auto"
})

if weather:
    current_temp=weather["current"]["temperature_2m"]
    forecast=weather["daily"]["temperature_2m_max"][0]
    consensus=max(current_temp,forecast)

    st.subheader("Current Temperature")
    st.write(current_temp)

    st.subheader("Forecast High")
    st.write(forecast)

    st.subheader("Model Consensus High")
    st.write(consensus)

    rows=bracket_probs(consensus,ladder)
    df=pd.DataFrame(rows,columns=["Bracket","Probability"])
    df["Probability"]=df["Probability"].apply(lambda x:f"{x*100:.1f}%")

    st.subheader("Kalshi Bracket Probabilities")
    st.dataframe(df)

else:
    st.error("Weather data unavailable")
