import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

st.set_page_config(page_title="Kalshi Temperature Model v10.5")

st.title("Kalshi Temperature Model v10.5")

cities = {
    "Phoenix": (33.4342,-112.0116),
    "Las Vegas": (36.0840,-115.1537),
    "Los Angeles": (33.9416,-118.4085),
    "Dallas": (32.8998,-97.0403),
    "Austin": (30.1945,-97.6699),
    "Houston": (29.9902,-95.3368)
}

city = st.selectbox("City", list(cities.keys()))

lat,lon = cities[city]

def nws_high():

    url=f"https://api.weather.gov/points/{lat},{lon}"
    r=requests.get(url).json()

    grid=r["properties"]["forecastHourly"]

    data=requests.get(grid).json()

    temps=[]

    for p in data["properties"]["periods"][:24]:
        temps.append(p["temperature"])

    return max(temps)

try:
    nws=nws_high()
except:
    nws=None

consensus=nws

sigma=1.0

st.subheader("Consensus High")
st.write(consensus)

st.write("Sigma")
st.write(sigma)

# Kalshi bracket structure
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
        p=norm.cdf(69-consensus,sigma)

    elif hi==200:
        p=1-norm.cdf(78-consensus,sigma)

    else:
        p=norm.cdf(hi-consensus,sigma)-norm.cdf(lo-consensus,sigma)

    probs.append(p)

df=pd.DataFrame({
"Bracket":[b[0] for b in brackets],
"Win Probability":np.round(np.array(probs)*100,1)
})

st.subheader("Model Bracket Probabilities")
st.dataframe(df)

best=df.iloc[df["Win Probability"].idxmax()]

st.success(f"BET SIGNAL: {best['Bracket']}")
