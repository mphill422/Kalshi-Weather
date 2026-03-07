import streamlit as st

st.set_page_config(page_title="Kalshi Weather Model v9.5", layout="wide")

st.title("Kalshi Temperature Model v9.5 (Stable)")

# --- City Filters ---
CITY_PROB_FILTERS = {
    "Phoenix": 0.55,
    "Las Vegas": 0.55,
    "Houston": 0.62
}

DEFAULT_PROB_FILTER = 0.58
MIN_TOP_TWO_GAP = 0.12

city = st.selectbox(
    "City",
    ["Phoenix","Las Vegas","Los Angeles","Dallas","Austin","Houston"]
)

prob_filter = CITY_PROB_FILTERS.get(city, DEFAULT_PROB_FILTER)

st.markdown(f"""
### Active Filters
Probability filter: **{prob_filter:.2f}**  
Minimum top-two gap: **{MIN_TOP_TWO_GAP*100:.0f}%**
""")

st.divider()

st.header("Model Inputs")

top_prob = st.number_input("Top bracket probability",0.0,1.0,0.60,0.01)
gap = st.number_input("Top-two bracket gap",0.0,1.0,0.20,0.01)

st.divider()

st.header("Decision")

reasons = []

if top_prob < prob_filter:
    reasons.append("Top probability below filter")

if gap < MIN_TOP_TWO_GAP:
    reasons.append("Top-two gap too small")

if reasons:
    st.error("PASS â " + " | ".join(reasons))
else:
    st.success("BET â Model conditions satisfied")

st.divider()

st.caption("v9.5 stable build â simplified to prevent runtime errors")
