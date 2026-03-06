# streamlit_app_v7_7_do_not_bet_filter.py
# Kalshi Weather Model – Daily High
# v7.7 upgrade: DO NOT BET ZONE filter

import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Kalshi Weather Model – v7.7", layout="centered")

st.title("Kalshi Weather Model – Daily High (v7.7)")
st.caption("Adds DO NOT BET filter to avoid low‑confidence trades.")

# --- Example inputs (normally from your forecast model) ---
consensus_high = st.number_input("Consensus High (°F)", value=82.6)
sigma = st.number_input("Model Uncertainty σ", value=1.5)

# --- Settings ---
st.subheader("Risk Controls")
do_not_bet_threshold = st.slider(
    "Do Not Bet Zone (top probability must exceed)",
    min_value=0.40,
    max_value=0.70,
    value=0.55,
    step=0.01
)

# --- Probability model ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def prob_between(mu, sigma, a, b):
    return max(0, norm_cdf((b - mu) / sigma) - norm_cdf((a - mu) / sigma))

# --- Kalshi ladder (example structure) ---
bins = [
    ("77° or below", None, 77),
    ("78° to 79°", 78, 79),
    ("80° to 81°", 80, 81),
    ("82° to 83°", 82, 83),
    ("84° to 85°", 84, 85),
    ("86° or above", 86, None),
]

rows = []

for label, lo, hi in bins:
    if lo is None:
        p = norm_cdf((hi - consensus_high) / sigma)
    elif hi is None:
        p = 1 - norm_cdf((lo - consensus_high) / sigma)
    else:
        p = prob_between(consensus_high, sigma, lo, hi+0.999)

    rows.append({
        "Bracket": label,
        "Probability": p
    })

df = pd.DataFrame(rows)
df["Probability"] = df["Probability"] / df["Probability"].sum()

top_row = df.loc[df["Probability"].idxmax()]
top_prob = float(top_row["Probability"])

st.subheader("Model Probabilities")
st.dataframe(
    df.assign(Probability=lambda x: (x["Probability"]*100).round(1).astype(str) + "%"),
    hide_index=True,
    use_container_width=True
)

# --- Do Not Bet Logic ---
st.subheader("Bet Decision")

if top_prob < do_not_bet_threshold:
    st.error(
        f"DO NOT BET: Top probability only {top_prob*100:.1f}% "
        f"(below {do_not_bet_threshold*100:.0f}% threshold)."
    )
else:
    st.success(
        f"BET OK: {top_row['Bracket']} "
        f"(model probability {top_prob*100:.1f}%)"
    )

st.caption(
    "The Do Not Bet filter prevents coin‑flip weather outcomes from triggering trades."
)
