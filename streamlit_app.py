import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass

st.set_page_config(page_title="Kalshi Model v17", layout="wide")

BRACKET_ORDER = [
    "78 or below",
    "79-80",
    "81-82",
    "83-84",
    "85-86",
    "87 or above",
]

@dataclass
class LadderConfig:
    num_core_brackets: int = 2
    include_tails: bool = False
    core_edge_threshold: float = 0.06
    tail_edge_threshold: float = 0.10
    stake1: float = 10.0
    stake2: float = 20.0
    stake3: float = 30.0
    max_total_risk: float = 30.0
    prefer_middle_brackets: bool = True


def american_to_implied_prob(odds):
    if odds is None:
        return np.nan
    try:
        odds = float(odds)
    except Exception:
        return np.nan

    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return np.nan


def choose_sigma(hour_local):
    if hour_local < 11:
        return 2.2
    elif hour_local < 15:
        return 1.8
    return 1.5


def apply_intraday_adjustment(adjusted_high, current_temp, expected_temp_now, hour_local):
    if current_temp is None or expected_temp_now is None:
        return adjusted_high

    if 11 <= hour_local <= 14:
        lag = expected_temp_now - current_temp
        if lag > 2.0:
            adjusted_high -= 1.0
        elif lag < -2.0:
            adjusted_high += 0.7
    return adjusted_high


def simulate_bracket_probs(
    src1,
    src2,
    src3,
    station_bias=0.0,
    current_temp=None,
    expected_temp_now=None,
    hour_local=12,
    n_sims=10000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    base_high = 0.55 * src1 + 0.30 * src2 + 0.15 * src3
    adjusted_high = base_high + station_bias
    adjusted_high = apply_intraday_adjustment(
        adjusted_high, current_temp, expected_temp_now, hour_local
    )

    sigma = choose_sigma(hour_local)

    sims = rng.normal(adjusted_high, sigma, n_sims)
    sims = np.rint(sims).astype(int)

    probs = {
        "78 or below": float(np.mean(sims <= 78)),
        "79-80": float(np.mean((sims >= 79) & (sims <= 80))),
        "81-82": float(np.mean((sims >= 81) & (sims <= 82))),
        "83-84": float(np.mean((sims >= 83) & (sims <= 84))),
        "85-86": float(np.mean((sims >= 85) & (sims <= 86))),
        "87 or above": float(np.mean(sims >= 87)),
    }

    return {
        "base_high": round(base_high, 2),
        "adjusted_high": round(adjusted_high, 2),
        "sigma": sigma,
        "bracket_probs": probs,
    }


def build_market_table(model_probs, kalshi_yes_odds):
    rows = []
    for bracket in BRACKET_ORDER:
        model_p = model_probs.get(bracket, np.nan)
        yes_odds = kalshi_yes_odds.get(bracket, np.nan)
        market_p = american_to_implied_prob(yes_odds)
        edge = model_p - market_p if pd.notna(market_p) else np.nan

        rows.append({
            "Bracket": bracket,
            "Model Prob %": round(model_p * 100, 1),
            "Kalshi YES Odds": yes_odds,
            "Market Prob %": round(market_p * 100, 1) if pd.notna(market_p) else np.nan,
            "Edge % Pts": round(edge * 100, 1) if pd.notna(edge) else np.nan,
            "is_tail": bracket in ["78 or below", "87 or above"],
            "is_middle": bracket in ["79-80", "81-82", "83-84"],
        })
    return rows


def bracket_priority(row, prefer_middle_brackets=True):
    edge = row["Edge % Pts"] / 100.0 if pd.notna(row["Edge % Pts"]) else -999
    middle_bonus = 0.01 if prefer_middle_brackets and row["is_middle"] else 0.0
    tail_penalty = -0.005 if row["is_tail"] else 0.0
    return edge + middle_bonus + tail_penalty


def auto_generate_ladder(rows, cfg: LadderConfig):
    candidates = []
    for row in rows:
        edge = row["Edge % Pts"] / 100.0 if pd.notna(row["Edge % Pts"]) else np.nan
        threshold = cfg.tail_edge_threshold if row["is_tail"] else cfg.core_edge_threshold

        if pd.notna(edge) and edge >= threshold:
            if row["is_tail"] and not cfg.include_tails:
                continue
            candidates.append(row)

    candidates = sorted(
        candidates,
        key=lambda r: bracket_priority(r, cfg.prefer_middle_brackets),
        reverse=True
    )

    selected = candidates[:cfg.num_core_brackets]
    stakes = [cfg.stake1, cfg.stake2, cfg.stake3]

    ladder = []
    total_risk = 0.0

    for i, row in enumerate(selected):
        stake = stakes[min(i, len(stakes) - 1)]
        if total_risk + stake > cfg.max_total_risk:
            break

        ladder.append({
            "Bracket": row["Bracket"],
            "Side": "YES",
            "Stake": stake,
            "Model Prob %": row["Model Prob %"],
            "Market Prob %": row["Market Prob %"],
            "Edge % Pts": row["Edge % Pts"],
            "Kalshi YES Odds": row["Kalshi YES Odds"],
        })
        total_risk += stake

    return ladder


st.title("Kalshi Model v17 - Market Replicator")
st.caption("Lean version with Dynamic Ladder Mode")

with st.sidebar:
    st.header("Forecast Inputs")
    src1 = st.number_input("Primary forecast high", value=82.0, step=0.1)
    src2 = st.number_input("Secondary forecast high", value=81.0, step=0.1)
    src3 = st.number_input("Third forecast high", value=83.0, step=0.1)
    station_bias = st.number_input("Station bias", value=-0.8, step=0.1)
    current_temp = st.number_input("Current temp", value=76.0, step=0.1)
    expected_temp_now = st.number_input("Expected temp by now", value=77.5, step=0.1)
    hour_local = st.slider("Local hour", min_value=6, max_value=20, value=13)
    n_sims = st.selectbox("Simulations", [5000, 10000, 20000], index=1)

    st.header("Ladder Settings")
    num_core_brackets = st.selectbox("Number of core brackets", [1, 2, 3], index=1)
    include_tails = st.checkbox("Include tails", value=False)
    core_edge_threshold = st.slider("Core edge threshold", 0.01, 0.20, 0.06, 0.01)
    tail_edge_threshold = st.slider("Tail edge threshold", 0.01, 0.25, 0.10, 0.01)
    stake1 = st.number_input("Stake 1", value=10.0, step=1.0)
    stake2 = st.number_input("Stake 2", value=20.0, step=1.0)
    stake3 = st.number_input("Stake 3", value=30.0, step=1.0)
    max_total_risk = st.number_input("Max total risk", value=30.0, step=1.0)

cfg = LadderConfig(
    num_core_brackets=num_core_brackets,
    include_tails=include_tails,
    core_edge_threshold=core_edge_threshold,
    tail_edge_threshold=tail_edge_threshold,
    stake1=stake1,
    stake2=stake2,
    stake3=stake3,
    max_total_risk=max_total_risk,
)

st.subheader("Kalshi YES Odds")
c1, c2, c3 = st.columns(3)
with c1:
    odd_78 = st.number_input("78 or below", value=3233.0, step=1.0)
    odd_79_80 = st.number_input("79-80", value=1328.0, step=1.0)
with c2:
    odd_81_82 = st.number_input("81-82", value=233.0, step=1.0)
    odd_83_84 = st.number_input("83-84", value=117.0, step=1.0)
with c3:
    odd_85_86 = st.number_input("85-86", value=354.0, step=1.0)
    odd_87 = st.number_input("87 or above", value=3233.0, step=1.0)

kalshi_yes_odds = {
    "78 or below": odd_78,
    "79-80": odd_79_80,
    "81-82": odd_81_82,
    "83-84": odd_83_84,
    "85-86": odd_85_86,
    "87 or above": odd_87,
}

model = simulate_bracket_probs(
    src1=src1,
    src2=src2,
    src3=src3,
    station_bias=station_bias,
    current_temp=current_temp,
    expected_temp_now=expected_temp_now,
    hour_local=hour_local,
    n_sims=n_sims,
)

rows = build_market_table(model["bracket_probs"], kalshi_yes_odds)
ladder = auto_generate_ladder(rows, cfg)

m1, m2, m3 = st.columns(3)
m1.metric("Base High", model["base_high"])
m2.metric("Adjusted High", model["adjusted_high"])
m3.metric("Sigma", model["sigma"])

st.subheader("Bracket Table")
df_rows = pd.DataFrame(rows).drop(columns=["is_tail", "is_middle"])
st.dataframe(df_rows, use_container_width=True)

st.subheader("Dynamic Ladder")
if ladder:
    st.dataframe(pd.DataFrame(ladder), use_container_width=True)
else:
    st.info("No brackets currently meet your edge threshold.")

st.subheader("Saved Strategy Logic")
st.write({
    "num_core_brackets": cfg.num_core_brackets,
    "include_tails": cfg.include_tails,
    "core_edge_threshold": cfg.core_edge_threshold,
    "tail_edge_threshold": cfg.tail_edge_threshold,
    "stakes": [cfg.stake1, cfg.stake2, cfg.stake3],
    "max_total_risk": cfg.max_total_risk,
})
