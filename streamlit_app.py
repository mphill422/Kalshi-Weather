import numpy as np
from dataclasses import dataclass, asdict

# -----------------------------
# Kalshi Model v17 - Market Replicator
# with Dynamic Ladder Mode
# -----------------------------

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
    name: str = "Default Ladder"
    enabled: bool = True

    # strategy settings
    num_core_brackets: int = 2
    include_tails: bool = False
    core_edge_threshold: float = 0.06   # 6%
    tail_edge_threshold: float = 0.10   # 10%

    # sizing
    stake1: float = 10.0
    stake2: float = 20.0
    stake3: float = 30.0

    # exposure controls
    max_total_risk: float = 60.0
    allow_multiple_sides: bool = False  # usually False

    # optional preference
    prefer_middle_brackets: bool = True


def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    +117 -> 0.4608
    -133 -> 0.5708
    """
    if odds is None:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def choose_sigma(hour_local: int) -> float:
    if hour_local < 11:
        return 2.2
    elif hour_local < 15:
        return 1.8
    return 1.5


def apply_intraday_adjustment(adjusted_high, current_temp, expected_temp_now, hour_local):
    """
    Very light touch. No constant fiddling.
    """
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
    """
    Main v17 probability engine.
    """
    rng = np.random.default_rng(seed)

    # weighted expected high
    base_high = 0.55 * src1 + 0.30 * src2 + 0.15 * src3
    adjusted_high = base_high + station_bias

    # light intraday tweak
    adjusted_high = apply_intraday_adjustment(
        adjusted_high=adjusted_high,
        current_temp=current_temp,
        expected_temp_now=expected_temp_now,
        hour_local=hour_local,
    )

    sigma = choose_sigma(hour_local)

    sims = rng.normal(adjusted_high, sigma, n_sims)
    sims = np.rint(sims).astype(int)

    probs = {
        "78 or below": np.mean(sims <= 78),
        "79-80": np.mean((sims >= 79) & (sims <= 80)),
        "81-82": np.mean((sims >= 81) & (sims <= 82)),
        "83-84": np.mean((sims >= 83) & (sims <= 84)),
        "85-86": np.mean((sims >= 85) & (sims <= 86)),
        "87 or above": np.mean(sims >= 87),
    }

    return {
        "base_high": round(base_high, 2),
        "adjusted_high": round(adjusted_high, 2),
        "sigma": sigma,
        "bracket_probs": probs,
    }


def build_market_table(model_probs, kalshi_yes_odds):
    """
    kalshi_yes_odds example:
    {
        "78 or below": +3233,
        "79-80": +1328,
        "81-82": +233,
        "83-84": +117,
        "85-86": +354,
        "87 or above": +3233,
    }
    """
    rows = []
    for bracket in BRACKET_ORDER:
        model_p = model_probs.get(bracket, np.nan)
        yes_odds = kalshi_yes_odds.get(bracket)
        market_p = american_to_implied_prob(yes_odds) if yes_odds is not None else np.nan
        edge = model_p - market_p if not np.isnan(model_p) and not np.isnan(market_p) else np.nan

        rows.append({
            "bracket": bracket,
            "model_prob": model_p,
            "yes_odds": yes_odds,
            "market_prob": market_p,
            "edge": edge,
            "is_tail": bracket in ["78 or below", "87 or above"],
            "is_middle": bracket in ["79-80", "81-82", "83-84"],
        })
    return rows


def bracket_priority(row, prefer_middle_brackets=True):
    """
    Sort candidates sensibly.
    """
    edge = row["edge"]
    middle_bonus = 0.01 if prefer_middle_brackets and row["is_middle"] else 0.0
    tail_penalty = -0.005 if row["is_tail"] else 0.0
    return edge + middle_bonus + tail_penalty


def auto_generate_ladder(rows, cfg: LadderConfig):
    """
    Dynamic Ladder Mode:
    Save the strategy, not exact temp picks.
    Rebuild live from current model + current odds.
    """
    if not cfg.enabled:
        return []

    candidates = []
    for row in rows:
        threshold = cfg.tail_edge_threshold if row["is_tail"] else cfg.core_edge_threshold
        if row["edge"] is not None and not np.isnan(row["edge"]) and row["edge"] >= threshold:
            if not cfg.include_tails and row["is_tail"]:
                continue
            candidates.append(row)

    # sort by best edge, with mild preference for middle contracts
    candidates = sorted(
        candidates,
        key=lambda r: bracket_priority(r, cfg.prefer_middle_brackets),
        reverse=True
    )

    # keep top core brackets
    selected = candidates[:cfg.num_core_brackets]

    stakes = [cfg.stake1, cfg.stake2, cfg.stake3]
    ladder = []
    total_risk = 0.0

    for i, row in enumerate(selected):
        stake = stakes[min(i, len(stakes) - 1)]
        if total_risk + stake > cfg.max_total_risk:
            break

        ladder.append({
            "bracket": row["bracket"],
            "side": "YES",
            "stake": stake,
            "model_prob": round(row["model_prob"] * 100, 1),
            "market_prob": round(row["market_prob"] * 100, 1),
            "edge_pct_pts": round((row["edge"]) * 100, 1),
            "yes_odds": row["yes_odds"],
        })
        total_risk += stake

    return ladder


def summarize_changes(old_ladder, new_ladder):
    """
    Lets the app show 'changed since save' without breaking your saved strategy.
    """
    old_keys = {(x["bracket"], x["side"]) for x in old_ladder}
    new_keys = {(x["bracket"], x["side"]) for x in new_ladder}

    added = new_keys - old_keys
    removed = old_keys - new_keys

    return {
        "added": list(added),
        "removed": list(removed),
        "changed": bool(added or removed),
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example forecast inputs
    src1 = 82.0
    src2 = 81.0
    src3 = 83.0
    station_bias = -0.8
    current_temp = 76.0
    expected_temp_now = 77.5
    hour_local = 13

    # Example Kalshi yes odds
    kalshi_yes_odds = {
        "78 or below": 3233,
        "79-80": 1328,
        "81-82": 233,
        "83-84": 117,
        "85-86": 354,
        "87 or above": 3233,
    }

    model = simulate_bracket_probs(
        src1=src1,
        src2=src2,
        src3=src3,
        station_bias=station_bias,
        current_temp=current_temp,
        expected_temp_now=expected_temp_now,
        hour_local=hour_local,
        n_sims=10000,
        seed=42,
    )

    rows = build_market_table(
        model_probs=model["bracket_probs"],
        kalshi_yes_odds=kalshi_yes_odds,
    )

    cfg = LadderConfig(
        name="Michael Core Ladder",
        num_core_brackets=2,
        include_tails=False,
        core_edge_threshold=0.06,
        tail_edge_threshold=0.10,
        stake1=10,
        stake2=20,
        stake3=30,
        max_total_risk=30,
        prefer_middle_brackets=True,
    )

    ladder = auto_generate_ladder(rows, cfg)

    print("MODEL SUMMARY")
    print(model)
    print("\nLADDER CONFIG")
    print(asdict(cfg))
    print("\nAUTO LADDER")
    for item in ladder:
        print(item)
