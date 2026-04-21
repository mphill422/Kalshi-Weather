"""
V5.14 Trust Score Module
========================
Computes a composite 0-100 trust score per signal, combining 7 factors
that gate whether a green edge signal should actually be bet.

Philosophy: The existing BET/AVOID system flags edge + ensemble only.
This module adds contextual factors (bracket adjacency to 2DC, NBM active,
GFS-NWS gap, bias adj magnitude, MAE, Model %) to catch cases where the
model's own output contradicts itself (e.g. Phoenix 2DC=93-94 but signaled
91-92) or where there are structural problems with the forecast.

Designed to run in PARALLEL with existing BET/AVOID during A/B period.
Outputs are additive — they never remove existing columns.

Author: Mike + Claude
Version: V5.14 initial
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import math


# ----------------------------- Constants ------------------------------

# Weighting of each factor in final composite score (must sum to 100)
FACTOR_WEIGHTS = {
    "adjacency": 28,
    "ensemble":  17,
    "mae":       17,
    "nbm":       11,
    "gfs_gap":   11,
    "bias":       9,
    "model_pct":  7,
}
assert sum(FACTOR_WEIGHTS.values()) == 100, "Factor weights must sum to 100"

# Tier thresholds on final composite (0-100)
TIER_BET      = 75   # >= 75 is full green
TIER_CAUTION  = 55   # 55-74 is caution
# < 55 is skip

# Stake-suggestion tiers based on Model %. These are ADVISORY only.
# Returned as a note alongside the existing Kelly figure; never enforced.
STAKE_TIER_BANKROLL_PCT = {
    "conviction": 0.05,   # Model% >= 50: up to 5% bankroll
    "solid":      0.03,   # Model% 25-49: up to 3%
    "lottery":    0.01,   # Model% 10-24: up to 1%  (lottery ticket)
    "tail":       0.005,  # Model% <10:   0.5% (pure tail flyer)
}


# ----------------------------- Data class ------------------------------

@dataclass
class SignalInputs:
    """
    Inputs needed to compute a trust score for a single bet candidate.

    All values are READ from the existing model outputs — no new data
    sources required. Caller populates these fields from the same
    dataframes/dicts already used in streamlit_app.py.
    """
    # Bet identity
    city: str
    bracket_label: str              # e.g. "72 or below", "81-82"
    direction: str                  # "YES" or "NO"

    # Bracket / model distribution
    two_degree_call: str            # e.g. "72-73" or "72 or below"
    bracket_midpoint: Optional[float]  # numeric center of the bet bracket; None for open-ended
    twodc_midpoint: Optional[float]    # numeric center of 2DC bracket; None for open-ended

    # Model outputs already on the page
    model_pct: float                # 0-100 (the "Model %" column)
    edge_cents: float               # the +xxc edge
    ensemble_tier: str              # "HIGH" / "MED" / "LOW"
    mae_color: str                  # "green" / "yellow" / "red"
    nbm_active: bool                # True = NBM fired, False = sigma fallback
    nws_forecast_f: Optional[float] # NWS forecast in F
    gfs_ensemble_f: Optional[float] # GFS ensemble mean in F
    bias_adj_f: float               # Signed bias adjustment in F


@dataclass
class TrustScoreResult:
    """Output of trust score computation."""
    composite: float                        # 0-100 composite
    tier: str                               # "BET" / "CAUTION" / "SKIP"
    factor_scores: dict[str, float]         # 0-100 per factor, for debugging/logging
    warnings: list[str] = field(default_factory=list)
    stake_suggestion_label: str = ""        # "conviction"/"solid"/"lottery"/"tail"
    stake_suggestion_pct: float = 0.0       # Fraction of bankroll suggested

    def as_row(self) -> dict:
        """Flatten for dataframe display or Supabase logging."""
        out = {
            "trust_score": round(self.composite, 1),
            "trust_tier": self.tier,
            "trust_warnings": "; ".join(self.warnings) if self.warnings else "",
            "stake_tier": self.stake_suggestion_label,
            "stake_pct_bankroll": self.stake_suggestion_pct,
        }
        for k, v in self.factor_scores.items():
            out[f"ts_{k}"] = round(v, 1)
        return out


# ------------------------- Individual factor scorers ------------------------

def _score_adjacency(inp: SignalInputs) -> tuple[float, Optional[str]]:
    """
    How close is the bet bracket to the model's 2 Degree Call?

    - Same bracket: 100  (perfect alignment)
    - 1 bracket away (adjacent): 70
    - 2 brackets away: 30
    - 3+ brackets away: 0

    Open-ended brackets ("X or below", "Y or above") treat the 2DC direction
    as a sanity check — if 2DC is 72-73 and bet is "72 or below", they are
    considered adjacent (distance 1) because the 72-below range INCLUDES 72.
    """
    # Normalize strings
    bet = (inp.bracket_label or "").strip().lower()
    twodc = (inp.two_degree_call or "").strip().lower()

    if not bet or not twodc:
        return 50.0, "Bracket or 2DC missing — adjacency not computable"

    # Exact text match is the cleanest case
    if bet == twodc:
        return 100.0, None

    # If we have midpoints, use numeric distance
    if inp.bracket_midpoint is not None and inp.twodc_midpoint is not None:
        delta = abs(inp.bracket_midpoint - inp.twodc_midpoint)
        # Brackets are 2F wide, so distance ~= delta/2
        bracket_dist = delta / 2.0
        if bracket_dist <= 0.5:
            return 100.0, None
        if bracket_dist <= 1.5:
            return 70.0, f"Bet bracket adjacent to 2DC ({inp.two_degree_call})"
        if bracket_dist <= 2.5:
            return 30.0, f"Bet bracket 2 away from 2DC ({inp.two_degree_call}) — tail bet"
        return 0.0, f"Bet bracket far from 2DC ({inp.two_degree_call}) — deep tail"

    # Fallback: no midpoints, different labels — treat as distance 1
    return 60.0, f"Bet bracket differs from 2DC ({inp.two_degree_call})"


def _score_ensemble(inp: SignalInputs) -> tuple[float, Optional[str]]:
    tier = (inp.ensemble_tier or "").upper()
    if tier == "HIGH":
        return 100.0, None
    if tier == "MED":
        return 55.0, None   # MED passes protocol but nudges score
    if tier == "LOW":
        return 0.0, "LOW ensemble — GFS members disagree widely"
    return 40.0, f"Unknown ensemble tier: {inp.ensemble_tier}"


def _score_mae(inp: SignalInputs) -> tuple[float, Optional[str]]:
    color = (inp.mae_color or "").lower()
    if color == "green":
        return 100.0, None
    if color == "yellow":
        return 50.0, "Yellow MAE — city calibration moderate"
    if color == "red":
        return 0.0, "Red MAE — city calibration poor"
    return 60.0, None


def _score_nbm(inp: SignalInputs) -> tuple[float, Optional[str]]:
    if inp.nbm_active:
        return 100.0, None
    return 0.0, "NBM not active — using sigma/normal fallback (less reliable)"


def _score_gfs_gap(inp: SignalInputs) -> tuple[float, Optional[str]]:
    if inp.nws_forecast_f is None or inp.gfs_ensemble_f is None:
        return 50.0, "Cannot compute GFS-NWS gap (missing data)"
    gap = abs(inp.nws_forecast_f - inp.gfs_ensemble_f)
    if gap < 2.0:
        return 100.0, None
    if gap < 3.5:
        return 70.0, None
    if gap < 5.0:
        return 30.0, f"GFS-NWS gap {gap:.1f}F — forecasters disagree"
    return 0.0, f"GFS-NWS gap {gap:.1f}F — large forecaster disagreement"


def _score_bias(inp: SignalInputs) -> tuple[float, Optional[str]]:
    mag = abs(inp.bias_adj_f)
    if mag < 0.5:
        return 100.0, None
    if mag < 1.5:
        return 80.0, None
    if mag < 2.5:
        return 50.0, f"Bias adj {inp.bias_adj_f:+.1f}F — notable correction"
    return 20.0, f"Bias adj {inp.bias_adj_f:+.1f}F — large correction; consensus uncertain"


def _score_model_pct(inp: SignalInputs) -> tuple[float, Optional[str]]:
    m = inp.model_pct
    if m >= 50:
        return 100.0, None
    if m >= 30:
        return 80.0, None
    if m >= 20:
        return 60.0, None
    if m >= 15:
        return 40.0, "Low Model % — lottery-ticket profile"
    if m >= 10:
        return 20.0, "Very low Model % — deep tail"
    return 0.0, "Model % below 10 — noise territory"


def _stake_suggestion(model_pct: float) -> tuple[str, float]:
    if model_pct >= 50:
        return "conviction", STAKE_TIER_BANKROLL_PCT["conviction"]
    if model_pct >= 25:
        return "solid", STAKE_TIER_BANKROLL_PCT["solid"]
    if model_pct >= 10:
        return "lottery", STAKE_TIER_BANKROLL_PCT["lottery"]
    return "tail", STAKE_TIER_BANKROLL_PCT["tail"]


# --------------------------- Main entry point -------------------------

def compute_trust_score(inp: SignalInputs) -> TrustScoreResult:
    """
    Compute the composite trust score and tier for a single signal.

    Returns a TrustScoreResult with:
      - composite 0-100
      - tier: BET / CAUTION / SKIP
      - factor_scores dict for debugging/logging
      - warnings list (human-readable flags)
      - stake suggestion tier and bankroll %
    """
    factor_scorers = {
        "adjacency": _score_adjacency,
        "ensemble":  _score_ensemble,
        "mae":       _score_mae,
        "nbm":       _score_nbm,
        "gfs_gap":   _score_gfs_gap,
        "bias":      _score_bias,
        "model_pct": _score_model_pct,
    }

    factor_scores: dict[str, float] = {}
    warnings: list[str] = []

    composite = 0.0
    for name, scorer in factor_scorers.items():
        raw, warn = scorer(inp)
        factor_scores[name] = raw
        composite += raw * (FACTOR_WEIGHTS[name] / 100.0)
        if warn:
            warnings.append(warn)

    composite = max(0.0, min(100.0, composite))

    # ---- Veto / downgrade rules ----
    # Certain severe single-factor problems force tier downgrades even if
    # the composite score looks OK. These are Mike's protocol rules hardcoded.
    vetoes: list[str] = []

    # Hard veto: LOW ensemble is a documented skip per protocol
    if (inp.ensemble_tier or "").upper() == "LOW":
        vetoes.append("SKIP_LOW_ENSEMBLE")

    # Hard veto: bet bracket 2+ brackets away from 2DC — tail bet against own model
    if factor_scores["adjacency"] <= 30.0:
        vetoes.append("SKIP_NOT_ADJACENT")

    # Downgrade: NBM off + large GFS gap = fallback mode + forecasters disagree
    if not inp.nbm_active and factor_scores["gfs_gap"] <= 30.0:
        vetoes.append("DOWNGRADE_FALLBACK_AND_GAP")

    # Downgrade: bracket not the exact 2DC match (adjacency score <100)
    #   AND model_pct is in lottery tier (<25%)
    # This catches bets like New Orleans today: off-2DC + low Model % = skip even
    # though individually each factor is borderline OK.
    if factor_scores["adjacency"] < 100.0 and inp.model_pct < 25.0:
        vetoes.append("DOWNGRADE_OFF_2DC_LOTTERY")

    # Apply tier logic
    if composite >= TIER_BET:
        tier = "BET"
    elif composite >= TIER_CAUTION:
        tier = "CAUTION"
    else:
        tier = "SKIP"

    # Veto adjustments
    if "SKIP_LOW_ENSEMBLE" in vetoes or "SKIP_NOT_ADJACENT" in vetoes:
        tier = "SKIP"
    elif tier == "BET" and vetoes:
        tier = "CAUTION"

    # Fold veto reasons into warnings for display
    if "SKIP_LOW_ENSEMBLE" in vetoes:
        # Already added by _score_ensemble
        pass
    if "SKIP_NOT_ADJACENT" in vetoes and not any("far from 2DC" in w or "2 away from 2DC" in w for w in warnings):
        warnings.append("Bet bracket too far from 2DC — skipping")
    if "DOWNGRADE_FALLBACK_AND_GAP" in vetoes:
        warnings.append("NBM fallback + large GFS gap — downgrading")
    if "DOWNGRADE_OFF_2DC_LOTTERY" in vetoes:
        warnings.append("Off-2DC + low Model % — downgrading lottery tail bet")

    stake_label, stake_pct = _stake_suggestion(inp.model_pct)

    return TrustScoreResult(
        composite=composite,
        tier=tier,
        factor_scores=factor_scores,
        warnings=warnings,
        stake_suggestion_label=stake_label,
        stake_suggestion_pct=stake_pct,
    )


# ----------------------------- Utility helpers -----------------------------

def bracket_midpoint_from_label(label: str) -> Optional[float]:
    """
    Parse bracket strings like '72-73', '72 or below', '82 or above' into
    a numeric midpoint usable for adjacency distance math.

    Returns None if parse fails.
    """
    if not label:
        return None
    s = label.lower().strip()

    # Range like "72-73"
    if "-" in s and "or" not in s:
        parts = s.split("-")
        try:
            lo = float(parts[0].strip())
            hi = float(parts[1].strip())
            return (lo + hi) / 2.0
        except ValueError:
            return None

    # "X or below" -> treat midpoint as X - 0.5 (represents "X or anything cooler")
    if "or below" in s:
        try:
            num = float(s.split("or below")[0].strip())
            return num - 0.5
        except ValueError:
            return None

    # "X or above" -> X + 0.5
    if "or above" in s:
        try:
            num = float(s.split("or above")[0].strip())
            return num + 0.5
        except ValueError:
            return None

    # Bare number
    try:
        return float(s)
    except ValueError:
        return None
