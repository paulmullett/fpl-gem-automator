"""
fpl_funcs.py — Core Mathematical & Utility Functions Engine

This module serves as the quantitative foundation for the FPL decision pipeline.
It handles:
1. Expected Minutes (xMins) estimation using sigmoid logistic curves.
2. Bayesian League Strength Translation for foreign arrivals.
3. Player data dictionary normalization.
4. Base Expected Value (EV) calculation incorporating Poisson clean-sheet decay,
   sigmoid appearance points, DefCon BPS math, and continuous finisher curves.
5. Macro 4-GW Horizon projections and dynamic bench discounting.
6. Algorithmic chip threshold evaluations.
"""

import math
from typing import Dict, Any, Optional

def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely converts input values to float, defaulting on None or ValueError."""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default

def estimate_xmins(p: Dict[str, Any]) -> float:
    """
    Estimates a player's expected minutes (xMins) using a logistic sigmoid curve
    calibrated against player cost, ownership, position, and FPL playing status.
    
    Formula:
        x = 2.5 * (Effective_Cost - (Base_Cost + 0.5))
        Sigmoid xMins = 90.0 / (1.0 + exp(-x))
    """
    chance_raw = p.get("chance_of_playing_next_round")
    
    # Handle non-available status or explicit 0% chance
    if chance_raw is not None and str(chance_raw) == "0":
        return 0.0
    if p.get("status") not in ["a", "d", ""]:
        return 0.0

    own = _safe_float(p.get("own"), 0.0)
    cost = _safe_float(p.get("cost"), 4.0)
    pos_id = p.get("pos_id", 3)
    
    # Position baseline cost floor (GK/DEF = 4.0, MID/FWD = 4.5)
    base_cost = 4.0 if pos_id in [1, 2] else 4.5
    
    # Ownership acts as a proxy for manager consensus on role security
    own_boost = min(1.5, (own / 10.0))
    effective_cost = cost + own_boost
    
    # Calculate sigmoid expected minutes
    x = 2.5 * (effective_cost - (base_cost + 0.5))
    raw_xmins = 90.0 / (1.0 + math.exp(-x))
    
    # Apply FPL yellow-flag percentage multipliers if present
    if chance_raw is not None:
        chance_str = str(chance_raw)
        if chance_str == "25": raw_xmins *= 0.25
        elif chance_str == "50": raw_xmins *= 0.50
        elif chance_str == "75": raw_xmins *= 0.75
        
    return min(90.0, max(0.0, raw_xmins))

# Matrix mapping domestic/foreign leagues to position-specific strength coefficients
LEAGUE_BASE_STRENGTHS = {
    "Champions_League": {1: 0.98, 2: 0.96, 3: 0.96, 4: 0.96},
    "Bundesliga":       {1: 0.92, 2: 0.92, 3: 0.88, 4: 0.88},
    "La_Liga":          {1: 0.95, 2: 0.93, 3: 0.87, 4: 0.85},
    "Serie_A":          {1: 0.94, 2: 0.94, 3: 0.86, 4: 0.84},
    "Championship":     {1: 0.85, 2: 0.85, 3: 0.88, 4: 0.89},
    "Ligue_1":          {1: 0.90, 2: 0.88, 3: 0.82, 4: 0.80},
    "Eredivisie":       {1: 0.85, 2: 0.82, 3: 0.75, 4: 0.70},
    "Pro_League":       {1: 0.82, 2: 0.80, 3: 0.74, 4: 0.68},
    "Other_Foreign":    {1: 0.80, 2: 0.75, 3: 0.70, 4: 0.65},
    "Premier_League":   {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00}
}

def calculate_tier1_translation_factor(p: Dict[str, Any]) -> float:
    """
    Calculates Bayesian translation scaling for players transferring from foreign leagues.
    Adjusts underlying expected output based on source competition tier, age adaptation curve,
    and team expected goals (xG/xGA) scaling.
    """
    league = p.get("source_league", "Premier_League")
    if league == "Premier_League":
        return 1.00
        
    pos_id = p.get("pos_id", 3)
    league_matrix = LEAGUE_BASE_STRENGTHS.get(league, {})
    base_coef = league_matrix.get(pos_id, 0.75)
    
    # Age adaptation modifier
    age = int(p.get("age", 25))
    if age <= 22:
        age_modifier = 1.05  # High developmental ceiling
    elif age >= 29:
        age_modifier = 0.92  # Adaptation/physical decline risk
    else:
        age_modifier = 1.00

    # Team dominance ratio scaling (comparing former vs new club attacking/defensive strength)
    try:
        if pos_id in [3, 4]:
            prev_xg = max(0.1, _safe_float(p.get("prev_team_xg_for_90"), 1.35))
            new_xg = _safe_float(p.get("new_team_xg_for_90"), 1.35)
            xg_scaling = min(1.2, max(0.8, new_xg / prev_xg))
        else:
            prev_xga = _safe_float(p.get("prev_team_xga_for_90"), 1.35)
            new_xga = max(0.1, _safe_float(p.get("new_team_xga_for_90"), 1.35))
            xg_scaling = min(1.2, max(0.8, prev_xga / new_xga))
    except Exception:
        xg_scaling = 1.0

    # Exception logic for returning premier assets with recent elite European peaks
    if p.get("has_stale_pl_history") and p.get("recent_european_peak", False):
        return 0.94

    final_multiplier = base_coef * age_modifier * xg_scaling
    return max(0.65, min(0.98, final_multiplier))

def normalize_player(raw_p: Dict[str, Any], teams_map: Optional[Dict[int, str]] = None,
                     element_types_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    Coalesces raw FPL API bootstrap element dictionaries into clean, standardized numeric formats.
    """
    p = {}
    p["id"] = int(raw_p.get("id"))
    p["name"] = raw_p.get("web_name") or raw_p.get("name") or "Unknown"
    
    team_id = raw_p.get("team")
    p["team_id"] = int(team_id) if team_id is not None else None
    p["team"] = teams_map.get(team_id, "UNK") if teams_map else str(raw_p.get("team", "UNK"))
    
    pos_id = raw_p.get("element_type") or raw_p.get("position") or raw_p.get("pos_id")
    p["pos_id"] = int(pos_id) if pos_id is not None else 3
    p["pos"] = element_types_map.get(p["pos_id"], "UNK") if element_types_map else str(p["pos_id"])

    p["cost"] = _safe_float(raw_p.get("now_cost") or raw_p.get("cost")) / 10.0
    p["selling_price"] = p["cost"]  # Fallback for solver budget calculations
    p["status"] = raw_p.get("status", "")
    p["news"] = raw_p.get("news", "")

    p["ep_next"] = _safe_float(raw_p.get("ep_next") or raw_p.get("ep") or 0.0)
    p["form"] = _safe_float(raw_p.get("form") or 0.0)
    p["total_points"] = int(raw_p.get("total_points") or 0)
    p["own"] = _safe_float(raw_p.get("selected_by_percent") or raw_p.get("own") or 0.0)
    p["chance_of_playing_next_round"] = raw_p.get("chance_of_playing_next_round", "")

    p["xgi_90"] = _safe_float(raw_p.get("expected_goal_involvements_per_90") or 0.0)
    p["xgc_90"] = _safe_float(raw_p.get("expected_goals_conceded_per_90") or 1.35)
    if p["xgc_90"] <= 0.0: p["xgc_90"] = 1.35

    p["cost_change_start"] = int(raw_p.get("cost_change_start") or 0)
    p["source_league"] = raw_p.get("source_league", "Premier_League")
    p["age"] = int(raw_p.get("age") or 25)
    p["has_stale_pl_history"] = bool(raw_p.get("has_stale_pl_history", False))
    p["recent_european_peak"] = bool(raw_p.get("recent_european_peak", False))

    p["transfers_in_event"] = int(raw_p.get("transfers_in_event") or 0)
    p["transfers_out_event"] = int(raw_p.get("transfers_out_event") or 0)

    return p

def get_gameweek_state(bootstrap_data: Dict[str, Any]):
    """Returns (active_gw, target_gw) based on FPL API event states."""
    current_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_next")), None)
    
    target_gw = next_gw_id or current_gw_id or 1
    active_gw = current_gw_id or (target_gw if target_gw > 1 else 1)
    
    return active_gw, target_gw

def get_base_ev(p: Dict[str, Any], xmins_overrides: Optional[Dict[str, float]] = None, 
                weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculates standard single-gameweek Expected Value (EV).
    Blends:
    - Bayesian shrunken expected goal involvements (xGI).
    - Sigmoid 60-minute appearance point thresholds.
    - Exponential Poisson clean sheet decay.
    - DefCon BPS and save volume bonuses.
    - Continuous market-priced finisher scaling.
    """
    # Intelligent parameter handling if weights and overrides are passed swapped
    if isinstance(xmins_overrides, dict) and "xgi_weight" in xmins_overrides and weights is None:
        weights = xmins_overrides
        xmins_overrides = {}
        
    if weights is None: weights = {}
    if xmins_overrides is None: xmins_overrides = {}

    pid_str = str(p.get("id"))
    if pid_str in xmins_overrides:
        xmins = float(xmins_overrides[pid_str])
    else:
        xmins = estimate_xmins(p)
        
    if xmins < 5.0:
        return 0.0

    ep = _safe_float(p.get("ep_next"), 0.0)
    xgi = _safe_float(p.get("xgi_90"), 0.0)
    xgc = _safe_float(p.get("xgc_90"), 1.35)
    cost = _safe_float(p.get("cost"), 4.0)
    own = _safe_float(p.get("own"), 0.0)

    pos_id = p.get("pos_id", 3)
    mins_factor = xmins / 90.0

    # 1. Bayesian Shrinkage towards positional baseline xGI
    baseline_xgi = 0.01 if pos_id == 1 else (0.08 if pos_id == 2 else (0.25 if pos_id == 3 else 0.35))
    cost_threshold = 4.0 if pos_id in [1, 2] else 4.5
    cost_premium = max(0.0, cost - cost_threshold)
    confidence = min(1.0, (own / 15.0) + (cost_premium / 2.0))
    
    translation_mult = calculate_tier1_translation_factor(p)
    adjusted_xgi = xgi * translation_mult
    shrunken_xgi = (adjusted_xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    # 2. Appearance Points (Logistic threshold around 60 mins)
    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)

    # 3. Poisson Clean Sheet Engine (Requires 60 mins played)
    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    
    if pos_id in [1, 2]: cs_points = (cs_prob * 4.0) * prob_60
    elif pos_id == 3: cs_points = (cs_prob * 1.0) * prob_60
    else: cs_points = 0.0

    # 4. DefCon BPS & Save Volume Modeling
    extra_defensive_points = 0.0
    if pos_id == 1:
        estimated_saves = max(1.5, (xgc * 1.4))
        extra_defensive_points = (estimated_saves / 3.0) * 0.33 * mins_factor
    elif pos_id == 2:
        extra_defensive_points = 0.22 * mins_factor if cost >= 5.5 else 0.08

    # 5. Attacking Points & Continuous Finisher Curve
    market_premium_factor = 1.0 + (max(0, cost - 5.5) * 0.04)
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium_factor

    raw_ev = app_points + attacking_points + cs_points + extra_defensive_points
    
    # Blended EV output based on state weights
    xgi_mult = weights.get("xgi_weight", 0.70)
    final_ev = (raw_ev * xgi_mult) + (ep * (1.0 - xgi_mult))
    return max(0.0, final_ev)

def get_variance_penalty(xmins: float) -> float:
    """Applies a minutes volatility penalty to macro projections."""
    return 0.8 + (min(xmins, 90.0) / 90.0) * 0.2

def get_macro_ev(p: Dict[str, Any], team_avg_fdr: Dict[int, float], 
                 weights: Optional[Dict[str, float]] = None, 
                 xmins_overrides: Optional[Dict[str, float]] = None) -> float:
    """Calculates cumulative 4-Gameweek Expected Points (xP) horizon adjusted for FDR."""
    if weights is None: weights = {}
    if xmins_overrides is None: xmins_overrides = {}

    base_ev = get_base_ev(p, xmins_overrides, weights)
    if base_ev <= 0.0:
        return 0.0

    pid_str = str(p.get("id"))
    xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)

    variance_penalty = get_variance_penalty(xmins)
    ev_4gw = (base_ev * variance_penalty) * 4.0

    team_id = p.get("team_id")
    avg_fdr = team_avg_fdr.get(team_id, 3.0) if team_avg_fdr and team_id in team_avg_fdr else 3.0
    fdr_impact = weights.get("fdr_impact_factor", 0.10)
    fdr_multiplier = 1.0 + ((3.0 - avg_fdr) * fdr_impact)

    return ev_4gw * fdr_multiplier

def calculate_dynamic_bench_discount(starters: list, xmins_overrides: Optional[Dict[str, float]] = None) -> float:
    """Calculates dynamic bench discount based on starter rotation risk (scales 0.01 to 0.20)."""
    if not starters:
        return 0.01
    if xmins_overrides is None:
        xmins_overrides = {}

    total_rotation_risk = 0.0
    for p in starters:
        pid_str = str(p.get("id"))
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        risk = max(0.0, (90.0 - xmins) / 90.0)
        total_rotation_risk += risk

    dynamic_discount = 0.01 + (total_rotation_risk * 0.03)
    return round(min(0.20, dynamic_discount), 4)

def get_ensemble_ev(p: Dict[str, Any], xmins_overrides: Optional[Dict[str, float]] = None, 
                    market_data: Optional[Dict[str, Any]] = None, 
                    weights: Optional[Dict[str, float]] = None) -> float:
    """Blends structural EV with live bookmaker odds and short-term momentum metrics."""
    if xmins_overrides is None: xmins_overrides = {}

    ev_a = get_base_ev(p, xmins_overrides, weights)
    
    # Adjust structural EV using live bookmaker odds if available
    if market_data and p.get("team") in market_data:
        m_metrics = market_data[p.get("team")]
        pos_id = p.get("pos_id", 3)
        if pos_id in [1, 2]:
            market_cs_mult = m_metrics.get("cs_prob", 0.35) / 0.35
            ev_a *= (0.75 + (0.25 * market_cs_mult))
        else:
            market_xg_mult = m_metrics.get("xG", 1.35) / 1.35
            ev_a *= (0.75 + (0.25 * market_xg_mult))

    form = _safe_float(p.get("form"), 0.0)
    ep = _safe_float(p.get("ep_next"), 0.0)
    ev_b = max(0.0, (form * 0.6) + (ep * 0.4))
    
    return (0.70 * ev_a) + (0.30 * ev_b)

def evaluate_chip_thresholds(starters: list, bench: list, ev_matrix: dict, active_chip: str) -> list:
    """Scans projections against mathematical thresholds to recommend optimal chip timing."""
    if active_chip and active_chip != "NONE":
        return [f"CHIP ACTIVE: {active_chip}"]

    recommendations = []

    # 1. Bench Boost: Bench reserves project > 12.0 total EV
    bench_ev = sum(ev_matrix[p["id"]][0] for p in bench)
    if bench_ev >= 12.0:
        recommendations.append(f"BENCH BOOST THRESHOLD MET: Bench reserves project an elite {bench_ev:.1f} combined xP.")

    # 2. Triple Captain: Single asset projects > 8.5 1-GW EV
    best_starter = max(starters, key=lambda p: ev_matrix[p["id"]][0])
    best_ev = ev_matrix[best_starter["id"]][0]
    if best_ev >= 8.5:
        recommendations.append(f"TRIPLE CAPTAIN THRESHOLD MET: {best_starter['name']} projects a massive {best_ev:.1f} xP ceiling.")

    # 3. Free Hit / Wildcard Warning: Starters < 9 or total 1-GW EV <= 35.0
    active_starters = len([p for p in starters if ev_matrix[p["id"]][0] >= 2.0])
    starting_ev = sum(ev_matrix[p["id"]][0] for p in starters)
    if active_starters <= 8 or starting_ev <= 35.0:
        recommendations.append(f"FREE HIT / WILDCARD WARNING: Squad decay detected. Only {active_starters} viable starters. 1-GW EV is critical ({starting_ev:.1f} xP).")

    if not recommendations:
        recommendations.append("Hold all chips. No mathematical variance thresholds met for the current gameweek.")

    return recommendations
