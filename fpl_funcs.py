"""
fpl_funcs.py — Core Mathematical & Utility Functions Engine (Pre-Season Calibrated & Soft-Capped)
"""

import math
from typing import Dict, Any, Optional

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception:
        return default

def estimate_xmins(p: Dict[str, Any]) -> float:
    chance_raw = p.get("chance_of_playing_next_round")
    if chance_raw is not None and str(chance_raw) == "0": return 0.0
    if p.get("status") not in ["a", "d", ""]: return 0.0

    own = _safe_float(p.get("own"), 0.0)
    cost = _safe_float(p.get("cost"), 4.0)
    pos_id = p.get("pos_id", 3)
    team = p.get("team", "UNK")
    
    # PREMIUM CLUB TAX: Mathematically differentiates an academy kid at Arsenal from a starter at Fulham
    PREMIUM_TEAMS = {"ARS", "MCI", "LIV", "CHE", "TOT", "MUN"}
    
    if pos_id in [1, 2]: # GK/DEF
        base_cost = 4.5 if team in PREMIUM_TEAMS else 4.0
    else: # MID/FWD
        base_cost = 5.5 if team in PREMIUM_TEAMS else 4.5

    own_boost = min(1.5, (own / 10.0))
    effective_cost = cost + own_boost
    
    x = 2.5 * (effective_cost - (base_cost + 0.5))
    base_xmins = 90.0 / (1.0 + math.exp(-x))
    
    # MARKET TRUTH OVERRIDE: High ownership instantly validates starting status
    if own >= 10.0:
        base_xmins = max(base_xmins, 80.0)
        
    # PREMIUM PRICE OVERRIDE: Top-tier transfers are guaranteed to play regardless of initial low ownership
    if cost >= 7.0:
        base_xmins = max(base_xmins, 80.0)

    if chance_raw is not None:
        chance_str = str(chance_raw)
        if chance_str == "25": base_xmins *= 0.25
        elif chance_str == "50": base_xmins *= 0.50
        elif chance_str == "75": base_xmins *= 0.75

    return min(90.0, max(0.0, base_xmins))

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
    league = p.get("source_league", "Premier_League")
    if league == "Premier_League": return 1.00

    pos_id = p.get("pos_id", 3)
    league_matrix = LEAGUE_BASE_STRENGTHS.get(league, {})
    base_coef = league_matrix.get(pos_id, 0.75)

    age = int(p.get("age", 25))
    if age <= 22: age_modifier = 1.05
    elif age >= 29: age_modifier = 0.92
    else: age_modifier = 1.00

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

    if p.get("has_stale_pl_history") and p.get("recent_european_peak", False): return 0.94

    final_multiplier = base_coef * age_modifier * xg_scaling
    return max(0.65, min(0.98, final_multiplier))

def normalize_player(raw_p: Dict[str, Any], teams_map: Optional[Dict[int, str]] = None,
                     element_types_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
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
    p["selling_price"] = p["cost"]
    p["status"] = raw_p.get("status", "")
    p["news"] = raw_p.get("news", "")

    p["ep_next"] = _safe_float(raw_p.get("ep_next") or raw_p.get("ep") or 0.0)
    p["form"] = _safe_float(raw_p.get("form") or 0.0)
    p["total_points"] = int(raw_p.get("total_points") or 0)
    p["own"] = _safe_float(raw_p.get("selected_by_percent") or raw_p.get("own") or 0.0)

    p["top_10k_eo"] = _safe_float(raw_p.get("top_10k_eo") or p["own"])
    p["price_delta_prob"] = _safe_float(raw_p.get("price_delta_prob") or 0.0)

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

    p["transfers_in_event"] = int(raw_p.get("transfers_in_event") or 0)
    p["transfers_out_event"] = int(raw_p.get("transfers_out_event") or 0)
    
    # NEW: Dead-Ball Data Extraction
    p["pen_order"] = raw_p.get("penalties_order")
    p["set_piece_order"] = raw_p.get("corners_and_indirect_freekicks_order")

    return p

def get_gameweek_state(bootstrap_data: Dict[str, Any]):
    current_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_next")), None)
    target_gw = next_gw_id or current_gw_id or 1
    active_gw = current_gw_id or (target_gw if target_gw > 1 else 1)
    return active_gw, target_gw

def get_live_price_deltas(players_dict: dict) -> dict:
    deltas = {}
    for pid, p in players_dict.items():
        transfers_in = _safe_float(p.get("transfers_in_event"), 0.0)
        transfers_out = _safe_float(p.get("transfers_out_event"), 0.0)
        net_transfers = transfers_in - transfers_out
        
        own_percent = max(0.5, _safe_float(p.get("own"), 1.0))
        
        # Ownership-relative velocity metric
        velocity = net_transfers / (own_percent * 2500.0)
        
        # Sigmoid probability scaling for price movement (-1.0 to +1.0)
        if velocity > 0:
            prob = 1.0 / (1.0 + math.exp(-1.5 * (velocity - 1.0)))
        else:
            prob = -1.0 / (1.0 + math.exp(1.5 * (velocity + 1.0)))
            
        deltas[pid] = round(max(-1.0, min(1.0, prob)), 3)
        
    return deltas

def get_base_ev(p: Dict[str, Any], xmins_overrides: Optional[Dict[str, float]] = None, 
                weights: Optional[Dict[str, float]] = None,
                market_data: Optional[Dict[str, Any]] = None) -> float:
    if isinstance(xmins_overrides, dict) and "xgi_weight" in xmins_overrides and weights is None:
        weights = xmins_overrides
        xmins_overrides = {}

    if weights is None: weights = {}
    if xmins_overrides is None: xmins_overrides = {}

    pid_str = str(p.get("id"))
    xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
    if xmins < 5.0: return 0.0

    raw_xgi = _safe_float(p.get("xgi_90"), 0.0)
    
    # ASYMPTOTIC SOFT-CAP: Dampens extreme small-sample anomalies without limiting true elites
    if raw_xgi > 1.0:
        xgi = 1.0 + ((raw_xgi - 1.0) * 0.4)
    else:
        xgi = raw_xgi
        
    xgc = _safe_float(p.get("xgc_90"), 1.35)
    cost = _safe_float(p.get("cost"), 4.0)
    own = _safe_float(p.get("own"), 0.0)
    ep_next = _safe_float(p.get("ep_next"), 0.0)
    pos_id = p.get("pos_id", 3)

    if xgi <= 0.01:
        if pos_id == 4:    xgi = 0.40 + max(0.0, cost - 6.0) * 0.04
        elif pos_id == 3:  xgi = 0.22 + max(0.0, cost - 5.5) * 0.03
        elif pos_id == 2:  xgi = 0.08 + max(0.0, cost - 4.5) * 0.02
        else:              xgi = 0.01

    mins_factor = xmins / 90.0

    team_name = p.get("team", "")
    team_xg_base = market_data[team_name].get("xG", 1.5) if market_data and team_name in market_data else 1.5
    baseline_xgi = team_xg_base * 0.01 if pos_id == 1 else (team_xg_base * 0.06 if pos_id == 2 else (team_xg_base * 0.18 if pos_id == 3 else team_xg_base * 0.30))

    cost_threshold = 4.0 if pos_id in [1, 2] else 4.5
    cost_premium = max(0.0, cost - cost_threshold)
    
    # MULTI-VARIATE CONFIDENCE: Blends Cost, Ownership, and FPL's native expectation
    raw_confidence = (cost_premium / 4.0) + (own / 30.0) + (max(0, ep_next - 2.0) * 0.15)
    
    translation_mult = calculate_tier1_translation_factor(p)
    # Protect elite foreign transfers with strong UEFA coefficients from washing out due to low initial ownership
    if translation_mult > 0.90 and own < 5.0 and pos_id in [3, 4]:
        confidence = min(1.0, max(raw_confidence, 0.60))
    else:
        confidence = min(1.0, max(raw_confidence, 0.25))

    adjusted_xgi = xgi * translation_mult
    shrunken_xgi = (adjusted_xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)

    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0

    if pos_id in [1, 2]: cs_points = (cs_prob * 4.0) * prob_60
    elif pos_id == 3: cs_points = (cs_prob * 1.0) * prob_60
    else: cs_points = 0.0

    extra_defensive_points = 0.0
    if pos_id == 1:
        estimated_saves = max(1.5, (xgc * 1.4))
        extra_defensive_points = (estimated_saves / 3.0) * 0.33 * mins_factor
    elif pos_id == 2:
        extra_defensive_points = 0.22 * mins_factor if cost >= 5.5 else 0.08

    market_premium_factor = 1.0 + (max(0, cost - 5.5) * 0.04)
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium_factor

    final_ev = app_points + attacking_points + cs_points + extra_defensive_points
    return max(0.0, final_ev)

def get_variance_penalty(xmins: float) -> float:
    return 0.8 + (min(xmins, 90.0) / 90.0) * 0.2

def get_ensemble_ev(p: Dict[str, Any], xmins_overrides: Optional[Dict[str, float]] = None, 
                    market_data: Optional[Dict[str, Any]] = None, 
                    weights: Optional[Dict[str, float]] = None,
                    risk_posture: str = "NEUTRAL") -> float:
    if xmins_overrides is None: xmins_overrides = {}
    ev_a = get_base_ev(p, xmins_overrides, weights, market_data)

    pos_id = p.get("pos_id", 3)
    sigma = ev_a * (0.45 if pos_id == 3 else (0.4 if pos_id == 4 else 0.3))
    if risk_posture == "SHIELD":
        ev_a -= (sigma * 0.15)
    elif risk_posture == "CHASE":
        ev_a += (sigma * 0.15)

    if market_data and p.get("team") in market_data:
        m_metrics = market_data[p.get("team")]
        if pos_id in [1, 2]:
            market_cs_mult = m_metrics.get("cs_prob", 0.35) / 0.35
            ev_a *= (0.75 + (0.25 * market_cs_mult))
        else:
            market_xg_mult = m_metrics.get("xG", 1.35) / 1.35
            ev_a *= (0.75 + (0.25 * market_xg_mult))

    form = _safe_float(p.get("form"), 0.0)
    
    if form > 0.1:
        ep = _safe_float(p.get("ep_next"), 0.0)
        ev_b = max(0.0, (form * 0.6) + (ep * 0.4))
        return max(0.0, (0.70 * ev_a) + (0.30 * ev_b))
    else:
        return max(0.0, ev_a)

def get_macro_ev(p: Dict[str, Any], team_avg_fdr: Dict[int, float], 
                 weights: Optional[Dict[str, float]] = None, 
                 xmins_overrides: Optional[Dict[str, float]] = None,
                 market_data: Optional[Dict[str, Any]] = None,
                 horizons: int = 8,
                 risk_posture: str = "NEUTRAL") -> float:
    if weights is None: weights = {}
    if xmins_overrides is None: xmins_overrides = {}

    base_ev = get_base_ev(p, xmins_overrides, weights, market_data)
    pos_id = p.get("pos_id", 3)
    sigma = base_ev * (0.45 if pos_id == 3 else (0.4 if pos_id == 4 else 0.3))

    if risk_posture == "SHIELD":
        base_ev -= (sigma * 0.15)
    elif risk_posture == "CHASE":
        base_ev += (sigma * 0.15)

    if base_ev <= 0.0: return 0.0

    pid_str = str(p.get("id"))
    xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
    variance_penalty = get_variance_penalty(xmins)
    ev_macro = (base_ev * variance_penalty) * float(horizons)

    team_id = p.get("team_id")
    avg_fdr = team_avg_fdr.get(team_id, 3.0) if team_avg_fdr and team_id in team_avg_fdr else 3.0
    fdr_impact = weights.get("fdr_impact_factor", 0.10)
    fdr_multiplier = 1.0 + ((3.0 - avg_fdr) * fdr_impact)

    return ev_macro * fdr_multiplier

def evaluate_chip_thresholds(starters: list, bench: list, ev_matrix: dict, active_chip: str) -> list:
    if active_chip and active_chip != "NONE": return [f"CHIP ACTIVE: {active_chip}"]
    recommendations = []

    bench_ev = sum(ev_matrix[p["id"]][0] for p in bench)
    if bench_ev >= 12.0: recommendations.append(f"BENCH BOOST THRESHOLD MET: Bench reserves project an elite {bench_ev:.1f} combined xP.")

    best_starter = max(starters, key=lambda p: ev_matrix[p["id"]][0])
    best_ev = ev_matrix[best_starter["id"]][0]
    if best_ev >= 8.5: recommendations.append(f"TRIPLE CAPTAIN THRESHOLD MET: {best_starter['name']} projects a massive {best_ev:.1f} xP ceiling.")

    active_starters = len([p for p in starters if ev_matrix[p["id"]][0] >= 2.0])
    starting_ev = sum(ev_matrix[p["id"]][0] for p in starters)
    if active_starters <= 8 or starting_ev <= 35.0:
        recommendations.append(f"FREE HIT / WILDCARD WARNING: Squad decay detected. Only {active_starters} viable starters. 1-GW EV is critical ({starting_ev:.1f} xP).")

    if not recommendations: recommendations.append("Hold all chips. No mathematical variance thresholds met for the current gameweek.")
    return recommendations