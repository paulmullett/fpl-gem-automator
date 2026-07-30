#Combined re-used functions and operations
import math

# estimated minute calculations

def estimate_xmins(p):
    chance = str(p.get("chance_of_playing_next_round", ""))
    
    if chance == "0" or p.get("status") not in ["a", "d"]:
        return 0.0

    try: 
        own = float(p.get("own", 0.0))
    except (ValueError, TypeError): 
        own = 0.0
        
    try: 
        cost = float(p.get("cost", 0.0))
    except (ValueError, TypeError): 
        cost = 4.0
        
    pos_id = p.get("pos_id", 3)
    base_cost = 4.0 if pos_id in [1, 2] else 4.5
    
    own_boost = min(1.5, (own / 10.0))
    effective_cost = cost + own_boost
    
    x = 2.5 * (effective_cost - (base_cost + 0.5))
    raw_xmins = 90.0 / (1.0 + math.exp(-x))
    
    if chance == "25": raw_xmins *= 0.25
    elif chance == "50": raw_xmins *= 0.50
    elif chance == "75": raw_xmins *= 0.75
        
    return min(90.0, max(0.0, raw_xmins))

# Position-aware translation matrix (1: GK, 2: DEF, 3: MID, 4: FWD)
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

def calculate_tier1_translation_factor(p):
    league = p.get("source_league", "Premier_League")
    pos_id = p.get("pos_id", 3)
    
    # Fetch position-specific base coefficient, default to 0.75 if missing
    league_matrix = LEAGUE_BASE_STRENGTHS.get(league, {})
    base_coef = league_matrix.get(pos_id, 0.75)
    
    if league == "Premier_League":
        return 1.00
        
    try:
        age = int(p.get("age", 25))
    except (ValueError, TypeError):
        age = 25
        
    if age <= 22:
        age_modifier = 1.05
    elif age >= 29:
        age_modifier = 0.92
    else:
        age_modifier = 1.00

    # Dynamic xG Translation replacing Dominance Delta
    # Uses safe defaults (1.35) so it functions normally if data isn't ingested yet
    try:
        if pos_id in [3, 4]:
            # Attackers care about offensive volume (xG For)
            prev_xg = float(p.get("prev_team_xg_for_90", 1.35))
            new_xg = float(p.get("new_team_xg_for_90", 1.35))
            prev_xg = max(0.1, prev_xg) # Prevent division by zero
            xg_scaling = min(1.2, max(0.8, new_xg / prev_xg))
        else:
            # Defenders/GKs care about defensive solidity (xG Against)
            prev_xga = float(p.get("prev_team_xga_for_90", 1.35))
            new_xga = float(p.get("new_team_xga_for_90", 1.35))
            new_xga = max(0.1, new_xga) # Prevent division by zero
            # Lower xG against in new team is better, so prev / new
            xg_scaling = min(1.2, max(0.8, prev_xga / new_xga))
    except (ValueError, TypeError):
        xg_scaling = 1.0

    if p.get("has_stale_pl_history") and p.get("recent_european_peak", False):
        return 0.94

    final_multiplier = base_coef * age_modifier * xg_scaling
    return max(0.65, min(0.98, final_multiplier))

# Gameweek state

def get_gameweek_state(bootstrap_data):
    """Returns (active_gw, target_gw) based on current FPL API state."""
    current_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw_id = next((e["id"] for e in bootstrap_data["events"] if e.get("is_next")), None)
    
    target_gw = next_gw_id or current_gw_id or 1
    active_gw = current_gw_id or (target_gw if target_gw > 1 else 1)
    
    return active_gw, target_gw

# Base EV

def get_base_ev(p, xmins_overrides, weights=None):
    """
    Calculates the base Expected Value (EV) for an FPL player.
    Accepts an optional weights dictionary for dynamic blending.
    """
    if weights is None:
        weights = {}

    pid_str = str(p.get("id"))
    
    if xmins_overrides and pid_str in xmins_overrides:
        xmins = float(xmins_overrides[pid_str])
    else:
        xmins = estimate_xmins(p)
        
    if xmins < 5.0:
        return 0.0
        
    # Safe, isolated parsing
    try: ep = float(p.get("ep_next", 0.0))
    except (ValueError, TypeError): ep = 0.0
    
    try: xgi = float(p.get("xgi_90", 0.0))
    except (ValueError, TypeError): xgi = 0.0
    
    try: xgc = float(p.get("xgc_90", 0.0))
    except (ValueError, TypeError): xgc = 1.35
    if xgc <= 0.0: 
        xgc = 1.35
        
    try: cost = float(p.get("cost", 0.0))
    except (ValueError, TypeError): cost = 4.0
    
    try: own = float(p.get("own", 0.0))
    except (ValueError, TypeError): own = 0.0

    pos_id = p.get("pos_id", 3)
    mins_factor = xmins / 90.0
    
    # 1. Bayesian Shrinkage
    if pos_id == 1: baseline_xgi = 0.01
    elif pos_id == 2: baseline_xgi = 0.08
    elif pos_id == 3: baseline_xgi = 0.25
    elif pos_id == 4: baseline_xgi = 0.35
    else: baseline_xgi = 0.10
    
    cost_threshold = 4.0 if pos_id in [1, 2] else 4.5
    cost_premium = max(0.0, cost - cost_threshold)
    confidence = min(1.0, (own / 15.0) + (cost_premium / 2.0))
    
    # Apply translation factor to xgi before shrinkage
    translation_mult = calculate_tier1_translation_factor(p)
    adjusted_xgi = xgi * translation_mult
    
    shrunken_xgi = (adjusted_xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    # 2. Logistic Regression (Sigmoid) Appearance Points
    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    prob_1_59 = (1.0 - prob_60)
    app_points = (prob_60 * 2.0) + (prob_1_59 * 1.0)

    # 3. Poisson Clean Sheet Engine (60-minute rule)
    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    
    if pos_id in [1, 2]: 
        cs_points = (cs_prob * 4.0) * prob_60
    elif pos_id == 3: 
        cs_points = (cs_prob * 1.0) * prob_60
    else:
        cs_points = 0.0
        
    # 4. Extra Defensive Points (Saves & BPS)
    extra_defensive_points = 0.0
    if pos_id == 1:
        estimated_saves = max(1.5, (xgc * 1.4))
        extra_defensive_points = (estimated_saves / 3.0) * 0.33 * mins_factor
    elif pos_id == 2:
        extra_defensive_points = 0.22 * mins_factor if cost >= 5.5 else 0.08
        
    # 5. Continuous Market-Priced Finisher Curve
    market_premium_factor = 1.0 + (max(0, cost - 5.5) * 0.04) 
    
    if pos_id == 2: pos_mult = 4.2       
    elif pos_id == 3: pos_mult = 4.0     
    else: pos_mult = 3.6                      
        
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium_factor
    
    raw_ev = app_points + attacking_points + cs_points + extra_defensive_points
    
    # Dynamic blending using the optional weights parameter
    xgi_mult = weights.get("xgi_weight", 0.70)
    final_ev = (raw_ev * xgi_mult) + (ep * (1.0 - xgi_mult))
    
    return final_ev
