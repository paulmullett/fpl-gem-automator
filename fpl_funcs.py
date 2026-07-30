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
