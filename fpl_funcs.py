#Combined re-used functions and operations
import math

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
