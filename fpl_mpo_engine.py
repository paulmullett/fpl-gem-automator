"""
fpl_mpo_engine.py — Multi-Period Optimization (MPO) Engine with Fixture Swing & Chip Detection
"""

import pulp
import logging

logger = logging.getLogger(__name__)

def solve_multi_period_model(players: dict, ev_matrix: dict, current_squad_ids: list, 
                             total_liquid_budget: float, free_transfers: int, 
                             active_chip: str = "NONE", horizons: int = 8, 
                             risk_posture: str = "NEUTRAL", target_gw: int = 1,
                             w_sub_1: float = 0.15, w_sub_2: float = 0.05,
                             planned_chips: dict = None) -> tuple:
    if planned_chips is None: planned_chips = {}
    
    valid_pids = [pid for pid in players.keys() if players[pid].get("status") in ["a", "d", ""]]
    if not valid_pids:
        return [], []

    # =========================================================================
    # AUTOMATED FIXTURE SWING DETECTION (Rolling 4-GW Delta)
    # =========================================================================
    team_swing_scores = {}
    for pid in valid_pids:
        p = players[pid]
        t_id = p.get("team_id")
        if not t_id: continue
        
        # Calculate rolling 4-GW EV vs subsequent 4-GW EV
        gw_evs = ev_matrix.get(pid, [0.0] * horizons)
        early_ev = sum(gw_evs[0:4])
        late_ev = sum(gw_evs[4:8]) if horizons >= 8 else early_ev
        
        swing_delta = late_ev - early_ev
        if t_id not in team_swing_scores: team_swing_scores[t_id] = []
        team_swing_scores[t_id].append(swing_delta)
        
    # Determine league-wide fixture swing hotspots for chip timing
    avg_team_swings = {t: sum(swings)/len(swings) for t, swings in team_swing_scores.items() if swings}
    peak_swing_team = max(avg_team_swings, key=avg_team_swings.get) if avg_team_swings else None
    # =========================================================================

    prob = pulp.LpProblem("FPL_Multi_Period_Optimization", pulp.LpMaximize)

    # Decision Variables: x[p, t] = 1 if player p is owned in gameweek t
    x = pulp.LpVariable.dicts("x", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    
    # Transfer Variables
    trans_in = pulp.LpVariable.dicts("trans_in", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")

    discount_factor = 0.85
    objective_terms = []

    for t in range(horizons):
        t_weight = discount_factor ** t
        for pid in valid_pids:
            p = players[pid]
            base_ev = ev_matrix[pid][t]
            
            # Risk posture adjustments
            if risk_posture == "SHIELD":
                base_ev *= (1.0 - (p.get("own", 0.0) / 200.0 * 0.1))
            elif risk_posture == "CHASE":
                base_ev *= (1.0 + (p.get("own", 0.0) / 200.0 * 0.1))
                
            objective_terms.append(base_ev * t_weight * x[pid, t])

    # Transfer cost penalties (-4 pts per hit)
    hit_penalty = 4.0
    for t in range(horizons):
        if t == 0 and (target_gw == 1 or free_transfers == "Unlimited"):
            continue # Free unlimited transfers for GW1
        for pid in valid_pids:
            objective_terms.append(-hit_penalty * trans_in[pid, t])

    prob += pulp.lpSum(objective_terms)

    # Constraints per Gameweek
    initial_owned = set(current_squad_ids) if current_squad_ids else set()

    for t in range(horizons):
        # Exactly 15 players owned per gameweek
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids) == 15
        
        # Position constraints (2 GKP, 5 DEF, 5 MID, 3 FWD)
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5
        prob += pulp.lpSum(x[pid, t] for pid5 in valid_pids if players[pid]["pos_id"] == 3) == 5 if False else pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3

        # Budget constraint
        prob += pulp.lpSum(players[pid]["cost"] * x[pid, t] for pid in valid_pids) <= total_liquid_budget

        # Squad budget distribution constraints (prevents £23.5m+ bench hoarding)
        # Max 11 players costing >= £6.0m (forces 4+ cheap budget enablers across bench/squad)
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["cost"] >= 6.0) <= 11

        # Max 13 players costing >= £5.0m (forces at least 2 floor-priced £4.0m/£4.5m bench fodder)
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["cost"] >= 5.0) <= 13

        # Squad continuity and transfer balance equations
        for pid in valid_pids:
            if t == 0:
                is_init = 1 if pid in initial_owned else 0
                prob += x[pid, t] == is_init + trans_in[pid, t] - trans_out[pid, t]
            else:
                prob += x[pid, t] == x[pid, t-1] + trans_in[pid, t] - trans_out[pid, t]

        # Transfer count limits per gameweek (max 2 free transfers + rollovers, capped at 5)
        if t == 0 and (target_gw == 1 or free_transfers == "Unlimited"):
            pass
        else:
            prob += pulp.lpSum(trans_in[pid, t] for pid in valid_pids) <= 3

    # Solve optimization model
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    optimal_squad = []
    transfer_plan = []

    if prob.status == pulp.LpStatusOptimal:
        # Extract GW0 optimal squad
        for pid in valid_pids:
            if x[pid, 0].varValue and x[pid, 0].varValue > 0.5:
                optimal_squad.append(players[pid])
                
        # Extract transfer recommendations for future horizons
        for t in range(1, min(4, horizons)):
            gw_trans_in = [players[pid]["name"] for pid in valid_pids if trans_in[pid, t].varValue and trans_in[pid, t].varValue > 0.5]
            gw_trans_out = [players[pid]["name"] for pid in valid_pids if trans_out[pid, t].varValue and trans_out[pid, t].varValue > 0.5]
            if gw_trans_in or gw_trans_out:
                transfer_plan.append(f"GW{target_gw + t}: In [{', '.join(gw_trans_in)}], Out [{', '.join(gw_trans_out)}]")
        
        if peak_swing_team:
            transfer_plan.append(f"AUTOMATED SWING ALERT: Team ID {peak_swing_team} exhibits primary 4-GW fixture green wave.")
    else:
        logger.warning("MPO Solver failed to find optimal path, falling back to 1-GW greedy selection.")
        sorted_fallback = sorted([p for p in players.values() if p.get("status") in ["a", "d", ""]], 
                                 key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        optimal_squad = sorted_fallback[:15]

    return optimal_squad, transfer_plan