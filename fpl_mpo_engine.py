"""
fpl_mpo_engine.py — Multi-Period Optimization (MPO) Engine with Native Bench Weighting
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

    # Automated Fixture Swing Detection
    team_swing_scores = {}
    for pid in valid_pids:
        p = players[pid]
        t_id = p.get("team_id")
        if not t_id: continue
        
        gw_evs = ev_matrix.get(pid, [0.0] * horizons)
        early_ev = sum(gw_evs[0:4])
        late_ev = sum(gw_evs[4:8]) if horizons >= 8 else early_ev
        
        swing_delta = late_ev - early_ev
        if t_id not in team_swing_scores: team_swing_scores[t_id] = []
        team_swing_scores[t_id].append(swing_delta)
        
    avg_team_swings = {t: sum(swings)/len(swings) for t, swings in team_swing_scores.items() if swings}
    peak_swing_team = max(avg_team_swings, key=avg_team_swings.get) if avg_team_swings else None

    prob = pulp.LpProblem("FPL_Multi_Period_Optimization", pulp.LpMaximize)

    # Decision Variables:
    # x[p, t] = 1 if player p is owned in 15-man squad in GW t
    # s[p, t] = 1 if player p is in Starting XI in GW t
    # c[p, t] = 1 if player p is Captain in GW t
    x = pulp.LpVariable.dicts("x", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    s = pulp.LpVariable.dicts("s", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    c = pulp.LpVariable.dicts("c", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    
    # Transfer Variables
    trans_in = pulp.LpVariable.dicts("trans_in", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")

    discount_factor = 0.85
    objective_terms = []
    w_bench = 0.05  # Bench EV weighted at 5% to force budget into starting XI premiums

    for t in range(horizons):
        t_weight = discount_factor ** t
        cap_mult = 2.0 if (t == 0 and active_chip == "TRIPLE_CAPTAIN") else 1.0

        for pid in valid_pids:
            p = players[pid]
            base_ev = ev_matrix[pid][t]
            
            # Risk posture adjustments
            if risk_posture == "SHIELD":
                base_ev *= (1.0 - (p.get("own", 0.0) / 200.0 * 0.1))
            elif risk_posture == "CHASE":
                base_ev *= (1.0 + (p.get("own", 0.0) / 200.0 * 0.1))
            
            # OBJECTIVE FUNCTION: Starting XI EV + Captain EV + Bench EV (5% weight)
            objective_terms.append(t_weight * base_ev * s[pid, t])
            objective_terms.append(t_weight * base_ev * cap_mult * c[pid, t])
            objective_terms.append(t_weight * base_ev * w_bench * (x[pid, t] - s[pid, t]))

        # Transfer hit penalties (-4 pts per hit)
        if not (t == 0 and (target_gw == 1 or free_transfers == "Unlimited")):
            for pid in valid_pids:
                objective_terms.append(-4.0 * trans_in[pid, t])

    prob += pulp.lpSum(objective_terms)

    # Constraints per Gameweek
    initial_owned = set(current_squad_ids) if current_squad_ids else set()

    for t in range(horizons):
        # Squad and Lineup Size
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids) == 15
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids) == 11
        prob += pulp.lpSum(c[pid, t] for pid in valid_pids) == 1

        # Linking constraints: s <= x and c <= s
        for pid in valid_pids:
            prob += s[pid, t] <= x[pid, t]
            prob += c[pid, t] <= s[pid, t]

        # 15-Man Squad Position Constraints (2 GKP, 5 DEF, 5 MID, 3 FWD)
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3

        # Starting XI Position Constraints (1 GKP, 3-4 DEF, 3-5 MID, 1-3 FWD)
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 4
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 3
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1

        # Max 3 players per real-world club
        team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
        for team_id in team_ids:
            prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3

        # Total Liquid Budget
        prob += pulp.lpSum(players[pid]["cost"] * x[pid, t] for pid in valid_pids) <= total_liquid_budget

        # Squad continuity and transfer balance
        for pid in valid_pids:
            if t == 0:
                is_init = 1 if pid in initial_owned else 0
                prob += x[pid, t] == is_init + trans_in[pid, t] - trans_out[pid, t]
            else:
                prob += x[pid, t] == x[pid, t-1] + trans_in[pid, t] - trans_out[pid, t]

        # Transfer limits per GW
        if not (t == 0 and (target_gw == 1 or free_transfers == "Unlimited")):
            prob += pulp.lpSum(trans_in[pid, t] for pid in valid_pids) <= 3

    # Solve optimization model
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    optimal_squad = []
    transfer_plan = []

    if prob.status == pulp.LpStatusOptimal:
        for pid in valid_pids:
            if x[pid, 0].varValue and x[pid, 0].varValue > 0.5:
                p_copy = dict(players[pid])
                p_copy["is_starter"] = bool(s[pid, 0].varValue and s[pid, 0].varValue > 0.5)
                p_copy["is_captain"] = bool(c[pid, 0].varValue and c[pid, 0].varValue > 0.5)
                optimal_squad.append(p_copy)
        
        for t in range(1, min(4, horizons)):
            gw_trans_in = [players[pid]["name"] for pid in valid_pids if trans_in[pid, t].varValue and trans_in[pid, t].varValue > 0.5]
            gw_trans_out = [players[pid]["name"] for pid in valid_pids if trans_out[pid, t].varValue and trans_out[pid, t].varValue > 0.5]
            if gw_trans_in or gw_trans_out:
                transfer_plan.append(f"GW{target_gw + t}: In [{', '.join(gw_trans_in)}], Out [{', '.join(gw_trans_out)}]")
        
        if peak_swing_team:
            team_name_str = next((p["team"] for p in players.values() if p.get("team_id") == peak_swing_team), str(peak_swing_team))
            transfer_plan.append(f"AUTOMATED SWING ALERT: {team_name_str} exhibits primary 4-GW fixture green wave.")
    else:
        logger.warning("MPO Solver failed to find optimal path, falling back to 1-GW greedy selection.")
        sorted_fallback = sorted([p for p in players.values() if p.get("status") in ["a", "d", ""]], 
                                 key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        optimal_squad = sorted_fallback[:15]

    return optimal_squad, transfer_plan