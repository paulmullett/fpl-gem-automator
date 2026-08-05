"""
fpl_mpo_engine.py — Mathematical Multi-Period Optimization (MILP) Engine
"""

import pulp
import logging
from fpl_funcs import estimate_xmins

logger = logging.getLogger(__name__)

def get_combinatorial_bench_weights(likely_starters_xmins):
    """
    Calculates exact probabilities of needing 1, 2, or 3 outfield subs 
    using a Poisson Binomial distribution matrix.
    """
    # Array of probabilities that each starter MISSES the game entirely
    p_miss = [max(0.0, 1.0 - (xm / 90.0)) for xm in likely_starters_xmins]
    
    # dp[absences] = exact probability of that many simultaneous absences
    dp = {0: 1.0}
    for p in p_miss:
        next_dp = {}
        for absences, prob in dp.items():
            # Player plays (0 absences added)
            next_dp[absences] = next_dp.get(absences, 0.0) + prob * (1.0 - p)
            # Player misses (1 absence added)
            next_dp[absences + 1] = next_dp.get(absences + 1, 0.0) + prob * p
        dp = next_dp
        
    # Cumulative probabilities for FPL bench slots
    # Bench 1 triggers if 1 OR MORE starters miss
    b1_weight = sum(prob for absences, prob in dp.items() if absences >= 1)
    # Bench 2 triggers if 2 OR MORE starters miss
    b2_weight = sum(prob for absences, prob in dp.items() if absences >= 2)
    # Bench 3 triggers if 3 OR MORE starters miss
    b3_weight = sum(prob for absences, prob in dp.items() if absences >= 3)
    
    return b1_weight, b2_weight, b3_weight

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

    # Decision Variables
    x = pulp.LpVariable.dicts("x", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    s = pulp.LpVariable.dicts("s", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    c = pulp.LpVariable.dicts("c", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    
    trans_in = pulp.LpVariable.dicts("trans_in", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")

    # Same-Team Goalkeeper Coupling Variables
    gk_by_team = {}
    for pid in valid_pids:
        if players[pid]["pos_id"] == 1:
            t_id = players[pid].get("team_id")
            if t_id:
                if t_id not in gk_by_team: gk_by_team[t_id] = []
                gk_by_team[t_id].append(pid)
                
    gk_pair = pulp.LpVariable.dicts("gk_pair", 
                ((t_id, t) for t_id in gk_by_team for t in range(horizons)), 
                cat="Binary")

    discount_factor = 0.85
    objective_terms = []
    initial_owned = set(current_squad_ids) if current_squad_ids else set()

    # --- PRIMARY OPTIMIZATION LOOP ---
    for t in range(horizons):
        t_weight = discount_factor ** t
        cap_mult = 2.0 if (t == 0 and active_chip == "TRIPLE_CAPTAIN") else 1.0

        # Store adjusted EVs for this specific gameweek to prevent bench tax-evasion
        adjusted_evs = {}

        for pid in valid_pids:
            p = players[pid]
            base_ev = ev_matrix[pid][t]
                
            # Optimization & Game Theory Gating (Variance & EO)
            floor_ev = p.get("mc_floor_ev", base_ev * 0.8) 
            ceiling_ev = p.get("mc_ceiling_ev", base_ev * 1.2)
            eo = p.get("top_10k_eo", p.get("own", 0.0))
                
            if risk_posture == "SHIELD":
                eo_shield_multiplier = min(1.0, 0.7 + (eo / 100.0) * 0.3)
                base_ev = floor_ev * eo_shield_multiplier
                    
            elif risk_posture == "CHASE":
                eo_chase_tax = max(0.6, 1.0 - (eo / 200.0))
                base_ev = ceiling_ev * eo_chase_tax

            adjusted_evs[pid] = base_ev

            # Co-optimize Starting XI EV + Captain 2x Multiplier
            objective_terms.append(t_weight * base_ev * s[pid, t])
            objective_terms.append(t_weight * base_ev * cap_mult * c[pid, t])

        # Transfer hit penalties (-4 pts per hit)
        if not (t == 0 and (target_gw == 1 or free_transfers == "Unlimited")):
            for pid in valid_pids:
                objective_terms.append(-4.0 * trans_in[pid, t])

        # Pure Game-Rule Constraints
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids) == 15
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids) == 11
        prob += pulp.lpSum(c[pid, t] for pid in valid_pids) == 1

        for pid in valid_pids:
            prob += s[pid, t] <= x[pid, t]
            prob += c[pid, t] <= s[pid, t]

        # 15-Man Squad Constraints (2 GKP, 5 DEF, 5 MID, 3 FWD)
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5
        prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3

        # Starting XI Position Constraints (1 GKP, 3-4 DEF, 3-5 MID, 1-3 FWD)
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 4 # Strictly kills 5-at-the-back
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 3
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1

       # --- NEW: Dynamic Time-Horizon Pricing ---
        # Base cost compounds by the predicted delta multiplied by the gameweek horizon index
        
        # Squad Structure Guardrail: Force £4.0m/£4.5m GK Fodder Meta using projected costs
        prob += pulp.lpSum(
            (players[pid]["cost"] + (players[pid].get("predicted_price_delta", 0.0) * t)) * x[pid, t] 
            for pid in valid_pids if players[pid]["pos_id"] == 1
        ) <= 9.5
        
        # Max 3 players per club rule
        team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
        for team_id in team_ids:
            prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3

        # Total Financial Budget using projected costs
        prob += pulp.lpSum(
            (players[pid]["cost"] + (players[pid].get("predicted_price_delta", 0.0) * t)) * x[pid, t] 
            for pid in valid_pids
        ) <= total_liquid_budget

        # Same-Team Goalkeeper Handshake Constraints
        for t_id, gks in gk_by_team.items():
            if len(gks) >= 2:
                gks_sorted = sorted(gks, key=lambda p: adjusted_evs[p], reverse=True)
                starter_id = gks_sorted[0]
                backup_id = gks_sorted[1]
                
                prob += gk_pair[t_id, t] <= x[starter_id, t]
                prob += gk_pair[t_id, t] <= x[backup_id, t]
                prob += gk_pair[t_id, t] >= x[starter_id, t] + x[backup_id, t] - 1
                
                starter_xmins = players[starter_id].get("xmins", players[starter_id].get("ml_xmins", 90.0))
                missing_xmins_factor = max(0.0, 90.0 - starter_xmins) / 90.0
                
                backup_current_xmins = max(0.01, players[backup_id].get("xmins", players[backup_id].get("ml_xmins", 0.01)))
                backup_full_ev = (adjusted_evs[backup_id] / (backup_current_xmins / 90.0))
                
                handshake_bonus = backup_full_ev * missing_xmins_factor * t_weight
                objective_terms.append(handshake_bonus * gk_pair[t_id, t])

        # Combinatorial Auto-Sub Probability Matrix
        outfield_pids = [pid for pid in valid_pids if players[pid]["pos_id"] != 1]
        gk_pids = [pid for pid in valid_pids if players[pid]["pos_id"] == 1]
        
        top_10_outfield = sorted(outfield_pids, key=lambda p: adjusted_evs[p], reverse=True)[:10]
        likely_xmins = [players[p].get("xmins", 90.0) for p in top_10_outfield]
        
        b1_wt, b2_wt, b3_wt = get_combinatorial_bench_weights(likely_xmins)
        blended_bench_wt = (b1_wt + b2_wt + b3_wt) / 3.0
        
        for p in outfield_pids:
            objective_terms.append(adjusted_evs[p] * (x[p, t] - s[p, t]) * blended_bench_wt * (discount_factor**t))

        # --- UPGRADED: Smarter Sub GK Tie-Breaker ---
        # Coexists with the Handshake Bonus. If a handshake isn't used, 
        # this ensures the best independent £4.0m keeper is selected.
        # --- UPGRADED: Smarter Sub GK Tie-Breaker ---
        for p in gk_pids:
            p_obj = players[p]
            # Check explicit xmins or ml_xmins overrides prior to falling back to heuristics
            p_xmins = p_obj.get("xmins", p_obj.get("ml_xmins", estimate_xmins(p_obj)))
            p_ev = adjusted_evs[p]
        
            # Primary Tie-Breaker: xMins (0.001 weight) | Secondary: EV (0.0001 weight)
            sub_gk_score = (p_xmins * 0.001) + (p_ev * 0.0001)
            objective_terms.append(sub_gk_score * (x[p, t] - s[p, t]) * (discount_factor**t))

        # Squad continuity
        for pid in valid_pids:
            if t == 0:
                is_init = 1 if pid in initial_owned else 0
                prob += x[pid, t] == is_init + trans_in[pid, t] - trans_out[pid, t]
            else:
                prob += x[pid, t] == x[pid, t-1] + trans_in[pid, t] - trans_out[pid, t]

        if not (t == 0 and (target_gw == 1 or free_transfers == "Unlimited")):
            prob += pulp.lpSum(trans_in[pid, t] for pid in valid_pids) <= 3

    # Add all terms to the objective function AFTER the loop is complete
    prob += pulp.lpSum(objective_terms)

    # Attempt HiGHS solver for superior branch-and-bound speed, fallback to CBC
    try:
        # 30-second time limit prevents infinite branching loops
        prob.solve(pulp.HiGHS_CMD(msg=False, timeLimit=30))
    except Exception as e:
        print(f"HiGHS solver not available ({e}), falling back to CBC.")
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