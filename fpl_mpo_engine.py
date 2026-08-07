"""
fpl_mpo_engine.py — Mathematical Multi-Period Optimization (MILP) Engine
"""

import pulp
import logging
from fpl_funcs import estimate_xmins

logger = logging.getLogger(__name__)

def get_combinatorial_bench_weights(likely_starters_xmins):
    p_miss = [max(0.0, 1.0 - (xm / 90.0)) for xm in likely_starters_xmins]
    
    dp = {0: 1.0}
    for p in p_miss:
        next_dp = {}
        for absences, prob in dp.items():
            next_dp[absences] = next_dp.get(absences, 0.0) + prob * (1.0 - p)
            next_dp[absences + 1] = next_dp.get(absences + 1, 0.0) + prob * p
        dp = next_dp
    
    b1_weight = sum(prob for absences, prob in dp.items() if absences >= 1)
    b2_weight = sum(prob for absences, prob in dp.items() if absences >= 2)
    b3_weight = sum(prob for absences, prob in dp.items() if absences >= 3)
    
    return b1_weight, b2_weight, b3_weight

def solve_multi_period_model(players: dict, ev_matrix: dict, current_squad_ids: list, 
                             total_liquid_budget: float, free_transfers: int, 
                             active_chip: str = "NONE", horizons: int = 8, 
                             risk_posture: str = "NEUTRAL", target_gw: int = 1,
                             w_sub_1: float = 0.15, w_sub_2: float = 0.05,
                             planned_chips: dict = None, bank: float = 0.0,
                             available_chips: dict = None) -> tuple:
    if planned_chips is None: planned_chips = {}
    if available_chips is None: available_chips = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}
    
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

    # Core Decision Variables
    x = pulp.LpVariable.dicts("x", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    s = pulp.LpVariable.dicts("s", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    c = pulp.LpVariable.dicts("c", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    trans_in = pulp.LpVariable.dicts("trans_in", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((pid, t) for pid in valid_pids for t in range(horizons)), cat="Binary")

    # Endogenous Chip Decision Variables
    y_wc = pulp.LpVariable.dicts("y_wc", range(horizons), cat="Binary")
    y_tc = pulp.LpVariable.dicts("y_tc", range(horizons), cat="Binary")
    y_bb = pulp.LpVariable.dicts("y_bb", range(horizons), cat="Binary")

    # Dynamic Cash Flow Ledger Variable
    bank_balance = pulp.LpVariable.dicts("bank_balance", range(horizons), lowBound=0.0, cat="Continuous")

    # Global Chip Rules: Max 1 of each per horizon, Max 1 chip total per gameweek
    prob += pulp.lpSum(y_wc[t] for t in range(horizons)) <= (1 if available_chips.get("wildcard", True) else 0)
    prob += pulp.lpSum(y_tc[t] for t in range(horizons)) <= (1 if available_chips.get("3xc", True) else 0)
    prob += pulp.lpSum(y_bb[t] for t in range(horizons)) <= (1 if available_chips.get("bboost", True) else 0)
    
    for t in range(horizons):
        prob += y_wc[t] + y_tc[t] + y_bb[t] <= 1
        if t == 0:
            if active_chip == "WILDCARD" and available_chips.get("wildcard", True): prob += y_wc[0] == 1
            elif active_chip == "TRIPLE_CAPTAIN" and available_chips.get("3xc", True): prob += y_tc[0] == 1
            elif active_chip == "BENCH_BOOST" and available_chips.get("bboost", True): prob += y_bb[0] == 1
            else:
                prob += y_wc[0] == 0
                prob += y_tc[0] == 0
                prob += y_bb[0] == 0

    # Prevent automated Wildcard burning in early weeks (t=0, t=1) unless explicitly selected
    for t in range(horizons):
        if t < 2 and active_chip != "WILDCARD":
            prob += y_wc[t] == 0

    # The Gameweek 19 "Use It or Lose It" Constraint
    if target_gw <= 19:
        gw19_idx = 19 - target_gw
        if 0 <= gw19_idx < horizons:
            active_vars = []
            if available_chips.get("wildcard", True): active_vars.append(y_wc)
            if available_chips.get("3xc", True): active_vars.append(y_tc)
            if available_chips.get("bboost", True): active_vars.append(y_bb)
            
            available_slots = gw19_idx + 1
            if available_slots >= len(active_vars):
                for y_var in active_vars:
                    prob += pulp.lpSum(y_var[t] for t in range(available_slots)) == 1

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
    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    # Define the Opportunity Cost of burning the first Wildcard
    WILDCARD_OPPORTUNITY_COST = 14.0 

    # --- PRIMARY OPTIMIZATION LOOP ---
    for t in range(horizons):
        t_weight = discount_factor ** t
        
        # Penalize Wildcard usage to simulate long-term holding value
        objective_terms.append(-1.0 * WILDCARD_OPPORTUNITY_COST * y_wc[t])

        adjusted_evs = {}
        for pid in valid_pids:
            p = players[pid]
            base_ev = ev_matrix[pid][t]
            
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

            # Base Starting XI EV & Captain 2x Multiplier
            objective_terms.append(t_weight * base_ev * s[pid, t])
            objective_terms.append(t_weight * base_ev * c[pid, t])

        # Triple Captain Multiplier Linearization (Un-discounted to prevent early burn)
        tc_cap = pulp.LpVariable.dicts(f"tc_cap_{t}", valid_pids, lowBound=0.0, upBound=1.0, cat="Continuous")
        for pid in valid_pids:
            prob += tc_cap[pid] <= c[pid, t]
            prob += tc_cap[pid] <= y_tc[t]
            prob += tc_cap[pid] >= c[pid, t] + y_tc[t] - 1
            if adjusted_evs[pid] > 0:
                objective_terms.append(adjusted_evs[pid] * tc_cap[pid])

        # Transfer Hit Penalty & Wildcard Hit Suppression
        if not (t == 0 and (target_gw == 1 or str(free_transfers).lower() in ["unlimited", "999"])):
            hit_cost = pulp.LpVariable(f"hit_cost_{t}", lowBound=0.0, cat="Continuous")
            trans_sum = pulp.lpSum(trans_in[pid, t] for pid in valid_pids)
            
            free_tf_val = free_transfers if (isinstance(free_transfers, int) and free_transfers > 0) else 1
            prob += hit_cost >= 4.0 * (trans_sum - free_tf_val) - (100.0 * y_wc[t])
            objective_terms.append(-hit_cost)

        # Bench Boost Objective Linearization (Un-discounted to prevent early burn)
        bb_active = pulp.LpVariable.dicts(f"bb_active_{t}", valid_pids, lowBound=0.0, upBound=1.0, cat="Continuous")
        for pid in valid_pids:
            prob += bb_active[pid] <= x[pid, t] - s[pid, t]
            prob += bb_active[pid] <= y_bb[t]
            prob += bb_active[pid] >= (x[pid, t] - s[pid, t]) + y_bb[t] - 1
            if adjusted_evs[pid] > 0:
                objective_terms.append(adjusted_evs[pid] * bb_active[pid])

        # Core Game-Rule Constraints
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

        # Starting XI Position Constraints
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 5
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 2
        prob += pulp.lpSum(s[pid, t] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1

        # Goalkeeper Budget Guardrail (Max £9.5m spend on GKs)
        prob += pulp.lpSum(
            (players[pid]["cost"] + (players[pid].get("price_delta_prob", 0.0) * t)) * x[pid, t] 
            for pid in valid_pids if players[pid]["pos_id"] == 1
        ) <= 9.5
        
        # Max 3 players per club rule
        team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
        for team_id in team_ids:
            prob += pulp.lpSum(x[pid, t] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3

        # Dynamic Cash Flow & Exact Selling Value Constraints
        if t == 0:
            if is_fresh_squad:
                squad_cost_0 = pulp.lpSum(players[pid]["cost"] * x[pid, 0] for pid in valid_pids)
                prob += squad_cost_0 <= total_liquid_budget
                prob += bank_balance[0] == total_liquid_budget - squad_cost_0
                prob += pulp.lpSum(trans_in[pid, 0] for pid in valid_pids) == 0
                prob += pulp.lpSum(trans_out[pid, 0] for pid in valid_pids) == 0
            else:
                cash_in = pulp.lpSum(players[pid].get("selling_price", players[pid]["cost"]) * trans_out[pid, 0] for pid in valid_pids)
                cash_out = pulp.lpSum(players[pid]["cost"] * trans_in[pid, 0] for pid in valid_pids)
                prob += bank_balance[0] == bank + cash_in - cash_out
        else:
            future_cash_in_terms = []
            future_cash_out_terms = []
            
            for pid in valid_pids:
                p_cost = players[pid]["cost"]
                p_delta = players[pid].get("price_delta_prob", 0.0) * t
                future_market_price = p_cost + p_delta
                
                # Exact FPL Non-Linear Sell-on Profit Logic
                if p_delta > 0.0:
                    exact_profit = math.floor(round(p_delta * 10) / 2) * 0.1
                    exact_sell_price = p_cost + exact_profit
                else:
                    exact_sell_price = future_market_price
                    
                future_cash_in_terms.append(exact_sell_price * trans_out[pid, t])
                future_cash_out_terms.append(future_market_price * trans_in[pid, t])

            prob += bank_balance[t] == bank_balance[t-1] + pulp.lpSum(future_cash_in_terms) - pulp.lpSum(future_cash_out_terms)

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

        # Smarter Sub GK Tie-Breaker
        for p in gk_pids:
            p_obj = players[p]
            p_xmins = p_obj.get("xmins", p_obj.get("ml_xmins", estimate_xmins(p_obj)))
            p_ev = adjusted_evs[p]
            sub_gk_score = (p_xmins * 0.001) + (p_ev * 0.0001)
            objective_terms.append(sub_gk_score * (x[p, t] - s[p, t]) * (discount_factor**t))

        # Squad Continuity & Transfer Mutual Exclusivity Constraints
        for pid in valid_pids:
            # Prevent buying and selling the same player in the same gameweek
            prob += trans_in[pid, t] + trans_out[pid, t] <= 1
            
            if t == 0:
                if not is_fresh_squad:
                    is_init = 1 if pid in initial_owned else 0
                    prob += x[pid, 0] == is_init + trans_in[pid, 0] - trans_out[pid, 0]
            else:
                prob += x[pid, t] == x[pid, t-1] + trans_in[pid, t] - trans_out[pid, t]

        if not (t == 0 and (target_gw == 1 or str(free_transfers).lower() in ["unlimited", "999"])):
            prob += pulp.lpSum(trans_in[pid, t] for pid in valid_pids) <= 3

        # --- THE SOFT PENALTY: Wildcard Anti-Churn Logic ---
        if t < horizons - 1:
            wc_churn = pulp.LpVariable.dicts(f"wc_churn_{t}", valid_pids, lowBound=0.0, cat="Continuous")
            for pid in valid_pids:
                # If Wildcard is played (y_wc=1) AND player bought (trans_in=1) AND player sold next week (trans_out=1), wc_churn >= 1.0
                prob += wc_churn[pid] >= y_wc[t] + trans_in[pid, t] + trans_out[pid, t+1] - 2.0
                # Apply the -4.0 Expected Value penalty to the solver's objective function
                objective_terms.append(-4.0 * wc_churn[pid])

    # Add all terms to objective function
    prob += pulp.lpSum(objective_terms)

    # Solve using HiGHS with CBC fallback
    try:
        prob.solve(pulp.HiGHS_CMD(msg=False, timeLimit=90))
    except Exception as e:
        logger.warning(f"HiGHS solver not available ({e}), falling back to CBC.")
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=90))

    optimal_squad = []
    transfer_plan = []

    def has_feasible_squad():
        try:
            return sum(1 for pid in valid_pids if x[pid, 0].varValue is not None and x[pid, 0].varValue > 0.5) == 15
        except Exception:
            return False

    if prob.status == pulp.LpStatusOptimal or has_feasible_squad():
        for pid in valid_pids:
            if x[pid, 0].varValue and x[pid, 0].varValue > 0.5:
                p_copy = dict(players[pid])
                p_copy["is_starter"] = bool(s[pid, 0].varValue and s[pid, 0].varValue > 0.5)
                p_copy["is_captain"] = bool(c[pid, 0].varValue and c[pid, 0].varValue > 0.5)
                optimal_squad.append(p_copy)
        
        for t in range(1, min(4, horizons)):
            gw_trans_in = [players[pid]["name"] for pid in valid_pids if trans_in[pid, t].varValue and trans_in[pid, t].varValue > 0.5]
            gw_trans_out = [players[pid]["name"] for pid in valid_pids if trans_out[pid, t].varValue and trans_out[pid, t].varValue > 0.5]
            chip_str = ""
            if y_wc[t].varValue and y_wc[t].varValue > 0.5: chip_str = " [WILDCARD DEPLOYED]"
            elif y_tc[t].varValue and y_tc[t].varValue > 0.5: chip_str = " [TRIPLE CAPTAIN DEPLOYED]"
            elif y_bb[t].varValue and y_bb[t].varValue > 0.5: chip_str = " [BENCH BOOST DEPLOYED]"

            if gw_trans_in or gw_trans_out or chip_str:
                transfer_plan.append(f"GW{target_gw + t}: In [{', '.join(gw_trans_in)}], Out [{', '.join(gw_trans_out)}]{chip_str}")
        
        if peak_swing_team:
            team_name_str = next((p["team"] for p in players.values() if p.get("team_id") == peak_swing_team), str(peak_swing_team))
            transfer_plan.append(f"AUTOMATED SWING ALERT: {team_name_str} exhibits primary 4-GW fixture green wave.")
    else:
        logger.warning("MPO Solver failed to find optimal path, falling back to Constrained Greedy Heuristic.")
        valid_players = [p for p in players.values() if p.get("status") in ["a", "d", ""]]
        
        # Sort all players by true 8-GW Expected Value
        gks = sorted([p for p in valid_players if p["pos_id"] == 1], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        defs = sorted([p for p in valid_players if p["pos_id"] == 2], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        mids = sorted([p for p in valid_players if p["pos_id"] == 3], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        fwds = sorted([p for p in valid_players if p["pos_id"] == 4], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        
        optimal_squad = []
        current_budget = total_liquid_budget
        
        # 1. Allocate minimum viable base (2 GK, 5 DEF, 5 MID, 3 FWD) at cheapest prices
        base_gks = sorted(gks, key=lambda x: x["cost"])[:2]
        base_defs = sorted(defs, key=lambda x: x["cost"])[:5]
        base_mids = sorted(mids, key=lambda x: x["cost"])[:5]
        base_fwds = sorted(fwds, key=lambda x: x["cost"])[:3]
        
        base_squad = base_gks + base_defs + base_mids + base_fwds
        base_cost = sum(p["cost"] for p in base_squad)
        remaining_budget = current_budget - base_cost
        
        if remaining_budget < 0:
            logger.error("Total market floor exceeds budget. Returning illegal raw heuristic.")
            return base_squad, transfer_plan
            
        optimal_squad = base_squad.copy()
        
        # 2. Greedy Knapsack Upgrade Loop
        # Repeatedly upgrade the weakest player in the squad to the highest EV player we can afford
        upgrade_made = True
        while upgrade_made:
            upgrade_made = False
            # Find the player in our squad with the lowest 8-GW EV
            weakest_player = min(optimal_squad, key=lambda x: sum(ev_matrix[x["id"]]))
            
            # Find all available players in the same position
            pos_pool = [p for p in valid_players if p["pos_id"] == weakest_player["pos_id"] and p["id"] not in [s["id"] for s in optimal_squad]]
            # Filter for players we can afford with our remaining budget + the cost of the weakest player
            affordable_upgrades = [p for p in pos_pool if p["cost"] <= (remaining_budget + weakest_player["cost"])]
            
            if affordable_upgrades:
                # Pick the affordable player with the highest 8-GW EV
                best_upgrade = max(affordable_upgrades, key=lambda x: sum(ev_matrix[x["id"]]))
                
                # If the upgrade is actually better than our weakest player, execute the swap
                if sum(ev_matrix[best_upgrade["id"]]) > sum(ev_matrix[weakest_player["id"]]):
                    remaining_budget += weakest_player["cost"]
                    remaining_budget -= best_upgrade["cost"]
                    optimal_squad.remove(weakest_player)
                    optimal_squad.append(best_upgrade)
                    upgrade_made = True

    return optimal_squad, transfer_plan