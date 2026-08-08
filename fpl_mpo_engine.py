"""
fpl_mpo_engine.py — Two-Stage Stochastic MILP Engine (PuLP / Native System CBC)
Includes 2024/25 Banked Free Transfer Mechanics (Up to 5 FTs, Chip Retention, Mini-Wildcards)
and Hard Hit Constraints.
"""

import os
import json
import logging
import pulp

logger = logging.getLogger(__name__)

def solve_multi_period_model(players: dict, ev_matrix: dict, current_squad_ids: list, 
                             total_liquid_budget: float, free_transfers: int, 
                             active_chip: str = "NONE", horizons: int = 8, 
                             risk_posture: str = "NEUTRAL", target_gw: int = 1,
                             w_sub_1: float = 0.15, w_sub_2: float = 0.05,
                             planned_chips: dict = None, bank: float = 0.0,
                             available_chips: dict = None,
                             scenarios_path: str = "stochastic_scenarios.json") -> tuple:
    
    if planned_chips is None: planned_chips = {}
    if available_chips is None: available_chips = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}

    valid_pids = [pid for pid in players.keys() if players[pid].get("status") in ["a", "d", ""]]
    if not valid_pids:
        return [], []

    # --- MATRIX COMPRESSION ---
    viable_pids = set(current_squad_ids)
    sorted_by_ev = sorted(valid_pids, key=lambda p: sum(ev_matrix[p]), reverse=True)
    
    gks_added, defs_added, mids_added, fwds_added = 0, 0, 0, 0
    for pid in sorted_by_ev:
        if pid in viable_pids: continue
        pos = players[pid]["pos_id"]
        
        if pos == 1 and gks_added < 12:
            viable_pids.add(pid); gks_added += 1
        elif pos == 2 and defs_added < 35:
            viable_pids.add(pid); defs_added += 1
        elif pos == 3 and mids_added < 35:
            viable_pids.add(pid); mids_added += 1
        elif pos == 4 and fwds_added < 20:
            viable_pids.add(pid); fwds_added += 1

    valid_pids = list(viable_pids)
    # --------------------------

    # --- SAA SCENARIO PRUNING ---
    stochastic_scenarios = None
    if os.path.exists(scenarios_path):
        try:
            with open(scenarios_path, "r") as f:
                stochastic_scenarios = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {scenarios_path}: {e}")

    raw_scenarios = stochastic_scenarios.get("scenarios", {}) if stochastic_scenarios else {}
    
    MAX_SCENARIOS = 40
    if raw_scenarios:
        sorted_scenarios = sorted(raw_scenarios.items(), key=lambda item: item[1]["weight"], reverse=True)[:MAX_SCENARIOS]
        total_weight = sum(item[1]["weight"] for item in sorted_scenarios)
        scenarios = {str(i): {"weight": item[1]["weight"]/total_weight, "player_ev_matrix": item[1]["player_ev_matrix"]} for i, item in enumerate(sorted_scenarios)}
    else:
        scenarios = {}
        
    num_scenarios = len(scenarios) if scenarios else 1
    # ----------------------------

    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    prob = pulp.LpProblem("Stochastic_FPL_Solver", pulp.LpMaximize)

    # --- ENDOGENOUS CHIP DECISION VARIABLES ---
    y_wc = pulp.LpVariable.dicts("y_wc", range(horizons), cat="Binary")
    y_tc = pulp.LpVariable.dicts("y_tc", range(horizons), cat="Binary")
    y_bb = pulp.LpVariable.dicts("y_bb", range(horizons), cat="Binary")

    prob += pulp.lpSum(y_wc[t] for t in range(horizons)) <= (1 if available_chips.get("wildcard", True) else 0)
    prob += pulp.lpSum(y_tc[t] for t in range(horizons)) <= (1 if available_chips.get("3xc", True) else 0)
    prob += pulp.lpSum(y_bb[t] for t in range(horizons)) <= (1 if available_chips.get("bboost", True) else 0)

    for t in range(horizons):
        prob += y_wc[t] + y_tc[t] + y_bb[t] <= 1

    if active_chip == "WILDCARD" and available_chips.get("wildcard", True): prob += y_wc[0] == 1
    elif active_chip == "TRIPLE_CAPTAIN" and available_chips.get("3xc", True): prob += y_tc[0] == 1
    elif active_chip == "BENCH_BOOST" and available_chips.get("bboost", True): prob += y_bb[0] == 1
    else:
        prob += y_wc[0] == 0
        prob += y_tc[0] == 0
        prob += y_bb[0] == 0

    for t in range(horizons):
        if t < 2 and active_chip != "WILDCARD":
            prob += y_wc[t] == 0

    # --- STAGE 1 VARIABLES (GW1) ---
    x = pulp.LpVariable.dicts("x_gw1", valid_pids, cat="Binary")
    s = pulp.LpVariable.dicts("s_gw1", valid_pids, cat="Binary")
    c = pulp.LpVariable.dicts("c_gw1", valid_pids, cat="Binary")
    trans_in = pulp.LpVariable.dicts("tin_gw1", valid_pids, cat="Binary")
    trans_out = pulp.LpVariable.dicts("tout_gw1", valid_pids, cat="Binary")

    # --- STAGE 1 BANKED FT VARIABLES ---
    ft_avail_0 = pulp.LpVariable("ft_avail_0", lowBound=0, upBound=5, cat="Continuous")
    ft_used_0 = pulp.LpVariable("ft_used_0", lowBound=0, upBound=5, cat="Continuous")
    hit_cost_0 = pulp.LpVariable("hit_cost_0", lowBound=0.0, cat="Continuous")

    # --- STAGE 2 VARIABLES (GW2-8 across scenarios) ---
    x_scen = pulp.LpVariable.dicts("x_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tin_scen = pulp.LpVariable.dicts("tin_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tout_scen = pulp.LpVariable.dicts("tout_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    
    s_scen = pulp.LpVariable.dicts("s_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), lowBound=0.0, upBound=1.0, cat="Continuous")
    c_scen = pulp.LpVariable.dicts("c_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), lowBound=0.0, upBound=1.0, cat="Continuous")
    
    # --- STAGE 2 BANKED FT VARIABLES ---
    ft_avail_scen = pulp.LpVariable.dicts("ft_avail_scen", ((t, k) for t in range(1, horizons) for k in range(num_scenarios)), lowBound=1, upBound=5, cat="Continuous")
    ft_used_scen = pulp.LpVariable.dicts("ft_used_scen", ((t, k) for t in range(1, horizons) for k in range(num_scenarios)), lowBound=0, upBound=5, cat="Continuous")
    hit_cost_scen = pulp.LpVariable.dicts("hit_scen", ((t, k) for t in range(1, horizons) for k in range(num_scenarios)), lowBound=0.0, cat="Continuous")

    # Opportunity Cost Mathematics
    if target_gw <= 19:
        remaining_half = max(1.0, 19.0 - target_gw)
        decay = (remaining_half / 18.0) ** 0.5
        wc_cost = 14.0 * decay; tc_cost = 9.0 * decay; bb_cost = 10.0 * decay
    else:
        remaining_half = max(1.0, 38.0 - target_gw)
        decay = (remaining_half / 18.0) ** 0.5
        wc_cost = 16.0 * decay; tc_cost = 14.0 * decay; bb_cost = 18.0 * decay

    # Stage 1 Constraints
    prob += pulp.lpSum(x[pid] for pid in valid_pids) == 15
    prob += pulp.lpSum(s[pid] for pid in valid_pids) == 11
    prob += pulp.lpSum(c[pid] for pid in valid_pids) == 1

    for pid in valid_pids:
        prob += s[pid] <= x[pid]
        prob += c[pid] <= s[pid]
        prob += trans_in[pid] + trans_out[pid] <= 1

    prob += pulp.lpSum(x[pid] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2
    prob += pulp.lpSum(x[pid] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5
    prob += pulp.lpSum(x[pid] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5
    prob += pulp.lpSum(x[pid] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3

    prob += pulp.lpSum(s[pid] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1
    prob += pulp.lpSum(s[pid] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3
    prob += pulp.lpSum(s[pid] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 5
    prob += pulp.lpSum(s[pid] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 2
    prob += pulp.lpSum(s[pid] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1

    team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
    for team_id in team_ids:
        prob += pulp.lpSum(x[pid] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3

    if is_fresh_squad or str(free_transfers).lower() == "unlimited":
        prob += pulp.lpSum(players[pid]["cost"] * x[pid] for pid in valid_pids) <= total_liquid_budget
        prob += pulp.lpSum(trans_in[pid] for pid in valid_pids) == 0
        prob += pulp.lpSum(trans_out[pid] for pid in valid_pids) == 0
        
        prob += ft_avail_0 == 0
        prob += ft_used_0 == 0
        prob += hit_cost_0 == 0
    else:
        for pid in valid_pids:
            is_init = 1 if pid in initial_owned else 0
            prob += x[pid] == is_init + trans_in[pid] - trans_out[pid]
        cash_in = pulp.lpSum(players[pid].get("selling_price", players[pid]["cost"]) * trans_out[pid] for pid in valid_pids)
        cash_out = pulp.lpSum(players[pid]["cost"] * trans_in[pid] for pid in valid_pids)
        prob += bank + cash_in - cash_out >= 0.0
        
        prob += ft_avail_0 == min(5, max(1, int(free_transfers)))
        trans_sum_0 = pulp.lpSum(trans_in[pid] for pid in valid_pids)
        prob += ft_used_0 <= ft_avail_0
        prob += ft_used_0 <= trans_sum_0
        prob += ft_used_0 <= 5 * (1 - y_wc[0])
        prob += hit_cost_0 >= 4.0 * (trans_sum_0 - ft_used_0) - (100.0 * y_wc[0])

    # Stage 2 Constraints
    for k in range(num_scenarios):
        for t in range(1, horizons):
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids) == 15
            prob += pulp.lpSum(s_scen[pid, t, k] for pid in valid_pids) == 11
            prob += pulp.lpSum(c_scen[pid, t, k] for pid in valid_pids) == 1
            prob += pulp.lpSum(players[pid]["cost"] * x_scen[pid, t, k] for pid in valid_pids) <= total_liquid_budget

            # --- 2024/25 BANKED FREE TRANSFER MECHANICS ---
            trans_sum_scen = pulp.lpSum(tin_scen[pid, t, k] for pid in valid_pids)
            
            # Carryover calculation: Retained through Wildcards without accumulating +1
            if t == 1:
                prob += ft_avail_scen[t, k] <= (ft_avail_0 - ft_used_0) + 1 - y_wc[0]
            else:
                prob += ft_avail_scen[t, k] <= (ft_avail_scen[t-1, k] - ft_used_scen[t-1, k]) + 1 - y_wc[t-1]
                
            prob += ft_used_scen[t, k] <= ft_avail_scen[t, k]
            prob += ft_used_scen[t, k] <= trans_sum_scen
            prob += ft_used_scen[t, k] <= 5 * (1 - y_wc[t])
            
            # HARD CAP: Maximum of 1 extra transfer for a -4 hit per week unless Wildcard is active
            prob += trans_sum_scen <= ft_avail_scen[t, k] + 1 + (15 * y_wc[t])
            
            # Hit cost safely evaluates to 0 during Mini-Wildcards (where FTs >= transfers)
            prob += hit_cost_scen[t, k] >= 4.0 * (trans_sum_scen - ft_used_scen[t, k]) - (100.0 * y_wc[t])
            # ----------------------------------------------

            for pid in valid_pids:
                prob += s_scen[pid, t, k] <= x_scen[pid, t, k]
                prob += c_scen[pid, t, k] <= s_scen[pid, t, k]
                prob += tin_scen[pid, t, k] + tout_scen[pid, t, k] <= 1

                if t == 1:
                    prob += x_scen[pid, 1, k] == x[pid] + tin_scen[pid, 1, k] - tout_scen[pid, 1, k]
                else:
                    prob += x_scen[pid, t, k] == x_scen[pid, t-1, k] + tin_scen[pid, t, k] - tout_scen[pid, t, k]

            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3

            for team_id in team_ids:
                prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3

    # Objective Function
    objective_terms = []
    discount_factor = 0.85

    for t in range(horizons):
        objective_terms.append(-1.0 * wc_cost * y_wc[t])
        objective_terms.append(-1.0 * tc_cost * y_tc[t])
        objective_terms.append(-1.0 * bb_cost * y_bb[t])

    # Stage 1 Objective
    objective_terms.append(-1.0 * hit_cost_0)
    objective_terms.append(0.01 * (ft_avail_0 - ft_used_0))

    for pid in valid_pids:
        base_ev = ev_matrix[pid][0]
        objective_terms.append(base_ev * s[pid])
        objective_terms.append(base_ev * c[pid])
        
        # FIX: Bench players contribute a fractional EV based on auto-sub probability
        objective_terms.append((base_ev * w_sub_1) * (x[pid] - s[pid]))

    # Stage 2 Objective Across Scenarios
    for k in range(num_scenarios):
        scen_key = str(k)
        scen_weight = scenarios[scen_key]["weight"] if scenarios and scen_key in scenarios else (1.0 / max(1, num_scenarios))
        scen_ev_matrix = scenarios.get(scen_key, {}).get("player_ev_matrix", {})
        objective_terms.append((ev_val * t_discount) * s_scen[pid, t, k])
        objective_terms.append((ev_val * t_discount) * c_scen[pid, t, k])
                
        # FIX: Apply auto-sub probability to the bench across future scenarios
        objective_terms.append((ev_val * t_discount * w_sub_1) * (x_scen[pid, t, k] - s_scen[pid, t, k]))


        for t in range(1, horizons):
            t_discount = (discount_factor ** t) * scen_weight
            for pid in valid_pids:
                pid_str = str(pid)
                if pid_str in scen_ev_matrix and len(scen_ev_matrix[pid_str]) > t:
                    ev_val = float(scen_ev_matrix[pid_str][t])
                else:
                    ev_val = float(ev_matrix[pid][t]) if t < len(ev_matrix[pid]) else 0.0

                objective_terms.append((ev_val * t_discount) * s_scen[pid, t, k])
                objective_terms.append((ev_val * t_discount) * c_scen[pid, t, k])

            objective_terms.append(-1.0 * t_discount * hit_cost_scen[t, k])
            objective_terms.append(0.01 * t_discount * (ft_avail_scen[t, k] - ft_used_scen[t, k]))

    prob += pulp.lpSum(objective_terms)

    cbc_path = "/usr/bin/cbc"
    try:
        if os.path.exists(cbc_path):
            solver_cmd = pulp.COIN_CMD(path=cbc_path, msg=False, timeLimit=120)
        else:
            solver_cmd = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
        prob.solve(solver_cmd)
    except Exception as e:
        logger.error(f"Solver error: {e}")

    optimal_squad = []
    transfer_plan = []

    if prob.status == pulp.LpStatusOptimal:
        logger.info("Stochastic Solution Found!")

        for pid in valid_pids:
            if x[pid].varValue and x[pid].varValue > 0.5:
                p_copy = dict(players[pid])
                p_copy["is_starter"] = bool(s[pid].varValue and s[pid].varValue > 0.5)
                p_copy["is_captain"] = bool(c[pid].varValue and c[pid].varValue > 0.5)
                optimal_squad.append(p_copy)

        for t in range(1, min(4, horizons)):
            gw_trans_in = [players[pid]["name"] for pid in valid_pids if tin_scen[pid, t, 0].varValue and tin_scen[pid, t, 0].varValue > 0.5]
            gw_trans_out = [players[pid]["name"] for pid in valid_pids if tout_scen[pid, t, 0].varValue and tout_scen[pid, t, 0].varValue > 0.5]
            
            chip_str = ""
            if y_wc[t].varValue and y_wc[t].varValue > 0.5: chip_str = " [WILDCARD DEPLOYED]"
            elif y_tc[t].varValue and y_tc[t].varValue > 0.5: chip_str = " [TRIPLE CAPTAIN DEPLOYED]"
            elif y_bb[t].varValue and y_bb[t].varValue > 0.5: chip_str = " [BENCH BOOST DEPLOYED]"

            if gw_trans_in or gw_trans_out or chip_str:
                transfer_plan.append(f"GW{target_gw + t}: In [{', '.join(gw_trans_in)}], Out [{', '.join(gw_trans_out)}]{chip_str}")

        return optimal_squad, transfer_plan

    else:
        logger.warning("Stochastic Solver failed to mathematically converge in time. Executing Constrained Greedy Heuristic.")
        valid_players = [p for p in players.values() if p.get("status") in ["a", "d", ""]]
        
        gks = sorted([p for p in valid_players if p["pos_id"] == 1], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        defs = sorted([p for p in valid_players if p["pos_id"] == 2], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        mids = sorted([p for p in valid_players if p["pos_id"] == 3], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        fwds = sorted([p for p in valid_players if p["pos_id"] == 4], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)

        base_squad = gks[:2] + defs[:5] + mids[:5] + fwds[:3]
        return base_squad, []