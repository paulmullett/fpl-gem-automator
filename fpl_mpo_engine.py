"""
fpl_mpo_engine.py — Two-Stage Stochastic MILP Engine (PuLP / Native System CBC)
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
    
    valid_pids = [pid for pid in players.keys() if players[pid].get("status") in ["a", "d", ""]]
    if not valid_pids:
        return [], []

    # --- MATRIX COMPRESSION (Player Pool Trimming) ---
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
    # -------------------------------------------------

    # Load SAA Stochastic Scenarios
    stochastic_scenarios = None
    if os.path.exists(scenarios_path):
        try:
            with open(scenarios_path, "r") as f:
                stochastic_scenarios = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {scenarios_path}: {e}")

    # Set the scenario limit parameter here (e.g., 40, 30, 20)
    MAX_SCENARIOS_TO_SOLVE = 30  

    raw_scenarios = stochastic_scenarios.get("scenarios", {}) if stochastic_scenarios else {}
    
    if raw_scenarios:
        # Isolate top N scenarios and re-normalize weights to equal 1.0
        sorted_scenarios = sorted(raw_scenarios.items(), key=lambda item: item[1]["weight"], reverse=True)[:MAX_SCENARIOS_TO_SOLVE]
        total_weight = sum(item[1]["weight"] for item in sorted_scenarios)
        scenarios = {str(i): {"weight": item[1]["weight"]/total_weight, "player_ev_matrix": item[1]["player_ev_matrix"]} for i, item in enumerate(sorted_scenarios)}
    else:
        scenarios = {}
        
    num_scenarios = len(scenarios) if scenarios else 1

    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    prob = pulp.LpProblem("Stochastic_FPL_Solver", pulp.LpMaximize)

    # Stage 1: Here-and-Now
    x = pulp.LpVariable.dicts("x_gw1", valid_pids, cat="Binary")
    s = pulp.LpVariable.dicts("s_gw1", valid_pids, cat="Binary")
    c = pulp.LpVariable.dicts("c_gw1", valid_pids, cat="Binary")
    trans_in = pulp.LpVariable.dicts("tin_gw1", valid_pids, cat="Binary")
    trans_out = pulp.LpVariable.dicts("tout_gw1", valid_pids, cat="Binary")

    # Stage 2: Wait-and-See
    x_scen = pulp.LpVariable.dicts("x_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    s_scen = pulp.LpVariable.dicts("s_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    c_scen = pulp.LpVariable.dicts("c_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tin_scen = pulp.LpVariable.dicts("tin_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tout_scen = pulp.LpVariable.dicts("tout_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")

    # Hit cost accounting variables for future weeks
    hit_cost_scen = pulp.LpVariable.dicts("hit_scen", ((t, k) for t in range(1, horizons) for k in range(num_scenarios)), lowBound=0.0, cat="Continuous")

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

    if is_fresh_squad:
        prob += pulp.lpSum(players[pid]["cost"] * x[pid] for pid in valid_pids) <= total_liquid_budget
        prob += pulp.lpSum(trans_in[pid] for pid in valid_pids) == 0
        prob += pulp.lpSum(trans_out[pid] for pid in valid_pids) == 0
    else:
        for pid in valid_pids:
            is_init = 1 if pid in initial_owned else 0
            prob += x[pid] == is_init + trans_in[pid] - trans_out[pid]
        cash_in = pulp.lpSum(players[pid].get("selling_price", players[pid]["cost"]) * trans_out[pid] for pid in valid_pids)
        cash_out = pulp.lpSum(players[pid]["cost"] * trans_in[pid] for pid in valid_pids)
        prob += bank + cash_in - cash_out >= 0.0

    # Stage 2 Constraints
    for k in range(num_scenarios):
        for t in range(1, horizons):
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids) == 15
            prob += pulp.lpSum(s_scen[pid, t, k] for pid in valid_pids) == 11
            prob += pulp.lpSum(c_scen[pid, t, k] for pid in valid_pids) == 1

            # STAGE 2 BUDGET GUARDRAIL
            prob += pulp.lpSum(players[pid]["cost"] * x_scen[pid, t, k] for pid in valid_pids) <= total_liquid_budget

            # STAGE 2 HIT COST ACCOUNTING: Transfers > 1 incur a -4 pt penalty
            trans_sum_scen = pulp.lpSum(tin_scen[pid, t, k] for pid in valid_pids)
            prob += hit_cost_scen[t, k] >= 4.0 * (trans_sum_scen - 1)

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

    for pid in valid_pids:
        base_ev = ev_matrix[pid][0]
        objective_terms.append(base_ev * s[pid])
        objective_terms.append(base_ev * c[pid])

    for k in range(num_scenarios):
        scen_key = str(k)
        scen_weight = scenarios[scen_key]["weight"] if scenarios and scen_key in scenarios else (1.0 / max(1, num_scenarios))
        scen_ev_matrix = scenarios.get(scen_key, {}).get("player_ev_matrix", {})

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

            # Deduct hit cost penalty from objective
            objective_terms.append(-1.0 * t_discount * hit_cost_scen[t, k])

    prob += pulp.lpSum(objective_terms)

    # Use native system CBC binary via COIN_CMD
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

    def has_feasible_squad():
        try:
            return sum(1 for pid in valid_pids if x[pid].varValue is not None and x[pid].varValue > 0.5) == 15
        except Exception:
            return False

    if prob.status == pulp.LpStatusOptimal or has_feasible_squad():
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
            if gw_trans_in or gw_trans_out:
                transfer_plan.append(f"GW{target_gw + t}: In [{', '.join(gw_trans_in)}], Out [{', '.join(gw_trans_out)}]")

        return optimal_squad, transfer_plan

    else:
        logger.warning("Stochastic Solver failed to converge. Executing Constrained Greedy Heuristic.")
        valid_players = [p for p in players.values() if p.get("status") in ["a", "d", ""]]
        
        gks = sorted([p for p in valid_players if p["pos_id"] == 1], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        defs = sorted([p for p in valid_players if p["pos_id"] == 2], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        mids = sorted([p for p in valid_players if p["pos_id"] == 3], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)
        fwds = sorted([p for p in valid_players if p["pos_id"] == 4], key=lambda i: sum(ev_matrix[i["id"]]), reverse=True)

        base_squad = gks[:2] + defs[:5] + mids[:5] + fwds[:3]
        return base_squad, transfer_plan