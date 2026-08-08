"""
fpl_mpo_engine.py — Two-Stage Stochastic Mixed-Integer Linear Programming Engine (PuLP/CBC)

Stage 1 (Here-and-Now): Resolves binding current-gameweek squad & transfer decisions.
Stage 2 (Wait-and-See): Evaluates 50 clustered SAA multiverses across GW2-8.
"""

import os
import json
import logging
import pulp
from fpl_funcs import estimate_xmins

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

    # 1. Load SAA Stochastic Scenarios
    stochastic_scenarios = None
    if os.path.exists(scenarios_path):
        try:
            with open(scenarios_path, "r") as f:
                stochastic_scenarios = json.load(f)
            logger.info(f"Loaded stochastic scenarios from '{scenarios_path}'.")
        except Exception as e:
            logger.warning(f"Failed to load {scenarios_path}: {e}")

    scenarios = stochastic_scenarios.get("scenarios", {}) if stochastic_scenarios else {}
    scenario_weights = stochastic_scenarios.get("scenario_weights", [1.0]) if stochastic_scenarios else [1.0]
    num_scenarios = len(scenarios) if scenarios else 1

    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    prob = pulp.LpProblem("Stochastic_FPL_Solver", pulp.LpMaximize)

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # Stage 1: Here-and-Now (Universal GW1 decisions)
    x = pulp.LpVariable.dicts("x_gw1", valid_pids, cat="Binary")
    s = pulp.LpVariable.dicts("s_gw1", valid_pids, cat="Binary")
    c = pulp.LpVariable.dicts("c_gw1", valid_pids, cat="Binary")
    trans_in = pulp.LpVariable.dicts("tin_gw1", valid_pids, cat="Binary")
    trans_out = pulp.LpVariable.dicts("tout_gw1", valid_pids, cat="Binary")

    # Stage 2: Wait-and-See (Scenario-indexed GW2-8 decisions)
    x_scen = pulp.LpVariable.dicts("x_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    s_scen = pulp.LpVariable.dicts("s_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    c_scen = pulp.LpVariable.dicts("c_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tin_scen = pulp.LpVariable.dicts("tin_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")
    tout_scen = pulp.LpVariable.dicts("tout_scen", ((pid, t, k) for pid in valid_pids for t in range(1, horizons) for k in range(num_scenarios)), cat="Binary")

    # =========================================================================
    # CONSTRAINTS: STAGE 1 (GW1)
    # =========================================================================
    
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

    # =========================================================================
    # CONSTRAINTS: STAGE 2 (GW2-8 Across Scenarios)
    # =========================================================================
    for k in range(num_scenarios):
        for t in range(1, horizons):
            prob += pulp.lpSum(x_scen[pid, t, k] for pid in valid_pids) == 15
            prob += pulp.lpSum(s_scen[pid, t, k] for pid in valid_pids) == 11
            prob += pulp.lpSum(c_scen[pid, t, k] for pid in valid_pids) == 1

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

    # =========================================================================
    # OBJECTIVE FUNCTION: Maximizing Expected Yield Across Multiverse
    # =========================================================================
    objective_terms = []
    discount_factor = 0.85

    for pid in valid_pids:
        base_ev = ev_matrix[pid][0]
        objective_terms.append(base_ev * s[pid])
        objective_terms.append(base_ev * c[pid])

    for k in range(num_scenarios):
        scen_weight = scenario_weights[k] if k < len(scenario_weights) else (1.0 / num_scenarios)
        scen_key = str(k)
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

    prob += pulp.lpSum(objective_terms)

    # =========================================================================
    # SOLVER EXECUTION
    # =========================================================================
    logger.info("Executing Two-Stage Stochastic MILP via PuLP/CBC...")
    
    cbc_path = "/usr/bin/cbc"
    try:
        if os.path.exists(cbc_path):
            prob.solve(pulp.COIN_CMD(path=cbc_path, msg=False, timeLimit=120))
        else:
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
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