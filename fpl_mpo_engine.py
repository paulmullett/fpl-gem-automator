"""
fpl_mpo_engine.py — Two-Stage Stochastic MILP Engine (OR-Tools SCIP)
Optimized for x86_64 Cloud Execution with Light Matrix Compression.
"""

import os
import json
import logging
from ortools.linear_solver import pywraplp
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

    # --- LIGHT MATRIX COMPRESSION ---
    # Retains the top 200 players to maintain maximal search quality
    # while ensuring the SCIP solver clears the multiverse in < 120 seconds.
    viable_pids = set(current_squad_ids)
    sorted_by_ev = sorted(valid_pids, key=lambda p: sum(ev_matrix[p]), reverse=True)
    
    gks_added, defs_added, mids_added, fwds_added = 0, 0, 0, 0
    for pid in sorted_by_ev:
        if pid in viable_pids: continue
        pos = players[pid]["pos_id"]
        
        if pos == 1 and gks_added < 20:
            viable_pids.add(pid); gks_added += 1
        elif pos == 2 and defs_added < 65:
            viable_pids.add(pid); defs_added += 1
        elif pos == 3 and mids_added < 75:
            viable_pids.add(pid); mids_added += 1
        elif pos == 4 and fwds_added < 40:
            viable_pids.add(pid); fwds_added += 1

    valid_pids = list(viable_pids)
    # --------------------------------

    # Load SAA Stochastic Scenarios
    stochastic_scenarios = None
    if os.path.exists(scenarios_path):
        try:
            with open(scenarios_path, "r") as f:
                stochastic_scenarios = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {scenarios_path}: {e}")

    scenarios = stochastic_scenarios.get("scenarios", {}) if stochastic_scenarios else {}
    scenario_weights = stochastic_scenarios.get("scenario_weights", [1.0]) if stochastic_scenarios else [1.0]
    num_scenarios = len(scenarios) if scenarios else 1

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        logger.error("Could not initialize SCIP solver.")
        return [], []

    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    x = {}
    s = {}
    c = {}
    trans_in = {}
    trans_out = {}

    for pid in valid_pids:
        x[pid, 0] = solver.BoolVar(f"x_{pid}_0")
        s[pid, 0] = solver.BoolVar(f"s_{pid}_0")
        c[pid, 0] = solver.BoolVar(f"c_{pid}_0")
        trans_in[pid, 0] = solver.BoolVar(f"tin_{pid}_0")
        trans_out[pid, 0] = solver.BoolVar(f"tout_{pid}_0")

    x_scen = {}
    s_scen = {}
    c_scen = {}
    trans_in_scen = {}
    trans_out_scen = {}

    for k in range(num_scenarios):
        for t in range(1, horizons):
            for pid in valid_pids:
                x_scen[pid, t, k] = solver.BoolVar(f"x_{pid}_{t}_k{k}")
                s_scen[pid, t, k] = solver.BoolVar(f"s_{pid}_{t}_k{k}")
                c_scen[pid, t, k] = solver.BoolVar(f"c_{pid}_{t}_k{k}")
                trans_in_scen[pid, t, k] = solver.BoolVar(f"tin_{pid}_{t}_k{k}")
                trans_out_scen[pid, t, k] = solver.BoolVar(f"tout_{pid}_{t}_k{k}")

    # Stage 1 Constraints
    solver.Add(sum(x[pid, 0] for pid in valid_pids) == 15)
    solver.Add(sum(s[pid, 0] for pid in valid_pids) == 11)
    solver.Add(sum(c[pid, 0] for pid in valid_pids) == 1)

    for pid in valid_pids:
        solver.Add(s[pid, 0] <= x[pid, 0])
        solver.Add(c[pid, 0] <= s[pid, 0])
        solver.Add(trans_in[pid, 0] + trans_out[pid, 0] <= 1)

    solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2)
    solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5)
    solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5)
    solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3)

    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 5)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 2)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1)

    team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
    for team_id in team_ids:
        solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3)

    if is_fresh_squad:
        solver.Add(sum(players[pid]["cost"] * x[pid, 0] for pid in valid_pids) <= total_liquid_budget)
        solver.Add(sum(trans_in[pid, 0] for pid in valid_pids) == 0)
        solver.Add(sum(trans_out[pid, 0] for pid in valid_pids) == 0)
    else:
        for pid in valid_pids:
            is_init = 1 if pid in initial_owned else 0
            solver.Add(x[pid, 0] == is_init + trans_in[pid, 0] - trans_out[pid, 0])
        cash_in = sum(players[pid].get("selling_price", players[pid]["cost"]) * trans_out[pid, 0] for pid in valid_pids)
        cash_out = sum(players[pid]["cost"] * trans_in[pid, 0] for pid in valid_pids)
        solver.Add(bank + cash_in - cash_out >= 0.0)

    # Stage 2 Constraints
    for k in range(num_scenarios):
        for t in range(1, horizons):
            solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids) == 15)
            solver.Add(sum(s_scen[pid, t, k] for pid in valid_pids) == 11)
            solver.Add(sum(c_scen[pid, t, k] for pid in valid_pids) == 1)

            for pid in valid_pids:
                solver.Add(s_scen[pid, t, k] <= x_scen[pid, t, k])
                solver.Add(c_scen[pid, t, k] <= s_scen[pid, t, k])
                solver.Add(trans_in_scen[pid, t, k] + trans_out_scen[pid, t, k] <= 1)

                if t == 1:
                    solver.Add(x_scen[pid, 1, k] == x[pid, 0] + trans_in_scen[pid, 1, k] - trans_out_scen[pid, 1, k])
                else:
                    solver.Add(x_scen[pid, t, k] == x_scen[pid, t-1, k] + trans_in_scen[pid, t, k] - trans_out_scen[pid, t, k])

            solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 1) == 2)
            solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 2) == 5)
            solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 3) == 5)
            solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids if players[pid]["pos_id"] == 4) == 3)

            for team_id in team_ids:
                solver.Add(sum(x_scen[pid, t, k] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3)

    objective = solver.Objective()
    discount_factor = 0.85

    for pid in valid_pids:
        base_ev = ev_matrix[pid][0]
        objective.SetCoefficient(s[pid, 0], base_ev)
        objective.SetCoefficient(c[pid, 0], base_ev)

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

                objective.SetCoefficient(s_scen[pid, t, k], ev_val * t_discount)
                objective.SetCoefficient(c_scen[pid, t, k], ev_val * t_discount)

    objective.SetMaximization()

    logger.info("Executing Two-Stage Stochastic MILP via OR-Tools SCIP...")
    solver.set_time_limit(180000)
    status = solver.Solve()

    optimal_squad = []
    transfer_plan = []

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        for pid in valid_pids:
            if x[pid, 0].solution_value() > 0.5:
                p_copy = dict(players[pid])
                p_copy["is_starter"] = bool(s[pid, 0].solution_value() > 0.5)
                p_copy["is_captain"] = bool(c[pid, 0].solution_value() > 0.5)
                optimal_squad.append(p_copy)

        for t in range(1, min(4, horizons)):
            gw_trans_in = [players[pid]["name"] for pid in valid_pids if trans_in_scen[pid, t, 0].solution_value() > 0.5]
            gw_trans_out = [players[pid]["name"] for pid in valid_pids if trans_out_scen[pid, t, 0].solution_value() > 0.5]
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