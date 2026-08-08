"""
fpl_mpo_engine.py — Two-Stage Stochastic Mixed-Integer Linear Programming Engine

Powered by Google OR-Tools (SCIP/CBC backend).
Stage 1 (Here-and-Now): Resolves binding current-gameweek squad & transfer decisions.
Stage 2 (Wait-and-See): Evaluates 50 clustered SAA multiverses across GW2-8.
"""

import os
import json
import math
import logging
from ortools.linear_solver import pywraplp
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
                             available_chips: dict = None,
                             scenarios_path: str = "stochastic_scenarios.json") -> tuple:
    
    if planned_chips is None: planned_chips = {}
    if available_chips is None: available_chips = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}
    
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
            logger.warning(f"Failed to load {scenarios_path}: {e}. Falling back to single-scenario mode.")

    scenarios = stochastic_scenarios.get("scenarios", {}) if stochastic_scenarios else {}
    scenario_weights = stochastic_scenarios.get("scenario_weights", [1.0]) if stochastic_scenarios else [1.0]
    num_scenarios = len(scenarios) if scenarios else 1

    # 2. Instantiate Google OR-Tools Solver (SCIP backend with CBC fallback)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        logger.error("Could not initialize SCIP or CBC solver in OR-Tools.")
        return [], []

    initial_owned = set(current_squad_ids) if (current_squad_ids and len(current_squad_ids) == 15) else set()
    is_fresh_squad = len(initial_owned) == 0

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # Stage 1: Here-and-Now (Universal GW1 decisions)
    x = {}         # Squad inclusion (15-man)
    s = {}         # Starting XI inclusion
    c = {}         # Captaincy
    trans_in = {}  # Transfer In
    trans_out = {} # Transfer Out

    for pid in valid_pids:
        x[pid, 0] = solver.BoolVar(f"x_{pid}_0")
        s[pid, 0] = solver.BoolVar(f"s_{pid}_0")
        c[pid, 0] = solver.BoolVar(f"c_{pid}_0")
        trans_in[pid, 0] = solver.BoolVar(f"tin_{pid}_0")
        trans_out[pid, 0] = solver.BoolVar(f"tout_{pid}_0")

    # Stage 2: Wait-and-See (Scenario-indexed GW2-8 decisions)
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

    # =========================================================================
    # CONSTRAINTS: STAGE 1 (GW1)
    # =========================================================================
    
    # 15-man squad size & positional allocation
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

    # Formation constraints
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 1) == 1)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 2) >= 3)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 2) <= 5)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 3) >= 2)
    solver.Add(sum(s[pid, 0] for pid in valid_pids if players[pid]["pos_id"] == 4) >= 1)

    # Team limits (max 3 per club)
    team_ids = set(players[pid]["team_id"] for pid in valid_pids if players[pid].get("team_id"))
    for team_id in team_ids:
        solver.Add(sum(x[pid, 0] for pid in valid_pids if players[pid].get("team_id") == team_id) <= 3)

    # Budget & Squad continuity
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

    # =========================================================================
    # CONSTRAINTS: STAGE 2 (GW2-8 Across Scenarios)
    # =========================================================================
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

    # =========================================================================
    # OBJECTIVE FUNCTION: Maximizing Expected Yield Across Multiverse
    # =========================================================================
    objective = solver.Objective()
    discount_factor = 0.85

    # Stage 1 Objective Terms
    for pid in valid_pids:
        base_ev = ev_matrix[pid][0]
        objective.SetCoefficient(s[pid, 0], base_ev)
        objective.SetCoefficient(c[pid, 0], base_ev) # Captain 2x multiplier

    # Stage 2 Objective Terms (Weighted by Scenario Probability)
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

    # =========================================================================
    # SOLVER EXECUTION
    # =========================================================================
    logger.info("Executing Two-Stage Stochastic MILP via OR-Tools SCIP...")
    solver.set_time_limit(120000) # 2-minute hard timeout
    status = solver.Solve()

    optimal_squad = []
    transfer_plan = []

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        logger.info(f"Stochastic Solution Found! Multiverse Horizon xP: {objective.Value():.2f}")

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
        
        gks = sorted([p for p in valid_players if p["pos_id"] == 1], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        defs = sorted([p for p in valid_players if p["pos_id"] == 2], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        mids = sorted([p for p in valid_players if p["pos_id"] == 3], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)
        fwds = sorted([p for p in valid_players if p["pos_id"] == 4], key=lambda x: sum(ev_matrix[x["id"]]), reverse=True)

        base_squad = gks[:2] + defs[:5] + mids[:5] + fwds[:3]
        return base_squad, transfer_plan