import os
import sys
import json
import requests
import pulp
import math

# 1. Environment & Pre-Flight Check
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
STATE_FILE_PATH = "fpl_state.json"

if not FPL_TEAM_ID:
    print("CRITICAL ERROR: Missing FPL_TEAM_ID environment variable.")
    sys.exit(1)

# 2. State Management (Independent Instance)
def load_state():
    default_state = {
        "calibration_weights": {
            "xgi_weight": 0.70,
            "fdr_impact_factor": 0.10,
            "bench_discount": 0.01
        },
        "xmins_overrides": {}
    }
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                saved = json.load(f)
                for k, v in default_state.items():
                    if k not in saved:
                        saved[k] = v
                return saved
        except Exception:
            pass
    return default_state

# 3. Core Mathematical Engines (Stream A & Stream B)
def estimate_xmins(p):
    chance = str(p.get("chance_of_playing_next_round", ""))
    if chance == "0" or p.get("status") not in ["a", "d"]:
        return 0.0
    try:
        own = float(p.get("own", 0.0))
        cost = float(p.get("cost", 0.0))
    except:
        own, cost = 0.0, 4.0

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

def get_stream_a_ev(p, weights, xmins_overrides):
    pid_str = str(p["id"])
    xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
    if xmins < 5.0:
        return 0.0
        
    try:
        ep = float(p.get("ep_next", 0.0))
        xgi = float(p.get("xgi_90", 0.0))
        xgc = float(p.get("xgc_90", 0.0) or 1.35)
        cost = float(p.get("cost", 0.0))
        own = float(p.get("own", 0.0))
    except:
        ep, xgi, xgc, cost, own = 0.0, 0.0, 1.35, 4.0, 0.0

    pos_id = p["pos_id"]
    mins_factor = xmins / 90.0
    
    # Bayesian Shrinkage
    baseline_xgi = 0.01 if pos_id == 1 else (0.08 if pos_id == 2 else (0.25 if pos_id == 3 else 0.35))
    cost_threshold = 4.0 if pos_id in [1, 2] else 4.5
    confidence = min(1.0, (own / 15.0) + (max(0.0, cost - cost_threshold) / 2.0))
    shrunken_xgi = (xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    # Sigmoid Appearance & Poisson Clean Sheet
    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)
    
    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    cs_points = (cs_prob * (4.0 if pos_id in [1, 2] else (1.0 if pos_id == 3 else 0.0))) * prob_60
    
    market_premium = 1.0 + (max(0, cost - 5.5) * 0.04)
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium
    
    raw_ev = app_points + attacking_points + cs_points
    xgi_mult = weights.get("xgi_weight", 0.70)
    return (raw_ev * xgi_mult) + (ep * (1.0 - xgi_mult))

def get_stream_b_ev(p):
    """Stream B: Momentum & Short-Term Form Regressor"""
    try:
        form = float(p.get("form", 0.0))
        ep = float(p.get("ep_next", 0.0))
        total_pts = float(p.get("total_points", 0))
    except:
        form, ep, total_pts = 0.0, 0.0, 0.0
        
    # Momentum scaling factor emphasizing recent form and FPL expected points
    momentum_score = (form * 0.6) + (ep * 0.4)
    return max(0.0, momentum_score)

def get_ensemble_ev(p, weights, xmins_overrides):
    ev_a = get_stream_a_ev(p, weights, xmins_overrides)
    ev_b = get_stream_b_ev(p)
    
    # Blending Function: 70% Structural Realism (Stream A) + 30% Momentum (Stream B)
    blended_ev = (0.70 * ev_a) + (0.30 * ev_b)
    return blended_ev

def get_macro_ev(p, team_avg_fdr, weights, xmins_overrides):
    base_ev = get_ensemble_ev(p, weights, xmins_overrides)
    if base_ev <= 0.0:
        return 0.0
    xmins = estimate_xmins(p)
    variance_penalty = 0.8 + (min(xmins, 90.0) / 90.0) * 0.2
    ev_4gw = (base_ev * variance_penalty) * 4.0
    avg_fdr = team_avg_fdr.get(p["team_id"], 3.0)
    fdr_multiplier = 1.0 + ((3.0 - avg_fdr) * 0.10)
    return ev_4gw * fdr_multiplier

# 4. Ensemble Solver Execution Pipeline
def run_ensemble_optimization():
    headers = {"User-Agent": "FPL-Ensemble-Script/1.0"}
    state = load_state()
    weights = state["calibration_weights"]
    xmins_overrides = state.get("xmins_overrides", {})

    resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    if resp.status_code != 200:
        print("CRITICAL ERROR: Failed to reach FPL API.")
        sys.exit(1)
    data = resp.json()

    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in data["element_types"]}

    players = {}
    for p in data["elements"]:
        players[p["id"]] = {
            "id": p["id"], "name": p["web_name"], "team": teams.get(p["team"], "UNK"),
            "team_id": p["team"], "pos": element_types.get(p["element_type"], "UNK"),
            "pos_id": p["element_type"], "cost": p["now_cost"] / 10.0,
            "status": p["status"], "news": p["news"], "ep_next": str(p.get("ep_next", "0.0")),
            "form": str(p.get("form", "0.0")), "total_points": p.get("total_points", 0),
            "own": str(p.get("selected_by_percent", "0.0")),
            "chance_of_playing_next_round": str(p.get("chance_of_playing_next_round", "")),
            "xgi_90": str(p.get("expected_goal_involvements_per_90", "0.0")),
            "xgc_90": str(p.get("expected_goals_conceded_per_90", "0.0"))
        }

    # Fixtures & FDR
    fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
    fixtures_data = fixtures_resp.json() if fixtures_resp.status_code == 200 else []
    
    current_gw = next((e["id"] for e in data["events"] if e.get("is_current")), 1)
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and current_gw <= event < current_gw + 4:
            if f.get("team_a") in team_fdr_sum:
                team_fdr_sum[f["team_a"]] += f.get("team_a_difficulty", 3)
                team_fdr_count[f["team_a"]] += 1
            if f.get("team_h") in team_fdr_sum:
                team_fdr_sum[f["team_h"]] += f.get("team_h_difficulty", 3)
                team_fdr_count[f["team_h"]] += 1
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    # MILP Knapsack Solver
    prob = pulp.LpProblem("FPL_Ensemble_Optimization", pulp.LpMaximize)
    valid_ids = list(players.keys())
    squad_vars = pulp.LpVariable.dicts("squad", valid_ids, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", valid_ids, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", valid_ids, cat="Binary")

    objective = []
    for i in valid_ids:
        p = players[i]
        ev = get_macro_ev(p, team_avg_fdr, weights, xmins_overrides)
        own_pct = float(p.get("own", 0.0)) / 100.0
        rank_gravity = (ev * (own_pct ** 2) * 0.75)
        
        objective.append(
            (ev * starter_vars[i]) + 
            ((ev + rank_gravity) * captain_vars[i]) + 
            (0.01 * ev * (squad_vars[i] - starter_vars[i]))
        )

    prob += pulp.lpSum(objective)

    for i in valid_ids:
        p = players[i]
        prob += starter_vars[i] <= squad_vars[i]
        prob += captain_vars[i] <= starter_vars[i]
        if p["status"] not in ["a", "d"] or p.get("chance_of_playing_next_round") == "0":
            prob += squad_vars[i] == 0

    prob += pulp.lpSum([squad_vars[i] for i in valid_ids]) == 15
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids]) == 11
    prob += pulp.lpSum([captain_vars[i] for i in valid_ids]) == 1
    
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players[i]["pos_id"] == 1]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players[i]["pos_id"] == 2]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players[i]["pos_id"] == 3]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players[i]["pos_id"] == 4]) == 3
    
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players[i]["pos_id"] == 1]) == 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players[i]["pos_id"] == 2]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players[i]["pos_id"] == 3]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players[i]["pos_id"] == 4]) >= 1

    for t_id in set(p["team_id"] for p in players.values()):
        prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players[i]["team_id"] == t_id]) <= 3

    prob += pulp.lpSum([players[i]["cost"] * squad_vars[i] for i in valid_ids]) <= 100.0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print("--- ENSEMBLE SQUAD RUN COMPLETE ---")
    selected_starters = [players[i] for i in valid_ids if starter_vars[i].varValue and starter_vars[i].varValue > 0.5]
    captain = next((players[i] for i in valid_ids if captain_vars[i].varValue and captain_vars[i].varValue > 0.5), None)
    
    print(f"Captain: {captain['name'] if captain else 'None'}")
    print("Starting XI (Ensemble Blended Model):")
    for s in selected_starters:
        print(f" - {s['name']} ({s['team']}, {s['pos']}, £{s['cost']}m) | Blended EV: {get_ensemble_ev(s, weights, xmins_overrides):.2f}")

if __name__ == "__main__":
    run_ensemble_optimization()
