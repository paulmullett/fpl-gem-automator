import os
import sys
import requests
import pulp

# Import the production engine directly to ensure identical baseline math
from fpl_bot import get_base_ev, estimate_xmins, load_state

FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
if not FPL_TEAM_ID:
    print("CRITICAL ERROR: Missing FPL_TEAM_ID environment variable.")
    sys.exit(1)

def get_ensemble_ev(p, xmins_overrides, weights):
    ev_a = get_base_ev(p, xmins_overrides)
    try:
        form = float(p.get("form", 0.0))
        ep = float(p.get("ep_next", 0.0))
    except:
        form, ep = 0.0, 0.0
    ev_b = max(0.0, (form * 0.6) + (ep * 0.4))
    return (0.70 * ev_a) + (0.30 * ev_b)

def solve_model(players_dict, use_ensemble=False):
    state = load_state()
    weights = state["calibration_weights"]
    xmins_overrides = state.get("xmins_overrides", {})

    prob = pulp.LpProblem(f"FPL_{'Ensemble' if use_ensemble else 'Baseline'}", pulp.LpMaximize)
    valid_ids = list(players_dict.keys())
    
    squad_vars = pulp.LpVariable.dicts("squad", valid_ids, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", valid_ids, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", valid_ids, cat="Binary")

    objective = []
    for i in valid_ids:
        p = players_dict[i]
        ev = get_ensemble_ev(p, xmins_overrides, weights) if use_ensemble else get_base_ev(p, xmins_overrides)
        own_pct = float(p.get("own", 0.0)) / 100.0
        rank_gravity = (ev * (own_pct ** 2) * 0.75)
        
        objective.append(
            (ev * starter_vars[i]) + 
            ((ev + rank_gravity) * captain_vars[i]) + 
            (0.01 * ev * (squad_vars[i] - starter_vars[i]))
        )

    prob += pulp.lpSum(objective)

    for i in valid_ids:
        p = players_dict[i]
        prob += starter_vars[i] <= squad_vars[i]
        prob += captain_vars[i] <= starter_vars[i]
        if p["status"] not in ["a", "d"] or p.get("chance_of_playing_next_round") == "0":
            prob += squad_vars[i] == 0

    prob += pulp.lpSum([squad_vars[i] for i in valid_ids]) == 15
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids]) == 11
    prob += pulp.lpSum([captain_vars[i] for i in valid_ids]) == 1
    
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
    
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) >= 1

    for t_id in set(p["team_id"] for p in players_dict.values()):
        prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

    prob += pulp.lpSum([players_dict[i]["cost"] * squad_vars[i] for i in valid_ids]) <= 100.0
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    starters = [players_dict[i] for i in valid_ids if starter_vars[i].varValue and starter_vars[i].varValue > 0.5]
    captain = next((players_dict[i] for i in valid_ids if captain_vars[i].varValue and captain_vars[i].varValue > 0.5), None)
    
    ev_func = lambda p, x: get_ensemble_ev(p, x, weights) if use_ensemble else get_base_ev(p, x)
    total_xp = sum(ev_func(s, xmins_overrides) for s in starters)
    if captain:
        total_xp += ev_func(captain, xmins_overrides)

    return starters, captain, total_xp

def run_comparison():
    headers = {"User-Agent": "FPL-Compare-Script/1.0"}
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
            "chance_of_playing_next_round": str(p.get("chance_of_playing_next_round", ""))
        }

    print("========================================================")
    print("               MODEL COMPARISON RUN                     ")
    print("========================================================")

    base_starters, base_cap, base_xp = solve_model(players, use_ensemble=False)
    ens_starters, ens_cap, ens_xp = solve_model(players, use_ensemble=True)

    base_ids = {s["id"] for s in base_starters}
    ens_ids = {s["id"] for s in ens_starters}
    diffs = ens_ids.symmetric_difference(base_ids)

    print(f"\n[BASELINE MODEL] Projected Starting xP: {base_xp:.2f} | Captain: {base_cap['name'] if base_cap else 'None'}")
    for s in base_starters:
        print(f"  - {s['name']} ({s['team']}, {s['pos']}, £{s['cost']}m)")

    print(f"\n[ENSEMBLE MODEL] Projected Starting xP: {ens_xp:.2f} | Captain: {ens_cap['name'] if ens_cap else 'None'}")
    for s in ens_starters:
        print(f"  - {s['name']} ({s['team']}, {s['pos']}, £{s['cost']}m)")

    print("\n--------------------------------------------------------")
    print(f"PROJECTED XP DELTA (Ensemble vs Baseline): {ens_xp - base_xp:+.2f} pts")
    print(f"STARTING XI DIFFERENCES: {len(diffs)} player(s) divergent.")
    print("========================================================")

if __name__ == "__main__":
    run_comparison()
