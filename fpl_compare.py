import os
import sys
import requests
import pulp
import math

from fpl_odds_engine import get_market_adjustments
from fpl_mpo_engine import solve_multi_period_model
from fpl_monte_carlo import run_monte_carlo_simulations

# combined functions
from fpl_funcs import (
    estimate_xmins, 
    calculate_tier1_translation_factor,
    get_gameweek_state
)

# 1. Fetch the master FPL data payload FIRST
headers = {"User-Agent": "FPL-Compare-Script/1.0"}
try:
    resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10, headers=headers)
    bootstrap_data = resp.json()
except Exception as e:
    print(f"Failed to fetch FPL data: {e}")
    exit(1)

# 2. Now you can resolve gameweek targets because bootstrap_data exists
active_gw, target_gw = get_gameweek_state(bootstrap_data)

FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

if not FPL_TEAM_ID:
    print("CRITICAL ERROR: Missing FPL_TEAM_ID environment variable.")
    sys.exit(1)

def get_user_current_squad(team_id):
    """Fetch the user's starting picks from the active/last-completed gameweek."""
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{active_gw}/picks/"
    try:
        resp = requests.get(url, headers={"User-Agent": "FPL-Compare-Script/1.0"}, timeout=5)
        if resp.status_code == 200:
            picks = resp.json().get("picks", [])
            return [p["element"] for p in picks]
        else:
            print(f"Could not fetch picks for team {team_id} on GW{active_gw} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"Could not fetch user picks for team {team_id}: {e}")
    return []

# xmins Moved to fpl_funcs.py

# league translations moved to fpl_funcs.py

def get_base_ev(p, xmins_overrides):
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
    
    baseline_xgi = 0.01 if pos_id == 1 else (0.08 if pos_id == 2 else (0.25 if pos_id == 3 else 0.35))
    cost_threshold = 4.0 if pos_id in [1, 2] else 4.5
    cost_premium = max(0.0, cost - cost_threshold)
    confidence = min(1.0, (own / 15.0) + (cost_premium / 2.0))
    
    translation_mult = calculate_tier1_translation_factor(p)
    adjusted_xgi = xgi * translation_mult
    
    shrunken_xgi = (adjusted_xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)
    
    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    cs_points = (cs_prob * (4.0 if pos_id in [1, 2] else (1.0 if pos_id == 3 else 0.0))) * prob_60
    
    extra_defensive_points = 0.0
    if pos_id == 1:
        estimated_saves = max(1.5, (xgc * 1.4))
        extra_defensive_points = (estimated_saves / 3.0) * 0.33 * mins_factor
    elif pos_id == 2:
        extra_defensive_points = 0.22 * mins_factor if cost >= 5.5 else 0.08
    
    market_premium = 1.0 + (max(0, cost - 5.5) * 0.04)
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium
    
    raw_ev = app_points + attacking_points + cs_points + extra_defensive_points
    return (raw_ev * 0.70) + (ep * 0.30)

def get_ensemble_ev(p, xmins_overrides, market_data):
    ev_a = get_base_ev(p, xmins_overrides)
    team_name = p.get("team")
    
    if market_data and team_name in market_data:
        m_metrics = market_data[team_name]
        pos_id = p["pos_id"]
        if pos_id in [1, 2]:
            market_cs_mult = m_metrics["cs_prob"] / 0.35
            ev_a *= (0.75 + (0.25 * market_cs_mult))
        else:
            market_xg_mult = m_metrics["xG"] / 1.35
            ev_a *= (0.75 + (0.25 * market_xg_mult))

    try:
        form = float(p.get("form", 0.0))
        ep = float(p.get("ep_next", 0.0))
    except:
        form, ep = 0.0, 0.0
    ev_b = max(0.0, (form * 0.6) + (ep * 0.4))
    return (0.70 * ev_a) + (0.30 * ev_b)

def solve_model(players_dict, market_data, use_ensemble=False):
    prob = pulp.LpProblem(f"FPL_{'Ensemble' if use_ensemble else 'Baseline'}", pulp.LpMaximize)
    valid_ids = list(players_dict.keys())
    
    squad_vars = pulp.LpVariable.dicts("squad", valid_ids, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", valid_ids, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", valid_ids, cat="Binary")

    objective = []
    for i in valid_ids:
        p = players_dict[i]
        ev = get_ensemble_ev(p, {}, market_data) if use_ensemble else get_base_ev(p, {})
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
    
    ev_func = (lambda s, x: get_ensemble_ev(s, x, market_data)) if use_ensemble else get_base_ev
    
    eligible_vcs = [s for s in starters if captain and s["id"] != captain["id"]]
    vice_captain = max(eligible_vcs, key=lambda s: ev_func(s, {})) if eligible_vcs else None

    total_xp = sum(ev_func(s, {}) for s in starters)
    if captain:
        total_xp += ev_func(captain, {})

    return starters, captain, vice_captain, total_xp

def send_to_discord(base_xp, ens_xp, mpo_xp, mc_results, base_starters, ens_starters, mpo_starters, base_cap, ens_cap, mpo_cap, base_vc, ens_vc, mpo_vc):
    if not DISCORD_WEBHOOK_URL:
        return
        
    base_starter_ids = {s["id"] for s in base_starters}

    total_mc_floor = sum(mc_results[pid]["floor"] for pid in base_starter_ids if pid in mc_results)
    total_mc_ceiling = sum(mc_results[pid]["ceiling"] for pid in base_starter_ids if pid in mc_results)
    
    def format_position_swaps(model_starters):
        base_map = {s["id"]: s for s in base_starters}
        model_map = {s["id"]: s for s in model_starters}
        
        out_ids = set(base_map.keys()) - set(model_map.keys())
        in_ids = set(model_map.keys()) - set(base_map.keys())
        
        out_players = [base_map[pid] for pid in out_ids]
        in_players = [model_map[pid] for pid in in_ids]
        
        swaps = []
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_outs = [p for p in out_players if p["pos"] == pos]
            pos_ins = [p for p in in_players if p["pos"] == pos]
            for o, i in zip(pos_outs, pos_ins):
                swaps.append(f"{o['name']} ➔ {i['name']}")
                out_players.remove(o)
                in_players.remove(i)
                
        for o, i in zip(out_players, in_players):
            swaps.append(f"{o['name']} ➔ {i['name']}")
            
        return swaps

    ens_swaps = format_position_swaps(ens_starters)
    mpo_swaps = format_position_swaps(mpo_starters)

    # --- ADDED EXPLICIT MODEL INDICATORS ---
    ens_diff_text = "Swaps vs Base `[Odds/Market Shift]`:\n" + "\n".join([f"  └ {s}" for s in ens_swaps]) if ens_swaps else "Swaps vs Base `[Odds/Market Shift]`: `None (Identical XI)`"
    mpo_diff_text = "Swaps vs Base `[3W Horizon / Fixture Shift]`:\n" + "\n".join([f"  └ {s}" for s in mpo_swaps]) if mpo_swaps else "Swaps vs Base `[3W Horizon / Fixture Shift]`: `None (Identical XI)`"

    base_c_str = f"C: `{base_cap['name']}` | V: `{base_vc['name']}`" if base_cap and base_vc else "None"
    ens_c_str = f"C: `{ens_cap['name']}` | V: `{ens_vc['name']}`" if ens_cap and ens_vc else "None"
    mpo_c_str = f"C: `{mpo_cap['name']}` | V: `{mpo_vc['name']}`" if mpo_cap and mpo_vc else "None"

    content = (
        f"**[Master Model Audit: Odds + MPO + Monte Carlo Side-by-Side]**\n\n"
        f"• **Baseline Model:** `{base_xp:.2f} xP` | {base_c_str}\n"
        f"• **Ensemble Model:** `{ens_xp:.2f} xP` | {ens_c_str}\n"
        f"  {ens_diff_text}\n"
        f"• **Multi-Period (3W) Model:** `{mpo_xp:.2f} xP` | {mpo_c_str}\n"
        f"  {mpo_diff_text}\n\n"
        f"• **Stochastic Starter Floor (10th %):** `{total_mc_floor:.1f} pts`\n"
        f"• **Stochastic Starter Ceiling (90th %):** `{total_mc_ceiling:.1f} pts`"
    )
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
    except Exception as e:
        print(f"Failed to send Discord webhook: {e}")

def run_comparison():
    #headers = {"User-Agent": "FPL-Compare-Script/1.0"}
    #resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    #if resp.status_code != 200:
    #    print("CRITICAL ERROR: Failed to reach FPL API.")
    #    sys.exit(1)
    #bootstrap_data = resp.json()

    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}

    players = {}
    for p in bootstrap_data["elements"]:
        est_mins = estimate_xmins(p)
        players[p["id"]] = {
        "id": p["id"], "name": p["web_name"], "team": teams.get(p["team"], "UNK"),
        "team_id": p["team"], "pos": element_types.get(p["element_type"], "UNK"),
        "pos_id": p["element_type"], "cost": p["now_cost"] / 10.0,
        "status": p["status"], "news": p["news"], "ep_next": str(p.get("ep_next", "0.0")),
        "total_points": p.get("total_points", 0), "form": str(p.get("form", "0.0")),
        "own": str(p.get("selected_by_percent", "0.0")),
        "chance_of_playing_next_round": str(p.get("chance_of_playing_next_round", "")),
        "xgi_90": str(p.get("expected_goal_involvements_per_90", "0.0")),
        "xgc_90": str(p.get("expected_goals_conceded_per_90", "0.0")),
        "cost_change_start": p.get("cost_change_start", 0),
        "source_league": p.get("source_league", "Premier_League"),
        "age": p.get("age", 25),
        "former_team_possession_pct": p.get("former_team_possession_pct", 50.0),
        "has_stale_pl_history": p.get("has_stale_pl_history", False),
        "recent_european_peak": p.get("recent_european_peak", False)
    }

    print("Fetching live market odds adjustments...")
    market_data = get_market_adjustments()
    
    print("Running Monte Carlo simulations...")
    mc_results = run_monte_carlo_simulations(players, num_trials=1000)

    print("Fetching user squad picks...")
    user_squad_ids = get_user_current_squad(FPL_TEAM_ID)
    
    base_starters, base_cap, base_vc, base_xp = solve_model(players, market_data, use_ensemble=False)
    ens_starters, ens_cap, ens_vc, ens_xp = solve_model(players, market_data, use_ensemble=True)
    mpo_starters, mpo_cap, mpo_xp = solve_multi_period_model(players, current_squad_ids=user_squad_ids, horizons=3)
    
    mpo_eligible_vcs = [s for s in mpo_starters if mpo_cap and s["id"] != mpo_cap["id"]]
    mpo_vc = max(mpo_eligible_vcs, key=lambda s: get_base_ev(s, {})) if mpo_eligible_vcs else None

    send_to_discord(
        base_xp, ens_xp, mpo_xp, mc_results, 
        base_starters, ens_starters, mpo_starters, 
        base_cap, ens_cap, mpo_cap,
        base_vc, ens_vc, mpo_vc
    )

if __name__ == "__main__":
    run_comparison()
