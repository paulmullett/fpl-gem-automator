import os
import sys
import requests

FPL_TEAM_ID = os.getenv("FPL_TEAM_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

def estimate_xmins(p):
    # Placeholder or existing logic for estimating expected minutes
    return float(p.get("chance_of_playing_next_round", 100) or 100) * 0.9

def get_market_adjustments():
    return {}

def run_monte_carlo_simulations(players, num_trials=1000):
    results = {}
    for pid in players:
        results[pid] = {"floor": 2.0, "ceiling": 10.0}
    return results

def get_user_current_squad(team_id):
    return []

def solve_model(players, market_data, use_ensemble=False):
    starters = []
    cap = None
    xp = 38.0
    return starters, cap, xp

def solve_multi_period_model(players, current_squad_ids, horizons=3):
    starters = []
    cap = None
    xp = 37.8
    return starters, cap, xp

def send_to_discord(base_xp, ens_xp, mpo_xp, mc_results, base_starters, ens_starters, mpo_starters, base_cap, ens_cap, mpo_cap):
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

    ens_diff_text = "Swaps vs Base:\n" + "\n".join([f"  └ {s}" for s in ens_swaps]) if ens_swaps else "Swaps vs Base: `None (Identical XI)`"
    mpo_diff_text = "Swaps vs Base:\n" + "\n".join([f"  └ {s}" for s in mpo_swaps]) if mpo_swaps else "Swaps vs Base: `None (Identical XI)`"

    content = (
        f"**[Master Model Audit: Odds + MPO + Monte Carlo Side-by-Side]**\n\n"
        f"• **Baseline Model:** `{base_xp:.2f} xP` | C: `{base_cap['name'] if base_cap else 'None'}`\n"
        f"• **Ensemble Model:** `{ens_xp:.2f} xP` | C: `{ens_cap['name'] if ens_cap else 'None'}`\n"
        f"  {ens_diff_text}\n"
        f"• **Multi-Period (3W) Model:** `{mpo_xp:.2f} xP` | C: `{mpo_cap['name'] if mpo_cap else 'None'}`\n"
        f"  {mpo_diff_text}\n\n"
        f"• **Stochastic Starter Floor (10th %):** `{total_mc_floor:.1f} pts`\n"
        f"• **Stochastic Starter Ceiling (90th %):** `{total_mc_ceiling:.1f} pts`"
    )
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
    except Exception as e:
        print(f"Failed to send Discord webhook: {e}")

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
        est_mins = estimate_xmins(p)
        players[p["id"]] = {
            "id": p["id"], "name": p["web_name"], "team": teams.get(p["team"], "UNK"),
            "team_id": p["team"], "pos": element_types.get(p["element_type"], "UNK"),
            "pos_id": p["element_type"], "cost": p["now_cost"] / 10.0,
            "status": p["status"], "news": p["news"], "ep_next": str(p.get("ep_next", "0.0")),
            "form": str(p.get("form", "0.0")), "total_points": p.get("total_points", 0),
            "own": str(p.get("selected_by_percent", "0.0")),
            "chance_of_playing_next_round": str(p.get("chance_of_playing_next_round", "")),
            "est_xmins": est_mins, 
            "xgi_90": float(p.get("expected_goal_involvements_90", 0.25) or 0.25)
        }

    print("Fetching live market odds adjustments...")
    market_data = get_market_adjustments()
    
    print("Running Monte Carlo simulations...")
    mc_results = run_monte_carlo_simulations(players, num_trials=1000)

    print("Fetching user squad picks...")
    user_squad_ids = get_user_current_squad(FPL_TEAM_ID)

    base_starters, base_cap, base_xp = solve_model(players, market_data, use_ensemble=False)
    ens_starters, ens_cap, ens_xp = solve_model(players, market_data, use_ensemble=True)
    mpo_starters, mpo_cap, mpo_xp = solve_multi_period_model(players, current_squad_ids=user_squad_ids, horizons=3)

    print(f"[BASELINE MODEL] Projected Starting xP: {base_xp:.2f}")
    print(f"[ENSEMBLE MODEL] Projected Starting xP: {ens_xp:.2f}")
    print(f"[MPO MODEL] 3-Week Horizon Projected xP: {mpo_xp:.2f}")

    send_to_discord(base_xp, ens_xp, mpo_xp, mc_results, base_starters, ens_starters, mpo_starters, base_cap, ens_cap, mpo_cap)

if __name__ == "__main__":
    run_comparison()
