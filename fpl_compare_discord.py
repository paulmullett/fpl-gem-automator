import os
import sys
import requests
import math

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PLAYER_A = os.environ.get("PLAYER_A", "").strip()
PLAYER_B = os.environ.get("PLAYER_B", "").strip()

if not DISCORD_WEBHOOK_URL:
    print("CRITICAL ERROR: Missing DISCORD_WEBHOOK_URL secret.")
    sys.exit(1)

if not PLAYER_A or not PLAYER_B:
    print("CRITICAL ERROR: Both PLAYER_A and PLAYER_B inputs must be provided.")
    sys.exit(1)

def get_bootstrap_players():
    headers = {"User-Agent": "FPL-Comparison-Automation/1.0"}
    resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    if resp.status_code != 200:
        print("CRITICAL ERROR: Could not fetch FPL bootstrap data.")
        sys.exit(1)
    data = resp.json()
    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in data["element_types"]}
    
    players = {}
    for p in data["elements"]:
        players[p["id"]] = {
            "id": p["id"],
            "name": p["web_name"],
            "full_name": f"{p['first_name']} {p['second_name']}",
            "team": teams.get(p["team"], "UNK"),
            "pos": element_types.get(p["element_type"], "UNK"),
            "pos_id": p["element_type"],
            "cost": p["now_cost"] / 10.0,
            "status": p["status"],
            "own": float(p.get("selected_by_percent", 0.0)),
            "ep_next": float(p.get("ep_next", 0.0) or 0.0),
            "xgi_90": float(p.get("expected_goal_involvements_per_90", 0.0) or 0.0),
            "xgc_90": float(p.get("expected_goals_conceded_per_90", 1.35) or 1.35),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round", 100)
        }
    return players

def evaluate_single_player(p):
    xmins = 85.0 if p["status"] == "a" else 45.0
    chance = p["chance_of_playing_next_round"]
    if chance is not None:
        try:
            xmins *= (float(chance) / 100.0)
        except:
            pass
            
    pos_id = p["pos_id"]
    mins_factor = xmins / 90.0
    
    baseline_xgi = 0.01 if pos_id == 1 else (0.08 if pos_id == 2 else (0.25 if pos_id == 3 else 0.35))
    confidence = min(1.0, (p["own"] / 15.0) + (max(0.0, p["cost"] - 4.5) / 2.0))
    shrunken_xgi = (p["xgi_90"] * confidence) + (baseline_xgi * (1.0 - confidence))
    
    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)
    
    team_xga = p["xgc_90"] * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    cs_points = (cs_prob * (4.0 if pos_id in [1, 2] else (1.0 if pos_id == 3 else 0.0))) * prob_60
    
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult
    
    raw_ev = app_points + attacking_points + cs_points
    return round(raw_ev, 2), round(xmins, 1)

def main():
    players = get_bootstrap_players()
    
    p1 = next((p for p in players.values() if PLAYER_A.lower() in p["name"].lower() or PLAYER_A.lower() in p["full_name"].lower()), None)
    p2 = next((p for p in players.values() if PLAYER_B.lower() in p["name"].lower() or PLAYER_B.lower() in p["full_name"].lower()), None)
    
    if not p1 or not p2:
        error_msg = f"Comparison Error: Could not resolve players. Found A: {p1['name'] if p1 else 'None'}, Found B: {p2['name'] if p2 else 'None'}"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": error_msg})
        sys.exit(1)

    ev1, xmins1 = evaluate_single_player(p1)
    ev2, xmins2 = evaluate_single_player(p2)
    
    winner = p1['name'] if ev1 > ev2 else (p2['name'] if ev2 > ev1 else "Statistical Tie")
    
    discord_output = (
        f"**[FPL Player Head-to-Head Audit]**\n\n"
        f"```text\n"
        f"================================================================================\n"
        f"METRIC                  | {p1['name']} ({p1['team']})     | {p2['name']} ({p2['team']})\n"
        f"--------------------------------------------------------------------------------\n"
        f"Position                | {p1['pos']}                         | {p2['pos']}\n"
        f"Cost                    | £{p1['cost']}m                        | £{p2['cost']}m\n"
        f"Ownership               | {p1['own']}%                        | {p2['own']}%\n"
        f"Status                  | {p1['status']}                           | {p2['status']}\n"
        f"Proj. Expected Mins     | {xmins1}m                       | {xmins2}m\n"
        f"Raw xGI / 90            | {p1['xgi_90']}                      | {p2['xgi_90']}\n"
        f"Calculated 1-GW EV      | {ev1} pts                   | {ev2} pts\n"
        f"================================================================================\n"
        f"Model Recommendation: {winner} projects higher structural value.\n"
        f"================================================================================\n"
        f"```"
    )
    
    requests.post(DISCORD_WEBHOOK_URL, json={"content": discord_output})
    print("Comparison successfully posted to Discord.")

if __name__ == "__main__":
    main()
