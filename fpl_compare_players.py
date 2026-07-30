import os
import sys
import requests
import math

# combined functions
from fpl_funcs import estimate_xmins

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PLAYER_A = os.environ.get("PLAYER_A", "").strip()
PLAYER_B = os.environ.get("PLAYER_B", "").strip()

if not DISCORD_WEBHOOK_URL:
    print("CRITICAL ERROR: Missing DISCORD_WEBHOOK_URL secret.")
    sys.exit(1)

if not PLAYER_A or not PLAYER_B:
    print("CRITICAL ERROR: Both PLAYER_A and PLAYER_B inputs must be provided.")
    sys.exit(1)

# Tier 1 Dynamic Translation Engine
LEAGUE_BASE_STRENGTHS = {
    "Champions_League": 0.96, "Bundesliga": 0.90, "Serie_A": 0.88,
    "La_Liga": 0.89, "Ligue_1": 0.84, "Eredivisie": 0.79,
    "Pro_League": 0.77, "Championship": 0.87, "Premier_League": 1.00, "Other_Foreign": 0.72
}

def calculate_tier1_translation_factor(p):
    league = p.get("source_league", "Premier_League")
    base_coef = LEAGUE_BASE_STRENGTHS.get(league, 0.75)
    if league == "Premier_League":
        return 1.00
    try: age = int(p.get("age", 25))
    except: age = 25
    age_modifier = 1.05 if age <= 22 else (0.92 if age >= 29 else 1.00)
    pos_id = p["pos_id"]
    pos_resilience = 1.02 if pos_id in [3, 4] else 0.96
    return max(0.65, min(0.98, base_coef * age_modifier * pos_resilience))

# Replaced xmins centrally

def get_bootstrap_players():
    headers = {"User-Agent": "FPL-Comparison-Automation/2.0"}
    resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    if resp.status_code != 200:
        print("CRITICAL ERROR: Could not fetch FPL bootstrap data.")
        sys.exit(1)
    data = resp.json()
    teams = {t["id"]: t["short_name"] for t in data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in data["element_types"]}
    
    players = {}
    for p in data["elements"]:
        raw_xgi = float(p.get("expected_goal_involvements_per_90", 0.0) or 0.0)
        cost = p["now_cost"] / 10.0
        pos_id = p["element_type"]
        
        # Zero-history foreign transfer proxy fallback (fixes Tzolis 0.0 API bug)
        if raw_xgi == 0.0 and cost >= 6.0:
            raw_xgi = 0.15 if pos_id == 2 else (0.30 if pos_id == 3 else 0.45)
            
        players[p["id"]] = {
            "id": p["id"],
            "name": p["web_name"],
            "full_name": f"{p['first_name']} {p['second_name']}",
            "team": teams.get(p["team"], "UNK"),
            "pos": element_types.get(pos_id, "UNK"),
            "pos_id": pos_id,
            "cost": cost,
            "status": p["status"],
            "own": float(p.get("selected_by_percent", 0.0)),
            "ep_next": float(p.get("ep_next", 0.0) or 0.0),
            "xgi_90": raw_xgi,
            "xgc_90": float(p.get("expected_goals_conceded_per_90", 1.35) or 1.35),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round", 100),
            "source_league": p.get("source_league", "Premier_League"),
            "age": p.get("age", 25)
        }
    return players

def evaluate_player_models(p):
    xmins = estimate_xmins(p)
    if xmins < 5.0:
        return 0.0, 0.0, 0.0, 0.0
        
    pos_id = p["pos_id"]
    mins_factor = xmins / 90.0
    
    baseline_xgi = 0.01 if pos_id == 1 else (0.08 if pos_id == 2 else (0.25 if pos_id == 3 else 0.35))
    confidence = min(1.0, (p["own"] / 15.0) + (max(0.0, p["cost"] - 4.5) / 2.0))
    
    translation_mult = calculate_tier1_translation_factor(p)
    adjusted_xgi = p["xgi_90"] * translation_mult
    shrunken_xgi = (adjusted_xgi * confidence) + (baseline_xgi * (1.0 - confidence))
    
    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)
    
    team_xga = p["xgc_90"] * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    cs_points = (cs_prob * (4.0 if pos_id in [1, 2] else (1.0 if pos_id == 3 else 0.0))) * prob_60
    
    # --- DEFCON & GOALKEEPER SAVE/BPS MODELLING ---
    extra_defensive_points = 0.0
    if pos_id == 1:
        # Goalkeeper save volume estimation (factoring in save-point baselines)
        estimated_saves = max(1.5, (p["xgc_90"] * 1.4))
        extra_defensive_points = (estimated_saves / 3.0) * 0.33 * mins_factor
    elif pos_id == 2:
        # Centre-back / Full-back defensive contribution & BPS weighting
        extra_defensive_points = 0.22 * mins_factor if p["cost"] >= 5.5 else 0.08
    
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    market_premium = 1.0 + (max(0, p["cost"] - 5.5) * 0.04)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium
    
    raw_ev = app_points + attacking_points + cs_points + extra_defensive_points
    ep = p["ep_next"]
    
    # Model Variations Calculation
    base_ev = (raw_ev * 0.70) + (ep * 0.30)
    ensemble_ev = (raw_ev * 0.85) + (ep * 0.15) * 0.95
    multi_period_ev = base_ev * 0.92 + (ep * 0.08)
    
    return round(base_ev, 2), round(ensemble_ev, 2), round(multi_period_ev, 2), round(xmins, 1)

def main():
    players = get_bootstrap_players()
    
    def find_player(search_term):
        term = search_term.lower().strip()
        for p in players.values():
            if term in p["name"].lower() or term in p["full_name"].lower():
                return p
        return None

    p1 = find_player(PLAYER_A)
    p2 = find_player(PLAYER_B)
    
    if not p1 or not p2:
        missing = []
        if not p1: missing.append(f"Player A ('{PLAYER_A}')")
        if not p2: missing.append(f"Player B ('{PLAYER_B}')")
        error_msg = f"Comparison Error: Could not resolve {' and '.join(missing)}. Check spelling."
        requests.post(DISCORD_WEBHOOK_URL, json={"content": error_msg})
        sys.exit(1)

    base1, ens1, mp1, xmins1 = evaluate_player_models(p1)
    base2, ens2, mp2, xmins2 = evaluate_player_models(p2)
    
    # --- DROP THE TWO LINES HERE ---
    val_ratio1 = round(base1 / p1['cost'], 2)
    val_ratio2 = round(base2 / p2['cost'], 2)
    
    winner = p1['name'] if base1 > base2 else (p2['name'] if base2 > base1 else "Statistical Tie")
    
    discord_output = (
        f"**[Multi-Model Player Head-to-Head Audit]**\n\n"
        f"```text\n"
        f"================================================================================\n"
        f"METRIC                  | {p1['name']} ({p1['team']})     | {p2['name']} ({p2['team']})\n"
        f"--------------------------------------------------------------------------------\n"
        f"Position                | {p1['pos']}                         | {p2['pos']}\n"
        f"Cost                    | £{p1['cost']}m                        | £{p2['cost']}m\n"
        f"Ownership               | {p1['own']}%                        | {p2['own']}%\n"
        f"Proj. Expected Mins     | {xmins1}m                       | {xmins2}m\n"
        f"Adjusted xGI / 90       | {p1['xgi_90']}                      | {p2['xgi_90']}\n"
        f"--------------------------------------------------------------------------------\n"
        f"Baseline Model EV       | {base1} pts                  | {base2} pts\n"
        f"Ensemble Model EV       | {ens1} pts                  | {ens2} pts\n"
        f"Multi-Period Model EV   | {mp1} pts                  | {mp2} pts\n"
        # --- ADD THE NEW ROW TO THE TABLE TEMPLATE HERE ---
        f"Value Ratio (EV / £m)   | {val_ratio1} pts/£m              | {val_ratio2} pts/£m\n"
        f"================================================================================\n"
        f"Model Recommendation: {winner} projects higher structural value.\n"
        f"================================================================================\n"
        f"```"
    )
    
    requests.post(DISCORD_WEBHOOK_URL, json={"content": discord_output})
    print("Multi-model comparison posted to Discord.")

if __name__ == "__main__":
    main()
