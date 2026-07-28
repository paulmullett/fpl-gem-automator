import os
import sys
import requests
import pulp
import math
from scipy.optimize import fsolve

FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

if not FPL_TEAM_ID:
    print("CRITICAL ERROR: Missing FPL_TEAM_ID environment variable.")
    sys.exit(1)

# ==========================================
# 1. LIVE BOOKMAKER ODDS ENGINE
# ==========================================
def fetch_live_odds():
    if not ODDS_API_KEY:
        print("WARNING: Missing ODDS_API_KEY. Skipping live market odds and falling back to base heuristics.")
        return []
    
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching odds: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception during odds API query: {e}")
    return []

def remove_vig(odds_list):
    implied_probs = [1.0 / d for d in odds_list if d > 1.0]
    total = sum(implied_probs)
    if total <= 0:
        return [1.0 / len(odds_list)] * len(odds_list)
    return [p / total for p in implied_probs]

def compute_implied_metrics(match_event):
    home_team = match_event.get("home_team")
    away_team = match_event.get("away_team")
    bookmakers = match_event.get("bookmakers", [])
    
    if not bookmakers:
        return None

    target_bookie = next((b for b in bookmakers if b["key"] in ["pinnacle", "bet365", "skybet"]), bookmakers[0])
    markets = {m["key"]: m["outcomes"] for m in target_bookie.get("markets", [])}

    h2h = markets.get("h2h", [])
    totals = markets.get("totals", [])

    if not h2h or not totals:
        return None

    h_odds = next((o["price"] for o in h2h if o["name"] == home_team), 2.0)
    a_odds = next((o["price"] for o in h2h if o["name"] == away_team), 2.0)
    d_odds = next((o["price"] for o in h2h if o["name"] == "Draw"), 3.5)

    clean_h2h = remove_vig([h_odds, d_odds, a_odds])
    p_home, p_away = clean_h2h[0], clean_h2h[2]

    over_odds = next((o["price"] for o in totals if o["name"] == "Over" and o.get("point", 2.5) == 2.5), 1.9)
    under_odds = next((o["price"] for o in totals if o["name"] == "Under" and o.get("point", 2.5) == 2.5), 1.9)
    clean_totals = remove_vig([over_odds, under_odds])
    p_under_25 = clean_totals[1]

    def objective(lam):
        if lam[0] <= 0:
            return 1.0
        p_under_model = math.exp(-lam[0]) * (1.0 + lam[0] + (lam[0]**2) / 2.0)
        return p_under_model - p_under_25

    lam_total_initial = [2.5]
    try:
        lam_total = fsolve(objective, lam_total_initial)[0]
    except:
        lam_total = 2.5
    lam_total = max(0.5, min(5.0, lam_total))

    advantage_factor = max(0.1, p_home / max(0.01, p_away))
    fraction_home = advantage_factor / (1.0 + advantage_factor)
    fraction_home = max(0.2, min(0.8, fraction_home))
    
    lam_home = lam_total * fraction_home
    lam_away = lam_total * (1.0 - fraction_home)

    cs_home = math.exp(-lam_away)
    cs_away = math.exp(-lam_home)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
        "cs_probability_home": round(cs_home, 3),
        "cs_probability_away": round(cs_away, 3)
    }

def get_market_adjustments():
    events = fetch_live_odds()
    adjustments = {}
    for event in events:
        metrics = compute_implied_metrics(event)
        if metrics:
            adjustments[metrics["home_team"]] = {
                "xG": metrics["lambda_home"],
                "xGC": metrics["lambda_away"],
                "cs_prob": metrics["cs_probability_home"]
            }
            adjustments[metrics["away_team"]] = {
                "xG": metrics["lambda_away"],
                "xGC": metrics["lambda_home"],
                "cs_prob": metrics["cs_probability_away"]
            }
    return adjustments

# ==========================================
# 2. FPL EVALUATION & MODEL MATH
# ==========================================
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
    confidence = min(1.0, (own / 15.0) + (max(0.0, cost - cost_threshold) / 2.0))
    shrunken_xgi = (xgi * confidence) + (baseline_xgi * (1.0 - confidence))

    prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (xmins - 60.0)))
    app_points = (prob_60 * 2.0) + ((1.0 - prob_60) * 1.0)
    
    team_xga = xgc * mins_factor
    cs_prob = math.exp(-team_xga) if team_xga > 0 else 1.0
    cs_points = (cs_prob * (4.0 if pos_id in [1, 2] else (1.0 if pos_id == 3 else 0.0))) * prob_60
    
    market_premium = 1.0 + (max(0, cost - 5.5) * 0.04)
    pos_mult = 4.2 if pos_id == 2 else (4.0 if pos_id == 3 else 3.6)
    attacking_points = (shrunken_xgi * mins_factor) * pos_mult * market_premium
    
    raw_ev = app_points + attacking_points + cs_points
    return (raw_ev * 0.70) + (ep * 0.30)

def get_ensemble_ev(p, xmins_overrides, market_data):
    ev_a = get_base_ev(p, xmins_overrides)
    team_name = p.get("team")
    
    # Apply Live Bookmaker Odds Calibration Stream B Multipliers if available
    if market_data and team_name in market_data:
        m_metrics = market_data[team_name]
        pos_id = p["pos_id"]
        if pos_id in [1, 2]: # GKP / DEF Clean Sheet odds adjustment
            market_cs_mult = m_metrics["cs_prob"] / 0.35
            ev_a *= (0.75 + (0.25 * market_cs_mult))
        else: # MID / FWD Attacking xG odds adjustment
            market_xg_mult = m_metrics["xG"] / 1.35
            ev_a *= (0.75 + (0.25 * market_xg_mult))

    try:
        form = float(p.get("form", 0.0))
        ep = float(p.get("ep_next", 0.0))
    except:
        form, ep = 0.0, 0.0
    ev_b = max(0.0, (form * 0.6) + (ep * 0.4))
    return (0.70 * ev_a) + (0.30 * ev_b)

# ==========================================
# 3. MILP OPTIMIZATION SOLVER
# ==========================================
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
    total_xp = sum(ev_func(s, {}) for s in starters)
    if captain:
        total_xp += ev_func(captain, {})

    return starters, captain, total_xp

# ==========================================
# 4. DISCORD NOTIFICATION & PIPELINE EXECUTION
# ==========================================
def send_to_discord(base_xp, ens_xp, diffs, base_cap, ens_cap):
    if not DISCORD_WEBHOOK_URL:
        return
    
    diff_text = f"{len(diffs)} divergent starter(s)." if diffs else "No starting XI differences."
    content = (
        f"**[Model Comparison Audit with Live Odds]**\n"
        f"• **Baseline xP:** `{base_xp:.2f}` | Captain: `{base_cap['name'] if base_cap else 'None'}`\n"
        f"• **Ensemble xP:** `{ens_xp:.2f}` | Captain: `{ens_cap['name'] if ens_cap else 'None'}`\n"
        f"• **Delta:** `{ens_xp - base_xp:+.2f} pts` | {diff_text}"
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
        players[p["id"]] = {
            "id": p["id"], "name": p["web_name"], "team": teams.get(p["team"], "UNK"),
            "team_id": p["team"], "pos": element_types.get(p["element_type"], "UNK"),
            "pos_id": p["element_type"], "cost": p["now_cost"] / 10.0,
            "status": p["status"], "news": p["news"], "ep_next": str(p.get("ep_next", "0.0")),
            "form": str(p.get("form", "0.0")), "total_points": p.get("total_points", 0),
            "own": str(p.get("selected_by_percent", "0.0")),
            "chance_of_playing_next_round": str(p.get("chance_of_playing_next_round", ""))
        }

    print("Fetching live market odds adjustments...")
    market_data = get_market_adjustments()

    base_starters, base_cap, base_xp = solve_model(players, market_data, use_ensemble=False)
    ens_starters, ens_cap, ens_xp = solve_model(players, market_data, use_ensemble=True)

    base_ids = {s["id"] for s in base_starters}
    ens_ids = {s["id"] for s in ens_starters}
    diffs = ens_ids.symmetric_difference(base_ids)

    print(f"[BASELINE MODEL] Projected Starting xP: {base_xp:.2f} | Captain: {base_cap['name'] if base_cap else 'None'}")
    print(f"[ENSEMBLE MODEL] Projected Starting xP: {ens_xp:.2f} | Captain: {ens_cap['name'] if ens_cap else 'None'}")
    print(f"PROJECTED XP DELTA: {ens_xp - base_xp:+.2f} pts | STARTING XI DIFFERENCES: {len(diffs)}")

    send_to_discord(base_xp, ens_xp, diffs, base_cap, ens_cap)

if __name__ == "__main__":
    run_comparison()
