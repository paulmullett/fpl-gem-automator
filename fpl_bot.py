import os
import sys
import json
import requests
import pulp
from google import genai
from google.genai import types
from ddgs import DDGS

# 1. Environment & Pre-Flight Check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")
STATE_FILE_PATH = "fpl_state.json"

# CONFIGURATION: Update this list at the start of each season
UEFA_TEAMS = {"MCI", "ARS", "LIV", "AVL", "MUN", "NEW", "CHE", "TOT", "SUN"}

if not all([GEMINI_API_KEY, DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing GitHub Secrets (GEMINI_API_KEY, DISCORD_WEBHOOK_URL, or FPL_TEAM_ID).")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine. Your purpose is to evaluate squad selections, transfers, captaincy choices, and chip strategies using advanced underlying metrics, pitch geometry, game-state data, and FPL market economics. You must strictly adhere to the following analytical laws.

### CORE ANALYTICAL LAWS & METRIC DEFINITIONS

1. GAME-STATE NORMALISED ATTACKING & ACCIDENTAL ASSISTS
- Game-State Filter: Normalize non-penalty expected goals (npxG) and expected assisted goals (xAG) strictly to periods when the game state is level (0-0, 1-1) or trailing. Exclude all scoreline-skewed "garbage time" data.
- Finisher's Multiplier: Apply a multiplier based on a player's multi-season conversion history over their xG.
- Accidental Assist Rule: FPL awards assists for deflected passes if there is only one defensive touch inside the box. Prioritize "Raw Crosses into the Penalty Box" alongside xAG.

2. DEFENSIVE CONTRIBUTION (DefCon) & 2026/27 BPS MATHEMATICS
- DefCon Thresholds: Target baseline assets averaging >8.5 CBIT (Clearances, Blocks, Interceptions, Tackles) for defenders, and >10.5 CBIRT (+ Recoveries) for midfielders/forwards.
- BPS Adjustments: Centre-backs earn 1 BPS per 3 CBI actions. Dribblers face no -1 penalty for being tackled. Penalty goals are a flat 12 BPS for all positions.
- Goalkeeper Save/Variance Balance: Target goalkeepers only if their team concedes <1.5 Expected Goals Against (xGA) per match.
- Bench Fodder Exemption: Players priced \u00a34.0m-\u00a34.5m in Bench Slots 3-4 are exempt from starter metric thresholds.

3. SPATIAL GEOMETRY & FLANK MISMATCHES
- Cross-reference an attacker's primary pitch zone with the opposition's specific spatial weaknesses.

4. CHIP STRUCTURE & 5-TRANSFER ECONOMICS
- 5-Transfer Banking: Evaluate 0-transfer rolls as an appreciating asset building toward a 3-to-5 transfer "mini-wildcard".
- Point Hit Constraint: Forbidden from recommending a point hit (-4) unless the 4-Gameweek Expected Value (EV) of the incoming player exceeds the outgoing player by >5.5 points, OR if required to field 11 starters.

5. QUALITATIVE OVERRIDES & TIER-1 LEAK VERIFICATION
- Expected Minutes (xMins) Overrides: Verified Tier-1 ITK/Injury intelligence acts as absolute mathematical overrides to base xMins.
- 15-Minute Panic Rule: Late leaks affecting >2 decisions within 15 mins of deadline default to the original multi-week EV plan.

6. MATCH STAKES & FATIGUE MULTIPLIERS
- Cup Congestion Law: Apply a 25% xMins penalty to starters with <72 hours turnaround from cup matches.

7. FOREIGN TRANSFERS & ZERO-HISTORY ASSETS
- Translation Discount: Apply a 20% discount to expected attacking output (npxG/xAG) for unproven foreign arrivals.
- xMins Integration Cap: Cap initial xMins for new arrivals at 45-60 mins for their first 3 Gameweeks.

### OUTPUT FORMAT & AESTHETIC DIRECTIVES
You MUST format your analysis with extreme visual precision for Discord rendering.

SECTION 1: EXECUTIVE SUMMARY & CORE MOVES
- Strategic summary of squad state, economics, and core moves.
- Include a monospaced ASCII Pitch Map block inside a markdown text codeblock displaying the starting formation and bench hierarchy.
- Detail the Foreign Transfer / Law 7 Audit and Tactical Mitigation Flags.

SECTION 2: QUANTITATIVE TRADE-OFF SUMMARY
- Do NOT use wide Markdown tables (they break on mobile/Discord).
- Format strictly as compact, high-density, single-line bullet points per player:
  • **[Player Name]** ([POS] | [TEAM] | \u00a3[X.X]m) — **[xMins] xMins** | **[EV] EV** | [DefCon / xGA / Metric Status]

SECTION 3: TRANSFER ECONOMICS & CHIP STATUS
- Capital management, rolling transfer economics, market volatility, and macro chip timeline.

SECTION 4: SPATIAL, GAME-STATE & MOTIVATION JUSTIFICATION
- Include an ASCII Spatial Overload / Pitch Mismatch Diagram inside a markdown text codeblock.
- Detail Law 1 Game-State Normalization, Law 3 Spatial Mismatches, and Law 2 DefCon BPS math.

SECTION 5: ITK & CONGESTION AUDIT
- ASCII Matrix or structured breakdown covering ITK, Ben Dinnery injuries, and Ben Crellin schedule congestion.

MANDATORY SIGN-OFF: FINAL LOCKED-IN SQUAD SUMMARY
- You MUST conclude with an ASCII Box block formatted exactly as below, and you MUST wrap this entire block inside a markdown text codeblock:

================================================================================
FINAL LOCKED-IN SQUAD SUMMARY: GAMEWEEK [X]
================================================================================
Formation: [X-X-X] | Bank: \u00a3[X.X]m | Free Transfers: [X]

STARTING XI:
  GKP: [Player] (\u00a3[X.X]m)
  DEF: [Player] (\u00a3[X.X]m)
  ...

BENCH RESERVES:
  Slot 1: [Player] (\u00a3[X.X]m, [POS])
  Slot 2: [Player] (\u00a3[X.X]m, [POS])
  Slot 3: [Player] (\u00a3[X.X]m, [POS])
  Slot 4: [Player] (\u00a3[X.X]m, [POS])
================================================================================

CRITICAL FORMATTING RULE: Strictly AVOID LaTeX syntax (e.g. $,$$, \\text{}, \\times). Use plain text and standard Markdown formatting exclusively.
"""

# 2. State Persistence & Calibration Engine
def load_state():
    default_state = {
        "buyback_targets": {},
        "last_updated_gw": 0,
        "calibration_weights": {
            "xgi_weight": 0.70,
            "fdr_impact_factor": 0.10,
            "bench_discount": 0.05
        },
        "pending_evaluation": None,
        "performance_history": []
    }
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                saved_state = json.load(f)
                for key, val in default_state.items():
                    if key not in saved_state:
                        saved_state[key] = val
                return saved_state
        except Exception as e:
            print(f"WARNING: Error reading state file: {e}")
    return default_state

def save_state(state):
    try:
        with open(STATE_FILE_PATH, "w") as f:
            json.dump(state, f, indent=4)
        print("STATE ENGINE: Successfully saved updated strategy state to fpl_state.json")
    except Exception as e:
        print(f"ERROR: Failed to save state file: {e}")

def recalibrate_model(state, headers, active_gw):
    pending = state.get("pending_evaluation")
    if not pending:
        return state

    eval_gw = pending.get("gw")
    if active_gw <= eval_gw:
        return state

    print(f"CALIBRATION ENGINE: Evaluating Gameweek {eval_gw} model accuracy...")
    gw_history_url = f"[https://fantasy.premierleague.com/api/entry/](https://fantasy.premierleague.com/api/entry/){FPL_TEAM_ID}/event/{eval_gw}/picks/"
    try:
        resp = requests.get(gw_history_url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            actual_points = data.get("entry_history", {}).get("points", 0)
            projected_xP = pending.get("projected_xP", 0.0)
            residual_error = actual_points - projected_xP

            state["performance_history"].append({
                "gw": eval_gw,
                "projected_xP": round(projected_xP, 2),
                "actual_points": actual_points,
                "residual_error": round(residual_error, 2),
                "captain": pending.get("captain")
            })

            recent_history = state["performance_history"][-6:]
            if len(recent_history) >= 2:
                mean_error = sum(h["residual_error"] for h in recent_history) / len(recent_history)
                weights = state["calibration_weights"]
                learning_rate = 0.02

                if mean_error < -5.0:
                    weights["xgi_weight"] = max(0.50, round(weights["xgi_weight"] - learning_rate, 3))
                elif mean_error > 5.0:
                    weights["xgi_weight"] = min(1.00, round(weights["xgi_weight"] + learning_rate, 3))

            state["pending_evaluation"] = None
    except Exception as e:
        print(f"WARNING: Error during model recalibration: {e}")

    return state

# 3. Live Search Data Fetcher
def get_live_fpl_news():
    news_text = "### LIVE ITK NEWS & SCHEDULE DATA (Automatically Fetched)\n"
    try:
        crellin_results = DDGS().text("Ben Crellin FPL blank double gameweek updates", max_results=3)
        news_text += "--- SCHEDULE CHANGES (Ben Crellin) ---\n"
        for r in crellin_results:
            news_text += f"- {r.get('body', '')}\n"
            
        dinnery_results = DDGS().text("Ben Dinnery FPL injuries team news press conference", max_results=3)
        news_text += "\n--- INJURY UPDATES (Ben Dinnery) ---\n"
        for r in dinnery_results:
            news_text += f"- {r.get('body', '')}\n"
    except Exception as e:
        news_text += f"[Search tool failed to retrieve live data: {e}]\n"
    return news_text

# 4. Portfolio Risk-Adjusted Predictive Engine with Mitigation Flags
def estimate_xmins(p):
    chance = str(p.get("chance_of_playing_next_round", ""))
    if chance == "0":
        return 0
    if p.get("status") not in ["a", "d"]:
        return 0
    
    try: own = float(p.get("own", 0.0))
    except: own = 0.0
    try: cost = float(p.get("cost", 0.0))
    except: cost = 0.0
    
    if cost >= 8.0:
        return 90 if own >= 5.0 else 75
    if cost >= 6.0:
        if own >= 5.0: return 85
        elif own >= 1.5: return 65
        else: return 30
    
    if own >= 5.0: return 85
    elif own >= 1.5: return 65
    else: return 0

def get_variance_penalty(p):
    xmins = estimate_xmins(p)
    cost = float(p.get("cost", 0.0))
    
    if cost >= 10.0 and xmins >= 85:
        return 1.0
    if xmins < 80:
        return 0.85
    return 0.95

def get_base_ev(p, weights):
    xmins = estimate_xmins(p)
    if xmins < 15:
        return 0.0
        
    try: ep = float(p.get("ep_next", 0.0))
    except: ep = 0.0
    try: xgi = float(p.get("xgi_90", 0.0))
    except: xgi = 0.0
    try: xgc = float(p.get("xgc_90", 0.0))
    except: xgc = 0.0

    xgi_mult = weights.get("xgi_weight", 0.70)
    
    if ep <= 0.0:
        try:
            tp = float(p.get("total_points", 0.0))
            form = float(p.get("form", 0.0))
            ep = (tp / 38.0) + form
        except:
            ep = 0.0

    mins_factor = xmins / 90.0

    if p["pos_id"] in [3, 4]: 
        base_points = (ep * (1.0 - (xgi_mult / 2.0))) + (xgi * (xgi_mult * 2.8))
    elif p["pos_id"] == 2: 
        cs_boost = max(0, 1.5 - xgc)
        base_points = (ep * 0.7) + cs_boost + (xgi * xgi_mult)
    elif p["pos_id"] == 1: 
        cs_boost = max(0, 1.5 - xgc)
        base_points = (ep * 0.7) + (cs_boost * 1.5)
        
    return base_points * mins_factor

def get_macro_ev(p, team_avg_fdr, weights):
    base_ev = get_base_ev(p, weights)
    if base_ev <= 0.0:
        return 0.0
    
    variance_penalty = get_variance_penalty(p)
    ev_4gw = (base_ev * variance_penalty) * 4.0
    
    avg_fdr = team_avg_fdr.get(p["team_id"], 3.0)
    fdr_impact = weights.get("fdr_impact_factor", 0.10)
    fdr_multiplier = 1.0 + ((3.0 - avg_fdr) * fdr_impact)
    
    return ev_4gw * fdr_multiplier

def check_european_congestion_flags(starters, fixtures_data, target_gw):
    flags = []
    for p in starters:
        if p["team"] in UEFA_TEAMS:
            flags.append(f"[FLAG OPTION: European Turnaround Risk detected for {p['name']} ({p['team']}) due to mid-week fixture congestion. Consider bench contingency or rotation guard.]")
    return flags

def evaluate_dynamic_opportunity_cost(free_transfers, starters):
    flags = []
    try:
        ft = int(''.join(filter(str.isdigit, str(free_transfers))))
        if ft >= 3:
            flags.append("[FLAG OPTION: Mini-Wildcard Trigger Recommended — Transfer hoarding threshold reached (3+ FTs banked). Evaluate breaking transfer lock to capture imminent price/fixture swings rather than risking cap loss.]")
    except ValueError:
        pass
    return flags

# 5. Execution Engine: Portfolio Optimization MILP Solver
def solve_fpl_knapsack(players_dict, current_squad_ids, total_budget, free_transfers, team_avg_fdr, required_bank_reservation, weights):
    prob = pulp.LpProblem("FPL_Portfolio_Optimization", pulp.LpMaximize)
    valid_ids = list(players_dict.keys())
    bench_discount = weights.get("bench_discount", 0.05)
    
    squad_vars = pulp.LpVariable.dicts("squad", valid_ids, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", valid_ids, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", valid_ids, cat="Binary")
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat="Continuous")
    
    objective = []
    for i in valid_ids:
        p = players_dict[i]
        ev = get_macro_ev(p, team_avg_fdr, weights)
        
        try: ownership = float(p.get("own", 0.0))
        except: ownership = 0.0
        
        eo_defensive_shield = (ownership / 100.0) * 2.5 if ownership >= 40.0 else (ownership / 100.0) * 1.0
        own_tiebreaker = ownership * 0.0001
        
        objective.append(
            (ev * starter_vars[i]) + 
            ((ev + eo_defensive_shield) * captain_vars[i]) + 
            (bench_discount * ev * (squad_vars[i] - starter_vars[i])) + 
            (own_tiebreaker * squad_vars[i])
        )
        
    prob += pulp.lpSum(objective) - (4.0 * extra_transfers)
    
    for i in valid_ids:
        p = players_dict[i]
        
        prob += starter_vars[i] <= squad_vars[i]
        prob += captain_vars[i] <= starter_vars[i]
        
        if i not in current_squad_ids:
            chance = str(p.get("chance_of_playing_next_round", ""))
            if chance == "0" or p.get("status") not in ["a", "d"]:
                prob += squad_vars[i] == 0
                
        if estimate_xmins(p) < 60:
            prob += starter_vars[i] == 0

    prob += pulp.lpSum([squad_vars[i] for i in valid_ids]) == 15
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids]) == 11
    prob += pulp.lpSum([captain_vars[i] for i in valid_ids]) == 1
    
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
    
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) <= 5
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) >= 2
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) <= 5
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) >= 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) <= 3

    team_ids = set(players_dict[i]["team_id"] for i in valid_ids)
    for t_id in team_ids:
        prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

    effective_budget = total_budget - required_bank_reservation
    prob += pulp.lpSum([players_dict[i]["cost"] * squad_vars[i] for i in valid_ids]) <= effective_budget

    if str(free_transfers).lower() != "unlimited":
        try: ft = int(''.join(filter(str.isdigit, str(free_transfers))))
        except: ft = 1
        if current_squad_ids and len(current_squad_ids) == 15:
            overlap = [i for i in current_squad_ids if i in valid_ids]
            players_kept = pulp.lpSum([squad_vars[i] for i in overlap])
            transfers_made = 15 - players_kept
            prob += extra_transfers >= transfers_made - ft

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # --- PHASE 2: MICRO-OPTIMIZATION (SINGLE GW) ---
    squad_players = []
    for i in valid_ids:
        if squad_vars[i].varValue and squad_vars[i].varValue > 0.5:
            squad_players.append(players_dict[i])
            
    # Sub-solver to pick optimal starting XI strictly on 1-GW EV
    sub_prob = pulp.LpProblem("Phase2_StartingXI", pulp.LpMaximize)
    s_vars = pulp.LpVariable.dicts("s", [p["id"] for p in squad_players], cat="Binary")
    c_vars = pulp.LpVariable.dicts("c", [p["id"] for p in squad_players], cat="Binary")
    
    sub_objective = []
    for p in squad_players:
        ev_1gw = get_base_ev(p, weights) # 1-GW EV engine
        try: ownership = float(p.get("own", 0.0))
        except: ownership = 0.0
        eo_shield = (ownership / 100.0) * 2.5 if ownership >= 40.0 else (ownership / 100.0) * 1.0
        
        # Maximize 1-GW Starter EV + 1-GW Captaincy EV
        sub_objective.append( (ev_1gw * s_vars[p["id"]]) + ((ev_1gw + eo_shield) * c_vars[p["id"]]) )
        
    sub_prob += pulp.lpSum(sub_objective)
    
    # Standard Gameweek Constraints
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players]) == 11
    sub_prob += pulp.lpSum([c_vars[p["id"]] for p in squad_players]) == 1
    
    for p in squad_players:
        sub_prob += c_vars[p["id"]] <= s_vars[p["id"]]
        if estimate_xmins(p) < 60:
            sub_prob += s_vars[p["id"]] == 0
            
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 1]) == 1
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 2]) >= 3
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 3]) >= 2
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 4]) >= 1
    
    sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters, bench = [], []
    cap = None
    
    for p in squad_players:
        if s_vars[p["id"]].varValue and s_vars[p["id"]].varValue > 0.5:
            starters.
