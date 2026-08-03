"""
fpl_bot.py — Primary Execution Script & Discord Interface
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import requests
import pulp
import math
from google import genai
from google.genai import types
from ddgs import DDGS

from fpl_funcs import (
    estimate_xmins,
    calculate_tier1_translation_factor,
    get_gameweek_state,
    get_live_price_deltas,
    get_base_ev,
    get_macro_ev,
    get_ensemble_ev,
    normalize_player,
    evaluate_chip_thresholds
)
from fpl_odds_engine import get_market_adjustments
from fpl_mpo_engine import solve_multi_period_model

# 1. Environment & Pre-Flight Check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")
XMINS_INPUT = os.environ.get("XMINS_INPUT", "")
ACTIVE_CHIP = os.environ.get("ACTIVE_CHIP", "NONE").upper()
RISK_POSTURE = os.environ.get("RISK_POSTURE", "NEUTRAL")
STATE_FILE_PATH = "fpl_state.json"

UEFA_TEAMS = {"MCI", "ARS", "LIV", "AVL", "MUN", "NEW", "CHE", "TOT", "SUN"}

if not all([GEMINI_API_KEY, DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing GitHub Secrets.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine. Your purpose is to evaluate squad selections, transfers, captaincy choices, and chip strategies.

### CORE DATA INTEGRITY GUARDRAILS
1. STRICT DATA IMMUTABILITY: You must copy player names, team tags, positions, costs, xMins, and EV values EXACTLY as presented in the MATHEMATICALLY LOCKED SQUAD payload.
2. ACTIVE MARKET ISOLATION: Base all qualitative commentary exclusively on players active within the provided Premier League database.

### CORE ANALYTICAL LAWS & METRIC DEFINITIONS
1. GAME-STATE NORMALISED ATTACKING & BAYESIAN SHRINKAGE
- Expected goals (xG/xAG) for low-minute fringe players are mathematically shrunken toward positional averages.
2. DEFENSIVE CONTRIBUTION (DefCon) & BPS MATHEMATICS
- Target baseline assets averaging >8.5 CBIT for defenders, and >10.5 CBIRT for midfielders/forwards.
3. CHIP STRUCTURE & MULTI-PERIOD ECONOMICS
- Forbidden from recommending a point hit (-4) unless the 8-Gameweek Expected Value (EV) of the incoming player exceeds the outgoing player by >5.5 points.

### OUTPUT FORMAT & AESTHETIC DIRECTIVES
You MUST format your analysis with extreme visual precision for Discord rendering. 
- Ensure there is a clean, empty line between paragraphs.
- Use `---` on its own line to divide sections.
- NEVER use LaTeX syntax or emojis.

SECTION 1: EXECUTIVE SUMMARY & CORE MOVES
- Provide a concise strategic summary.
- Generate a visual ASCII representation of the STARTING XI and BENCH inside a `text` codeblock. Center the player names symmetrically. YOU MUST FORMAT IT EXACTLY LIKE THIS TEMPLATE:
================================================================================
                    MATHEMATICALLY LOCKED SQUAD (X-X-X)
================================================================================
                                [Raya (4.03)]

     [Calafiori (4.59)]  [Gabriel (4.61)]  [Dalot (3.34)]  [O'Reilly (4.24)]

  [Estêvão (3.86)] [Maddison (4.28)] [Enzo (3.93)] [Cherki (4.39)] [Ngumoha (3.58)]

                               [Haaland (5.63)]
================================================================================
BENCH: [GK] Kinsky (3.38) | [S1] Zirkzee (2.95) | [S2] Beto (3.07) | [S3] Cash (3.01)
================================================================================

SECTION 2: QUANTITATIVE TRADE-OFF SUMMARY
- Format strictly as clean bullet points:
• Player Name (POS | TEAM | £X.Xm) — XX.X xMins | X.XX EV | [One-sentence justification]

SECTION 3: TRANSFER ECONOMICS & CHIP STATUS
- Detail capital management, rolling transfers, and MPO roadmap clearly.

SECTION 4: REAL-WORLD TACTICAL EXPLOIT & MATCHUP ANALYSIS
- STRICT REAL-WORLD ISOLATION: FPL players play for different clubs. Do NOT describe them passing to each other unless they play for the exact same real-world team.
- Generate an aligned ASCII Tactical Exploit Diagram inside a `text` codeblock. You MUST include at least 3 distinct club blocks representing key starting assets (e.g., ARSENAL, MAN CITY, CHELSEA/LIV) inside the diagram box. Do not output a single-club box. Format it EXACTLY like this example structure:
+-------------------------------------------------------------------------------+
|                      REAL-WORLD TACTICAL EXPLOIT MATRIX                       |
+-------------------------------------------------------------------------------+
| ARSENAL TACTICAL BLOCK:                                                       |
| [Gabriel] <---> [Calafiori] (Set-Piece Dominance & High Defensive Line)       |
|                                                                               |
| MANCHESTER CITY TACTICAL BLOCK:                                               |
| [Cherki] ---> [Haaland] (Central Half-Space Overload)                         |
|                                                                               |
| [THIRD CLUB] TACTICAL BLOCK:                                                  |
| [Player A] <---> [Player B] (Specific Tactical Exploit)                       |
+-------------------------------------------------------------------------------+

SECTION 5: HUMAN ORACLE INTELLIGENCE BRIEFING
- STRICT NULL-STATE RULE: Unless the LIVE ITK NEWS section contains specific, direct injury or press conference quotes for players in our payload, you are STRICTLY FORBIDDEN from writing generic summaries or commentary. Output EXACTLY and ONLY: 'Status: Awaiting live press conference data.' and nothing else.

MANDATORY SIGN-OFF: FINAL LOCKED-IN SQUAD SUMMARY
- You MUST conclude by pasting the EXACT pre-formatted text block provided at the bottom of the payload, verbatim, without altering its alignment.
"""

def load_state():
    default_state = {
        "buyback_targets": {}, "last_updated_gw": 0, "xmins_overrides": {}, 
        "price_watchlist": {},
        "calibration_weights": {"xgi_weight": 0.70, "fdr_impact_factor": 0.10, "bench_discount": 0.01},
        "pending_evaluation": None, "performance_history": []
    }
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                saved_state = json.load(f)
                for key, val in default_state.items():
                    if key not in saved_state: saved_state[key] = val
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
    if not pending: return state

    eval_gw = pending.get("gw")
    if active_gw <= eval_gw: return state

    try:
        resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{eval_gw}/picks/", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            actual_points = data.get("entry_history", {}).get("points", 0)
            projected_xP = pending.get("projected_xP", 0.0)
            residual_error = actual_points - projected_xP

            state["performance_history"].append({
                "gw": eval_gw, "projected_xP": round(projected_xP, 2),
                "actual_points": actual_points, "residual_error": round(residual_error, 2),
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

def get_live_fpl_news():
    news_text = "### LIVE ITK NEWS & SCHEDULE DATA (Automatically Fetched)\n"
    try:
        crellin_results = DDGS().text("Ben Crellin FPL blank double gameweek updates", max_results=3, timelimit='w')
        news_text += "--- SCHEDULE CHANGES (Ben Crellin) ---\n"
        for r in crellin_results: news_text += f"- {r.get('body', '')}\n"
            
        dinnery_results = DDGS().text("Ben Dinnery FPL injuries team news press conference", max_results=3, timelimit='w')
        news_text += "\n--- INJURY UPDATES (Ben Dinnery) ---\n"
        for r in dinnery_results: news_text += f"- {r.get('body', '')}\n"

        press_results = DDGS().news("Premier League manager press conference injury updates today", max_results=3, timelimit='d')
        news_text += "\n--- PRESS CONFERENCE UPDATES ---\n"
        for r in press_results: news_text += f"- {r.get('body', '')}\n"

        leak_results = DDGS().text("FPL late team news leaks traveling squad omitted", max_results=3, timelimit='d')
        news_text += "\n--- SQUAD OMISSIONS & LEAKS ---\n"
        for r in leak_results: news_text += f"- {r.get('body', '')}\n"
            
    except Exception as e:
        news_text += f"[Search tool failed to retrieve live data: {e}]\n"
    return news_text

def check_european_congestion_flags(starters, fixtures_data, target_gw):
    flags = []
    for p in starters:
        if p["team"] in UEFA_TEAMS:
            flags.append(f"[FLAG OPTION: European Turnaround Risk detected for {p['name']} ({p['team']}) due to mid-week fixture congestion.]")
    return flags

def get_fpl_data():
    print("Fetching live market odds for tactical alignment...")
    market_data = get_market_adjustments()
    
    headers = {"User-Agent": "FPL-Auto-Script/14.0"}
    state = load_state()
    xmins_overrides = state.get("xmins_overrides", {})
    
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if bootstrap_resp.status_code != 200: sys.exit(1)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}"); sys.exit(1)

    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}
    
    state = recalibrate_model(state, headers, active_gw)
    state["last_updated_gw"] = target_gw
    weights = state["calibration_weights"]

    players = {}
    new_arrivals = []
    for raw_p in bootstrap_data.get("elements", []):
        p = normalize_player(raw_p, teams, element_types)
        players[p["id"]] = p
        if p.get("has_stale_pl_history") or p.get("total_points", 0) == 0:
            new_arrivals.append(f"- {p['name']} ({p['team']})")

    new_arrivals_str = "\n".join(new_arrivals) if new_arrivals else "None"
    
    price_deltas = get_live_price_deltas(players)
    for pid, p in players.items():
        p["price_delta_prob"] = price_deltas.get(pid, 0.0)
    market_str = "Market data & live price deltas initialized."

    if XMINS_INPUT:
        print(f"Processing Human Oracle Input: {XMINS_INPUT}")
        for override in XMINS_INPUT.split(","):
            if ":" in override:
                name_part, min_part = override.split(":")
                name_part = name_part.strip().lower()
                try:
                    target_mins = float(min_part.strip())
                    for pid, p in players.items():
                        if name_part in p["name"].lower():
                            xmins_overrides[str(pid)] = target_mins
                            print(f"   -> ORACLE OVERRIDE: {p['name']} locked to {target_mins} mins.")
                except ValueError:
                    print(f"   -> WARNING: Could not parse override '{override}'")
    
    current_squad_ids = []
    bank = 0.0
    total_liquid_budget = 100.0
    free_transfers = "Unlimited" if target_gw == 1 else 1

    if target_gw > 1:
        try:
            team_resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{target_gw-1}/picks/", headers=headers)
            if team_resp.status_code == 200:
                team_data = team_resp.json()
                current_squad_ids = [pick["element"] for pick in team_data.get("picks", [])]
                entry_history = team_data.get("entry_history", {})
                bank = entry_history.get("bank", 0) / 10.0
                total_liquid_budget = entry_history.get("value", 1000) / 10.0
        except Exception as e:
            print(f"WARNING: Could not fetch squad for team {FPL_TEAM_ID}: {e}")

    planned_chips_env = os.environ.get("PLANNED_CHIPS", "")
    planned_chips = {}
    if planned_chips_env:
        for pair in planned_chips_env.split(","):
            parts = pair.split(":")
            if len(parts) == 2:
                try:
                    gw_num = int(parts[0])
                    chip_name = parts[1].strip().upper()
                    t_idx = gw_num - target_gw
                    if 0 < t_idx < 8: planned_chips[t_idx] = chip_name
                except ValueError: pass

    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        fixtures_data = fixtures_resp.json() if fixtures_resp.status_code == 200 else []
    except Exception:
        fixtures_data = []
        
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 8:
            team_a, team_h = f.get("team_a"), f.get("team_h")
            if team_a in team_fdr_sum:
                team_fdr_sum[team_a] += f.get("team_a_difficulty", 3)
                team_fdr_count[team_a] += 1
            if team_h in team_fdr_sum:
                team_fdr_sum[team_h] += f.get("team_h_difficulty", 3)
                team_fdr_count[team_h] += 1
                
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    team_gw_fdr = {t_id: [6.0] * 8 for t_id in teams.keys()}
    team_gw_fixtures = {t_id: [0.0] * 8 for t_id in teams.keys()}
    
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 8:
            t = event - target_gw
            team_a, team_h = f.get("team_a"), f.get("team_h")
            if team_a in team_gw_fdr:
                team_gw_fdr[team_a][t] = f.get("team_a_difficulty", 3)
                team_gw_fixtures[team_a][t] += 1.0
            if team_h in team_gw_fdr:
                team_gw_fdr[team_h][t] = f.get("team_h_difficulty", 3)
                team_gw_fixtures[team_h][t] += 1.0

    # LOAD MACHINE LEARNING PROJECTIONS FROM ML PIPELINE
    ml_projections = {}
    if os.path.exists("ml_projections.json"):
        try:
            with open("ml_projections.json", "r") as f:
                ml_projections = json.load(f)
            print("ML ENGINE: Successfully loaded ml_projections.json into decision pipeline.")
        except Exception as e:
            print(f"WARNING: Could not read ml_projections.json: {e}")

    ev_matrix = {}
    valid_ids = list(players.keys())
    for pid in valid_ids:
        p = players[pid]
        ev_matrix[pid] = [0.0] * 8
        if p.get("status") not in ["a", "d", ""]: continue
            
        t_id = p["team_id"]
        pid_str = str(pid)

        # Calculate baseline heuristic & market EV
        heuristic_ev = get_ensemble_ev(p, xmins_overrides, market_data, weights, RISK_POSTURE)

        # Inject XGBoost ML EV if available (70% ML / 30% Market & Heuristic blend)
        if pid_str in ml_projections and ml_projections[pid_str].get("ml_ev_1gw", 0) > 0:
            ml_ev = float(ml_projections[pid_str]["ml_ev_1gw"])
            ev_matrix[pid][0] = round((0.70 * ml_ev) + (0.30 * heuristic_ev), 2)
        else:
            ev_matrix[pid][0] = heuristic_ev

        base_ev = ev_matrix[pid][0]
        
        pos_id = p.get("pos_id", 3)
        sigma = base_ev * (0.45 if pos_id == 3 else (0.4 if pos_id == 4 else 0.3))
        if RISK_POSTURE == "SHIELD": base_ev -= (sigma * 0.15)
        elif RISK_POSTURE == "CHASE": base_ev += (sigma * 0.15)
            
        for t in range(1, 8):
            fdr = team_gw_fdr[t_id][t]
            fix_count = team_gw_fixtures[t_id][t]
            fdr_multiplier = 1.0 + (3.0 - fdr) * 0.10 if fdr != 6.0 else 0.0
            ev_matrix[pid][t] = max(0.0, base_ev * fdr_multiplier * fix_count)

    starter_candidates = sorted([p for p in valid_ids if players[p].get("cost", 0) >= 5.0], 
                                key=lambda x: ev_matrix[x][0], reverse=True)[:11]
    
    p_zero_mins = []
    for pid in starter_candidates:
        p = players[pid]
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        p_z = 1.0 / (1.0 + math.exp(0.1 * (xmins - 35.0)))
        p_zero_mins.append(p_z)
        
    p_0 = 1.0
    for p_z in p_zero_mins: p_0 *= (1.0 - p_z)
    
    p_1 = 0.0
    for i, p_z in enumerate(p_zero_mins):
        term = p_z
        for j, p_other in enumerate(p_zero_mins):
            if i != j: term *= (1.0 - p_other)
        p_1 += term
        
    dynamic_w_sub_1 = round(max(0.04, min(0.30, 1.0 - p_0)), 3)
    dynamic_w_sub_2 = round(max(0.01, min(0.15, 1.0 - p_0 - p_1)), 3)

    optimal_squad, transfer_plan = solve_multi_period_model(
        players, ev_matrix, current_squad_ids, total_liquid_budget, 
        free_transfers, active_chip=ACTIVE_CHIP, horizons=8, risk_posture=RISK_POSTURE, target_gw=target_gw,
        w_sub_1=dynamic_w_sub_1, w_sub_2=dynamic_w_sub_2, planned_chips=planned_chips
    )

    if not optimal_squad or len(optimal_squad) < 15:
        sorted_all = sorted([p for p in players.values() if p.get("status") in ["a", "d", ""]], 
                            key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        optimal_squad = sorted_all[:15]

    sub_prob = pulp.LpProblem("Phase2_StartingXI", pulp.LpMaximize)
    s_vars = pulp.LpVariable.dicts("s", [p["id"] for p in optimal_squad], cat="Binary")
    c_vars = pulp.LpVariable.dicts("c", [p["id"] for p in optimal_squad], cat="Binary")
    
    cap_mult = 2.0 if ACTIVE_CHIP == "TRIPLE_CAPTAIN" else 1.0
    sub_objective = []
    for p in optimal_squad:
        ev_1gw = ev_matrix[p["id"]][0]
        ownership = p.get("own", 0.0) / 100.0
        rank_gravity = (ev_1gw * (ownership ** 2) * 1.50) if RISK_POSTURE == "SHIELD" else (
                       -(ev_1gw * ownership * 0.50) if RISK_POSTURE == "CHASE" else 
                       (ev_1gw * (ownership ** 2) * 0.75))
        sub_objective.append((ev_1gw * s_vars[p["id"]]) + ((ev_1gw * cap_mult + rank_gravity) * c_vars[p["id"]]))
        
    sub_prob += pulp.lpSum(sub_objective)
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad]) == 11
    sub_prob += pulp.lpSum([c_vars[p["id"]] for p in optimal_squad]) == 1
    for p in optimal_squad: sub_prob += c_vars[p["id"]] <= s_vars[p["id"]]
            
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 1]) == 1
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 2]) >= 3
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 3]) >= 3
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 4]) >= 1
    
    sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters, bench = [], []
    cap = None
    for p in optimal_squad:
        if s_vars[p["id"]].varValue and s_vars[p["id"]].varValue > 0.5:
            starters.append(p)
            if c_vars[p["id"]].varValue and c_vars[p["id"]].varValue > 0.5: cap = p
        else: bench.append(p)
            
    starters.sort(key=lambda x: x["pos_id"])
    
    starters_sorted_by_1gw = sorted(starters, key=lambda x: ev_matrix[x["id"]][0], reverse=True)
    vice = next((p for p in starters_sorted_by_1gw if not cap or (p["id"] != cap["id"] and p["team_id"] != cap["team_id"])), starters_sorted_by_1gw[0])
    
    optimal_squad = starters + bench

    european_flags = check_european_congestion_flags(starters, fixtures_data, target_gw)

    projected_starting_xP = sum(ev_matrix[p["id"]][0] for p in starters)
    if cap:
        cap_mult = 2.0 if ACTIVE_CHIP == "TRIPLE_CAPTAIN" else 1.0
        projected_starting_xP += (ev_matrix[cap["id"]][0] * (cap_mult - 1.0))
        
    state["pending_evaluation"] = {
        "gw": target_gw, "projected_xP": round(projected_starting_xP, 2), "captain": cap["name"] if cap else None
    }
    save_state(state)
    
    def count_pos(group, pos_id): return len([p for p in group if p["pos_id"] == pos_id])
    formation = f"{count_pos(starters, 2)}-{count_pos(starters, 3)}-{count_pos(starters, 4)}"

    squad_lines = [
        "```text",
        "================================================================================",
        "                       FINAL LOCKED-IN SQUAD SUMMARY",
        "================================================================================",
        f"GW{target_gw} Formation: {formation:<5} | Liquid Value: £{total_liquid_budget:<5.1f}m | Bank: £{bank:<4.1f}m",
        f"Captain (C): {cap['name'] if cap else 'N/A'} ({cap['team'] if cap else ''}) | Vice-Captain (VC): {vice['name'] if vice else 'N/A'} ({vice['team'] if vice else ''})",
        "",
        "STARTING XI:"
    ]

    for p in starters:
        is_cap = " (C)" if cap and p["id"] == cap["id"] else ""
        is_vice = " (VC)" if vice and p["id"] == vice["id"] else ""
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        actual_ev = round(ev_matrix[p["id"]][0], 2)
        pos_str = f"{p['pos']}:"
        name_team = f"{p['name']} ({p['team']})"
        squad_lines.append(f"{pos_str:<4} {name_team:<22} | £{p['cost']:>4.1f}m | {xmins:>4.1f} xMins | {actual_ev:>4.2f} EV{is_cap}{is_vice}")
        
    squad_lines.extend(["", "BENCH ORDER:"])
    bench_gk = [p for p in bench if p["pos_id"] == 1]
    bench_outfield = sorted([p for p in bench if p["pos_id"] != 1], key=lambda x: ev_matrix[x["id"]][0], reverse=True)
    
    for i, p in enumerate(bench_gk + bench_outfield):
        prefix = "GK:" if p["pos_id"] == 1 else f"S{i}:"
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        actual_ev = round(ev_matrix[p["id"]][0], 2)
        name_team = f"{p['name']} ({p['team']})"
        squad_lines.append(f"{prefix:<4} {name_team:<22} | £{p['cost']:>4.1f}m | {xmins:>4.1f} xMins | {actual_ev:>4.2f} EV")
        
    squad_lines.extend([
        "",
        "================================================================================",
        "                         MULTI-PERIOD TRANSFER TREE",
        "================================================================================"
    ])
    
    if transfer_plan:
        for tp in transfer_plan: squad_lines.append(f"• {tp}")
    else:
        squad_lines.append("• GW1: Hold / Bank Transfer (Unlimited pre-season allocation locked)")
        
    chip_recommendations = evaluate_chip_thresholds(starters, bench, ev_matrix, ACTIVE_CHIP)
    squad_lines.extend([
        "================================================================================",
        "                    ALGORITHMIC CHIP RECOMMENDATIONS",
        "================================================================================"
    ])
    for rec in chip_recommendations: squad_lines.append(f"• {rec}")

    macro_squad_8gw_xp = sum(get_macro_ev(p, team_avg_fdr, weights, xmins_overrides, market_data, 8, RISK_POSTURE) for p in optimal_squad)
    
    from fpl_monte_carlo import run_monte_carlo_simulations
    mc_players = {p["id"]: {"est_xmins": estimate_xmins(p), "xgi_90": p["xgi_90"], "pos_id": p["pos_id"], "ep_next": p["ep_next"]} for p in starters}
    mc_results = run_monte_carlo_simulations(mc_players, num_trials=1000)
    
    starter_floor = sum(res["floor"] for res in mc_results.values())
    starter_ceiling = sum(res["ceiling"] for res in mc_results.values())

    squad_lines.extend([
        "================================================================================",
        "                    SQUAD HEALTH & VARIANCE REPORT",
        "================================================================================",
        f"• 1-GW Expected Yield (Odds-Adjusted): {projected_starting_xP:>6.2f} pts",
        f"• 8-GW Deep-Tree Horizon xP:           {macro_squad_8gw_xp:>6.2f} pts",
        f"• Stochastic 10th Percentile Floor:    {starter_floor:>6.1f} pts",
        f"• Stochastic 90th Percentile Ceiling:  {starter_ceiling:>6.1f} pts",
        "================================================================================",
        "```"
    ])

    locked_squad_str = "\n".join(squad_lines)

    return target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str

def build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news):
    gw1_override = "\n    6. PRE-SEASON RULE OVERRIDE: GW1 has UNLIMITED free transfers." if (target_gw == 1 or str(free_transfers).lower() == "unlimited") else ""

    if WORKFLOW_INPUT == "post_gameweek_review":
        action_type = "Post-Gameweek Strategic Review & Market Volatility Audit"
        phase_instructions = (
            "- FOCUS: Backward-looking performance evaluation, market equity, and medium-term planning.\n"
            "- STRICT PHASE ISOLATION: Do NOT generate a pre-match pitch mismatch diagram or starting XI tactical justifications.\n"
            "- Detail the MULTI-PERIOD TRANSFER TREE (MPO) roadmap for GW+1 through GW+7.\n"
            "- GW1 ZERO-STATE RULE: If target_gw is 1, acknowledge actual recalibration data is pending."
        )
    elif WORKFLOW_INPUT == "pre_gameweek_deadline":
        action_type = "Pre-Gameweek Final Deadline Lock & Late ITK Leak Audit"
        phase_instructions = (
            "- FOCUS: Forward-looking immediate execution for the upcoming deadline.\n"
            "- Confirm any Human Oracle xMins overrides applied and lock Starting XI, Captain (C), Vice-Captain (VC)."
        )
    else:
        action_type = "Full Weekly Execution & Analytical Breakdown"
        phase_instructions = "- Balanced breakdown covering transfer economics, market volatility, and upcoming fixture geometry."

    focus_instructions = (
        f"1. 11-Man Verification Lock: Output exact mathematically locked Starting XI and Bench.\n"
        f"2. Phase-Specific Focus ({action_type}):\n{phase_instructions}\n"
        f"3. Analytical Justification: Use EXACT 'TRUE 1-GW EV' numbers provided.\n"
        f"4. MANDATORY SIGN-OFF: Conclude response with boxed 'FINAL LOCKED-IN SQUAD SUMMARY' block exactly as provided."
    )

    return f"""
    Run {action_type} for Gameweek {target_gw}.
    ### CURRENT SQUAD STATE & ECONOMICS: Bank: £{bank}m | Saved Transfers: {free_transfers}
    ### NEW ARRIVALS & FOREIGN TRANSFERS:\n{new_arrivals_str}
    ### ACTIVE 2026/27 MARKET WATCHLIST:\n{market_str}\n{live_news}
    ### DATA INSTRUCTIONS:\n{focus_instructions}{gw1_override}

    ### MATHEMATICALLY LOCKED SQUAD PAYLOAD (INSERT VERBATIM AT END OF RESPONSE):
    \n{locked_squad_str}\n
    """

def send_to_discord(webhook_url, text):
    chunks, current_chunk = [], ""
    for line in text.split("\n"):
        while len(line) > 1800:
            if len(current_chunk) > 0: chunks.append(current_chunk); current_chunk = ""
            chunks.append(line[:1800]); line = line[1800:]
        if len(current_chunk) + len(line) + 1 > 1800:
            chunks.append(current_chunk); current_chunk = line + "\n"
        else: current_chunk += line + "\n"
    if current_chunk.strip(): chunks.append(current_chunk)
    for chunk in chunks: requests.post(webhook_url, json={"content": chunk})

def main():
    target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str = get_fpl_data()
    print("--- FETCHING LIVE WEB SEARCH DATA ---")
    live_news = get_live_fpl_news()
    prompt = build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news)
    
    print(f"--- QUERYING GEMINI API (Target GW: {target_gw} | Chip: {ACTIVE_CHIP}) ---")
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash', contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2)
        )
    except Exception as e:
        print(f"CRITICAL ERROR generating content with Gemini: {str(e)}")
        sys.exit(1)
        
    content = response.text if response and response.text else ""
    if not content: sys.exit(1)

    print(f"--- GEMINI RESPONSE RECEIVED ({len(content)} chars) ---")
    send_to_discord(DISCORD_WEBHOOK_URL, content)
    print("--- DISCORD DELIVERY COMPLETE ---")

if __name__ == "__main__":
    main()