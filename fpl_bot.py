import os
import sys
import json
import requests
import pulp
import math
from google import genai
from google.genai import types
from ddgs import DDGS
# combined functions
from fpl_funcs import (
    estimate_xmins,
    calculate_tier1_translation_factor,
    get_gameweek_state,
    get_base_ev,
    get_macro_ev,
    normalize_player
)

# 1. Environment & Pre-Flight Check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")

ACTIVE_CHIP = os.environ.get("ACTIVE_CHIP", "NONE").upper() 
STATE_FILE_PATH = "fpl_state.json"

UEFA_TEAMS = {"MCI", "ARS", "LIV", "AVL", "MUN", "NEW", "CHE", "TOT", "SUN"}

if not all([GEMINI_API_KEY, DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing GitHub Secrets (GEMINI_API_KEY, DISCORD_WEBHOOK_URL, or FPL_TEAM_ID).")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine. Your purpose is to evaluate squad selections, transfers, captaincy choices, and chip strategies using advanced underlying metrics, pitch geometry, game-state data, and FPL market economics. You must strictly adhere to the following analytical laws.

### CORE DATA INTEGRITY GUARDRAILS
1. STRICT DATA IMMUTABILITY: You must copy player names, team tags, positions, costs, xMins, and EV values EXACTLY as presented in the MATHEMATICALLY LOCKED SQUAD payload. Do NOT alter, overwrite, or assume club affiliations or positions from past historical memory.
2. ACTIVE MARKET ISOLATION: Base all qualitative commentary exclusively on players active within the provided Premier League database. Ignore mentions of non-Premier League players that may appear in external ITK or search feeds.

### CORE ANALYTICAL LAWS & METRIC DEFINITIONS

1. GAME-STATE NORMALISED ATTACKING & BAYESIAN SHRINKAGE
- Bayesian Filter: Expected goals (xG/xAG) for low-minute fringe players are mathematically shrunken toward positional averages to eliminate small-sample-size data explosions.
- Continuous Market Scalar: The system applies a fluid multiplier to xG based on player cost, reflecting the market's inherent pricing of elite finishing ability.
- Accidental Assist Rule: FPL awards assists for deflected passes if there is only one defensive touch inside the box. Prioritize "Raw Crosses into the Penalty Box" alongside xAG.

2. DEFENSIVE CONTRIBUTION (DefCon) & 2026/27 BPS MATHEMATICS
- DefCon Thresholds: Target baseline assets averaging >8.5 CBIT (Clearances, Blocks, Interceptions, Tackles) for defenders, and >10.5 CBIRT (+ Recoveries) for midfielders/forwards.
- BPS Adjustments: Centre-backs earn 1 BPS per 3 CBI actions. Dribblers face no -1 penalty for being tackled. Penalty goals are a flat 12 BPS for all positions.
- Poisson Clean Sheet Probability: Goalkeeper and Defender EV is mathematically derived from an exponential Poisson decay of their team's Expected Goals Against (xGA), strictly gated by 60-minute threshold probability.

3. CHIP STRUCTURE & 5-TRANSFER ECONOMICS
- 5-Transfer Banking: Evaluate 0-transfer rolls as an appreciating asset building toward a 3-to-5 transfer "mini-wildcard".
- Point Hit Constraint: Forbidden from recommending a point hit (-4) unless the 4-Gameweek Expected Value (EV) of the incoming player exceeds the outgoing player by >5.5 points, OR if required to field 11 starters.

4. QUALITATIVE OVERRIDES & TIER-1 LEAK VERIFICATION
- 15-Minute Panic Rule: Late leaks affecting >2 decisions within 15 mins of deadline default to the original multi-week EV plan.
- Cup Congestion Law: Apply a 25% xMins penalty to starters with <72 hours turnaround from cup matches.

5. FOREIGN TRANSFERS & TIERED LEAGUE TRANSLATION
- Tiered Translation Coefficients: Apply granular scaling multipliers to expected attacking output (npxG/xAG) based on competition source tiers (e.g., Champions League at 0.95, Pro League/Bundesliga tiers ranging from 0.76 to 0.88) rather than a flat penalty.
- Returning Player Exception Logic: Assets with stale early Premier League history but recent elite European peaks bypass blunt rookie discounts via intelligent exception overrides.
- xMins Integration Cap: Cap initial xMins for new arrivals at 45-60 mins for their first 3 Gameweeks until role security stabilizes.

### OUTPUT FORMAT & AESTHETIC DIRECTIVES
You MUST format your analysis with extreme visual precision for Discord rendering.

SECTION 1: EXECUTIVE SUMMARY & CORE MOVES
- Strategic summary of squad state, economics, and core moves.
- Include a monospaced ASCII Pitch Map block inside a markdown text codeblock displaying the starting formation and bench hierarchy.
- Detail the Foreign Transfer / Law 7 Audit and Tactical Mitigation Flags.

SECTION 2: QUANTITATIVE TRADE-OFF SUMMARY
- Do NOT use wide Markdown tables (they break on mobile/Discord).
- Format strictly as compact, high-density, single-line bullet points per player:
  • **[Player Name]** ([POS] | [TEAM] | £[X.X]m) — **[xMins] xMins** | **[EV] EV** | [DefCon / xGA / Metric Status]

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
Formation: [X-X-X] | Bank: £[X.X]m | Free Transfers: [X]

STARTING XI:
  GKP: [Player] (£[X.X]m)
  DEF: [Player] (£[X.X]m)
  ...

BENCH RESERVES:
  Slot 1: [Player] (£[X.X]m, [POS])
  Slot 2: [Player] (£[X.X]m, [POS])
  Slot 3: [Player] (£[X.X]m, [POS])
  Slot 4: [Player] (£[X.X]m, [POS])
================================================================================

CRITICAL FORMATTING RULE: Strictly AVOID LaTeX syntax (e.g. $, $$, \text{}, \times). Use plain text and standard Markdown formatting exclusively.
"""

# 2. State Persistence & Calibration Engine
def load_state():
    default_state = {
        "buyback_targets": {},
        "last_updated_gw": 0,
        "xmins_overrides": {}, 
        "price_watchlist": {},
        "calibration_weights": {
            "xgi_weight": 0.70,
            "fdr_impact_factor": 0.10,
            "bench_discount": 0.01
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

    try:
        resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{eval_gw}/picks/", headers=headers)
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

def check_european_congestion_flags(starters, fixtures_data, target_gw):
    flags = []
    for p in starters:
        if p["team"] in UEFA_TEAMS:
            flags.append(f"[FLAG OPTION: European Turnaround Risk detected for {p['name']} ({p['team']}) due to mid-week fixture congestion.]")
    return flags

# 5. Execution Engine: Portfolio Optimization MILP Solver
def solve_fpl_knapsack(players_dict, current_squad_ids, total_budget, free_transfers, team_avg_fdr, required_bank_reservation, weights, xmins_overrides, active_chip):
    prob = pulp.LpProblem("FPL_Portfolio_Optimization", pulp.LpMaximize)
    valid_ids = list(players_dict.keys())
    
    # Aggressive override to prevent budget bleeding onto the bench
    if active_chip == "BENCH_BOOST":
        bench_discount = 1.0
    else:
        bench_discount = 0.01 
        
    captain_multiplier = 2.0 if active_chip == "TRIPLE_CAPTAIN" else 1.0
        
    squad_vars = pulp.LpVariable.dicts("squad", valid_ids, cat="Binary")
    starter_vars = pulp.LpVariable.dicts("starter", valid_ids, cat="Binary")
    captain_vars = pulp.LpVariable.dicts("captain", valid_ids, cat="Binary")
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat="Continuous")
    
    objective = []
    for i in valid_ids:
        p = players_dict[i]
        ev = get_macro_ev(p, team_avg_fdr, weights, xmins_overrides)
        
        try: ownership = float(p.get("own", 0.0))
        except: ownership = 0.0
        
        own_pct = ownership / 100.0
        rank_threat_gravity = (ev * (own_pct ** 2) * 0.75)
            
        own_tiebreaker = ownership * 0.0001
        
        objective.append(
            (ev * starter_vars[i]) + 
            ((ev * captain_multiplier + rank_threat_gravity) * captain_vars[i]) + 
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

    prob += pulp.lpSum([squad_vars[i] for i in valid_ids]) == 15
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids]) == 11
    prob += pulp.lpSum([captain_vars[i] for i in valid_ids]) == 1
    
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
    
    # STRICT LEGAL FORMATION CONSTRAINTS (Min 3 DEF, Min 3 MID, Min 1 FWD)
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) <= 5
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) >= 3
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) <= 5
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) >= 1
    prob += pulp.lpSum([starter_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) <= 3

    team_ids = set(players_dict[i]["team_id"] for i in valid_ids)
    for t_id in team_ids:
        prob += pulp.lpSum([squad_vars[i] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

    effective_budget = total_budget - required_bank_reservation
    prob += pulp.lpSum([players_dict[i]["cost"] * squad_vars[i] for i in valid_ids]) <= effective_budget

    if active_chip not in ["FREE_HIT", "WILDCARD"] and str(free_transfers).lower() != "unlimited":
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
            
    sub_prob = pulp.LpProblem("Phase2_StartingXI", pulp.LpMaximize)
    s_vars = pulp.LpVariable.dicts("s", [p["id"] for p in squad_players], cat="Binary")
    c_vars = pulp.LpVariable.dicts("c", [p["id"] for p in squad_players], cat="Binary")
    
    sub_objective = []
    for p in squad_players:
        ev_1gw = get_base_ev(p, weights, xmins_overrides) 
        try: ownership = float(p.get("own", 0.0))
        except: ownership = 0.0
        
        own_pct = ownership / 100.0
        rank_threat_gravity = (ev_1gw * (own_pct ** 2) * 0.75)
            
        sub_objective.append( (ev_1gw * s_vars[p["id"]]) + ((ev_1gw * captain_multiplier + rank_threat_gravity) * c_vars[p["id"]]) )
        
    sub_prob += pulp.lpSum(sub_objective)
    
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players]) == 11
    sub_prob += pulp.lpSum([c_vars[p["id"]] for p in squad_players]) == 1
    
    for p in squad_players:
        sub_prob += c_vars[p["id"]] <= s_vars[p["id"]]
            
    # STRICT LEGAL FORMATION CONSTRAINTS (Phase 2 Sub-Solver)
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 1]) == 1
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 2]) >= 3
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 3]) >= 3
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in squad_players if p["pos_id"] == 4]) >= 1
    
    sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters, bench = [], []
    cap = None
    
    for p in squad_players:
        if s_vars[p["id"]].varValue and s_vars[p["id"]].varValue > 0.5:
            starters.append(p)
            if c_vars[p["id"]].varValue and c_vars[p["id"]].varValue > 0.5:
                cap = p
        else:
            bench.append(p)
            
    starters.sort(key=lambda x: x["pos_id"])
    
    bench_gk = [p for p in bench if p["pos_id"] == 1]
    bench_outfield = sorted([p for p in bench if p["pos_id"] != 1], key=lambda x: get_base_ev(x, weights, xmins_overrides), reverse=True)
    sorted_bench = bench_gk + bench_outfield
    
    starters_sorted_by_1gw = sorted(starters, key=lambda x: get_base_ev(x, weights, xmins_overrides), reverse=True)
    vice = next((p for p in starters_sorted_by_1gw if not cap or p["id"] != cap["id"]), starters_sorted_by_1gw[0])
            
    return starters, sorted_bench, cap, vice

# 6. Main Data Pipeline & Integration
def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/13.0"}
    state = load_state()
    xmins_overrides = state.get("xmins_overrides", {})
    price_watchlist = state.get("price_watchlist", {})
    
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if bootstrap_resp.status_code != 200:
            print(f"CRITICAL ERROR: FPL API returned status {bootstrap_resp.status_code}. The game is likely updating.")
            sys.exit(1)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}")
        sys.exit(1)

    # Resolve gameweek targets
    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}
    
    state = recalibrate_model(state, headers, active_gw)
    state["last_updated_gw"] = target_gw
    weights = state["calibration_weights"]

    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        if fixtures_resp.status_code == 200:
            fixtures_data = fixtures_resp.json()
        else:
            fixtures_data = []
    except Exception as e:
        fixtures_data = []
        
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 4:
            team_a, team_h = f.get("team_a"), f.get("team_h")
            if team_a in team_fdr_sum:
                team_fdr_sum[team_a] += f.get("team_a_difficulty", 3)
                team_fdr_count[team_a] += 1
            if team_h in team_fdr_sum:
                team_fdr_sum[team_h] += f.get("team_h_difficulty", 3)
                team_fdr_count[team_h] += 1
                
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    players = {}
    for raw_p in bootstrap_data["elements"]:
        p_norm = normalize_player(raw_p, teams, element_types)
        players[p_norm["id"]] = p_norm
        
    market_list = []
    available_players = [p for p in players.values() if p.get("status") in ["a", "d"]]
    for pos_id in [1, 2, 3, 4]: 
        pos_players = [p for p in available_players if p["pos_id"] == pos_id]
        def safe_float_own(player):
            try: return float(player.get("own", 0.0))
            except: return 0.0
        top_pos = sorted(pos_players, key=safe_float_own, reverse=True)[:30]
        for p in top_pos:
            news_flag = f" | FLAG: {p['news']}" if p['news'] else ""
            market_list.append(f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, {p['own']}% owned, Status: {p['status']}{news_flag})")
    market_str = "\n".join(market_list)

    new_arrivals = []
    for p in players.values():
        is_new_transfer = (p["total_points"] == 0 and p["cost"] >= 5.5 and p["status"] in ["a", "d"])
        if is_new_transfer:
            new_arrivals.append(f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, {p['own']}% owned) | NOTE: Zero Baseline Data")
    new_arrivals_str = "\n".join(new_arrivals) if new_arrivals else "No recent high-profile foreign arrivals detected in API."
    
    squad_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{active_gw}/picks/"
    transfers_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/transfers/"
    
    current_squad_ids, transfer_ledger = [], []
    
    try:
        sq_resp = requests.get(squad_url, headers=headers)
        if sq_resp.status_code == 200:
            current_squad_ids = [pick["element"] for pick in sq_resp.json().get("picks", [])]
    except Exception as e:
        print(f"WARNING: Error fetching squad picks: {e}")

    try:
        tr_resp = requests.get(transfers_url, headers=headers)
        if tr_resp.status_code == 200:
            transfer_ledger = tr_resp.json()
    except Exception as e:
        print(f"WARNING: Error fetching transfer ledger: {e}")

    liquid_squad_value = 0.0
    for pid in current_squad_ids:
        p = players[pid]
        now_cost_raw = int(p["cost"] * 10)
        
        purchase_price_raw = None
        for t in reversed(transfer_ledger):
            if t["element_in"] == pid:
                purchase_price_raw = t["element_in_cost"]
                break
                
        if purchase_price_raw is None:
            purchase_price_raw = now_cost_raw - p["cost_change_start"]
            
        profit = now_cost_raw - purchase_price_raw
        selling_price_raw = purchase_price_raw + (profit // 2) if profit > 0 else now_cost_raw
            
        p["purchase_price"] = purchase_price_raw / 10.0
        p["selling_price"] = selling_price_raw / 10.0
        liquid_squad_value += p["selling_price"]
        
        pid_str = str(pid)
        if pid_str not in price_watchlist:
            price_watchlist[pid_str] = p["purchase_price"]

    bank = 0.0
    free_transfers = "Unlimited (Pre-Season GW1)" if target_gw == 1 else "1+"
    try:
        hist_resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/", headers=headers)
        if hist_resp.status_code == 200:
            h_data = hist_resp.json()
            if h_data.get("current"):
                bank = h_data["current"][-1].get("bank", 0) / 10.0
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")

    total_liquid_budget = liquid_squad_value + bank if current_squad_ids else 100.0

    required_bank_reservation = 0.0
    active_targets = state.get("buyback_targets", {})
    updated_targets = {}

    for pid_str, target_info in active_targets.items():
        if target_gw <= target_info["return_gw"]:
            updated_targets[pid_str] = target_info
            if int(pid_str) not in current_squad_ids:
                required_bank_reservation += target_info["reserved_bank"]

    for pid in current_squad_ids:
        p = players[pid]
        chance = str(p.get("chance_of_playing_next_round", ""))
        
        if chance in ["0", "25", "50"] and p["cost"] >= 9.0:
            trapped_equity = p["cost"] - p["selling_price"]
            
            if chance == "0": injury_weeks = 4
            elif chance == "25": injury_weeks = 2
            else: injury_weeks = 1
                
            remaining_season_gws = max(1, 38 - target_gw)
            short_term_replacement_gain = injury_weeks * 4.0
            long_term_equity_loss = trapped_equity * 0.25 * remaining_season_gws
            crossover_ev = short_term_replacement_gain - long_term_equity_loss

            if crossover_ev > 0:
                target_buyback_price = p["cost"]
                buyback_reserve = max(0.0, target_buyback_price - 7.5)
                
                updated_targets[str(pid)] = {
                    "name": p["name"], "sell_gw": target_gw, "return_gw": target_gw + injury_weeks,
                    "selling_price": p["selling_price"], "target_buyback_price": target_buyback_price,
                    "reserved_bank": buyback_reserve
                }
                required_bank_reservation += buyback_reserve

    state["buyback_targets"] = updated_targets
    state["price_watchlist"] = price_watchlist

    starters, bench, cap, vice = solve_fpl_knapsack(
        players, current_squad_ids, total_liquid_budget, free_transfers, team_avg_fdr, required_bank_reservation, weights, xmins_overrides, ACTIVE_CHIP
    )
    optimal_squad = starters + bench

    european_flags = check_european_congestion_flags(starters, fixtures_data, target_gw)
    mitigation_flags_str = "\n".join(european_flags) if european_flags else "[No active mitigation warning flags triggered]"

    projected_starting_xP = sum(get_base_ev(p, weights, xmins_overrides) for p in starters)
    if cap:
        cap_mult = 2.0 if ACTIVE_CHIP == "TRIPLE_CAPTAIN" else 1.0
        projected_starting_xP += get_base_ev(cap, weights, xmins_overrides) * cap_mult
        
    state["pending_evaluation"] = {
        "gw": target_gw,
        "projected_xP": round(projected_starting_xP, 2),
        "captain": cap["name"] if cap else None
    }

    save_state(state)
    
    locked_squad_str = f"--- MATHEMATICALLY LOCKED SQUAD (Liquid Value: £{total_liquid_budget:.1f}m | Reserved Bank: £{required_bank_reservation:.1f}m) ---\n"
    locked_squad_str += f"ACTIVE CHIP STATUS: {ACTIVE_CHIP}\n"
    
    if current_squad_ids:
        optimal_ids = [p["id"] for p in optimal_squad]
        transfers_in = [p["name"] for p in optimal_squad if p["id"] not in current_squad_ids]
        transfers_out = [players[i]["name"] for i in current_squad_ids if i not in optimal_ids]
        locked_squad_str += f"TRANSFERS OUT: {', '.join(transfers_out) if transfers_out else 'None'}\n"
        locked_squad_str += f"TRANSFERS IN: {', '.join(transfers_in) if transfers_in else 'None'}\n\n"
        
    def count_pos(group, pos_id): return len([p for p in group if p["pos_id"] == pos_id])
    formation = f"{count_pos(starters, 2)}-{count_pos(starters, 3)}-{count_pos(starters, 4)}"
    
    locked_squad_str += f"STARTING XI (Formation: {formation}):\n"
    for p in starters:
        is_cap = " (C)" if cap and p["id"] == cap["id"] else ""
        is_vice = " (VC)" if vice and p["id"] == vice["id"] else ""
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        actual_ev = round(get_base_ev(p, weights, xmins_overrides), 2)
        locked_squad_str += f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, Proj. Mins: {xmins:.1f}, TRUE 1-GW EV: {actual_ev}){is_cap}{is_vice}\n"
        
    locked_squad_str += "\nBENCH:\n"
    for i, p in enumerate(bench):
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else estimate_xmins(p)
        actual_ev = round(get_base_ev(p, weights, xmins_overrides), 2)
        locked_squad_str += f"Slot {i+1}: {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, Proj. Mins: {xmins:.1f}, TRUE 1-GW EV: {actual_ev})\n"
    
    locked_squad_str += f"\n### TACTICAL MITIGATION OPTION FLAGS\n{mitigation_flags_str}\n"

    # --- DEBUG: SPECIFIC PLAYER EV AUDIT ---
    debug_names = ["Saka", "Fernandes", "Gabriel", "Kinsky", "Raya", "Haaland"]
    print("\n--- DEBUG: PLAYER EV AUDIT ---")
    for p in players.values():
        if any(name.lower() in p["name"].lower() for name in debug_names):
            ev = get_base_ev(p, weights, xmins_overrides)
            print(f"{p['name']} ({p['team']}) - £{p['cost']}m | 1-GW EV: {ev:.3f} | xGI: {p['xgi_90']} | xGC: {p['xgc_90']}")
    print("------------------------------\n")

    return target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str

def build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news):
    gw1_override = ""
    if target_gw == 1 or str(free_transfers).lower() == "unlimited":
        gw1_override = "\n    6. PRE-SEASON RULE OVERRIDE: Gameweek 1 has UNLIMITED free transfers. Ignore point-hit constraints (Law 4)."

    if WORKFLOW_INPUT == "post_gameweek_review":
        action_type = "Post-Gameweek Strategic Review & Market Volatility Audit"
        phase_instructions = (
            "- Focus heavily on post-mortem recalibration results from the previous Gameweek.\n"
            "- Evaluate early transfer moves to catch impending price rises or dodge price crashes.\n"
            "- Focus on structural squad health, bank reservation strategy, and long-term 4-GW rolling planning."
        )
    elif WORKFLOW_INPUT == "pre_gameweek_deadline":
        action_type = "Pre-Gameweek Final Deadline Lock & Late ITK Leak Audit"
        phase_instructions = (
            "- Focus heavily on verified press conference news, injury updates, and late lineup leaks.\n"
            "- Apply Law 5 (15-Minute Panic Rule) if late leaks alter team selections.\n"
            "- Confirm absolute final Starting XI, Captaincy, Vice-Captaincy, and bench order before deadline."
        )
    else:
        action_type = "Full Weekly Execution & Analytical Breakdown"
        phase_instructions = (
            "- Provide a complete balanced breakdown covering both transfer economics and upcoming fixture geometry."
        )

    focus_instructions = (
        f"1. 11-Man Verification Lock & Strict xMins Audit: Output the exact mathematically locked Starting XI and Bench provided. Do NOT change any player, captain, bench order, or team affiliation tag.\n"
        f"2. Phase-Specific Focus ({action_type}):\n{phase_instructions}\n"
        f"3. Visual Aesthetics: Include monospaced ASCII pitch maps in Section 1 and Section 4 using markdown text codeblocks. Format Section 2 strictly as single-line bullet points with bold player names. Avoid wide Markdown tables.\n"
        f"4. Analytical Justification: Use the EXACT 'TRUE 1-GW EV' numbers provided in the mathematically locked squad list below. Do NOT invent or hallucinate Expected Value (EV) numbers. Format Section 2 strictly as single-line bullet points with bold player names.\n"
        f"5. Tactical Mitigation Option Flags: Review any generated TACTICAL MITIGATION OPTION FLAGS and present them clearly as optional human decisions.\n"
        f"6. Transfer Economics & Chip Status: Outline banking EV, market volatility, reserved bank capital, and macro chip alignment.\n"
        f"7. MANDATORY SIGN-OFF: Conclude your response with the boxed 'FINAL LOCKED-IN SQUAD SUMMARY' ASCII block."
    )

    prompt = f"""
    Run the {action_type} for Gameweek {target_gw}.
    
    ### CURRENT SQUAD STATE & ECONOMICS
    - Current Bank Balance: £{bank}m | Saved Free Transfers: {free_transfers}
    
{locked_squad_str}

    ### NEW ARRIVALS & FOREIGN TRANSFERS (ZERO PL HISTORY)
    The following players are new additions or foreign signings lacking Premier League historical baselines:
{new_arrivals_str}

    ### ACTIVE 2026/27 TRANSFER MARKET WATCHLIST
{market_str}
    
    {live_news}
    
    ### MANDATORY ANALYTICAL CONSTRAINTS
    1. Base all transfer and squad analysis STRICTLY on the Mathematically Locked Squad provided.
    2. Do NOT hallucinate players who are not currently active in the Premier League.
    3. Evaluate incoming transfer replacements STRICTLY using the ACTIVE 2026/27 TRANSFER MARKET WATCHLIST provided.
    4. LIVE NEWS OVERRIDE: Cross-reference the Market Watchlist against LIVE ITK NEWS and LONG TERM INJURIES. Ignore search items referencing non-Premier League players.
    5. FOREIGN TRANSFERS: Apply Law 7 to any players listed under NEW ARRIVALS. Highlight them for manual monitoring in Section 1.{gw1_override}
    
    ### DATA INSTRUCTIONS FOR EVALUATION
    {focus_instructions}
    
    Execute the full 5-section quantitative breakdown based strictly on your system instructions.
    """
    return prompt

def send_to_discord(webhook_url, text):
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        while len(line) > 1800:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(line[:1800])
            line = line[1800:]
            
        if len(current_chunk) + len(line) + 1 > 1800:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk)
        
    for chunk in chunks:
        requests.post(webhook_url, json={"content": chunk})

def main():
    target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str = get_fpl_data()
    print("--- FETCHING LIVE WEB SEARCH DATA ---")
    live_news = get_live_fpl_news()
    
    prompt = build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news)
    
    print("--- DATA FETCHED ---")
    print(f"Target GW: {target_gw} | Bank: {bank} | Transfers: {free_transfers} | Chip: {ACTIVE_CHIP}")
    print("--- QUERYING GEMINI API ---")
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2 
            )
        )
    except Exception as e:
        print(f"CRITICAL ERROR generating content with Gemini: {str(e)}")
        sys.exit(1)
        
    content = response.text if response and response.text else ""
    
    if not content:
        print("ERROR: Gemini returned an empty response string.")
        sys.exit(1)

    print(f"--- GEMINI RESPONSE RECEIVED ({len(content)} characters) ---")
    send_to_discord(DISCORD_WEBHOOK_URL, content)
    print("--- DISCORD DELIVERY COMPLETE ---")

if __name__ == "__main__":
    main()
