import os
import sys
import requests
import datetime
import pulp
from google import genai
from google.genai import types
from ddgs import DDGS

# 1. Load Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")

# 2. Pre-Flight Check
if not all([GEMINI_API_KEY, DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing base GitHub Secrets.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """
You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine. Your purpose is to evaluate squad selections, transfers, captaincy choices, and chip strategies using advanced underlying metrics, pitch geometry, game-state data, and FPL market economics. You must strictly adhere to the following analytical laws.

### CORE ANALYTICAL LAWS & METRIC DEFINITIONS

1. GAME-STATE NORMALISED ATTACKING & ACCIDENTAL ASSISTS
- Game-State Filter: You must normalize non-penalty expected goals (npxG) and expected assisted goals (xAG) strictly to periods when the game state is level (0-0, 1-1) or trailing. Exclude all scoreline-skewed "garbage time" data when a team is leading by 2+ goals.
- Finisher's Multiplier: Never project points using raw xG alone. Apply a multiplier based on a player's multi-season conversion history over their xG to accurately value elite shot placement versus low-efficiency volume shooters.
- Accidental Assist Rule: FPL awards assists for deflected passes if there is only one defensive touch inside the box. Therefore, track and prioritize "Raw Crosses into the Penalty Box" alongside xAG to raise the points ceiling for high-volume wide players.

2. DEFENSIVE CONTRIBUTION (DefCon) & 2026/27 BPS MATHEMATICS
- DefCon Thresholds: A 2-point floor is awarded for 10 CBIT (Clearances, Blocks, Interceptions, Tackles) for defenders, and 12 CBIRT (+ Recoveries) for midfielders/forwards. You must exclusively target baseline assets averaging >8.5 CBIT or >10.5 CBIRT per 90 minutes.
- BPS Centre-Back Nerf: Centre-backs now earn 1 BPS per 3 CBI actions (down from 2). You must downgrade pure centre-backs in bonus point projections.
- BPS Dribbler Buff: The -1 BPS penalty for being tackled has been removed. You must upgrade the projected BPS of direct, high-volume dribbling wingers and full-backs.
- BPS Penalty Flattening: Penalty goals are now a flat 12 BPS for all positions. You must manually strip out historical BPS spikes when projecting premium penalty-taking midfielders and forwards.
- Goalkeeper Save/Variance Balance: Goalkeepers earn 2 BPS per save, +1 for inside the box, +1 for big chances. However, FPL penalizes -1 point per 2 goals conceded. You must ONLY target high-save goalkeepers if their team concedes <1.5 Expected Goals Against (xGA) per match. Do not recommend goalkeepers facing heavy bombardment without a solid xGA baseline.
- Bench Fodder Exemption: Players priced at £4.0m or £4.5m who are strictly assigned to Bench Slots 3 and 4 are exempt from minimum DefCon or Goalkeeper xGA baseline thresholds. Do not discard viable bench fodder solely for failing starter metric requirements.

3. SPATIAL GEOMETRY & FLANK MISMATCHES
- Do not evaluate overall team defense vs overall team attack. You must cross-reference an attacker's primary pitch zone with the opposition's specific spatial weaknesses.
- Evaluate: Opposition Passes Allowed into Penalty Area, Box Touches Conceded, and Expected Crosses Conceded (xA_cross) broken down strictly by Left Flank, Right Flank, and Centre.

4. CHIP STRUCTURE & 5-TRANSFER ECONOMICS
- 5-Transfer Banking: Evaluate 0-transfer rolls not just as a "hold", but as an appreciating asset building toward a 3-to-5 transfer "mini-wildcard" that avoids point hits. 
- Point Hit Execution Constraint: You are forbidden from recommending a point hit (-4) unless the 4-Gameweek Expected Value (EV) of the incoming player exceeds the outgoing player by >5.5 points, OR if a -4 is mathematically required to field 11 starting players due to sudden suspensions/injuries.
- 8-Chip Macro Strategy: You must actively monitor and flag the deployment of the first set of chips (Wildcard, Free Hit, Triple Captain, Bench Boost) to ensure they are utilized optimally before the Gameweek 19 deadline lock.

5. QUALITATIVE OVERRIDES & TIER-1 LEAK VERIFICATION
- Expected Minutes (xMins) Overrides: Treat verified Tier-1 ITK/Injury intelligence (e.g., Ben Dinnery, Teamnewsandtix, Paul O'Keefe, Official Press Conferences, Physio Scout) as absolute mathematical overrides to base xMins (reducing them to 0 or 45 depending on the leak).
- The 15-Minute Panic Rule: If a lineup leak drops within 15 minutes of the Gameweek deadline and alters more than two squad decisions, you must default to the original multi-week EV plan and reject the leak to prevent execution errors.
- Tactical Hook Rates: Penalize the xMins of assets who possess a high "Sub-65 Minute Substitution Rate" under their current manager.

6. MATCH STAKES & FATIGUE MULTIPLIERS
- Cup Congestion Law: Apply a 25% xMins penalty to any outfield starter with <72 hours turnaround from a domestic/European cup match. Simultaneously, apply a 10% boost to the opposition's attacking metrics to account for pressing fatigue in the defending team.
- Motivation Adjustments: Boost xG projections for direct, counter-attacking forwards facing "Desperation Teams" (relegation/title chasers pushing high defensive lines). Increase overall returns for premium assets facing "Dead Rubber" opponents (mathematically secure teams with dropping PPDA/pressing intensity).

7. FOREIGN TRANSFERS & ZERO-HISTORY ASSETS
- Translation Discount: For players arriving from foreign leagues with no Premier League historical baseline data, apply a mandatory 20% discount to their expected attacking output (npxG/xAG) to account for Premier League adaptation and physical intensity.
- xMins Integration Cap: Cap initial Expected Minutes (xMins) for new foreign arrivals at 45-60 minutes for their first 3 Gameweeks to account for tactical integration, unless Tier-1 sources confirm they are direct, unchallengeable starters.
- Manual Flagging Constraint: You are forbidden from recommending immediate transfer-in executions for unproven foreign arrivals in GW1-GW3. You MUST isolate them in a dedicated "High-Potential Watchlist / Manual Audit" box within Section 1, highlighting what metrics (e.g., set-piece duties, starter confirmation) need to be verified before buying.

### AUTOMATED LIVE DATA SEARCH PROTOCOL
Before processing any weekly prompt or evaluating squad metrics, ingest and cross-reference all live fixture state data provided:
1. FIXTURE & SCHEDULE INGESTION: Read all provided schedule change data (Ben Crellin updates) for current/upcoming gameweeks. Automatically identify any postponed fixtures (Blanks) or rescheduled fixtures (Doubles) and adjust player xMins (0 xMins for Blanks, updated 180-min potential for Doubles).
2. INJURY & LINEUP INGESTION: Cross-reference the active watchlist against live injury news (Ben Dinnery / Physio Scout updates).
3. LIVE OVERRIDE EXECUTION: If live news explicitly confirms a player is out injured or suspended, override their FPL API status flag and treat them as 0 xMins, excluding them from immediate transfer consideration.

### OUTPUT FORMAT REQUIREMENTS
When responding to weekly prompts, structure your analysis strictly into these 5 sections:
1. Executive Summary & Core Moves (Immediate actions, starting XI decisions, and a dedicated "High-Potential Watchlist / Manual Audit" box for foreign arrivals).
2. Quantitative Trade-off Matrix (Table showing xPts, xMins, EV, and Keeper xGA limits).
3. Transfer Economics & Chip Status (Banking strategy, EV of rolling vs hitting, GW19 chip countdown).
4. Spatial, Game-State & Motivation Justification (Flank mismatches and stakes).
5. ITK & Congestion Audit (Impact of verified Tier-1 leaks, hook rates, and 72-hour cup turnarounds).
"""

def get_live_fpl_news():
    """Executes a free web search to inject live ITK data into the prompt."""
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
            
        long_term_results = DDGS().text("Premier League long term injuries ACL out for months", max_results=3)
        news_text += "\n--- LONG TERM INJURIES (Cross-Reference List) ---\n"
        for r in long_term_results:
            news_text += f"- {r.get('body', '')}\n"
    except Exception as e:
        news_text += f"[Search tool failed to retrieve live data: {e}]\n"
        
    return news_text

def get_base_ev(p):
    """Calculates a baseline 1-Gameweek Expected Value, ejecting confirmed absentees."""
    chance = p.get("chance_of_playing_next_round")
    if chance in [0, "0"] or p.get("status") not in ["a", "d"]:
        return -100.0  # Tweak 3: Hard Injury Ejection
    try:
        ep = float(p.get("ep_next", 0.0))
        if ep <= 0.0:
            tp = float(p.get("total_points", 0.0))
            form = float(p.get("form", 0.0))
            ep = (tp / 38.0) + form
        return ep
    except:
        return 0.0

def get_macro_ev(p, team_avg_fdr):
    """Calculates a 4-Gameweek EV factoring in algorithmic fixture difficulty."""
    base_ev = get_base_ev(p)
    if base_ev <= -100.0:
        return -100.0
    ev_4gw = base_ev * 4.0
    
    # Tweak 1: Algorithmic Multi-Week EV (FDR Adjustment)
    avg_fdr = team_avg_fdr.get(p["team_id"], 3.0)
    # Deduct 10% per FDR tier above 3.0, add 10% per tier below 3.0
    fdr_multiplier = 1.0 + ((3.0 - avg_fdr) * 0.1)
    
    return ev_4gw * fdr_multiplier

def solve_fpl_knapsack(players_dict, current_squad_ids, total_budget, free_transfers, team_avg_fdr):
    """Uses linear programming to mathematically select the optimal 15-player squad."""
    prob = pulp.LpProblem("FPL_Optimal_Squad", pulp.LpMaximize)
    valid_ids = list(players_dict.keys())
    
    decision_vars = pulp.LpVariable.dicts("player", valid_ids, cat="Binary")
    
    # Tweak 2: Linear Point-Hit Penalties
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat="Continuous")
    
    # Objective: Maximize total Macro EV minus 4 points per extra transfer beyond FT limit
    prob += pulp.lpSum([get_macro_ev(players_dict[i], team_avg_fdr) * decision_vars[i] for i in valid_ids]) - (4.0 * extra_transfers)
    
    # Constraint: Exclude injured/unavailable players from being transferred IN
    for i in valid_ids:
        if i not in current_squad_ids:
            p = players_dict[i]
            chance = p.get("chance_of_playing_next_round")
            if chance in [0, "0"] or p.get("status") not in ["a", "d"]:
                prob += decision_vars[i] == 0
                
    # Core Constraints
    prob += pulp.lpSum([decision_vars[i] for i in valid_ids]) == 15
    prob += pulp.lpSum([players_dict[i]["cost"] * decision_vars[i] for i in valid_ids]) <= total_budget
    
    # Positional Limits
    prob += pulp.lpSum([decision_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
    prob += pulp.lpSum([decision_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
    prob += pulp.lpSum([decision_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
    prob += pulp.lpSum([decision_vars[i] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
    
    # Max 3 players per Premier League team
    team_ids = set(players_dict[i]["team_id"] for i in valid_ids)
    for t_id in team_ids:
        prob += pulp.lpSum([decision_vars[i] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

    # Transfer Cost Mathematics
    if free_transfers != "Unlimited":
        try:
            ft = int(str(free_transfers).replace("+", "").strip())
        except:
            ft = 1
        
        if current_squad_ids and len(current_squad_ids) == 15:
            overlap = [i for i in current_squad_ids if i in valid_ids]
            players_kept = pulp.lpSum([decision_vars[i] for i in overlap])
            transfers_made = 15 - players_kept
            prob += extra_transfers >= transfers_made - ft

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    optimal_squad = [players_dict[i] for i in valid_ids if decision_vars[i].varValue and decision_vars[i].varValue > 0.5]
    return optimal_squad

def solve_starting_xi(squad_players):
    """Uses linear programming to mathematically select the optimal 11 starters and bench order."""
    prob = pulp.LpProblem("Starting_XI", pulp.LpMaximize)
    ids = [p["id"] for p in squad_players]
    decision_vars = pulp.LpVariable.dicts("starter", ids, cat="Binary")
    
    # Objective: Maximize immediate 1-GW EV for the starting lineup
    prob += pulp.lpSum([get_base_ev(p) * decision_vars[p["id"]] for p in squad_players])
    
    prob += pulp.lpSum([decision_vars[i] for i in ids]) == 11
    
    # Valid Formation Geometry Constraints
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 1]) == 1
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 2]) >= 3
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 2]) <= 5
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 3]) >= 2
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 3]) <= 5
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 4]) >= 1
    prob += pulp.lpSum([decision_vars[p["id"]] for p in squad_players if p["pos_id"] == 4]) <= 3
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters = [p for p in squad_players if decision_vars[p["id"]].varValue and decision_vars[p["id"]].varValue > 0.5]
    bench = [p for p in squad_players if not (decision_vars[p["id"]].varValue and decision_vars[p["id"]].varValue > 0.5)]
    
    starters.sort(key=lambda x: x["pos_id"])
    
    bench_gk = [p for p in bench if p["pos_id"] == 1]
    bench_outfield = sorted([p for p in bench if p["pos_id"] != 1], key=lambda x: get_base_ev(x), reverse=True)
    sorted_bench = bench_gk + bench_outfield
    
    starters_sorted_by_ep = sorted(starters, key=lambda x: get_base_ev(x), reverse=True)
    captain = starters_sorted_by_ep[0] if starters_sorted_by_ep else None
    vice = starters_sorted_by_ep[1] if len(starters_sorted_by_ep) > 1 else None
    
    return starters, sorted_bench, captain, vice

def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/4.0"}
    
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}")
        sys.exit(1)
        
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}
    
    current_gw = next((e for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw = next((e for e in bootstrap_data["events"] if e.get("is_next")), None)
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    active_gw = current_gw['id'] if current_gw else (target_gw if target_gw > 1 else 1)
    
    # Pre-fetch FDR metrics
    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        fixtures_data = fixtures_resp.json()
    except Exception as e:
        print(f"WARNING: Error fetching fixtures: {e}")
        fixtures_data = []
        
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 4:
            team_a = f.get("team_a")
            team_h = f.get("team_h")
            if team_a in team_fdr_sum:
                team_fdr_sum[team_a] += f.get("team_a_difficulty", 3)
                team_fdr_count[team_a] += 1
            if team_h in team_fdr_sum:
                team_fdr_sum[team_h] += f.get("team_h_difficulty", 3)
                team_fdr_count[team_h] += 1
                
    team_avg_fdr = {}
    for t in teams.keys():
        if team_fdr_count[t] > 0:
            team_avg_fdr[t] = team_fdr_sum[t] / team_fdr_count[t]
        else:
            team_avg_fdr[t] = 3.0

    players = {}
    for p in bootstrap_data["elements"]:
        players[p["id"]] = {
            "id": p["id"],
            "name": p["web_name"],
            "team": teams.get(p["team"], "UNK"),
            "team_id": p["team"],
            "pos": element_types.get(p["element_type"], "UNK"),
            "pos_id": p["element_type"],
            "cost": p["now_cost"] / 10.0,
            "status": p["status"],
            "news": p["news"],
            "ep_next": p.get("ep_next", "0.0"),
            "total_points": p.get("total_points", 0),
            "form": p.get("form", "0.0"),
            "own": p.get("selected_by_percent", 0),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round")
        }
        
    market_list = []
    available_players = [p for p in players.values() if p.get("status") in ["a", "d"]]
    for pos_id in [1, 2, 3, 4]: 
        pos_players = [p for p in available_players if p["pos_id"] == pos_id]
        top_pos = sorted(pos_players, key=lambda x: float(x.get("own", 0)), reverse=True)[:30]
        
        for p in top_pos:
            news_flag = f" | FLAG: {p['news']}" if p['news'] else ""
            market_list.append(f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, {p['own']}% owned, Status: {p['status']}{news_flag})")
    market_str = "\n".join(market_list)

    new_arrivals = []
    for p in players.values():
        news_text = p["news"].lower() if p["news"] else ""
        is_new_transfer = "joined" in news_text or "transferred" in news_text or "signed" in news_text
        is_high_value_zero_min = (p["cost"] >= 6.0) and (p["total_points"] == 0)
        if (is_new_transfer or is_high_value_zero_min) and p["status"] in ["a", "d"]:
            news_msg = p["news"] if p["news"] else "New Transfer / Foreign Arrival"
            new_arrivals.append(f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m, {p['own']}% owned) | NOTE: {news_msg}")
    new_arrivals_str = "\n".join(new_arrivals) if new_arrivals else "No recent high-profile foreign arrivals detected in API."
    
    team_history_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/"
    bank = 0.0
    free_transfers = "Unlimited (Pre-Season GW1)" if target_gw == 1 else "1+"
    
    try:
        hist_resp = requests.get(team_history_url, headers=headers)
        if hist_resp.status_code == 200:
            h_data = hist_resp.json()
            if h_data.get("current"):
                last_gw_data = h_data["current"][-1]
                bank = last_gw_data.get("bank", 0) / 10.0
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")
        
    squad_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{active_gw}/picks/"
    current_squad_ids = []
    current_squad_value = 0.0
    try:
        squad_resp = requests.get(squad_url, headers=headers)
        if squad_resp.status_code == 200:
            picks_data = squad_resp.json()
            for pick in picks_data.get("picks", []):
                pid = pick["element"]
                current_squad_ids.append(pid)
                current_squad_value += players.get(pid, {}).get("cost", 0.0)
    except Exception as e:
        print(f"WARNING: Error fetching squad picks: {e}")

    total_budget = (current_squad_value + bank) if current_squad_ids else 100.0

    # Execute Python Math Optimization
    optimal_squad = solve_fpl_knapsack(players, current_squad_ids, total_budget, free_transfers, team_avg_fdr)
    starters, bench, cap, vice = solve_starting_xi(optimal_squad)
    
    # Format The Locked Output for the LLM
    locked_squad_str = f"--- MATHEMATICALLY LOCKED SQUAD (Total Value: £{total_budget}m) ---\n"
    
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
        locked_squad_str += f"- {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m){is_cap}{is_vice}\n"
        
    locked_squad_str += "\nBENCH:\n"
    for i, p in enumerate(bench):
        locked_squad_str += f"Slot {i+1}: {p['name']} ({p['team']}, {p['pos']}, £{p['cost']}m)\n"
    
    return target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str

def build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news):
    gw1_override = ""
    if target_gw == 1 or "Unlimited" in str(free_transfers):
        gw1_override = "\n    6. PRE-SEASON RULE OVERRIDE: Gameweek 1 has UNLIMITED free transfers. Ignore all point-hit penalty constraints (Law 4)."

    action_type = "Full Weekly Execution & Analytical Breakdown"
    
    focus_instructions = """1. 11-Man Verification Lock: You MUST output the exact mathematically locked Starting XI and Bench provided below. Do NOT change a single player, captain, or bench order. The linear programming solver has already optimized the budget knapsack.
    2. Analytical Justification: Provide the quantitative trade-off matrix and explain the geometric mismatches (Law 3) that validate this mathematical selection.
    3. Transfer Economics & Chip Status: Outline banking EV, market volatility, and macro chip alignment based on the executed transfers provided.
    4. MANDATORY SIGN-OFF: You must conclude your entire response with a highly visible 'FINAL LOCKED-IN SQUAD SUMMARY' block mirroring the exact locked structure provided."""

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
    1. Base all transfer and squad analysis STRICTLY on the Mathematically Locked Squad provided. Read any appended injury FLAGs carefully.
    2. Do NOT hallucinate players who are not currently active in the Premier League.
    3. Evaluate incoming transfer replacements STRICTLY using the ACTIVE 2026/27 TRANSFER MARKET WATCHLIST provided. Drop any players from consideration if their FLAG indicates a serious injury.
    4. LIVE NEWS OVERRIDE: You must meticulously cross-reference the Market Watchlist against the LIVE ITK NEWS and LONG TERM INJURIES sections.
    5. FOREIGN TRANSFERS: Apply Law 7 to any players listed under NEW ARRIVALS. Highlight them for manual monitoring in Section 1.{gw1_override}
    
    ### DATA INSTRUCTIONS FOR EVALUATION
    {focus_instructions}
    
    Execute the full 5-section quantitative breakdown based strictly on your system instructions.
    """
    return prompt

def send_to_discord(webhook_url, text):
    lines = text.split("\n")
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > 1800:
            requests.post(webhook_url, json={"content": current_chunk})
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk.strip():
        requests.post(webhook_url, json={"content": current_chunk})

def main():
    target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str = get_fpl_data()
    print("--- FETCHING LIVE WEB SEARCH DATA ---")
    live_news = get_live_fpl_news()
    
    prompt = build_prompt(target_gw, bank, free_transfers, locked_squad_str, market_str, new_arrivals_str, live_news)
    
    print("--- DATA FETCHED ---")
    print(f"Target GW: {target_gw} | Bank: {bank} | Transfers: {free_transfers}")
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
    print(f"--- GEMINI RESPONSE RECEIVED ({len(content)} characters) ---")
    
    if not content:
        print("ERROR: Gemini returned an empty response string.")
        sys.exit(1)
        
    send_to_discord(DISCORD_WEBHOOK_URL, content)
    print("--- DISCORD DELIVERY COMPLETE ---")

if __name__ == "__main__":
    main()
