import os
import sys
import requests
import datetime
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

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

# PASTE YOUR FULL V5 MASTER SYSTEM INSTRUCTION HERE
SYSTEM_INSTRUCTION = """
You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine... [PASTE ENTIRE V5 MASTER INSTRUCTION HERE]
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

def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/1.8"}
    
    # 1. Fetch bootstrap static to build current player database
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}")
        sys.exit(1)
        
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}
    
    # Map all active players
    players = {}
    for p in bootstrap_data["elements"]:
        players[p["id"]] = {
            "name": p["web_name"],
            "team": teams.get(p["team"], "UNK"),
            "pos": element_types.get(p["element_type"], "UNK"),
            "cost": p["now_cost"] / 10.0,
            "status": p["status"],
            "news": p["news"]
        }
        
    # Build a "Smart 120" Active Market Watchlist
    available_players = []
    for p in bootstrap_data["elements"]:
        status = p.get("status", "a")
        chance_next = p.get("chance_of_playing_next_round")
        
        # Hard drop players officially ruled out (0% chance) or fully flagged
        # Note: chance_next is 'None' when a player is 100% fit
        if chance_next in [0, "0"] or status in ["i", "s", "u", "n"]:
            continue
            
        available_players.append(p)
    
    market_list = []
    # 1=GK, 2=DEF, 3=MID, 4=FWD
    for pos_id in [1, 2, 3, 4]: 
        pos_players = [p for p in available_players if p["element_type"] == pos_id]
        top_pos = sorted(pos_players, key=lambda x: float(x.get("selected_by_percent", 0)), reverse=True)[:30]
        
        for p in top_pos:
            name = p["web_name"]
            team = teams.get(p["team"], "UNK")
            pos = element_types.get(p["element_type"], "UNK")
            cost = p["now_cost"] / 10.0
            own = p.get("selected_by_percent", 0)
            status = p.get("status", "a")
            news = p.get("news", "")
            
            # Inject the exact injury news into the string if it exists
            news_flag = f" | FLAG: {news}" if news else ""
            market_list.append(f"- {name} ({team}, {pos}, £{cost}m, {own}% owned, Status: {status}{news_flag})")
            
    market_str = "\n".join(market_list)
        
    current_gw = next((e for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw = next((e for e in bootstrap_data["events"] if e.get("is_next")), None)
    
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    active_gw = current_gw['id'] if current_gw else (target_gw if target_gw > 1 else 1)
    
    # 2. Fetch user history for bank balance
    team_history_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/"
    bank = "0.0"
    free_transfers = "1+"
    
    try:
        hist_resp = requests.get(team_history_url, headers=headers)
        if hist_resp.status_code == 200:
            h_data = hist_resp.json()
            if h_data.get("current"):
                last_gw_data = h_data["current"][-1]
                bank = str(last_gw_data.get("bank", 0) / 10.0)
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")
        
    # 3. Fetch user's actual 15-player squad
    squad_list = []
    squad_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{active_gw}/picks/"
    try:
        squad_resp = requests.get(squad_url, headers=headers)
        if squad_resp.status_code == 200:
            picks_data = squad_resp.json()
            for pick in picks_data.get("picks", []):
                p_info = players.get(pick["element"], {})
                role = "Starter" if pick["position"] <= 11 else "Bench"
                cap = " (C)" if pick.get("is_captain") else (" (VC)" if pick.get("is_vice_captain") else "")
                
                # Fetch news for squad members too so AI can assess current injuries
                s_news = p_info.get("news", "")
                s_news_flag = f" | FLAG: {s_news}" if s_news else ""
                squad_list.append(f"- {p_info.get('name', 'Unknown')} ({p_info.get('team')}, {p_info.get('pos')}, £{p_info.get('cost')}m) - {role}{cap}{s_news_flag}")
    except Exception as e:
        print(f"WARNING: Error fetching squad picks: {e}")

    squad_str = "\n".join(squad_list) if squad_list else "Squad picks not yet public/locked for this gameweek."
    
    return target_gw, bank, free_transfers, squad_str, market_str
    
def build_prompt(target_gw, bank, free_transfers, squad_str, market_str, live_news):
    day_of_week = datetime.datetime.today().weekday()
    
    if WORKFLOW_INPUT == "monday" or (WORKFLOW_INPUT == "auto" and day_of_week <= 2):
        action_type = "Monday Market Assessment & FPL Optimization Protocol"
        focus_instructions = "1. Price Volatility Check\n2. Transfer Banking EV\n3. Macro Chip Alignment"
    else:
        action_type = "Friday Execution Protocol"
        focus_instructions = "1. Finalize Starting XI and Bench Order\n2. Calculate Captain and Vice-Captain EV\n3. Recalculate all xMins based strictly on the provided Live ITK News."

    prompt = f"""
    Run the {action_type} for Gameweek {target_gw}.
    
    ### CURRENT SQUAD STATE & ECONOMICS
    - Current Bank Balance: £{bank}m | Saved Free Transfers: {free_transfers}
    - ACTUAL 15-PLAYER SQUAD:
{squad_str}

    ### ACTIVE 2026/27 TRANSFER MARKET WATCHLIST
    The following are the top 120 most relevant active players in the game currently, split by position. 
    YOU MAY ONLY RECOMMEND INCOMING TRANSFERS FROM THIS SPECIFIC LIST:
{market_str}
    
    {live_news}
    
    ### MANDATORY ANALYTICAL CONSTRAINTS
    1. Base all transfer and squad analysis STRICTLY on the actual 15 players listed in the current squad state above. Read any appended injury FLAGs carefully.
    2. Do NOT hallucinate players who are not currently active in the Premier League.
    3. Evaluate incoming transfer replacements STRICTLY using the ACTIVE 2026/27 TRANSFER MARKET WATCHLIST provided. Drop any players from consideration if their FLAG indicates a serious injury.
    4. LIVE NEWS OVERRIDE: You must meticulously cross-reference the Market Watchlist against the LIVE ITK NEWS and LONG TERM INJURIES sections. If the live news states a player is injured (e.g., Gabriel, Saliba, etc.), you MUST treat them as having 0 xMins and ban them from transfer consideration, even if the official FPL Watchlist lists them as fit.
    
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
    target_gw, bank, free_transfers, squad_str, market_str = get_fpl_data()
    print("--- FETCHING LIVE WEB SEARCH DATA ---")
    live_news = get_live_fpl_news()
    
    prompt = build_prompt(target_gw, bank, free_transfers, squad_str, market_str, live_news)
    
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
