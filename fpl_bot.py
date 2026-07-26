import os
import sys
import requests
import datetime
from google import genai

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

def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/1.2"}
    
    # 1. Fetch public bootstrap data
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap: {e}")
        sys.exit(1)
        
    current_gw = next((e for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw = next((e for e in bootstrap_data["events"] if e.get("is_next")), None)
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    
    # 2. Fetch public team history to get bank data WITHOUT login
    team_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/"
    bank = "Unknown (Check manually if making early transfers)"
    free_transfers = "Unknown (Check manually)"
    
    try:
        team_resp = requests.get(team_url, headers=headers)
        if team_resp.status_code == 200:
            history_data = team_resp.json()
            if history_data.get("current"):
                # Get the bank from the most recently completed Gameweek
                last_gw_data = history_data["current"][-1]
                bank = last_gw_data.get("bank", 0) / 10.0
                free_transfers = "1+ (Confirm exact max bank manually)"
        else:
            print(f"WARNING: Could not fetch public history. HTTP {team_resp.status_code}.")
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")
        
    return target_gw, bank, free_transfers

def build_prompt(target_gw, bank, free_transfers):
    day_of_week = datetime.datetime.today().weekday()
    
    if WORKFLOW_INPUT == "monday" or (WORKFLOW_INPUT == "auto" and day_of_week <= 2):
        action_type = "Monday Market Assessment & FPL Optimization Protocol"
        focus_instructions = "1. Price Volatility Check \n2. Transfer Banking EV \n3. Macro Chip Alignment"
    else:
        action_type = "Friday Execution Protocol"
        focus_instructions = "1. Finalize Starting XI and Bench Order \n2. Calculate Captain and Vice-Captain EV \n3. Recalculate all xMins"

    prompt = f"""
    Run the {action_type} for Gameweek {target_gw}.
    
    ### CURRENT SQUAD STATE & ECONOMICS
    - Current Bank Balance: £{bank}m | Saved Free Transfers: {free_transfers}
    
    ### AUTOMATED LIVE DATA SEARCH PROTOCOL
    Before processing, execute a live web search to pull the latest fixture state:
    1. Search for "Premier League fixture changes blank double gameweeks Ben Crellin".
    2. Search for "Premier League team news injuries Ben Dinnery".
    
    ### DATA INSTRUCTIONS FOR EVALUATION
    {focus_instructions}
    
    Execute the 5-section quantitative breakdown based on the V5 System Instructions.
    """
    return prompt

def main():
    target_gw, bank, free_transfers = get_fpl_data()
    prompt = build_prompt(target_gw, bank, free_transfers)
    
    print(f"--- DATA FETCHED ---")
    print(f"Target GW: {target_gw} | Bank: {bank} | Transfers: {free_transfers}")
    print("--- QUERYING GEMINI API ---")
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config={
                "system_instruction": "You are an institutional-grade Quantitative Fantasy Premier League Analyst... [PASTE ENTIRE V5 MASTER INSTRUCTION HERE]"
            }
        )
    except Exception as e:
        print(f"CRITICAL ERROR generating content with Gemini: {str(e)}")
        sys.exit(1)
        
    content = response.text if response and response.text else ""
    print(f"--- GEMINI RESPONSE RECEIVED ({len(content)} characters) ---")
    
    if not content:
        print("ERROR: Gemini returned an empty response string.")
        sys.exit(1)
        
    # Send to Discord in 1700-character clean chunks without backtick wrapping
    for i in range(0, len(content), 1700):
        chunk = content[i:i+1700]
        discord_resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk})
        print(f"Discord POST HTTP Status: {discord_resp.status_code}")
        
        if discord_resp.status_code not in [200, 204]:
            print(f"DISCORD ERROR RESPONSE: {discord_resp.text}")

if __name__ == "__main__":
    main()
