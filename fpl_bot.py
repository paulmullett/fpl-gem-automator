import os
import sys
import requests
import datetime
from google import genai

# 1. Load Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
FPL_EMAIL = os.environ.get("FPL_EMAIL")
FPL_PASSWORD = os.environ.get("FPL_PASSWORD")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")

# 2. Pre-Flight Check
if not all([GEMINI_API_KEY, DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing base GitHub Secrets.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

def get_fpl_data():
    session = requests.Session()
    headers = {"User-Agent": "FPL-Auto-Script/1.1"}
    
    # Authenticate to access the protected /my-team/ endpoint
    if FPL_EMAIL and FPL_PASSWORD:
        login_url = "https://users.premierleague.com/accounts/login/"
        payload = {
            "login": FPL_EMAIL,
            "password": FPL_PASSWORD,
            "app": "plfpl-web",
            "redirect_uri": "https://fantasy.premierleague.com/a/login"
        }
        session.post(login_url, data=payload, headers=headers)
    else:
        print("WARNING: Email/Password secrets missing. The protected /my-team/ endpoint will fail.")
    
    # Fetch public bootstrap data
    bootstrap_resp = session.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    if bootstrap_resp.status_code != 200:
        print(f"ERROR: Failed to fetch bootstrap. HTTP {bootstrap_resp.status_code}")
        sys.exit(1)
        
    bootstrap_data = bootstrap_resp.json()
    
    current_gw = next((e for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw = next((e for e in bootstrap_data["events"] if e.get("is_next")), None)
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    
    # Fetch private team data (Bank & Transfers)
    team_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/my-team/"
    team_resp = session.get(team_url, headers=headers)
    
    if team_resp.status_code == 200:
        team_data = team_resp.json()
        bank = team_data.get("transfers", {}).get("bank", 0) / 10.0
        limit = team_data.get("transfers", {}).get("limit", 1)
        made = team_data.get("transfers", {}).get("made", 0)
        free_transfers = limit - made
    else:
        print(f"ERROR: Could not fetch private team data. HTTP {team_resp.status_code}. Using fallback values.")
        bank = "Unknown"
        free_transfers = "Unknown"
        
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
    - Current Bank Balance: £{bank}m | Saved Free Transfers: {free_transfers} (Max 5)
    
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
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                "system_instruction": "You are an institutional-grade Quantitative Fantasy Premier League Analyst... [PASTE ENTIRE V5 MASTER INSTRUCTION HERE]"
            }
        )
    except Exception as e:
        print(f"CRITICAL ERROR generating content with Gemini: {str(e)}")
        sys.exit(1)
        
    content = response.text
    for i in range(0, len(content), 1900):
        chunk = content[i:i+1900]
        discord_resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": f"```markdown\n{chunk}\n```"})
        if discord_resp.status_code not in [200, 204]:
            print(f"WARNING: Failed to send to Discord. HTTP {discord_resp.status_code}")

if __name__ == "__main__":
    main()
