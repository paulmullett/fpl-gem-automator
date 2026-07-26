import os
import requests
import datetime
from google import genai

# 1. Load Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Fetch Live Data from Official FPL API
def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/1.0"}
    
    # Fetch general data (events, price changes, blank/double schedules)
    bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
    bootstrap_data = bootstrap_resp.json()
    
    # Fetch your specific squad state & bank
    team_resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/my-team/", headers=headers)
    team_data = team_resp.json()
    
    # Identify the current/next Gameweek
    current_gw = next((event for event in bootstrap_data["events"] if event["is_current"]), None)
    next_gw = next((event for event in bootstrap_data["events"] if event["is_next"]), None)
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    
    # Extract Bank & Transfers
    bank = team_data.get("transfers", {}).get("bank", 0) / 10.0
    free_transfers = team_data.get("transfers", {}).get("limit", 1) - team_data.get("transfers", {}).get("made", 0)
    
    return target_gw, bank, free_transfers

# 3. Determine Prompt Type (Monday vs Friday vs Manual)
def build_prompt(target_gw, bank, free_transfers):
    day_of_week = datetime.datetime.today().weekday() # 0 = Monday, 4 = Friday
    
    if WORKFLOW_INPUT == "monday" or (WORKFLOW_INPUT == "auto" and day_of_week <= 2):
        action_type = "Monday Market Assessment & FPL Optimization Protocol"
        focus_instructions = "1. Price Volatility Check \n2. Transfer Banking EV \n3. Macro Chip Alignment"
    else:
        action_type = "Friday Execution Protocol"
        focus_instructions = "1. Finalize Starting XI and Bench Order \n2. Calculate Captain and Vice-Captain EV \n3. Recalculate all xMins"

    # Assemble the dynamic prompt combining live data and your V5 instructions
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

# 4. Generate AI Advice & Send to Discord
def main():
    target_gw, bank, free_transfers = get_fpl_data()
    prompt = build_prompt(target_gw, bank, free_transfers)
    
    # Send to Gemini with your hardcoded V5 Master Prompt as system instructions
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            "system_instruction": "You are an institutional-grade Quantitative Fantasy Premier League Analyst... [PASTE ENTIRE V5 MASTER INSTRUCTION HERE]"
        }
    )
    
    # Deliver payload to Discord
    # Discord limits messages to 2000 characters, so we chunk the response
    content = response.text
    for i in range(0, len(content), 1900):
        chunk = content[i:i+1900]
        requests.post(DISCORD_WEBHOOK_URL, json={"content": f"```markdown\n{chunk}\n```"})

if __name__ == "__main__":
    main()
