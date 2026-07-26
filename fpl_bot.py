import os
import sys
import requests
import datetime
from google import genai
from google.genai import types

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
SYSTEM_INSTRUCTION = """You are an institutional-grade Quantitative Fantasy Premier League (FPL) Analyst and Tactical Decision Engine. Your purpose is to evaluate squad selections, transfers, captaincy choices, and chip strategies using advanced underlying metrics, pitch geometry, game-state data, and FPL market economics. You must strictly adhere to the following analytical laws.

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

### AUTOMATED LIVE DATA SEARCH PROTOCOL
Before processing any weekly prompt or evaluating squad metrics, you must execute a live web search to pull the latest fixture state:
1. FIXTURE SEARCH: Search for "Premier League fixture changes blank double gameweeks Ben Crellin" for the current gameweeks.
2. INJURY/LINEUP SEARCH: Search for "Premier League team news injuries Ben Dinnery" for the upcoming gameweek.
3. AUTOMATED INGESTION: Automatically identify any postponed fixtures (Blanks) or rescheduled fixtures (Doubles) and adjust player xMins (0 xMins for Blanks, updated 180-min potential for Doubles) before outputting the evaluation. Do not require the user to manually input fixture schedule changes if they are publicly confirmed.

### OUTPUT FORMAT REQUIREMENTS
When responding to weekly prompts, structure your analysis strictly into these 5 sections:
1. Executive Summary & Core Moves (Immediate actions).
2. Quantitative Trade-off Matrix (Table showing xPts, xMins, EV, and Keeper xGA limits).
3. Transfer Economics & Chip Status (Banking strategy, EV of rolling vs hitting, GW19 chip countdown).
4. Spatial, Game-State & Motivation Justification (Flank mismatches and stakes).
5. ITK & Congestion Audit (Impact of verified Tier-1 leaks, hook rates, and 72-hour cup turnarounds).
"""

def get_fpl_data():
    headers = {"User-Agent": "FPL-Auto-Script/1.3"}
    
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}")
        sys.exit(1)
        
    current_gw = next((e for e in bootstrap_data["events"] if e.get("is_current")), None)
    next_gw = next((e for e in bootstrap_data["events"] if e.get("is_next")), None)
    target_gw = next_gw['id'] if next_gw else (current_gw['id'] if current_gw else 1)
    
    team_url = f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/"
    bank = "Unknown (Check manually if making early transfers)"
    free_transfers = "Unknown (Check manually)"
    
    try:
        team_resp = requests.get(team_url, headers=headers)
        if team_resp.status_code == 200:
            history_data = team_resp.json()
            if history_data.get("current"):
                last_gw_data = history_data["current"][-1]
                bank = last_gw_data.get("bank", 0) / 10.0
                free_transfers = "1+ (Confirm exact count manually)"
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")
        
    return target_gw, bank, free_transfers

def build_prompt(target_gw, bank, free_transfers):
    day_of_week = datetime.datetime.today().weekday()
    
    if WORKFLOW_INPUT == "monday" or (WORKFLOW_INPUT == "auto" and day_of_week <= 2):
        action_type = "Monday Market Assessment & FPL Optimization Protocol"
        focus_instructions = "1. Price Volatility Check\n2. Transfer Banking EV\n3. Macro Chip Alignment"
    else:
        action_type = "Friday Execution Protocol"
        focus_instructions = "1. Finalize Starting XI and Bench Order\n2. Calculate Captain and Vice-Captain EV\n3. Recalculate all xMins based on recent team news"

    prompt = f"""
    Run the {action_type} for Gameweek {target_gw}.
    
    ### CURRENT SQUAD STATE & ECONOMICS
    - Current Bank Balance: £{bank}m | Saved Free Transfers: {free_transfers}
    
    ### LIVE DATA REQUIREMENTS
    Perform a live web search for:
    - Premier League fixture changes, blank/double gameweeks (Ben Crellin).
    - Latest injury news and press conference updates (Ben Dinnery / Physio Scout).
    
    ### DATA INSTRUCTIONS FOR EVALUATION
    {focus_instructions}
    
    Execute the full 5-section quantitative breakdown based on your system instructions.
    """
    return prompt

def send_to_discord(webhook_url, text):
    """Splits output cleanly by line boundaries to preserve Discord Markdown formatting."""
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
    target_gw, bank, free_transfers = get_fpl_data()
    prompt = build_prompt(target_gw, bank, free_transfers)
    
    print("--- DATA FETCHED ---")
    print(f"Target GW: {target_gw} | Bank: {bank} | Transfers: {free_transfers}")
    print("--- QUERYING GEMINI API WITH LIVE SEARCH GROUNDING ---")
    
    try:
        response = client.models.generate_content(
            model='emini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[{"google_search": {}}]  # Native Search Grounding Enabled
            )
        )
    except Exception as e:
        print(f"CRITICAL ERROR generating content with Gemini: {str(e)}")
        sys.exit(1)
        
    content = response.text if response and response.text else ""
    print(f"--- GEMINI RESPONSE RECEIVED ({len(content)} characters) ---")
    
    if not content:
        if response and response.candidates:
            print(f"Candidate Finish Reason: {response.candidates[0].finish_reason}")
        print("ERROR: Gemini returned an empty response string.")
        sys.exit(1)
        
    send_to_discord(DISCORD_WEBHOOK_URL, content)
    print("--- DISCORD DELIVERY COMPLETE ---")

if __name__ == "__main__":
    main()
