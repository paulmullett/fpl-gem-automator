"""
fpl_odds_engine.py — Live Bookmaker Odds Ingestion & Poisson Intensity Solver

Integrates live betting odds (The Odds API) to calibrate team-level attacking (xG)
and defensive (xGC) expected values. Uses scipy.optimize.fsolve to remove bookmaker
vigorish (margin) and solve for implied Poisson intensity parameters.
"""

import os
import requests
import math
from scipy.optimize import fsolve

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

TEAM_NAME_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "AFC Bournemouth": "BOU", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Brighton and Hove Albion": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS",
    "Ipswich Town": "IPS", "Leicester": "LEI", "Leicester City": "LEI", "Liverpool": "LIV",
    "Manchester City": "MCI", "Man City": "MCI", "Manchester United": "MUN", "Man Utd": "MUN",
    "Newcastle": "NEW", "Newcastle United": "NEW", "Nottingham Forest": "NFO", "Southampton": "SOU",
    "Tottenham": "TOT", "Tottenham Hotspur": "TOT", "West Ham": "WHU", "Wolves": "WOL"
}

def fetch_live_odds():
    """Fetches raw market odds for Premier League fixtures from The Odds API."""
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not found. Operating on standard baseline heuristics.")
        return []
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {"apiKey": ODDS_API_KEY, "regions": "uk,eu", "markets": "h2h,totals", "oddsFormat": "decimal"}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        print(f"Odds API error: {e}")
        return []

def remove_vig(odds_list: list) -> list:
    """Removes bookmaker overround to return clean implied probabilities."""
    implied_probs = [1.0 / d for d in odds_list if d > 1.0]
    total = sum(implied_probs)
    if total <= 0: return [1.0 / len(odds_list)] * len(odds_list)
    return [p / total for p in implied_probs]

def compute_implied_metrics(match_event: dict):
    """Solves for team expected goals (lambda) and clean sheet probabilities."""
    home_team = match_event.get("home_team")
    away_team = match_event.get("away_team")
    bookmakers = match_event.get("bookmakers", [])
    if not bookmakers: return None
    
    target_bookie = next((b for b in bookmakers if b["key"] in ["pinnacle", "bet365", "skybet"]), bookmakers[0])
    markets = {m["key"]: m["outcomes"] for m in target_bookie.get("markets", [])}
    
    h2h, totals = markets.get("h2h", []), markets.get("totals", [])
    if not h2h or not totals: return None

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
        if lam[0] <= 0: return 1.0
        p_under_model = math.exp(-lam[0]) * (1.0 + lam[0] + (lam[0]**2) / 2.0)
        return p_under_model - p_under_25

    try: lam_total = fsolve(objective, [2.5])[0]
    except: lam_total = 2.5
    lam_total = max(0.5, min(5.0, lam_total))

    advantage_factor = max(0.1, p_home / max(0.01, p_away))
    fraction_home = max(0.2, min(0.8, advantage_factor / (1.0 + advantage_factor)))
    
    lam_home = lam_total * fraction_home
    lam_away = lam_total * (1.0 - fraction_home)

    return {
        "home_team": home_team, "away_team": away_team,
        "lambda_home": round(lam_home, 3), "lambda_away": round(lam_away, 3),
        "cs_probability_home": round(math.exp(-lam_away), 3),
        "cs_probability_away": round(math.exp(-lam_home), 3)
    }

def get_market_adjustments() -> dict:
    """Parses live fixtures into team-level shortcode market dictionaries."""
    events = fetch_live_odds()
    adjustments = {}
    for event in events:
        metrics = compute_implied_metrics(event)
        if metrics:
            h_short = TEAM_NAME_MAP.get(metrics["home_team"], metrics["home_team"])
            a_short = TEAM_NAME_MAP.get(metrics["away_team"], metrics["away_team"])
            adjustments[h_short] = {"xG": metrics["lambda_home"], "xGC": metrics["lambda_away"], "cs_prob": metrics["cs_probability_home"]}
            adjustments[a_short] = {"xG": metrics["lambda_away"], "xGC": metrics["lambda_home"], "cs_prob": metrics["cs_probability_away"]}
    return adjustments
