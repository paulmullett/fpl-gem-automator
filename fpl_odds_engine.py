import os
import requests
import math
from scipy.optimize import fsolve

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
BASE_URL = "https://api.v4.the-odds-api.com/v4/sports/soccer_epl/odds/"

def fetch_live_odds():
    if not ODDS_API_KEY:
        print("WARNING: Missing ODDS_API_KEY. Falling back to native FPL expected models.")
        return {}
    
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching odds: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception during odds API query: {e}")
    return {}

def remove_vig(odds_list):
    """Removes bookmaker margin via proportional normalization."""
    implied_probs = [1.0 / d for d in odds_list if d > 1.0]
    total = sum(implied_probs)
    if total <= 0:
        return [1.0 / len(odds_list)] * len(odds_list)
    return [p / total for p in implied_probs]

def compute_implied_metrics(match_event):
    """
    Converts decimal odds for H2H and Totals (2.5) into 
    expected goals (lambda_home, lambda_away) and clean sheet probabilities.
    """
    home_team = match_event.get("home_team")
    away_team = match_event.get("away_team")
    bookmakers = match_event.get("bookmakers", [])
    
    if not bookmakers:
        return None

    # Use consensus or first available major bookmaker (e.g., bet365 / Pinnacle)
    target_bookie = next((b for b in bookmakers if b["key"] in ["pinnacle", "bet365", "skybet"]), bookmakers[0])
    markets = {m["key"]: m["outcomes"] for m in target_bookie.get("markets", [])}

    h2h = markets.get("h2h", [])
    totals = markets.get("totals", [])

    if not h2h or not totals:
        return None

    # Parse H2H Decimal Odds
    h_odds = next((o["price"] for o in h2h if o["name"] == home_team), 2.0)
    a_odds = next((o["price"] for o in h2h if o["name"] == away_team), 2.0)
    d_odds = next((o["price"] for o in h2h if o["name"] == "Draw"), 3.5)

    clean_h2h = remove_vig([h_odds, d_odds, a_odds])
    p_home, p_draw, p_away = clean_h2h[0], clean_h2h[1], clean_h2h[2]

    # Parse Over/Under 2.5 Goals Line
    over_odds = next((o["price"] for o in totals if o["name"] == "Over" and o.get("point", 2.5) == 2.5), 1.9)
    under_odds = next((o["price"] for o in totals if o["name"] == "Under" and o.get("point", 2.5) == 2.5), 1.9)
    clean_totals = remove_vig([over_odds, under_odds])
    p_under_25 = clean_totals[1]

    # Solve total expected goals (lambda_total) from Under 2.5 probability via Poisson CDF:
    # P(X <= 2) = exp(-lam) * (1 + lam + lam^2 / 2)
    def objective(lam):
        if lam[0] <= 0:
            return 1.0
        p_under_model = math.exp(-lam[0]) * (1.0 + lam[0] + (lam[0]**2) / 2.0)
        return p_under_model - p_under_25

    lam_total_initial = [2.5]
    lam_total = fsolve(objective, lam_total_initial)[0]
    lam_total = max(0.5, min(5.0, lam_total))

    # Split lambda_total between Home and Away using win probability ratio mapping
    # Using a standard logistic strength differential assumption
    advantage_factor = max(0.1, p_home / max(0.01, p_away))
    fraction_home = advantage_factor / (1.0 + advantage_factor)
    
    # Bounded proportional split
    fraction_home = max(0.2, min(0.8, fraction_home))
    
    lam_home = lam_total * fraction_home
    lam_away = lam_total * (1.0 - fraction_home)

    # Poisson Clean Sheet Probabilities: P(X = 0) = exp(-lambda)
    cs_home = math.exp(-lam_away)
    cs_away = math.exp(-lam_home)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
        "cs_probability_home": round(cs_home, 3),
        "cs_probability_away": round(cs_away, 3)
    }

def get_market_adjustments():
    events = fetch_live_odds()
    adjustments = {}
    for event in events:
        metrics = compute_implied_metrics(event)
        if metrics:
            adjustments[metrics["home_team"]] = {
                "xG": metrics["lambda_home"],
                "xGC": metrics["lambda_away"],
                "cs_prob": metrics["cs_probability_home"]
            }
            adjustments[metrics["away_team"]] = {
                "xG": metrics["lambda_away"],
                "xGC": metrics["lambda_home"],
                "cs_prob": metrics["cs_probability_away"]
            }
    return adjustments

if __name__ == "__main__":
    print(get_market_adjustments())
