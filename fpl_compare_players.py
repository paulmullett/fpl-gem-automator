import os
import sys
import json
import requests
import math
import unicodedata

# Combined engine functions
from fpl_funcs import (
    estimate_xmins,
    get_base_ev,
    get_macro_ev,
    get_ensemble_ev,
    normalize_player,
    get_gameweek_state
)
from fpl_odds_engine import get_market_adjustments

# Environment Variables
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PLAYER_A_INPUT = os.environ.get("PLAYER_A", "").strip()
PLAYER_B_INPUT = os.environ.get("PLAYER_B", "").strip()
STATE_FILE_PATH = "fpl_state.json"

if not DISCORD_WEBHOOK_URL:
    print("CRITICAL ERROR: Missing DISCORD_WEBHOOK_URL secret.")
    sys.exit(1)

if not PLAYER_A_INPUT or not PLAYER_B_INPUT:
    print("CRITICAL ERROR: Both PLAYER_A and PLAYER_B inputs must be provided.")
    sys.exit(1)

# Helper: Load calibration weights from state engine
def load_calibration_weights():
    default_weights = {"xgi_weight": 0.70, "fdr_impact_factor": 0.10, "bench_discount": 0.01}
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                state = json.load(f)
                return state.get("calibration_weights", default_weights)
        except Exception:
            pass
    return default_weights

# Helper: Diacritic & Special Character Normalizer
def normalize_text(text):
    if not text:
        return ""
    # Strip accents (e.g., Estêvão -> Estevao)
    nfkd_form = unicodedata.normalize('NFD', text)
    clean_text = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Character substitutions
    char_map = {'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'ð': 'd', 'Ð': 'D', 'ß': 'ss'}
    for char, repl in char_map.items():
        clean_text = clean_text.replace(char, repl)
    return clean_text.lower().strip()

# Alias dictionary for common shorthand names
COMMON_ALIASES = {
    "kdb": "de bruyne",
    "vvd": "van dijk",
    "trent": "alexander-arnold",
    "taa": "alexander-arnold",
    "bruno": "fernandes"
}

def resolve_player(search_term, players):
    raw_term = search_term.strip()
    norm_term = normalize_text(raw_term)
    
    # Check alias lookup
    if norm_term in COMMON_ALIASES:
        norm_term = COMMON_ALIASES[norm_term]

    # 1. Exact match on normalized web_name or full_name
    for p in players.values():
        p_web = normalize_text(p["name"])
        p_full = normalize_text(p.get("full_name", ""))
        if norm_term == p_web or norm_term == p_full:
            return p, None

    # 2. Substring matching
    candidates = []
    for p in players.values():
        p_web = normalize_text(p["name"])
        p_full = normalize_text(p.get("full_name", ""))
        if norm_term in p_web or norm_term in p_full:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0], None
    elif len(candidates) > 1:
        # Check if any single candidate has an exact web_name match
        exact_web = [c for c in candidates if normalize_text(c["name"]) == norm_term]
        if len(exact_web) == 1:
            return exact_web[0], None
        
        cand_list = ", ".join([f"**{c['name']}** ({c['team']})" for c in candidates[:5]])
        return None, f"Multiple players matched '{raw_term}': {cand_list}. Please refine search."

    return None, f"Could not resolve player '{raw_term}'. Check spelling."

def fetch_data_and_compare():
    headers = {"User-Agent": "FPL-Comparison-Tool/3.0"}
    
    # Fetch Bootstrap Data
    try:
        resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if resp.status_code != 200:
            print("CRITICAL ERROR: Failed to connect to FPL API.")
            sys.exit(1)
        bootstrap_data = resp.json()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}

    # Parse and normalize all players
    players = {}
    for raw_p in bootstrap_data.get("elements", []):
        p = normalize_player(raw_p, teams, element_types)
        players[p["id"]] = p

    # Resolve input players
    p1, err1 = resolve_player(PLAYER_A_INPUT, players)
    p2, err2 = resolve_player(PLAYER_B_INPUT, players)

    if err1 or err2:
        err_msg = "### ⚠️ PLAYER RESOLUTION FAILURE\n"
        if err1: err_msg += f"• {err1}\n"
        if err2: err_msg += f"• {err2}\n"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": err_msg})
        sys.exit(1)

    # Fetch live odds and fixtures
    market_data = get_market_adjustments()
    weights = load_calibration_weights()

    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        fixtures_data = fixtures_resp.json() if fixtures_resp.status_code == 200 else []
    except Exception:
        fixtures_data = []

    # Calculate fixture difficulty metrics
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 4:
            ta, th = f.get("team_a"), f.get("team_h")
            if ta in team_fdr_sum:
                team_fdr_sum[ta] += f.get("team_a_difficulty", 3)
                team_fdr_count[ta] += 1
            if th in team_fdr_sum:
                team_fdr_sum[th] += f.get("team_h_difficulty", 3)
                team_fdr_count[th] += 1
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    # Quantitative Evaluation
    xmins1 = estimate_xmins(p1)
    ev_1gw_1 = get_ensemble_ev(p1, {}, market_data, weights)
    macro_4gw_1 = get_macro_ev(p1, team_avg_fdr, weights, {})
    val_1gw_1 = ev_1gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0
    val_4gw_1 = macro_4gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0

    xmins2 = estimate_xmins(p2)
    ev_1gw_2 = get_ensemble_ev(p2, {}, market_data, weights)
    macro_4gw_2 = get_macro_ev(p2, team_avg_fdr, weights, {})
    val_1gw_2 = ev_1gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0
    val_4gw_2 = macro_4gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0

    # Model Recommendation Summary (1-2 lines)
    diff_1gw = abs(ev_1gw_1 - ev_1gw_2)
    diff_4gw = abs(macro_4gw_1 - macro_4gw_2)

    if macro_4gw_1 > macro_4gw_2:
        rec = f"{p1['name']} projects higher structural value (+{diff_1gw:.2f} 1-GW EV | +{diff_4gw:.2f} 4-GW xP)."
    elif macro_4gw_2 > macro_4gw_1:
        rec = f"{p2['name']} projects higher structural value (+{diff_1gw:.2f} 1-GW EV | +{diff_4gw:.2f} 4-GW xP)."
    else:
        rec = "Both assets are mathematically deadlocked across short and medium-term horizons."

    if p1["pos"] != p2["pos"]:
        rec += f" [Cross-Position Audit: {p1['pos']} vs. {p2['pos']}]"

    # Monospaced ASCII Output Formatting
    p1_head = f"{p1['name']} ({p1['team']})"
    p2_head = f"{p2['name']} ({p2['team']})"

    def fmt_row(lbl, val1, val2):
        return f"{lbl:<23} | {str(val1):<25} | {str(val2):<25}\n"

    ascii_table = (
        "================================================================================\n"
        + fmt_row("METRIC", p1_head, p2_head)
        + "--------------------------------------------------------------------------------\n"
        + fmt_row("Position", p1["pos"], p2["pos"])
        + fmt_row("Cost", f"£{p1['cost']:.1f}m", f"£{p2['cost']:.1f}m")
        + fmt_row("Ownership", f"{p1['own']:.1f}%", f"{p2['own']:.1f}%")
        + fmt_row("Proj. Expected Mins", f"{xmins1:.1f}m", f"{xmins2:.1f}m")
        + fmt_row("Adjusted xGI / 90", f"{p1['xgi_90']:.2f}", f"{p2['xgi_90']:.2f}")
        + "--------------------------------------------------------------------------------\n"
        + fmt_row("1-GW EV (Odds-Adjusted)", f"{ev_1gw_1:.2f} pts", f"{ev_1gw_2:.2f} pts")
        + fmt_row("4-GW Horizon xP (Total)", f"{macro_4gw_1:.2f} pts", f"{macro_4gw_2:.2f} pts")
        + fmt_row("1-GW Value (EV / £m)", f"{val_1gw_1:.2f} pts/£m", f"{val_1gw_2:.2f} pts/£m")
        + fmt_row("4-GW Value (xP / £m)", f"{val_4gw_1:.2f} pts/£m", f"{val_4gw_2:.2f} pts/£m")
        + "================================================================================\n"
        + f"Model Recommendation: {rec}\n"
        + "================================================================================\n"
    )

    discord_payload = f"**[Multi-Model Player Head-to-Head Audit]**\n```text\n{ascii_table}```"
    requests.post(DISCORD_WEBHOOK_URL, json={"content": discord_payload})
    print("Head-to-head comparison posted successfully to Discord.")

if __name__ == "__main__":
    fetch_data_and_compare()
