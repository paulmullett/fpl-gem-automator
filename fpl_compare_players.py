"""
fpl_compare_players.py — Head-to-Head Player Audit Tool

Executes side-by-side player comparisons across 1-GW EV and 8-GW Deep-Tree Horizon xP.
Features:
- Unicode NFD decomposition to normalize accents (Estêvão -> Estevao, Ødegaard -> Odegaard).
- Alias dictionary mapping shorthand inputs (KDB, TAA, VVD, Bruno).
- Cross-position auditing support (e.g., DEF vs MID).
- 8-GW Deep-Tree Horizon & Market Odds Integration.
- Stochastic Risk Posture Gating (SHIELD / CHASE / NEUTRAL).
- Formatted monospaced YAML table delivery directly to Discord.
"""

import os
import sys
import json
import requests
import unicodedata

from fpl_funcs import (
    estimate_xmins, get_base_ev, get_macro_ev, 
    get_ensemble_ev, normalize_player, get_gameweek_state,
    get_live_price_deltas
)
from fpl_odds_engine import get_market_adjustments

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PLAYER_COMPARE_INPUT = os.environ.get("PLAYER_COMPARE", "")
# Parse comma-separated input from GitHub Actions
if "," in PLAYER_COMPARE_INPUT:
    parts = PLAYER_COMPARE_INPUT.split(",")
    PLAYER_A_INPUT = parts[0].strip()
    PLAYER_B_INPUT = parts[1].strip()
else:
    PLAYER_A_INPUT = os.environ.get("PLAYER_A", "").strip()
    PLAYER_B_INPUT = os.environ.get("PLAYER_B", "").strip()
RISK_POSTURE = os.environ.get("RISK_POSTURE", "NEUTRAL").upper()
STATE_FILE_PATH = "fpl_state.json"

if not DISCORD_WEBHOOK_URL or not PLAYER_A_INPUT or not PLAYER_B_INPUT:
    print("CRITICAL ERROR: Missing DISCORD_WEBHOOK_URL secret or player inputs.")
    sys.exit(1)

def load_calibration_weights():
    default_weights = {"xgi_weight": 0.70, "fdr_impact_factor": 0.10, "bench_discount": 0.01}
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                return json.load(f).get("calibration_weights", default_weights)
        except Exception: pass
    return default_weights

def normalize_text(text: str) -> str:
    """Strips accents and normalizes special characters (Ø, Æ, ð, etc.)."""
    if not text: return ""
    nfkd_form = unicodedata.normalize('NFD', text)
    clean_text = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    char_map = {'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'ð': 'd', 'Ð': 'D', 'ß': 'ss'}
    for char, repl in char_map.items(): clean_text = clean_text.replace(char, repl)
    return clean_text.lower().strip()

COMMON_ALIASES = {
    "kdb": "de bruyne", 
    "vvd": "van dijk", 
    "trent": "alexander-arnold", 
    "taa": "alexander-arnold", 
    "bruno": "fernandes"
}

def resolve_player(search_term: str, players: dict):
    """Resolves player search term using exact, alias, or substring matching."""
    norm_term = normalize_text(search_term)
    if norm_term in COMMON_ALIASES: norm_term = COMMON_ALIASES[norm_term]

    for p in players.values():
        if norm_term in [normalize_text(p["name"]), normalize_text(p.get("full_name", ""))]:
            return p, None

    candidates = [p for p in players.values() if norm_term in normalize_text(p["name"]) or norm_term in normalize_text(p.get("full_name", ""))]
    if len(candidates) == 1: return candidates[0], None
    elif len(candidates) > 1:
        exact_web = [c for c in candidates if normalize_text(c["name"]) == norm_term]
        if len(exact_web) == 1: return exact_web[0], None
        cand_list = ", ".join([f"**{c['name']}** ({c['team']})" for c in candidates[:5]])
        return None, f"Multiple players matched '{search_term}': {cand_list}."

    return None, f"Could not resolve player '{search_term}'."

def fetch_data_and_compare():
    headers = {"User-Agent": "FPL-Comparison-Tool/4.0"}
    try:
        resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if resp.status_code != 200: sys.exit(1)
        bootstrap_data = resp.json()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}"); sys.exit(1)

    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}

    players = {raw_p["id"]: normalize_player(raw_p, teams, element_types) for raw_p in bootstrap_data.get("elements", [])}

    price_deltas = get_live_price_deltas(players)
    for pid, p in players.items():
        p["price_delta_prob"] = price_deltas.get(pid, 0.0)

    p1, err1 = resolve_player(PLAYER_A_INPUT, players)
    p2, err2 = resolve_player(PLAYER_B_INPUT, players)

    if err1 or err2:
        err_msg = f"### ⚠️ PLAYER RESOLUTION FAILURE\n• {err1 or ''}\n• {err2 or ''}"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": err_msg}); sys.exit(1)

    market_data = get_market_adjustments()
    weights = load_calibration_weights()

    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        fixtures_data = fixtures_resp.json() if fixtures_resp.status_code == 200 else []
    except Exception: fixtures_data = []

    # 8-GW Deep-Tree Fixture Horizon Scanning
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 8:
            ta, th = f.get("team_a"), f.get("team_h")
            if ta in team_fdr_sum: team_fdr_sum[ta] += f.get("team_a_difficulty", 3); team_fdr_count[ta] += 1
            if th in team_fdr_sum: team_fdr_sum[th] += f.get("team_h_difficulty", 3); team_fdr_count[th] += 1
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    # Player 1 Model Execution
    xmins1 = estimate_xmins(p1)
    ev_1gw_1 = get_ensemble_ev(p1, {}, market_data, weights, RISK_POSTURE)
    macro_8gw_1 = get_macro_ev(p1, team_avg_fdr, weights, {}, market_data, 8, RISK_POSTURE)
    val_1gw_1 = ev_1gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0
    val_8gw_1 = macro_8gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0

    # Player 2 Model Execution
    xmins2 = estimate_xmins(p2)
    ev_1gw_2 = get_ensemble_ev(p2, {}, market_data, weights, RISK_POSTURE)
    macro_8gw_2 = get_macro_ev(p2, team_avg_fdr, weights, {}, market_data, 8, RISK_POSTURE)
    val_1gw_2 = ev_1gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0
    val_8gw_2 = macro_8gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0

    diff_1gw, diff_8gw = abs(ev_1gw_1 - ev_1gw_2), abs(macro_8gw_1 - macro_8gw_2)
    winner = p1['name'] if macro_8gw_1 > macro_8gw_2 else (p2['name'] if macro_8gw_2 > macro_8gw_1 else "Equal")
    rec = f"{winner} projects higher structural value (+{diff_1gw:.2f} 1-GW EV | +{diff_8gw:.2f} 8-GW xP)."
    if p1["pos"] != p2["pos"]: rec += f" [Cross-Position Audit: {p1['pos']} vs. {p2['pos']}]"

    def price_trend_str(prob):
        if prob > 0.0: return "Rising (+£0.1m)"
        if prob < 0.0: return "Falling (-£0.1m)"
        return "Stable (£0.0m)"

    p1_str = f"{p1['name']} ({p1['team']})"
    p2_str = f"{p2['name']} ({p2['team']})"

    def fmt_row(lbl, v1, v2):
        return f" {lbl:<24} | {str(v1):<25} | {str(v2):<25}\n"

    ascii_table = "```yaml\n"
    ascii_table += "================================================================================\n"
    ascii_table += "               HEAD-TO-HEAD QUANTITATIVE PLAYER AUDIT (GW " + str(target_gw) + ")\n"
    ascii_table += "================================================================================\n"
    ascii_table += fmt_row("METRIC", p1_str, p2_str)
    ascii_table += "--------------------------------------------------------------------------------\n"
    ascii_table += fmt_row("Position", p1["pos"], p2["pos"])
    ascii_table += fmt_row("Cost", f"£{p1['cost']:.1f}m", f"£{p2['cost']:.1f}m")
    ascii_table += fmt_row("Ownership", f"{p1['own']:.1f}%", f"{p2['own']:.1f}%")
    ascii_table += fmt_row("Price Delta Trend", price_trend_str(p1["price_delta_prob"]), price_trend_str(p2["price_delta_prob"]))
    ascii_table += fmt_row("Proj. Expected Mins", f"{xmins1:.1f} xMins", f"{xmins2:.1f} xMins")
    ascii_table += fmt_row("Adjusted xGI / 90", f"{p1['xgi_90']:.2f}", f"{p2['xgi_90']:.2f}")
    ascii_table += "--------------------------------------------------------------------------------\n"
    ascii_table += fmt_row("1-GW EV (Odds-Adjusted)", f"{ev_1gw_1:.2f} pts", f"{ev_1gw_2:.2f} pts")
    ascii_table += fmt_row("8-GW Deep Horizon xP", f"{macro_8gw_1:.2f} pts", f"{macro_8gw_2:.2f} pts")
    ascii_table += fmt_row("1-GW Value (EV / £m)", f"{val_1gw_1:.2f} pts/£m", f"{val_1gw_2:.2f} pts/£m")
    ascii_table += fmt_row("8-GW Value (xP / £m)", f"{val_8gw_1:.2f} pts/£m", f"{val_8gw_2:.2f} pts/£m")
    ascii_table += "================================================================================\n"
    ascii_table += f" Risk Posture: {RISK_POSTURE}\n"
    ascii_table += f" Model Recommendation: {rec}\n"
    ascii_table += "================================================================================\n"
    ascii_table += "```"

    requests.post(DISCORD_WEBHOOK_URL, json={"content": ascii_table})

if __name__ == "__main__":
    fetch_data_and_compare()
