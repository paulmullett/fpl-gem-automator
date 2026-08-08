"""
fpl_compare_players.py — Head-to-Head Player Audit Tool (Multi-Period Matrix Aligned)

Executes side-by-side player comparisons across 1-GW EV, 8-GW Deep-Tree Horizon xP, 
week-by-week EV/xMins matrices, and stochastic risk bounds from ml_projections.json.
"""

import os
import sys
import json
import requests
import unicodedata
import re

from fpl_funcs import (
    estimate_xmins, normalize_player, get_gameweek_state, get_live_price_deltas
)

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
PLAYER_COMPARE_INPUT = os.environ.get("PLAYER_COMPARE", "").strip()

# Flexible Input Parsing: supports comma, 'vs', 'v', or space-delimited inputs
if "," in PLAYER_COMPARE_INPUT:
    parts = PLAYER_COMPARE_INPUT.split(",")
    PLAYER_A_INPUT = parts[0].strip()
    PLAYER_B_INPUT = parts[1].strip()
elif " vs " in PLAYER_COMPARE_INPUT.lower():
    parts = re.split(r'\svs\s', PLAYER_COMPARE_INPUT, flags=re.IGNORECASE)
    PLAYER_A_INPUT = parts[0].strip()
    PLAYER_B_INPUT = parts[1].strip()
elif " v " in PLAYER_COMPARE_INPUT.lower():
    parts = re.split(r'\sv\s', PLAYER_COMPARE_INPUT, flags=re.IGNORECASE)
    PLAYER_A_INPUT = parts[0].strip()
    PLAYER_B_INPUT = parts[1].strip()
else:
    parts = PLAYER_COMPARE_INPUT.split()
    if len(parts) >= 2:
        PLAYER_A_INPUT = parts[0].strip()
        PLAYER_B_INPUT = parts[1].strip()
    else:
        PLAYER_A_INPUT = os.environ.get("PLAYER_A", "").strip()
        PLAYER_B_INPUT = os.environ.get("PLAYER_B", "").strip()

RISK_POSTURE = os.environ.get("RISK_POSTURE", "NEUTRAL").upper()

if not DISCORD_WEBHOOK_URL or not PLAYER_A_INPUT or not PLAYER_B_INPUT:
    print(f"CRITICAL ERROR: Missing DISCORD_WEBHOOK_URL secret or valid player inputs. Received: '{PLAYER_COMPARE_INPUT}'")
    sys.exit(1)

def send_to_discord(webhook_url, content):
    """Delivers output to Discord with payload safety checks and error handling."""
    if len(content) <= 1900:
        resp = requests.post(webhook_url, json={"content": content})
        if resp.status_code >= 400:
            print(f"ERROR: Discord Webhook rejected payload (HTTP {resp.status_code}): {resp.text}")
        else:
            print("Head-to-head audit delivered successfully to Discord.")
    else:
        lines = content.split("\n")
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > 1850:
                resp = requests.post(webhook_url, json={"content": chunk})
                if resp.status_code >= 400:
                    print(f"ERROR: Discord Webhook rejected chunk (HTTP {resp.status_code}): {resp.text}")
                chunk = line + "\n"
            else:
                chunk += line + "\n"
        if chunk.strip():
            resp = requests.post(webhook_url, json={"content": chunk})
            if resp.status_code >= 400:
                print(f"ERROR: Discord Webhook rejected final chunk (HTTP {resp.status_code}): {resp.text}")
            else:
                print("Head-to-head audit delivered successfully to Discord.")

def normalize_text(text: str) -> str:
    """Strips accents and normalizes special characters (Ø, Æ, ð, etc.)."""
    if not text: return ""
    nfkd_form = unicodedata.normalize('NFD', str(text))
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

def resolve_player(search_term: str, raw_elements: list, players: dict, ml_proj_data: dict):
    """Resolves player search term using exact, alias, or substring matching."""
    norm_term = normalize_text(search_term)
    if not norm_term:
        return None, None, f"Empty search term provided."
    if norm_term in COMMON_ALIASES:
        norm_term = COMMON_ALIASES[norm_term]

    # Exact Match Pass
    for raw_p in raw_elements:
        pid = raw_p["id"]
        pid_str = str(pid)
        web_name = normalize_text(raw_p.get("web_name", ""))
        first_name = normalize_text(raw_p.get("first_name", ""))
        second_name = normalize_text(raw_p.get("second_name", ""))
        full_name = f"{first_name} {second_name}".strip()
        ml_name = normalize_text(ml_proj_data.get(pid_str, {}).get("name", ""))

        if norm_term in [web_name, full_name, ml_name]:
            return players[pid], pid_str, None

    # Substring Match Pass
    candidates = []
    for raw_p in raw_elements:
        pid = raw_p["id"]
        pid_str = str(pid)
        web_name = normalize_text(raw_p.get("web_name", ""))
        first_name = normalize_text(raw_p.get("first_name", ""))
        second_name = normalize_text(raw_p.get("second_name", ""))
        full_name = f"{first_name} {second_name}".strip()
        ml_name = normalize_text(ml_proj_data.get(pid_str, {}).get("name", ""))

        if (norm_term in web_name) or (norm_term in full_name) or (norm_term in ml_name):
            candidates.append((players[pid], pid_str))

    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], None
    elif len(candidates) > 1:
        exact_web = [c for c in candidates if normalize_text(c[0]["name"]) == norm_term]
        if len(exact_web) == 1:
            return exact_web[0][0], exact_web[0][1], None
        cand_list = ", ".join([f"**{c[0]['name']}** ({c[0]['team']})" for c in candidates[:5]])
        return None, None, f"Multiple players matched '{search_term}': {cand_list}."

    return None, None, f"Could not resolve player '{search_term}'."

def fetch_data_and_compare():
    headers = {"User-Agent": "FPL-Comparison-Tool/5.0"}
    try:
        resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if resp.status_code != 200: sys.exit(1)
        bootstrap_data = resp.json()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}"); sys.exit(1)

    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    raw_elements = bootstrap_data.get("elements", [])
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}

    players = {raw_p["id"]: normalize_player(raw_p, teams, element_types) for raw_p in raw_elements}

    from ml_engine.data_ingestion import fetch_price_targets
    
    oracle_targets = fetch_price_targets()
    price_deltas = get_live_price_deltas(players, oracle_targets)
    
    for pid, p in players.items():
        p["price_delta_prob"] = price_deltas.get(pid, 0.0)

    ml_proj_data = {}
    if os.path.exists("ml_projections.json"):
        try:
            with open("ml_projections.json", "r") as f:
                ml_proj_data = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load ml_projections.json: {e}")

    p1, pid1_str, err1 = resolve_player(PLAYER_A_INPUT, raw_elements, players, ml_proj_data)
    p2, pid2_str, err2 = resolve_player(PLAYER_B_INPUT, raw_elements, players, ml_proj_data)

    if err1 or err2:
        err_msg = f"### ⚠️ PLAYER RESOLUTION FAILURE\n• {err1 or ''}\n• {err2 or ''}"
        send_to_discord(DISCORD_WEBHOOK_URL, err_msg)
        sys.exit(1)

    # Extract Metrics for Player 1
    p1_proj = ml_proj_data.get(pid1_str, {})
    ev_1gw_1 = float(p1_proj.get("ml_ev_1gw", p1.get("ep_next", 0.0)))
    macro_8gw_1 = float(p1_proj.get("ml_ev_8gw", ev_1gw_1 * 8))
    xmins1 = float(p1_proj.get("ml_xmins", estimate_xmins(p1)))
    floor1 = float(p1_proj.get("mc_floor_ev", ev_1gw_1 * 0.7))
    ceil1 = float(p1_proj.get("mc_ceiling_ev", ev_1gw_1 * 1.3))
    ev_matrix_1 = p1_proj.get("ml_ev_matrix", [ev_1gw_1] * 8)
    xmins_matrix_1 = p1_proj.get("ml_xmins_matrix", [xmins1] * 8)
    val_1gw_1 = ev_1gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0
    val_8gw_1 = macro_8gw_1 / p1["cost"] if p1["cost"] > 0 else 0.0

    # Extract Metrics for Player 2
    p2_proj = ml_proj_data.get(pid2_str, {})
    ev_1gw_2 = float(p2_proj.get("ml_ev_1gw", p2.get("ep_next", 0.0)))
    macro_8gw_2 = float(p2_proj.get("ml_ev_8gw", ev_1gw_2 * 8))
    xmins2 = float(p2_proj.get("ml_xmins", estimate_xmins(p2)))
    floor2 = float(p2_proj.get("mc_floor_ev", ev_1gw_2 * 0.7))
    ceil2 = float(p2_proj.get("mc_ceiling_ev", ev_1gw_2 * 1.3))
    ev_matrix_2 = p2_proj.get("ml_ev_matrix", [ev_1gw_2] * 8)
    xmins_matrix_2 = p2_proj.get("ml_xmins_matrix", [xmins2] * 8)
    val_1gw_2 = ev_1gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0
    val_8gw_2 = macro_8gw_2 / p2["cost"] if p2["cost"] > 0 else 0.0

    diff_1gw = abs(ev_1gw_1 - ev_1gw_2)
    diff_8gw = abs(macro_8gw_1 - macro_8gw_2)
    winner = p1['name'] if macro_8gw_1 > macro_8gw_2 else (p2['name'] if macro_8gw_2 > macro_8gw_1 else "Equal")
    rec = f"{winner} projects higher 8-GW structural value (+{diff_1gw:.2f} 1-GW EV | +{diff_8gw:.2f} 8-GW xP)."
    if p1["pos"] != p2["pos"]: 
        rec += f" [{p1['pos']} vs. {p2['pos']}]"

    def price_trend_str(prob):
        if prob > 0.0: return "Rising (+£0.1m)"
        if prob < 0.0: return "Falling (-£0.1m)"
        return "Stable (£0.0m)"

    p1_str = f"{p1['name']} ({p1['team']})"
    p2_str = f"{p2['name']} ({p2['team']})"

    def fmt_row(lbl, v1, v2):
        return f" {lbl:<22} | {str(v1):<24} | {str(v2):<24}\n"

    def fmt_matrix(mat):
        return "[" + ", ".join([f"{x:.1f}" for x in mat[:8]]) + "]"

    ascii_table = "```yaml\n"
    ascii_table += "================================================================================\n"
    ascii_table += f" HEAD-TO-HEAD QUANTITATIVE PLAYER AUDIT (GW {target_gw})\n"
    ascii_table += "================================================================================\n"
    ascii_table += fmt_row("METRIC", p1_str, p2_str)
    ascii_table += "--------------------------------------------------------------------------------\n"
    ascii_table += fmt_row("Position", p1["pos"], p2["pos"])
    ascii_table += fmt_row("Cost", f"£{p1['cost']:.1f}m", f"£{p2['cost']:.1f}m")
    ascii_table += fmt_row("Ownership", f"{p1['own']:.1f}%", f"{p2['own']:.1f}%")
    ascii_table += fmt_row("Top 10k EO", f"{p1_proj.get('top_10k_eo', p1['own']):.1f}%", f"{p2_proj.get('top_10k_eo', p2['own']):.1f}%")
    ascii_table += fmt_row("Price Delta Trend", price_trend_str(p1["price_delta_prob"]), price_trend_str(p2["price_delta_prob"]))
    ascii_table += fmt_row("Baseline xMins", f"{xmins1:.1f}", f"{xmins2:.1f}")
    ascii_table += fmt_row("Adjusted xGI / 90", f"{p1['xgi_90']:.2f}", f"{p2['xgi_90']:.2f}")
    ascii_table += "--------------------------------------------------------------------------------\n"
    ascii_table += fmt_row("1-GW Expected Yield", f"{ev_1gw_1:.2f} pts", f"{ev_1gw_2:.2f} pts")
    ascii_table += fmt_row("8-GW Horizon xP", f"{macro_8gw_1:.2f} pts", f"{macro_8gw_2:.2f} pts")
    ascii_table += fmt_row("Stochastic Floor/Ceil", f"{floor1:.1f} / {ceil1:.1f}", f"{floor2:.1f} / {ceil2:.1f}")
    ascii_table += fmt_row("1-GW Value (EV/£m)", f"{val_1gw_1:.2f}", f"{val_1gw_2:.2f}")
    ascii_table += fmt_row("8-GW Value (xP/£m)", f"{val_8gw_1:.2f}", f"{val_8gw_2:.2f}")
    ascii_table += "--------------------------------------------------------------------------------\n"
    ascii_table += fmt_row("8-GW EV Matrix", fmt_matrix(ev_matrix_1), fmt_matrix(ev_matrix_2))
    ascii_table += fmt_row("8-GW xMins Matrix", fmt_matrix(xmins_matrix_1), fmt_matrix(xmins_matrix_2))
    ascii_table += "================================================================================\n"
    ascii_table += f" Posture: {RISK_POSTURE} | Rec: {rec}\n"
    ascii_table += "================================================================================\n"
    ascii_table += "```"

    send_to_discord(DISCORD_WEBHOOK_URL, ascii_table)

if __name__ == "__main__":
    fetch_data_and_compare()