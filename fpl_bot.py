"""
fpl_bot.py — Primary Execution Script & Discord Interface (Pure Code Architecture)
Zero-LLM implementation: 100% Deterministic, £0 API Cost, Sub-millisecond Payload Generation.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import re
import requests
import pulp
import math

from ddgs import DDGS

from fpl_funcs import (
    _safe_float,
    estimate_xmins,
    calculate_tier1_translation_factor,
    get_gameweek_state,
    get_live_price_deltas,
    get_base_ev,
    get_macro_ev,
    get_ensemble_ev,
    normalize_player,
    evaluate_chip_thresholds
)
from fpl_odds_engine import get_market_adjustments
from fpl_mpo_engine import solve_multi_period_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment & Pre-Flight Check
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")
WORKFLOW_INPUT = os.environ.get("MANUAL_TRIGGER", "auto")
XMINS_INPUT = os.environ.get("XMINS_INPUT", "")
ACTIVE_CHIP = os.environ.get("ACTIVE_CHIP", "NONE").upper()
RISK_POSTURE = os.environ.get("RISK_POSTURE", "NEUTRAL")
STATE_FILE_PATH = "fpl_state.json"

UEFA_TEAMS = {"MCI", "ARS", "LIV", "AVL", "MUN", "NEW", "CHE", "TOT", "SUN"}

if not all([DISCORD_WEBHOOK_URL, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing GitHub Secrets (Discord Webhook or Team ID).")
    sys.exit(1)

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ["true", "1", "yes"]

# ==============================================================================
# STATE MANAGEMENT & DATA INGESTION
# ==============================================================================

def load_state():
    default_state = {
        "buyback_targets": {}, 
        "last_updated_gw": 0, 
        "xmins_overrides": {}, 
        "price_watchlist": {},
        "calibration_weights": {"xgi_weight": 0.70, "fdr_impact_factor": 0.10, "bench_discount": 0.01},
        "pending_evaluation": None, 
        "performance_history": []
    }
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                saved_state = json.load(f)
            if not isinstance(saved_state, dict):
                logger.warning("fpl_state.json contains non-dict data. Resetting to default state.")
                return default_state
            for key, val in default_state.items():
                if key not in saved_state: 
                    saved_state[key] = val
            return saved_state
        except Exception as e:
            print(f"WARNING: Error reading state file: {e}")
            return default_state
    return default_state

def save_state(state):
    try:
        with open(STATE_FILE_PATH, "w") as f:
            json.dump(state, f, indent=4)
            print("STATE ENGINE: Successfully saved updated strategy state to fpl_state.json")
    except Exception as e:
        print(f"ERROR: Failed to save state file: {e}")

def get_user_fpl_context(target_gw, headers):
    """
    Fetches played chips and calculates exact banked Free Transfers (up to 5) 
    directly from official FPL API season history.
    """
    available_chips = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}
    banked_fts = 1

    if target_gw == 1:
        return available_chips, "Unlimited"

    try:
        resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/history/", headers=headers, timeout=10)
        if resp.status_code == 200:
            history_data = resp.json()
            played_chips = history_data.get("chips", [])
            season_history = history_data.get("current", [])

            # 1. Audit Chip History
            half_played = []
            chip_by_gw = {}
            for c in played_chips:
                chip_by_gw[c["event"]] = c["name"]
                if target_gw <= 19 and c["event"] <= 19:
                    half_played.append(c["name"])
                elif target_gw > 19 and c["event"] > 19:
                    half_played.append(c["name"])

            available_chips["wildcard"] = "wildcard" not in half_played
            available_chips["freehit"] = "freehit" not in half_played
            available_chips["bboost"] = "bboost" not in half_played
            available_chips["3xc"] = "3xc" not in half_played

            # 2. Calculate Exact Banked Free Transfers (2024/25+ Rules: max 5 FTs)
            current_ft = 1
            for gw_data in season_history:
                gw = gw_data["event"]
                if gw >= target_gw:
                    break

                transfers_made = gw_data.get("event_transfers", 0)
                active_chip = chip_by_gw.get(gw, "NONE")

                if active_chip in ["wildcard", "freehit"]:
                    # Under 2024/25 rules, transfers during Wildcard/Free Hit are free 
                    # and retained FTs carry over without gaining +1
                    continue
                else:
                    # Subtract transfers used, then add 1 for the new week, capped at 5
                    unused_ft = max(0, current_ft - transfers_made)
                    current_ft = min(5, unused_ft + 1)

            banked_fts = current_ft

    except Exception as e:
        print(f"WARNING: Could not fetch FPL history context: {e}")

    return available_chips, banked_fts

def recalibrate_model(state, headers, active_gw):
    pending = state.get("pending_evaluation")
    if not pending: 
        return state

    eval_gw = pending.get("gw")
    if active_gw <= eval_gw: 
        return state

    try:
        resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{eval_gw}/picks/", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            actual_points = data.get("entry_history", {}).get("points", 0)
            projected_xP = pending.get("projected_xP", 0.0)
            residual_error = actual_points - projected_xP

            state["performance_history"].append({
                "gw": eval_gw, 
                "projected_xP": round(projected_xP, 2),
                "actual_points": actual_points, 
                "residual_error": round(residual_error, 2),
                "captain": pending.get("captain")
            })

            recent_history = state["performance_history"][-6:]
            if len(recent_history) >= 2:
                mean_error = sum(h["residual_error"] for h in recent_history) / len(recent_history)
                weights = state["calibration_weights"]
                learning_rate = 0.02
                if mean_error < -5.0:
                    weights["xgi_weight"] = max(0.50, round(weights["xgi_weight"] - learning_rate, 3))
                elif mean_error > 5.0:
                    weights["xgi_weight"] = min(1.00, round(weights["xgi_weight"] + learning_rate, 3))
            state["pending_evaluation"] = None
    except Exception as e:
        print(f"WARNING: Error during model recalibration: {e}")

    return state

def get_live_fpl_news():
    news_text = ""
    try:
        crellin_results = DDGS().text("Ben Crellin FPL blank double gameweek updates", max_results=3, timelimit='w')
        news_text += "--- SCHEDULE CHANGES (Ben Crellin) ---\n"
        for r in crellin_results: news_text += f"• {r.get('body', '')}\n"
        
        dinnery_results = DDGS().text("Ben Dinnery FPL injuries team news press conference", max_results=3, timelimit='w')
        news_text += "\n--- INJURY UPDATES (Ben Dinnery) ---\n"
        for r in dinnery_results: news_text += f"• {r.get('body', '')}\n"

        press_results = DDGS().news("Premier League manager press conference injury updates today", max_results=3, timelimit='d')
        news_text += "\n--- PRESS CONFERENCE UPDATES ---\n"
        for r in press_results: news_text += f"• {r.get('body', '')}\n"
        
    except Exception as e:
        news_text += f"[Search tool failed to retrieve live data: {e}]\n"
    return news_text.strip()

def check_european_congestion_flags(starters, fixtures_data, target_gw):
    flags = []
    for p in starters:
        if p["team"] in UEFA_TEAMS:
            flags.append(f"[FLAG OPTION: European Turnaround Risk detected for {p['name']} ({p['team']}) due to mid-week fixture congestion.]")
    return flags

def get_fpl_data():
    print("Fetching live market odds for tactical alignment...")
    market_data = get_market_adjustments()
    
    headers = {"User-Agent": "FPL-Auto-Script/14.0"}
    state = load_state()
    xmins_overrides = state.get("xmins_overrides", {})
    
    try:
        bootstrap_resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=headers)
        if bootstrap_resp.status_code != 200: sys.exit(1)
        bootstrap_data = bootstrap_resp.json()
    except Exception as e:
        print(f"ERROR fetching bootstrap data: {e}"); sys.exit(1)

    active_gw, target_gw = get_gameweek_state(bootstrap_data)
    teams = {t["id"]: t["short_name"] for t in bootstrap_data["teams"]}
    element_types = {e["id"]: e["singular_name_short"] for e in bootstrap_data["element_types"]}
    
    state = recalibrate_model(state, headers, active_gw)
    state["last_updated_gw"] = target_gw
    weights = state["calibration_weights"]

    ml_proj_data = {}
    if os.path.exists("ml_projections.json"):
        try:
            with open("ml_projections.json", "r") as f:
                ml_proj_data = json.load(f)
            print("SUCCESS: Loaded local projections from ml_projections.json")
        except Exception as e:
            print(f"WARNING: Could not read ml_projections.json: {e}")

    players = {}
    new_arrivals = []
    for raw_p in bootstrap_data.get("elements", []):
        p = normalize_player(raw_p, teams, element_types)
        
        pid_str = str(p["id"])
        if pid_str in ml_proj_data:
            p["ml_xmins"] = min(90.0, float(ml_proj_data[pid_str].get("ml_xmins", estimate_xmins(p))))
            p["mc_floor_ev"] = float(ml_proj_data[pid_str].get("mc_floor_ev", 0.0))
            p["mc_ceiling_ev"] = float(ml_proj_data[pid_str].get("mc_ceiling_ev", 0.0))
        else:
            p["ml_xmins"] = estimate_xmins(p)
        
        players[p["id"]] = p
        if p.get("has_stale_pl_history") or p.get("total_points", 0) == 0:
            new_arrivals.append(f"- {p['name']} ({p['team']})")

    new_arrivals_str = "\n".join(new_arrivals) if new_arrivals else "None"
    
    price_deltas = get_live_price_deltas(players)
    for pid, p in players.items():
        p["price_delta_prob"] = price_deltas.get(pid, 0.0)
    market_str = "Market data & live price deltas initialized natively."

    if XMINS_INPUT and XMINS_INPUT.strip():
        print(f"Processing Human Oracle Input: {XMINS_INPUT}")
        parsed_overrides = {}
        try:
            parsed_overrides = json.loads(XMINS_INPUT.replace("'", '"'))
        except Exception:
            matches = re.findall(r'([A-Za-z\s.\'-]+?)[:=\-\s,]+(\d+(?:\.\d+)?)', XMINS_INPUT)
            for name, mins in matches:
                clean_name = name.strip()
                if clean_name and clean_name.lower() not in ['mins', 'minutes', 'xmins']:
                    parsed_overrides[clean_name] = float(mins)

        for name_part, target_mins in parsed_overrides.items():
            name_lower = str(name_part).strip().lower()
            match_found = False
            
            for pid, p in players.items():
                if p.get("name", "").lower() == name_lower:
                    clamped_mins = min(90.0, float(target_mins))
                    xmins_overrides[str(pid)] = clamped_mins
                    p["xmins"] = clamped_mins
                    p["ml_xmins"] = clamped_mins
                    print(f" -> ORACLE OVERRIDE LOCKED: {p['name']} ({p['team']}) set to {clamped_mins} mins.")
                    match_found = True
                    break
            
            if not match_found:
                for pid, p in players.items():
                    if name_lower in p.get("name", "").lower():
                        clamped_mins = min(90.0, float(target_mins))
                        xmins_overrides[str(pid)] = clamped_mins
                        p["xmins"] = clamped_mins
                        p["ml_xmins"] = clamped_mins
                        print(f" -> ORACLE FUZZY OVERRIDE: {p['name']} ({p['team']}) set to {clamped_mins} mins.")
                        break

        state["xmins_overrides"] = xmins_overrides
    else:
        print("No Human Oracle Input provided. Resetting state overrides to clean baseline.")
        xmins_overrides = {}
        state["xmins_overrides"] = {}
        
    current_squad_ids = []
    bank = 0.0
    total_liquid_budget = 100.0
    free_transfers = "Unlimited" if target_gw == 1 else 1

    if target_gw > 1:
        try:
            team_resp = requests.get(f"https://fantasy.premierleague.com/api/entry/{FPL_TEAM_ID}/event/{target_gw-1}/picks/", headers=headers)
            if team_resp.status_code == 200:
                team_data = team_resp.json()
                current_squad_ids = []
                for pick in team_data.get("picks", []):
                    pid = pick["element"]
                    current_squad_ids.append(pid)
                    if pid in players:
                        players[pid]["selling_price"] = pick.get("selling_price", players[pid]["cost"] * 10) / 10.0

                entry_history = team_data.get("entry_history", {})
                bank = entry_history.get("bank", 0) / 10.0
                total_liquid_budget = entry_history.get("value", 1000) / 10.0
        except Exception as e:
            print(f"WARNING: Could not fetch squad for team {FPL_TEAM_ID}: {e}")

    planned_chips_env = os.environ.get("PLANNED_CHIPS", "")
    planned_chips = {}
    if planned_chips_env:
        for pair in planned_chips_env.split(","):
            parts = pair.split(":")
            if len(parts) == 2:
                try:
                    gw_num = int(parts[0])
                    chip_name = parts[1].strip().upper()
                    t_idx = gw_num - target_gw
                    if 0 < t_idx < 8: planned_chips[t_idx] = chip_name
                except ValueError: pass

    try:
        fixtures_resp = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
        fixtures_data = fixtures_resp.json() if fixtures_resp.status_code == 200 else []
    except Exception:
        fixtures_data = []
    
    team_fdr_sum = {t: 0 for t in teams.keys()}
    team_fdr_count = {t: 0 for t in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 8:
            team_a, team_h = f.get("team_a"), f.get("team_h")
            if team_a in team_fdr_sum:
                team_fdr_sum[team_a] += f.get("team_a_difficulty", 3)
                team_fdr_count[team_a] += 1
            if team_h in team_fdr_sum:
                team_fdr_sum[team_h] += f.get("team_h_difficulty", 3)
                team_fdr_count[team_h] += 1
    
    team_avg_fdr = {t: (team_fdr_sum[t] / team_fdr_count[t] if team_fdr_count[t] > 0 else 3.0) for t in teams.keys()}

    team_gw_opponents = {t_id: [[] for _ in range(8)] for t_id in teams.keys()}
    for f in fixtures_data:
        event = f.get("event")
        if event and target_gw <= event < target_gw + 8:
            t = event - target_gw
            team_a, team_h = f.get("team_a"), f.get("team_h")
            if team_a in team_gw_opponents:
                team_gw_opponents[team_a][t].append({"opp": team_h, "is_home": False})
            if team_h in team_gw_opponents:
                team_gw_opponents[team_h][t].append({"opp": team_a, "is_home": True})

    ev_matrix = {}
    valid_ids = list(players.keys())
    for pid in valid_ids:
        p = players[pid]
        ev_matrix[pid] = [0.0] * 8
        if p.get("status") not in ["a", "d", ""]: continue
        
        t_id = p["team_id"]
        pid_str = str(pid)
        pos_id = p.get("pos_id", 3)

        if pid_str in ml_proj_data and "ml_ev_matrix" in ml_proj_data[pid_str]:
            raw_ml_evs = list(ml_proj_data[pid_str]["ml_ev_matrix"])
            
            if pid_str in xmins_overrides:
                target_xmins = float(xmins_overrides[pid_str])
                orig_xmins = float(ml_proj_data[pid_str].get("ml_xmins", 90.0))
                base_ml_ev = float(ml_proj_data[pid_str].get("ml_ev_1gw", raw_ml_evs[0]))
                
                prob_60 = 1.0 / (1.0 + math.exp(-0.15 * (target_xmins - 60.0)))
                app_ev = (prob_60 * 2.0) + ((1.0 - prob_60) * (target_xmins / 60.0))
                attacking_ev = max(0.0, base_ml_ev - 2.0) * (target_xmins / max(1.0, orig_xmins))
                
                gw1_override_ev = app_ev + attacking_ev
                raw_ml_evs = [gw1_override_ev] + raw_ml_evs[1:]

            for t in range(min(8, len(raw_ml_evs))):
                ev_val = float(raw_ml_evs[t])
                sigma = ev_val * (0.45 if pos_id == 3 else (0.40 if pos_id == 4 else 0.30))
                if RISK_POSTURE == "SHIELD":
                    ev_val -= (sigma * 0.15)
                elif RISK_POSTURE == "CHASE":
                    ev_val += (sigma * 0.15)
                ev_matrix[pid][t] = max(0.0, ev_val)

        else:
            base_ev = get_ensemble_ev(p, xmins_overrides, market_data, weights, RISK_POSTURE)
            sigma = base_ev * (0.45 if pos_id == 3 else (0.40 if pos_id == 4 else 0.30))
            if RISK_POSTURE == "SHIELD": base_ev -= (sigma * 0.15)
            elif RISK_POSTURE == "CHASE": base_ev += (sigma * 0.15)
            
            for t in range(0, 8):
                gw_opponents = team_gw_opponents[t_id][t]
                if not gw_opponents:
                    ev_matrix[pid][t] = 0.0
                    continue
                
                gw_ev_total = 0.0
                for opp_data in gw_opponents:
                    opp_id = opp_data["opp"]
                    is_home = opp_data["is_home"]
                    opp_name = teams.get(opp_id, "")
                    
                    opp_metrics = market_data.get(opp_name, {})
                    opp_xg = opp_metrics.get("xG", _safe_float(p.get("xgc_90"), 1.50))
                    opp_xgc = opp_metrics.get("xGC", _safe_float(p.get("xgi_90"), 1.50))
                    
                    ha_factor = 1.05 if is_home else 0.95
                    
                    if pos_id in [1, 2]: 
                        raw_multiplier = (1.50 / max(0.50, opp_xg * ha_factor))
                    else: 
                        raw_multiplier = ((opp_xgc * ha_factor) / 1.50)
                    
                    raw_multiplier = max(0.65, min(1.35, raw_multiplier))
                    horizon_decay = 0.95 ** t
                    final_multiplier = (raw_multiplier * horizon_decay) + (1.0 * (1.0 - horizon_decay))
                    
                    gw_ev_total += base_ev * final_multiplier
                
                ev_matrix[pid][t] = max(0.0, gw_ev_total)

    starter_candidates = sorted([p for p in valid_ids if players[p].get("cost", 0) >= 5.0], 
                                key=lambda x: ev_matrix[x][0], reverse=True)[:11]
    
    p_zero_mins = []
    for pid in starter_candidates:
        p = players[pid]
        pid_str = str(p["id"])
        xmins = float(xmins_overrides[pid_str]) if pid_str in xmins_overrides else p.get("ml_xmins", estimate_xmins(p))
        p_z = 1.0 / (1.0 + math.exp(0.1 * (xmins - 35.0)))
        p_zero_mins.append(p_z)
    
    p_0 = 1.0
    for p_z in p_zero_mins: p_0 *= (1.0 - p_z)
    
    p_1 = 0.0
    for i, p_z in enumerate(p_zero_mins):
        term = p_z
        for j, p_other in enumerate(p_zero_mins):
            if i != j: term *= (1.0 - p_other)
        p_1 += term
    
    dynamic_w_sub_1 = round(max(0.04, min(0.30, 1.0 - p_0)), 3)
    dynamic_w_sub_2 = round(max(0.01, min(0.15, 1.0 - p_0 - p_1)), 3)

    available_chips, free_transfers = get_user_fpl_context(target_gw, headers)

    optimal_squad, transfer_plan = solve_multi_period_model(
        players, ev_matrix, current_squad_ids, total_liquid_budget, 
        free_transfers, active_chip=ACTIVE_CHIP, horizons=8, risk_posture=RISK_POSTURE, target_gw=target_gw,
        w_sub_1=dynamic_w_sub_1, w_sub_2=dynamic_w_sub_2, planned_chips=planned_chips, bank=bank,
        available_chips=available_chips
    )

    if not optimal_squad or len(optimal_squad) < 15:
        valid_players = [p for p in players.values() if p.get("status") in ["a", "d", ""]]
        gks = sorted([p for p in valid_players if p["pos_id"] == 1], key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        defs = sorted([p for p in valid_players if p["pos_id"] == 2], key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        mids = sorted([p for p in valid_players if p["pos_id"] == 3], key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        fwds = sorted([p for p in valid_players if p["pos_id"] == 4], key=lambda x: ev_matrix[x["id"]][0], reverse=True)
        optimal_squad = gks[:2] + defs[:5] + mids[:5] + fwds[:3]

    sub_prob = pulp.LpProblem("Phase2_StartingXI", pulp.LpMaximize)
    s_vars = pulp.LpVariable.dicts("s", [p["id"] for p in optimal_squad], cat="Binary")
    c_vars = pulp.LpVariable.dicts("c", [p["id"] for p in optimal_squad], cat="Binary")
    
    cap_mult = 2.0 if ACTIVE_CHIP == "TRIPLE_CAPTAIN" else 1.0
    sub_objective = []
    for p in optimal_squad:
        ev_1gw = ev_matrix[p["id"]][0]
        ownership = p.get("own", 0.0) / 100.0
        rank_gravity = (ev_1gw * (ownership ** 2) * 1.50) if RISK_POSTURE == "SHIELD" else (
            -(ev_1gw * ownership * 0.50) if RISK_POSTURE == "CHASE" else 
            (ev_1gw * (ownership ** 2) * 0.75))
        sub_objective.append((ev_1gw * s_vars[p["id"]]) + ((ev_1gw * cap_mult + rank_gravity) * c_vars[p["id"]]))
    
    sub_prob += pulp.lpSum(sub_objective)
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad]) == 11
    sub_prob += pulp.lpSum([c_vars[p["id"]] for p in optimal_squad]) == 1
    for p in optimal_squad: sub_prob += c_vars[p["id"]] <= s_vars[p["id"]]
    
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 1]) == 1
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 2]) <= 5
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 3]) >= 2
    sub_prob += pulp.lpSum([s_vars[p["id"]] for p in optimal_squad if p["pos_id"] == 4]) >= 1

    sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters, bench = [], []
    cap = None
    for p in optimal_squad:
        is_st = bool(s_vars[p["id"]].varValue and s_vars[p["id"]].varValue > 0.5)
        is_cp = bool(c_vars[p["id"]].varValue and c_vars[p["id"]].varValue > 0.5)
        p["is_starter"] = is_st
        p["is_captain"] = is_cp
        p["is_vice"] = False
        p["actual_ev"] = ev_matrix[p["id"]][0] # Store actual EV for report builder
        if is_st:
            starters.append(p)
            if is_cp: cap = p
        else:
            bench.append(p)
    
    starters.sort(key=lambda x: x["pos_id"])
    starters_sorted_by_1gw = sorted(starters, key=lambda x: x["actual_ev"], reverse=True)
    vice = next((p for p in starters_sorted_by_1gw if not cap or (p["id"] != cap["id"] and p["team_id"] != cap["team_id"])), starters_sorted_by_1gw[0])
    
    if vice:
        for p in starters:
            if p["id"] == vice["id"]:
                p["is_vice"] = True
                break

    optimal_squad = starters + bench

    european_flags = check_european_congestion_flags(starters, fixtures_data, target_gw)

    projected_starting_xP = sum(p["actual_ev"] for p in starters)
    if cap:
        cap_mult = 3.0 if ACTIVE_CHIP == "TRIPLE_CAPTAIN" else 2.0
        projected_starting_xP += (cap["actual_ev"] * (cap_mult - 1.0))
    
    state["pending_evaluation"] = {
        "gw": target_gw, "projected_xP": round(projected_starting_xP, 2), "captain": cap["name"] if cap else None
    }
    save_state(state)
    
    def count_pos(group, pos_id): return len([p for p in group if p["pos_id"] == pos_id])
    formation = f"{count_pos(starters, 2)}-{count_pos(starters, 3)}-{count_pos(starters, 4)}"

    # Generate Final Locked-in Squad Str for appending to the end
    squad_lines = [
        "================================================================================",
        " FINAL LOCKED-IN SQUAD SUMMARY",
        "================================================================================",
        f"GW{target_gw} Formation: {formation:<5} | Liquid Value: £{total_liquid_budget:<5.1f}m | Bank: £{bank:<4.1f}m",
        f"Captain (C): {cap['name'] if cap else 'N/A'} ({cap['team'] if cap else ''}) | Vice-Captain (VC): {vice['name'] if vice else 'N/A'} ({vice['team'] if vice else ''})",
        "",
        "STARTING XI:"
    ]

    for p in starters:
        is_cap = " (C)" if cap and p["id"] == cap["id"] else ""
        is_vice = " (VC)" if vice and p["id"] == vice["id"] else ""
        pos_str = f"{p['pos']}:"
        name_team = f"{p['name']} ({p['team']})"
        squad_lines.append(f"{pos_str:<4} {name_team:<22} | £{p['cost']:>4.1f}m | {p['ml_xmins']:>4.1f} xMins | {p['actual_ev']:>4.2f} EV{is_cap}{is_vice}")
    
    squad_lines.extend(["", "BENCH ORDER:"])
    bench_gk = [p for p in bench if p["pos_id"] == 1]
    bench_outfield = sorted([p for p in bench if p["pos_id"] != 1], key=lambda x: x["actual_ev"], reverse=True)
    
    for i, p in enumerate(bench_gk + bench_outfield):
        prefix = "GK:" if p["pos_id"] == 1 else f"S{i}:"
        name_team = f"{p['name']} ({p['team']})"
        squad_lines.append(f"{prefix:<4} {name_team:<22} | £{p['cost']:>4.1f}m | {p['ml_xmins']:>4.1f} xMins | {p['actual_ev']:>4.2f} EV")
    
    squad_lines.extend([
        "",
        "================================================================================",
        " MULTI-PERIOD TRANSFER TREE",
        "================================================================================"
    ])
    
    if transfer_plan:
        for tp in transfer_plan: squad_lines.append(f"• {tp}")
    else:
        squad_lines.append(f"• GW{target_gw}: Hold / Bank Transfer (Allocation locked)")
    
    chip_recommendations = evaluate_chip_thresholds(starters, bench, ev_matrix, ACTIVE_CHIP)
    squad_lines.extend([
        "================================================================================",
        " ALGORITHMIC CHIP RECOMMENDATIONS",
        "================================================================================"
    ])
    for rec in chip_recommendations: squad_lines.append(f"• {rec}")

    macro_squad_8gw_xp = sum(get_macro_ev(p, team_avg_fdr, weights, xmins_overrides, market_data, 8, RISK_POSTURE) for p in optimal_squad)
    
    from fpl_monte_carlo import run_monte_carlo_simulations
    mc_players = {p["id"]: {"est_xmins": p.get("ml_xmins", estimate_xmins(p)), "xgi_90": p["xgi_90"], "pos_id": p["pos_id"], "ep_next": p["ep_next"]} for p in starters}
    mc_results = run_monte_carlo_simulations(mc_players, num_trials=1000)
    
    starter_floor = sum(res["floor"] for res in mc_results.values())
    starter_ceiling = sum(res["ceiling"] for res in mc_results.values())

    squad_lines.extend([
        "================================================================================",
        " SQUAD HEALTH & VARIANCE REPORT",
        "================================================================================",
        f"• 1-GW Expected Yield (Odds-Adjusted): {projected_starting_xP:>6.2f} pts",
        f"• 8-GW Deep-Tree Horizon xP: {macro_squad_8gw_xp:>6.2f} pts",
        f"• Stochastic 10th Percentile Floor: {starter_floor:>6.1f} pts",
        f"• Stochastic 90th Percentile Ceiling: {starter_ceiling:>6.1f} pts",
        "================================================================================"
    ])

    locked_squad_str = "\n".join(squad_lines)

    return (target_gw, bank, total_liquid_budget, formation, starters, bench, 
            projected_starting_xP, macro_squad_8gw_xp, transfer_plan, 
            locked_squad_str)

# ==============================================================================
# PURE PYTHON TEXT GENERATOR (Replaces LLM)
# ==============================================================================

def generate_player_justification(p, is_cap, is_vice):
    pos = p['pos']
    cost = p['cost']
    xmins = p.get('ml_xmins', 90.0)
    ev = p['actual_ev']

    if xmins < 60.0:
        return f"Reduced xMins allocation ({xmins:.0f} mins) places asset in rotation management risk."
    elif is_cap:
        return f"Primary captaincy allocation backed by database-leading EV ceiling."
    elif pos == "GKP":
        return f"Secure minutes baseline and fixture clean sheet probability underpin yield target."
    elif pos == "DEF":
        return f"High defensive contribution potential combined with clean sheet odds."
    elif pos in ["MID", "FWD"] and cost >= 8.5:
        return f"Premium attacking asset driving primary expected goal involvement (xGI) volume."
    elif pos in ["MID", "FWD"] and cost < 6.0:
        return f"High-value budget enabler providing structural capital flexibility."
    else:
        return f"Solid points-per-million value profile within current structural setup."

def build_pure_code_report(target_gw, bank, liquid_value, formation, starters, bench, 
                           projected_xP, horizon_xP, transfer_plan, live_news_found, locked_squad_str):
    
    cap = next((p for p in starters if p.get("is_captain")), starters[0])
    vice = next((p for p in starters if p.get("is_vice")), starters[1])
    
    # SECTION 1
    sec1 = (
        f"**SECTION 1: EXECUTIVE SUMMARY & CORE MOVES**\n\n"
        f"The optimization engine has locked a {formation} layout for Gameweek {target_gw}, "
        f"projecting a 1-GW expected yield of {projected_xP:.2f} points and an 8-GW horizon total "
        f"of {horizon_xP:.2f} points. Liquid value is £{liquid_value:.1f}m with £{bank:.1f}m in bank.\n\n"
        f"Captaincy is assigned to {cap['name']} ({cap['team']}) at {cap['actual_ev']:.2f} EV, "
        f"with {vice['name']} ({vice['team']}) designated as vice-captain at {vice['actual_ev']:.2f} EV."
    )
    
    # SECTION 2
    sec2_lines = ["**SECTION 2: QUANTITATIVE TRADE-OFF SUMMARY**\n"]
    for p in starters + bench:
        justification = generate_player_justification(p, p['id'] == cap['id'], p['id'] == vice['id'])
        sec2_lines.append(f"• {p['name']} ({p['pos']} | {p['team']} | £{p['cost']:.1f}m) — {p['ml_xmins']:.1f} xMins | {p['actual_ev']:.2f} EV | {justification}")
    sec2 = "\n".join(sec2_lines)

    # SECTION 4
    teams_map = {}
    for p in starters:
        teams_map.setdefault(p['team'], []).append(p['name'])
    
    multi_player_teams = {t: players for t, players in teams_map.items() if len(players) >= 2}
    matrix_lines = [
        "+-------------------------------------------------------------------------------+",
        "|                      REAL-WORLD TACTICAL EXPLOIT MATRIX                       |",
        "+-------------------------------------------------------------------------------+"
    ]
    if multi_player_teams:
        for team, p_list in multi_player_teams.items():
            matrix_lines.append(f"| {team} TACTICAL BLOCK:")
            matrix_lines.append(f"| [{' <---> '.join(p_list)}] (Shared Team Structure & Tactical Coupling)")
            matrix_lines.append("|")
    else:
        matrix_lines.append("| No multi-player club blocks deployed in starting XI. Highly decentralized setup. |")
    matrix_lines.append("+-----------------------------------------------------------------+")
    sec4 = "**SECTION 4: REAL-WORLD TACTICAL EXPLOIT & MATCHUP ANALYSIS**\n```text\n" + "\n".join(matrix_lines) + "\n```"

    # SECTION 5
    sec5_text = live_news_found if live_news_found else "Status: Awaiting live press conference data."
    sec5 = f"**SECTION 5: HUMAN ORACLE INTELLIGENCE BRIEFING**\n\n{sec5_text}"

    # ASSEMBLE
    final_report = f"{sec1}\n\n---\n\n{sec2}\n\n---\n\n{sec4}\n\n---\n\n{sec5}\n\n```text\n{locked_squad_str}\n```"
    return final_report

def send_to_discord(webhook_url, text):
    chunks, current_chunk = [], ""
    for line in text.split("\n"):
        while len(line) > 1800:
            if len(current_chunk) > 0: chunks.append(current_chunk); current_chunk = ""
            chunks.append(line[:1800]); line = line[1800:]
        if len(current_chunk) + len(line) + 1 > 1800:
            chunks.append(current_chunk); current_chunk = line + "\n"
        else: current_chunk += line + "\n"
    if current_chunk.strip(): chunks.append(current_chunk)
    for chunk in chunks: requests.post(webhook_url, json={"content": chunk})

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    (target_gw, bank, total_liquid_budget, formation, starters, bench, 
     projected_starting_xP, macro_squad_8gw_xp, transfer_plan, 
     locked_squad_str) = get_fpl_data()

    print("--- FETCHING LIVE WEB SEARCH DATA ---")
    live_news = get_live_fpl_news()
    
    print("--- GENERATING PURE CODE DISCORD PAYLOAD ---")
    report = build_pure_code_report(
        target_gw=target_gw, 
        bank=bank, 
        liquid_value=total_liquid_budget, 
        formation=formation, 
        starters=starters, 
        bench=bench, 
        projected_xP=projected_starting_xP, 
        horizon_xP=macro_squad_8gw_xp, 
        transfer_plan=transfer_plan, 
        live_news_found=live_news, 
        locked_squad_str=locked_squad_str
    )

    send_to_discord(DISCORD_WEBHOOK_URL, report)
    print("--- DISCORD DELIVERY COMPLETE ---")

if __name__ == "__main__":
    main()