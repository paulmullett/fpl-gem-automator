"""
ml_engine/train_models.py — Unified Dataset Alignment & Projections Engine
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import unicodedata
import difflib
import json
import requests # NEW: Required for LiveFPL API fetching

# Ensure root directory is on sys.path to import fpl_funcs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fpl_funcs import get_ensemble_ev, estimate_xmins

logger = logging.getLogger(__name__)

FPL_TO_FBREF_TEAM = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford",
    "BHA": "Brighton", "CHE": "Chelsea", "COV": "Coventry", "CRY": "Crystal Palace",
    "EVE": "Everton", "FUL": "Fulham", "HUL": "Hull", "IPI": "Ipswich",
    "LEE": "Leeds", "LEI": "Leicester", "LIV": "Liverpool", "MCI": "Manchester City",
    "MUN": "Manchester Utd", "NEW": "Newcastle", "NFO": "Nottingham", "SOU": "Southampton",
    "SUN": "Sunderland", "TOT": "Tottenham", "WHU": "West Ham", "WOL": "Wolves"
}

NAME_OVERRIDES = {
    "bruno guimaraes rodriguez moura": "bruno guimaraes",
    "bruno g.": "bruno guimaraes",
    "stefan bajcetic maquieira": "stefan bajcetic",
    "bajcetic": "stefan bajcetic",
    "nico gonzalez iglesias": "nico gonzalez",
    "n.gonzalez": "nico gonzalez",
    "dejan kulusevski": "dejan kulusevski",
    "kulusevski": "dejan kulusevski",
    "kalvin phillips": "kalvin phillips",
    "illan meslier": "illan meslier",
    "christos tzolis": "christos tzolis",
    "erling haaland": "erling haaland",
    "haaland": "erling haaland"
}

def strip_accents(text):
    try:
        text = str(text)
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8").lower().strip()
    except Exception:
        return str(text).lower().strip()

def match_logic(full_name, web_name, second_name, first_name, name_pool):
    if full_name in name_pool: return full_name
    if web_name in name_pool: return web_name
    
    sn_tokens = set(second_name.split()) if second_name else set()
    wn_tokens = set(web_name.split('-')) if web_name else set()
    
    for fb_name in name_pool:
        fb_tokens = set(fb_name.split())
        if sn_tokens.intersection(fb_tokens):
            if not first_name or first_name[0] == fb_name[0] or wn_tokens.intersection(fb_tokens):
                return fb_name
        if wn_tokens.intersection(fb_tokens):
            return fb_name
            
    fuzzy = difflib.get_close_matches(full_name, name_pool, n=1, cutoff=0.70)
    if fuzzy: return fuzzy[0]
    fuzzy_web = difflib.get_close_matches(web_name, name_pool, n=1, cutoff=0.70)
    if fuzzy_web: return fuzzy_web[0]
    return None

def find_best_match(fpl_row, fbref_df):
    web_name = strip_accents(fpl_row.get('web_name', ''))
    first_name = strip_accents(fpl_row.get('first_name', ''))
    second_name = strip_accents(fpl_row.get('second_name', ''))
    full_name = f"{first_name} {second_name}".strip()
    
    global_pool = fbref_df['clean_fbref_name'].tolist()
    
    if full_name in NAME_OVERRIDES and NAME_OVERRIDES[full_name] in global_pool:
        return NAME_OVERRIDES[full_name]
    if web_name in NAME_OVERRIDES and NAME_OVERRIDES[web_name] in global_pool:
        return NAME_OVERRIDES[web_name]
    
    team_code = fpl_row.get('team_code', '')
    target_team = strip_accents(FPL_TO_FBREF_TEAM.get(team_code, ''))
    
    if target_team:
        team_pool = fbref_df[fbref_df['clean_team'].str.contains(target_team, na=False, regex=False)]['clean_fbref_name'].tolist()
        if team_pool:
            match = match_logic(full_name, web_name, second_name, first_name, team_pool)
            if match: return match

    return match_logic(full_name, web_name, second_name, first_name, global_pool)

def get_livefpl_top10k_eo():
    """Attempts to fetch real-time Top 10k EO from LiveFPL, with heuristic fallbacks."""
    logger.info("Attempting to fetch Top 10k EO from LiveFPL...")
    eo_dict = {}
    try:
        # 1. Dynamically identify the current Gameweek from the FPL API
        bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
        current_events = [e for e in bootstrap['events'] if e['is_current']]
        if not current_events:
            current_events = [e for e in bootstrap['events'] if e['is_next']] # Pre-season fallback
        current_gw = current_events[0]['id'] if current_events else 1

        # 2. Query the standard LiveFPL endpoint
        headers = {"User-Agent": "FPL-ML-Pipeline/1.0"}
        response = requests.get(f"https://www.livefpl.net/api/gw_players_data/{current_gw}", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for player in data:
                pid = str(player.get("id"))
                eo = float(player.get("top10k_eo", 0.0))
                eo_dict[pid] = eo
            logger.info(f"Successfully downloaded Top 10k EO for {len(eo_dict)} players.")
            return eo_dict
    except Exception as e:
        logger.warning(f"LiveFPL API unavailable or route changed. Defaulting to mathematical EO fallback. ({e})")
        
    return eo_dict

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets...")
    
    # --- NEW: Fetch Top 10k EO ---
    top10k_eo_dict = get_livefpl_top10k_eo()
    
    # Parse External xMins Overrides from GitHub Actions
    custom_xmins_dict = {}
    xmins_env = os.getenv("XMINS_INPUT", "")
    if xmins_env and xmins_env.strip():
        try:
            clean_env = xmins_env.replace("'", '"')
            custom_xmins_dict = json.loads(clean_env)
            logger.info(f"Loaded {len(custom_xmins_dict)} manual xMins overrides from environment.")
        except Exception as e:
            logger.warning(f"Failed to parse XMINS_INPUT JSON. Ignoring overrides. Error: {e}")

    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fbref_df['clean_team'] = fbref_df['team'].apply(strip_accents) if 'team' in fbref_df.columns else ""
        
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        df = pd.merge(fpl_df, fbref_df, left_on='matched_fbref_name', right_on='clean_fbref_name', how='left')
        
        match_rate = df['matched_fbref_name'].notna().mean() * 100
        logger.info(f"FPL to FBref Match Rate: {match_rate:.1f}%")
    else:
        df = fpl_df.copy()

    projections = {}
    for _, row in df.iterrows():
        pid = str(row['id'])
        web_name = row.get('web_name', 'Unknown')
        
        fb_xg = float(row.get('fbref_xg', 0.0)) if pd.notna(row.get('fbref_xg')) else 0.0
        fb_xag = float(row.get('fbref_xag', 0.0)) if pd.notna(row.get('fbref_xag')) else 0.0
        fb_mins = float(row.get('minutes_played', 0.0)) if pd.notna(row.get('minutes_played')) else 0.0
        
        native_xgi = float(row.get('expected_goal_involvements_per_90', 0.0) or 0.0)
        
        if fb_mins > 270.0:
            fb_xgi_90 = (fb_xg + fb_xag) / (fb_mins / 90.0)
            combined_xgi = (0.60 * fb_xgi_90) + (0.40 * native_xgi) if native_xgi > 0 else fb_xgi_90
        else:
            combined_xgi = native_xgi

        # --- NEW: Top 10k EO Extraction & Heuristic Fallback ---
        global_own = float(row.get('selected_by_percent', 0.0) or 0.0)
        cost_float = float(row.get('now_cost', 40)) / 10.0
        
        if pid in top10k_eo_dict:
            top_10k_eo = top10k_eo_dict[pid]
        else:
            # Mathematical Heuristic Fallback (Concentration of premium assets in Top 10k)
            if global_own > 30.0 and cost_float >= 10.0:
                top_10k_eo = min(200.0, global_own * 1.6) # Accounts for Captaincy spikes
            elif global_own > 20.0 and cost_float >= 7.0:
                top_10k_eo = min(150.0, global_own * 1.3)
            elif global_own < 10.0:
                top_10k_eo = global_own * 0.5 # Casual accounts inflate low-tier fringe players
            else:
                top_10k_eo = global_own

        player_obj = {
            "id": row['id'],
            "name": web_name,
            "team": row.get('team_code', 'UNK'),
            "pos_id": int(row.get('element_type', 3)),
            "cost": cost_float,
            "own": global_own,
            "top_10k_eo": round(top_10k_eo, 2), # --- NEW INJECTION ---
            "status": str(row.get('status', 'a')),
            "ep_next": float(row.get('ep_next', 0.0) or 0.0),
            "form": float(row.get('form', 0.0) or 0.0),
            "xgi_90": combined_xgi,
            "xgc_90": float(row.get('expected_goals_conceded_per_90', 1.35) or 1.35),
            "chance_of_playing_next_round": row.get('chance_of_playing_next_round'),
            "age": int(row.get('age', 25) or 25),
            "has_stale_pl_history": bool(row.get('has_stale_pl_history', False)),
            "recent_european_peak": bool(row.get('recent_european_peak', False)),
            "fb_mins": fb_mins 
        }
        
        original_xmins = estimate_xmins(player_obj)
        calculated_ev = get_ensemble_ev(player_obj)
        
        # Dynamic xMins Mathematical Scaling
        if web_name in custom_xmins_dict:
            xmins = float(custom_xmins_dict[web_name])
            if original_xmins > 0:
                calculated_ev = calculated_ev * (xmins / original_xmins)
            else:
                calculated_ev = (combined_xgi * (xmins / 90.0)) + (2.0 * (xmins / 90.0))
        else:
            xmins = original_xmins
        
        projections[pid] = {
            "ml_xmins": round(xmins, 1),
            "ml_ev_1gw": round(calculated_ev, 2),
            "ml_ev_8gw": round(calculated_ev * 8 * 0.95, 2),
            "mc_floor_ev": round(calculated_ev * 0.6, 2),    # Aligned key for solver
            "mc_ceiling_ev": round(calculated_ev * 1.5, 2),  # Aligned key for solver
            "top_10k_eo": round(top_10k_eo, 2)               # Export EO to solver
        }
        
    logger.info(f"Successfully generated projections for {len(projections)} players using rich math engine.")
    return projections