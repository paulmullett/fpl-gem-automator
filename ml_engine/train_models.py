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

def get_crowdsourced_xmins(fpl_df: pd.DataFrame) -> dict:
    """
    Attempts to load crowdsourced xMins from a local 'fplreview.csv' or an external JSON API.
    Fuzzy matches external names to FPL web_names.
    """
    logger.info("Checking for crowdsourced xMins data...")
    crowd_xmins = {}
    name_pool = fpl_df['web_name'].dropna().tolist()
    
    # 1. Try local CSV (Standard for FPL ML community)
    if os.path.exists("fplreview.csv"):
        try:
            df = pd.read_csv("fplreview.csv")
            # FPLReview standard columns: 'Name', '1_xMins'
            name_col = next((c for c in df.columns if 'name' in c.lower()), None)
            mins_col = next((c for c in df.columns if 'xmins' in c.lower() or 'mins' in c.lower()), None)
            
            if name_col and mins_col:
                for _, row in df.iterrows():
                    match = difflib.get_close_matches(str(row[name_col]), name_pool, n=1, cutoff=0.70)
                    if match:
                        crowd_xmins[match[0]] = float(row[mins_col])
                logger.info(f"Loaded {len(crowd_xmins)} xMins projections from local fplreview.csv")
                return crowd_xmins
        except Exception as e:
            logger.warning(f"Failed to parse local fplreview.csv: {e}")

    # 2. Automated Remote Fallback Hook
    url = os.getenv("CROWD_XMINS_URL", "")
    if url:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for ext_name, mins in data.items():
                    match = difflib.get_close_matches(str(ext_name), name_pool, n=1, cutoff=0.70)
                    if match:
                        crowd_xmins[match[0]] = float(mins)
                logger.info(f"Loaded {len(crowd_xmins)} xMins projections from external API.")
        except Exception as e:
            logger.warning(f"Failed to fetch remote xMins: {e}")

    if not crowd_xmins:
        logger.info("No crowdsourced xMins found. Pipeline will rely on internal heuristics.")
        
    return crowd_xmins

def get_upcoming_opponent_mapping(current_gw: int = None) -> dict:
    """Fetches upcoming fixtures and returns a mapping of team_id (int) -> opponent_team_id (int)."""
    if current_gw is None:
        try:
            bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
            next_events = [e for e in bootstrap['events'] if e['is_next']]
            if not next_events:
                next_events = [e for e in bootstrap['events'] if e['is_current']]
            current_gw = next_events[0]['id'] if next_events else 1
        except Exception:
            current_gw = 1

    opp_map = {}
    try:
        url = f"https://fantasy.premierleague.com/api/fixtures/?event={current_gw}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fixtures = response.json()
            for f in fixtures:
                h_team = f.get('team_h')
                a_team = f.get('team_a')
                if h_team and a_team:
                    opp_map[int(h_team)] = int(a_team)
                    opp_map[int(a_team)] = int(h_team)
    except Exception as e:
        logger.warning(f"Could not fetch upcoming fixture mapping: {e}")
    return opp_map

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets...")
    
    top10k_eo_dict = get_livefpl_top10k_eo()
    
    # --- NEW: Initialize Crowdsourced xMins ---
    crowd_xmins_dict = get_crowdsourced_xmins(fpl_df)
    
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

    # --- Dynamic Matchup Rating Wiring ---
    from ml_engine.data_ingestion import get_team_matchup_ratings
    
    team_ratings = get_team_matchup_ratings(fbref_df)
    opp_mapping = get_upcoming_opponent_mapping()
    
    # Map team IDs to FPL Short Names ("ARS", "MCI")
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
    teams_short_by_id = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}
    
    def calculate_player_opp_rating(row):
        # 1. Extract integer FPL team ID
        player_team_id = int(row['team']) if pd.notna(row.get('team')) else None
        if not player_team_id:
            return 1.0
            
        # 2. Lookup integer opponent team ID
        opp_id = opp_mapping.get(player_team_id)
        if not opp_id:
            return 1.0
            
        # 3. Resolve opponent FPL short name -> FBref full name
        opp_short = teams_short_by_id.get(opp_id)
        opp_fbref_name = FPL_TO_FBREF_TEAM.get(opp_short, "")
        
        # 4. Extract opponent defensive frailty rating
        if opp_fbref_name and opp_fbref_name in team_ratings:
            return team_ratings[opp_fbref_name].get('def_rating', 1.0)
            
        return 1.0

    df['opponent_def_rating'] = df.apply(calculate_player_opp_rating, axis=1)

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

        opponent_def_rating = row.get('opponent_def_rating', 1.0) 
        combined_xgi = combined_xgi * opponent_def_rating

        global_own = float(row.get('selected_by_percent', 0.0) or 0.0)
        cost_float = float(row.get('now_cost', 40)) / 10.0
        
        if pid in top10k_eo_dict:
            top_10k_eo = top10k_eo_dict[pid]
        else:
            if global_own > 30.0 and cost_float >= 10.0:
                top_10k_eo = min(200.0, global_own * 1.6) 
            elif global_own > 20.0 and cost_float >= 7.0:
                top_10k_eo = min(150.0, global_own * 1.3)
            elif global_own < 10.0:
                top_10k_eo = global_own * 0.5 
            else:
                top_10k_eo = global_own
                
        transfers_in = int(row.get('transfers_in_event', 0) or 0)
        transfers_out = int(row.get('transfers_out_event', 0) or 0)
        net_transfers = transfers_in - transfers_out
        
        predicted_delta = 0.0
        if net_transfers > 75000:
            predicted_delta = 0.1
        elif net_transfers < -75000:
            predicted_delta = -0.1

        player_obj = {
            "id": row['id'],
            "name": web_name,
            "team": row.get('team_code', 'UNK'),
            "pos_id": int(row.get('element_type', 3)),
            "cost": cost_float,
            "predicted_price_delta": predicted_delta, 
            "own": global_own,
            "top_10k_eo": round(top_10k_eo, 2),
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
        
        # --- NEW: Three-Tier xMins Hierarchy & Mathematical EV Scaling ---
        # 1. Base Heuristic
        final_xmins = original_xmins
        
        # 2. Crowdsourced Anchor
        if web_name in crowd_xmins_dict:
            final_xmins = crowd_xmins_dict[web_name]
            
        # 3. Human-In-The-Loop Override (Highest Priority)
        if web_name in custom_xmins_dict:
            final_xmins = float(custom_xmins_dict[web_name])

        # Execute Mathematical EV Scaling if xMins shifted from the baseline heuristic
        if final_xmins != original_xmins:
            if original_xmins > 0:
                calculated_ev = calculated_ev * (final_xmins / original_xmins)
            else:
                calculated_ev = (combined_xgi * (final_xmins / 90.0)) + (2.0 * (final_xmins / 90.0))
        
        projections[pid] = {
            "ml_xmins": round(final_xmins, 1),
            "ml_ev_1gw": round(calculated_ev, 2),
            "ml_ev_8gw": round(calculated_ev * 8 * 0.95, 2),
            "mc_floor_ev": round(calculated_ev * 0.6, 2),    
            "mc_ceiling_ev": round(calculated_ev * 1.5, 2),  
            "top_10k_eo": round(top_10k_eo, 2),               
            "predicted_price_delta": predicted_delta         
        }
        
    logger.info(f"Successfully generated projections for {len(projections)} players using rich math engine.")
    return projections