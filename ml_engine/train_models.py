"""
ml_engine/train_models.py — Unified Dataset Alignment & Projections Engine
"""

import os
import sys
import json
import logging
import requests
import difflib
import unicodedata
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fpl_funcs import estimate_xmins

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
    """Fetches upcoming fixtures and returns a mapping of integer team_id -> opponent integer team_id."""
    if current_gw is None:
        try:
            bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
            next_events = [e for e in bootstrap['events'] if e.get('is_next')]
            if not next_events:
                next_events = [e for e in bootstrap['events'] if e.get('is_current')]
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

def execute_tri_model_regression(df: pd.DataFrame) -> np.ndarray:
    """
    Trains a Tri-Model ML Regressor Suite to predict 1-GW Expected Value.
    """
    logger.info("Initializing Tri-Model Regressor Suite (XGBoost, LightGBM, Random Forest)...")
    
    # 1. Feature Matrix Construction
    features = [
        'cost_float', 'global_own', 'combined_xgi', 'xgc_90_num', 
        'opponent_def_rating', 'fb_mins', 'age_num', 
        'fpl_cbit_90', 'fpl_cbirt_90'
    ]
    X = df[features].fillna(0.0)
    
    # 2. Target Variable (y) Definition
    # Pre-season proxy: Blend of proprietary algorithmic expectations. 
    # To be swapped to actual points_per_90 once the season has accrued 4+ gameweeks.
    ep = pd.to_numeric(df['ep_next_raw'], errors='coerce').fillna(0.0)
    form = pd.to_numeric(df['form_raw'], errors='coerce').fillna(0.0)
    y = (ep * 0.7) + (form * 0.3)
    
    # 3. Model Initialization
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    
    # 4. In-Memory Training
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)
    rf_model.fit(X, y)
    
    # 5. Ensemble Prediction Generation
    # Weighting: 40% XGBoost, 40% LightGBM, 20% Random Forest (Variance Reducer)
    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X)
    rf_preds = rf_model.predict(X)
    
    ensemble_preds = (0.4 * xgb_preds) + (0.4 * lgb_preds) + (0.2 * rf_preds)
    
    # Ensure no negative EVs are returned
    return np.maximum(0.0, ensemble_preds)

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets for ML Pipeline...")
    
    top10k_eo_dict = get_livefpl_top10k_eo()
    crowd_xmins_dict = get_crowdsourced_xmins(fpl_df)
    
    custom_xmins_dict = {}
    xmins_env = os.getenv("XMINS_INPUT", "")
    if xmins_env and xmins_env.strip():
        try:
            custom_xmins_dict = json.loads(xmins_env.replace("'", '"'))
            logger.info(f"Loaded {len(custom_xmins_dict)} manual xMins overrides.")
        except Exception as e:
            logger.warning(f"Failed to parse XMINS_INPUT JSON: {e}")

    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        df = pd.merge(fpl_df, fbref_df, left_on='matched_fbref_name', right_on='clean_fbref_name', how='left', suffixes=('', '_fbref'))
        logger.info(f"FPL to FBref Match Rate: {df['matched_fbref_name'].notna().mean() * 100:.1f}%")
    else:
        df = fpl_df.copy()

    from ml_engine.data_ingestion import get_team_matchup_ratings
    team_ratings = get_team_matchup_ratings(fbref_df, fpl_df)
    opp_mapping = get_upcoming_opponent_mapping()
    
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
    teams_short_by_id = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}

    # --- VECTORIZED FEATURE ENGINEERING ---
    def resolve_opp_rating(team_id):
        try:
            opp_id = opp_mapping.get(int(team_id))
            if not opp_id: return 1.0
            opp_short = teams_short_by_id.get(opp_id, "")
            if opp_short in team_ratings: return team_ratings[opp_short]
            opp_fbref = FPL_TO_FBREF_TEAM.get(opp_short, "")
            if opp_fbref in team_ratings: return team_ratings[opp_fbref]
        except: pass
        return 1.0

    df['opponent_def_rating'] = df['team'].apply(resolve_opp_rating)
    df['cost_float'] = pd.to_numeric(df.get('now_cost', 40), errors='coerce').fillna(40) / 10.0
    df['global_own'] = pd.to_numeric(df.get('selected_by_percent', 0.0), errors='coerce').fillna(0.0)
    df['fb_mins'] = pd.to_numeric(df.get('minutes_played', 0.0), errors='coerce').fillna(0.0)
    df['age_num'] = pd.to_numeric(df.get('age', 25), errors='coerce').fillna(25)
    df['xgc_90_num'] = pd.to_numeric(df.get('expected_goals_conceded_per_90', 1.35), errors='coerce').fillna(1.35)
    df['ep_next_raw'] = df.get('ep_next', 0.0)
    df['form_raw'] = df.get('form', 0.0)

    # Calculate native DEFCON metrics for the ML Matrix
    df['fpl_mins_played'] = pd.to_numeric(df.get('minutes', 0.0), errors='coerce').clip(lower=0.001)
    df['fpl_cbit'] = df[['clearances', 'blocks', 'interceptions', 'tackles']].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    df['fpl_cbirt'] = df['fpl_cbit'] + pd.to_numeric(df.get('recoveries', 0), errors='coerce').fillna(0)
    df['fpl_cbit_90'] = (df['fpl_cbit'] / df['fpl_mins_played']) * 90.0
    df['fpl_cbirt_90'] = (df['fpl_cbirt'] / df['fpl_mins_played']) * 90.0

    # Calculate blended xGI
    fb_xg = pd.to_numeric(df.get('fbref_xg', 0.0), errors='coerce').fillna(0.0)
    fb_xag = pd.to_numeric(df.get('fbref_xag', 0.0), errors='coerce').fillna(0.0)
    native_xgi = pd.to_numeric(df.get('expected_goal_involvements_per_90', 0.0), errors='coerce').fillna(0.0)
    
    df['combined_xgi'] = np.where(
        df['fb_mins'] > 270.0,
        np.where(native_xgi > 0, (0.60 * ((fb_xg + fb_xag) / (df['fb_mins'] / 90.0))) + (0.40 * native_xgi), ((fb_xg + fb_xag) / (df['fb_mins'] / 90.0))),
        native_xgi
    )

    # --- EXECUTE TRI-MODEL MACHINE LEARNING REGRESSION ---
    ml_predictions = execute_tri_model_regression(df)
    df['ml_base_ev'] = ml_predictions

    # --- POPULATE FINAL JSON PAYLOAD ---
    projections = {}
    for idx, row in df.iterrows():
        pid = str(row['id'])
        web_name = str(row.get('web_name', 'Unknown'))
        
        # Ownership calculations
        if pid in top10k_eo_dict:
            top_10k_eo = top10k_eo_dict[pid]
        else:
            if row['global_own'] > 30.0 and row['cost_float'] >= 10.0: top_10k_eo = min(200.0, row['global_own'] * 1.6) 
            elif row['global_own'] > 20.0 and row['cost_float'] >= 7.0: top_10k_eo = min(150.0, row['global_own'] * 1.3)
            elif row['global_own'] < 10.0: top_10k_eo = row['global_own'] * 0.5 
            else: top_10k_eo = row['global_own']
                
        transfers_in = int(row.get('transfers_in_event', 0) or 0)
        transfers_out = int(row.get('transfers_out_event', 0) or 0)
        net_transfers = transfers_in - transfers_out
        
        predicted_delta = 0.1 if net_transfers > 75000 else (-0.1 if net_transfers < -75000 else 0.0)

        player_obj = {
            "id": row['id'], "name": web_name, "team": row.get('team_code', 'UNK'),
            "pos_id": int(row.get('element_type', 3)), "cost": row['cost_float'],
            "own": row['global_own'], "status": str(row.get('status', 'a')),
            "chance_of_playing_next_round": row.get('chance_of_playing_next_round'),
            "fb_mins": row['fb_mins']
        }
        
        # xMins Hierarchy Resolution
        original_xmins = estimate_xmins(player_obj)
        final_xmins = custom_xmins_dict.get(web_name, crowd_xmins_dict.get(web_name, original_xmins))

        # Apply Matchup Multiplier & xMins Delta to ML output
        calculated_ev = row['ml_base_ev'] * row['opponent_def_rating']
        if final_xmins != original_xmins and original_xmins > 0:
            calculated_ev = calculated_ev * (final_xmins / original_xmins)
        
        projections[pid] = {
            "ml_xmins": round(final_xmins, 1),
            "ml_ev_1gw": round(calculated_ev, 2),
            "ml_ev_8gw": round(calculated_ev * 8 * 0.95, 2),
            "mc_floor_ev": round(calculated_ev * 0.6, 2),    
            "mc_ceiling_ev": round(calculated_ev * 1.5, 2),  
            "top_10k_eo": round(top_10k_eo, 2),               
            "predicted_price_delta": predicted_delta         
        }
        
    logger.info(f"Successfully generated Tri-Model ML projections for {len(projections)} players.")
    return projections