"""
ml_engine/train_models.py — Unified Dataset Alignment & Projections Engine

Academic Attribution: 
Features position-specific ensemble segregation and asymmetric sample weighting 
methodologies adapted from OpenFPL (Groos, 2025, arXiv:2508.09992).
"""

import os
import sys
import json
import logging
import requests
import difflib
import unicodedata
import re
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpl_funcs import estimate_xmins
from ml_engine.data_ingestion import get_team_matchup_ratings, fetch_understat_rolling_data

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

logger = logging.getLogger(__name__)

def get_crowdsourced_data(fpl_df: pd.DataFrame) -> tuple:
    """
    Loads crowdsourced xMins and Pts matrices from local fplreview*.csv files
    or an external JSON API fallback. Handles multi-period, single-column,
    and non-standard column naming conventions.
    
    Returns:
        tuple: (crowd_xmins_dict, crowd_pts_dict)
    """
    logger.info("Checking for crowdsourced xMins and Pts matrix data...")
    crowd_xmins = {}
    crowd_pts = {}
    name_pool = fpl_df['web_name'].dropna().tolist()

    # 1. Look for any matching local CSV file
    csv_files = glob.glob("fplreview*.csv")
    target_csv = csv_files[0] if csv_files else ("fplreview.csv" if os.path.exists("fplreview.csv") else None)

    if target_csv and os.path.exists(target_csv):
        try:
            df = pd.read_csv(target_csv)
            id_col = next((c for c in df.columns if str(c).lower().strip() in ['id', 'element']), None)

            # Identify xMins columns (e.g., 1_xMins, xMins_1, 1_mins)
            xmins_cols = [c for c in df.columns if re.search(r'\d+_xmins|\d+_mins|xmins_\d+|mins_\d+', str(c), re.IGNORECASE)]
            
            # Identify Pts columns (e.g., 1_Pts, Pts_1, 1_points)
            pts_cols = [c for c in df.columns if re.search(r'\d+_pts|\d+_points|pts_\d+|points_\d+', str(c), re.IGNORECASE)]

            def get_gw_num(col_name):
                match = re.search(r'\d+', str(col_name))
                return int(match.group()) if match else 99

            xmins_cols = sorted(xmins_cols, key=get_gw_num)[:8]
            pts_cols = sorted(pts_cols, key=get_gw_num)[:8]

            # Primary Route: Multi-Period Columns
            if id_col and xmins_cols:
                for _, row in df.iterrows():
                    try:
                        pid = int(row[id_col])
                        crowd_xmins[pid] = [float(row[c]) for c in xmins_cols]
                        if pts_cols and len(pts_cols) == len(xmins_cols):
                            crowd_pts[pid] = [float(row[c]) for c in pts_cols]
                    except (ValueError, TypeError):
                        continue
                logger.info(f"Loaded 8-GW xMins/Pts matrix for {len(crowd_xmins)} players from {target_csv}")
                return crowd_xmins, crowd_pts

            # Fallback Route 1: Single-Column CSVs
            elif id_col:
                mins_col = next((c for c in df.columns if 'xmins' in str(c).lower() or 'mins' in str(c).lower()), None)
                pts_col = next((c for c in df.columns if 'pts' in str(c).lower() or 'points' in str(c).lower()), None)
                
                if mins_col:
                    for _, row in df.iterrows():
                        try:
                            pid = int(row[id_col])
                            val_mins = float(row[mins_col])
                            crowd_xmins[pid] = [val_mins] * 8
                            if pts_col:
                                crowd_pts[pid] = [float(row[pts_col])] * 8
                        except (ValueError, TypeError):
                            continue
                    logger.info(f"Loaded single-column xMins/Pts for {len(crowd_xmins)} players from {target_csv}")
                    return crowd_xmins, crowd_pts

        except Exception as e:
            logger.warning(f"Failed to parse local {target_csv}: {e}")

    # 2. Fallback Route 2: Automated Remote API
    url = os.getenv("CROWD_XMINS_URL", "")
    if url:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for ext_name, val in data.items():
                    match = difflib.get_close_matches(str(ext_name), name_pool, n=1, cutoff=0.70)
                    if match:
                        matched_name = match[0]
                        if isinstance(val, list):
                            crowd_xmins[matched_name] = [float(x) for x in val[:8]]
                        else:
                            crowd_xmins[matched_name] = [float(val)] * 8
                logger.info(f"Loaded xMins projections from remote API endpoint.")
        except Exception as e:
            logger.warning(f"Failed to fetch remote xMins: {e}")

    if not crowd_xmins:
        logger.info("No crowdsourced matrix found. Pipeline will rely on internal heuristics.")

    return crowd_xmins, crowd_pts

def get_upcoming_opponent_mapping(current_gw: int = None) -> dict:
    if current_gw is None:
        try:
            bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
            next_events = [e for e in bootstrap['events'] if e.get('is_next')]
            if not next_events: next_events = [e for e in bootstrap['events'] if e.get('is_current')]
            current_gw = next_events[0]['id'] if next_events else 1
        except Exception:
            current_gw = 1

    opp_map = {}
    try:
        url = f"https://fantasy.premierleague.com/api/fixtures/?event={current_gw}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            for f in response.json():
                h_team, a_team = f.get('team_h'), f.get('team_a')
                if h_team and a_team:
                    if int(h_team) not in opp_map: opp_map[int(h_team)] = []
                    if int(a_team) not in opp_map: opp_map[int(a_team)] = []
                    opp_map[int(h_team)].append(int(a_team))
                    opp_map[int(a_team)].append(int(h_team))
    except Exception as e:
        logger.warning(f"Could not fetch upcoming fixture mapping: {e}")
    return opp_map

def execute_tri_model_regression(df: pd.DataFrame) -> np.ndarray:
 """
 Trains Position-Specific Tri-Model ML Regressor Suites with Asymmetric 
 'Hauler' Sample Weighting and Automated K-Best Hyperparameter Tuning.
 """
 logger.info("Initializing Position-Specific Tri-Model Regressor Suites...")
 
 features = [
 'cost_float', 'global_own', 'combined_xgi', 'xgc_90_num', 
 'opponent_def_rating', 'fb_mins', 'fpl_cbit_90', 'fpl_cbirt_90',
 'xg_per_shot', 'xa_per_kp'
 ]
 
 df['ml_scalar'] = 1.0 # Default fallback
 
 # Iterate through each FPL position (1: GK, 2: DEF, 3: MID, 4: FWD)
 for pos_id in [1, 2, 3, 4]:
     pos_mask = pd.to_numeric(df['element_type'], errors='coerce').fillna(3) == pos_id
     pos_df = df[pos_mask]
     
     if len(pos_df) < 10: # Safety net for tiny datasets (pre-season)
         continue
         
     X = pos_df[features].fillna(0.0)
     
     expected_base = np.maximum(1.0, pos_df['cost_float'] * 0.5)
     raw_target = pd.to_numeric(pos_df['ep_next_raw'], errors='coerce').fillna(0.0)
     
     y = np.where(raw_target > 0, raw_target / expected_base, 1.0)
     y = np.clip(y, 0.5, 2.0)
     
     # Asymmetric Sample Weighting for True "Haulers" (≥ 8 points)
     # A threshold of 5.0 over-indexed Goalkeepers/Defenders (standard clean sheets = 6 pts).
     # 8.0 isolates true explosive upside (goals, double returns, penalty saves, max bonus).
     sample_weights = np.where(raw_target >= 8.0, 2.5, 1.0)
     
     # Lightweight Grid Search for Hyperparameters (Prevents GitHub Action timeouts)
     param_grid = {'max_depth': [2, 3], 'learning_rate': [0.03, 0.05]}
     
     xgb_base = XGBRegressor(n_estimators=50, random_state=42)
     xgb_tuned = GridSearchCV(xgb_base, param_grid, cv=2, scoring='neg_root_mean_squared_error')
     xgb_tuned.fit(X, y, sample_weight=sample_weights)
     
     lgb_base = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
     lgb_tuned = GridSearchCV(lgb_base, param_grid, cv=2, scoring='neg_root_mean_squared_error')
     lgb_tuned.fit(X, y, sample_weight=sample_weights)
     
     rf_model = RandomForestRegressor(n_estimators=50, max_depth=xgb_tuned.best_params_['max_depth'], random_state=42)
     rf_model.fit(X, y, sample_weight=sample_weights)
     
     xgb_preds = xgb_tuned.predict(X)
     lgb_preds = lgb_tuned.predict(X)
     rf_preds = rf_model.predict(X)
     
     # Weighted ensemble adjustment scalar
     ensemble_scalars = (0.4 * xgb_preds) + (0.4 * lgb_preds) + (0.2 * rf_preds)
     df.loc[pos_mask, 'ml_scalar'] = np.clip(ensemble_scalars, 0.6, 1.6)

 return df['ml_scalar'].values

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets for ML Pipeline...")
    
    top10k_eo_dict = get_livefpl_top10k_eo()
    crowd_xmins_dict = get_crowdsourced_xmins(fpl_df)
    
    custom_xmins_dict = {}
    xmins_env = os.getenv("XMINS_INPUT", "")
    if xmins_env and xmins_env.strip():
        try:
            raw_dict = json.loads(xmins_env.replace("'", '"'))
            custom_xmins_dict = {str(k).strip().lower(): float(v) for k, v in raw_dict.items()}
        except json.JSONDecodeError:
            for override in xmins_env.split(","):
                if ":" in override:
                    k, v = override.split(":")
                    try:
                        custom_xmins_dict[k.strip().lower()] = float(v.strip())
                    except ValueError: pass
        logger.info(f"Loaded {len(custom_xmins_dict)} manual xMins overrides into ML Pipeline.")
    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fbref_df['clean_team'] = fbref_df['team'].apply(strip_accents) if 'team' in fbref_df.columns else ""
        
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        df = pd.merge(fpl_df, fbref_df, left_on='matched_fbref_name', right_on='clean_fbref_name', how='left', suffixes=('', '_fbref'))
        logger.info(f"FPL to FBref Match Rate: {df['matched_fbref_name'].notna().mean() * 100:.1f}%")
    else:
        df = fpl_df.copy()

    # --- UNDERSTAT 4-GW ROLLING DATA MERGE ---
    understat_df = fetch_understat_rolling_data()
    if not understat_df.empty:
        understat_df['clean_understat_name'] = understat_df['name'].apply(strip_accents)
        df['clean_full_name'] = (df['first_name'].astype(str) + " " + df['second_name'].astype(str)).apply(strip_accents)
        df = pd.merge(df, understat_df, left_on='clean_full_name', right_on='clean_understat_name', how='left', suffixes=('', '_understat'))
        
    df['xg_per_shot'] = pd.to_numeric(df.get('xg_per_shot'), errors='coerce').fillna(0.0)
    df['xa_per_kp'] = pd.to_numeric(df.get('xa_per_kp'), errors='coerce').fillna(0.0)

    team_ratings = get_team_matchup_ratings(fbref_df, fpl_df)
    opp_mapping = get_upcoming_opponent_mapping()
    
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
    teams_short_by_id = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}

    def resolve_opp_rating(team_id):
        try:
            opp_ids = opp_mapping.get(int(team_id), [])
            if not opp_ids: return 0.0 # Blank Gameweek (0.0 multiplier kills EV)
            
            ratings = []
            for opp_id in opp_ids:
                opp_short = teams_short_by_id.get(opp_id, "")
                if opp_short in team_ratings: ratings.append(team_ratings[opp_short])
                else:
                    opp_fbref = FPL_TO_FBREF_TEAM.get(opp_short, "")
                    if opp_fbref in team_ratings: ratings.append(team_ratings[opp_fbref])
                    else: ratings.append(1.0)
            
            # Sum the ratings: 2 matches = ~2.0 multiplier, accurately scaling DGW Expected Value
            return sum(ratings)
        except: pass
        return 1.0

    df['opponent_def_rating'] = df['team'].apply(resolve_opp_rating)
    df['cost_float'] = (pd.to_numeric(df['now_cost'], errors='coerce').fillna(40) if 'now_cost' in df.columns else 40) / 10.0
    df['global_own'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0.0) if 'selected_by_percent' in df.columns else 0.0
    df['fb_mins'] = pd.to_numeric(df['minutes_played'], errors='coerce').fillna(0.0) if 'minutes_played' in df.columns else 0.0
    df['xgc_90_num'] = pd.to_numeric(df['expected_goals_conceded_per_90'], errors='coerce').fillna(1.35) if 'expected_goals_conceded_per_90' in df.columns else 1.35
    df['ep_next_raw'] = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0.0) if 'ep_next' in df.columns else 0.0
    df['form_raw'] = pd.to_numeric(df['form'], errors='coerce').fillna(0.0) if 'form' in df.columns else 0.0

    df['fpl_mins_played'] = (pd.to_numeric(df['minutes'], errors='coerce').fillna(0.0) if 'minutes' in df.columns else pd.Series(0.0, index=df.index)).clip(lower=0.001)
    
    cbit_cols = [c for c in ['clearances', 'blocks', 'interceptions', 'tackles'] if c in df.columns]
    df['fpl_cbit'] = df[cbit_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1) if cbit_cols else 0.0
    df['fpl_cbirt'] = df['fpl_cbit'] + (pd.to_numeric(df['recoveries'], errors='coerce').fillna(0) if 'recoveries' in df.columns else 0)

    df['fpl_cbit_90'] = (df['fpl_cbit'] / df['fpl_mins_played']) * 90.0
    df['fpl_cbirt_90'] = (df['fpl_cbirt'] / df['fpl_mins_played']) * 90.0

    fb_xg = pd.to_numeric(df['fbref_xg'], errors='coerce').fillna(0.0) if 'fbref_xg' in df.columns else 0.0
    fb_xag = pd.to_numeric(df['fbref_xag'], errors='coerce').fillna(0.0) if 'fbref_xag' in df.columns else 0.0
    native_xgi = pd.to_numeric(df['expected_goal_involvements_per_90'], errors='coerce').fillna(0.0) if 'expected_goal_involvements_per_90' in df.columns else 0.0
    
    df['combined_xgi'] = np.where(
        df['fb_mins'] > 270.0,
        np.where(native_xgi > 0, (0.60 * ((fb_xg + fb_xag) / (df['fb_mins'] / 90.0))) + (0.40 * native_xgi), ((fb_xg + fb_xag) / (df['fb_mins'] / 90.0))),
        native_xgi
    )

    # --- DOMAIN BASELINE EVALUATION ---
    # Calculates structural baseline EV preserving premium player pricing curves
    pos_ids = pd.to_numeric(df['element_type'], errors='coerce').fillna(3).astype(int)
    base_evs = np.where(
        pos_ids == 1, 3.5 + (df['cost_float'] - 4.0) * 0.20,
        np.where(
            pos_ids == 2, 3.2 + (df['cost_float'] - 4.0) * 0.30 + (df['combined_xgi'] * 0.10),
            np.where(
                pos_ids == 3, 3.0 + (df['cost_float'] - 4.5) * 0.35 + (df['combined_xgi'] * 0.15),
                3.2 + (df['cost_float'] - 4.5) * 0.40 + (df['combined_xgi'] * 0.20)
            )
        )
    )
    df['domain_base_ev'] = base_evs

    # --- EXECUTE TRI-MODEL MACHINE LEARNING REGRESSION ---
    ml_scalars = execute_tri_model_regression(df)
    df['ml_base_ev'] = df['domain_base_ev'] * ml_scalars

    projections = {}
    crowd_xmins_dict, crowd_evs_dict = get_crowdsourced_data(fpl_df)

    for idx, row in df.iterrows():
        pid_int = int(row['id'])
        pid = str(pid_int)
        web_name = str(row.get('web_name', 'Unknown'))
        
        # ... [Keep existing EO and Price Delta calculation logic] ...

        original_xmins = estimate_xmins(player_obj)
        
        # Retrieve crowdsourced xMins and EV arrays if available
        raw_xmins_list = crowd_xmins_dict.get(pid_int, [original_xmins] * 8)
        raw_ev_list = crowd_evs_dict.get(pid_int, None)

        # Handle Human Overrides passed via XMINS_INPUT
        if web_name.lower() in custom_xmins_dict or strip_accents(web_name) in custom_xmins_dict:
            override_val = custom_xmins_dict.get(web_name.lower(), custom_xmins_dict.get(strip_accents(web_name)))
            raw_xmins_list = [float(override_val)] * 8

        # Ensure arrays are exactly 8 items long
        if len(raw_xmins_list) < 8:
            raw_xmins_list += [raw_xmins_list[-1]] * (8 - len(raw_xmins_list))
        raw_xmins_list = raw_xmins_list[:8]

        weekly_evs = []
        weekly_xmins = []

        for t in range(8):
            target_xmins = min(90.0, float(raw_xmins_list[t]))
            
            if raw_ev_list and len(raw_ev_list) >= 8:
                # Use the exact 8-GW point projection from the CSV
                base_ev_t = float(raw_ev_list[t])
                
                # Rescale EV if a human override modified xMins for this gameweek
                orig_csv_mins = max(1.0, float(raw_xmins_list[t]))
                if web_name.lower() in custom_xmins_dict or strip_accents(web_name) in custom_xmins_dict:
                    calculated_ev = base_ev_t * (target_xmins / orig_csv_mins)
                else:
                    calculated_ev = base_ev_t
            else:
                # Heuristic fallback if CSV player match is missing
                base_ev = row['ml_base_ev']
                prob_60 = 1.0 / (1.0 + np.exp(-0.15 * (target_xmins - 60.0)))
                app_ev = (prob_60 * 2.0) + ((1.0 - prob_60) * (target_xmins / 60.0))
                attacking_ev = max(0.0, base_ev - 2.0) * (target_xmins / 90.0)
                calculated_ev = app_ev + attacking_ev

            weekly_evs.append(round(max(0.0, calculated_ev), 2))
            weekly_xmins.append(round(target_xmins, 1))

        final_xmins = weekly_xmins[0]
        calculated_ev = weekly_evs[0]

        # Position-specific variance boundaries
        pos_id = int(row.get('element_type', 3))
        mins_variance_penalty = max(0.5, final_xmins / 90.0)
        sigma = calculated_ev * (0.45 if pos_id == 3 else (0.40 if pos_id == 4 else 0.30)) / mins_variance_penalty

        projections[pid] = {
            "name": web_name,
            "ml_xmins": final_xmins,
            "ml_xmins_matrix": weekly_xmins,
            "ml_ev_1gw": calculated_ev,
            "ml_ev_matrix": weekly_evs,
            "ml_ev_8gw": round(sum(weekly_evs), 2),
            "mc_floor_ev": round(max(0.0, calculated_ev - sigma), 2),
            "mc_ceiling_ev": round(calculated_ev + (sigma * 1.25), 2),
            "top_10k_eo": round(top_10k_eo, 2),
            "predicted_price_delta": predicted_delta
        }
        
    logger.info(f"Successfully generated Residual ML projections for {len(projections)} players.")
    return projections