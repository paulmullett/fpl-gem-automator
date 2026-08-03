"""
ml_engine/train_models.py — XGBoost Predictive Engine (Bayesian Shrinkage)
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import logging
import unicodedata
import difflib

logger = logging.getLogger(__name__)

NAME_OVERRIDES = {}

def strip_accents(text):
    try:
        text = str(text)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    except Exception:
        return str(text).lower().strip()

def find_best_match(fpl_row, fbref_df):
    web_name = strip_accents(fpl_row.get('web_name', ''))
    first_name = strip_accents(fpl_row.get('first_name', ''))
    second_name = strip_accents(fpl_row.get('second_name', ''))
    full_name = f"{first_name} {second_name}".strip()
    fpl_team = strip_accents(fpl_row.get('team_code', ''))
    
    if web_name in NAME_OVERRIDES: return NAME_OVERRIDES[web_name]
    if full_name in NAME_OVERRIDES: return NAME_OVERRIDES[full_name]
    
    fbref_names = fbref_df['clean_fbref_name'].tolist()
    
    if full_name in fbref_names: return full_name
    if web_name in fbref_names: return web_name
        
    web_compact = web_name.replace(" ", "").replace("-", "").replace(".", "")
    full_compact = full_name.replace(" ", "").replace("-", "").replace(".", "")
    second_compact = second_name.replace(" ", "").replace("-", "").replace(".", "")
    
    for idx, row in fbref_df.iterrows():
        fb_name = row['clean_fbref_name']
        fb_compact = fb_name.replace(" ", "").replace("-", "").replace(".", "")
        if web_compact in fb_compact and len(web_compact) > 2: return fb_name
        if second_compact in fb_compact and len(second_compact) > 3: return fb_name

    if 'team' in fbref_df.columns:
        team_subset = fbref_df[fbref_df['clean_team'].str.contains(fpl_team, na=False)]
        if not team_subset.empty:
            team_names = team_subset['clean_fbref_name'].tolist()
            fuzzy_team = difflib.get_close_matches(full_name, team_names, n=1, cutoff=0.50)
            if fuzzy_team: return fuzzy_team[0]
            fuzzy_web = difflib.get_close_matches(web_name, team_names, n=1, cutoff=0.50)
            if fuzzy_web: return fuzzy_web[0]

    fuzzy = difflib.get_close_matches(full_name, fbref_names, n=1, cutoff=0.50)
    if fuzzy: return fuzzy[0]
        
    return None

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets using Context-Aware Fuzzy Logic & Overrides...")
    
    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        if 'team' in fbref_df.columns:
            fbref_df['clean_team'] = fbref_df['team'].apply(strip_accents)
        else:
            fbref_df['clean_team'] = ""
            
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        
        df = pd.merge(
            fpl_df, 
            fbref_df, 
            left_on='matched_fbref_name', 
            right_on='clean_fbref_name', 
            how='left'
        )
        
        match_rate = df['matched_fbref_name'].notna().mean() * 100
        logger.info(f"FPL to FBref Match Rate: {match_rate:.1f}%")
        
        unmatched_fpl = df[df['matched_fbref_name'].isnull()]
        if not unmatched_fpl.empty:
            logger.warning(f"--- UNMATCHED FPL PLAYERS ({len(unmatched_fpl)}) ---")
            for _, row in unmatched_fpl.iterrows():
                cost = float(row.get('now_cost', 0)) / 10
                if cost > 4.5:
                    logger.warning(f"HIGH VALUE UNMATCHED: {row.get('web_name')} ({row.get('team_code')}) - £{cost}m [Full Name: {row.get('first_name')} {row.get('second_name')}]")
            logger.warning("-----------------------------------------")
            
    else:
        df = fpl_df.copy()
    
    essential_cols = ['fbref_xg', 'fbref_npxg', 'fbref_xag', 'minutes_played']
    for col in essential_cols:
        if col not in df.columns: df[col] = 0.0
            
    df.fillna({col: 0.0 for col in essential_cols}, inplace=True)
    
    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING: BAYESIAN SHRINKAGE
    # ---------------------------------------------------------
    logger.info("Applying Positional Bayesian Shrinkage to per-90 metrics...")
    df['cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0
    
    # FPL element_types: 1=GK, 2=DEF, 3=MID, 4=FWD
    df['pos_type'] = pd.to_numeric(df['element_type'], errors='coerce').fillna(3)
    
    # Define Positional Priors (Expected League Averages per 90)
    prior_xg = {1: 0.00, 2: 0.04, 3: 0.15, 4: 0.38}
    prior_xag = {1: 0.00, 2: 0.06, 3: 0.12, 4: 0.15}
    
    # The "Gravity" of the prior (900 minutes = 10 full matches)
    prior_90s = 900.0 / 90.0 
    
    df['prior_xg90'] = df['pos_type'].map(prior_xg)
    df['prior_xag90'] = df['pos_type'].map(prior_xag)
    
    player_90s = df['minutes_played'] / 90.0
    
    # Bayesian Formula: ((Player Total Stat) + (Prior Stat/90 * Prior 90s)) / (Player 90s + Prior 90s)
    df['xg_per_90'] = (df['fbref_xg'] + (df['prior_xg90'] * prior_90s)) / (player_90s + prior_90s)
    df['xag_per_90'] = (df['fbref_xag'] + (df['prior_xag90'] * prior_90s)) / (player_90s + prior_90s)
    
    # ---------------------------------------------------------
    # 3. DEFINE MODEL ARCHITECTURE
    # ---------------------------------------------------------
    features = ['cost', 'xg_per_90', 'xag_per_90']
    X = df[features].copy()
    
    y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0.0)
    
    logger.info(f"Training XGBoost Regressor on {len(X)} players...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X, y)
    
    df['ml_predicted_ev'] = model.predict(X)
    df['ml_predicted_ev'] = df['ml_predicted_ev'].clip(lower=0.0)
    
    projections = {}
    for _, row in df.iterrows():
        pid = str(row['id'])
        
        status_mult = 1.0 if row['status'] == 'a' else (0.0 if row['status'] in ['d', 'i', 's', 'u'] else 0.5)
        ml_xmins = min(90.0, float(row.get('minutes_played', 0.0) / 38.0) * status_mult)
        if row['status'] == 'a' and float(row.get('ep_next', 0)) > 1.5:
            ml_xmins = max(75.0, ml_xmins)
        
        base_ev = float(row['ml_predicted_ev'])
        
        projections[pid] = {
            "ml_xmins": round(ml_xmins, 1),
            "ml_ev_1gw": round(base_ev, 2),
            "ml_ev_8gw": round(base_ev * 8 * 0.95, 2),
            "ml_variance_floor": round(base_ev * 0.6, 2),
            "ml_variance_ceiling": round(base_ev * 1.5, 2)
        }
        
    logger.info(f"Successfully generated ML projections for {len(projections)} players.")
    return projections