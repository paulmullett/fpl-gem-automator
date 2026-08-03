"""
ml_engine/train_models.py — XGBoost Predictive Engine (Transfer-Aware Global Fallback)
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import logging
import unicodedata
import difflib

logger = logging.getLogger(__name__)

FPL_TO_FBREF_TEAM = {
    "ARS": "Arsenal",
    "AVL": "Aston Villa",
    "BOU": "Bournemouth",
    "BRE": "Brentford",
    "BHA": "Brighton",
    "CHE": "Chelsea",
    "COV": "Coventry",
    "CRY": "Crystal Palace",
    "EVE": "Everton",
    "FUL": "Fulham",
    "HUL": "Hull",
    "IPI": "Ipswich",
    "LEE": "Leeds",
    "LEI": "Leicester",
    "LIV": "Liverpool",
    "MCI": "Manchester City",
    "MUN": "Manchester Utd",
    "NEW": "Newcastle",
    "NFO": "Nottingham", 
    "SOU": "Southampton",
    "SUN": "Sunderland",
    "TOT": "Tottenham",
    "WHU": "West Ham",
    "WOL": "Wolves"
}

def strip_accents(text):
    try:
        text = str(text)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    except Exception:
        return str(text).lower().strip()

def match_logic(full_name, web_name, second_name, first_name, name_pool):
    """Reusable logic for both Team-Bounded and Global searches."""
    # 1. Exact Exact
    if full_name in name_pool: return full_name
    if web_name in name_pool: return web_name
    
    # 2. Token Matching (e.g., "Senesi" inside "Marcos Senesi")
    for fb_name in name_pool:
        fb_tokens = set(fb_name.split())
        if second_name and second_name in fb_tokens:
            if not first_name or (first_name[0] == fb_name[0]):
                return fb_name
        if web_name and web_name in fb_tokens:
            return fb_name
        if '-' in second_name:
            for part in second_name.split('-'):
                if len(part) > 3 and part in fb_tokens:
                    return fb_name
                    
    # 3. Forgiving Fuzzy Match
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
    
    team_code = fpl_row.get('team_code', '')
    target_team = strip_accents(FPL_TO_FBREF_TEAM.get(team_code, ''))
    
    # --- PHASE 1: TEAM-BOUNDED SEARCH (Catches 95% of stable players) ---
    if target_team:
        team_pool = fbref_df[fbref_df['clean_team'].str.contains(target_team, na=False, regex=False)]['clean_fbref_name'].tolist()
        if team_pool:
            match = match_logic(full_name, web_name, second_name, first_name, team_pool)
            if match: return match

    # --- PHASE 2: GLOBAL FALLBACK SEARCH (Catches Summer Transfers) ---
    global_pool = fbref_df['clean_fbref_name'].tolist()
    match = match_logic(full_name, web_name, second_name, first_name, global_pool)
    if match: return match
    
    return None

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets using Team-Bounded & Transfer-Aware Global Logic...")
    
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
    df['pos_type'] = pd.to_numeric(df['element_type'], errors='coerce').fillna(3)
    
    prior_xg = {1: 0.00, 2: 0.04, 3: 0.15, 4: 0.38}
    prior_xag = {1: 0.00, 2: 0.06, 3: 0.12, 4: 0.15}
    prior_90s = 900.0 / 90.0 
    
    df['prior_xg90'] = df['pos_type'].map(prior_xg)
    df['prior_xag90'] = df['pos_type'].map(prior_xag)
    
    player_90s = df['minutes_played'] / 90.0
    
    df['xg_per_90'] = (df['fbref_xg'] + (df['prior_xg90'] * prior_90s)) / (player_90s + prior_90s)
    df['xag_per_90'] = (df['fbref_xag'] + (df['prior_xag90'] * prior_90s)) / (player_90s + prior_90s)
    
    # ---------------------------------------------------------
    # 3. DEFINE MODEL ARCHITECTURE
    # ---------------------------------------------------------
    features = ['cost', 'xg_per_90', 'xag_per_90']
    X = df[features].copy()
    
    y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0.0)
    
    logger.info(f"Training XGBoost Regressor on {len(X)} players...")
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, objective='reg:squarederror')
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
            "ml_ev_1gw": round(base_ev, 2), "ml_ev_8gw": round(base_ev * 8 * 0.95, 2),
            "ml_variance_floor": round(base_ev * 0.6, 2), "ml_variance_ceiling": round(base_ev * 1.5, 2)
        }
        
    logger.info(f"Successfully generated ML projections for {len(projections)} players.")
    return projections