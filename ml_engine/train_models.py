"""
ml_engine/train_models.py — XGBoost Predictive Engine (Context-Aware Matching)
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import logging
import unicodedata
import difflib

logger = logging.getLogger(__name__)

def strip_accents(text):
    try:
        text = str(text)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    except Exception:
        return str(text).lower().strip()

def find_best_match(fpl_row, fbref_df):
    """
    Multi-tier matching: Exact -> Compact Substring -> Team-Filtered Fuzzy -> Direct Fuzzy.
    """
    web_name = strip_accents(fpl_row.get('web_name', ''))
    first_name = strip_accents(fpl_row.get('first_name', ''))
    second_name = strip_accents(fpl_row.get('second_name', ''))
    full_name = f"{first_name} {second_name}".strip()
    fpl_team = strip_accents(fpl_row.get('team_code', ''))
    
    fbref_names = fbref_df['clean_fbref_name'].tolist()
    
    # 1. Exact full name match
    if full_name in fbref_names:
        return full_name
        
    # 2. Exact web name match
    if web_name in fbref_names:
        return web_name
        
    # Prepare compact strings (strip spaces, dashes, dots)
    web_compact = web_name.replace(" ", "").replace("-", "").replace(".", "")
    full_compact = full_name.replace(" ", "").replace("-", "").replace(".", "")
    second_compact = second_name.replace(" ", "").replace("-", "").replace(".", "")
    
    # 3. Compact substring matching
    for idx, row in fbref_df.iterrows():
        fb_name = row['clean_fbref_name']
        fb_compact = fb_name.replace(" ", "").replace("-", "").replace(".", "")
        if web_compact in fb_compact and len(web_compact) > 2:
            return fb_name
        if second_compact in fb_compact and len(second_compact) > 3:
            return fb_name

    # 4. Team-filtered fuzzy match (if FBref team data is available)
    if 'team' in fbref_df.columns:
        team_subset = fbref_df[fbref_df['clean_team'].str.contains(fpl_team, na=False)]
        if not team_subset.empty:
            team_names = team_subset['clean_fbref_name'].tolist()
            fuzzy_team = difflib.get_close_matches(full_name, team_names, n=1, cutoff=0.50)
            if fuzzy_team:
                return fuzzy_team[0]
            fuzzy_web = difflib.get_close_matches(web_name, team_names, n=1, cutoff=0.50)
            if fuzzy_web:
                return fuzzy_web[0]

    # 5. Global forgiving fuzzy match
    fuzzy = difflib.get_close_matches(full_name, fbref_names, n=1, cutoff=0.50)
    if fuzzy:
        return fuzzy[0]
        
    return None

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets using Context-Aware Fuzzy Logic...")
    
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
    else:
        df = fpl_df.copy()
    
    essential_cols = ['fbref_xg', 'fbref_npxg', 'fbref_xag', 'minutes_played']
    for col in essential_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df.fillna({col: 0.0 for col in essential_cols}, inplace=True)
    
    logger.info("Engineering per-90 metrics for the XGBoost model...")
    df['cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0
    
    df['xg_per_90'] = np.where(df['minutes_played'] > 0, (df['fbref_xg'] / df['minutes_played']) * 90, 0)
    df['xag_per_90'] = np.where(df['minutes_played'] > 0, (df['fbref_xag'] / df['minutes_played']) * 90, 0)
    
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