"""
ml_engine/train_models.py — XGBoost Predictive Engine (Robust FBref Matching)
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import logging
import unicodedata
import difflib

logger = logging.getLogger(__name__)

def strip_accents(text):
    """Removes special characters and accents, converts to lowercase."""
    try:
        text = str(text)
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    except Exception:
        return str(text).lower().strip()

def find_best_match(fpl_row, fbref_names_list):
    """Cascading logic to find the best FBref name for an FPL player."""
    web_name = strip_accents(fpl_row.get('web_name', ''))
    first_name = strip_accents(fpl_row.get('first_name', ''))
    second_name = strip_accents(fpl_row.get('second_name', ''))
    full_name = f"{first_name} {second_name}".strip()
    
    # 1. Exact match on full name
    if full_name in fbref_names_list:
        return full_name
        
    # 2. Exact match on web name
    if web_name in fbref_names_list:
        return web_name
        
    # 3. Substring match (e.g., FPL 'salah' is in FBref 'mohamed salah')
    # We split into words to prevent 'dan' matching 'jordan'
    possible_matches = [name for name in fbref_names_list if web_name in name.split()]
    if len(possible_matches) == 1:
        return possible_matches[0]
        
    # 4. Fuzzy Match (string similarity > 75%)
    fuzzy = difflib.get_close_matches(full_name, fbref_names_list, n=1, cutoff=0.75)
    if fuzzy:
        return fuzzy[0]
        
    return None

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    """
    Merges datasets robustly and runs an XGBoost regression.
    """
    logger.info("Aligning FPL and FBref datasets using Fuzzy Logic...")
    
    if not fbref_df.empty:
        # Clean FBref names
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fbref_names_list = fbref_df['clean_fbref_name'].tolist()
        
        # Apply the matching algorithm to every FPL player
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_names_list), axis=1)
        
        # Merge on the newly matched names
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
    
    # ---------------------------------------------------------
    # FAILSAFE: Guarantee columns exist
    # ---------------------------------------------------------
    essential_cols = ['fbref_xg', 'fbref_npxg', 'fbref_xag', 'minutes_played']
    for col in essential_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df.fillna({col: 0.0 for col in essential_cols}, inplace=True)
    
    # 2. FEATURE ENGINEERING
    logger.info("Engineering per-90 metrics for the XGBoost model...")
    df['cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0
    
    df['xg_per_90'] = np.where(df['minutes_played'] > 0, (df['fbref_xg'] / df['minutes_played']) * 90, 0)
    df['xag_per_90'] = np.where(df['minutes_played'] > 0, (df['fbref_xag'] / df['minutes_played']) * 90, 0)
    
    # 3. DEFINE MODEL ARCHITECTURE
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
    
    # 4. GENERATE PREDICTIONS
    df['ml_predicted_ev'] = model.predict(X)
    df['ml_predicted_ev'] = df['ml_predicted_ev'].clip(lower=0.0)
    
    # 5. BUILD JSON PAYLOAD
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