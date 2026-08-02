"""
ml_engine/train_models.py — XGBoost Predictive Engine (FBref integration)
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    """
    Merges datasets and runs an XGBoost regression to predict Expected Value (EV).
    """
    logger.info("Aligning FPL and FBref datasets for Machine Learning...")
    
    # 1. Clean names for merging
    fpl_df['match_name'] = fpl_df['web_name'].astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
    
    if not fbref_df.empty:
        fbref_df['match_name'] = fbref_df['name'].astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
        fbref_df = fbref_df.drop_duplicates(subset=['match_name'])
        df = pd.merge(fpl_df, fbref_df, on='match_name', how='left')
    else:
        df = fpl_df.copy()
    
    # ---------------------------------------------------------
    # FAILSAFE: Guarantee columns exist even if FBref fails
    # ---------------------------------------------------------
    essential_cols = ['fbref_xg', 'fbref_npxg', 'fbref_xag', 'minutes_played']
    for col in essential_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing. Defaulting to 0.0.")
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