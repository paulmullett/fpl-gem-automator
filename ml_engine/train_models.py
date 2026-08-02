"""
ml_engine/train_models.py — XGBoost Predictive Engine (Native FPL Opta Data)
"""

import pandas as pd
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

def generate_ml_projections(df: pd.DataFrame) -> dict:
    """
    Runs an XGBoost regression to predict Expected Value (EV) using native Opta metrics.
    """
    logger.info("Engineering ML metrics using FPL's native Opta data...")
    
    # 1. FEATURE ENGINEERING
    df['cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0
    
    # Extract official FPL Opta underlying stats
    df['opt_xg_90'] = pd.to_numeric(df.get('expected_goals_per_90', 0), errors='coerce').fillna(0.0)
    df['opt_xa_90'] = pd.to_numeric(df.get('expected_assists_per_90', 0), errors='coerce').fillna(0.0)
    df['opt_xgc_90'] = pd.to_numeric(df.get('expected_goals_conceded_per_90', 0), errors='coerce').fillna(0.0)
    df['minutes_played'] = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0.0)
    
    # 2. DEFINE MODEL ARCHITECTURE
    # We now train the model based on Price, xG/90, xA/90, and Expected Goals Conceded/90
    features = ['cost', 'opt_xg_90', 'opt_xa_90', 'opt_xgc_90']
    X = df[features].copy()
    
    # Proxy Target for initial weights
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
    
    # 3. GENERATE PREDICTIONS
    df['ml_predicted_ev'] = model.predict(X)
    df['ml_predicted_ev'] = df['ml_predicted_ev'].clip(lower=0.0)
    
    # 4. BUILD JSON PAYLOAD
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