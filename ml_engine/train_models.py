import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("ModelTrainer")

class ModelTrainer:
    def __init__(self):
        self.team_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.share_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.xmins_model = lgb.LGBMRegressor(n_estimators=100)

    def run_pipeline(self, df):
        logger.info("Executing individualized projection engine...")
        projections = {}

        if df.empty:
            return projections

        for idx, row in df.iterrows():
            pid = str(row.get('id', 0))
            if pid == "0":
                continue

            xmins = row.get('calculated_xmins', 0.0)
            xmins = 0.0 if pd.isna(xmins) else float(xmins)

            ev_1gw = row.get('calculated_ev_1gw', 0.0)
            ev_1gw = 0.0 if pd.isna(ev_1gw) else float(ev_1gw)

            pos_id = row.get('element_type', 3)

            decay_factor = 6.8 if pos_id in [1, 2] else 7.2
            ev_8gw = round(ev_1gw * decay_factor, 2)

            sigma = ev_1gw * (0.45 if pos_id == 3 else (0.40 if pos_id == 4 else 0.30))
            floor = max(0.0, round(ev_1gw - (sigma * 1.5), 1))
            ceiling = round(ev_1gw + (sigma * 2.2), 1)

            projections[pid] = {
                "ml_xmins": xmins,
                "ml_ev_1gw": ev_1gw,
                "ml_ev_8gw": ev_8gw,
                "ml_variance_floor": floor,
                "ml_variance_ceiling": ceiling
            }

        return projections