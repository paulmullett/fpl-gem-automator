import pandas as pd
import numpy as np

class FeatureEngine:
    def __init__(self):
        self.ewma_span = 5

    def build_features(self, df):
        if df.empty:
            return df

        # Convert core numerical columns safely
        df['cost_m'] = pd.to_numeric(df['now_cost'], errors='coerce').fillna(40) / 10.0
        df['ep_next_num'] = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0.0)
        df['xgi_num'] = pd.to_numeric(df.get('expected_goal_involvements', 0), errors='coerce').fillna(0.0)
        df['xG_num'] = pd.to_numeric(df.get('expected_goals', 0), errors='coerce').fillna(0.0)
        df['xA_num'] = pd.to_numeric(df.get('expected_assists', 0), errors='coerce').fillna(0.0)
        df['minutes_num'] = pd.to_numeric(df.get('minutes', 0), errors='coerce').fillna(0.0)

        # Expected Minutes Engine (xMins with strict pd.isna check)
        def calculate_xmins(row):
            status = str(row.get('status', 'a')).lower()
            chance = row.get('chance_of_playing_next_round')
            
            # In FPL API, null/NaN chance means 100% available (no news flag)
            if pd.isna(chance) or chance is None:
                chance_val = 100.0
            else:
                try:
                    chance_val = float(chance)
                except (ValueError, TypeError):
                    chance_val = 100.0

            if status in ['u', 'i', 's'] or chance_val == 0.0:
                return 0.0
            
            base_mins = 90.0 if row['cost_m'] >= 6.0 or row['minutes_num'] > 500 else 65.0
            base_mins *= (chance_val / 100.0)

            if pd.isna(base_mins):
                return 0.0

            return round(float(base_mins), 1)

        df['calculated_xmins'] = df.apply(calculate_xmins, axis=1)

        # Baseline Position Weights for 1-GW EV modeling
        def calculate_base_ev(row):
            pos = row.get('element_type', 3)
            cost = row['cost_m']
            ep = row['ep_next_num']
            xgi = row['xgi_num']
            xmins = row['calculated_xmins']

            if pd.isna(xmins) or xmins == 0.0:
                return 0.0

            if ep > 0 and not pd.isna(ep):
                ev_base = ep
            else:
                if pos == 1:   # Goalkeeper
                    ev_base = 3.5 + (cost - 4.0) * 0.20
                elif pos == 2: # Defender
                    ev_base = 3.2 + (cost - 4.0) * 0.30 + (xgi * 0.10)
                elif pos == 3: # Midfielder
                    ev_base = 3.0 + (cost - 4.5) * 0.35 + (xgi * 0.15)
                else:          # Forward
                    ev_base = 3.2 + (cost - 4.5) * 0.40 + (xgi * 0.20)

            mins_factor = xmins / 90.0
            ev = ev_base * mins_factor

            if pd.isna(ev) or ev < 0:
                return 0.0
            return round(float(ev), 2)

        df['calculated_ev_1gw'] = df.apply(calculate_base_ev, axis=1)

        return df