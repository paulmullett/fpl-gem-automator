"""
ml_engine/train_models.py — Unified Dataset Alignment Engine
"""
import pandas as pd
import logging
import unicodedata
import difflib
from fpl_funcs import get_ensemble_ev

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
    "erling haaland": "erling haaland",
    "haaland": "erling haaland"
}

def strip_accents(text):
    try:
        text = str(text)
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8").lower().strip()
    except Exception:
        return str(text).lower().strip()

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
    
    fuzzy = difflib.get_close_matches(full_name, global_pool, n=1, cutoff=0.70)
    return fuzzy[0] if fuzzy else None

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets...")
    
    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        df = pd.merge(fpl_df, fbref_df, left_on='matched_fbref_name', right_on='clean_fbref_name', how='left')
    else:
        df = fpl_df.copy()

    projections = {}
    for _, row in df.iterrows():
        pid = str(row['id'])
        
        # Extract FBref underlying metrics
        fb_xg = float(row.get('fbref_xg', 0.0)) if pd.notna(row.get('fbref_xg')) else 0.0
        fb_xag = float(row.get('fbref_xag', 0.0)) if pd.notna(row.get('fbref_xag')) else 0.0
        fb_mins = float(row.get('minutes_played', 0.0)) if pd.notna(row.get('minutes_played')) else 0.0
        
        # Calculate underlying xGI per 90 from FBref if available
        fb_xgi_90 = ((fb_xg + fb_xag) / (fb_mins / 90.0)) if fb_mins > 270 else float(row.get('expected_goal_involvements_per_90', 0.0))
        
        # Structure enriched player payload to pass into fpl_funcs
        player_obj = {
            "id": row['id'],
            "name": row.get('web_name', 'Unknown'),
            "team": row.get('team_code', 'UNK'),
            "pos_id": int(row.get('element_type', 3)),
            "cost": float(row.get('now_cost', 40)) / 10.0,
            "own": float(row.get('selected_by_percent', 0.0)),
            "status": str(row.get('status', 'a')),
            "ep_next": float(row.get('ep_next', 0.0) or 0.0),
            "form": float(row.get('form', 0.0) or 0.0),
            "xgi_90": fb_xgi_90,
            "xgc_90": float(row.get('expected_goals_conceded_per_90', 1.35) or 1.35),
            "chance_of_playing_next_round": row.get('chance_of_playing_next_round'),
            "source_league": row.get('source_league', 'Premier_League'),
            "age": int(row.get('age', 25) or 25)
        }
        
        # DELEGATE TO THE RICH MATH ENGINE (fpl_funcs.py)
        calculated_ev = get_ensemble_ev(player_obj)
        
        projections[pid] = {
            "ml_xmins": 90.0 if player_obj["cost"] >= 7.0 else 65.0,
            "ml_ev_1gw": round(calculated_ev, 2),
            "ml_ev_8gw": round(calculated_ev * 8 * 0.95, 2),
            "ml_variance_floor": round(calculated_ev * 0.6, 2),
            "ml_variance_ceiling": round(calculated_ev * 1.5, 2)
        }
        
    logger.info(f"Successfully generated projections for {len(projections)} players using rich math engine.")
    return projections