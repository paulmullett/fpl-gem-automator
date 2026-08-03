"""
ml_engine/train_models.py — Bottom-Up Expected Points (xP) Engine
"""

import pandas as pd
import numpy as np
import logging
import unicodedata
import difflib

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
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
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

def generate_ml_projections(fpl_df: pd.DataFrame, fbref_df: pd.DataFrame) -> dict:
    logger.info("Aligning FPL and FBref datasets...")
    
    if not fbref_df.empty:
        fbref_df['clean_fbref_name'] = fbref_df['name'].apply(strip_accents)
        fbref_df['clean_team'] = fbref_df['team'].apply(strip_accents) if 'team' in fbref_df.columns else ""
        fpl_df['matched_fbref_name'] = fpl_df.apply(lambda row: find_best_match(row, fbref_df), axis=1)
        df = pd.merge(fpl_df, fbref_df, left_on='matched_fbref_name', right_on='clean_fbref_name', how='left')
    else:
        df = fpl_df.copy()
        
    essential_cols = ['fbref_xg', 'fbref_npxg', 'fbref_xag', 'minutes_played']
    for col in essential_cols:
        if col not in df.columns: df[col] = 0.0
    df.fillna({col: 0.0 for col in essential_cols}, inplace=True)

    # ---------------------------------------------------------
    # BOTTOM-UP EXPECTED POINTS (xP) CALCULATOR
    # ---------------------------------------------------------
    logger.info("Generating Bottom-Up Expected Points (xP) Projections...")
    
    df['cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0
    df['pos_type'] = pd.to_numeric(df['element_type'], errors='coerce').fillna(3) # 1:GK, 2:DEF, 3:MID, 4:FWD
    
    # Adaptive Bayesian Priors (Reduced gravity to 3 full matches to preserve superstar metrics)
    prior_xg = {1: 0.00, 2: 0.03, 3: 0.12, 4: 0.32}
    prior_xag = {1: 0.00, 2: 0.04, 3: 0.10, 4: 0.12}
    
    player_90s = df['minutes_played'] / 90.0
    prior_90s = 3.0 
    
    df['prior_xg90'] = df['pos_type'].map(prior_xg)
    df['prior_xag90'] = df['pos_type'].map(prior_xag)
    
    df['xg_per_90'] = (df['fbref_xg'] + (df['prior_xg90'] * prior_90s)) / (player_90s + prior_90s)
    df['xag_per_90'] = (df['fbref_xag'] + (df['prior_xag90'] * prior_90s)) / (player_90s + prior_90s)
    
    # Position-Specific Point Rules
    goal_pts = {1: 6.0, 2: 6.0, 3: 5.0, 4: 4.0}
    cs_pts = {1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}
    
    df['pts_goal'] = df['pos_type'].map(goal_pts)
    df['pts_cs'] = df['pos_type'].map(cs_pts)
    
    top_teams = ['ARS', 'MCI', 'LIV']
    df['cs_per_90'] = df['team_code'].apply(lambda t: 0.38 if t in top_teams else 0.22)

    projections = {}
    for _, row in df.iterrows():
        pid = str(row['id'])
        status = str(row.get('status', 'a'))
        
        past_mins = float(row.get('minutes_played', 0.0))
        cost = float(row.get('cost', 0.0))
        
        if status != 'a':
            status_mult = 0.0 if status in ['i', 's', 'u'] else 0.5
        else:
            status_mult = 1.0
            
        # Realistic xMins baseline
        if status_mult > 0:
            if past_mins > 1500 or cost >= 9.0:
                xmins = 85.0 * status_mult
            elif past_mins > 500:
                xmins = 65.0 * status_mult
            else:
                xmins = 35.0 * status_mult
        else:
            xmins = 0.0

        match_fraction = xmins / 90.0
        
        # Component Calculations
        exp_goals = row['xg_per_90'] * match_fraction
        exp_assists = row['xag_per_90'] * match_fraction
        exp_cs = row['cs_per_90'] * match_fraction
        
        pts_goals = exp_goals * row['pts_goal']
        pts_assists = exp_assists * 3.0
        pts_cs = exp_cs * row['pts_cs']
        pts_appearance = 2.0 if xmins >= 60 else (1.0 if xmins > 0 else 0.0)
        
        # Base Single Gameweek Expected Value (xP)
        base_ev = pts_goals + pts_assists + pts_cs + pts_appearance
        
        if status == 'a' and xmins >= 60:
            base_ev = max(base_ev, 2.0)

        projections[pid] = {
            "ml_xmins": round(xmins, 1),
            "ml_ev_1gw": round(base_ev, 2),
            "ml_ev_8gw": round(base_ev * 8 * 0.95, 2),
            "ml_variance_floor": round(base_ev * 0.6, 2),
            "ml_variance_ceiling": round(base_ev * 1.5, 2)
        }
        
    logger.info(f"Successfully generated ML projections for {len(projections)} players.")
    return projections