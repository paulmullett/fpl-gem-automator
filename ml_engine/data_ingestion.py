"""
ml_engine/data_ingestion.py — Multi-Source Football Data Ingestion Engine
"""
import logging
import pandas as pd
import requests
import soccerdata as sd

logger = logging.getLogger(__name__)

def fetch_fpl_data() -> pd.DataFrame:
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    logger.info("Fetching live player data from official FPL API...")
    try:
        response = requests.get(url, headers={"User-Agent": "FPL-ML-Pipeline/1.0"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        elements_df = pd.DataFrame(data['elements'])
        teams_map = {t['id']: t['short_name'] for t in data['teams']}
        types_map = {e['id']: e['singular_name_short'] for e in data['element_types']}
        
        elements_df['team_code'] = elements_df['team'].map(teams_map)
        elements_df['position'] = elements_df['element_type'].map(types_map)
        
        logger.info(f"Successfully retrieved {len(elements_df)} active players from FPL API.")
        return elements_df
    except Exception as e:
        logger.error(f"Failed to fetch FPL API data: {e}")
        return pd.DataFrame()

def fetch_fbref_data(leagues="ENG-Premier League", seasons="2425") -> pd.DataFrame:
    logger.info(f"Fetching FBref underlying stats for {leagues} (Season {seasons})...")
    try:
        fbref = sd.FBref(leagues=leagues, seasons=seasons)
        stats_df = fbref.read_player_season_stats(stat_type="standard")
        stats_df = stats_df.reset_index()
        
        clean_df = pd.DataFrame()
        str_columns = [str(c).lower() for c in stats_df.columns]
        
        for orig_col, str_col in zip(stats_df.columns, str_columns):
            if ('player' in str_col or 'name' in str_col) and 'name' not in clean_df:
                clean_df['name'] = stats_df[orig_col]
            elif ('team' in str_col or 'squad' in str_col) and 'team' not in clean_df:
                clean_df['team'] = stats_df[orig_col]
            elif 'min' in str_col and '90' not in str_col and 'minutes_played' not in clean_df:
                clean_df['minutes_played'] = stats_df[orig_col]
            elif 'npxg' in str_col and '90' not in str_col and 'fbref_npxg' not in clean_df:
                clean_df['fbref_npxg'] = stats_df[orig_col]
            elif 'xag' in str_col and '90' not in str_col and 'fbref_xag' not in clean_df:
                clean_df['fbref_xag'] = stats_df[orig_col]
            elif 'xg' in str_col and 'npxg' not in str_col and '90' not in str_col and 'fbref_xg' not in clean_df:
                clean_df['fbref_xg'] = stats_df[orig_col]

        for col in ['name', 'team', 'minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']:
            if col not in clean_df:
                clean_df[col] = 0.0 if col not in ['name', 'team'] else "Unknown"
        
        clean_df['name'] = clean_df['name'].astype(str)
        clean_df['team'] = clean_df['team'].astype(str)
        
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)

        logger.info(f"Successfully scraped {len(clean_df)} player records from FBref natively.")
        return clean_df

    except Exception as e:
        logger.error(f"Error fetching FBref data: {e}")
        return pd.DataFrame()