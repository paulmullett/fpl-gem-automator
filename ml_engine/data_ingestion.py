"""
ml_engine/data_ingestion.py — Core Data Ingestion Module (Stabilized)
"""
import pandas as pd
import soccerdata as sd
import requests
import logging

logger = logging.getLogger(__name__)

def fetch_fpl_data() -> pd.DataFrame:
    logger.info("Fetching live player data from official FPL API...")
    try:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get('elements', []))
            logger.info(f"Successfully retrieved {len(df)} active players from FPL API.")
            return df
        else:
            logger.error("Failed to fetch FPL data.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching FPL data: {e}")
        return pd.DataFrame()

def fetch_fbref_data(leagues=None, seasons="2526") -> pd.DataFrame:
    if leagues is None:
        # Restricted strictly to soccerdata-supported FBref identifiers
        leagues = ["Big 5 European Leagues Combined"]
        
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
            elif 'min' in str_col and '90' not in str_col and 'minutes_played' not in clean_df:
                clean_df['minutes_played'] = stats_df[orig_col]
            elif 'npxg' in str_col and '90' not in str_col and 'fbref_npxg' not in clean_df:
                clean_df['fbref_npxg'] = stats_df[orig_col]
            elif 'xag' in str_col and '90' not in str_col and 'fbref_xag' not in clean_df:
                clean_df['fbref_xag'] = stats_df[orig_col]
            elif 'xg' in str_col and 'npxg' not in str_col and '90' not in str_col and 'fbref_xg' not in clean_df:
                clean_df['fbref_xg'] = stats_df[orig_col]

        # Failsafe for missing columns
        for col in ['name', 'minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']:
            if col not in clean_df:
                clean_df[col] = 0.0 if col != 'name' else "Unknown"
        
        # Clean and convert types
        clean_df['name'] = clean_df['name'].astype(str)
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
            
        # AGGREGATION: Combine stats for multi-club players
        grouped_df = clean_df.groupby('name', as_index=False)[numeric_cols].sum()

        logger.info(f"Successfully scraped {len(grouped_df)} global player records from FBref natively.")
        return grouped_df

    except Exception as e:
        logger.error(f"Error fetching FBref data: {e}")
        return pd.DataFrame()