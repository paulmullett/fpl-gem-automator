"""
ml_engine/data_ingestion.py — Core Data Ingestion Module (MultiIndex Stabilized)
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
        leagues = ["Big 5 European Leagues Combined"]
        
    logger.info(f"Fetching FBref underlying stats for {leagues} (Season {seasons})...")
    try:
        fbref = sd.FBref(leagues=leagues, seasons=seasons)
        stats_df = fbref.read_player_season_stats(stat_type="standard")
        
        # ROBUST MULTI-INDEX FLATTENING: Handles hierarchical soccerdata column tuples safely
        if isinstance(stats_df.columns, pd.MultiIndex):
            stats_df.columns = [str(col[-1]).strip() if col[-1] else str(col[0]).strip() for col in stats_df.columns.values]
        else:
            stats_df.columns = [str(c).strip() for c in stats_df.columns]
            
        stats_df = stats_df.reset_index()
        
        # Clean any leftover tuple artifacts after reset_index
        stats_df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in stats_df.columns]
        
        clean_df = pd.DataFrame()
        col_lower = {str(c).lower(): c for c in stats_df.columns}
        
        # Safe column mapping dictionary
        name_key = next((col_lower[k] for k in col_lower if 'player' in k or 'name' in k), None)
        if name_key:
            clean_df['name'] = stats_df[name_key]
            
        min_key = next((col_lower[k] for k in col_lower if ('min' in k or 'minute' in k) and '90' not in k), None)
        if min_key:
            clean_df['minutes_played'] = stats_df[min_key]
            
        xg_key = next((col_lower[k] for k in col_lower if k == 'xg' or (('xg' in k or 'goal' in k) and 'npxg' not in k and 'xag' not in k)), None)
        clean_df['fbref_xg'] = stats_df[xg_key] if xg_key else 0.0

        npxg_key = next((col_lower[k] for k in col_lower if 'npxg' in k), None)
        clean_df['fbref_npxg'] = stats_df[npxg_key] if npxg_key else 0.0

        xag_key = next((col_lower[k] for k in col_lower if 'xag' in k or 'xa' in k), None)
        clean_df['fbref_xag'] = stats_df[xag_key] if xag_key else 0.0

        # Failsafes
        for col in ['name', 'minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']:
            if col not in clean_df:
                clean_df[col] = 0.0 if col != 'name' else "Unknown"
        
        clean_df['name'] = clean_df['name'].astype(str)
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
            
        grouped_df = clean_df.groupby('name', as_index=False)[numeric_cols].sum()

        logger.info(f"Successfully scraped {len(grouped_df)} global player records from FBref natively.")
        return grouped_df

    except Exception as e:
        logger.error(f"Error fetching FBref data: {e}")
        return pd.DataFrame()