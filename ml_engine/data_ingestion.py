"""
ml_engine/data_ingestion.py — Core Data Ingestion Module (Guaranteed Name Resolution)
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
        
        # Flatten MultiIndex columns robustly
        stats_df = stats_df.reset_index()
        flat_cols = []
        for col in stats_df.columns:
            if isinstance(col, tuple):
                parts = [str(p).strip() for p in col if p and not str(p).lower().startswith('unnamed')]
                flat_cols.append('_'.join(parts) if parts else str(col[-1]))
            else:
                flat_cols.append(str(col).strip())
        stats_df.columns = flat_cols
        
        clean_df = pd.DataFrame()
        col_lower = {str(c).lower(): c for c in stats_df.columns}
        
        # GUARANTEED PLAYER NAME RESOLUTION: Explicitly find 'player' column, ignoring team/squad/nation
        name_col = None
        for k, orig_col in col_lower.items():
            if 'player' in k and 'team' not in k and 'squad' not in k and 'nation' not in k:
                name_col = orig_col
                break
        if not name_col:
            # Fallback search for any column containing 'name'
            name_col = next((col_lower[k] for k in col_lower if 'name' in k), stats_df.columns[0])
            
        clean_df['name'] = stats_df[name_col]
        
        # Extract metrics safely
        min_key = next((col_lower[k] for k in col_lower if ('min' in k or 'minute' in k) and '90' not in k), None)
        clean_df['minutes_played'] = stats_df[min_key] if min_key else 0.0
        
        xg_key = next((col_lower[k] for k in col_lower if 'xg' in k and 'npxg' not in k and 'xag' not in k), None)
        clean_df['fbref_xg'] = stats_df[xg_key] if xg_key else 0.0

        npxg_key = next((col_lower[k] for k in col_lower if 'npxg' in k), None)
        clean_df['fbref_npxg'] = stats_df[npxg_key] if npxg_key else 0.0

        xag_key = next((col_lower[k] for k in col_lower if 'xag' in k or ('xa' in k and 'xg' not in k)), None)
        clean_df['fbref_xag'] = stats_df[xag_key] if xag_key else 0.0

        # Type cleaning & Failsafes
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