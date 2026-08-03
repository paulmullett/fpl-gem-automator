"""
ml_engine/data_ingestion.py — Core Data Ingestion Module (Bulletproof Index Extraction)
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
        
        clean_df = pd.DataFrame()
        
        # 1. BULLETPROOF NAME EXTRACTION: Pull directly from the soccerdata MultiIndex level.
        # This completely bypasses the pandas column flattening bugs that caused the 69% match rate.
        if 'player' in stats_df.index.names:
            clean_df['name'] = stats_df.index.get_level_values('player').astype(str).str.strip()
        else:
            # Absolute fallback if index is structured differently
            temp_df = stats_df.reset_index()
            name_col = next((c for c in temp_df.columns if 'player' in str(c).lower() and 'team' not in str(c).lower()), temp_df.columns[0])
            clean_df['name'] = temp_df[name_col].astype(str).str.strip()
            
        # 2. Safely reset the index to move metrics into accessible columns
        stats_df = stats_df.reset_index()
        
        # 3. Bulletproof metric extraction bypassing tuple naming issues
        def get_metric(target_substring, exclude):
            for col in stats_df.columns:
                # Handle both tuple MultiIndex and flat string columns safely
                c_end = str(col[-1]).lower().strip() if isinstance(col, tuple) else str(col).lower().strip()
                if target_substring in c_end:
                    if not any(ex in c_end for ex in exclude):
                        s = stats_df[col]
                        # If duplicate columns exist, take the first one to avoid DataFrame errors
                        if isinstance(s, pd.DataFrame): 
                            return s.iloc[:, 0]
                        return s
            return pd.Series(0.0, index=stats_df.index)

        clean_df['minutes_played'] = pd.to_numeric(get_metric('min', ['90', 'per']), errors='coerce').fillna(0.0)
        clean_df['fbref_xg'] = pd.to_numeric(get_metric('xg', ['npxg', 'xag', '90', 'per']), errors='coerce').fillna(0.0)
        clean_df['fbref_npxg'] = pd.to_numeric(get_metric('npxg', ['90', 'per']), errors='coerce').fillna(0.0)
        clean_df['fbref_xag'] = pd.to_numeric(get_metric('xag', ['90', 'per']), errors='coerce').fillna(0.0)
        
        # Fallback for xA if xAG is not found
        if clean_df['fbref_xag'].sum() == 0.0:
            clean_df['fbref_xag'] = pd.to_numeric(get_metric('xa', ['90', 'per', 'xg']), errors='coerce').fillna(0.0)

        # 4. Final aggregation for multi-club players
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        grouped_df = clean_df.groupby('name', as_index=False)[numeric_cols].sum()

        logger.info(f"Successfully scraped {len(grouped_df)} global player records from FBref natively.")
        return grouped_df

    except Exception as e:
        logger.error(f"Error fetching FBref data: {e}")
        return pd.DataFrame()