"""
ml_engine/data_ingestion.py — Multi-Source Football Data Ingestion Engine (Team-Bounded)
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

def fetch_fbref_data(leagues="Big 5 European Leagues Combined", seasons="2526") -> pd.DataFrame:
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
            elif ('squad' in str_col or 'team' in str_col) and 'team' not in clean_df:
                clean_df['team'] = stats_df[orig_col]
            elif 'min' in str_col and '90' not in str_col and 'minutes_played' not in clean_df:
                clean_df['minutes_played'] = stats_df[orig_col]
            elif 'npxg' in str_col and '90' not in str_col and 'fbref_npxg' not in clean_df:
                clean_df['fbref_npxg'] = stats_df[orig_col]
            elif 'xag' in str_col and '90' not in str_col and 'fbref_xag' not in clean_df:
                clean_df['fbref_xag'] = stats_df[orig_col]
            elif 'xg' in str_col and 'npxg' not in str_col and '90' not in str_col and 'fbref_xg' not in clean_df:
                clean_df['fbref_xg'] = stats_df[orig_col]

        # Failsafes
        for col in ['name', 'team', 'minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']:
            if col not in clean_df:
                clean_df[col] = 0.0 if col not in ['name', 'team'] else "Unknown"
        
        clean_df['name'] = clean_df['name'].astype(str)
        clean_df['team'] = clean_df['team'].astype(str)
        
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
        
        # AGGREGATION: Group by name and preserve the last recorded team for mid-season transfers
        grouped_df = clean_df.groupby('name', as_index=False).agg({
            'team': 'last',
            'minutes_played': 'sum',
            'fbref_xg': 'sum',
            'fbref_npxg': 'sum',
            'fbref_xag': 'sum'
        })

        logger.info(f"Successfully scraped {len(grouped_df)} global player records from FBref natively.")
        return grouped_df

    except Exception as e:
        logger.error(f"Error fetching FBref data: {e}")
        return pd.DataFrame()

def get_team_matchup_ratings(fbref_df: pd.DataFrame) -> dict:
    """
    Aggregates FBref team data to generate dynamic Attack and Defense multipliers.
    A defense rating > 1.0 means the opponent concedes more xG than average (easier fixture).
    """
    logger.info("Calculating dynamic team matchup modifiers...")
    if fbref_df.empty or 'team' not in fbref_df.columns:
        return {}
        
    # Aggregate team totals
    team_stats = fbref_df.groupby('team', as_index=False).agg({
        'fbref_xg': 'sum',
        'fbref_xag': 'sum',
        'minutes_played': 'sum'
    })
    
    # Calculate league averages per 90
    total_mins = team_stats['minutes_played'].sum()
    if total_mins == 0:
        return {}
        
    avg_xg_90 = (team_stats['fbref_xg'].sum() / total_mins) * 90.0
    
    ratings = {}
    for _, row in team_stats.iterrows():
        team_name = str(row['team'])
        mins = float(row['minutes_played'])
        if mins > 270.0:
            team_xg_90 = (float(row['fbref_xg']) / mins) * 90.0
            # Multiplier centered around 1.0
            attack_rating = team_xg_90 / avg_xg_90 if avg_xg_90 > 0 else 1.0
        else:
            attack_rating = 1.0
            
        ratings[team_name] = {
            "attack_rating": round(attack_rating, 2)
        }
        
    return ratings