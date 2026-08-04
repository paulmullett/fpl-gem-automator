"""
ml_engine/data_ingestion.py — Multi-Source Football Data Ingestion Engine (Team-Bounded)
"""
import logging
import pandas as pd
import requests
import soccerdata as sd
import io  # NEW: Required for parsing native HTML tables

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

def fetch_fbref_data(leagues=("Big 5 European Leagues Combined", "Championship"), seasons="2526") -> pd.DataFrame:
    logger.info(f"Fetching FBref underlying stats for {leagues} (Season {seasons})...")
    all_dfs = []
    
    # Ensure leagues is iterable
    if isinstance(leagues, str):
        leagues = [leagues]

    for league in leagues:
        if "Championship" in league:
            # soccerdata hardcodes Tier-1 leagues and rejects the Championship.
            # We bypass it here by natively scraping the FBref historical URL.
            logger.info("Bypassing soccerdata to scrape Championship data natively...")
            
            # Convert "2526" to "2025-2026" for the URL path
            s_yr = f"20{seasons[:2]}-20{seasons[2:]}"
            url = f"https://fbref.com/en/comps/10/{s_yr}/stats/{s_yr}-Championship-Stats"

            if "Championship" in league:
              # soccerdata hardcodes Tier-1 leagues and rejects the Championship.
              # We bypass it here by natively scraping the FBref historical URL.
              logger.info("Bypassing soccerdata to scrape Championship data natively...")
            
              # Direct URL for the current Championship player stats
              url = "https://fbref.com/en/comps/10/stats/Championship-Stats"

            try:
                # FBref strictly requires 'sec-ch-ua' and realistic browser headers to bypass 403 blocks
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
                    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9"
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # FBref optimizes page loads by hiding secondary tables inside HTML comments.
                # We strip the comments so pandas can parse the DOM correctly.
                html_content = response.text.replace('<!--', '').replace('-->', '')
                
                # Extract the "Standard Stats" table
                tables = pd.read_html(io.StringIO(html_content), match="Standard Stats")
                if not tables:
                    raise ValueError("Could not locate Standard Stats table in HTML.")
                    
                stats_df = tables[0]
                
                # FBref uses multi-level headers (e.g., "Expected" -> "xG"). We flatten them.
                if isinstance(stats_df.columns, pd.MultiIndex):
                    stats_df.columns = ['_'.join(str(c) for c in col).strip() for col in stats_df.columns.values]
                    
            except Exception as e:
                logger.error(f"Error fetching Championship data natively: {e}")
                continue
                
        else:
            try:
                logger.info(f"Scraping FBref via soccerdata for league: {league}...")
                fbref = sd.FBref(leagues=league, seasons=seasons)
                stats_df = fbref.read_player_season_stats(stat_type="standard")
                if stats_df.empty:
                    continue
                stats_df = stats_df.reset_index()
            except Exception as e:
                logger.error(f"Error fetching FBref data for {league}: {e}")
                continue

        # Common Parsing Logic for BOTH soccerdata and native pandas tables
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

        # Tag source league context
        clean_df['source_league'] = "Championship" if "Championship" in league else "Premier_League"

        # Failsafes
        for col in ['name', 'team', 'minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag', 'source_league']:
            if col not in clean_df:
                clean_df[col] = 0.0 if col not in ['name', 'team', 'source_league'] else "Unknown"
        
        clean_df['name'] = clean_df['name'].astype(str)
        clean_df['team'] = clean_df['team'].astype(str)
        
        numeric_cols = ['minutes_played', 'fbref_xg', 'fbref_npxg', 'fbref_xag']
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
            
        # Drop summary rows (native FBref tables often include a "Squad Total" row at the bottom)
        clean_df = clean_df[~clean_df['name'].str.contains('Total|Opponent', case=False, na=False)]
            
        all_dfs.append(clean_df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # AGGREGATION: Group by name and preserve source league and last team
    grouped_df = combined_df.groupby('name', as_index=False).agg({
        'team': 'last',
        'minutes_played': 'sum',
        'fbref_xg': 'sum',
        'fbref_npxg': 'sum',
        'fbref_xag': 'sum',
        'source_league': 'first'
    })

    logger.info(f"Successfully scraped {len(grouped_df)} global player records from FBref natively (including Championship).")
    return grouped_df