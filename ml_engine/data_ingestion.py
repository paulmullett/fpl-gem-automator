import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataIngestion")

class IngestionEngine:
    def __init__(self):
        self.fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.headers = {"User-Agent": "FPL-ML-Pipeline/1.0"}

    def fetch_fpl_base(self):
        logger.info("Fetching live FPL API static data...")
        try:
            resp = requests.get(self.fpl_url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            return pd.DataFrame(data['elements'])
        except Exception as e:
            logger.error(f"FPL API fetch failed: {e}")
            return pd.DataFrame()

    def fetch_event_data(self):
        logger.info("Initializing soccerdata ingestion (FBref/Understat)...")
        # In full production, this hooks into soccerdata to pull match logs.
        # Returning an empty DataFrame for initial structural mapping.
        return pd.DataFrame()

    def execute(self):
        fpl_data = self.fetch_fpl_base()
        event_data = self.fetch_event_data()
        return fpl_data, event_data
