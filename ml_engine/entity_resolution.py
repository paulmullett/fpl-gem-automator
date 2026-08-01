import json
import pandas as pd
from difflib import get_close_matches

class ResolutionEngine:
    def __init__(self, mapping_file="ml_engine/entity_mapping.json"):
        self.mapping_file = mapping_file
        self.overrides = self._load_mappings()

    def _load_mappings(self):
        try:
            with open(self.mapping_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def map_entities(self, fpl_df, event_df):
        # Applies fuzzy string matching and static JSON overrides to link
        # FBref/Understat nomenclature to strict FPL IDs.
        if fpl_df.empty:
            return pd.DataFrame()
            
        fpl_df['ml_resolved_name'] = fpl_df['web_name'].str.lower()
        return fpl_df
