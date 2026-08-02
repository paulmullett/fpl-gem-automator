"""
run_ml_pipeline.py — Master Execution File for the ML Suite
"""
import json
import logging
import sys

from ml_engine.data_ingestion import fetch_fpl_data
from ml_engine.train_models import generate_ml_projections

# Setup clean terminal logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STARTING FPL MACHINE LEARNING PIPELINE ===")
    
    # 1. Scrape the official FPL data (includes Opta xG/xA stats)
    fpl_data = fetch_fpl_data()
    
    if fpl_data.empty:
        logger.error("Critical FPL API data missing. Aborting pipeline.")
        sys.exit(1)

    # 2. Train Models & Predict using native data
    projections = generate_ml_projections(fpl_data)
    
    # 3. Save JSON Payload for the Bot to read
    output_file = "ml_projections.json"
    with open(output_file, 'w') as f:
        json.dump(projections, f, indent=4)
        
    logger.info(f"=== PIPELINE COMPLETE: Outputs locked to {output_file} ===")

if __name__ == "__main__":
    main()