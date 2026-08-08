"""
run_ml_pipeline.py — Master Execution File for the ML & Stochastic Suite
"""
import json
import logging
import sys

from ml_engine.data_ingestion import fetch_fpl_data, fetch_fbref_data
from ml_engine.train_models import generate_ml_projections
from stochastic_oracle import generate_stochastic_scenarios

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STARTING FPL MACHINE LEARNING & STOCHASTIC PIPELINE ===")
    
    # 1. Scrape data
    fpl_data = fetch_fpl_data()
    fbref_data = fetch_fbref_data()
    
    if fpl_data.empty:
        logger.error("Critical FPL API data missing. Aborting pipeline.")
        sys.exit(1)
        
    if fbref_data.empty:
        logger.warning("FBref data empty. Pushing to ML Failsafe...")

    # 2. Train Models & Predict
    projections = generate_ml_projections(fpl_data, fbref_data)
    
    # 3. Save Projections
    output_file = "ml_projections.json"
    with open(output_file, 'w') as f:
        json.dump(projections, f, indent=4)
    logger.info(f"ML Projections saved to {output_file}.")

    # 4. Generate SAA Stochastic Multiverse Scenarios
    logger.info("=== GENERATING SAA STOCHASTIC MULTIVERSE SCENARIOS ===")
    stochastic_success = generate_stochastic_scenarios(
        ml_proj_path=output_file, 
        output_path="stochastic_scenarios.json"
    )

    if not stochastic_success:
        logger.warning("Stochastic Scenario generation encountered errors.")

    logger.info("=== PIPELINE COMPLETE: Outputs & Scenarios Locked ===")

if __name__ == "__main__":
    main()