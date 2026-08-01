import os
import json
from ml_engine.data_ingestion import IngestionEngine
from ml_engine.entity_resolution import ResolutionEngine
from ml_engine.feature_engineering import FeatureEngine
from ml_engine.train_models import ModelTrainer

def main():
    print("--- FPL ML PIPELINE INITIALIZATION ---")
    
    ingestor = IngestionEngine()
    raw_fpl, raw_event = ingestor.execute()
    
    resolver = ResolutionEngine()
    merged_data = resolver.map_entities(raw_fpl, raw_event)
    
    engineer = FeatureEngine()
    model_ready_data = engineer.build_features(merged_data)
    
    trainer = ModelTrainer()
    projections = trainer.run_pipeline(model_ready_data)
    
    with open("ml_projections.json", "w") as f:
        json.dump(projections, f, indent=4)
        
    print(f"--- PIPELINE COMPLETE: ml_projections.json generated ({len(projections)} entities) ---")

if __name__ == "__main__":
    main()
