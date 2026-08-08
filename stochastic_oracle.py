"""
stochastic_oracle.py — Sample Average Approximation (SAA) Multiverse Generator
"""

import json
import logging
import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

def generate_stochastic_scenarios(ml_proj_path: str = "ml_projections.json", 
                                   output_path: str = "stochastic_scenarios.json", 
                                   num_simulations: int = 1000, 
                                   num_clusters: int = 50) -> bool:
    
    # Lock the seed so Monte Carlo draws are 100% reproducible across runs
    # Change BEFORE LAUNCH test sessions only update here
    np.random.seed(42)
    
    logger.info("Initializing Stochastic Oracle (SAA Scenario Generator)...")
    
    try:
        with open(ml_proj_path, "r") as f:
            projections = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load projections from {ml_proj_path}: {e}")
        return False

    player_ids = list(projections.keys())
    num_players = len(player_ids)
    
    if num_players == 0:
        logger.error("No player projections found.")
        return False

    logger.info(f"Loaded projections for {num_players} players.")

    # 1. Generate 1,000 Monte Carlo Futures across 8 Gameweeks
    # Shape: (1000 simulations, num_players, 8 gameweeks)
    simulated_futures = np.zeros((num_simulations, num_players, 8))

    for p_idx, pid in enumerate(player_ids):
        pdata = projections[pid]
        ev_matrix = pdata.get("ml_ev_matrix", [pdata.get("ml_ev_1gw", 2.0)] * 8)
        xmins_matrix = pdata.get("ml_xmins_matrix", [pdata.get("ml_xmins", 90.0)] * 8)
        
        for t in range(min(8, len(ev_matrix))):
            base_ev = float(ev_matrix[t])
            xmins = float(xmins_matrix[t])
            
            if xmins <= 5.0 or base_ev <= 0.1:
                simulated_futures[:, p_idx, t] = 0.0
                continue

            # Stochastic variance bounds based on expected minutes
            sigma = base_ev * (0.35 + (0.15 * (1.0 - (xmins / 90.0))))
            draws = np.random.normal(loc=base_ev, scale=sigma, size=num_simulations)
            simulated_futures[:, p_idx, t] = np.maximum(0.0, draws)

    # 2. Feature Slicing for Fast Clustering
    # Isolate top 100 players by 8-GW EV to reduce dimensions from (1000, 4584) to (1000, 800)
    top_100_indices = sorted(range(num_players), key=lambda i: sum(projections[player_ids[i]].get("ml_ev_matrix", [0]*8)), reverse=True)[:100]
    
    # Reshape top player futures to (1000, 800)
    X_clustering = simulated_futures[:, top_100_indices, :].reshape(num_simulations, -1)

    # 3. Fast Clustering via MiniBatchKMeans
    logger.info(f"Clustering {num_simulations} futures into {num_clusters} SAA representative scenarios...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, 
        batch_size=256, 
        n_init=3, 
        max_iter=50, 
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(X_clustering)

    # 4. Extract Clustered Scenarios and Normalize Weights
    scenarios = {}
    cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
    
    for k in range(num_clusters):
        weight = float(cluster_counts[k]) / float(num_simulations)
        if weight == 0:
            continue
            
        # Compute mean 8-GW EV matrix across all 573 players for this cluster
        cluster_mask = (cluster_labels == k)
        cluster_mean_matrix = np.mean(simulated_futures[cluster_mask, :, :], axis=0)
        
        player_ev_matrix = {}
        for p_idx, pid in enumerate(player_ids):
            player_ev_matrix[pid] = np.round(cluster_mean_matrix[p_idx, :], 2).tolist()
            
        scenarios[str(k)] = {
            "weight": round(weight, 4),
            "player_ev_matrix": player_ev_matrix
        }

    output_data = {
        "num_scenarios": len(scenarios),
        "scenarios": scenarios
    }

    try:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Successfully locked {len(scenarios)} SAA scenarios to {output_path}.")
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False

if __name__ == "__main__":
    generate_stochastic_scenarios()