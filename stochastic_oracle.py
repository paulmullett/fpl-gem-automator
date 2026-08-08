"""
stochastic_oracle.py — Sample Average Approximation (SAA) Scenario Generator

Generates 1,000 Monte Carlo futures across an 8-GW horizon using bimodal 
appearance probabilities and Poisson scoring distributions, then collapses 
them into 50 weighted scenarios via K-Means clustering.
"""

import os
import json
import logging
import numpy as np
from sklearn.cluster import KMeans

from fpl_funcs import get_bimodal_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_SIMULATIONS = 1000
NUM_CLUSTERS = 50
HORIZON_GWS = 8

def generate_stochastic_scenarios(ml_proj_path: str = "ml_projections.json", 
                                 output_path: str = "stochastic_scenarios.json") -> bool:
    """
    Simulates N Monte Carlo gameweek futures and clusters them into 
    weighted representative scenarios for the stochastic MILP solver.
    """
    logger.info("Initializing Stochastic Oracle (SAA Scenario Generator)...")
    
    if not os.path.exists(ml_proj_path):
        logger.error(f"Cannot find {ml_proj_path}. Run ML pipeline first.")
        return False

    with open(ml_proj_path, "r") as f:
        ml_proj_data = json.load(f)

    player_ids = list(ml_proj_data.keys())
    num_players = len(player_ids)
    logger.info(f"Loaded projections for {num_players} players.")

    # Tensor shape: (NUM_SIMULATIONS, num_players, HORIZON_GWS)
    simulated_matrix = np.zeros((NUM_SIMULATIONS, num_players, HORIZON_GWS), dtype=np.float32)

    for p_idx, pid in enumerate(player_ids):
        p_data = ml_proj_data[pid]
        ev_matrix = p_data.get("ml_ev_matrix", [p_data.get("ml_ev_1gw", 2.0)] * HORIZON_GWS)
        xmins_matrix = p_data.get("ml_xmins_matrix", [p_data.get("ml_xmins", 75.0)] * HORIZON_GWS)

        for gw_idx in range(HORIZON_GWS):
            ev_base = float(ev_matrix[gw_idx])
            xmins_target = float(xmins_matrix[gw_idx])

            p_start, p_sub, p_bench = get_bimodal_probabilities(xmins_target)
            probs = np.array([p_bench, p_sub, p_start], dtype=np.float64)
            probs /= probs.sum()  # Normalize to guarantee exact 1.0 sum

            # 1. Sample appearance states for 1,000 simulations
            # States: 0 = Bench (0 mins), 1 = Sub (~20 mins), 2 = Start (~85 mins)
            states = np.random.choice([0, 1, 2], size=NUM_SIMULATIONS, p=probs)

            # 2. Derive action EV (stripping standard 2pt appearance baseline)
            pure_action_ev = max(0.0, ev_base - 2.0)
            action_ev_start = pure_action_ev * (85.0 / 90.0)
            action_ev_sub = pure_action_ev * (20.0 / 90.0)

            # Expected yield given pitch time state
            sim_lambda = np.where(states == 2, 2.0 + action_ev_start,
                         np.where(states == 1, 1.0 + action_ev_sub, 0.0))

            # 3. Sample realized points from Poisson distribution
            points_sampled = np.random.poisson(lam=sim_lambda).astype(np.float32)
            simulated_matrix[:, p_idx, gw_idx] = points_sampled

    # Flatten simulations across players & GWs for clustering: (NUM_SIMULATIONS, num_players * HORIZON_GWS)
    flattened_sims = simulated_matrix.reshape(NUM_SIMULATIONS, -1)

    logger.info(f"Clustering {NUM_SIMULATIONS} futures into {NUM_CLUSTERS} SAA representative scenarios...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(flattened_sims)
    cluster_centers = kmeans.cluster_centers_

    # Calculate cluster probability weights
    counts = np.bincount(cluster_labels, minlength=NUM_CLUSTERS)
    weights = (counts / NUM_SIMULATIONS).tolist()

    # Reconstruct clustered matrix: (NUM_CLUSTERS, num_players, HORIZON_GWS)
    clustered_matrix = cluster_centers.reshape(NUM_CLUSTERS, num_players, HORIZON_GWS)

    scenarios_output = {
        "metadata": {
            "num_simulations": NUM_SIMULATIONS,
            "num_clusters": NUM_CLUSTERS,
            "horizon_gws": HORIZON_GWS,
            "player_ids": player_ids
        },
        "scenario_weights": weights,
        "scenarios": {}
    }

    for c_idx in range(NUM_CLUSTERS):
        scenarios_output["scenarios"][str(c_idx)] = {
            "weight": round(weights[c_idx], 4),
            "player_ev_matrix": {
                player_ids[p_idx]: [round(float(clustered_matrix[c_idx, p_idx, gw_idx]), 2) 
                                    for gw_idx in range(HORIZON_GWS)]
                for p_idx in range(num_players)
            }
        }

    with open(output_path, "w") as f:
        json.dump(scenarios_output, f, indent=2)

    logger.info(f"Successfully generated {NUM_CLUSTERS} SAA scenarios saved to '{output_path}'.")
    return True

if __name__ == "__main__":
    generate_stochastic_scenarios()