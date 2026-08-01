"""
fpl_monte_carlo.py — Stochastic Risk Simulation Engine

Executes 1,000 Monte Carlo trials per starter using Beta distributions for expected
minutes variance and Poisson distributions for attacking returns. Derives explicit 
10th percentile floors and 90th percentile ceilings for the starting XI.
"""

import numpy as np

def run_monte_carlo_simulations(players_dict: dict, num_trials: int = 1000) -> dict:
    """Runs stochastic simulation trials to return floor, mean, and ceiling point distributions."""
    simulated_results = {}
    
    for pid, p in players_dict.items():
        try:
            xmins = float(p.get("est_xmins", 85.0))
            xgi_90 = float(p.get("xgi_90", 0.0))
            if xgi_90 <= 0.01:
                ep = float(p.get("ep_next", 3.0))
                xgi_90 = max(0.15, ep / 4.0)
            pos_id = int(p.get("pos_id", 3))
        except Exception:
            xmins, xgi_90, pos_id = 85.0, 0.25, 3

        mean_fraction = min(1.0, max(0.0, xmins / 90.0))
        if mean_fraction <= 0:
            simulated_results[pid] = {"floor": 0.0, "expected": 0.0, "ceiling": 0.0}
            continue
        
        # Sample minutes played from Beta distribution
        alpha = max(1.0, mean_fraction * 10.0)
        beta_param = max(1.0, (1.0 - mean_fraction) * 10.0)
        sim_mins = np.random.beta(alpha, beta_param, num_trials) * 90.0
        
        # Sample goal/assist involvements from Poisson distribution
        adjusted_xgi = xgi_90 * (sim_mins / 90.0)
        sim_g_a = np.random.poisson(np.maximum(0.05, adjusted_xgi), num_trials)
        
        # Calculate appearance points and positional goal multipliers
        app_points = np.where(sim_mins >= 60, 2.0, np.where(sim_mins > 0, 1.0, 0.0))
        base_pts = 2.0 if pos_id in [1, 2] else 1.0
        goal_pts = 6.0 if pos_id in [1, 2] else (5.0 if pos_id == 3 else 4.0)
        
        sim_total_points = app_points + base_pts + (sim_g_a * goal_pts)
        
        simulated_results[pid] = {
            "floor": float(np.percentile(sim_total_points, 10)),
            "expected": float(np.mean(sim_total_points)),
            "ceiling": float(np.percentile(sim_total_points, 90))
        }
        
    return simulated_results
