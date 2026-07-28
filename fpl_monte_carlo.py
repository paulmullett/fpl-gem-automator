import numpy as np

def run_monte_carlo_simulations(players_dict, num_trials=1000):
    """
    Runs Monte Carlo simulations with robust pre-season fallback metrics 
    to evaluate floor risk, median expectation, and ceiling upside.
    """
    simulated_results = {}
    
    for pid, p in players_dict.items():
        try:
            xmins = float(p.get("est_xmins", 85.0))
            xgi_90 = float(p.get("xgi_90", 0.0))
            # Fallback estimation if pre-season xgi_90 is missing or zero
            if xgi_90 <= 0.01:
                ep = float(p.get("ep_next", 3.0))
                xgi_90 = max(0.15, ep / 4.0)
            pos_id = int(p.get("pos_id", 3))
        except:
            xmins, xgi_90, pos_id = 85.0, 0.25, 3

        mean_fraction = min(1.0, max(0.0, xmins / 90.0))
        if mean_fraction <= 0:
            simulated_results[pid] = {"floor": 0.0, "expected": 0.0, "ceiling": 0.0}
            continue
        
        # Beta distribution for minutes played volatility
        alpha = max(1.0, mean_fraction * 10.0)
        beta_param = max(1.0, (1.0 - mean_fraction) * 10.0)
        sim_mins = np.random.beta(alpha, beta_param, num_trials) * 90.0
        
        # Poisson distribution for goal/assist involvements scaled by simulated minutes
        adjusted_xgi = xgi_90 * (sim_mins / 90.0)
        sim_g_a = np.random.poisson(np.maximum(0.05, adjusted_xgi), num_trials)
        
        # Appearance points based on simulated minutes
        app_points = np.where(sim_mins >= 60, 2.0, np.where(sim_mins > 0, 1.0, 0.0))
        
        # Base positional points (clean sheet / baseline floor) and goal point mapping
        base_pts = 2.0 if pos_id in [1, 2] else 1.0
        goal_pts = 6.0 if pos_id in [1, 2] else (5.0 if pos_id == 3 else 4.0)
        
        sim_total_points = app_points + base_pts + (sim_g_a * goal_pts)
        
        simulated_results[pid] = {
            "floor": float(np.percentile(sim_total_points, 10)),
            "expected": float(np.mean(sim_total_points)),
            "ceiling": float(np.percentile(sim_total_points, 90))
        }
        
    return simulated_results
