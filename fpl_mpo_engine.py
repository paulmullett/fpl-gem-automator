import pulp
import math

def solve_multi_period_model(players_dict, horizons=3):
    """
    Solves a multi-period optimization problem across a rolling horizon of 'horizons' gameweeks.
    """
    model = pulp.LpProblem("FPL_Multi_Period_Optimization", pulp.LpMaximize)
    
    valid_ids = list(players_dict.keys())
    gameweeks = list(range(horizons))
    
    squad = pulp.LpVariable.dicts("squad", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    starter = pulp.LpVariable.dicts("starter", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    trans_in = pulp.LpVariable.dicts("trans_in", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    
    bank = pulp.LpVariable.dicts("bank", gameweeks, lowBound=0.0, upBound=100.0, cat="Continuous")
    hits = pulp.LpVariable.dicts("hits", gameweeks, lowBound=0, upBound=5, cat="Integer")
    free_transfers = pulp.LpVariable.dicts("ft", gameweeks, lowBound=1, upBound=5, cat="Integer")

    objective = []
    
    for t in gameweeks:
        gw_points = []
        for i in valid_ids:
            p = players_dict[i]
            try:
                ev = float(p.get("ep_next", 3.0))
            except:
                ev = 3.0
                
            own_pct = float(p.get("own", 0.0)) / 100.0
            rank_gravity = (ev * (own_pct ** 2) * 0.75)
            
            gw_points.append(
                (ev * starter[i, t]) + 
                ((ev + rank_gravity) * captain[i, t]) + 
                (0.01 * ev * (squad[i, t] - starter[i, t]))
            )
        
        objective.append(pulp.lpSum(gw_points) - 4 * hits[t])

    model += pulp.lpSum(objective)

    for t in gameweeks:
        model += pulp.lpSum([squad[i, t] for i in valid_ids]) == 15
        model += pulp.lpSum([starter[i, t] for i in valid_ids]) == 11
        model += pulp.lpSum([captain[i, t] for i in valid_ids]) == 1
        
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
        
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 1
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 2]) >= 3
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 3]) >= 3
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 4]) >= 1

        for i in valid_ids:
            model += starter[i, t] <= squad[i, t]
            model += captain[i, t] <= starter[i, t]

        for t_id in set(p["team_id"] for p in players_dict.values()):
            model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

        model += pulp.lpSum([players_dict[i]["cost"] * squad[i, t] for i in valid_ids]) + bank[t] <= 100.0

        if t > 0:
            for i in valid_ids:
                model += squad[i, t] == squad[i, t-1] + trans_in[i, t] - trans_out[i, t]
            
            total_transfers_in = pulp.lpSum([trans_in[i, t] for i in valid_ids])
            model += total_transfers_in <= free_transfers[t-1] + hits[t]

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    starters = [players_dict[i] for i in valid_ids if starter[i, 0].varValue and starter[i, 0].varValue > 0.5]
    captain = next((players_dict[i] for i in valid_ids if captain[i, 0].varValue and captain[i, 0].varValue > 0.5), None)
    total_xp = sum(float(p.get("ep_next", 3.0)) for p in starters)
    if captain:
        total_xp += float(captain.get("ep_next", 3.0))

    return starters, captain, total_xp
