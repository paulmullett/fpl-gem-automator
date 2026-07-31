import pulp

def solve_multi_period_model(players_dict, ev_matrix, current_squad_ids, current_bank, free_transfers_avail, bench_discount=0.01, horizons=4, risk_posture="NEUTRAL"):
    """
    True Multi-Period Optimization (MPO) with Transfer Trees.
    Solves across horizons (t = 0 to 3) to find the optimal transfer sequence.
    """
    model = pulp.LpProblem("FPL_Multi_Period_Optimization", pulp.LpMaximize)
    
    valid_ids = list(players_dict.keys())
    gameweeks = list(range(horizons))
    
    # Core Decision Variables across all Time Horizons
    squad = pulp.LpVariable.dicts("squad", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    starter = pulp.LpVariable.dicts("starter", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    trans_in = pulp.LpVariable.dicts("trans_in", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    trans_out = pulp.LpVariable.dicts("trans_out", ((i, t) for i in valid_ids for t in gameweeks), cat="Binary")
    
    bank = pulp.LpVariable.dicts("bank", gameweeks, lowBound=0.0, cat="Continuous")
    hits = pulp.LpVariable.dicts("hits", gameweeks, lowBound=0, upBound=5, cat="Integer")
    ft = pulp.LpVariable.dicts("ft", gameweeks, lowBound=1, upBound=5, cat="Integer")

    objective = []
    gamma = 0.95 # Temporal decay factor (prioritizes immediate points over distant projections)

    for t in gameweeks:
        gw_points = []
        for i in valid_ids:
            p = players_dict[i]
            ev = ev_matrix[i][t]
            
            # Risk Posture: Top 10k EO Shielding vs. Differential Chasing
            ownership = p.get("own", 0.0) / 100.0
            
            if risk_posture == "SHIELD":
                # Heavily penalize missing highly-owned players (protect rank)
                rank_gravity = (ev * (ownership ** 2) * 1.50)
            elif risk_posture == "CHASE":
                # Penalize highly-owned players, reward low-owned differentials (hunt rank)
                rank_gravity = -(ev * ownership * 0.50)
            else:
                # Standard structural baseline
                rank_gravity = (ev * (ownership ** 2) * 0.75)
            
            # Price Momentum Matrix
            trans_in_ev = p.get("transfers_in_event", 0)
            trans_out_ev = p.get("transfers_out_event", 0)
            momentum_boost = max(0.0, ((trans_in_ev - trans_out_ev) / 100000.0) * 0.05)
            
            gw_points.append(
                (ev * starter[i, t]) + 
                ((ev + rank_gravity) * captain[i, t]) + 
                (bench_discount * ev * (squad[i, t] - starter[i, t])) +
                (momentum_boost * squad[i, t])
            )
        
        # Apply temporal decay and subtract -4 hit penalties
        objective.append((gamma ** t) * pulp.lpSum(gw_points) - (4.0 * hits[t]))

    model += pulp.lpSum(objective)

    for t in gameweeks:
        # Legal Team Constraints
        model += pulp.lpSum([squad[i, t] for i in valid_ids]) == 15
        model += pulp.lpSum([starter[i, t] for i in valid_ids]) == 11
        model += pulp.lpSum([captain[i, t] for i in valid_ids]) == 1
        
        # Positional Squad Constraints
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 2
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 2]) == 5
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 3]) == 5
        model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 4]) == 3
        
        # Positional Starting XI Constraints (Valid Formations)
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 1]) == 1
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 2]) >= 3
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 3]) >= 3
        model += pulp.lpSum([starter[i, t] for i in valid_ids if players_dict[i]["pos_id"] == 4]) >= 1

        for i in valid_ids:
            model += starter[i, t] <= squad[i, t]
            model += captain[i, t] <= starter[i, t]

        # Club Limit Constraint
        team_ids = set(p["team_id"] for p in players_dict.values())
        for t_id in team_ids:
            model += pulp.lpSum([squad[i, t] for i in valid_ids if players_dict[i]["team_id"] == t_id]) <= 3

        # Rolling Budget Constraint
        if t == 0:
            starting_budget = current_bank + sum(players_dict[i]["selling_price"] for i in current_squad_ids if i in valid_ids) if current_squad_ids else 100.0
            model += pulp.lpSum([players_dict[i]["cost"] * squad[i, t] for i in valid_ids]) + bank[t] <= starting_budget
        else:
            model += pulp.lpSum([players_dict[i]["cost"] * squad[i, t] for i in valid_ids]) + bank[t] <= pulp.lpSum([players_dict[i]["cost"] * squad[i, t-1] for i in valid_ids]) + bank[t-1]

        # Transfer Logic & Roll Constraints
        if t == 0:
            if current_squad_ids and len(current_squad_ids) == 15 and free_transfers_avail != "Unlimited":
                for i in valid_ids:
                    if i in current_squad_ids:
                        model += squad[i, 0] == 1 - trans_out[i, 0] + trans_in[i, 0]
                    else:
                        model += squad[i, 0] == trans_in[i, 0]
                
                try: starting_ft = int(''.join(filter(str.isdigit, str(free_transfers_avail))))
                except: starting_ft = 1
                
                model += pulp.lpSum([trans_in[i, 0] for i in valid_ids]) <= starting_ft + hits[0]
                model += ft[0] == starting_ft - pulp.lpSum([trans_in[i, 0] for i in valid_ids]) + hits[0] + 1
            else:
                model += hits[0] == 0
                model += ft[0] == 1
        else:
            for i in valid_ids:
                model += squad[i, t] == squad[i, t-1] + trans_in[i, t] - trans_out[i, t]
            
            model += pulp.lpSum([trans_in[i, t] for i in valid_ids]) <= ft[t-1] + hits[t]
            model += ft[t] <= ft[t-1] - pulp.lpSum([trans_in[i, t] for i in valid_ids]) + hits[t] + 1
            model += ft[t] <= 5 # Maximum banked transfers constraint

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    optimal_squad = [players_dict[i] for i in valid_ids if squad[i, 0].varValue and squad[i, 0].varValue > 0.5]
    
    transfer_plan = []
    if current_squad_ids and len(current_squad_ids) == 15 and free_transfers_avail != "Unlimited":
        for t in gameweeks:
            ins = [players_dict[i]["name"] for i in valid_ids if trans_in[i, t].varValue and trans_in[i, t].varValue > 0.5]
            outs = [players_dict[i]["name"] for i in valid_ids if trans_out[i, t].varValue and trans_out[i, t].varValue > 0.5]
            if ins or outs:
                transfer_plan.append(f"GW+{t} -> OUT: {', '.join(outs)} | IN: {', '.join(ins)}")
    
    return optimal_squad, transfer_plan
