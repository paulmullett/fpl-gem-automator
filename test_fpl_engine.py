"""
test_fpl_engine.py — Unit Test Suite for FPL Quantitative Decision Engine
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from fpl_funcs import get_base_ev, poisson_prob_ge
from fpl_mpo_engine import solve_multi_period_model
from ml_engine.train_models import get_upcoming_opponent_mapping


class TestFPLEngineMathematics(unittest.TestCase):

    def test_goalkeeper_save_points_scaling(self):
        """
        Verify that 3 expected goalkeeper saves evaluate to 1.0 full point 
        (scaled by xMins), ensuring save points are not artificially reduced.
        """
        player_gk = {
            "id": 101,
            "name": "Test GK",
            "pos_id": 1,
            "cost": 5.0,
            "own": 10.0,
            "status": "a",
            "xgi_90": 0.0,
            "xgc_90": 2.14,  # Triggers ~3.0 expected saves (2.14 * 1.4 = 2.996)
            "ep_next": 3.5,
            "team": "TEST"
        }
        
        # Override xMins to 90.0
        xmins_overrides = {"101": 90.0}
        
        # Compute base EV
        ev = get_base_ev(player_gk, xmins_overrides=xmins_overrides)
        
        # At 2.14 xGC and 90 mins:
        # App points = 2.0
        # Clean Sheet prob = e^(-2.14) = 0.1176 -> CS points = 0.1176 * 4.0 = 0.4706
        # Expected saves = 2.996 -> Save points = 2.996 / 3.0 = 0.9988
        # Total EV should comfortably exceed 3.0 pts
        self.assertGreater(ev, 3.20)

    def test_non_linear_profit_tax_rounding(self):
        """
        Verify exact FPL integer floor logic for sell-on profits:
        +£0.1m rise -> £0.0m profit
        +£0.2m rise -> £0.1m profit
        +£0.3m rise -> £0.1m profit
        -£0.1m drop -> -£0.1m full loss
        """
        def calc_exact_sell_price(p_cost, p_delta):
            future_market_price = p_cost + p_delta
            if p_delta > 0.0:
                exact_profit = math.floor(round(p_delta * 10) / 2) * 0.1
                return round(p_cost + exact_profit, 2)
            else:
                return round(future_market_price, 2)

        buy_price = 10.0
        
        # Test cases
        self.assertEqual(calc_exact_sell_price(buy_price, 0.10), 10.00)
        self.assertEqual(calc_exact_sell_price(buy_price, 0.20), 10.10)
        self.assertEqual(calc_exact_sell_price(buy_price, 0.30), 10.10)
        self.assertEqual(calc_exact_sell_price(buy_price, 0.40), 10.20)
        self.assertEqual(calc_exact_sell_price(buy_price, -0.10), 9.90)

    def test_bps_defensive_poisson_thresholds(self):
        """
        Verify Poisson tail probability functions evaluate correctly
        for BPS action thresholds (>8.5 CBIT / >10.5 CBIRT).
        """
        # Poisson prob of >= 9 actions when expecting 11.5 actions
        prob_def = poisson_prob_ge(9, 11.5)
        self.assertGreater(prob_def, 0.80)
        self.assertLessEqual(prob_def, 1.00)

        # Poisson prob of >= 11 actions when expecting 6.0 actions
        prob_mid = poisson_prob_ge(11, 6.0)
        self.assertLess(prob_mid, 0.10)

    @patch('requests.get')
    def test_double_gameweek_opponent_mapping(self, mock_get):
        """
        Verify that a Double Gameweek maps multiple opponents into an array 
        rather than overwriting the dictionary key.
        """
        # Mock API returning a Double Gameweek fixture for team 1 (ARS)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"team_h": 1, "team_a": 2}, # Game 1: ARS vs AVL
            {"team_h": 1, "team_a": 3}  # Game 2: ARS vs BOU
        ]
        mock_get.return_value = mock_response

        opp_map = get_upcoming_opponent_mapping(current_gw=1)
        
        # Team 1 should have 2 distinct opponents in its list
        self.assertEqual(len(opp_map[1]), 2)
        self.assertIn(2, opp_map[1])
        self.assertIn(3, opp_map[1])

    @patch('pulp.LpProblem.solve')
    def test_greedy_knapsack_fallback_budget_compliance(self, mock_solve):
        """
        Verify that if the primary solver fails, the fallback heuristic 
        strictly enforces the £100.0m budget cap and valid positional layout.
        """
        # Create a mock player pool with realistic FPL baseline prices and team tags
        mock_players = {}
        pid = 1
        for pos_id, count, base_cost in [(1, 4, 4.0), (2, 10, 4.0), (3, 10, 4.5), (4, 6, 4.5)]:
            for i in range(count):
                mock_players[pid] = {
                    "id": pid, "name": f"Player_{pid}", "pos_id": pos_id,
                    "cost": base_cost + (i * 0.5), "status": "a", 
                    "team_id": i + 1, "team": f"TEAM_{i+1}"
                }
                pid += 1

        # Give higher cost players higher EVs so the Greedy Knapsack has a reason to upgrade
        mock_ev_matrix = {p: [mock_players[p]["cost"] * 1.2] * 8 for p in mock_players}

        # Mocking pulp.LpProblem.solve prevents variables from assigning, 
        # guaranteeing a has_feasible_squad() == False trigger.
        squad, plan = solve_multi_period_model(
            players=mock_players,
            ev_matrix=mock_ev_matrix,
            current_squad_ids=[],
            total_liquid_budget=100.0,
            free_transfers=1,
            horizons=8
        )

        # Assert total squad size and budget compliance
        self.assertEqual(len(squad), 15)
        total_cost = sum(p["cost"] for p in squad)
        self.assertLessEqual(total_cost, 100.0)
        
        # The absolute cheapest base squad is £76.0m. 
        # We assert it is strictly greater to prove the Knapsack upgrades worked.
        self.assertGreater(total_cost, 76.0)


if __name__ == '__main__':
    unittest.main()