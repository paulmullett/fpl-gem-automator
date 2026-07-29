# FPL Optimization & Master Audit Pipeline

An advanced, institutional-grade Fantasy Premier League (FPL) modeling and automation pipeline. This project utilizes Mixed-Integer Linear Programming (MILP), multi-period optimization, Monte Carlo stochastic simulations, and live bookmaker odds integration to maximize expected points (xP) and long-term squad value.

## Core Architecture & Features

* **Baseline EV Engine (`fpl_compare.py`):** Evaluates players using Poisson clean-sheet probability models, sigmoid expected minutes curves, Bayesian-shrunken Expected Goal Involvements (xGI), and integrated DefCon (Clearances, Blocks, Interceptions/CBIT) BPS bonuses alongside goalkeeper save-point estimations.
* **Tier 1 Dynamic Translation Engine:** Normalizes metrics for non-Premier League arrivals by combining league baseline strengths, age-performance adaptation curves, and team dominance deltas.
* **Multi-Period Optimization (MPO) (`fpl_mpo_engine.py`):** Evaluates a rolling multi-week horizon, incorporating transfer banking economics (up to 5 free transfers) and fixture difficulty swings.
* **Stochastic Risk Bounds (`fpl_monte_carlo.py`):** Runs Monte Carlo simulations to calculate explicit 10th percentile floors and 90th percentile ceilings for active starting lineups.
* **Market Odds Blending (`fpl_odds_engine.py`):** Fuses baseline expectations with live bookmaker odds and short-term form metrics through an ensemble model layer.
* **Automated Discord Audits:** Pushes comprehensive pre-gameweek audit summaries, model comparisons, and structural shift indicators directly to Discord via webhooks (`fpl_compare_discord.py`).

## Repository Structure

```text
├── .github/workflows/         # Automated GitHub Action pipelines
│   ├── ensemble.yml
│   ├── fpl_automation.yml
│   ├── fpl_compare.yml
│   ├── fpl_engine.yml
│   ├── fpl_player_compare.yml
│   └── test_auth.yml
├── fpl_bot.py                 # Bot integration script
├── fpl_compare.py             # Master orchestration script
├── fpl_compare_discord.py     # Discord webhook formatting and transmission
├── fpl_ensemble.py            # Ensemble model logic
├── fpl_monte_carlo.py         # Stochastic risk simulation engine
├── fpl_mpo_engine.py          # Multi-period horizon optimization solver
├── fpl_odds_engine.py         # Market odds data ingestion and adjustment
├── fpl_state.json             # State tracking storage
└── test_auth.py               # Authentication testing script
