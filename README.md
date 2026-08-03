# FPL Quantitative Decision Engine & Tactical Audit Pipeline (2026/27 Season)

An institutional-grade Fantasy Premier League (FPL) quantitative modeling suite. The pipeline fuses Mixed-Integer Linear Programming (MILP), an 8-Gameweek deep-tree horizon optimization engine, Monte Carlo stochastic risk modeling, live bookmaker odds intensity solvers, and Google Gemini AI orchestration to deliver automated, data-driven squad optimizations directly to Discord.

---

## System Architecture

┌────────────────────────┐     ┌────────────────────────┐
 │ Local GitHub Runner    │     │ Live Bookmaker API     │
 │ (Scrapes FBref Data)   │     │ (fpl_odds_engine.py)   │
 └───────────┬────────────┘     └───────────┬────────────┘
             │                              │
             ▼                              ▼
 ┌───────────────────────────────────────────────────────┐
 │                ML & Math Data Pipeline                │
 │     (run_ml_pipeline.py + fpl_funcs.py EV Engine)     │
 └──────────────────────────┬────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────┐     ┌──────────────────┐
 │  Python MILP Solver    │ ──► │  Google Gemini   │
 │ (fpl_bot / mpo_engine) │     │ (Orchestration)  │
 └───────────┬────────────┘     └───────────┬──────┘
             │                              │
             ▼                              ▼
 ┌────────────────────────┐     ┌──────────────────┐
 │ fpl_state.json Ledger  │     │  Discord Webhook │
 └────────────────────────┘     └──────────────────┘
```

The system operates across a unified, synchronized execution track:
1. **FPL Data & FBref Ingestion (`run_ml_pipeline.py`):** Runs on a self-hosted runner to bypass scraping blocks, aligning native FPL API data with underlying FBref metrics into a master JSON payload.
2. **Mathematical EV Engine (`fpl_funcs.py`):** Computes expected values integrating current form, bookmaker odds, historical minutes, and structural Bonus Points System (BPS) modeling. 
3. **MILP Co-Optimization Engine (`fpl_mpo_engine.py`):** Solves the multi-period Linear Programming puzzle to lock the maximum-yield 15-man squad, captaincy, and transfer path.

---

## Mathematical & Quantitative Foundation

### 1. The Bayesian Pricing Prior (Zero-Data Resolution)
During Gameweek 1 or when historical Expected Goal Involvement (xGI) data evaluates to zero, the engine relies on the official FPL market price as the ultimate Bayesian prior, scaling expected attacking threat based on premium cost thresholds. 

For forwards, the prior is calculated as:
$$\text{Prior xGI} = 0.35 + (\max(0.0, \text{Cost} - 5.0) \times 0.12)$$

### 2. Statistical Bonus Points System (BPS) Modeling
Rather than hardcoding arbitrary price logic to favor attackers, the engine uses structural BPS physics derived directly from underlying expected goals ($xG$) and expected assisted goals ($xAG$):
$$\text{Expected BPS} = ((xGI \times 0.7 \times \text{BPS}_{\text{goal\_weight}}) + (xGI \times 0.3 \times \text{BPS}_{\text{assist\_weight}})) \times \text{Mins Factor}$$
This naturally boosts premium forwards and goalscoring midfielders to correctly reflect their ceiling and capture their BPS monopoly.

### 3. Asymptotic Soft-Caps & Multi-Variate Confidence
To dampen small-sample anomalies while protecting proven elite assets, a confidence shrinkage algorithm mathematically blends raw metrics against positional baselines based on FPL ownership, price premiums, and foreign transfer translation matrices.

### 4. Exponential Poisson Clean Sheet Engine
Defender and Goalkeeper Expected Value is anchored by clean sheet probabilities derived through exponential decay equations, combined with live bookmaker non-linear odds intensity.

---

## Multi-Period Optimization (MPO) & Squad Architecture

The core solver relies on PuLP and the CBC/HiGHS command-line engines to solve an 8-Gameweek deep-tree horizon while respecting strict FPL constraints.

*   **Simultaneous Captaincy Co-Optimization:** The objective function natively co-optimizes the Starting XI EV and the $2.0\times$ Captain multiplier simultaneously. Elite assets yield massive marginal returns that mathematically lock them into the starting framework.
*   **Native Bench Weighting (0.05):** Bench Expected Value is weighted at $5\%$. This inherently punishes capital traps (like hoarding premium bench defenders or rotating goalkeepers) and forces funds into the Starting XI.
*   **Positional Guardrails:** Enforces the strict maximum 3-players-per-club rule and kills structurally sub-optimal 5-at-the-back formations by capping defensive starters to 4.
*   **Automated Fixture Swings:** Evaluates moving 4-GW clusters against the back-half 4-GW horizons to flag distinct positive and negative swing teams.
*   **Pre-Calculated FPL Chip Triggers:** Reads the 2026/27 dual-set FPL chip framework (two sets of four chips). Automatically flags optimal chip windows based on bench strength, fixture density, and premium player ceilings.

---

## Repository Directory Map

*   **.github/workflows/**
    *   `fpl_orchestrator.yml`: Master scheduled orchestrator workflow (Runs on self-hosted runner).
    *   `fpl_engine.yml`: Primary manual UI execution workflow.
    *   `fpl_player_compare.yml`: Head-to-Head player comparison workflow.
*   **ml_engine/**
    *   `data_ingestion.py`: FPL API & FBref scraper module.
    *   `train_models.py`: Unified dataset alignment that delegates xP modeling directly to `fpl_funcs.py`.
*   **Core Logic Files**
    *   `fpl_bot.py`: Primary orchestrator, Gemini AI client, & Discord publisher.
    *   `fpl_compare_players.py`: Player comparison engine.
    *   `fpl_funcs.py`: Core mathematical EV models, Bayesian pricing priors, & BPS scaling.
    *   `fpl_monte_carlo.py`: Stochastic 1,000-trial risk simulator.
    *   `fpl_mpo_engine.py`: Mixed-Integer Linear Programming solver.
    *   `fpl_odds_engine.py`: Live bookmaker odds & fsolve Poisson engine.
    *   `run_ml_pipeline.py`: Standalone ML pipeline runner.
*   **State & Payload Files**
    *   `fpl_state.json`: Persistent PID state & multi-period ledger.
    *   `ml_projections.json`: Combined payload from FBref + `fpl_funcs.py`.
    *   `requirements.txt`: Python dependencies.

---

## Environment Secrets & Configuration Setup

Configure the following secrets in GitHub Repository Settings (**Settings -> Secrets and variables -> Actions**):

| Secret Name | Description |
| :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DISCORD_WEBHOOK_URL` | Discord Channel Webhook URL |
| `FPL_TEAM_ID` | Numeric FPL Entry Team ID |
| `ODDS_API_KEY` | *(Optional)* The Odds API key for betting odds ingestion |

---

## Execution Guide

1.  Navigate to **Actions** -> **FPL Automated Tactical Engine & Comparison**.
2.  Click **Run workflow**.
3.  Configure parameters (e.g., `NEUTRAL`, `SHIELD`, `CHASE`, Chip Deployment, xMins Overrides).
4.  Monitor your local runner for the successful FBref scrape and Discord for the final payload.