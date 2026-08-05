# FPL Quantitative Decision Engine & Tactical Audit Pipeline (2026/27 Season)

An institutional-grade Fantasy Premier League (FPL) quantitative modeling suite. The pipeline fuses Mixed-Integer Linear Programming (MILP), an 8-Gameweek deep-tree horizon optimization engine, Monte Carlo stochastic risk modeling, live bookmaker odds intensity solvers, and Google Gemini AI orchestration to deliver automated, data-driven squad optimizations directly to Discord.

## Acknowledgements & Academic Citations
This project implements machine learning methodologies inspired by the OpenFPL forecasting framework. 
* **Citation:** Groos, D. (2025). *OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services*. arXiv preprint arXiv:2508.09992.
* **License Note:** Certain structural machine learning techniques herein are adapted from the OpenFPL methodology, originally distributed under the MIT License by Daniel Groos.

---

## System Architecture

```
┌────────────────────────┐      ┌────────────────────────┐
│   Local GitHub Runner  │      │  Live Bookmaker API    │
│  (Scrapes FBref Data)  │      │  (fpl_odds_engine.py)  │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            ▼                               ▼
┌────────────────────────────────────────────────────────┐
│  ML Data Pipeline (run_ml_pipeline.py / train_models)  │
│  Outputs: ml_projections.json                          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Python MILP Solver    │ ───► │  Google Gemini AI      │
│  (fpl_bot / mpo_engine)│      │  (gemini-3.6-flash)    │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  fpl_state.json Ledger │      │  Discord Webhook       │
└────────────────────────┘      └────────────────────────┘
```

The system operates across a synchronized two-step execution track:
1. **ML Data Pipeline (`run_ml_pipeline.py` -> `ml_engine/train_models.py`):** Ingests official FPL API statistics and scrapes global FBref data natively via `soccerdata`. Maps players across datasets using fuzzy name matching, computes underlying Defensive Contribution metrics (`CBIT`/`CBIRT`), trains a Tri-Model Regressor Ensemble, and outputs a normalized `ml_projections.json` payload.
2. **MILP Co-Optimization & Discord Oracle (`fpl_bot.py` -> `fpl_mpo_engine.py`):** Solves an 8-Gameweek Multi-Period Linear Program (`HiGHS` / `CBC` solver), co-optimizing Starting XI Expected Value (EV), captaincy multipliers, goalkeeper handcuffs, and dynamic bench probabilities. Gemini 3.6 Flash formats the solver output under strict manager agnosticism and null-state guardrails before delivering the report to Discord.

---

## Mathematical & Quantitative Foundation

### 1. Bayesian Pricing Priors & Zero-Data Resolution
During pre-season or early gameweeks when historical Expected Goal Involvement ($xGI$) data evaluates to zero, the engine uses official FPL market prices as a Bayesian prior to establish baseline attacking threat:

$$\text{Prior xGI} = 0.35 + (\max(0.0, \text{Cost} - 5.0) \times 0.12) \quad \text{(Forwards)}$$

### 2. Live Bookmaker Odds Calibration (`scipy.optimize.fsolve`)
Using `The Odds API`, the engine fetches decimal odds for H2H and Over/Under 2.5 markets, removes bookmaker margin (`remove_vig`), and solves for team Poisson intensity parameters ($\lambda$) to derive true clean sheet and attacking probabilities:

$$P(\text{Under } 2.5) = e^{-\lambda} \left(1 + \lambda + \frac{\lambda^2}{2}\right)$$

### 3. Defensive Contribution (`DefCon`) BPS Mechanics
Defenders and Midfielders earn additional expected points based on Poisson tail probabilities of hitting defensive action thresholds (>8.5 `CBIT` for defenders, >10.5 `CBIRT` for midfielders):

$$P(X \ge k) = 1 - \sum_{i=0}^{k-1} \frac{e^{-\lambda} \lambda^i}{i!}$$

### 4. Tri-Model Residual Regressor Ensemble
The ML pipeline computes performance adjustment scalars using an ensemble of three models trained on underlying metrics ($xGI$, $xGC$, `CBIT`, Opponent Frailty):

$$\text{Scalar} = 0.40 \cdot \hat{y}_{\text{XGB}} + 0.40 \cdot \hat{y}_{\text{LGB}} + 0.20 \cdot \hat{y}_{\text{RF}}$$

---

## Multi-Period Optimization (MPO) & Strategy Rules

The solver (`fpl_mpo_engine.py`) enforces the following quantitative constraints:

* **Simultaneous Captaincy Optimization:** Co-optimizes the Starting XI EV and the $2.0\times$ captaincy multiplier simultaneously.
* **Goalkeeper Handshake Constraint:** Calculates the explicit expected value bonus of owning a premium starting goalkeeper and their direct £4.0m backup ("handcuff").
* **Smarter Bench GK Tie-Breaker:** Evaluates independent £4.0m bench goalkeepers via a weighted score ($xMins \times 0.001 + EV \times 0.0001$) to prioritize playing backups over non-playing fodder.
* **Combinatorial Bench Probability Matrix:** Calculates substitute triggering probabilities using a Poisson Binomial distribution DP matrix derived from starting XI expected minutes.
* **Game-Theory Risk Postures:**
  * `NEUTRAL`: Pure mathematical Expected Value optimization.
  * `SHIELD`: Scales EV by top-10k Effective Ownership (EO) floor metrics to protect rank.
  * `CHASE`: Scales EV by 90th percentile Monte Carlo ceilings to maximize upside.

---

## Directory Map

* **`.github/workflows/`**
  * `fpl_orchestrator.yml`: Master scheduled & manual execution workflow.
  * `fpl_player_compare.yml`: Head-to-Head player audit workflow.
* **`ml_engine/`**
  * `data_ingestion.py`: FPL API & FBref `soccerdata` ingestion.
  * `train_models.py`: Dataset alignment, name matching, & Tri-Model Regressor pipeline.
  * `entity_mapping.json`: Static string overrides for player matching.
  * `feature_engineering.py`: Utility feature extraction.
  * `resolution_engine.py`: Entity resolution utilities.
* **Core Modules**
  * `fpl_bot.py`: Main orchestrator, Gemini AI client, & Discord publisher.
  * `fpl_compare_players.py`: Head-to-Head player audit script.
  * `fpl_funcs.py`: Bayesian pricing priors, $xMins$ heuristics, & BPS equations.
  * `fpl_mpo_engine.py`: Mixed-Integer Linear Programming solver (`PuLP`).
  * `fpl_odds_engine.py`: Live bookmaker odds ingestion & `fsolve` Poisson solver.
  * `fpl_monte_carlo.py`: Stochastic 1,000-trial Beta/Poisson simulator.
  * `run_ml_pipeline.py`: Pipeline entry point.
* **State & Data Files**
  * `fpl_state.json`: Persistent strategy state, calibration weights, & historical residual errors.
  * `ml_projections.json`: Generated ML payload.
  * `requirements.txt`: Python package dependencies.
  * `.env`: Local environment variables.

---

## Environment Secrets & Configuration

Configure the following secrets in GitHub Repository Settings (**Settings -> Secrets and variables -> Actions**):

| Secret Name | Required | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Yes | Google Gemini API Key |
| `DISCORD_WEBHOOK_URL` | Yes | Discord Channel Webhook URL |
| `FPL_TEAM_ID` | Yes | Numeric FPL Team Entry ID |
| `ODDS_API_KEY` | Optional | The Odds API key for live betting odds ingestion |

---

## Execution Guide

### Scheduled Runs
The bot runs automatically via GitHub Actions:
* **Thursdays @ 18:00 UTC:** Pre-weekend deadline preparation.
* **Sundays @ 20:00 UTC:** Post-weekend strategic review and model recalibration.

### Manual Trigger (GitHub UI)
1. Go to **Actions -> FPL Master Quantitative Bot**.
2. Click **Run workflow**.
3. Select parameters:
   * **Analysis Type:** `auto`, `pre_gameweek_deadline`, or `post_gameweek_review`.
   * **Active Chip:** `NONE`, `WILDCARD`, `FREE_HIT`, `BENCH_BOOST`, or `TRIPLE_CAPTAIN`.
   * **Human Overrides:** `haaland: 60, saka: 0` (Supports colons, commas, or JSON).
   * **Risk Posture:** `NEUTRAL`, `SHIELD`, or `CHASE`.