# FPL Quantitative Decision Engine & Tactical Audit Pipeline (2026/27 Season)

An institutional-grade Fantasy Premier League (FPL) quantitative modeling suite. The pipeline fuses Mixed-Integer Linear Programming (MILP), an 8-Gameweek multi-period deep-tree horizon optimization engine, Monte Carlo stochastic risk modeling, live bookmaker odds intensity solvers, and Google Gemini AI orchestration to deliver automated, data-driven squad optimizations directly to Discord.

## Acknowledgements & Academic Citations
This project implements machine learning methodologies inspired by the OpenFPL forecasting framework and multi-period linear programming techniques.
* **Citation:** Groos, D. (2025). *OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services*. arXiv preprint arXiv:2508.09992.
* **License Note:** Certain structural machine learning techniques herein are adapted from the OpenFPL methodology, originally distributed under the MIT License by Daniel Groos.

---

## System Architecture

```
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ Local GitHub Runner   │ │ Live Bookmaker API     │ │ FPL Review Matrix      │
│ (Scrapes FBref Data)   │ │ (fpl_odds_engine.py)   │ │ (fplreview.csv)        │
└───────────┬────────────┘ └───────────┬────────────┘ └───────────┬────────────┘
            │                          │                          │
            ▼                          ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ ML Data Pipeline (run_ml_pipeline.py / train_models.py)                        │
│ - Position-Specific Tri-Model Regressor Ensemble                               │
│ - Piecewise Non-Linear xMins Scaling & 8-GW EV Matrix                          │
│ Outputs: ml_projections.json                                                   │
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Python MILP Optimization Engine (fpl_bot.py / fpl_mpo_engine.py)              │
│ - HiGHS / CBC Branch-and-Bound Solver (8-GW Horizon Matrix)                    │
│ - Dynamic Poisson Binomial Bench Probability Matrix                            │
│ - Game-Theory Risk Postures (NEUTRAL / SHIELD / CHASE)                         │
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ Google Gemini AI       │ │ fpl_state.json Ledger │ │ Discord Webhook        │
│ (gemini-3.6-flash)     │ │ (Persistent State)     │ │ (Formatted Output)     │
└────────────────────────┘ └────────────────────────┘ └────────────────────────┘
```

The system operates across a synchronized execution track:
1. **ML Data Pipeline (`run_ml_pipeline.py` -> `ml_engine/train_models.py`):** Ingests official FPL API statistics, scrapes global FBref data natively via `soccerdata`, and parses 8-Gameweek crowdsourced $xMins$ matrices from FPL Review. Computes underlying Defensive Contribution metrics (`CBIT`/`CBIRT`), trains a Tri-Model Regressor Ensemble, applies piecewise non-linear minute scaling, and outputs an 8-GW EV matrix in `ml_projections.json`.
2. **MILP Co-Optimization & Discord Oracle (`fpl_bot.py` -> `fpl_mpo_engine.py`):** Solves an 8-Gameweek Multi-Period Linear Program using the high-performance `HiGHS` solver. Co-optimizes Starting XI EV, $2.0\times$ captaincy multipliers, goalkeeper handcuffs, and dynamic bench probabilities. Gemini 3.6 Flash formats the output under strict manager agnosticism before publishing to Discord.

---

## Mathematical & Quantitative Foundation

### 1. 8-Gameweek Piecewise Non-Linear Expected Value
Player Expected Value ($EV$) is evaluated dynamically across an 8-Gameweek horizon ($t \in [0, 7]$). Appearance points lock in via a Sigmoid probability curve around the 60-minute threshold, while attacking and defensive returns scale linearly with time on pitch:

$$P(\text{Min} \ge 60) = \frac{1}{1 + e^{-0.15(xMins_t - 60.0)}}$$

$$EV_t = \left[ \left(P(\text{Min} \ge 60) \cdot 2.0 + (1 - P(\text{Min} \ge 60)) \cdot \frac{xMins_t}{60.0}\right) + \left(\max(0, \text{BaseEV} - 2.0) \cdot \frac{xMins_t}{90.0}\right) \right] \cdot \text{DefRating}_t$$

### 2. Live Bookmaker Odds Calibration (`scipy.optimize.fsolve`)
Using `The Odds API`, the engine fetches decimal odds for H2H and Over/Under 2.5 markets, removes bookmaker margin (`remove_vig`), and solves for team Poisson intensity parameters ($\lambda$) to derive true clean sheet and attacking probabilities:

$$P(\text{Under } 2.5) = e^{-\lambda} \left(1 + \lambda + \frac{\lambda^2}{2}\right)$$

### 3. Defensive Contribution (`DefCon`) BPS Mechanics
Defenders and Midfielders earn additional expected points based on Poisson tail probabilities of hitting defensive action thresholds (>8.5 `CBIT` for defenders, >10.5 `CBIRT` for midfielders) under the updated BPS scoring rules:

$$P(X \ge k) = 1 - \sum_{i=0}^{k-1} \frac{e^{-\lambda} \lambda^i}{i!}$$

### 4. Tri-Model Residual Regressor Ensemble
The ML pipeline computes performance adjustment scalars using an ensemble trained on underlying metrics ($xGI$, $xGC$, `CBIT`, Opponent Def Rating) with asymmetric weighting for true haulers ($\ge 8.0$ points):

$$\text{Scalar} = 0.40 \cdot \hat{y}_{\text{XGB}} + 0.40 \cdot \hat{y}_{\text{LGB}} + 0.20 \cdot \hat{y}_{\text{RF}}$$

---

## Multi-Period Optimization (MPO) & Strategy Rules

The solver (`fpl_mpo_engine.py`) enforces the following quantitative constraints:

* **Direct 8-GW EV Matrix Optimization:** Evaluates transfers across true forward-projected expected values, natively accounting for blanks, doubles, and phased rotation.
* **Simultaneous Captaincy Optimization:** Co-optimizes the Starting XI EV and the $2.0\times$ captaincy multiplier simultaneously.
* **Goalkeeper Handshake Constraint:** Calculates the explicit expected value bonus of owning a premium starting goalkeeper and their direct £4.0m backup ("handcuff").
* **Goalkeeper Budget Guardrail:** Restricts total squad goalkeeper spend to $\le £9.5\text{m}$, enforcing capital efficiency and eliminating bench tax.
* **Combinatorial Bench Probability Matrix:** Calculates substitute triggering probabilities ($w_{sub1}, w_{sub2}, w_{sub3}$) using a Poisson Binomial distribution DP matrix derived from starting XI expected minutes.
* **Game-Theory Risk Postures:**
  * `NEUTRAL`: Pure mathematical Expected Value optimization.
  * `SHIELD`: Scales EV by top-10k Effective Ownership (EO) floor metrics to protect rank.
  * `CHASE`: Scales EV by 90th percentile Monte Carlo ceilings to maximize upside.

---

## Directory Map

* **`.github/workflows/`**
  * `fpl_orchestrator.yml`: Master scheduled & manual execution workflow (`clean: false` configured for persistent runner cache).
  * `fpl_player_compare.yml`: Head-to-Head player audit workflow.
* **`ml_engine/`**
  * `data_ingestion.py`: FPL API, FBref, & Understat `soccerdata` ingestion.
  * `train_models.py`: Dataset alignment, 8-GW matrix parsing, & Tri-Model Regressor pipeline.
  * `entity_mapping.json`: Static string overrides for player matching.
* **Core Modules**
  * `fpl_bot.py`: Main orchestrator, Gemini AI client, & Discord publisher.
  * `fpl_compare_players.py`: Head-to-Head player audit script.
  * `fpl_funcs.py`: Bayesian pricing priors, $xMins$ heuristics, & BPS equations.
  * `fpl_mpo_engine.py`: Mixed-Integer Linear Programming solver (`PuLP` + `HiGHS`).
  * `fpl_odds_engine.py`: Live bookmaker odds ingestion & `fsolve` Poisson solver.
  * `fpl_monte_carlo.py`: Stochastic 1,000-trial Beta/Poisson simulator.
  * `run_ml_pipeline.py`: Pipeline entry point.
* **State & Data Files**
  * `fplreview.csv`: Raw 8-Gameweek crowdsourced projection export.
  * `fpl_state.json`: Persistent strategy state, calibration weights, & historical residual errors.
  * `ml_projections.json`: Generated 8-GW ML payload.
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