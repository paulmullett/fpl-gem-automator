# ⚽ FPL Quantitative Decision Engine & Tactical Audit Pipeline

An institutional-grade Fantasy Premier League (FPL) quantitative modeling suite. This pipeline fuses Mixed-Integer Linear Programming (MILP), multi-period transfer banking trees, Monte Carlo stochastic risk modeling, live bookmaker odds intensity solvers, and Google Gemini AI orchestration to deliver automated, data-driven squad optimizations directly to Discord.

---

## 🏛️ Core System Architecture

    ┌────────────────────────┐
    │  FPL API / Bookmakers  │
    └───────────┬────────────┘
                │
                ▼
    ┌──────────────────┐           ┌────────────────────────┐           ┌──────────────────┐
    │  GitHub Actions  │ ────────► │   Python MILP Solver   │ ────────► │  Google Gemini   │
    │  User Interface  │           │ (fpl_mpo_engine/funcs) │           │ (3.6-Flash Engine)│
    └──────────────────┘           └───────────┬────────────┘           └───────────┬──────┘
                                               │                                    │
                                               ▼                                    ▼
                                   ┌────────────────────────┐           ┌──────────────────┐
                                   │ fpl_state.json Ledger  │           │ Discord Webhook  │
                                   └────────────────────────┘           └──────────────────┘

The system comprises two distinct dispatch workflows:
1. **FPL Automated Tactical Engine (`fpl_bot.py`):** Runs complete squad optimization, chip threshold audits, risk posture calculations, and multi-period transfer roadmaps.
2. **FPL Player Comparison Tool (`fpl_compare_players.py`):** Performs head-to-head audits between any two players across 1-GW EV and 4-GW Horizon xP.

---

## 🧮 Mathematical & Quantitative Foundation

### 1. Game-State Normalisation & Bayesian Shrinkage
Expected goal involvements (xGI = npxG + xAG) for low-minute or fringe assets are mathematically shrunken toward positional baselines:
*Shrunken xGI = (Adjusted xGI * Confidence) + (Positional Baseline * (1 - Confidence))*
- **Confidence Modifier:** Scaled by player cost premiums and overall ownership consensus.
- **Accidental Assist Rule:** Raw cross volume into the 18-yard box is weighted to capture deflection assist returns.

### 2. Bayesian League Strength Translation Matrix
Foreign summer arrivals lacking Premier League historical baselines are scaled through position-aware competition multipliers, age adaptation curves, and team expected goal dominance ratios:
*Translation Factor = MAX(0.65, MIN(0.98, League Coef * Age Modifier * xG Ratio))*

### 3. DefCon & 2026/27 BPS Mathematics
- **Defenders (>8.5 CBIT Baseline):** Earn +1 BPS per 3 Clearances, Blocks, Interceptions, and Tackles.
- **Midfielders (>10.5 CBIRT Baseline):** Evaluated including recoveries to maintain floor scoring in low-margin fixtures.
- **Dribbler Protection:** Zero BPS deduction for being tackled. Penalty goals locked at flat 12 BPS across all positions.

### 4. Exponential Poisson Clean Sheet Engine
Defender and Goalkeeper Clean Sheet probability is derived using exponential decay gated by a logistic 60-minute appearance threshold:
*P(Clean Sheet) = exp(-xGA * Mins Factor) * [1 / (1 + exp(-0.15 * (xMins - 60)))]*

---

## 🌲 Multi-Period Optimization (MPO) & Transfer Banking

The solver (`fpl_mpo_engine.py`) models transfers across a rolling 4-gameweek horizon.

- **5-Transfer Banking Curve:** Evaluates rolling 0-transfer weeks as an appreciating capital asset, allowing free transfers to bank up to 5 for multi-transfer "mini-wildcards" without taking point hit penalties.
- **Point Hit Constraint (-4):** Hits are strictly forbidden unless the incoming asset's 4-GW Expected Value exceeds the outgoing asset by **> 5.5 points**, or if required to field 11 starting players.
- **Rank Threat Gravity (Risk Posture):**
  - **SHIELD Mode:** Applies a quadratic penalty to missing high-ownership template assets to protect high overall ranks.
  - **CHASE Mode:** Inverts ownership weightings to hunt differential gains.
  - **NEUTRAL Mode:** Standard structural baseline optimization.

---

## 📊 Stochastic Monte Carlo Simulation Engine

To account for expected minutes variance and explosive ceiling outcomes, `fpl_monte_carlo.py` runs **1,000 trial simulations** per starting asset using:
1. **Beta Distribution:** Models minutes played volatility based on expected substitution patterns.
2. **Poisson Distribution:** Simulates attacking returns per trial scaled by simulated pitch time.

The engine outputs explicit 10th percentile stochastic floors and 90th percentile stochastic ceilings in the Discord report.

---

## 🎲 Live Bookmaker Odds Intensity Solver

The odds engine (`fpl_odds_engine.py`) ingests live betting market lines via The Odds API.
1. **Overround Elimination:** Removes bookmaker vigorish to extract clean implied probabilities.
2. **Poisson Goal Intensity:** Solves non-linear equations using `scipy.optimize.fsolve` to derive expected match goals.
3. **Ensemble Blending:** Blends structural EV (70%) with live market intensity metrics (30%).

---

## 🛠️ Repository Directory Map

    ├── .github/workflows/
    │   ├── fpl_engine.yml          # Primary execution workflow (UI dispatch)
    │   └── fpl_player_compare.yml  # Head-to-Head player comparison workflow
    ├── fpl_bot.py                  # Gemini AI orchestrator & Discord publisher
    ├── fpl_compare_players.py      # Player comparison engine (Unicode & H2H)
    ├── fpl_funcs.py                # Core mathematical EV models & translation matrices
    ├── fpl_monte_carlo.py          # Stochastic 1,000-trial risk simulator
    ├── fpl_mpo_engine.py           # Mixed-Integer Linear Programming (PuLP) solver
    ├── fpl_odds_engine.py          # Live bookmaker odds & fsolve Poisson engine
    ├── fpl_state.json              # Persistent PID state & error tracking ledger
    └── README.md                   # System documentation

---

## ⚙️ Environment Secrets & Configuration Setup

To deploy this repository, configure the following secrets in your GitHub repository (**Settings -> Secrets and variables -> Actions**):

| Secret Name | Description |
| :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DISCORD_WEBHOOK_URL` | Discord Channel Webhook URL for automated alerts |
| `FPL_TEAM_ID` | Your numeric FPL Entry Team ID |
| `ODDS_API_KEY` | *(Optional)* The Odds API key for live betting odds ingestion |

---

## 🚀 Execution Guide

### Running Main Squad Optimization
1. Navigate to **Actions** -> **FPL Automated Tactical Engine & Comparison**.
2. Click **Run workflow**.
3. Configure parameters:
   - **Run Type:** `auto`, `pre_gameweek_deadline`, or `post_gameweek_review`.
   - **Risk Posture:** `NEUTRAL`, `SHIELD`, or `CHASE`.
   - **Chip Deployment:** `NONE`, `WILDCARD`, `FREE_HIT`, `BENCH_BOOST`, `TRIPLE_CAPTAIN`.
   - **xMins Overrides (Human Oracle):** e.g., `Haaland:0, Saka:45`.

### Running Player Head-to-Head Comparison
1. Navigate to **Actions** -> **FPL Player Comparison Tool**.
2. Input `player_a` (e.g., `Estevao`) and `player_b` (e.g., `Palmer`).
3. Click **Run workflow** to publish the comparison matrix to Discord.
