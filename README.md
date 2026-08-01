# ⚽ FPL Quantitative Decision Engine & Tactical Audit Pipeline

An institutional-grade Fantasy Premier League (FPL) quantitative modeling suite. This pipeline fuses Mixed-Integer Linear Programming (MILP), 8-Gameweek deep-tree horizon optimization, Monte Carlo stochastic risk modeling, live bookmaker odds intensity solvers, and Google Gemini AI orchestration to deliver automated, data-driven squad optimizations directly to Discord.

---

## 🏛️ Core System Architecture

```text
    ┌────────────────────────┐
    │  FPL API / Bookmakers  │
    └───────────┬────────────┘
                │
                ▼
    ┌──────────────────┐            ┌────────────────────────┐            ┌──────────────────┐
    │  GitHub Actions  │ ────────► │   Python MILP Solver   │ ────────► │  Google Gemini   │
    │  User Interface  │            │ (fpl_mpo_engine/funcs) │            │ (3.6-Flash Engine)│
    └──────────────────┘            └───────────┬────────────┘            └───────────┬──────┘
                                                │                                     │
                                                ▼                                     ▼
                                    ┌────────────────────────┐            ┌──────────────────┐
                                    │ fpl_state.json Ledger  │            │ Discord Webhook  │
                                    └────────────────────────┘            └──────────────────┘
```

The system comprises two distinct dispatch workflows:
1. **FPL Automated Tactical Engine (`fpl_bot.py`):** Runs complete squad optimization, chip threshold audits, risk posture calculations, future chip state planning, and multi-period transfer roadmaps.
2. **FPL Player Comparison Tool (`fpl_compare_players.py`):** Performs head-to-head audits between any two players across 1-GW EV and 8-GW Deep-Tree Horizon xP with live price deltas and risk posture gating.

---

## 🧮 Mathematical & Quantitative Foundation

### 1. Game-State Normalisation & Bayesian Shrinkage
Expected goal involvements (xGI = npxG + xAG) for low-minute or fringe assets are mathematically shrunken toward positional baselines:
$$\text{Shrunken xGI} = (\text{Adjusted xGI} \times \text{Confidence}) + (\text{Positional Baseline} \times (1 - \text{Confidence}))$$
- **Confidence Modifier:** Scaled by player cost premiums and overall ownership consensus.
- **Accidental Assist Rule:** Raw cross volume into the 18-yard box is weighted to capture deflection assist returns.

### 2. Bayesian League Strength Translation Matrix
Foreign summer arrivals lacking Premier League historical baselines are scaled through position-aware competition multipliers, age adaptation curves, and team expected goal dominance ratios:
$$\text{Translation Factor} = \max\left(0.65, \min\left(0.98, \text{League Coef} \times \text{Age Modifier} \times \text{xG Ratio}\right)\right)$$

### 3. DefCon & 2026/27 BPS Mathematics
- **Defenders (>8.5 CBIT Baseline):** Earn +1 BPS per 3 Clearances, Blocks, Interceptions, and Tackles.
- **Midfielders (>10.5 CBIRT Baseline):** Evaluated including recoveries to maintain floor scoring in low-margin fixtures.
- **Dribbler Protection:** Zero BPS deduction for being tackled. Penalty goals locked at flat 12 BPS across all positions.

### 4. Exponential Poisson Clean Sheet Engine
Defender and Goalkeeper Clean Sheet probability is derived using exponential decay gated by a logistic 60-minute appearance threshold:
$$P(\text{Clean Sheet}) = e^{-\text{xGA} \times \text{Mins Factor}} \times \frac{1}{1 + e^{-0.15 \times (\text{xMins} - 60)}}$$

### 5. Calibrated Sigmoid Zero-Minute Absence Model
Rather than applying linear absence assumptions, automatic substitution probabilities measure the exact chance of a starter registering precisely 0 minutes using a logistic sigmoid curve:
$$P(\text{Zero Mins}) = \frac{1}{1 + e^{0.1 \times (\text{xMins} - 35.0)}}$$

### 6. Pre-Solve Stochastic Variance Gating
Player point standard deviations ($\sigma$) are calculated *prior* to MILP execution to penalize or reward volatility based on `RISK_POSTURE`:
- **SHIELD Mode:** Applies a $-\sigma \times 0.15$ penalty to volatile assets to defend rank against template ownership.
- **CHASE Mode:** Applies a $+\sigma \times 0.15$ boost to high-ceiling outliers to target rapid rank gains.
- **NEUTRAL Mode:** Standard structural baseline optimization.

---

## 🌲 Multi-Period Optimization (MPO) & Transfer Banking

The solver (`fpl_mpo_engine.py`) models transfers across a rolling **8-gameweek deep-tree horizon** using a geometric decay factor ($\gamma = 0.85$).

- **Stratified Bucket Universe Filtering:** Prevents combinatorial solver hangs while guaranteeing absolute access to price-floor fodder. The engine isolates top EV selections per position while explicitly injecting the **8 cheapest active players** in every position into `valid_ids`.
- **Combinatorial Dynamic Bench Weighting:** Evaluates compound zero-minute probabilities across all 11 starters ($P(X \ge 1)$ and $P(X \ge 2)$) to dynamically scale Sub 1 ($w_{sub1}$) and Sub 2 ($w_{sub2}$) weights, preventing bench capital traps.
- **5-Transfer Banking Curve:** Evaluates rolling 0-transfer weeks as an appreciating capital asset, allowing free transfers to bank up to 5 for multi-transfer "mini-wildcards" without taking point hit penalties.
- **Point Hit Constraint (-4):** Hits are strictly forbidden unless the incoming asset's 8-GW Expected Value exceeds the outgoing asset by **> 5.5 points**, or if required to field 11 starting players.
- **Geographic Vice-Captain Isolation:** Mathematically forces the Vice-Captain to belong to a different real-world Premier League club than the Captain to eliminate single-match postponement risk.
- **Future-Node Chip State Mapping:** Accepts planned chip schedules (`PLANNED_CHIPS`), unlinking squad transition constraints at designated future horizon nodes (e.g., Wildcard in GW6).
- **Live Price Delta Accounting:** Integrates live net transfer momentum into squad financial constraints to preserve team value.
- **HiGHS Engine with CBC Fallback:** Attempts high-performance HiGHS solver execution (60s time limit) with seamless fallback to PuLP's built-in `PULP_CBC_CMD`.

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

```text
├── .github/workflows/
│   ├── fpl_engine.yml          # Primary execution workflow (UI dispatch)
│   └── fpl_player_compare.yml  # Head-to-Head player comparison workflow
├── fpl_bot.py                  # Gemini AI orchestrator, string builder & Discord publisher
├── fpl_compare_players.py      # Player comparison engine (Unicode, 8-GW H2H & Price Deltas)
├── fpl_funcs.py                # Core mathematical EV models, Sigmoid absence & translation matrices
├── fpl_monte_carlo.py          # Stochastic 1,000-trial risk simulator
├── fpl_mpo_engine.py           # Mixed-Integer Linear Programming (PuLP/HiGHS) solver
├── fpl_odds_engine.py          # Live bookmaker odds & fsolve Poisson engine
├── fpl_state.json              # Persistent PID state, Bayesian recalibration & error ledger
└── README.md                   # System documentation
```

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
   - **Planned Future Chips:** e.g., `6:WILDCARD,12:BENCH_BOOST`.
   - **xMins Overrides (Human Oracle):** e.g., `Haaland:0, Saka:45`.

### Running Player Head-to-Head Comparison
1. Navigate to **Actions** -> **FPL Player Comparison Tool**.
2. Configure parameters:
   - **First Player (`player_a`):** e.g., `Haaland`
   - **Second Player (`player_b`):** e.g., `Isak`
   - **Risk Posture:** `NEUTRAL`, `SHIELD`, or `CHASE`.
3. Click **Run workflow** to publish the 8-GW comparison matrix to Discord.
