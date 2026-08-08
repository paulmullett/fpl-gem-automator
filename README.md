# Quantitative Stochastic FPL Optimization Engine

An institutional-grade, zero-LLM Fantasy Premier League (FPL) squad optimization system built on Two-Stage Stochastic Mixed-Integer Linear Programming (MILP), Model Predictive Control (MPC), and Sample Average Approximation (SAA) scenario clustering.

---

## Core Engine Architecture

### 1. Two-Stage Stochastic MILP Solver (`fpl_mpo_engine.py`)
* **Stage 1 (Here-and-Now Decisions):** Optimizes immediate choices for the upcoming deadline (starting XI, formation, captaincy, vice-captaincy, and instant transfers).
* **Stage 2 (Receding Horizon):** Evaluates multi-period transfer trees across an 8-Gameweek horizon across 40 distinct stochastic scenarios. Future expected points (xP) apply an exponential discount factor (gamma = 0.85^t) to prioritize near-term yield while maintaining medium-term squad structure.

### 2. Banked Free Transfer State Tracking (2024/25+ Rules)
* **Accumulation & Carrying Capacity:** Continuous state variables (ft_avail, ft_used) track up to 5 banked Free Transfers across future weeks.
* **Controlled Transfer Hits:** Enforces a hard cap allowing a maximum of 1 extra transfer for a -4 point hit per future week beyond available banked transfers (unless a Wildcard is active). This prevents solver overfitting to noisy distant projections.
* **Transfer Hoarding Incentive:** Incorporates a fractional intrinsic value weight (0.01 pts) for unspent Free Transfers to encourage rolling transfers for multi-FT "mini-wildcards."

### 3. Auto-Sub Valuation & Squad Resilience
* Integrates fractional auto-substitution weighting (w_sub) into the Stage 1 and Stage 2 objective functions.
* Evaluates bench assets based on starting XI rotational risk, preventing the selection of non-playing £4.0m deadwood while maintaining squad depth.

---

## Pipeline Execution Flow

    [Live Data Ingestion] ──> [Stochastic Scenario Oracle] ──> [Two-Stage MILP Solver] ──> [Discord Payload]
     (FPL API / Odds)          (1,000 Futures ──> 40 SAA)      (PuLP / System CBC)        (Pure Python)

1. **`data_ingestion.py` / `scrape_prices.py`:** Pulls live player status, pricing, selling values, bank balance, and fixture difficulty metrics.
2. **`stochastic_oracle.py`:** Generates 1,000 Monte Carlo match simulations and clusters them into 40 weighted SAA scenario matrices.
3. **`fpl_mpo_engine.py`:** Solves the multi-period optimization model using the COIN-OR Branch and Cut (CBC) solver.
4. **`fpl_bot.py`:** Parses the optimal decision vectors and dispatches the deterministic report via Discord Webhook.

---

## Configuration & Environment Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| `FPL_TEAM_ID` | Integer | Official FPL Team ID for live API context ingestion. |
| `DISCORD_WEBHOOK_URL` | String | Discord Webhook URL for automated report delivery. |
| `RISK_POSTURE` | String | Strategy mode: `NEUTRAL` (Max xP), `SHIELD` (Min variance), or `CHASE` (Max upside). |
| `XMINS_INPUT` | String | Manual expected minutes overrides (e.g., `"Saka: 45, Foden: 0"`). |

---

## Local Development & Setup

1. **Clone Repository & Install Dependencies:**
    git clone https://github.com/paulmullett/fpl-gem-automator.git
    cd fpl-gem-automator
    pip install -r requirements.txt

2. **Configure Local Environment:**
    Create a local `.env` file (ensure `.env` is listed in `.gitignore`):
    FPL_TEAM_ID=123456
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    RISK_POSTURE=NEUTRAL

3. **Execute Pipeline:**
    python fpl_bot.py