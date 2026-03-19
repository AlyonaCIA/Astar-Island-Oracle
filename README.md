# Astar Island Oracle

## Challenge Overview

**Goal:** Predict the probability distribution of terrain types across a 40×40 Norse civilisation simulation after 50 years.

**Platform:** [app.ainm.no](https://app.ainm.no) | **API:** `https://api.ainm.no/astar-island/`

### How It Works

1. A round has a **fixed map** with **5 random seeds** and hidden simulation parameters
2. You get **50 viewport queries** (max 15×15 cells each), shared across all 5 seeds
3. Each query runs one stochastic simulation and shows the final state through your viewport
4. Submit a **H×W×6 probability tensor** per seed predicting terrain class probabilities

### Terrain Classes (Prediction)

| Index | Class      | Description                          |
|-------|------------|--------------------------------------|
| 0     | Empty      | Ocean (10), Plains (11), Empty (0)   |
| 1     | Settlement | Active Norse settlement              |
| 2     | Port       | Coastal settlement with harbour      |
| 3     | Ruin       | Collapsed settlement                 |
| 4     | Forest     | Provides food to adjacent settlements|
| 5     | Mountain   | Impassable, static                   |

### Simulation Phases (per year, 50 years total)

1. **Growth** — food production, population growth, port development, expansion
2. **Conflict** — raids, looting, conquest (longships extend range)
3. **Trade** — ports trade food/wealth, tech diffusion
4. **Winter** — food loss, potential collapse → Ruins
5. **Environment** — ruins reclaimed by settlements (→ outpost) or forest

### Key Constraints

- **50 queries total** per round across all 5 seeds
- **Stochastic** — same seed produces different outcomes each run
- **Hidden parameters** — govern world behavior, same across all seeds in a round
- Static cells: Ocean, Mountain (never change), Forest (mostly static)
- Dynamic cells: Settlement, Port, Ruin (the interesting predictions)

### Scoring

- **Entropy-weighted KL divergence** between prediction and ground truth
- Ground truth computed from hundreds of Monte Carlo runs
- Score: `100 × exp(-3 × weighted_kl)` → 0-100 scale
- **Critical:** Never assign 0.0 probability — use floor of 0.01, then renormalize
- Round score = average of 5 seed scores; leaderboard = best round ever

### Initial Data Available

- Map seed → full initial terrain grid (reconstructable locally)
- Settlement positions + port status (no internal stats like population/food)
- Map dimensions (W×H)

## Setup

### Prerequisites

- Python 3.8+
- An account on [app.ainm.no](https://app.ainm.no) (Google sign-in)
- A team (create or join one on the platform)

### Install dependencies

```bash
# Baseline only (no GPU needed)
pip install requests numpy

# CNN script (also needs PyTorch)
pip install requests numpy torch
```

> **PyTorch note:** If `import torch` fails with a DLL error on Windows, install the correct build for your system from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For CPU-only:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Set your JWT token

1. Log in at [app.ainm.no](https://app.ainm.no)
2. Open browser DevTools → Application → Cookies
3. Copy the value of `access_token`
4. Create your `.env` file:

```bash
cp .env.example .env
```

5. Edit `.env` and paste your token:

```
ASTAR_TOKEN=your_jwt_token_here
```

The scripts auto-load `.env` from the project directory — no need to export variables manually.

## Scripts

There are 3 scripts, from simplest to most advanced:

### 1. `astar_uniform.py` — Instant fallback (no queries used)

```bash
python astar_uniform.py
```

Submits a uniform 1/6 distribution for every cell on every seed. Uses **zero queries**. Scores low (~1-5) but guarantees a non-zero score on the board in seconds. Use this to verify your token works and get on the leaderboard immediately.

### 2. `astar_baseline.py` — Prior + observation blending

```bash
python astar_baseline.py
```

Uses your **50 query budget** to observe simulation outcomes through viewports. Builds predictions by blending terrain-based priors with observed data. Takes ~1-2 minutes (1s per query). Scores better than uniform.

### 3. `astar_cnn.py` — CNN-based prediction (recommended)

```bash
python astar_cnn.py
```

The full pipeline:
1. **Immediately submits prior-based fallback** — you have a score even if the CNN is slow
2. **Collects observations** using the query budget (same as baseline)
3. **Saves observations to `data/`** for offline retraining later
4. **Trains a small CNN** mapping initial terrain features → final terrain probabilities
5. **Resubmits CNN predictions** (overwrites the fallback — only last submission counts)

Has a built-in **time limit** (default: 120 minutes). If CNN training runs out of time, the fallback submission stands. Set a custom limit in `.env`:

```
ASTAR_TIME_LIMIT=90
```

Or via environment variable:

```bash
# Windows (PowerShell)
$env:ASTAR_TIME_LIMIT = "90"   # minutes
python astar_cnn.py

# Windows (CMD)
set ASTAR_TIME_LIMIT=90
python astar_cnn.py

# Linux/Mac
ASTAR_TIME_LIMIT=90 python astar_cnn.py
```

### Recommended first-run workflow

1. Run `astar_uniform.py` first to verify auth and get on the board
2. Run `astar_cnn.py` — it will submit a prior-based fallback immediately, then try to improve with the CNN
3. Observation data is saved in `data/` for offline experiments

## Improving the Baseline

Key areas to improve predictions:

- **Smarter query allocation** — focus viewports on dynamic areas (near settlements), not static ocean/mountain
- **Multi-observation averaging** — query the same viewport multiple times per seed for better statistics
- **Simulation modeling** — implement growth/conflict/trade/winter mechanics to forecast outcomes
- **Cross-seed learning** — all 5 seeds share the same hidden parameters; insights from one seed apply to others
- **Neighborhood features** — settlement survival depends on adjacent terrain (food from forests, coastal ports)