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

---

## Setup

### Prerequisites

- Python 3.8+
- An account on [app.ainm.no](https://app.ainm.no) (Google sign-in)
- A team (create or join one on the platform)

### Install dependencies

```bash
# Baseline / uniform (no GPU needed)
pip install requests numpy

# CNN training (also needs PyTorch)
pip install requests numpy torch
```

> **PyTorch note:** If `import torch` fails with a DLL error on Windows, install the correct build from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For CPU-only:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Configure your JWT token

1. Log in at [app.ainm.no](https://app.ainm.no)
2. Open browser DevTools → Application → Cookies → copy `access_token`
3. Create your `.env` file:

```bash
cp .env.example .env
```

4. Edit `.env` and paste your token:

```
ASTAR_TOKEN=your_jwt_token_here
```

All scripts auto-load `.env` from the project directory.

---

## Methods

The project contains **three live submission methods** and an **offline training + evaluation pipeline**.

### Method 1 — Uniform Fallback (`astar_uniform.py`)

The simplest possible submission: every cell gets a uniform 1/6 probability for each class.

- **Queries used:** 0
- **Expected score:** ~1–5 (low, but non-zero)
- **Use case:** Verify auth works, get on the leaderboard instantly

```bash
python astar_uniform.py
```

### Method 2 — Prior + Observation Blending (`astar_baseline.py`)

Uses terrain-based hand-tuned priors (e.g., ocean cells → 95% Empty) and refines them by spending the 50-query budget to observe actual simulation outcomes.

- **Queries used:** up to 50
- **Runtime:** ~1–2 min (1s sleep per query for rate limiting)
- **Expected score:** moderate — better than uniform, limited by hand-tuned priors

```bash
python astar_baseline.py
```

### Method 3 — CNN with Fallback (`astar_cnn.py`)

Full pipeline that guarantees a submission even if the CNN fails:

1. Fetches round info from API
2. **Immediately submits prior-based fallback** (same quality as Method 2, without using queries)
3. Spends query budget collecting observations → saves to `data/`
4. Trains a small CNN (14→32→32→6 channels) on the collected data
5. Resubmits CNN predictions (overwrites fallback — only the last submission counts)

If PyTorch is not installed, or the CNN crashes for any reason, the fallback from step 2 still stands.

- **Queries used:** up to 50
- **Runtime:** 2–10 min depending on training
- **Fallback:** always — PyTorch errors are caught gracefully

```bash
python astar_cnn.py
```

**Time limit** (default 120 min, configurable):

```bash
# In .env
ASTAR_TIME_LIMIT=90

# Or inline (PowerShell)
$env:ASTAR_TIME_LIMIT="90"; python astar_cnn.py
```

### Method 4 — Offline CNN Training (`train_cnn.py` + `eval_cnn.py`)

Train the CNN **offline** using ground truth data from completed rounds — no query budget consumed, no time pressure.

#### Step 1: Train

```bash
python train_cnn.py
```

What it does:

1. Queries `GET /rounds` to find completed/scoring rounds
2. Downloads ground truth via `GET /analysis/{round_id}/{seed_index}` for each seed
3. **Caches everything in `data/ground_truth/`** — re-running skips the download
4. Encodes initial grids into 14-channel feature tensors
5. Splits each 40×40 map into 4 quadrants: **3 for training, 1 for validation** (bottom-right by default)
6. Trains with **KL divergence loss** (same metric as competition scoring)
7. Saves checkpoints every 25 epochs to `checkpoints/`
8. Saves `checkpoints/cnn_latest.pt` at the end

**Resumable:** If interrupted (Ctrl+C), re-running picks up from the latest checkpoint automatically.

Training hyperparameters (via env vars or `.env`):

| Variable | Default | Description |
|---|---|---|
| `ASTAR_TRAIN_EPOCHS` | 300 | Total training epochs |
| `ASTAR_TRAIN_LR` | 0.001 | Learning rate |
| `ASTAR_TRAIN_BATCH` | 64 | Batch size (number of maps per batch) |
| `ASTAR_CKPT_EVERY` | 25 | Checkpoint frequency (epochs) |
| `ASTAR_VAL_QUADRANT` | 3 | Validation quadrant (0=TL, 1=TR, 2=BL, 3=BR) |

Example with custom settings:

```bash
# PowerShell
$env:ASTAR_TRAIN_EPOCHS="500"; $env:ASTAR_TRAIN_LR="5e-4"; python train_cnn.py
```

#### Step 2: Evaluate

```bash
python eval_cnn.py                                   # evaluates cnn_latest.pt
python eval_cnn.py checkpoints/cnn_epoch_0100.pt     # specific checkpoint
```

Compares the CNN checkpoint against two baselines on the validation quadrant:

- **CNN** — trained model predictions
- **Prior** — hand-tuned priors (same as Method 2)
- **Uniform** — 1/6 everywhere

Output includes:

- Per-seed simulated competition score (`100 × exp(-3 × wKL)`)
- Per-seed validation KL divergence
- Averages across all seeds
- Per-class KL breakdown (which terrain types the CNN struggles with)

---

## Recommended Workflow

### First time (get on the board fast)

```bash
python astar_uniform.py       # instant score, zero queries
python astar_cnn.py           # burns queries, submits fallback + CNN
```

### After a round completes (improve for next round)

```bash
python train_cnn.py           # download ground truth, train offline
python eval_cnn.py            # check how well the CNN does
# Tweak hyperparameters, retrain, repeat
```

### During an active round (submit the best you have)

```bash
# Quick fallback-only submission (no queries burned)
$env:ASTAR_TIME_LIMIT="0.1"; python astar_cnn.py

# Full pipeline
python astar_cnn.py
```

---

## Project Structure

```
Astar-Island-Oracle/
├── .env.example          # Token + config template
├── .env                  # Your actual config (git-ignored)
├── .gitignore
├── README.md
├── astar_uniform.py      # Method 1: uniform 1/6 submission
├── astar_baseline.py     # Method 2: prior + observation blending
├── astar_cnn.py          # Method 3: CNN with auto-fallback
├── train_cnn.py          # Method 4: offline training from ground truth
├── eval_cnn.py           # Evaluate a trained checkpoint
├── data/                 # Cached observations & ground truth (git-ignored)
│   └── ground_truth/     # Downloaded analysis data per round/seed
└── checkpoints/          # Saved model checkpoints (git-ignored)
```

---

## Improving Further

- **Smarter query allocation** — focus viewports on dynamic areas (near settlements), not static ocean/mountain
- **Multi-observation averaging** — query the same viewport multiple times per seed
- **Simulation modeling** — implement growth/conflict/trade/winter mechanics to forecast outcomes
- **Cross-seed learning** — all 5 seeds share hidden parameters; insights transfer
- **Larger model** — deeper CNN, attention layers, or graph-based approaches
- **Accumulate training data** — every completed round adds more ground truth for offline training