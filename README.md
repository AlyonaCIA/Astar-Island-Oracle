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
4. Trains a small CNN on the collected data
5. Resubmits CNN predictions (overwrites fallback — only the last submission counts)

If PyTorch is not installed, or the CNN crashes for any reason, the fallback from step 2 still stands.

- **Queries used:** up to 50
- **Runtime:** 2–10 min depending on training
- **Fallback:** always — PyTorch errors are caught gracefully

```bash
python astar_cnn.py
```

**Select architecture** via environment variable:

```bash
# PowerShell
$env:ASTAR_MODEL="unet"; python astar_cnn.py
```

Available architectures: `quick` (default), `quick3`, `unet` (see Model Architectures below).

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
python train_cnn.py                      # train default (quick) architecture
python train_cnn.py --model unet         # train a specific architecture
python train_cnn.py --model quick3 --reset  # train from scratch (clear old checkpoints)
python train_cnn.py --fetch              # fetch latest ground truth from API first
python train_cnn.py --forever            # train indefinitely until Ctrl+C
```

What it does:

1. Loads cached ground truth from `data/ground_truth/` (or fetches from API with `--fetch`)
2. Encodes initial grids into 14-channel feature tensors
3. Splits each 40×40 map into 4 quadrants for **4-fold cross-validation** (rotate held-out quadrant)
4. Trains with **KL divergence loss** (same metric as competition scoring)
5. Saves checkpoints every 25 epochs to the architecture's checkpoint directory
6. Saves `cnn_latest.pt` at the end

**Resumable:** If interrupted (Ctrl+C), re-running picks up from the latest checkpoint automatically.

Each architecture saves to its own checkpoint directory:

| Architecture | Checkpoint dir |
|---|---|
| `quick` | `checkpoints/` |
| `quick3` | `checkpoints_quick3/` |
| `unet` | `checkpoints_unet/` |

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
$env:ASTAR_TRAIN_EPOCHS="500"; $env:ASTAR_TRAIN_LR="5e-4"; python train_cnn.py --model unet
```

#### Step 2: Evaluate

```bash
python eval_cnn.py                                   # evaluates cnn_latest.pt (quick arch)
python eval_cnn.py --arch unet                       # evaluates latest unet checkpoint
python eval_cnn.py checkpoints/cnn_epoch_0100.pt     # specific checkpoint (auto-detects arch)
python eval_cnn.py --viewports                       # also evaluate on your actual viewport regions
python eval_cnn.py --arch unet --viewports           # combine options
```

Compares the CNN checkpoint against two baselines:

- **CNN** — trained model predictions
- **Prior** — hand-tuned priors (same as Method 2)
- **Uniform** — 1/6 everywhere

**Default output** (always shown):

- Per-seed simulated competition score (`100 × exp(-3 × wKL)`) on the full map
- Per-seed validation KL divergence on the held-out quadrant
- Averages across all seeds
- Per-class KL breakdown (which terrain types the CNN struggles with)

**`--viewports` flag** — Viewport-restricted evaluation:

When enabled, the evaluator loads your cached observation files (`data/observations_*.json`) from previous rounds and computes metrics **only on the pixels you actually queried** during that round. This tells you how well the model would have performed in the regions you chose to observe, vs the full map.

For each seed with observations, it reports:

- Number of observed pixels (out of 1600 total)
- Entropy-weighted KL and score restricted to those pixels, for CNN / Prior / Uniform
- Averages across all seeds

This is useful for understanding whether your **viewport strategy** (which regions you chose to observe) targeted pixels where the model was already accurate, or where the model was actually struggling.

> **Note:** Observation files are only saved by `astar_cnn.py` during live rounds. If you only have ground truth data (from `--fetch`) but no observation files for a round, the viewport section will be skipped for that round.

#### Step 3: Compare all architectures

```bash
python compare_models.py                          # train all 3 architectures, then compare
python compare_models.py --epochs 100             # fewer epochs for a quick test
python compare_models.py --eval-only              # skip training, just evaluate existing checkpoints
python compare_models.py --models quick unet      # only train/evaluate specific models
python compare_models.py --reset                  # clear all checkpoints and train from scratch
```

Trains each registered architecture on the same data with 4-fold cross-validation, then evaluates all of them plus baselines. Prints a side-by-side comparison table with full-map and validation scores per model.

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
├── eval_cnn.py           # Evaluate a trained checkpoint (--viewports for viewport-only eval)
├── compare_models.py     # Train & compare all architectures side-by-side
├── analyze.py            # Plot training history curves
├── data/                 # Cached observations & ground truth (git-ignored)
│   ├── observations_*.json   # Viewport queries saved by astar_cnn.py during live rounds
│   ├── round_*.json          # Cached round info
│   └── ground_truth/         # Downloaded analysis data per round/seed
├── checkpoints/          # QuickCNN checkpoints
├── checkpoints_quick3/   # QuickCNN3 checkpoints
└── checkpoints_unet/     # MiniUNet checkpoints
```

---

## Model Architectures

All models take a 14-channel input (8 one-hot terrain channels + 6 neighbour class frequency channels) and output a 6-class probability distribution per pixel.

| Name | Key | Layers | Receptive field | Parameters | Notes |
|---|---|---|---|---|---|
| QuickCNN | `quick` | 14→32→32→6 (3×3 convs) | 5×5 | ~12K | Original, fast to train |
| QuickCNN3 | `quick3` | 14→32→32→32→6 (3×3 convs) | 7×7 | ~22K | One extra hidden layer |
| MiniUNet | `unet` | Encoder-decoder with skip connections, 2 pooling stages (40→20→10), bottleneck 128ch | Full map | ~250K | Sees global structure |

All models use Dropout2d (default 0.2), softmax output, and probability floor clamping (0.01).

---

## Improving Further

- **Smarter query allocation** — focus viewports on dynamic areas (near settlements), not static ocean/mountain
- **Multi-observation averaging** — query the same viewport multiple times per seed
- **Simulation modeling** — implement growth/conflict/trade/winter mechanics to forecast outcomes
- **Cross-seed learning** — all 5 seeds share hidden parameters; insights transfer
- **Attention / ensemble** — spatial attention modules, or ensemble predictions from multiple architectures
- **Accumulate training data** — every completed round adds more ground truth for offline training