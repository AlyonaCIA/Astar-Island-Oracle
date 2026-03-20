# Astar Island Oracle

Our toolkit for the **Astar Island** ML challenge — a Norse civilisation simulator where you observe a black-box world through limited viewports and predict the final terrain state. See [CHALLENGE.md](CHALLENGE.md) for full challenge details.

---

## Table of Contents

- [The Challenge in Brief](#the-challenge-in-brief)
- [What This Codebase Does](#what-this-codebase-does)
- [Setup](#setup)
- [Production Pipeline](#production-pipeline)
  - [Architecture — unet\_cond](#architecture--unet_cond)
  - [Viewport Strategy — Coverage First](#viewport-strategy--coverage-first)
  - [Observation Encoding](#observation-encoding)
  - [Training — Offline with Multi-Replay Data](#training--offline-with-multi-replay-data)
  - [Cross-Validation](#cross-validation)
  - [Observation Dropout](#observation-dropout)
  - [Automated Pipeline — cron.py](#automated-pipeline--cronpy)
- [Manual Training & Evaluation](#manual-training--evaluation)
- [Data Analysis](#data-analysis)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)

---

## The Challenge in Brief

A 40×40 procedurally generated map runs a Norse simulation for 50 years: settlements grow, factions clash, trade routes form, winters destroy. Each round gives you **5 seeds** (same map, different random outcomes) and **50 viewport queries** (max 15×15 each, shared across seeds). You submit an **H×W×6 probability tensor** per seed predicting 6 terrain classes per cell. Scored by entropy-weighted KL divergence against Monte Carlo ground truth — only dynamic cells matter.

**Critical property:** Simulations are **stochastic** — querying the same viewport on the same seed twice gives different results (each query runs an independent simulation with a different RNG seed). This means overlapping viewports provide genuinely new information, not redundant data.

## What This Codebase Does

| File | Purpose |
|------|---------|
| `cron.py` | **Production pipeline** — automated 20-min cycle: submit fallback → query viewports → submit CNN predictions → retrain |
| `astar_cnn.py` | Live inference: viewport strategy, observation encoding, model loading, prediction submission |
| `train_cnn.py` | Offline training with ground truth + simulation replay data |
| `eval_cnn.py` | Evaluate checkpoints against competition metric |
| `compare_models.py` | Train & compare architectures side-by-side |
| `observations_viz.py` | Visualize viewport coverage |

---

## Setup

### Prerequisites

- Python 3.8+, PyTorch, requests, numpy
- Account on [app.ainm.no](https://app.ainm.no)

### Install

```bash
pip install requests numpy torch
```

### Configure JWT token

1. Log in at [app.ainm.no](https://app.ainm.no)
2. Browser DevTools → Application → Cookies → copy `access_token`
3. `cp .env.example .env` and paste your token as `ASTAR_TOKEN=...`

---

## Production Pipeline

The production system uses **unet_cond** — a MiniUNet trained on multi-replay observation data with entropy-weighted KL loss matching the competition metric.

### Architecture — unet_cond

**Input:** 21 channels = 14 terrain features + 7 observation channels
- Channels 0–7: one-hot terrain encoding (8 terrain types)
- Channels 8–13: neighbour class frequency counts (local spatial context)
- Channels 14–19: observed class frequency per pixel (from viewport queries)
- Channel 20: coverage indicator — `log(1 + observation_count)`

**Output:** 6-class probability distribution per pixel (softmax)

**Architecture:** MiniUNet (~472K params) with dropout=0.1. Encoder: 3 levels (64→128→256) with 3×3 convolutions + batch norm + ReLU + max pool. Decoder: transposed convolutions with skip connections. 1×1 output head → softmax.

**Key insight:** By conditioning on observations, the model learns `P(full map distribution | terrain, partial observation)`. Observed cells get near-perfect predictions; unobserved cells benefit from spatial context propagated through the UNet.

### Viewport Strategy — Coverage First

The query strategy guarantees **100% map coverage** before spending any budget on resampling. Implemented in `astar_cnn.py :: collect_observations()`.

#### Phase 1 — Full Coverage (45 queries)

Uses systematic tile grid: 9 tiles of 15×15 covering the entire 40×40 map.

```
Tile positions (3×3 grid):
      x=0       x=12      x=25
y=0   [A 15×15  ][B 15×15  ][C 15×15  ]
y=12  [D 15×15  ][E 15×15  ][F 15×15  ]
y=25  [G 15×15  ][H 15×15  ][I 15×15  ]

→ 100% coverage, 375 cells (23%) observed 2× from natural overlap
→ 9 tiles × 5 seeds = 45 queries
```

**Priority ordering:** Tiles are scored by terrain dynamism (`score_tile`) — settlements and ports first, ocean/mountain last. The first tile per seed acts as a "scout" — after all 5 scouts complete, cross-seed dynamism intelligence re-ranks remaining tiles so high-activity areas are observed sooner.

#### Phase 2 — Smart Resample (5 remaining queries)

Remaining budget re-queries the most dynamic viewports across all seeds:
1. Rank observed viewports by `change_rate` (fraction of cells that differ from initial terrain)
2. Include cross-seed-informed targets for under-sampled seeds
3. Round-robin through top candidates

Since each query runs an independent stochastic simulation, re-querying a viewport provides a genuinely new sample — improving the empirical distribution estimate for that region.

### Observation Encoding

Viewport query results are encoded into 7 channels via `encode_obs_channels()`:

| Channel | Content | Range |
|---------|---------|-------|
| 0–5 | Observed class frequency per pixel | [0, 1] — normalized count for each of 6 classes |
| 6 | Coverage indicator: `log(1 + n_observations)` | 0 = unobserved, >0 = observed |

Multiple observations of the same cell (from overlapping tiles or resample queries) are **averaged into frequencies**, capturing the stochastic variation. Unobserved cells are all zeros — the model learns to fall back to terrain-only prediction for these.

### Training — Offline with Multi-Replay Data

**Data sources:**
- **Ground truth:** 40 samples in `data/ground_truth/` (rounds 1–8, 5 seeds each). Each is a 40×40×6 probability tensor from Monte Carlo simulation.
- **Simulation replays:** 40 files in `simulation_replays/` (rounds 1–8, 5 seeds each). Each contains 51 frames (years 0–50) of the full 40×40 grid.

**Multi-replay observation generation** (`sample_multi_replay_obs_channels`):

In production, each viewport query triggers an independent stochastic simulation — so different viewports observe different random outcomes even on the same seed. The training pipeline replicates this:

1. Loads all 5 replay grids (final frames) for the round
2. Places viewports using the same systematic 3×3 tile grid as production
3. For each tile, randomly selects one of the 5 available replay grids — so overlapping tiles may show different stochastic outcomes, just like production
4. Encodes the result as 7 observation channels (6 class frequencies + coverage)

For each ground truth sample:
- **3 copies** with different random multi-replay observation patterns
- **1 copy** with zero observations (teaches terrain-only fallback)

This yields **160 training maps** from 40 ground truth samples (no rotation/flip augmentation — observation spatial patterns must match production).

**Loss function:** Entropy-weighted KL divergence, matching the competition scoring metric exactly:
$$\text{loss} = \frac{\sum_{\text{cell}} \text{KL}(\text{cell}) \cdot H(\text{cell})}{\sum_{\text{cell}} H(\text{cell})}$$
where $H(\text{cell})$ is the entropy of the ground truth distribution. Only dynamic cells (non-zero entropy) contribute — static cells like deep ocean are ignored. This aligns training directly with competition scoring: $\text{score} = 100 \cdot e^{-3 \cdot \text{loss}}$.

### Cross-Validation

`unet_cond` supports 4 cross-validation modes:

| Mode | Flag | What it does | Use case |
|------|------|-------------|----------|
| **All data** | `--cv all` | Trains on everything, no holdout | **Production** (default for `unet_cond`) |
| **Round K-fold** | `--cv round_kfold` | K fresh models (one per round as holdout), reports mean ± std | **Evaluation** — reliable generalization estimate |
| **Leave-one-round-out** | `--cv round` | Holds out latest round as validation | Quick single-split estimate |
| **4-fold quadrant** | `--cv quadrant` | Spatial split, holds out one map quadrant per fold | Terrain-only models (no obs channels) |

**Production (`--cv all`):** Trains on all available data with no holdout. Since the model will face entirely new maps in production, withholding data for validation only reduces training signal. The model is evaluated separately using `--cv round_kfold`.

**Evaluation (`--cv round_kfold`):** The gold-standard evaluation mode. Trains K separate models from scratch (K = number of rounds), each holding out one round. Reports per-fold and mean ± std validation loss. This answers: **how well does the model generalize to a never-before-seen map?**

```bash
# Evaluate generalization (trains K fresh models):
python train_cnn.py --model unet_cond --cv round_kfold

# Train for production (all data, no holdout):
python train_cnn.py --model unet_cond --cv all
```

**Why not quadrant CV for obs-conditioned models?** The model takes observation channels as input covering the full 40×40 map. In quadrant CV, the held-out quadrant's pixels are masked from the *target* but their observation channels remain visible in the *input* — leaking the answer. Round-based CV holds out entire maps, so the model never sees any data from the validation round.

**Quadrant CV is still available** for terrain-only architectures:
```bash
python train_cnn.py --model quick --cv quadrant
```

### Observation Dropout

During training, observation channels are randomly masked to build robustness:

| Action | Probability | Purpose |
|--------|-------------|---------|
| Keep all observations | 65% | Matches production (100% coverage) |
| Random partial masking (3–9 viewports of 15×15) | 25% | Handles incomplete coverage gracefully |
| Zero all observations | 10% | Safety net for terrain-only fallback |

### LR Scheduler & Early Stopping

When validation data is available (`--cv round`, `--cv round_kfold`, `--cv quadrant`):

- **ReduceLROnPlateau:** Halves the learning rate after 100 epochs of no validation improvement (min LR: 1e-6)
- **Early stopping:** Stops training after 300 epochs without validation improvement

When training with `--cv all` (production), neither applies since there is no validation set — the model trains for the full epoch count.

### Automated Pipeline — cron.py

Runs every 20 minutes in a continuous loop:

```
┌─ Check for active round
├─ Submit prior-based fallback (guarantees a score)
├─ Collect observations (2-phase viewport strategy)
├─ Load pretrained checkpoint → submit CNN predictions
├─ Wait 10 min for ground truth availability
└─ Retrain (up to 4,000 new epochs from latest checkpoint)
```

Configuration (in `cron.py`):

| Setting | Value | Description |
|---------|-------|-------------|
| `ARCH` | `unet_cond` | Model architecture |
| `POLL_INTERVAL_S` | 1200 (20 min) | Check interval |
| `GT_WAIT_S` | 600 (10 min) | Wait for GT before retrain |
| `MAX_TRAIN_EPOCHS` | 4000 | Max epochs per training cycle |

Retraining uses `--cv all` (train on everything) and inference uses snapshot ensemble (last 3 checkpoints).

```bash
python cron.py          # run forever
python cron.py --once   # single cycle
```

---

## Manual Training & Evaluation

### Train

```bash
# Production model (unet_cond)
python train_cnn.py --model unet_cond                     # default: --cv all (no holdout)
python train_cnn.py --model unet_cond --cv round_kfold    # evaluate generalization (K fresh models)
python train_cnn.py --model unet_cond --reset             # train from scratch
python train_cnn.py --model unet_cond --forever           # train until Ctrl+C
python train_cnn.py --model unet_cond --fetch             # fetch GT from API first
python train_cnn.py --model unet_cond --epochs 2000       # override epoch count

# Simple CNN baseline
python train_cnn.py --model quick --cv quadrant           # terrain-only, quadrant CV
```

### Evaluate

```bash
python eval_cnn.py --arch unet_cond                       # latest checkpoint
python eval_cnn.py --arch quick                           # simple CNN baseline
python eval_cnn.py --arch unet_cond --viewports           # viewport-restricted eval
```

### Compare architectures

```bash
python compare_models.py --eval-only                      # eval all existing checkpoints
python compare_models.py --eval-only --models unet_cond quick
```

---

## Data Analysis

```bash
python observations_viz.py   # visualize viewport coverage
```

---

## Key Findings

From analysis across 7 rounds of observation data:

1. **Simulations are stochastic:** 294/306 overlapping viewport pairs show cell-level differences. 28.8% of overlapping cells have different values across queries.

2. **Replays ≠ observations:** Static cells (ocean/mountain) match 100% between replay final frames and observation viewports. Dynamic cells match only 71.8% — confirming independent stochastic runs.

3. **Variation rate varies by round:** Some maps have low dynamism (~5% variation rate on round 3) while others are highly dynamic (~53% on round 6). The model must handle both extremes.

---

## Project Structure

```
Astar-Island-Oracle/
├── cron.py                  # Production pipeline (automated 20-min cycle)
├── astar_cnn.py             # Live inference + viewport strategy
├── train_cnn.py             # Offline training (ground truth + replays)
├── eval_cnn.py              # Checkpoint evaluation
├── compare_models.py        # Architecture comparison
├── observations_viz.py      # Viewport coverage visualization
├── CHALLENGE.md             # Full challenge specification
├── data/
│   ├── ground_truth/        # GT files: r{N}_s{M}_{id}.json (40×40×6 probabilities)
│   ├── observations_*.json  # Saved viewport queries per round
│   └── round_*.json         # Cached round metadata
├── simulation_replays/      # Replay files: r{N}s{M}.json (51 frames each)
├── checkpoints_unet_cond/   # unet_cond checkpoints (production)
└── checkpoints/             # QuickCNN checkpoints
```

---

## Model Architectures

| Architecture | Input Channels | Params | Description | CV Default | Loss |
|---|---|---|---|---|---|
| `unet_cond` | 21 (14 terrain + 7 obs) | ~472K | **Production.** Multi-replay obs, entropy-weighted KL, no augmentation | `all` | entropy-weighted KL |
| `quick` | 14 | ~35K | QuickCNN (2-layer, terrain-only baseline) | `quadrant` | KL divergence |
