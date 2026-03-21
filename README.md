# Astar Island Oracle

Our toolkit for the **Astar Island** ML challenge — a Norse civilisation simulator where you observe a black-box world through limited viewports and predict the final terrain state. See [CHALLENGE.md](CHALLENGE.md) for full challenge details.

---

## Table of Contents

- [The Challenge in Brief](#the-challenge-in-brief)
- [What This Codebase Does](#what-this-codebase-does)
- [Solution Architecture — Overview](#solution-architecture--overview)
  - [End-to-End Process](#end-to-end-process)
  - [Production Submission Flow](#production-submission-flow)
- [Setup](#setup)
- [Usage](#usage)
  - [Automated Pipeline (Production)](#automated-pipeline-production)
  - [Manual Training](#manual-training)
  - [Evaluation](#evaluation)
  - [Data Collection](#data-collection)
  - [Analysis & Visualization](#analysis--visualization)
- [Production Pipeline](#production-pipeline)
  - [What is unet\_cond?](#what-is-unet_cond)
  - [Architecture — unet\_cond](#architecture--unet_cond)
  - [Viewport Strategy — Coverage First](#viewport-strategy--coverage-first)
  - [Observation Encoding](#observation-encoding)
  - [Training — Offline with Multi-Replay Data](#training--offline-with-multi-replay-data)
  - [Cross-Validation](#cross-validation)
  - [Observation Dropout](#observation-dropout)
  - [LR Scheduler & Early Stopping](#lr-scheduler--early-stopping)
  - [Automated Pipeline — cron.py](#automated-pipeline--cronpy)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Solution Architecture — Deep Dive](#solution-architecture--deep-dive)
  - [Network Architecture: MiniUNet](#network-architecture-miniunet)
  - [Input Encoding: 21 Channels](#input-encoding-21-channels)
  - [Data Sources](#data-sources)
  - [Training Procedure](#training-procedure)
  - [Evaluation Pipeline](#evaluation-pipeline--eval_cnnpy)
  - [Post-Processing Steps](#post-processing-steps)
  - [Production Submission Pipeline](#production-submission-pipeline--cronpy--astar_cnnpy)

---

## Solution Architecture — Overview

### End-to-End Process

The full solution lifecycle — from collecting training data through to submitting live predictions:

```
                        ┌───────────────────────────────────────────┐
                        │          OFFLINE (one-time setup)         │
                        │                                           │
                        │  fetch_ground_truth.py                    │
                        │    └─ Download GT from completed rounds   │
                        │       └─ data/ground_truth/r{N}_s{M}.json│
                        │                                           │
                        │  get_replays.py                           │
                        │    └─ Download replay trajectories        │
                        │       └─ replays/r{N}_s{M}_*.json        │
                        │                                           │
                        │  train_cnn.py --model unet_cond --cv all  │
                        │    └─ Build training set:                 │
                        │       ├─ 14ch terrain + 7ch observations  │
                        │       ├─ Multi-replay obs synthesis       │
                        │       └─ Obs dropout augmentation         │
                        │    └─ Train MiniUNet (~472K params)       │
                        │       └─ checkpoints_unet_cond/           │
                        │                                           │
                        │  eval_cnn.py --arch unet_cond --viewports │
                        │    └─ Evaluate on held-out rounds         │
                        │       └─ Score = 100 × e^(-3 × wKL)      │
                        └───────────────┬───────────────────────────┘
                                        │
                        ┌───────────────▼───────────────────────────┐
                        │     PRODUCTION (every 20 min via cron.py) │
                        │                                           │
                        │  1. Detect active round (API)             │
                        │  2. Submit prior-based fallback           │
                        │     └─ Guarantees a score immediately     │
                        │  3. Collect observations (50 queries)     │
                        │     ├─ Phase 1: 45 queries → full map     │
                        │     └─ Phase 2: 5 queries → resample      │
                        │  4. Load checkpoint → predict → submit    │
                        │     ├─ Encode terrain + observations      │
                        │     ├─ MiniUNet forward pass              │
                        │     ├─ Bayesian blend (uncertain pixels)  │
                        │     ├─ Mountain override (deterministic)  │
                        │     └─ Submit H×W×6 per seed via API      │
                        │  5. Wait for ground truth availability    │
                        │  6. Retrain (+2,000 epochs from checkpoint)│
                        └───────────────────────────────────────────┘
```

### Production Submission Flow

What happens when we submit predictions for a live round:

```
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │   DATA SOURCES   │     │  PRE-PROCESSING  │     │     MODEL        │
  │                  │     │                  │     │                  │
  │ API: initial_grid├────►│ encode_initial   │     │                  │
  │   (40×40 terrain)│     │   _grid()        ├──┐  │                  │
  │                  │     │ → 14 channels:   │  │  │                  │
  │                  │     │   8 one-hot terr. │  │  │                  │
  │                  │     │   6 neighbour freq│  │  │                  │
  └──────────────────┘     └──────────────────┘  │  │                  │
                                                 ├──►  MiniUNet       │
  ┌──────────────────┐     ┌──────────────────┐  │  │  (unet_cond)    │
  │  VIEWPORT QUERIES│     │  PRE-PROCESSING  │  │  │                  │
  │  (stochastic)    │     │                  │  │  │  21ch input      │
  │                  │     │ encode_obs       │  │  │  → 6ch output    │
  │ 50 simulate()   ├────►│   _channels()    ├──┘  │  (softmax probs) │
  │  API calls       │     │ → 7 channels:   │     │                  │
  │  (9 tiles × 5    │     │   6 class freq.  │     └────────┬─────────┘
  │   seeds + 5      │     │   1 coverage     │              │
  │   resample)      │     │                  │              ▼
  └──────────────────┘     └──────────────────┘     ┌──────────────────┐
                                                    │ POST-PROCESSING  │
                                                    │                  │
                                                    │ 1. Prob floor    │
                                                    │    (min 0.01)    │
                                                    │    + renormalize │
                                                    │                  │
                                                    │ 2. Bayesian blend│
                                                    │    CNN + obs     │
                                                    │    counts where  │
                                                    │    entropy > 0.54│
                                                    │                  │
                                                    │ 3. Mountain      │
                                                    │    override      │
                                                    │    (terrain=5 →  │
                                                    │     95% class 5) │
                                                    └────────┬─────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │   SUBMISSION     │
                                                    │                  │
                                                    │ POST /submit     │
                                                    │ {round_id,       │
                                                    │  seed_index,     │
                                                    │  prediction:     │
                                                    │   H×W×6 probs}   │
                                                    │                  │
                                                    │ × 5 seeds        │
                                                    └──────────────────┘
```

**Viewport query strategy in detail:**

```
  Budget: 50 queries total, shared across 5 seeds

  Phase 1 — FULL COVERAGE (45 queries = 9 tiles × 5 seeds)
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Systematic 3×3 tile grid on the 40×40 map:         │
  │                                                     │
  │       x=0       x=12      x=25                      │
  │  y=0  ┌─────────┬─────────┬─────────┐              │
  │       │ A 15×15 │ B 15×15 │ C 15×15 │              │
  │  y=12 ├─────────┼─────────┼─────────┤              │
  │       │ D 15×15 │ E 15×15 │ F 15×15 │              │
  │  y=25 ├─────────┼─────────┼─────────┤              │
  │       │ G 15×15 │ H 15×15 │ I 15×15 │              │
  │       └─────────┴─────────┴─────────┘              │
  │                                                     │
  │  → 100% pixel coverage                              │
  │  → 375 cells (23%) observed 2× from natural overlap │
  │                                                     │
  │  Priority: tiles scored by terrain dynamism          │
  │    Settlement: +5 | Port: +6 | Forest: +0.3         │
  │    Plains/Empty: +0.5 | Ocean/Mountain: 0           │
  │    + coastal bonus + cluster bonus                   │
  │                                                     │
  │  First tile per seed = "scout" →                     │
  │    cross-seed dynamism re-ranks remaining tiles      │
  └─────────────────────────────────────────────────────┘

  Phase 2 — SMART RESAMPLE (5 remaining queries)
  ┌─────────────────────────────────────────────────────┐
  │  Re-query most dynamic viewports across all seeds:  │
  │  1. Rank viewports by change_rate                   │
  │     (fraction of cells differing from initial)      │
  │  2. Include cross-seed targets for under-sampled    │
  │  3. Round-robin through top candidates              │
  │                                                     │
  │  Each re-query = independent stochastic simulation  │
  │  → genuinely new sample, not redundant data         │
  └─────────────────────────────────────────────────────┘
```

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
| `eval_round.py` | Round-specific evaluation — compare submitted prediction vs re-inference vs baselines |
| `compare_models.py` | Train & compare architectures side-by-side |
| `fetch_ground_truth.py` | Download ground truth from completed rounds via API |
| `get_replays.py` | Download simulation replay trajectories (51 frames per seed) |
| `observations_viz.py` | Visualize viewport coverage |
| `qualitative_analysis.py` | Side-by-side CNN prediction vs ground truth heatmaps |
| `analyze_dynamics.py` | Empirical analysis of grid change rates across simulation strides |

---

## Setup

### Prerequisites

- Python 3.8+
- PyTorch (CPU or CUDA)
- `requests`, `numpy`
- An account on [app.ainm.no](https://app.ainm.no)

### Install Dependencies

```bash
pip install requests numpy torch
```

### Configure API Token

1. Log in at [app.ainm.no](https://app.ainm.no)
2. Browser DevTools → Application → Cookies → copy `access_token`
3. Create a `.env` file:

```bash
cp .env.example .env
```

4. Paste your token:

```
ASTAR_TOKEN=eyJhbGciOi...your_jwt_token
```

### Collect Training Data

Before training, you need ground truth and replay data from completed rounds:

```bash
# Download ground truth for all completed rounds
python fetch_ground_truth.py

# Download simulation replays (needed for multi-replay observation synthesis)
python get_replays.py

# Or fetch a specific round:
python fetch_ground_truth.py 10
python get_replays.py --round 10
```

This populates `data/ground_truth/` and `replays/`.

### Train a Model

```bash
# Train the production model (unet_cond, all data, no holdout)
python train_cnn.py --model unet_cond

# Evaluate generalization (K-fold across rounds)
python train_cnn.py --model unet_cond --cv round_kfold
```

Checkpoints are saved to `checkpoints_unet_cond/` every 25 epochs.

### Run the Automated Pipeline

```bash
# Run continuously (checks every 20 minutes)
python cron.py

# Single cycle (for testing)
python cron.py --once
```

---

## Usage

### Automated Pipeline (Production)

The recommended way to participate in live rounds:

```bash
python cron.py          # Run forever — checks every 20 min
python cron.py --once   # Single cycle — process one round and exit
```

What happens each cycle:
1. Checks for an active round via the API
2. Submits a prior-based fallback (guarantees a score even if the model fails)
3. Spends the query budget collecting viewport observations
4. Loads the latest `unet_cond` checkpoint, runs inference, and submits predictions per seed
5. After the round closes, waits for ground truth, then retrains (+2,000 epochs from last checkpoint)

State is tracked in `cron_state.json` to avoid reprocessing the same round.

### Manual Training

```bash
# Production model — train on all data (no holdout)
python train_cnn.py --model unet_cond

# Evaluate generalization — K-fold cross-validation across rounds
python train_cnn.py --model unet_cond --cv round_kfold

# Train from scratch (ignore existing checkpoints)
python train_cnn.py --model unet_cond --reset

# Train indefinitely until Ctrl+C
python train_cnn.py --model unet_cond --forever

# Fetch fresh ground truth from API before training
python train_cnn.py --model unet_cond --fetch

# Override epoch count
python train_cnn.py --model unet_cond --epochs 2000

# Simple CNN baseline (terrain-only, quadrant cross-validation)
python train_cnn.py --model quick --cv quadrant
```

### Evaluation

```bash
# Evaluate unet_cond on all ground truth data
python eval_cnn.py --arch unet_cond

# Include viewport-restricted metrics (KL on observed vs unobserved pixels)
python eval_cnn.py --arch unet_cond --viewports

# Evaluate baseline
python eval_cnn.py --arch quick

# Compare all architectures side-by-side
python compare_models.py --eval-only
python compare_models.py --eval-only --models unet_cond quick

# Diagnose a specific round (compare submitted vs re-inference vs baselines)
python eval_round.py 16
python eval_round.py 16 --arch unet_cond --detailed
```

### Data Collection

```bash
# Download ground truth for completed rounds
python fetch_ground_truth.py              # all rounds
python fetch_ground_truth.py 10           # round 10 only

# Download simulation replays
python get_replays.py                     # all completed rounds
python get_replays.py --round 5           # round 5 only
```

### Analysis & Visualization

```bash
# Visualize viewport coverage patterns
python observations_viz.py

# Side-by-side CNN predictions vs ground truth heatmaps
python qualitative_analysis.py --round 1 --seed 0
python qualitative_analysis.py --all

# Empirical dynamics analysis (change rates, settlement lifecycles)
python analyze_dynamics.py
```

---

## Production Pipeline

### What is unet_cond?

`unet_cond` (short for "**U-Net conditioned on observations**") is our production model architecture. It is a compact MiniUNet (~472K parameters) that takes **both** terrain features **and** live viewport observations as input — hence "conditioned."

The key insight: instead of treating terrain prediction and viewport observations as separate problems, `unet_cond` learns the joint mapping `P(full map distribution | terrain layout, partial stochastic observations)`. This means:

- **Observed pixels** get highly accurate predictions because the model sees empirical class frequencies from viewport queries
- **Unobserved pixels** benefit from spatial context propagated through the U-Net's encoder-decoder skip connections — the model learns to extrapolate from nearby observed areas
- **When no observations are available** (coverage channel = 0 everywhere), the model falls back to terrain-only prediction — trained explicitly via observation dropout during training

This contrasts with the `quick` baseline (a terrain-only CNN with 14 input channels) which cannot leverage viewport data at all.

### Architecture — unet_cond

**Input:** 21 channels = 14 terrain features + 7 observation channels
- Channels 0–7: one-hot terrain encoding (8 terrain types)
- Channels 8–13: neighbour class frequency counts (local spatial context)
- Channels 14–19: observed class frequency per pixel (from viewport queries)
- Channel 20: coverage indicator — `log(1 + observation_count)`

**Output:** 6-class probability distribution per pixel (softmax, floored at 0.01)

**Architecture:** MiniUNet (~472K params) with 2 encoder levels + bottleneck + 2 decoder levels:

| Layer | Channels | Spatial Size |
|-------|----------|--------------|
| Input | 21 | 40×40 |
| Encoder 1 | 32 | 40×40 → pool → 20×20 |
| Encoder 2 | 64 | 20×20 → pool → 10×10 |
| Bottleneck | 128 | 10×10 |
| Decoder 2 (+ skip from enc2) | 64 | 10×10 → up → 20×20 |
| Decoder 1 (+ skip from enc1) | 32 | 20×20 → up → 40×40 |
| Output (1×1 conv) | 6 | 40×40 |

All convolutions are 3×3 with `padding=1`. Dropout2d(0.1) at every block. Transpose convolutions for upsampling. Skip connections concatenate encoder features to decoder features.

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
- **Ground truth:** samples in `data/ground_truth/` (completed rounds, 5 seeds each). Each is a 40×40×6 probability tensor from Monte Carlo simulation (200 runs).
- **Simulation replays:** files in `replays/` (completed rounds, 5 seeds each). Each contains 51 frames (years 0–50) of the full 40×40 grid.

**Multi-replay observation generation** (`sample_multi_replay_obs_channels`):

In production, each viewport query triggers an independent stochastic simulation — so different viewports observe different random outcomes even on the same seed. The training pipeline replicates this:

1. Loads all available replay grids (final frames) for the round
2. Places viewports using the same systematic 3×3 tile grid as production
3. For each tile, randomly selects one of the available replay grids — so overlapping tiles may show different stochastic outcomes, just like production
4. Encodes the result as 7 observation channels (6 class frequencies + coverage)

For each ground truth sample:
- **3 copies** with different random multi-replay observation patterns
- **1 copy** with zero observations (teaches terrain-only fallback)

This yields ~4× training maps from the available ground truth samples (no rotation/flip augmentation — observation spatial patterns must match production).

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
└─ Retrain (+2,000 epochs from latest checkpoint)
```

Configuration (in `cron.py`):

| Setting | Value | Description |
|---------|-------|-------------|
| `ARCH` | `unet_cond` | Model architecture |
| `POLL_INTERVAL_S` | 1200 (20 min) | Check interval |
| `GT_WAIT_S` | 600 (10 min) | Wait for GT before retrain |
| `ADDITIONAL_EPOCHS` | 2000 | Additional epochs per training cycle |

Retraining resumes from the latest checkpoint (optimizer state, LR schedule preserved) and trains 2,000 additional epochs using `--cv all` (all data, no holdout). The model continuously improves as new data arrives.

```bash
python cron.py          # run forever
python cron.py --once   # single cycle
```

---

## Key Findings

From analysis across rounds of observation data:

1. **Simulations are stochastic:** The vast majority of overlapping viewport pairs show cell-level differences. ~29% of overlapping cells have different values across queries.

2. **Replays ≠ observations:** Static cells (ocean/mountain) match 100% between replay final frames and observation viewports. Dynamic cells match only ~72% — confirming independent stochastic runs.

3. **Variation rate varies by round:** Some maps have low dynamism (~5% variation rate) while others are highly dynamic (~53%). The model must handle both extremes.

---

## Project Structure

```
Astar-Island-Oracle/
├── cron.py                  # Production pipeline (automated 20-min cycle)
├── astar_cnn.py             # Live inference + viewport strategy + submission
├── train_cnn.py             # Offline training (ground truth + replays)
├── eval_cnn.py              # Checkpoint evaluation against competition metric
├── eval_round.py            # Round-specific score diagnosis
├── compare_models.py        # Architecture comparison
├── fetch_ground_truth.py    # Download GT from API for completed rounds
├── get_replays.py           # Download simulation replay trajectories
├── observations_viz.py      # Viewport coverage visualization
├── qualitative_analysis.py  # Side-by-side prediction vs GT heatmaps
├── analyze_dynamics.py      # Empirical dynamics analysis
├── CHALLENGE.md             # Full challenge specification
├── cron_state.json          # Tracks which rounds have been processed
├── .env                     # API token configuration
├── data/
│   ├── ground_truth/        # GT files: r{N}_s{M}_{id}.json (40×40×6 probabilities)
│   ├── observations_*.json  # Saved viewport queries per round
│   └── round_*.json         # Cached round metadata
├── replays/                 # Replay files: r{N}_s{M}_*.json (51 frames each)
├── checkpoints_unet_cond/   # unet_cond checkpoints (production)
│   ├── cnn_epoch_XXXX.pt    # Checkpoint every 25 epochs
│   ├── cnn_latest.pt        # Latest checkpoint (used by cron.py)
│   └── training_history.json
└── checkpoints/             # QuickCNN checkpoints (baseline)
```

---

## Model Architectures

| Architecture | Input Channels | Params | Description | CV Default | Loss |
|---|---|---|---|---|---|
| `unet_cond` | 21 (14 terrain + 7 obs) | ~472K | **Production.** MiniUNet conditioned on stochastic viewport observations. Multi-replay obs synthesis, entropy-weighted KL, observation dropout | `all` | entropy-weighted KL |
| `quick` | 14 | ~14K | QuickCNN (2-layer, terrain-only baseline) | `quadrant` | KL divergence |

---

## Solution Architecture — Deep Dive

This section provides a comprehensive explanation of the neural network design, data pipeline, training procedure, evaluation methodology, post-processing steps, and production submission flow.

### Network Architecture: MiniUNet

The production model (`unet_cond`) is a compact U-Net with **~472K parameters**, designed for 40×40 spatial maps. It follows a classic encoder–bottleneck–decoder structure with skip connections.

```
Input (21, 40, 40)
    │
    ▼
┌──────────────────────────┐
│  Encoder Block 1         │
│  Conv2d(21→32, 3×3) ReLU │──────────────────────┐ skip
│  Dropout2d(0.1)          │                       │
│  Conv2d(32→32, 3×3) ReLU │                       │
│  MaxPool2d(2)  → (32, 20, 20)                    │
└──────────────────────────┘                       │
    │                                              │
    ▼                                              │
┌──────────────────────────┐                       │
│  Encoder Block 2         │                       │
│  Conv2d(32→64, 3×3) ReLU │──────────┐ skip       │
│  Dropout2d(0.1)          │          │            │
│  Conv2d(64→64, 3×3) ReLU │          │            │
│  MaxPool2d(2)  → (64, 10, 10)       │            │
└──────────────────────────┘          │            │
    │                                 │            │
    ▼                                 │            │
┌──────────────────────────┐          │            │
│  Bottleneck              │          │            │
│  Conv2d(64→128, 3×3) ReLU│          │            │
│  Dropout2d(0.1)          │          │            │
│  Conv2d(128→128, 3×3) ReLU          │            │
│   → (128, 10, 10)        │          │            │
└──────────────────────────┘          │            │
    │                                 │            │
    ▼                                 │            │
┌──────────────────────────┐          │            │
│  Decoder Block 2         │          │            │
│  ConvT2d(128→64, 2×2, s2)│          │            │
│  Concat with skip2   ────┼──────────┘            │
│  Conv2d(128→64, 3×3) ReLU│   (cat: 64+64=128)   │
│  Dropout2d(0.1)          │                       │
│  Conv2d(64→64, 3×3) ReLU │                       │
│   → (64, 20, 20)         │                       │
└──────────────────────────┘                       │
    │                                              │
    ▼                                              │
┌──────────────────────────┐                       │
│  Decoder Block 1         │                       │
│  ConvT2d(64→32, 2×2, s2) │                       │
│  Concat with skip1   ────┼───────────────────────┘
│  Conv2d(64→32, 3×3) ReLU │   (cat: 32+32=64)
│  Dropout2d(0.1)          │
│  Conv2d(32→32, 3×3) ReLU │
│   → (32, 40, 40)         │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Output Head             │
│  Conv2d(32→6, 1×1)       │
│  Softmax(dim=1)          │
│  Clamp(min=0.01)         │
│  Renormalize             │
│   → (6, 40, 40)          │
└──────────────────────────┘
```

**Key design choices:**
- **2 encoder levels** (40→20→10) suffice for a 40×40 map — deeper would over-compress
- **Skip connections** preserve spatial detail from encoder to decoder, critical for pixel-level prediction
- **Dropout2d(0.1)** at every block for regularization (small dataset)
- **Probability floor** (`PROB_FLOOR = 0.01`) is enforced inside the model's `forward()` pass — the output is always a proper probability distribution with no zeros
- **All 3×3 convolutions** use `padding=1` to preserve spatial dimensions; reflective padding handles odd dimensions

### Input Encoding: 21 Channels

The model receives a **21-channel** tensor constructed from two sources:

#### Terrain Features (14 channels) — `encode_initial_grid()`

| Channels | Content | Description |
|----------|---------|-------------|
| 0–7 | One-hot terrain code | 8 terrain types: Empty(0), Settlement(1), Port(2), Ruin(3), Forest(4), Mountain(5), Ocean(10), Plains(11) |
| 8–13 | Neighbour class frequency | For each pixel, the fraction of its 8-neighbours belonging to each of the 6 output classes. Provides local spatial context |

Each pixel's terrain type activates exactly one of channels 0–7. Channels 8–13 encode the neighbourhood composition — e.g., a plains cell surrounded by forest will have channel 12 (forest class) close to 1.0.

#### Observation Features (7 channels) — `encode_obs_channels()`

| Channels | Content | Description |
|----------|---------|-------------|
| 14–19 | Observed class frequency | For each pixel, how often each class was observed across all viewport queries. Normalized to frequencies |
| 20 | Coverage indicator | $\log(1 + n_\text{observations})$ — tells the model how many times a pixel was observed (0 = never) |

Multiple viewport queries of the same pixel produce genuine frequency distributions because each query runs an independent stochastic simulation. A pixel observed 9 times (from overlapping tiles across 5 seeds) might show `[0.55, 0.22, 0.11, 0.00, 0.11, 0.00]` — this captures the actual variability in simulation outcomes.

**For unobserved pixels**, all 7 observation channels are zero. The model learns to fall back to terrain-only prediction when the coverage channel is 0.

### Data Sources

#### Training Data

1. **Ground truth** (`data/ground_truth/r{N}_s{M}_{id}.json`): 40×40×6 probability tensors from Monte Carlo simulation (200 runs). Available for past rounds after scoring completes. Each entry contains the `initial_grid`, map dimensions, and the full probability distribution over 6 classes per pixel.

2. **Simulation replays** (`replays/r{N}_s{M}_*.json`): Full 51-frame trajectories (years 0–50) for each seed. Downloaded via `get_replays.py`. Used during training to synthesize realistic observation patterns via `sample_multi_replay_obs_channels()`.

#### Evaluation Data

- Same ground truth files, evaluated with `eval_cnn.py` against the competition metric (entropy-weighted KL divergence)
- Observations from `data/observations_{round_id}.json` are used to construct observation channels for evaluation

#### Production Data (Live Rounds)

- `initial_grid` from the API for the current round
- Viewport observations collected during the 20-minute window via `collect_observations()`

### Training Procedure

#### Multi-Replay Observation Synthesis

The critical training challenge: in production, each viewport query runs an independent simulation, so different viewports observe different stochastic outcomes. Training must replicate this.

`sample_multi_replay_obs_channels()` simulates production-like observation patterns:

1. Places viewports on a systematic 3×3 tile grid (same as production)
2. **For each viewport, randomly selects one of the available replay grids** — so overlapping tiles may show different outcomes, just like production
3. Encodes the result as 7 observation channels

For each ground truth sample, training generates:
- **3 copies** with different random multi-replay observation patterns
- **1 copy** with zero observations (teaches the terrain-only fallback)

This produces ~4× training maps from the available ground truth samples.

#### Loss Function

**Entropy-weighted KL divergence**, matching the competition scoring metric exactly:

$$\mathcal{L} = \frac{\sum_{i} \text{KL}(q_i \| p_i) \cdot H(q_i)}{\sum_{i} H(q_i)}$$

where $q_i$ is the ground truth distribution at pixel $i$, $p_i$ is the prediction, and $H(q_i) = -\sum_c q_i(c) \log q_i(c)$ is the ground truth entropy.

**Effect:** Static cells (ocean, mountain) have $H = 0$ and contribute nothing to the loss. The model focuses its capacity on dynamic cells where the simulation outcome is uncertain. This directly matches competition scoring: $\text{score} = 100 \cdot e^{-3 \cdot \mathcal{L}}$.

#### Optimizer & Schedule

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | $10^{-3}$ (default, overridable) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=100 epochs, min=1e-6) |
| Early stopping | 300 epochs without improvement (when validation set exists) |
| Observation dropout | 65% full obs / 25% partial / 10% zero |

**Observation dropout** randomly masks observation channels during training to build robustness: the model must still produce reasonable predictions with incomplete or absent viewport data.

### Evaluation Pipeline — `eval_cnn.py`

When you run `python eval_cnn.py --arch unet_cond --viewports`, the following happens for each ground truth sample:

```
1. Load checkpoint from checkpoints_unet_cond/
2. encode_initial_grid(initial_grid)           → (14, H, W) terrain features
3. encode_obs_channels(seed_observations)       → (7, H, W)  obs features
4. Concatenate → (21, H, W) input tensor
5. Model forward pass                          → (6, H, W)  raw probabilities
6. Floor at PROB_FLOOR (0.01), renormalize     → cnn_pred (H, W, 6)
7. Bayesian blend with observations            → blend_pred (H, W, 6)
8. Mountain override (initial_grid == 5)       → final prediction
9. competition_score(prediction, ground_truth) → score
```

### Post-Processing Steps

These are applied **after** the model's forward pass, both in evaluation and in production submission:

#### 1. Bayesian Blend — `bayesian_blend()`

A post-hoc correction for observed pixels where the model remains uncertain despite having seen observations in its input channels.

**When it activates:** Only on pixels that are (a) observed by at least one viewport AND (b) where the CNN output entropy exceeds $0.3 \times \ln(6) \approx 0.54$ nats.

**How it works:**

$$\text{posterior}(c) \propto \underbrace{5.0 \times p_\text{CNN}(c)}_{\text{CNN prior (strength=5)}} + \underbrace{n_\text{obs}(c)}_{\text{empirical count}}$$

where $n_\text{obs}(c)$ is the number of times class $c$ was observed at that pixel across all viewport queries.

**Why it helps:** The model receives observations as input channels but may not fully exploit them for every pixel. The Bayesian blend acts as a cheap insurance layer — if the model is already confident (low entropy), the blend is skipped entirely. If the model is uncertain, empirical observation counts nudge the prediction toward what was actually observed.

**Effect on unobserved pixels:** None — predictions are returned unchanged.

#### 2. Mountain Override

Mountains are **static terrain** — they never change during the simulation. The ground truth for mountain pixels is always $[0, 0, 0, 0, 0, 1.0]$. After all other processing, mountain pixels are hard-set:

```python
prediction[y, x, :] = PROB_FLOOR           # 0.01 for all classes
prediction[y, x, 5] = 1.0 - (PROB_FLOOR * 5)  # 0.95 for mountain
```

This eliminates any residual model error on these deterministic pixels.

### Production Submission Pipeline — `cron.py` → `astar_cnn.py`

The automated pipeline runs every 20 minutes. For the prediction submission phase:

```
1. Load latest checkpoint from checkpoints_unet_cond/
2. For each seed (0–4):
   a. encode_initial_grid(initial_grid)              → (14, H, W)
   b. encode_obs_channels(seed_observations)          → (7, H, W)
   c. predict_full_map(model, features, obs_features) → (H, W, 6)
      ├── Concatenate features → (21, H, W)
      ├── Model forward pass → softmax → floor → normalize
      └── Return (H, W, 6) probabilities
   d. bayesian_blend(prediction, seed_obs, ...)       → (H, W, 6)
   e. Mountain override (initial_grid == 5)           → (H, W, 6)
   f. submit_prediction(round_id, seed_idx, prediction) → API call
```

**This is identical to the eval_cnn.py pipeline** — the same model, same encoding, same post-processing steps in the same order. Scores from `eval_cnn.py` are representative of what gets submitted in production.

