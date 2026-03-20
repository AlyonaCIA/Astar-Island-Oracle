# Astar Island Oracle

Our toolkit for the **Astar Island** ML challenge — a Norse civilisation simulator where you observe a black-box world through limited viewports and predict the final terrain state. See [CHALLENGE.md](CHALLENGE.md) for full challenge details.

---

## Table of Contents

- [The Challenge in Brief](#the-challenge-in-brief)
- [What This Codebase Does](#what-this-codebase-does)
- [Setup](#setup)
- [Production Pipeline](#production-pipeline)
  - [Architecture — unet\_sim](#architecture--unet_sim)
  - [Viewport Strategy — Coverage First](#viewport-strategy--coverage-first)
  - [Observation Encoding](#observation-encoding)
  - [Training — Offline with Replay Data](#training--offline-with-replay-data)
  - [Cross-Validation — Leave-One-Round-Out](#cross-validation--leave-one-round-out)
  - [Observation Dropout](#observation-dropout)
  - [Automated Pipeline — cron.py](#automated-pipeline--cronpy)
- [Manual Training & Evaluation](#manual-training--evaluation)
- [Data Analysis](#data-analysis)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [All Model Architectures](#all-model-architectures)

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
| `compare_models.py` | Train & compare all architectures side-by-side |
| `data_check.py` | Data analysis: verify stochastic behavior, compare replays vs observations |
| `analyze.py` | Plot training curves from `training_history.json` |
| `astar_uniform.py` | Baseline: uniform 1/6 probability everywhere (0 queries) |
| `astar_baseline.py` | Baseline: hand-tuned terrain priors + observation blending |

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

The production system uses **unet_sim** — a MiniUNet trained on simulation replay data with observation-conditioned input channels.

### Architecture — unet_sim

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

### Training — Offline with Replay Data

**Data sources:**
- **Ground truth:** 35 samples in `data/ground_truth/` (rounds 1–7, 5 seeds each). Each is a 40×40×6 probability tensor from Monte Carlo simulation.
- **Simulation replays:** 40 files in `simulation_replays/` (rounds 1–8, 5 seeds each). Each contains 51 frames (years 0–50) of the full 40×40 grid.

**Synthetic observation generation** (`sample_synthetic_obs_channels`):

For each ground truth sample with a matching replay, the training pipeline:
1. Takes the replay's final frame (year 50) as a plausible simulation outcome
2. Samples viewport observations from this frame, encoding them as 7 observation channels
3. Creates 3 copies per map with different viewport patterns:
   - **60% chance:** systematic tile grid (matches production layout)
   - **40% chance:** random 6–12 viewports (adds training diversity)
4. Creates 1 additional copy with **zero observations** (teaches terrain-only fallback)

After augmentation (4 rotations × 2 flips = 8×), this yields ~1,120 training maps from 35 ground truth samples.

**Known limitation:** Each training sample's observations come from a single replay realization. In production, observations aggregate multiple independent stochastic simulations, so the empirical frequencies are richer. The model compensates by training on varied viewport patterns and using observation dropout.

### Cross-Validation — Leave-One-Round-Out

**Production default: `--cv round`** (leave-one-round-out)

Holds out the most recent round as validation; trains on all other rounds. This directly tests the production scenario: **can the model predict a never-before-seen map?**

```bash
python train_cnn.py --model unet_sim --cv round
```

**Why not quadrant CV?** The 4-fold quadrant approach (hold out one spatial quadrant per fold) tests whether the model can interpolate within a known map. This is NOT what happens in production — the model faces a brand-new map with different terrain. Additionally, quadrant CV combined with rotation augmentation leaks validation data (a rotated map's held-out quadrant contains pixels from the original map's training region).

**Quadrant CV is still available** for architectures without observation channels:
```bash
python train_cnn.py --model unet --cv quadrant
```

### Observation Dropout

During training, observation channels are randomly masked to build robustness:

| Action | Probability | Purpose |
|--------|-------------|---------|
| Keep all observations | 65% | Matches production (100% coverage) |
| Random partial masking (3–9 viewports of 15×15) | 25% | Handles incomplete coverage gracefully |
| Zero all observations | 10% | Safety net for terrain-only fallback |

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
| `ARCH` | `unet_sim` | Model architecture |
| `POLL_INTERVAL_S` | 1200 (20 min) | Check interval |
| `GT_WAIT_S` | 600 (10 min) | Wait for GT before retrain |
| `MAX_TRAIN_EPOCHS` | 4000 | Max epochs per training cycle |

Retraining uses `--cv round` and snapshot ensemble (last 3 checkpoints).

```bash
python cron.py          # run forever
python cron.py --once   # single cycle
```

---

## Manual Training & Evaluation

### Train

```bash
python train_cnn.py --model unet_sim                     # default: --cv round
python train_cnn.py --model unet_sim --reset              # train from scratch
python train_cnn.py --model unet_sim --forever            # train until Ctrl+C
python train_cnn.py --model unet_sim --fetch              # fetch GT from API first
python train_cnn.py --model unet_sim --epochs 2000        # override epoch count
python train_cnn.py --model unet --cv quadrant            # older arch with quadrant CV
```

### Evaluate

```bash
python eval_cnn.py --arch unet_sim                        # latest checkpoint
python eval_cnn.py --arch unet_sim --viewports            # viewport-restricted eval
```

### Compare architectures

```bash
python compare_models.py --eval-only --models unet unet_sim
python compare_models.py --models unet_sim --epochs 500
```

---

## Data Analysis

```bash
python data_check.py         # verify stochastic behavior + replay vs observation comparison
python analyze.py            # plot training curves
python observations_viz.py   # visualize viewport coverage
```

---

## Key Findings

From `data_check.py` analysis across 7 rounds of observation data:

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
├── data_check.py            # Data analysis (stochastic verification)
├── analyze.py               # Training curve plots
├── observations_viz.py      # Viewport coverage visualization
├── astar_uniform.py         # Baseline: uniform 1/6
├── astar_baseline.py        # Baseline: terrain priors + blending
├── CHALLENGE.md             # Full challenge specification
├── data/
│   ├── ground_truth/        # GT files: r{N}_s{M}_{id}.json (40×40×6 probabilities)
│   ├── observations_*.json  # Saved viewport queries per round
│   └── round_*.json         # Cached round metadata
├── simulation_replays/      # Replay files: r{N}s{M}.json (51 frames each)
├── checkpoints_unet_sim/    # unet_sim checkpoints (production)
├── checkpoints_unet/        # MiniUNet checkpoints
├── checkpoints_unet_aug/    # Augmented MiniUNet checkpoints
├── checkpoints/             # QuickCNN checkpoints
└── checkpoints_quick3/      # QuickCNN3 checkpoints
```

---

## All Model Architectures

| Architecture | Input Channels | Params | Description | CV Default |
|---|---|---|---|---|
| `unet_sim` | 21 (14 terrain + 7 obs) | ~472K | **Production.** MiniUNet with synthetic replay observations | `round` |
| `unet_obs` | 21 (14 terrain + 7 obs) | ~472K | MiniUNet with real observation channels | `round` |
| `unet_v2` | 28 (14 + 7 obs + 7 cross-seed) | ~490K | MiniUNetV2 with cross-seed channels | `round` |
| `unet_aug` | 14 | ~470K | MiniUNet with augmentation + dropout | `quadrant` |
| `unet` | 14 | ~470K | MiniUNet (terrain only) | `quadrant` |
| `quick` | 14 | ~35K | QuickCNN (3-layer, no spatial context) | `quadrant` |
| `quick3` | 14 | ~35K | QuickCNN3 (3-layer variant) | `quadrant` |
