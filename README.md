# Astar Island Oracle

Our toolkit for the **Astar Island** ML challenge — a Norse civilisation simulator where you observe a black-box world through limited viewports and predict the final terrain state. See [CHALLENGE.md](CHALLENGE.md) for full challenge details.

---

## Table of Contents

- [The Challenge in Brief](#the-challenge-in-brief)
- [What This Codebase Does](#what-this-codebase-does)
- [Setup](#setup)
- [Methods](#methods)
  - [Method 1 — Uniform Fallback](#method-1-uniform-fallback-astar_uniformpy)
  - [Method 2 — Prior + Observation Blending](#method-2-prior-observation-blending-astar_baselinepy)
  - [Method 3 — CNN with Fallback](#method-3-cnn-with-fallback-astar_cnnpy)
  - [Method 4 — Offline CNN Training](#method-4-offline-cnn-training-train_cnnpy-eval_cnnpy)
- [Viewport Sampling & Query Strategy](#viewport-sampling-query-strategy)
  - [Constraints](#constraints)
  - [Phase 1 — Tile Grid Generation](#phase-1-tile-grid-generation-compute_tile_grid)
  - [Phase 2 — Tile Priority Scoring](#phase-2-tile-priority-scoring-score_tile)
  - [Phase 3 — Round-Robin Coverage with Dynamic Re-Ranking](#phase-3-round-robin-coverage-with-dynamic-re-ranking)
  - [Phase 4 — Extra Queries on Dynamic & High-Interest Areas](#phase-4-extra-queries-on-dynamic-high-interest-areas)
  - [Worked Example — 40×40 Map, 50 Budget, 5 Seeds](#worked-example-4040-map-50-budget-5-seeds)
  - [Visualising Coverage](#visualising-coverage)
- [Recommended Workflow](#recommended-workflow)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Improving Further](#improving-further)

---

## The Challenge in Brief

A 40×40 procedurally generated map runs a Norse simulation for 50 years: settlements grow, factions clash, trade routes form, winters destroy. Each round gives you **5 seeds** (same map, different random outcomes) and **50 viewport queries** (max 15×15 each, shared across seeds). You submit an **H×W×6 probability tensor** per seed predicting 6 terrain classes per cell. Scored by entropy-weighted KL divergence against Monte Carlo ground truth — only dynamic cells matter.

## What This Codebase Does

This repo contains a progression of approaches, from the simplest possible submission to offline-trained CNN models:

| File | Purpose | When to use |
|------|---------|-------------|
| `astar_uniform.py` | Submits uniform 1/6 probability everywhere. Zero queries used. | Get on the leaderboard instantly, verify auth works |
| `astar_baseline.py` | Hand-tuned terrain priors + 50-query observation blending | Quick reasonable baseline, collects observation data |
| `astar_cnn.py` | Live CNN pipeline: submits fallback immediately, then trains CNN on observations and resubmits | Active round submissions — guarantees a score even if CNN fails |
| `train_cnn.py` | Offline CNN training on ground truth from completed rounds | Between rounds — improve model with no time pressure |
| `eval_cnn.py` | Evaluates checkpoints against competition metric (entropy-weighted KL) | Check model quality, compare CNN vs prior vs uniform |
| `compare_models.py` | Trains and evaluates all 3 CNN architectures side-by-side | Architecture selection — find the best model |
| `analyze.py` | Plots training curves from `training_history.json` | Diagnose training progress, spot overfitting |

**Data flow:** Live scripts (`astar_baseline.py`, `astar_cnn.py`) save observations to `data/`. After rounds complete, `train_cnn.py --fetch` downloads ground truth for offline training. Checkpoints are saved per architecture (`checkpoints/`, `checkpoints_quick3/`, `checkpoints_unet/`). The live script `astar_cnn.py` can load pre-trained checkpoints to get a head-start during active rounds.

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

## Viewport Sampling & Query Strategy

The query strategy is the most critical part of the pipeline. With only **50 queries** shared across 5 seeds on a 1600-cell map, every viewport must extract maximum information. The strategy is implemented in `astar_cnn.py` and runs in four phases.

### Constraints

| Constraint | Value |
|---|---|
| Map size | 40×40 = 1,600 cells |
| Seeds per round | 5 (same map, different stochastic outcomes) |
| Total query budget | 50 (shared across all seeds) |
| Viewport size | 5×5 to 15×15 (always request max 15×15) |
| Cost per query | 1, regardless of viewport size |
| Simulation type | Stochastic — same viewport gives different results each time |

**Key insight:** Since every query costs 1 budget regardless of viewport size, we always request the full 15×15 viewport to maximise information per query.

### Phase 1 — Tile Grid Generation (`compute_tile_grid`)

Generates a grid of **full-size 15×15 viewports** that covers the entire 40×40 map. Instead of shrinking edge tiles (which wastes viewport capacity), edge tiles are **shifted inward** to stay within bounds, creating natural overlap.

**Algorithm:**
1. Calculate how many tiles are needed per axis: `ceil(40 / 15) = 3`
2. Distribute tile start positions evenly across `[0, max_start]` where `max_start = 40 - 15 = 25`
3. Result: positions at `x = [0, 12, 25]` and `y = [0, 12, 25]`

**40×40 tile layout (9 tiles, all 15×15):**

```
      x=0       x=12      x=25
      ├──15──┤  ├──15──┤  ├──15──┤
y=0   [A         ][B         ][C         ]     row 0: y covers [0..14]
         ┌─overlap─┐  ┌─overlap─┐
y=12  [D         ][E         ][F         ]     row 1: y covers [12..26]
         ┌─overlap─┐  ┌─overlap─┐
y=25  [G         ][H         ][I         ]     row 2: y covers [25..39]
```

- **100% map coverage** — every pixel is observed at least once
- **All tiles are 15×15** — no wasted viewport capacity
- **375 cells** (23%) are covered by 2+ tiles — providing natural overlap for better stochastic averaging
- **0 pixels outside the map** — no information wasted
- **2,025 total pixels** queried per full coverage cycle vs 1,600 with the old non-overlapping scheme (+26.6% more data)

### Phase 2 — Tile Priority Scoring (`score_tile`)

Each tile is scored based on the initial terrain it contains. Higher-scored tiles are queried first since they contain more dynamic content where observations matter most.

**Scoring weights per cell:**

| Terrain | Code | Points | Rationale |
|---|---|---|---|
| Port | 2 | 6.0 | Most dynamic — trade activity |
| Settlement | 1 | 5.0 | Dynamic — growth, conflict, death |
| Plains/Empty | 0, 11 | 0.5 | Expansion potential |
| Forest | 4 | 0.3 | Mostly static but supports settlements |
| Ocean | 10 | 0.0 | Completely static |
| Mountain | 5 | 0.0 | Completely static |

**Bonus modifiers:**
- **Coastal bonus (+0.3):** Land cells adjacent to ocean (potential port development)
- **Settlement cluster bonus (+N):** If a tile contains 2+ settlements/ports, add N bonus (more interaction = more dynamism)

**Example:** A tile containing 2 settlements, 1 port, and 50 plains cells near the coast would score: `2×5.0 + 1×6.0 + 50×0.5 + coastal_bonuses + cluster_bonus(3) ≈ 44+`

### Phase 3 — Round-Robin Coverage with Dynamic Re-Ranking

The main observation loop uses a **round-robin strategy** across seeds with **adaptive re-ranking** after each round.

**How it works:**

1. Each seed gets its own priority queue of 9 tiles, sorted by `score_tile`
2. In each "round", pop the highest-priority tile from each seed and query it
3. After each round (except the first), **re-rank** remaining tiles using `rescore_tile`

**Re-ranking (`rescore_tile`):**

After observing some tiles, unqueried tiles near high-activity areas get boosted. For each already-observed tile:
1. Compute `change_rate` = fraction of cells where observed class ≠ initial class
2. Compute `proximity` = linear falloff from 0 to 30 cells distance
3. Add bonus: `change_rate × proximity × 5.0`

This means if Tile A shows 40% of cells changed from their initial state, nearby unqueried Tile B gets a significant priority boost — the algorithm "follows the action."

**Round-robin sequence (first round, 5 seeds):**

```
Query 1:  Seed 0, highest-scored tile
Query 2:  Seed 1, highest-scored tile
Query 3:  Seed 2, highest-scored tile
Query 4:  Seed 3, highest-scored tile
Query 5:  Seed 4, highest-scored tile
--- re-rank remaining tiles using observed change rates ---
Query 6:  Seed 0, 2nd highest tile (re-ranked)
Query 7:  Seed 1, 2nd highest tile (re-ranked)
...
Query 45: Seed 4, 9th tile  (full coverage done)
```

**Full coverage cost:** 9 tiles × 5 seeds = **45 queries** → leaves **5 queries** for Phase 4.

### Phase 4 — Extra Queries on Dynamic & High-Interest Areas

After full coverage, remaining budget is spent on two interleaved sources:

#### Source A — Observed-Dynamic Re-Queries

Re-query tiles that showed the most changes from the initial state. These are tiles where the stochastic simulation produced different terrain than the initial grid, so additional observations improve our probability estimates.

Each previously-observed tile is scored by:
```
change_rate = (cells where observed_class ≠ initial_class) / total_cells
```

#### Source B — Interest-Centred Viewports (`compute_interest_viewports`)

Generate new viewport positions (not from the grid) that are **centred on the most interesting map regions**. These are always full 15×15 and always within bounds.

**Algorithm:**
1. Build a per-cell interest heatmap (same weights as `score_tile`: ports=6, settlements=5, etc.)
2. Compute a summed area table (SAT) for O(1) viewport scoring
3. Score every possible 15×15 viewport position (676 candidates on a 40×40 map)
4. Apply a mild diversity penalty for positions too close to already-queried tiles
5. Select top-N positions with minimum 3-cell separation between centres

**Interleaving:** Extra targets alternate between Source A (re-query dynamic) and Source B (new interest viewports), prioritised by score. This ensures the extra budget benefits both stochastic averaging and coverage of high-value regions.

### Worked Example — 40×40 Map, 50 Budget, 5 Seeds

```
Budget: 50 queries total

Phase 1: Generate 9 tiles (3×3 grid, all 15×15)
  Tiles at: (0,0) (12,0) (25,0) (0,12) (12,12) (25,12) (0,25) (12,25) (25,25)

Phase 2: Score and sort tiles per seed
  Seed 0 priority: tile(12,12)=42.3, tile(0,0)=31.1, tile(25,12)=28.7, ...
  Seed 1 priority: tile(12,12)=42.3, tile(0,12)=33.5, ...
  (same map, different priorities per seed due to initial state differences)

Phase 3: Round-robin coverage
  Round 1 (queries 1-5):   each seed queries its top tile
  Round 2 (queries 6-10):  re-rank, each seed queries next best
  ...
  Round 9 (queries 41-45): last tile per seed — full coverage!

Phase 4: Extra queries (5 remaining)
  Query 46: Re-query seed 2 tile(12,12) — 35% change rate
  Query 47: New interest viewport seed 0 at (8,10) — settlement cluster
  Query 48: Re-query seed 0 tile(0,12) — 28% change rate
  Query 49: New interest viewport seed 3 at (20,8) — port cluster
  Query 50: Re-query seed 4 tile(25,12) — 22% change rate

Result: 
  - 100% map coverage across all 5 seeds
  - 2,025 pixels observed per seed (all 15×15 tiles)
  - 375 cells with natural 2×+ overlap from tile grid
  - 5 extra observations on highest-value dynamic areas
```

### Visualising Coverage

Run `observations_viz.py` to see your query coverage after a live round:

```bash
python observations_viz.py                                    # auto-detect latest
python observations_viz.py data/observations_76909e29.json    # specific file
```

Produces a 5-panel figure (`observations_viz.png`) showing:
- Initial terrain as background
- Observation count heatmap (red = more observations)
- Viewport rectangles with query order labels
- Per-seed coverage statistics

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

- **Simulation modeling** — implement growth/conflict/trade/winter mechanics to forecast outcomes
- **Cross-seed learning** — all 5 seeds share hidden parameters; insights transfer
- **Attention / ensemble** — spatial attention modules, or ensemble predictions from multiple architectures
- **Accumulate training data** — every completed round adds more ground truth for offline training
- **Learned query strategy** — train a policy to select viewport positions based on observed data