"""
Astar Island — Monte Carlo v2: Variable-Horizon Transformer Dynamics

Two complementary models:

1. **VariableHorizonDynamics** — Predicts terrain state Δt steps into the
   future, trained on multi-stride (frame_t, Δt, frame_{t+Δt}) triples from
   all replays. Key innovations:
   - Horizon embedding: model knows how far ahead to predict
   - Identity skip connection: biases toward "no change" (correct for 95%+ cells)
   - Noise injection: robustness to own prediction errors during rollout
   - Stride-10 inference: 5 autoregressive steps instead of 50

2. **DirectPredictor** — Predicts 6-class probability distribution directly
   from initial state. Trained on ground truth data. No rollout needed.

Training data utilisation:
  - Multi-stride extraction: strides [1, 2, 5, 10] → ~9,300 transition pairs
    (vs 2,500 with stride-1 only)
  - Each stride teaches different temporal dynamics
  - All 50 replays × all valid frame pairs per stride

Usage:
  python monte_carlo_2.py train-dynamics [--epochs 300] [--reset]
  python monte_carlo_2.py train-direct [--epochs 500] [--reset]
  python monte_carlo_2.py evaluate [--rollouts 256] [--stride 10]
  python monte_carlo_2.py evaluate-direct [--epochs 500]
"""

import os
import sys
import json
import time
import math
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.join(SCRIPT_DIR, "simulation_replays")
GT_DIR = os.path.join(SCRIPT_DIR, "data", "ground_truth")
OBS_DIR = os.path.join(SCRIPT_DIR, "data")
CKPT_DIR_DYN = os.path.join(SCRIPT_DIR, "checkpoints_mc2_dyn")
CKPT_DIR_DIRECT = os.path.join(SCRIPT_DIR, "checkpoints_mc2_direct")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 terrain codes as they appear in replay grids
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CODE_TO_IDX = {code: i for i, code in enumerate(TERRAIN_CODES)}
IDX_TO_CODE = {i: code for code, i in CODE_TO_IDX.items()}
NUM_TERRAIN = 8

# 6-class submission mapping
SUBMIT_CLASSES = 6
TERRAIN_TO_SUBMIT = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
IDX8_TO_SUBMIT6 = [TERRAIN_TO_SUBMIT[IDX_TO_CODE[i]] for i in range(NUM_TERRAIN)]

PROB_FLOOR = 0.01
MAP_H, MAP_W = 40, 40
MAX_TIMESTEP = 50

# Multi-stride dynamics
TRAIN_STRIDES = [1, 2, 5, 10]     # horizons used during training
INFER_STRIDE = 10                   # stride for inference rollouts (5 steps for T=50)
MAX_HORIZON = max(TRAIN_STRIDES)

# Training defaults
DYN_EPOCHS = 300
DYN_LR = 3e-4
DYN_BATCH = 16
DIRECT_EPOCHS = 500
DIRECT_LR = 1e-3
DIRECT_BATCH = 8
CHECKPOINT_EVERY = 25

# MC inference
DEFAULT_K_ROLLOUTS = 256
DEFAULT_T_STEPS = 50

# Noise injection for dynamics training
NOISE_PROB = 0.3          # probability of injecting noise per sample
NOISE_MAGNITUDE = 0.15    # magnitude of Gaussian noise on one-hot input


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_replays():
    """Load all simulation replays."""
    replays = []
    if not os.path.isdir(REPLAY_DIR):
        print(f"ERROR: Replay directory not found: {REPLAY_DIR}")
        return replays
    for fname in sorted(os.listdir(REPLAY_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(REPLAY_DIR, fname)) as f:
            data = json.load(f)
        replays.append(data)
    print(f"Loaded {len(replays)} replays from {REPLAY_DIR}")
    return replays


def load_ground_truth():
    """Load all ground truth files."""
    if not os.path.isdir(GT_DIR):
        print(f"ERROR: Ground truth directory not found: {GT_DIR}")
        return []
    all_data = []
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        parts = fname.replace(".json", "").split("_")
        if len(parts) >= 3:
            data["_round_number"] = int(parts[0][1:])
            data["_seed_index"] = int(parts[1][1:])
            data["_round_id"] = parts[2]
        all_data.append(data)
    print(f"Loaded {len(all_data)} ground truth samples from {GT_DIR}")
    return all_data


def load_observations(round_id_short):
    """Load viewport observations for a round."""
    path = os.path.join(OBS_DIR, f"observations_{round_id_short}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("observations", [])
    return raw


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def grid_to_onehot(grid, height, width):
    """Convert 2D terrain grid to one-hot tensor. Returns (8, H, W) float32."""
    onehot = np.zeros((NUM_TERRAIN, height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = CODE_TO_IDX.get(grid[y][x], 0)
            onehot[idx, y, x] = 1.0
    return onehot


def grid_to_class_indices(grid, height, width):
    """Convert 2D terrain grid to int class index grid. Returns (H, W) int64."""
    indices = np.zeros((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            indices[y, x] = CODE_TO_IDX.get(grid[y][x], 0)
    return indices


# ---------------------------------------------------------------------------
# Positional / Temporal / Horizon Encodings
# ---------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for grid positions."""

    def __init__(self, d_model, max_h=50, max_w=50):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)

    def forward(self, h, w, device):
        """Returns (1, h*w, d_model) positional encoding."""
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        row_emb = self.row_embed(rows)
        col_emb = self.col_embed(cols)
        pos = torch.cat([
            row_emb.unsqueeze(1).expand(-1, w, -1),
            col_emb.unsqueeze(0).expand(h, -1, -1),
        ], dim=-1)
        return pos.reshape(1, h * w, -1)


class TimestepEmbedding(nn.Module):
    """Learnable timestep embedding (where in the simulation we are)."""

    def __init__(self, d_model, max_t=60):
        super().__init__()
        self.embed = nn.Embedding(max_t, d_model)

    def forward(self, t, n_tokens):
        emb = self.embed(t)  # (B, D)
        return emb.unsqueeze(1).expand(-1, n_tokens, -1)


class HorizonEmbedding(nn.Module):
    """
    Continuous horizon embedding: maps Δt → d_model vector.

    Uses log1p scaling since perceptual difference between Δt=1 and Δt=2
    is larger than between Δt=24 and Δt=25. MLP allows learning arbitrary
    horizon effects.
    """

    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, horizon, n_tokens):
        """
        Args:
            horizon: (B,) integer or float horizon values
            n_tokens: number of spatial tokens to broadcast to
        Returns: (B, n_tokens, d_model)
        """
        h = torch.log1p(horizon.float()).unsqueeze(1)  # (B, 1)
        emb = self.mlp(h)  # (B, d_model)
        return emb.unsqueeze(1).expand(-1, n_tokens, -1)


# ---------------------------------------------------------------------------
# Model 1: VariableHorizonDynamics
# ---------------------------------------------------------------------------

class VariableHorizonDynamics(nn.Module):
    """
    Variable-horizon dynamics model with CNN-Transformer architecture.

    Given terrain state at time t and a horizon Δt, predicts terrain at t+Δt.
    Trained on multiple strides simultaneously (1, 2, 5, 10 steps ahead).

    Key features:
    - Horizon embedding: knows how far ahead to predict
    - Identity skip connection: strong bias toward "no change" — correct for
      95-99% of cells. Output head only needs to predict the delta.
    - At inference, uses stride 10 → only 5 autoregressive steps for 50 years
      instead of 50 steps, massively reducing error accumulation.

    Input:  (B, 8, H, W) one-hot terrain at time t, timestep t, horizon Δt
    Output: (B, 8, H, W) logits for terrain at time t+Δt
    """

    def __init__(self, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1, skip_init=5.0):
        super().__init__()
        self.d_model = d_model

        # CNN stem: local feature extraction
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(NUM_TERRAIN, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Encodings
        self.pos_enc = PositionalEncoding2D(d_model)
        self.time_enc = TimestepEmbedding(d_model, max_t=MAX_TIMESTEP + 1)
        self.horizon_enc = HorizonEmbedding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head: predicts the DELTA from identity
        self.output_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, NUM_TERRAIN, kernel_size=1),
        )

        # Identity skip connection: biases output toward the input class
        # Learnable gain — initialized so that softmax strongly favors current class
        self.skip_gain = nn.Parameter(torch.tensor(skip_init))

    def forward(self, x, t=None, horizon=None):
        """
        Args:
            x: (B, 8, H, W) one-hot terrain state
            t: (B,) integer timestep [0..50]. Defaults to 0.
            horizon: (B,) integer/float horizon Δt. Defaults to 1.
        Returns: (B, 8, H, W) logits
        """
        B, C, H, W = x.shape

        # CNN stem
        feat = self.cnn_stem(x)

        # Flatten to token sequence
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, self.d_model)

        # Add positional encoding
        tokens = tokens + self.pos_enc(H, W, x.device)

        # Add timestep embedding
        if t is not None:
            tokens = tokens + self.time_enc(t, H * W)

        # Add horizon embedding
        if horizon is not None:
            tokens = tokens + self.horizon_enc(horizon, H * W)

        # Transformer encoder
        tokens = self.transformer(tokens)

        # Reshape back to grid
        grid_feat = tokens.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)

        # Output: delta logits + identity skip
        delta_logits = self.output_head(grid_feat)
        logits = delta_logits + self.skip_gain * x

        return logits

    def predict_probs(self, x, t=None, horizon=None):
        return F.softmax(self.forward(x, t, horizon), dim=1)


# ---------------------------------------------------------------------------
# Model 2: DirectPredictor — skip the rollout entirely
# ---------------------------------------------------------------------------

class DirectPredictor(nn.Module):
    """
    Predict the final 6-class probability distribution directly from the
    initial terrain state, without any step-by-step rollout.

    Trained on ground truth probability distributions from completed rounds.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=6,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Multi-scale CNN stem
        self.stem_local = nn.Sequential(
            nn.Conv2d(NUM_TERRAIN, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
        )
        self.stem_medium = nn.Sequential(
            nn.Conv2d(NUM_TERRAIN, d_model // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(d_model // 4),
            nn.GELU(),
        )
        self.stem_wide = nn.Sequential(
            nn.Conv2d(NUM_TERRAIN, d_model // 4, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(d_model // 4),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, SUBMIT_CLASSES, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        f_local = self.stem_local(x)
        f_medium = self.stem_medium(x)
        f_wide = self.stem_wide(x)
        feat = torch.cat([f_local, f_medium, f_wide], dim=1)

        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, self.d_model)
        tokens = tokens + self.pos_enc(H, W, x.device)
        tokens = self.transformer(tokens)

        grid_feat = tokens.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)
        logits = self.output_head(grid_feat)
        return logits

    def predict_probs(self, x):
        return F.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_grid(x, y_onehot=None):
    """Apply random rotation + horizontal flip to grid tensors."""
    k = random.randint(0, 3)
    flip = random.random() > 0.5

    x = torch.rot90(x, k, dims=[1, 2])
    if flip:
        x = torch.flip(x, dims=[2])

    if y_onehot is not None:
        y_onehot = torch.rot90(y_onehot, k, dims=[1, 2])
        if flip:
            y_onehot = torch.flip(y_onehot, dims=[2])
        return x, y_onehot

    return x


def augment_pair(x, y_indices):
    """Augment (input, class_index_target) pair consistently."""
    k = random.randint(0, 3)
    flip = random.random() > 0.5

    x = torch.rot90(x, k, dims=[1, 2])
    y_indices = torch.rot90(y_indices, k, dims=[0, 1])
    if flip:
        x = torch.flip(x, dims=[2])
        y_indices = torch.flip(y_indices, dims=[1])

    return x, y_indices


def add_input_noise(x, noise_prob=NOISE_PROB, noise_mag=NOISE_MAGNITUDE):
    """
    Inject noise into one-hot terrain inputs during training.

    Simulates the imperfect predictions the model will see at inference
    during autoregressive rollout. Makes the model robust to its own errors.

    Args:
        x: (B, 8, H, W) one-hot terrain (or close to it)
        noise_prob: probability of adding noise to each sample
        noise_mag: standard deviation of Gaussian noise
    Returns: (B, 8, H, W) possibly-noised input (still sums to ~1 per cell)
    """
    B = x.shape[0]
    mask = torch.rand(B, 1, 1, 1, device=x.device) < noise_prob
    noise = torch.randn_like(x) * noise_mag
    x_noisy = x + noise * mask.float()
    # Re-normalize to valid probability distribution per cell
    # Use temperature to keep it sharp (close to one-hot)
    x_noisy = F.softmax(x_noisy / 0.1, dim=1)
    return x_noisy


# ---------------------------------------------------------------------------
# Build training data — Multi-stride dynamics dataset
# ---------------------------------------------------------------------------

def build_multistride_dataset(replays, strides=None, held_out_rounds=None):
    """
    Extract (frame_t, timestep, horizon, frame_{t+horizon}) from replays
    at multiple strides.

    With strides [1, 2, 5, 10] and 50 replays, this yields ~9,300 pairs
    (vs 2,500 with stride-1 only). The model learns dynamics at multiple
    temporal scales simultaneously.

    Returns X, T_steps, Horizons, Y, meta.
    """
    if strides is None:
        strides = TRAIN_STRIDES

    X_list, T_list, H_list, Y_list, meta = [], [], [], [], []

    for replay in replays:
        rid = replay["round_id"][:8]
        if held_out_rounds and rid in held_out_rounds:
            continue

        height = replay["height"]
        width = replay["width"]
        frames = replay["frames"]
        n_frames = len(frames)

        for stride in strides:
            for i in range(n_frames - stride):
                j = i + stride
                grid_t = frames[i]["grid"]
                grid_tj = frames[j]["grid"]
                step = frames[i]["step"]

                x = grid_to_onehot(grid_t, height, width)
                y = grid_to_class_indices(grid_tj, height, width)

                X_list.append(x)
                T_list.append(step)
                H_list.append(stride)
                Y_list.append(y)
                meta.append({
                    "round_id": rid,
                    "seed_index": replay["seed_index"],
                    "step": step,
                    "horizon": stride,
                })

    X = np.stack(X_list)
    T_arr = np.array(T_list, dtype=np.int64)
    H_arr = np.array(H_list, dtype=np.int64)
    Y = np.stack(Y_list)

    # Report per-stride counts
    for s in sorted(set(H_arr)):
        count = (H_arr == s).sum()
        print(f"    Stride {s:2d}: {count:5d} pairs")
    n_rounds = len(set(m['round_id'] for m in meta))
    print(f"  Total: {len(X)} transition pairs from {n_rounds} rounds")

    return X, T_arr, H_arr, Y, meta


# ---------------------------------------------------------------------------
# Build training data — Direct predictor (initial → ground truth)
# ---------------------------------------------------------------------------

def build_direct_dataset(gt_data, held_out_rounds=None):
    """Build (initial_state, ground_truth_distribution) pairs."""
    X_list, Y_list, meta = [], [], []

    for gt in gt_data:
        rid = gt.get("_round_id", "")[:8] if gt.get("_round_id") else ""
        if held_out_rounds and rid in held_out_rounds:
            continue

        initial_grid = gt.get("initial_grid")
        ground_truth = gt.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            continue

        height = gt["height"]
        width = gt["width"]

        x = grid_to_onehot(initial_grid, height, width)
        y = np.array(ground_truth, dtype=np.float32).transpose(2, 0, 1)  # (6, H, W)

        X_list.append(x)
        Y_list.append(y)
        meta.append({"round_id": rid, "seed_index": gt.get("_seed_index", 0)})

    if not X_list:
        return (np.zeros((0, NUM_TERRAIN, MAP_H, MAP_W)),
                np.zeros((0, SUBMIT_CLASSES, MAP_H, MAP_W)), [])

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    print(f"  Direct samples: {len(X)} (from {len(set(m['round_id'] for m in meta))} rounds)")
    return X, Y, meta


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_divergence_loss(pred_logits, target_probs, eps=1e-8):
    """KL(target || pred) for DirectPredictor."""
    pred_log_probs = F.log_softmax(pred_logits, dim=1)
    target_clamped = target_probs.clamp(min=eps)
    kl = target_clamped * (target_clamped.log() - pred_log_probs)
    return kl.sum(dim=1).mean()


def weighted_ce_loss(logits, targets, class_weights=None):
    """Cross-entropy with optional per-class weighting."""
    if class_weights is not None:
        return F.cross_entropy(logits, targets, weight=class_weights)
    return F.cross_entropy(logits, targets)


def compute_class_weights(Y, num_classes=NUM_TERRAIN, smoothing=0.1):
    """Inverse-frequency class weights to boost rare classes."""
    counts = np.bincount(Y.flatten(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + smoothing)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint(ckpt_dir, prefix="dyn"):
    """Find latest checkpoint in dir."""
    os.makedirs(ckpt_dir, exist_ok=True)
    pattern = f"{prefix}_epoch_"
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith(pattern) and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f: int(f.replace(pattern, "").replace(".pt", "")))
    return os.path.join(ckpt_dir, files[-1])


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                    metadata, ckpt_dir, prefix="dyn"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{prefix}_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "metadata": metadata,
    }, path)
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Training — VariableHorizonDynamics
# ---------------------------------------------------------------------------

def train_dynamics(replays, epochs=DYN_EPOCHS, lr=DYN_LR,
                   batch_size=DYN_BATCH, cv="all", reset=False,
                   augment=True, noise=True):
    """Train the variable-horizon dynamics model on multi-stride transitions."""
    round_ids = sorted(set(r["round_id"][:8] for r in replays))

    if cv == "round_kfold":
        _train_dynamics_kfold(replays, round_ids, epochs, lr, batch_size,
                              augment, noise)
        return

    print(f"\n{'='*60}")
    print(f"  VariableHorizonDynamics Training (all data)")
    print(f"  strides={TRAIN_STRIDES}, augment={augment}, noise={noise}")
    print(f"{'='*60}")

    X, T_arr, H_arr, Y, meta = build_multistride_dataset(replays)
    class_weights = compute_class_weights(Y).to(DEVICE)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    _train_dynamics_loop(X, T_arr, H_arr, Y, None, None, None, None,
                         epochs, lr, batch_size, class_weights,
                         reset=reset, augment=augment, noise=noise,
                         save_ckpts=True, tag="all")


def _train_dynamics_kfold(replays, round_ids, epochs, lr, batch_size,
                          augment, noise):
    K = len(round_ids)
    print(f"\n{'='*60}")
    print(f"  VariableHorizonDynamics — {K}-fold Round CV")
    print(f"  strides={TRAIN_STRIDES}")
    print(f"{'='*60}")

    fold_losses = []
    for fold_i, val_rid in enumerate(round_ids):
        print(f"\n  --- Fold {fold_i+1}/{K}: holdout {val_rid} ---")

        X_train, T_train, H_train, Y_train, _ = build_multistride_dataset(
            replays, held_out_rounds={val_rid})
        X_val, T_val, H_val, Y_val, _ = build_multistride_dataset(
            [r for r in replays if r["round_id"][:8] == val_rid])

        class_weights = compute_class_weights(Y_train).to(DEVICE)

        best_val = _train_dynamics_loop(
            X_train, T_train, H_train, Y_train,
            X_val, T_val, H_val, Y_val,
            epochs, lr, batch_size, class_weights,
            reset=True, augment=augment, noise=noise,
            save_ckpts=False, tag=f"fold{fold_i+1}")
        fold_losses.append(best_val)

    mean_val = np.mean(fold_losses)
    std_val = np.std(fold_losses)
    print(f"\n  {K}-fold CV — Mean: {mean_val:.6f} ± {std_val:.6f}")


def _train_dynamics_loop(X_train, T_train, H_train, Y_train,
                         X_val, T_val, H_val, Y_val,
                         epochs, lr, batch_size, class_weights,
                         reset=False, augment=True, noise=True,
                         save_ckpts=True, tag=""):
    """Core training loop for VariableHorizonDynamics."""

    X_t = torch.tensor(X_train, device=DEVICE)
    T_t = torch.tensor(T_train, device=DEVICE)
    H_t = torch.tensor(H_train, device=DEVICE)
    Y_t = torch.tensor(Y_train, device=DEVICE)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_v = torch.tensor(X_val, device=DEVICE)
        T_v = torch.tensor(T_val, device=DEVICE)
        H_v = torch.tensor(H_val, device=DEVICE)
        Y_v = torch.tensor(Y_val, device=DEVICE)

    model = VariableHorizonDynamics().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Resume
    start_epoch = 1
    if save_ckpts and not reset:
        ckpt_path = latest_checkpoint(CKPT_DIR_DYN, "dyn")
        if ckpt_path:
            print(f"  Resuming from {os.path.basename(ckpt_path)}")
            ckpt = load_checkpoint(ckpt_path, model, optimizer)
            start_epoch = ckpt["epoch"] + 1

    dataset = TensorDataset(X_t, T_t, H_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 150
    training_start = time.time()

    metadata = {"tag": tag, "n_train": len(X_train),
                "model": "VariableHorizonDynamics",
                "strides": TRAIN_STRIDES}

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, T_b, H_b, Y_b in loader:
            # Data augmentation
            if augment:
                aug_x, aug_y = [], []
                for i in range(X_b.shape[0]):
                    ax, ay = augment_pair(X_b[i], Y_b[i])
                    aug_x.append(ax)
                    aug_y.append(ay)
                X_b = torch.stack(aug_x)
                Y_b = torch.stack(aug_y)

            # Noise injection (scheduled sampling)
            if noise:
                X_b = add_input_noise(X_b)

            optimizer.zero_grad()
            logits = model(X_b, T_b, H_b)
            loss = weighted_ce_loss(logits, Y_b, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation (no noise, no augmentation)
        val_loss = avg_train
        if has_val:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, len(X_v), batch_size):
                    xb = X_v[i:i+batch_size]
                    tb = T_v[i:i+batch_size]
                    hb = H_v[i:i+batch_size]
                    yb = Y_v[i:i+batch_size]
                    logits_v = model(xb, tb, hb)
                    val_losses.append(
                        weighted_ce_loss(logits_v, yb, class_weights).item())
                val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            marker = " ★"
        else:
            no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - training_start
            lr_now = optimizer.param_groups[0]["lr"]
            skip_g = model.skip_gain.item()
            val_str = f"val={val_loss:.6f}" if has_val else "no val"
            print(f"  Epoch {epoch:4d}/{epochs} | train={avg_train:.6f} | "
                  f"{val_str}{marker} | skip={skip_g:.2f} | "
                  f"lr={lr_now:.1e} | {elapsed:.0f}s")

        if save_ckpts and epoch % CHECKPOINT_EVERY == 0:
            metadata["best_val_loss"] = best_val
            path = save_checkpoint(model, optimizer, epoch, avg_train,
                                   val_loss, metadata, CKPT_DIR_DYN, "dyn")
            print(f"    → Saved {os.path.basename(path)}")

        if has_val and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save final
    if save_ckpts:
        metadata["total_epochs"] = epoch
        metadata["best_val_loss"] = best_val
        final_path = os.path.join(CKPT_DIR_DYN, "dyn_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train,
            "val_loss": val_loss,
            "metadata": metadata,
        }, final_path)
        print(f"  Final: {final_path}")

    print(f"  Best val CE: {best_val:.6f}")
    return best_val


# ---------------------------------------------------------------------------
# Training — DirectPredictor
# ---------------------------------------------------------------------------

def train_direct(gt_data, epochs=DIRECT_EPOCHS, lr=DIRECT_LR,
                 batch_size=DIRECT_BATCH, cv="all", reset=False,
                 augment=True):
    """Train the DirectPredictor on initial_grid → ground_truth_distribution."""
    round_ids = sorted(set(
        g.get("_round_id", "")[:8] for g in gt_data if g.get("_round_id")))

    if cv == "round_kfold":
        _train_direct_kfold(gt_data, round_ids, epochs, lr, batch_size, augment)
        return

    print(f"\n{'='*60}")
    print(f"  DirectPredictor Training (all data)")
    print(f"  augment={augment}")
    print(f"{'='*60}")

    X, Y, meta = build_direct_dataset(gt_data)
    if len(X) == 0:
        print("ERROR: No training data")
        return

    _train_direct_loop(X, Y, None, None, epochs, lr, batch_size,
                       reset=reset, augment=augment, save_ckpts=True, tag="all")


def _train_direct_kfold(gt_data, round_ids, epochs, lr, batch_size, augment):
    K = len(round_ids)
    print(f"\n{'='*60}")
    print(f"  DirectPredictor — {K}-fold Round CV")
    print(f"{'='*60}")

    fold_losses = []
    for fold_i, val_rid in enumerate(round_ids):
        print(f"\n  --- Fold {fold_i+1}/{K}: holdout {val_rid} ---")

        X_train, Y_train, _ = build_direct_dataset(
            gt_data, held_out_rounds={val_rid})
        X_val, Y_val, _ = build_direct_dataset(
            [g for g in gt_data if g.get("_round_id", "")[:8] == val_rid])

        if len(X_train) == 0 or len(X_val) == 0:
            print("    Skipping fold (no data)")
            continue

        best_val = _train_direct_loop(
            X_train, Y_train, X_val, Y_val,
            epochs, lr, batch_size, reset=True, augment=augment,
            save_ckpts=False, tag=f"fold{fold_i+1}")
        fold_losses.append(best_val)

    if fold_losses:
        print(f"\n  {K}-fold CV — Mean KL: {np.mean(fold_losses):.6f} "
              f"± {np.std(fold_losses):.6f}")


def _train_direct_loop(X_train, Y_train, X_val, Y_val, epochs, lr,
                       batch_size, reset=False, augment=True,
                       save_ckpts=True, tag=""):
    """Core training loop for DirectPredictor."""

    X_t = torch.tensor(X_train, device=DEVICE)
    Y_t = torch.tensor(Y_train, device=DEVICE)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_v = torch.tensor(X_val, device=DEVICE)
        Y_v = torch.tensor(Y_val, device=DEVICE)

    model = DirectPredictor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    start_epoch = 1
    if save_ckpts and not reset:
        ckpt_path = latest_checkpoint(CKPT_DIR_DIRECT, "direct")
        if ckpt_path:
            print(f"  Resuming from {os.path.basename(ckpt_path)}")
            ckpt = load_checkpoint(ckpt_path, model, optimizer)
            start_epoch = ckpt["epoch"] + 1

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 200
    training_start = time.time()

    metadata = {"tag": tag, "n_train": len(X_train), "model": "DirectPredictor"}

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, Y_b in loader:
            if augment:
                aug_x, aug_y = [], []
                for i in range(X_b.shape[0]):
                    ax, ay = augment_grid(X_b[i], Y_b[i])
                    aug_x.append(ax)
                    aug_y.append(ay)
                X_b = torch.stack(aug_x)
                Y_b = torch.stack(aug_y)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = kl_divergence_loss(logits, Y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        val_loss = avg_train
        if has_val:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, len(X_v), batch_size):
                    xb = X_v[i:i+batch_size]
                    yb = Y_v[i:i+batch_size]
                    logits_v = model(xb)
                    val_losses.append(kl_divergence_loss(logits_v, yb).item())
                val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            marker = " ★"
        else:
            no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - training_start
            lr_now = optimizer.param_groups[0]["lr"]
            val_str = f"val={val_loss:.6f}" if has_val else "no val"
            print(f"  Epoch {epoch:4d}/{epochs} | train={avg_train:.6f} | "
                  f"{val_str}{marker} | lr={lr_now:.1e} | {elapsed:.0f}s")

        if save_ckpts and epoch % CHECKPOINT_EVERY == 0:
            metadata["best_val_loss"] = best_val
            path = save_checkpoint(model, optimizer, epoch, avg_train,
                                   val_loss, metadata, CKPT_DIR_DIRECT, "direct")
            print(f"    → Saved {os.path.basename(path)}")

        if has_val and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if save_ckpts:
        metadata["total_epochs"] = epoch
        metadata["best_val_loss"] = best_val
        final_path = os.path.join(CKPT_DIR_DIRECT, "direct_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train,
            "val_loss": val_loss,
            "metadata": metadata,
        }, final_path)
        print(f"  Final: {final_path}")

    print(f"  Best val KL: {best_val:.6f}")
    return best_val


# ---------------------------------------------------------------------------
# MC Rollout Engine — Multi-step with variable stride
# ---------------------------------------------------------------------------

def rollout_trajectories(model, initial_grid, height, width,
                         K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS,
                         stride=INFER_STRIDE, temperature=1.0):
    """
    Sample K stochastic trajectories using stride-N steps.

    With stride=10 and T=50: takes 5 autoregressive steps (0→10→20→30→40→50)
    instead of 50 single-step transitions. This dramatically reduces
    error accumulation while maintaining trajectory diversity through sampling.

    Args:
        model: VariableHorizonDynamics model
        initial_grid: 2D terrain grid (list of lists)
        height, width: grid dimensions
        K: number of rollout trajectories
        T: total simulation time (50 years)
        stride: step size for autoregressive rollout
        temperature: sampling temperature (higher = more diversity)

    Returns: (K, n_steps+1, H, W) int64 array of terrain class indices
    """
    model.eval()
    s0 = grid_to_class_indices(initial_grid, height, width)

    # Compute timestep schedule: 0, stride, 2*stride, ..., T
    schedule = list(range(0, T, stride))
    if schedule[-1] + stride != T:
        schedule.append(T - stride)  # ensure we reach T
    n_jumps = len(schedule)

    # Store full trajectory (at computed timesteps + final)
    trajectories = np.zeros((K, n_jumps + 1, height, width), dtype=np.int64)
    trajectories[:, 0, :, :] = s0[np.newaxis, :, :]

    chunk_size = min(K, 64)

    with torch.no_grad():
        for chunk_start in range(0, K, chunk_size):
            chunk_end = min(chunk_start + chunk_size, K)
            B = chunk_end - chunk_start

            current = torch.tensor(
                s0, device=DEVICE).unsqueeze(0).expand(B, -1, -1).clone()

            for step_i, t_start in enumerate(schedule):
                t_end = t_start + stride
                actual_horizon = min(stride, T - t_start)

                # One-hot encode current state
                onehot = torch.zeros(
                    B, NUM_TERRAIN, height, width, device=DEVICE)
                onehot.scatter_(1, current.unsqueeze(1), 1.0)

                # Timestep and horizon tensors
                t_tensor = torch.full(
                    (B,), t_start, dtype=torch.long, device=DEVICE)
                h_tensor = torch.full(
                    (B,), actual_horizon, dtype=torch.long, device=DEVICE)

                # Predict
                logits = model(onehot, t_tensor, h_tensor)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=1)

                # Sample next state
                flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, NUM_TERRAIN)
                sampled = torch.multinomial(flat_probs, 1).squeeze(1)
                current = sampled.reshape(B, height, width)

                trajectories[chunk_start:chunk_end, step_i + 1, :, :] = \
                    current.cpu().numpy()

    return trajectories


# ---------------------------------------------------------------------------
# Observation scoring
# ---------------------------------------------------------------------------

_STATIC_CODES = {CODE_TO_IDX[10], CODE_TO_IDX[11], CODE_TO_IDX[5]}


def aggregate_observations(observations, height, width):
    """Aggregate viewport observations into per-cell empirical histograms."""
    obs_counts = np.zeros((height, width, NUM_TERRAIN), dtype=np.float32)
    obs_total = np.zeros((height, width), dtype=np.float32)

    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        grid = obs["grid"]
        for dy in range(len(grid)):
            for dx in range(len(grid[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    idx = CODE_TO_IDX.get(grid[dy][dx], 0)
                    obs_counts[gy, gx, idx] += 1.0
                    obs_total[gy, gx] += 1.0

    obs_probs = np.zeros_like(obs_counts)
    mask = obs_total > 0
    for c in range(NUM_TERRAIN):
        obs_probs[:, :, c][mask] = obs_counts[:, :, c][mask] / obs_total[mask]

    return obs_probs, obs_total


def score_rollouts(trajectories, obs_probs, obs_total, initial_grid_indices,
                   eps=1e-6):
    """Score each rollout against viewport observations (dynamic cells only)."""
    K = trajectories.shape[0]

    observed = obs_total > 0
    dynamic = np.ones_like(observed)
    for static_idx in _STATIC_CODES:
        dynamic &= (initial_grid_indices != static_idx)
    score_mask = observed & dynamic
    obs_ys, obs_xs = np.where(score_mask)
    n_scored = len(obs_ys)

    if n_scored == 0:
        return np.zeros(K, dtype=np.float64)

    # Use the LAST timestep of each trajectory as the final state
    final_states = trajectories[:, -1, :, :]
    final_classes = final_states[:, obs_ys, obs_xs]
    scored_obs_probs = obs_probs[obs_ys, obs_xs, :]

    log_weights = np.zeros(K, dtype=np.float64)
    for k in range(K):
        probs_k = scored_obs_probs[np.arange(n_scored), final_classes[k]]
        log_weights[k] = np.log(probs_k + eps).sum()

    log_weights /= n_scored
    return log_weights


def normalize_weights(log_weights, beta=1.0):
    """Softmax normalization of log weights with tempering."""
    lw = beta * log_weights
    lw = lw - lw.max()
    weights = np.exp(lw)
    total = weights.sum()
    if total < 1e-12:
        return np.ones_like(weights) / len(weights)
    return weights / total


# ---------------------------------------------------------------------------
# Aggregation: rollouts → submission tensor
# ---------------------------------------------------------------------------

def aggregate_predictions_fast(trajectories, weights, height, width):
    """Weighted class frequencies from final rollout states → (H, W, 6)."""
    K = trajectories.shape[0]
    final_states = trajectories[:, -1, :, :]

    mapping = np.array(IDX8_TO_SUBMIT6, dtype=np.int64)
    final_submit = mapping[final_states]

    pred = np.zeros((height, width, SUBMIT_CLASSES), dtype=np.float64)
    for k in range(K):
        for c in range(SUBMIT_CLASSES):
            pred[:, :, c] += weights[k] * (final_submit[k] == c).astype(np.float64)

    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def blend_with_observations(pred, observations, height, width, kappa=3.0):
    """Blend MC prediction with empirical observation distribution."""
    if not observations:
        return pred, 0

    obs_counts_6 = np.zeros((height, width, SUBMIT_CLASSES), dtype=np.float64)
    obs_total = np.zeros((height, width), dtype=np.float64)

    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        grid = obs["grid"]
        for dy in range(len(grid)):
            for dx in range(len(grid[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    terrain_code = grid[dy][dx]
                    submit_cls = TERRAIN_TO_SUBMIT.get(terrain_code, 0)
                    obs_counts_6[gy, gx, submit_cls] += 1.0
                    obs_total[gy, gx] += 1.0

    observed_mask = obs_total > 0
    n_blended = int(observed_mask.sum())
    if n_blended == 0:
        return pred, 0

    obs_empirical = np.zeros_like(obs_counts_6)
    for c in range(SUBMIT_CLASSES):
        obs_empirical[:, :, c][observed_mask] = (
            obs_counts_6[:, :, c][observed_mask] / obs_total[observed_mask])

    alpha = np.zeros((height, width, 1), dtype=np.float64)
    alpha[observed_mask, 0] = (
        obs_total[observed_mask] / (obs_total[observed_mask] + kappa))

    blended = pred.copy().astype(np.float64)
    blended = alpha * obs_empirical + (1.0 - alpha) * blended

    blended = np.maximum(blended, PROB_FLOOR).astype(np.float32)
    blended = blended / blended.sum(axis=-1, keepdims=True)
    return blended, n_blended


# ---------------------------------------------------------------------------
# Inference — Ensemble of DirectPredictor + MC rollouts
# ---------------------------------------------------------------------------

def predict_with_direct(direct_model, initial_grid, height, width):
    """DirectPredictor inference → (H, W, 6) probability tensor."""
    direct_model.eval()
    x = grid_to_onehot(initial_grid, height, width)
    x_t = torch.tensor(x, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        probs = direct_model.predict_probs(x_t)

    pred = probs[0].cpu().numpy().transpose(1, 2, 0)
    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def predict_round(dyn_model, direct_model, initial_grid, height, width,
                  observations, K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS,
                  stride=INFER_STRIDE, temperature=1.0, beta=1.0, kappa=3.0,
                  ensemble_weight=0.5):
    """
    Full prediction pipeline for one seed.

    1. DirectPredictor → baseline prediction
    2. If dyn_model and observations available: MC rollouts → conditioned prediction
    3. Ensemble: blend direct + MC predictions
    4. Blend with empirical observations
    """
    # --- Direct prediction ---
    pred_direct = None
    if direct_model is not None:
        print(f"    DirectPredictor inference...")
        pred_direct = predict_with_direct(direct_model, initial_grid,
                                          height, width)

    # --- MC rollouts ---
    pred_mc = None
    if dyn_model is not None:
        n_steps = T // stride
        print(f"    Rolling out {K} trajectories: {n_steps} steps "
              f"(stride={stride}, T={T}, temp={temperature})...")
        t0 = time.time()
        trajectories = rollout_trajectories(
            dyn_model, initial_grid, height, width,
            K, T, stride=stride, temperature=temperature)
        print(f"    Rollouts done in {time.time()-t0:.1f}s")

        if observations:
            obs_probs, obs_total = aggregate_observations(
                observations, height, width)
            initial_indices = grid_to_class_indices(
                initial_grid, height, width)

            log_weights = score_rollouts(
                trajectories, obs_probs, obs_total, initial_indices)
            weights = normalize_weights(log_weights, beta=beta)
            ess = 1.0 / (weights ** 2).sum()
            print(f"    ESS: {ess:.1f}/{K}")
        else:
            weights = np.ones(K, dtype=np.float64) / K

        pred_mc = aggregate_predictions_fast(
            trajectories, weights, height, width)

    # --- Ensemble ---
    if pred_direct is not None and pred_mc is not None:
        w_direct = ensemble_weight
        w_mc = 1.0 - ensemble_weight
        pred = (w_direct * pred_direct + w_mc * pred_mc).astype(np.float32)
        print(f"    Ensemble: direct={w_direct:.1%}, MC={w_mc:.1%}")
    elif pred_direct is not None:
        pred = pred_direct
    elif pred_mc is not None:
        pred = pred_mc
    else:
        raise RuntimeError("No model available for prediction!")

    # --- Blend with observations ---
    if observations:
        pred, n_blended = blend_with_observations(
            pred, observations, height, width, kappa=kappa)
        print(f"    Observation blending: {n_blended} cells (kappa={kappa})")

    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def entropy_per_pixel(target):
    t = np.clip(target, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def kl_per_pixel(pred, target):
    pred = np.clip(pred, 1e-8, None)
    target = np.clip(target, 1e-8, None)
    return (target * (np.log(target) - np.log(pred))).sum(axis=-1)


def weighted_kl(pred, target):
    kl = kl_per_pixel(pred, target)
    ent = entropy_per_pixel(target)
    total_ent = ent.sum()
    if total_ent < 1e-12:
        return 0.0
    return (kl * ent).sum() / total_ent


def competition_score(pred, target):
    wkl = weighted_kl(pred, target)
    return 100.0 * np.exp(-3.0 * wkl), wkl


# ---------------------------------------------------------------------------
# Evaluation — Round k-fold
# ---------------------------------------------------------------------------

def evaluate_all(replays, gt_data, K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS,
                 stride=INFER_STRIDE, dyn_epochs=DYN_EPOCHS,
                 direct_epochs=DIRECT_EPOCHS, use_obs=True):
    """
    Round k-fold evaluation of all prediction modes:
    1. DirectPredictor only
    2. VariableHorizonDynamics MC only (stride-N rollout)
    3. Ensemble (direct + MC)
    """
    replay_by_round = {}
    for r in replays:
        rid = r["round_id"][:8]
        replay_by_round.setdefault(rid, []).append(r)

    gt_by_round = {}
    for g in gt_data:
        rid = g.get("_round_id", "")[:8] if g.get("_round_id") else ""
        gt_by_round.setdefault(rid, []).append(g)

    common_rounds = sorted(
        set(replay_by_round.keys()) & set(gt_by_round.keys()))
    if len(common_rounds) < 2:
        print("ERROR: Need at least 2 rounds with both replays and GT")
        return

    print(f"\n{'='*60}")
    print(f"  MC v2 Evaluation — {len(common_rounds)}-fold Round CV")
    print(f"  Rounds: {common_rounds}")
    print(f"  K={K}, T={T}, stride={stride}, obs={'ON' if use_obs else 'OFF'}")
    print(f"  Training strides: {TRAIN_STRIDES}")
    print(f"{'='*60}")

    results = {"direct": [], "mc": [], "ensemble": []}

    for fold_i, val_rid in enumerate(common_rounds):
        print(f"\n  === Fold {fold_i+1}/{len(common_rounds)}: "
              f"holdout {val_rid} ===")

        # --- Train DirectPredictor ---
        print(f"\n  Training DirectPredictor...")
        X_d_train, Y_d_train, _ = build_direct_dataset(
            gt_data, held_out_rounds={val_rid})
        X_d_val, Y_d_val, _ = build_direct_dataset(
            [g for g in gt_data if g.get("_round_id", "")[:8] == val_rid])

        direct_model = None
        if len(X_d_train) > 0:
            direct_model = DirectPredictor().to(DEVICE)
            opt_d = torch.optim.AdamW(
                direct_model.parameters(), lr=DIRECT_LR, weight_decay=1e-4)

            X_dt = torch.tensor(X_d_train, device=DEVICE)
            Y_dt = torch.tensor(Y_d_train, device=DEVICE)
            ds = TensorDataset(X_dt, Y_dt)
            dl = DataLoader(ds, batch_size=DIRECT_BATCH, shuffle=True)

            t_start = time.time()
            for epoch in range(1, direct_epochs + 1):
                direct_model.train()
                e_loss, nb = 0.0, 0
                for xb, yb in dl:
                    aug_x, aug_y = [], []
                    for i in range(xb.shape[0]):
                        ax, ay = augment_grid(xb[i], yb[i])
                        aug_x.append(ax)
                        aug_y.append(ay)
                    xb = torch.stack(aug_x)
                    yb = torch.stack(aug_y)

                    opt_d.zero_grad()
                    loss = kl_divergence_loss(direct_model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        direct_model.parameters(), 1.0)
                    opt_d.step()
                    e_loss += loss.item()
                    nb += 1
                if epoch % 50 == 0 or epoch == 1:
                    print(f"    Direct epoch {epoch}/{direct_epochs} | "
                          f"loss={e_loss/max(nb,1):.6f} | "
                          f"{time.time()-t_start:.0f}s")

        # --- Train VariableHorizonDynamics ---
        print(f"\n  Training VariableHorizonDynamics...")
        train_replays = [r for r in replays if r["round_id"][:8] != val_rid]
        X_train, T_train, H_train, Y_train, _ = build_multistride_dataset(
            train_replays)
        class_weights = compute_class_weights(Y_train).to(DEVICE)

        dyn_model = VariableHorizonDynamics().to(DEVICE)
        opt_m = torch.optim.AdamW(
            dyn_model.parameters(), lr=DYN_LR, weight_decay=1e-4)

        X_mt = torch.tensor(X_train, device=DEVICE)
        T_mt = torch.tensor(T_train, device=DEVICE)
        H_mt = torch.tensor(H_train, device=DEVICE)
        Y_mt = torch.tensor(Y_train, device=DEVICE)
        ds_m = TensorDataset(X_mt, T_mt, H_mt, Y_mt)
        dl_m = DataLoader(ds_m, batch_size=DYN_BATCH, shuffle=True)

        t_start = time.time()
        for epoch in range(1, dyn_epochs + 1):
            dyn_model.train()
            e_loss, nb = 0.0, 0
            for xb, tb, hb, yb in dl_m:
                aug_x, aug_y = [], []
                for i in range(xb.shape[0]):
                    ax, ay = augment_pair(xb[i], yb[i])
                    aug_x.append(ax)
                    aug_y.append(ay)
                xb = torch.stack(aug_x)
                yb = torch.stack(aug_y)

                # Noise injection
                xb = add_input_noise(xb)

                opt_m.zero_grad()
                logits = dyn_model(xb, tb, hb)
                loss = weighted_ce_loss(logits, yb, class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), 1.0)
                opt_m.step()
                e_loss += loss.item()
                nb += 1
            if epoch % 25 == 0 or epoch == 1:
                print(f"    Dyn epoch {epoch}/{dyn_epochs} | "
                      f"loss={e_loss/max(nb,1):.6f} | "
                      f"skip={dyn_model.skip_gain.item():.2f} | "
                      f"{time.time()-t_start:.0f}s")

        # --- Evaluate on holdout ---
        val_gts = gt_by_round[val_rid]
        val_obs_all = load_observations(val_rid)

        for gt_sample in val_gts:
            initial_grid = gt_sample.get("initial_grid")
            ground_truth = gt_sample.get("ground_truth")
            if initial_grid is None or ground_truth is None:
                continue

            seed = gt_sample.get("_seed_index", 0)
            height = gt_sample["height"]
            width = gt_sample["width"]
            gt_tensor = np.array(ground_truth, dtype=np.float32)

            seed_obs = ([o for o in val_obs_all
                         if o.get("seed_index") == seed]
                        if use_obs else [])

            print(f"\n  Round {val_rid} seed {seed}:")

            # 1. Direct only
            if direct_model is not None:
                pred_d = predict_with_direct(
                    direct_model, initial_grid, height, width)
                score_d, wkl_d = competition_score(pred_d, gt_tensor)
                results["direct"].append(score_d)
                print(f"    Direct:   score={score_d:.2f}  wKL={wkl_d:.6f}")

            # 2. MC only (stride-N rollout)
            pred_mc_full = predict_round(
                dyn_model, None, initial_grid, height, width,
                seed_obs, K=K, T=T, stride=stride, ensemble_weight=0.0)
            score_mc, wkl_mc = competition_score(pred_mc_full, gt_tensor)
            results["mc"].append(score_mc)
            print(f"    MC only:  score={score_mc:.2f}  wKL={wkl_mc:.6f}")

            # 3. Ensemble
            if direct_model is not None:
                pred_ens = predict_round(
                    dyn_model, direct_model, initial_grid, height, width,
                    seed_obs, K=K, T=T, stride=stride,
                    ensemble_weight=0.5)
                score_e, wkl_e = competition_score(pred_ens, gt_tensor)
                results["ensemble"].append(score_e)
                print(f"    Ensemble: score={score_e:.2f}  wKL={wkl_e:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  MC v2 Evaluation Results (stride={stride})")
    for mode, scores in results.items():
        if scores:
            print(f"  {mode:10s}: {np.mean(scores):.2f} ± "
                  f"{np.std(scores):.2f}  "
                  f"(min={np.min(scores):.2f}, max={np.max(scores):.2f})")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Evaluate DirectPredictor only
# ---------------------------------------------------------------------------

def evaluate_direct_only(gt_data, epochs=DIRECT_EPOCHS):
    """Quick round k-fold evaluation of DirectPredictor alone."""
    round_ids = sorted(set(
        g.get("_round_id", "")[:8] for g in gt_data if g.get("_round_id")))

    if len(round_ids) < 2:
        print("ERROR: Need at least 2 rounds for evaluation")
        return

    print(f"\n{'='*60}")
    print(f"  DirectPredictor Only — {len(round_ids)}-fold Round CV")
    print(f"{'='*60}")

    all_scores, all_wkls = [], []

    for fold_i, val_rid in enumerate(round_ids):
        print(f"\n  --- Fold {fold_i+1}/{len(round_ids)}: "
              f"holdout {val_rid} ---")

        X_train, Y_train, _ = build_direct_dataset(
            gt_data, held_out_rounds={val_rid})
        X_val, Y_val, meta_val = build_direct_dataset(
            [g for g in gt_data if g.get("_round_id", "")[:8] == val_rid])

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = DirectPredictor().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=DIRECT_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=DIRECT_LR * 0.01)

        X_t = torch.tensor(X_train, device=DEVICE)
        Y_t = torch.tensor(Y_train, device=DEVICE)
        ds = TensorDataset(X_t, Y_t)
        dl = DataLoader(ds, batch_size=DIRECT_BATCH, shuffle=True)

        t_start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            e_loss, nb = 0.0, 0
            for xb, yb in dl:
                aug_x, aug_y = [], []
                for i in range(xb.shape[0]):
                    ax, ay = augment_grid(xb[i], yb[i])
                    aug_x.append(ax)
                    aug_y.append(ay)
                xb = torch.stack(aug_x)
                yb = torch.stack(aug_y)

                optimizer.zero_grad()
                loss = kl_divergence_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                e_loss += loss.item()
                nb += 1
            scheduler.step()
            if epoch % 50 == 0 or epoch == 1:
                print(f"    Epoch {epoch}/{epochs} | "
                      f"loss={e_loss/max(nb,1):.6f} | "
                      f"{time.time()-t_start:.0f}s")

        # Evaluate on holdout
        model.eval()
        val_gt_items = [g for g in gt_data
                        if g.get("_round_id", "")[:8] == val_rid]
        for gt_sample in val_gt_items:
            initial_grid = gt_sample.get("initial_grid")
            ground_truth = gt_sample.get("ground_truth")
            if initial_grid is None or ground_truth is None:
                continue

            seed = gt_sample.get("_seed_index", 0)
            height, width = gt_sample["height"], gt_sample["width"]
            gt_tensor = np.array(ground_truth, dtype=np.float32)

            pred = predict_with_direct(model, initial_grid, height, width)
            score, wkl = competition_score(pred, gt_tensor)
            all_scores.append(score)
            all_wkls.append(wkl)
            print(f"    Round {val_rid} seed {seed}: "
                  f"score={score:.2f} wKL={wkl:.6f}")

    if all_scores:
        print(f"\n{'='*60}")
        print(f"  DirectPredictor Results")
        print(f"  Samples: {len(all_scores)}")
        print(f"  Score: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
        print(f"  wKL:   {np.mean(all_wkls):.6f} ± {np.std(all_wkls):.6f}")
        print(f"  Range: [{np.min(all_scores):.2f}, {np.max(all_scores):.2f}]")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Astar Island — Monte Carlo v2: "
                    "Variable-Horizon Transformer Dynamics")
    sub = parser.add_subparsers(dest="command")

    # Train dynamics
    p_dyn = sub.add_parser(
        "train-dynamics",
        help="Train VariableHorizonDynamics on multi-stride transitions")
    p_dyn.add_argument("--cv", choices=["all", "round_kfold"], default="all")
    p_dyn.add_argument("--epochs", type=int, default=DYN_EPOCHS)
    p_dyn.add_argument("--lr", type=float, default=DYN_LR)
    p_dyn.add_argument("--batch", type=int, default=DYN_BATCH)
    p_dyn.add_argument("--reset", action="store_true")
    p_dyn.add_argument("--no-augment", action="store_true")
    p_dyn.add_argument("--no-noise", action="store_true",
                       help="Disable input noise injection")

    # Train direct predictor
    p_dir = sub.add_parser(
        "train-direct",
        help="Train DirectPredictor on ground truth distributions")
    p_dir.add_argument("--cv", choices=["all", "round_kfold"], default="all")
    p_dir.add_argument("--epochs", type=int, default=DIRECT_EPOCHS)
    p_dir.add_argument("--lr", type=float, default=DIRECT_LR)
    p_dir.add_argument("--batch", type=int, default=DIRECT_BATCH)
    p_dir.add_argument("--reset", action="store_true")
    p_dir.add_argument("--no-augment", action="store_true")

    # Evaluate all modes
    p_eval = sub.add_parser(
        "evaluate",
        help="Round k-fold evaluation of all modes")
    p_eval.add_argument("--rollouts", "-K", type=int,
                        default=DEFAULT_K_ROLLOUTS)
    p_eval.add_argument("--steps", "-T", type=int, default=DEFAULT_T_STEPS)
    p_eval.add_argument("--stride", type=int, default=INFER_STRIDE,
                        help="Rollout stride (default: 10)")
    p_eval.add_argument("--dyn-epochs", type=int, default=DYN_EPOCHS)
    p_eval.add_argument("--direct-epochs", type=int, default=DIRECT_EPOCHS)
    p_eval.add_argument("--no-obs", action="store_true")

    # Quick eval of DirectPredictor only
    p_eval_d = sub.add_parser(
        "evaluate-direct",
        help="Quick evaluation of DirectPredictor only")
    p_eval_d.add_argument("--epochs", type=int, default=DIRECT_EPOCHS)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    print(f"Device: {DEVICE}")

    if args.command == "train-dynamics":
        replays = load_all_replays()
        if not replays:
            return
        train_dynamics(replays, epochs=args.epochs, lr=args.lr,
                       batch_size=args.batch, cv=args.cv, reset=args.reset,
                       augment=not args.no_augment,
                       noise=not args.no_noise)

    elif args.command == "train-direct":
        gt_data = load_ground_truth()
        if not gt_data:
            return
        train_direct(gt_data, epochs=args.epochs, lr=args.lr,
                     batch_size=args.batch, cv=args.cv, reset=args.reset,
                     augment=not args.no_augment)

    elif args.command == "evaluate":
        replays = load_all_replays()
        gt_data = load_ground_truth()
        if not replays or not gt_data:
            return
        evaluate_all(replays, gt_data, K=args.rollouts, T=args.steps,
                     stride=args.stride,
                     dyn_epochs=args.dyn_epochs,
                     direct_epochs=args.direct_epochs,
                     use_obs=not args.no_obs)

    elif args.command == "evaluate-direct":
        gt_data = load_ground_truth()
        if not gt_data:
            return
        evaluate_direct_only(gt_data, epochs=args.epochs)


if __name__ == "__main__":
    main()
