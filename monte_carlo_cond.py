"""
Astar Island — Monte Carlo Conditional Prediction

Approach:
1. Train a dynamics model on replay transitions (frame_t → frame_t+1)
2. At inference, roll out K stochastic trajectories from the known initial map
3. Score each trajectory against viewport observations
4. Weight trajectories by observation likelihood
5. Aggregate weighted final states into H×W×6 submission tensor

The dynamics model is a simple 1-hidden-layer CNN that outputs per-cell
probabilities over 8 terrain types for the next timestep.

Usage:
  # Train dynamics model (round k-fold CV for evaluation)
  python monte_carlo_cond.py train --cv round_kfold

  # Train on all data (production)
  python monte_carlo_cond.py train --cv all

  # Evaluate: run MC rollouts on held-out rounds, compare to ground truth
  python monte_carlo_cond.py evaluate

  # Full inference on a specific round (requires observations)
  python monte_carlo_cond.py infer --round-id <short_id>
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
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_mc_cond")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8 terrain codes as they appear in replay grids
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CODE_TO_IDX = {code: i for i, code in enumerate(TERRAIN_CODES)}
IDX_TO_CODE = {i: code for code, i in CODE_TO_IDX.items()}
NUM_TERRAIN = 8  # dynamics model works in 8-class space

# 6-class submission mapping: terrain code → submission class
SUBMIT_CLASSES = 6
TERRAIN_TO_SUBMIT = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
# 8-class idx → 6-class submission idx
IDX8_TO_SUBMIT6 = [TERRAIN_TO_SUBMIT[IDX_TO_CODE[i]] for i in range(NUM_TERRAIN)]

PROB_FLOOR = 0.01

# Training defaults
DEFAULT_EPOCHS = 300
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 16
CHECKPOINT_EVERY = 25

# MC inference defaults
DEFAULT_K_ROLLOUTS = 256
DEFAULT_T_STEPS = 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_replays():
    """
    Load all simulation replays.
    Returns list of dicts with keys: round_id, seed_index, width, height, frames.
    Each frame is a 2D list of terrain codes.
    """
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
    """Load all ground truth files. Returns list of dicts."""
    if not os.path.isdir(GT_DIR):
        print(f"ERROR: Ground truth directory not found: {GT_DIR}")
        return []
    all_data = []
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        # Parse round info from filename: rN_sM_XXXX.json
        parts = fname.replace(".json", "").split("_")
        if len(parts) >= 3:
            data["_round_number"] = int(parts[0][1:])
            data["_seed_index"] = int(parts[1][1:])
            data["_round_id"] = parts[2]
        all_data.append(data)
    print(f"Loaded {len(all_data)} ground truth samples from {GT_DIR}")
    return all_data


def load_observations(round_id_short):
    """Load viewport observations for a round. Returns list of obs dicts or []."""
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
    """
    Convert a 2D terrain grid (list of lists of terrain codes) to one-hot tensor.
    Returns: (NUM_TERRAIN, H, W) float32 numpy array.
    """
    onehot = np.zeros((NUM_TERRAIN, height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = CODE_TO_IDX.get(grid[y][x], 0)
            onehot[idx, y, x] = 1.0
    return onehot


def grid_to_class_indices(grid, height, width):
    """
    Convert a 2D terrain grid to integer class index grid.
    Returns: (H, W) int64 numpy array.
    """
    indices = np.zeros((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            indices[y, x] = CODE_TO_IDX.get(grid[y][x], 0)
    return indices


# ---------------------------------------------------------------------------
# Dynamics Model — Simple 1-hidden-layer CNN
# ---------------------------------------------------------------------------

class DynamicsCNN(nn.Module):
    """
    One-step dynamics model: predicts P(terrain_{t+1} | terrain_t) per cell.

    Simple architecture: 1 hidden layer CNN with 3×3 convolutions.
    Input:  (B, 8, H, W) — one-hot terrain at time t
    Output: (B, 8, H, W) — logits for terrain at time t+1
    """
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_TERRAIN, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, NUM_TERRAIN, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Returns logits (B, 8, H, W). Apply softmax externally for probs."""
        return self.net(x)

    def predict_probs(self, x):
        """Returns probabilities (B, 8, H, W)."""
        return F.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Build transition training data from replays
# ---------------------------------------------------------------------------

def build_transition_dataset(replays, held_out_rounds=None):
    """
    Extract (frame_t, frame_t+1) pairs from replays.

    Args:
        replays: list of replay dicts
        held_out_rounds: set of round_id prefixes to exclude (for CV)

    Returns:
        X: (N, 8, H, W) one-hot inputs (frame t)
        Y: (N, H, W) integer class targets (frame t+1)
        meta: list of dicts with round info per sample
    """
    X_list, Y_list, meta = [], [], []

    for replay in replays:
        rid = replay["round_id"][:8]
        if held_out_rounds and rid in held_out_rounds:
            continue

        height = replay["height"]
        width = replay["width"]
        frames = replay["frames"]

        for i in range(len(frames) - 1):
            grid_t = frames[i]["grid"]
            grid_t1 = frames[i + 1]["grid"]

            x = grid_to_onehot(grid_t, height, width)
            y = grid_to_class_indices(grid_t1, height, width)

            X_list.append(x)
            Y_list.append(y)
            meta.append({
                "round_id": rid,
                "seed_index": replay["seed_index"],
                "step": frames[i]["step"],
            })

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    print(f"  Transitions: {len(X)} (from {len(set(m['round_id'] for m in meta))} rounds)")
    return X, Y, meta


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint():
    """Find latest checkpoint in CKPT_DIR."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    files = [f for f in os.listdir(CKPT_DIR)
             if f.startswith("dyn_epoch_") and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f: int(f.replace("dyn_epoch_", "").replace(".pt", "")))
    return os.path.join(CKPT_DIR, files[-1])


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, metadata):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"dyn_epoch_{epoch:04d}.pt")
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
# Training
# ---------------------------------------------------------------------------

def train_dynamics(replays, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
                   batch_size=DEFAULT_BATCH, cv="all", reset=False):
    """
    Train the dynamics model on replay transitions.

    CV modes:
      - "all": train on everything (production)
      - "round_kfold": K-fold over rounds (evaluation)
    """
    # Group replays by round
    round_ids = sorted(set(r["round_id"][:8] for r in replays))

    if cv == "round_kfold":
        _train_round_kfold(replays, round_ids, epochs, lr, batch_size)
        return

    # --- Train on all data ---
    print(f"\n{'='*60}")
    print(f"  Dynamics Model Training (all data, no holdout)")
    print(f"{'='*60}")

    X, Y, meta = build_transition_dataset(replays)
    _train_loop(X, Y, None, None, epochs, lr, batch_size, reset=reset, tag="all")


def _train_round_kfold(replays, round_ids, epochs, lr, batch_size):
    """Train K separate models, one per held-out round."""
    K = len(round_ids)
    print(f"\n{'='*60}")
    print(f"  Dynamics Model — {K}-fold Round CV")
    print(f"  Rounds: {round_ids}")
    print(f"{'='*60}")

    fold_losses = []

    for fold_i, val_rid in enumerate(round_ids):
        print(f"\n  --- Fold {fold_i+1}/{K}: holdout round {val_rid} ---")

        X_train, Y_train, _ = build_transition_dataset(
            replays, held_out_rounds={val_rid})
        X_val, Y_val, _ = build_transition_dataset(
            [r for r in replays if r["round_id"][:8] == val_rid])

        best_val = _train_loop(
            X_train, Y_train, X_val, Y_val,
            epochs, lr, batch_size, reset=True,
            tag=f"fold{fold_i+1}", save_ckpts=False)
        fold_losses.append(best_val)
        print(f"    Fold {fold_i+1} best val CE: {best_val:.6f}")

    mean_val = np.mean(fold_losses)
    std_val = np.std(fold_losses)
    print(f"\n{'='*60}")
    print(f"  {K}-fold Round CV Results (cross-entropy)")
    print(f"  Per-fold: {['%.6f' % s for s in fold_losses]}")
    print(f"  Mean: {mean_val:.6f} ± {std_val:.6f}")
    print(f"{'='*60}")


def _train_loop(X_train, Y_train, X_val, Y_val, epochs, lr, batch_size,
                reset=False, tag="", save_ckpts=True):
    """
    Core training loop. Returns best validation loss (or final train loss if no val).
    """
    X_t = torch.tensor(X_train).to(DEVICE)
    Y_t = torch.tensor(Y_train).to(DEVICE)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_v = torch.tensor(X_val).to(DEVICE)
        Y_v = torch.tensor(Y_val).to(DEVICE)

    model = DynamicsCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    ) if has_val else None

    # Resume from checkpoint (only for production training)
    start_epoch = 1
    if save_ckpts and not reset:
        ckpt_path = latest_checkpoint()
        if ckpt_path:
            print(f"  Resuming from {os.path.basename(ckpt_path)}")
            ckpt = load_checkpoint(ckpt_path, model, optimizer)
            start_epoch = ckpt["epoch"] + 1

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    no_improve = 0
    early_stop_patience = 200
    training_start = time.time()

    metadata = {
        "tag": tag,
        "n_train": len(X_train),
        "n_val": len(X_val) if has_val else 0,
        "lr": lr,
        "start_time": datetime.datetime.now().isoformat(),
    }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, Y_b in loader:
            optimizer.zero_grad()
            logits = model(X_b)  # (B, 8, H, W)
            loss = loss_fn(logits, Y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss = avg_train
        if has_val:
            model.eval()
            with torch.no_grad():
                # Process in chunks to avoid OOM
                val_losses = []
                for i in range(0, len(X_v), batch_size):
                    xb = X_v[i:i+batch_size]
                    yb = Y_v[i:i+batch_size]
                    logits_v = model(xb)
                    val_losses.append(loss_fn(logits_v, yb).item())
                val_loss = np.mean(val_losses)

        if scheduler:
            scheduler.step(val_loss)

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

        # Checkpoint
        if save_ckpts and epoch % CHECKPOINT_EVERY == 0:
            metadata["end_time"] = datetime.datetime.now().isoformat()
            metadata["best_val_loss"] = best_val
            path = save_checkpoint(model, optimizer, epoch, avg_train, val_loss, metadata)
            print(f"    → Saved {os.path.basename(path)}")

        # Early stopping
        if has_val and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    # Save final
    if save_ckpts:
        metadata["end_time"] = datetime.datetime.now().isoformat()
        metadata["total_epochs"] = epoch
        metadata["best_val_loss"] = best_val
        final_path = os.path.join(CKPT_DIR, "dyn_latest.pt")
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
# Monte Carlo Rollout Engine
# ---------------------------------------------------------------------------

def rollout_trajectories(model, initial_grid, height, width,
                         K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS):
    """
    Sample K stochastic trajectories of T steps from the initial map.

    At each step, the model predicts per-cell probabilities, and we sample
    from those categorical distributions to get the next state.

    Args:
        model: trained DynamicsCNN
        initial_grid: 2D list of terrain codes (the known t=0 map)
        height, width: map dimensions
        K: number of rollouts
        T: number of timesteps

    Returns:
        trajectories: (K, T+1, H, W) int64 numpy — terrain class indices per step
    """
    model.eval()

    # Initial state as class indices
    s0 = grid_to_class_indices(initial_grid, height, width)  # (H, W)

    # Store all trajectories: (K, T+1, H, W)
    trajectories = np.zeros((K, T + 1, height, width), dtype=np.int64)
    trajectories[:, 0, :, :] = s0[np.newaxis, :, :]

    with torch.no_grad():
        # Current states for all K rollouts: (K, H, W) as class indices
        current = torch.tensor(s0, device=DEVICE).unsqueeze(0).expand(K, -1, -1).clone()

        for t in range(T):
            # One-hot encode current state: (K, 8, H, W)
            onehot = torch.zeros(K, NUM_TERRAIN, height, width, device=DEVICE)
            onehot.scatter_(1, current.unsqueeze(1), 1.0)

            # Predict next-step probabilities: (K, 8, H, W)
            probs = model.predict_probs(onehot)

            # Sample next state from categorical distribution
            # Reshape to (K*H*W, 8), sample, reshape back
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, NUM_TERRAIN)
            sampled = torch.multinomial(flat_probs, 1).squeeze(1)  # (K*H*W,)
            current = sampled.reshape(K, height, width)

            trajectories[:, t + 1, :, :] = current.cpu().numpy()

    return trajectories


# ---------------------------------------------------------------------------
# Observation Scoring — Score rollouts against viewport evidence
# ---------------------------------------------------------------------------

def aggregate_observations(observations, height, width):
    """
    Aggregate viewport observations into per-cell empirical class histograms.

    Since we observe terrain codes (8-class), we build histograms in 8-class space.

    Returns:
        obs_counts: (T_max, H, W, 8) float — observation counts per timestep
        obs_total:  (T_max, H, W) float — total observations per cell per timestep

    Note: viewport queries in this challenge always observe the FINAL state (t=50),
    not intermediate timesteps. All observations go into timestep T.
    """
    # All viewport observations are from the final state (year 50 simulation)
    T = DEFAULT_T_STEPS

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

    # Normalize to empirical probabilities where observed
    obs_probs = np.zeros_like(obs_counts)
    mask = obs_total > 0
    for c in range(NUM_TERRAIN):
        obs_probs[:, :, c][mask] = obs_counts[:, :, c][mask] / obs_total[mask]

    return obs_probs, obs_total


# Static terrain codes that never change — scoring these just adds noise
_STATIC_CODES = {CODE_TO_IDX[10], CODE_TO_IDX[11], CODE_TO_IDX[5]}  # ocean, plains, mountain


def score_rollouts(trajectories, obs_probs, obs_total, initial_grid_indices,
                   eps=1e-6):
    """
    Score each rollout against viewport observations.

    Only scores DYNAMIC cells (not ocean/plains/mountain) since static cells
    match trivially and just push all log-likelihoods in the same direction
    without differentiating rollouts.

    Log-weights are normalized by the number of scored cells so the scale
    doesn't depend on viewport coverage, preventing ESS collapse.

    Args:
        trajectories: (K, T+1, H, W) int64 — class indices
        obs_probs: (H, W, 8) float — empirical class probabilities at final step
        obs_total: (H, W) float — observation counts (>0 means observed)
        initial_grid_indices: (H, W) int64 — initial terrain class indices
        eps: small constant for numerical stability

    Returns:
        log_weights: (K,) float64 array of normalized log-likelihood scores
    """
    K = trajectories.shape[0]
    T = trajectories.shape[1] - 1

    # Only score observed AND dynamic cells
    observed = obs_total > 0  # (H, W)
    dynamic = np.ones_like(observed)
    for static_idx in _STATIC_CODES:
        dynamic &= (initial_grid_indices != static_idx)
    score_mask = observed & dynamic
    obs_ys, obs_xs = np.where(score_mask)
    n_scored = len(obs_ys)

    if n_scored == 0:
        return np.zeros(K, dtype=np.float64)

    # Final states for all rollouts: (K, H, W)
    final_states = trajectories[:, T, :, :]

    # Vectorized scoring: gather obs_probs for each rollout's predicted class
    # final_classes shape: (K, n_scored)
    final_classes = final_states[:, obs_ys, obs_xs]  # (K, n_scored)

    # obs_probs at scored cells: (n_scored, 8)
    scored_obs_probs = obs_probs[obs_ys, obs_xs, :]  # (n_scored, 8)

    # For each rollout, look up the probability of its predicted class
    log_weights = np.zeros(K, dtype=np.float64)
    for k in range(K):
        probs_k = scored_obs_probs[np.arange(n_scored), final_classes[k]]  # (n_scored,)
        log_weights[k] = np.log(probs_k + eps).sum()

    # Normalize by number of scored cells to keep scale independent of coverage
    log_weights /= n_scored

    return log_weights


def normalize_weights(log_weights):
    """Softmax normalization of log weights. Returns (K,) normalized weights."""
    log_weights = log_weights - log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    total = weights.sum()
    if total < 1e-12:
        return np.ones_like(weights) / len(weights)
    return weights / total


# ---------------------------------------------------------------------------
# Final Aggregation — Weighted final states → submission tensor
# ---------------------------------------------------------------------------

def aggregate_predictions(trajectories, weights, height, width):
    """
    Compute weighted class frequencies from final rollout states.

    Maps from 8-class terrain indices to 6-class submission space.

    Returns: (H, W, 6) probability tensor ready for submission.
    """
    K = trajectories.shape[0]
    T = trajectories.shape[1] - 1
    final_states = trajectories[:, T, :, :]  # (K, H, W)

    pred = np.zeros((height, width, SUBMIT_CLASSES), dtype=np.float64)

    for k in range(K):
        w = weights[k]
        for y in range(height):
            for x in range(width):
                terrain_idx = final_states[k, y, x]
                submit_cls = IDX8_TO_SUBMIT6[terrain_idx]
                pred[y, x, submit_cls] += w

    # Apply probability floor and renormalize
    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    return pred


def aggregate_predictions_fast(trajectories, weights, height, width):
    """
    Vectorized version of aggregate_predictions for speed.
    """
    K = trajectories.shape[0]
    T = trajectories.shape[1] - 1
    final_states = trajectories[:, T, :, :]  # (K, H, W)

    # Map 8-class to 6-class
    mapping = np.array(IDX8_TO_SUBMIT6, dtype=np.int64)
    final_submit = mapping[final_states]  # (K, H, W)

    pred = np.zeros((height, width, SUBMIT_CLASSES), dtype=np.float64)
    for k in range(K):
        # One-hot encode final state and weight
        for c in range(SUBMIT_CLASSES):
            pred[:, :, c] += weights[k] * (final_submit[k] == c).astype(np.float64)

    # Probability floor and renormalize
    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    return pred


# ---------------------------------------------------------------------------
# End-to-end inference
# ---------------------------------------------------------------------------

def predict_round(model, initial_grid, height, width, observations,
                  K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS):
    """
    Full Monte Carlo conditional prediction for one seed.

    1. Roll out K trajectories from initial map
    2. Score against viewport observations
    3. Weight and aggregate into submission tensor

    Returns: (H, W, 6) probability tensor
    """
    print(f"    Rolling out {K} trajectories for {T} steps...")
    t0 = time.time()
    trajectories = rollout_trajectories(model, initial_grid, height, width, K, T)
    print(f"    Rollouts done in {time.time()-t0:.1f}s")

    if observations:
        print(f"    Scoring against {len(observations)} viewport observations...")
        obs_probs, obs_total = aggregate_observations(observations, height, width)
        n_observed = int((obs_total > 0).sum())
        initial_indices = grid_to_class_indices(initial_grid, height, width)
        n_dynamic = int(np.sum(
            np.all([initial_indices != s for s in _STATIC_CODES], axis=0)
            & (obs_total > 0)))
        print(f"    Observed cells: {n_observed}/{height*width} "
              f"(dynamic: {n_dynamic})")

        log_weights = score_rollouts(trajectories, obs_probs, obs_total,
                                     initial_indices)
        weights = normalize_weights(log_weights)

        # Effective sample size (diagnostic)
        ess = 1.0 / (weights ** 2).sum()
        print(f"    Effective sample size: {ess:.1f}/{K}")
    else:
        print("    No observations — using uniform weights")
        weights = np.ones(K, dtype=np.float64) / K

    pred = aggregate_predictions_fast(trajectories, weights, height, width)
    return pred


# ---------------------------------------------------------------------------
# Evaluation — Compare MC predictions to ground truth
# ---------------------------------------------------------------------------

def entropy_per_pixel(target):
    """Shannon entropy per pixel. target: (H, W, 6)."""
    t = np.clip(target, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def kl_per_pixel(pred, target):
    """KL(target || pred) per pixel. Both (H, W, 6)."""
    pred = np.clip(pred, 1e-8, None)
    target = np.clip(target, 1e-8, None)
    return (target * (np.log(target) - np.log(pred))).sum(axis=-1)


def weighted_kl(pred, target):
    """Entropy-weighted KL divergence (competition metric)."""
    kl = kl_per_pixel(pred, target)
    ent = entropy_per_pixel(target)
    total_ent = ent.sum()
    if total_ent < 1e-12:
        return 0.0
    return (kl * ent).sum() / total_ent


def competition_score(pred, target):
    """Returns (score, weighted_kl)."""
    wkl = weighted_kl(pred, target)
    return 100.0 * np.exp(-3.0 * wkl), wkl


def evaluate_mc(replays, gt_data, K=DEFAULT_K_ROLLOUTS, T=DEFAULT_T_STEPS,
                train_epochs=DEFAULT_EPOCHS, use_obs=True):
    """
    Evaluate the MC approach using round k-fold:
    - For each round, train dynamics on other rounds
    - Roll out trajectories from each seed's initial map
    - Score against ground truth
    """
    # Group replays and GT by round
    replay_by_round = {}
    for r in replays:
        rid = r["round_id"][:8]
        replay_by_round.setdefault(rid, []).append(r)

    gt_by_round = {}
    for g in gt_data:
        rid = g.get("_round_id", "")[:8] if g.get("_round_id") else ""
        gt_by_round.setdefault(rid, []).append(g)

    # Find rounds that have both replays and GT
    common_rounds = sorted(set(replay_by_round.keys()) & set(gt_by_round.keys()))
    if len(common_rounds) < 2:
        print("ERROR: Need at least 2 rounds with both replays and GT for evaluation")
        return

    print(f"\n{'='*60}")
    print(f"  MC Conditional Evaluation — {len(common_rounds)}-fold Round CV")
    print(f"  Rounds: {common_rounds}")
    print(f"  K={K} rollouts, T={T} steps, obs_weighting={'ON' if use_obs else 'OFF'}")
    print(f"{'='*60}")

    all_scores = []
    all_wkls = []

    for fold_i, val_rid in enumerate(common_rounds):
        print(f"\n  === Fold {fold_i+1}/{len(common_rounds)}: holdout {val_rid} ===")

        # Train dynamics on other rounds
        train_replays = [r for r in replays if r["round_id"][:8] != val_rid]
        print(f"  Training dynamics on {len(train_replays)} replays...")
        X_train, Y_train, _ = build_transition_dataset(train_replays)

        # Quick training for this fold
        model = DynamicsCNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
        loss_fn = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_train).to(DEVICE)
        Y_t = torch.tensor(Y_train).to(DEVICE)
        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=DEFAULT_BATCH, shuffle=True)

        # Train
        t_start = time.time()
        for epoch in range(1, train_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_b = 0
            for X_b, Y_b in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(X_b), Y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            avg_loss = epoch_loss / max(n_b, 1)
            if epoch == 1 or epoch % 25 == 0 or epoch == train_epochs:
                elapsed = time.time() - t_start
                print(f"    Epoch {epoch:4d}/{train_epochs} | "
                      f"loss={avg_loss:.6f} | {elapsed:.0f}s")

        # Evaluate on held-out round's GT samples
        val_gts = gt_by_round[val_rid]
        # Also try to load observations for this round
        val_obs_all = load_observations(val_rid)

        fold_scores = []
        for gt_sample in val_gts:
            initial_grid = gt_sample.get("initial_grid")
            ground_truth = gt_sample.get("ground_truth")
            if initial_grid is None or ground_truth is None:
                continue

            seed = gt_sample.get("_seed_index", 0)
            height = gt_sample["height"]
            width = gt_sample["width"]
            gt_tensor = np.array(ground_truth, dtype=np.float32)  # (H, W, 6)

            # Get observations for this specific seed (empty if --no-obs)
            seed_obs = [o for o in val_obs_all if o.get("seed_index") == seed] if use_obs else []

            # MC prediction
            print(f"\n  Round {val_rid} seed {seed}:")
            pred = predict_round(model, initial_grid, height, width,
                                 seed_obs, K=K, T=T)

            score, wkl = competition_score(pred, gt_tensor)
            fold_scores.append(score)
            all_scores.append(score)
            all_wkls.append(wkl)
            print(f"    Score: {score:.2f} (wKL={wkl:.6f})")

        if fold_scores:
            print(f"  Fold {fold_i+1} mean score: {np.mean(fold_scores):.2f}")

    print(f"\n{'='*60}")
    print(f"  MC Conditional Evaluation Results")
    print(f"  Samples: {len(all_scores)}")
    print(f"  Scores: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
    print(f"  wKL:    {np.mean(all_wkls):.6f} ± {np.std(all_wkls):.6f}")
    print(f"  Min/Max score: {np.min(all_scores):.2f} / {np.max(all_scores):.2f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Astar Island — Monte Carlo Conditional Prediction")
    sub = parser.add_subparsers(dest="command")

    # Train command
    p_train = sub.add_parser("train", help="Train dynamics model")
    p_train.add_argument("--cv", choices=["all", "round_kfold"], default="all")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p_train.add_argument("--reset", action="store_true")

    # Evaluate command
    p_eval = sub.add_parser("evaluate", help="Evaluate MC pipeline with round k-fold")
    p_eval.add_argument("--rollouts", "-K", type=int, default=DEFAULT_K_ROLLOUTS)
    p_eval.add_argument("--steps", "-T", type=int, default=DEFAULT_T_STEPS)
    p_eval.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                         help="Epochs for per-fold dynamics training")
    p_eval.add_argument("--no-obs", action="store_true",
                         help="Skip observation weighting (pure dynamics model)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"Device: {DEVICE}")

    if args.command == "train":
        replays = load_all_replays()
        if not replays:
            return
        train_dynamics(replays, epochs=args.epochs, lr=args.lr,
                       batch_size=args.batch, cv=args.cv, reset=args.reset)

    elif args.command == "evaluate":
        replays = load_all_replays()
        gt_data = load_ground_truth()
        if not replays or not gt_data:
            return
        evaluate_mc(replays, gt_data, K=args.rollouts, T=args.steps,
                    train_epochs=args.epochs, use_obs=not args.no_obs)


if __name__ == "__main__":
    main()
