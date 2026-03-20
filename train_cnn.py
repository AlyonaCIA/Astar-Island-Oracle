"""
Astar Island — Offline CNN Training Script

Workflow:
1. Fetch completed rounds and download ground truth from /analysis endpoint
2. Cache all data locally in data/ground_truth/ (skip re-download on subsequent runs)
3. Encode initial grids → 14-channel feature tensors
4. Split each map into 4 quadrants: 3 for training, 1 for validation
5. Train QuickCNN with checkpointing and resume support
6. Save final checkpoint with metadata
"""

import os
import sys
import time
import json
import math
import argparse
import datetime
import functools
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("ASTAR_TOKEN")
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "ground_truth")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
NUM_CLASSES = 6
PROB_FLOOR = 0.01
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyper-parameters (overridable via env)
EPOCHS = int(os.environ.get("ASTAR_TRAIN_EPOCHS", "300"))
LR = float(os.environ.get("ASTAR_TRAIN_LR", "1e-3"))
BATCH_SIZE = int(os.environ.get("ASTAR_TRAIN_BATCH", "64"))
CHECKPOINT_EVERY = int(os.environ.get("ASTAR_CKPT_EVERY", "25"))  # epochs
VAL_QUADRANT = int(os.environ.get("ASTAR_VAL_QUADRANT", "3"))  # 0-3, which quadrant is val

if not TOKEN:
    print("ERROR: Set ASTAR_TOKEN in .env file or as environment variable.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_rounds():
    resp = session.get(f"{BASE_URL}/rounds")
    resp.raise_for_status()
    return resp.json()


def get_analysis(round_id, seed_index):
    resp = session.get(f"{BASE_URL}/analysis/{round_id}/{seed_index}")
    resp.raise_for_status()
    time.sleep(1.0)
    return resp.json()


# ---------------------------------------------------------------------------
# Data loading (local) & fetching (remote)
# ---------------------------------------------------------------------------

def load_local_data():
    """
    Load ground truth from cached JSON files in DATA_DIR.
    No API calls — works fully offline.
    """
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("  Run with --fetch first to download ground truth.")
        return []

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    if not files:
        print(f"No JSON files found in {DATA_DIR}")
        print("  Run with --fetch first to download ground truth.")
        return []

    all_data = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        # Parse metadata from filename if not embedded (rN_sM_XXXX.json)
        if "_round_number" not in data:
            parts = fname.replace(".json", "").split("_")
            if len(parts) >= 3:
                data["_round_number"] = int(parts[0][1:]) if parts[0][0] == "r" else None
                data["_seed_index"] = int(parts[1][1:]) if parts[1][0] == "s" else None
                data["_round_id"] = parts[2]

        all_data.append(data)

    print(f"Loaded {len(all_data)} samples from {DATA_DIR}")
    return all_data


# ---------------------------------------------------------------------------
# Data fetching & caching
# ---------------------------------------------------------------------------

def fetch_ground_truth():
    """
    Download ground truth for all completed/scoring rounds.
    Caches each seed's analysis as a JSON file in DATA_DIR.
    Includes rate-limit safety with 1s sleep between API calls.
    Returns list of loaded analysis dicts.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    time.sleep(1.0)
    rounds = get_rounds()
    eligible = [r for r in rounds if r["status"] in ("completed", "scoring")]

    if not eligible:
        print("No completed/scoring rounds found. Available rounds:")
        for r in rounds:
            print(f"  Round {r['round_number']} — status: {r['status']}")
        sys.exit(0)

    all_data = []
    for rnd in eligible:
        round_id = rnd["id"]
        round_num = rnd["round_number"]
        seeds_count = rnd.get("seeds_count", 5)
        print(f"Round #{round_num} ({round_id[:8]}…) — {seeds_count} seeds")

        for seed_idx in range(seeds_count):
            cache_file = os.path.join(DATA_DIR, f"r{round_num}_s{seed_idx}_{round_id[:8]}.json")

            if os.path.exists(cache_file):
                print(f"  Seed {seed_idx}: cached ✓")
                with open(cache_file) as f:
                    data = json.load(f)
            else:
                print(f"  Seed {seed_idx}: downloading…", end=" ", flush=True)
                try:
                    data = get_analysis(round_id, seed_idx)
                    with open(cache_file, "w") as f:
                        json.dump(data, f)
                    print(f"score={data.get('score', '?')}")
                    time.sleep(1.0)
                except requests.HTTPError as e:
                    print(f"FAILED ({e})")
                    continue

            data["_round_id"] = round_id
            data["_round_number"] = round_num
            data["_seed_index"] = seed_idx
            all_data.append(data)

    print(f"\nTotal ground truth samples: {len(all_data)}")
    return all_data


def fetch_latest_round():
    """
    Download ground truth for only the most recent completed/scoring round,
    then load ALL local data (including previously cached rounds).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    time.sleep(1.0)
    rounds = get_rounds()
    eligible = [r for r in rounds if r["status"] in ("completed", "scoring")]

    if not eligible:
        print("No completed/scoring rounds found.")
        return load_local_data()

    # Pick the latest by round_number
    latest = max(eligible, key=lambda r: r["round_number"])
    round_id = latest["id"]
    round_num = latest["round_number"]
    seeds_count = latest.get("seeds_count", 5)
    print(f"Fetching latest: Round #{round_num} ({round_id[:8]}…) — {seeds_count} seeds")

    for seed_idx in range(seeds_count):
        cache_file = os.path.join(DATA_DIR, f"r{round_num}_s{seed_idx}_{round_id[:8]}.json")
        if os.path.exists(cache_file):
            print(f"  Seed {seed_idx}: cached ✓")
        else:
            print(f"  Seed {seed_idx}: downloading…", end=" ", flush=True)
            try:
                data = get_analysis(round_id, seed_idx)
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                print(f"score={data.get('score', '?')}")
                time.sleep(1.0)
            except requests.HTTPError as e:
                print(f"FAILED ({e})")

    # Now load everything from disk (all rounds)
    return load_local_data()


# ---------------------------------------------------------------------------
# Feature encoding (same as astar_cnn.py)
# ---------------------------------------------------------------------------

def terrain_to_class(cell_value):
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(cell_value, 0)


def encode_initial_grid(initial_grid, width, height):
    features = np.zeros((14, height, width), dtype=np.float32)
    code_to_channel = {code: i for i, code in enumerate(TERRAIN_CODES)}
    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            ch = code_to_channel.get(cell, 0)
            features[ch, y, x] = 1.0

    class_grid = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            class_grid[y, x] = terrain_to_class(initial_grid[y][x])

    for y in range(height):
        for x in range(width):
            counts = np.zeros(NUM_CLASSES, dtype=np.float32)
            n = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        counts[class_grid[ny, nx]] += 1.0
                        n += 1
            if n > 0:
                counts /= n
            features[8:14, y, x] = counts
    return features


# ---------------------------------------------------------------------------
# CNN models
# ---------------------------------------------------------------------------

class QuickCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.out_conv = nn.Conv2d(32, 6, kernel_size=1)

    def forward(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        logits = self.out_conv(x)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class QuickCNN3(nn.Module):
    """QuickCNN with one additional hidden conv layer (receptive field 7x7)."""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.out_conv = nn.Conv2d(32, 6, kernel_size=1)

    def forward(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        x = self.drop3(F.relu(self.conv3(x)))
        logits = self.out_conv(x)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class MiniUNet(nn.Module):
    """Small U-Net for 40x40 maps. Encoder-decoder with skip connections."""
    def __init__(self, dropout=0.2):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(14, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 40 -> 20
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)  # 20 -> 10
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 10 -> 20
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 20 -> 40
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(32, 6, kernel_size=1)

    def forward(self, x):
        # Pad to even dimensions if needed
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.out_conv(d1)

        # Crop back to original size
        if pad_h or pad_w:
            logits = logits[:, :, :H, :W]

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


# Registry for model selection
MODEL_REGISTRY = {
    "quick": QuickCNN,
    "quick3": QuickCNN3,
    "unet": MiniUNet,
    "unet_aug": functools.partial(MiniUNet, dropout=0.1),  # lower dropout for augmented training
}

# Separate checkpoint dirs per architecture
CHECKPOINT_DIR_MAP = {
    "quick": os.path.join(SCRIPT_DIR, "checkpoints"),
    "quick3": os.path.join(SCRIPT_DIR, "checkpoints_quick3"),
    "unet": os.path.join(SCRIPT_DIR, "checkpoints_unet"),
    "unet_aug": os.path.join(SCRIPT_DIR, "checkpoints_unet_aug"),
}


def make_model(arch="quick", **kwargs):
    """Instantiate a model by architecture name."""
    cls = MODEL_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY)}")
    return cls(**kwargs)


def get_checkpoint_dir(arch="quick"):
    """Return the checkpoint directory for the given architecture."""
    return CHECKPOINT_DIR_MAP.get(arch, CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Quadrant split
# ---------------------------------------------------------------------------

def quadrant_masks(height, width, val_quadrant=3):
    """
    Divide H×W into 4 quadrants (2×2 grid). Returns (train_mask, val_mask)
    as boolean arrays of shape (H, W).

    Quadrant layout:
      0 | 1
      -----
      2 | 3

    val_quadrant: which one is held out for validation (default: 3 = bottom-right)
    """
    mid_y = height // 2
    mid_x = width // 2

    val_mask = np.zeros((height, width), dtype=bool)
    if val_quadrant == 0:
        val_mask[:mid_y, :mid_x] = True
    elif val_quadrant == 1:
        val_mask[:mid_y, mid_x:] = True
    elif val_quadrant == 2:
        val_mask[mid_y:, :mid_x] = True
    else:
        val_mask[mid_y:, mid_x:] = True

    train_mask = ~val_mask
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------------

def build_datasets(all_data, val_quadrant=3):
    """
    For each sample: features = encoded initial_grid (14, H, W),
                     target = ground_truth probability tensor (H, W, 6).

    Returns full-map tensors split into train/val pixel sets.
    """
    train_X, train_Y = [], []
    val_X, val_Y = [], []

    skipped = 0
    for data in all_data:
        initial_grid = data.get("initial_grid")
        ground_truth = data.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            skipped += 1
            continue

        width = data["width"]
        height = data["height"]
        features = encode_initial_grid(initial_grid, width, height)  # (14, H, W)
        gt = np.array(ground_truth, dtype=np.float32)               # (H, W, 6)

        train_mask, val_mask = quadrant_masks(height, width, val_quadrant)

        # Extract pixel-level samples: input=(14,), target=(6,)
        for y in range(height):
            for x in range(width):
                feat = features[:, y, x]  # (14,)
                target = gt[y, x]         # (6,)
                if train_mask[y, x]:
                    train_X.append(feat)
                    train_Y.append(target)
                else:
                    val_X.append(feat)
                    val_Y.append(target)

    if skipped:
        print(f"  Skipped {skipped} samples with missing initial_grid or ground_truth")

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.float32)
    val_X = np.array(val_X, dtype=np.float32)
    val_Y = np.array(val_Y, dtype=np.float32)

    print(f"  Train pixels: {len(train_X)}, Val pixels: {len(val_X)}")
    return train_X, train_Y, val_X, val_Y


def build_fullmap_datasets(all_data, val_quadrant=3):
    """
    Build full-map tensors for CNN training (spatial context preserved).

    Returns:
      features_list: list of (14, H, W) numpy arrays
      targets_list:  list of (6, H, W) numpy arrays  (channels-first for loss)
      train_masks:   list of (H, W) bool arrays
      val_masks:     list of (H, W) bool arrays
      meta:          list of dicts with round/seed info
    """
    features_list, targets_list = [], []
    train_masks, val_masks = [], []
    meta = []

    for data in all_data:
        initial_grid = data.get("initial_grid")
        ground_truth = data.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            continue

        width = data["width"]
        height = data["height"]
        features = encode_initial_grid(initial_grid, width, height)  # (14, H, W)
        gt = np.array(ground_truth, dtype=np.float32)               # (H, W, 6)
        gt = gt.transpose(2, 0, 1)                                  # (6, H, W)

        tmask, vmask = quadrant_masks(height, width, val_quadrant)

        features_list.append(features)
        targets_list.append(gt)
        train_masks.append(tmask)
        val_masks.append(vmask)
        meta.append({
            "round": data.get("_round_number"),
            "seed": data.get("_seed_index"),
            "score": data.get("score"),
        })

    print(f"  Full maps: {len(features_list)} (val quadrant={val_quadrant})")
    return features_list, targets_list, train_masks, val_masks, meta


def augment_maps(features_list, targets_list, meta):
    """
    Augment full-map tensors with rotations (90°, 180°, 270°) and horizontal
    flips.  Each original map produces 8 variants (4 rotations × 2 flip states).

    Input/output shapes per element: features (14, H, W), targets (6, H, W).
    np.rot90 operates on the last two axes.
    """
    aug_features, aug_targets, aug_meta = [], [], []
    for feat, tgt, m in zip(features_list, targets_list, meta):
        for k in range(4):  # 0°, 90°, 180°, 270°
            rf = np.rot90(feat, k=k, axes=(1, 2)).copy()
            rt = np.rot90(tgt, k=k, axes=(1, 2)).copy()
            aug_features.append(rf)
            aug_targets.append(rt)
            aug_meta.append({**m, "aug": f"rot{k*90}"})
            # horizontal flip
            aug_features.append(np.flip(rf, axis=2).copy())
            aug_targets.append(np.flip(rt, axis=2).copy())
            aug_meta.append({**m, "aug": f"rot{k*90}_fliph"})
    print(f"  Augmented: {len(features_list)} → {len(aug_features)} maps "
          f"(4 rotations × 2 flip states)")
    return aug_features, aug_targets, aug_meta


# ---------------------------------------------------------------------------
# Loss: KL divergence (same metric used for scoring)
# ---------------------------------------------------------------------------

_kl_loss_fn = nn.KLDivLoss(reduction="batchmean")


def kl_divergence_loss(pred_probs, target_probs, mask=None):
    """
    Compute mean KL(target || pred) per pixel using nn.KLDivLoss.
    pred_probs, target_probs: (B, 6, H, W)
    mask: optional (B, H, W) bool tensor — only compute loss where True
    """
    pred = torch.clamp(pred_probs, min=1e-8)
    target = torch.clamp(target_probs, min=1e-8)

    if mask is not None:
        # Gather masked pixels: (M, 6)
        pred_m = pred.permute(0, 2, 3, 1)[mask]    # (M, 6)
        target_m = target.permute(0, 2, 3, 1)[mask]  # (M, 6)
        return _kl_loss_fn(pred_m.log(), target_m)
    else:
        # Reshape to (B*H*W, 6)
        B, C, H, W = pred.shape
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)
        return _kl_loss_fn(pred_flat.log(), target_flat)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint(ckpt_dir=None):
    """Find the latest checkpoint file in the given checkpoint directory."""
    d = ckpt_dir or CHECKPOINT_DIR
    os.makedirs(d, exist_ok=True)
    files = [f for f in os.listdir(d) if f.startswith("cnn_epoch_") and f.endswith(".pt")]
    if not files:
        return None
    # Parse epoch number
    def epoch_num(f):
        try:
            return int(f.replace("cnn_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1
    files.sort(key=epoch_num)
    return os.path.join(d, files[-1])


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, metadata,
                    ckpt_dir=None, arch=None):
    d = ckpt_dir or CHECKPOINT_DIR
    os.makedirs(d, exist_ok=True)
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "metadata": metadata,
    }
    if arch:
        save_dict["model_arch"] = arch
    path = os.path.join(d, f"cnn_epoch_{epoch:04d}.pt")
    torch.save(save_dict, path)
    return path


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def load_model_from_checkpoint(path):
    """Load a checkpoint, auto-detect architecture, return (model, ckpt dict)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("model_arch") or ckpt.get("metadata", {}).get("model_arch", "quick")
    model = make_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def _clear_checkpoints(ckpt_dir=None):
    """Delete all existing checkpoints and training history."""
    d = ckpt_dir or CHECKPOINT_DIR
    if not os.path.isdir(d):
        return
    removed = 0
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    if removed:
        print(f"  Cleared {removed} files from {d}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(all_data, reset=False, forever=False, arch="quick"):
    ckpt_dir = get_checkpoint_dir(arch)

    print(f"\n{'='*60}")
    print(f"  Building datasets (4-fold quadrant cross-validation)")
    print(f"  Architecture: {arch}")
    print(f"{'='*60}")

    features_list, targets_list, _, _, meta = \
        build_fullmap_datasets(all_data, val_quadrant=0)

    if not features_list:
        print("ERROR: No usable data found.")
        return

    # Apply rotation/flip augmentation for unet_aug
    if arch == "unet_aug":
        features_list, targets_list, meta = augment_maps(
            features_list, targets_list, meta)

    # Convert to tensors
    X = torch.tensor(np.stack(features_list)).to(DEVICE)       # (N, 14, H, W)
    Y = torch.tensor(np.stack(targets_list)).to(DEVICE)        # (N, 6, H, W)

    N, _, H, W = X.shape

    # Pre-build (H, W) masks for all 4 folds
    fold_masks = []  # list of (train_hw, val_hw) tuples
    for q in range(4):
        t_np, v_np = quadrant_masks(H, W, val_quadrant=q)
        fold_masks.append((
            torch.tensor(t_np, device=DEVICE),
            torch.tensor(v_np, device=DEVICE),
        ))

    pix_per_q = (H // 2) * (W // 2) * N
    print(f"  Tensors: X={list(X.shape)}, Y={list(Y.shape)}")
    print(f"  4-fold CV: {pix_per_q * 3} train / {pix_per_q} val pixels per fold")

    # Model & optimizer
    model = make_model(arch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1
    best_val = float("inf")
    training_start = time.time()

    # Resume from checkpoint if available (unless --reset)
    ckpt_path = None if reset else latest_checkpoint(ckpt_dir)
    if reset:
        _clear_checkpoints(ckpt_dir)
        print(f"\n  Training from scratch (--reset)")
    elif ckpt_path:
        print(f"\n  Resuming from {os.path.basename(ckpt_path)}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, prev val_loss={best_val:.6f}")
    else:
        print(f"\n  Training from scratch")

    max_label = "∞ (Ctrl+C to stop)" if forever else str(EPOCHS)
    print(f"  Epochs: {start_epoch}→{max_label}, LR={LR}, Device={DEVICE}")
    print(f"  Checkpoint every {CHECKPOINT_EVERY} epochs → {ckpt_dir}")
    print()

    # Dataset / DataLoader for map-level batching (masks applied per-fold)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    metadata = {
        "model_arch": arch,
        "num_maps": N,
        "map_size": f"{H}x{W}",
        "cross_validation": "4-fold quadrant",
        "rounds": [m["round"] for m in meta],
        "lr": LR,
        "start_time": datetime.datetime.now().isoformat(),
    }

    # Training history log (append-friendly)
    os.makedirs(ckpt_dir, exist_ok=True)
    history_path = os.path.join(ckpt_dir, "training_history.json")
    history = []
    if not reset and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    def _epoch_iter():
        """Yield epoch numbers: finite range or infinite counter."""
        if forever:
            epoch = start_epoch
            while True:
                yield epoch
                epoch += 1
        else:
            yield from range(start_epoch, EPOCHS + 1)

    last_epoch = start_epoch
    avg_train = 0.0
    val_loss = float("inf")

    try:
        for epoch in _epoch_iter():
            last_epoch = epoch
            model.train()
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            total_train_batches = 0

            # --- 4-fold cross-validation ---
            for t_hw, v_hw in fold_masks:
                # Train on 3 quadrants
                for X_b, Y_b in loader:
                    B = X_b.shape[0]
                    Tm_b = t_hw.unsqueeze(0).expand(B, -1, -1)
                    optimizer.zero_grad()
                    pred = model(X_b)  # (B, 6, H, W)
                    loss = kl_divergence_loss(pred, Y_b, mask=Tm_b)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    total_train_batches += 1

                # Validate on held-out quadrant (dropout stays active)
                with torch.no_grad():
                    V_full = v_hw.unsqueeze(0).expand(N, -1, -1)
                    pred_all = model(X)
                    epoch_val_loss += kl_divergence_loss(pred_all, Y, mask=V_full).item()

            avg_train = epoch_train_loss / max(total_train_batches, 1)
            val_loss = epoch_val_loss / 4

            if val_loss < best_val:
                best_val = val_loss
                best_marker = " ★"
            else:
                best_marker = ""

            # Print every epoch for visibility
            elapsed = time.time() - training_start
            max_str = "∞" if forever else str(EPOCHS)
            print(f"  Epoch {epoch:4d}/{max_str} | train_kl={avg_train:.6f} | "
                  f"val_kl={val_loss:.6f}{best_marker} | {elapsed:.0f}s")

            # Log to history
            history.append({
                "epoch": epoch,
                "train_kl": avg_train,
                "val_kl": val_loss,
                "best_val_kl": best_val,
                "elapsed_s": round(elapsed, 1),
                "timestamp": datetime.datetime.now().isoformat(),
            })

            # Checkpoint
            should_ckpt = (epoch % CHECKPOINT_EVERY == 0) or (not forever and epoch == EPOCHS)
            if should_ckpt:
                metadata["end_time"] = datetime.datetime.now().isoformat()
                metadata["total_epochs"] = epoch
                metadata["best_val_loss"] = best_val
                path = save_checkpoint(model, optimizer, epoch, avg_train, val_loss,
                                       metadata, ckpt_dir=ckpt_dir, arch=arch)
                print(f"    → Saved {os.path.basename(path)}")
                # Flush history to disk at each checkpoint
                with open(history_path, "w") as f:
                    json.dump(history, f)
                print(f"    → Saved training_history.json ({len(history)} entries)")

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted at epoch {last_epoch}. Saving checkpoint...")

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "cnn_latest.pt")
    metadata["end_time"] = datetime.datetime.now().isoformat()
    metadata["total_epochs"] = last_epoch
    metadata["best_val_loss"] = best_val
    metadata["total_training_time_s"] = time.time() - training_start
    save_dict = {
        "epoch": last_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train,
        "val_loss": val_loss,
        "metadata": metadata,
        "model_arch": arch,
    }
    torch.save(save_dict, final_path)
    # Final history flush
    with open(history_path, "w") as f:
        json.dump(history, f)

    print(f"\n  Final checkpoint: {final_path}")
    print(f"  Best val KL: {best_val:.6f}")
    print(f"  History: {history_path} ({len(history)} entries)")
    print(f"  Total time: {time.time() - training_start:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Astar Island — Offline CNN Training")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore existing checkpoints and train from scratch")
    parser.add_argument("--forever", action="store_true",
                        help="Train indefinitely until Ctrl+C (ignores ASTAR_TRAIN_EPOCHS)")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch/update ground truth from API (default: use local cache only)")
    parser.add_argument("--fetch-latest", action="store_true",
                        help="Fetch only the latest round, then load all cached data")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="quick",
                        help="Model architecture to train (default: quick)")
    args = parser.parse_args()

    arch = args.model
    ckpt_dir = get_checkpoint_dir(arch)

    print("=" * 60)
    print("  Astar Island — Offline CNN Training")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Architecture: {arch}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    if args.reset:
        print("  Mode: RESET (training from scratch)")
    if args.forever:
        print("  Mode: FOREVER (Ctrl+C to stop)")

    if args.fetch:
        print(f"\n--- Fetching ground truth from API ---")
        all_data = fetch_ground_truth()
    elif args.fetch_latest:
        print(f"\n--- Fetching latest round from API ---")
        all_data = fetch_latest_round()
    else:
        print(f"\n--- Loading local ground truth ---")
        all_data = load_local_data()

    if not all_data:
        print("No data to train on.")
        return

    train(all_data, reset=args.reset, forever=args.forever, arch=arch)


if __name__ == "__main__":
    main()
