"""
Astar Island — Qualitative Analysis

Side-by-side comparison of predicted vs ground-truth probability tensors.

For each sample, produces a 2×6 grid:
  Left column:  Our prediction for each of the 6 classes
  Right column: Ground-truth probability for each of the 6 classes

Color encodes class identity; brightness encodes probability.

Usage:
    python qualitative_analysis.py --round 1 --seed 0
    python qualitative_analysis.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GT_DIR = os.path.join(SCRIPT_DIR, "data", "ground_truth")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Empty/Ocean/Plains",
    "Settlement",
    "Port",
    "Ruin",
    "Forest",
    "Mountain",
]

# Base hue colors for each class (used at full probability=1.0)
# At probability=0.0, color is black.
CLASS_BASE_COLORS = [
    np.array([0.831, 0.784, 0.604]),  # 0 Empty/Plains — sandy tan #d4c89a
    np.array([0.878, 0.251, 0.251]),  # 1 Settlement   — red       #e04040
    np.array([0.251, 0.502, 0.878]),  # 2 Port         — blue      #4080e0
    np.array([0.627, 0.322, 0.176]),  # 3 Ruin         — brown     #a0522d
    np.array([0.133, 0.545, 0.133]),  # 4 Forest       — green     #228b22
    np.array([0.502, 0.502, 0.502]),  # 5 Mountain     — grey      #808080
]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Load all GT files from disk. Returns list of dicts.
    Parses round number, seed index, and round_id from filename pattern:
        r{N}_s{M}_{id8}.json
    """
    import re
    all_gt = []
    if not os.path.isdir(GT_DIR):
        print(f"ERROR: Ground truth directory not found: {GT_DIR}")
        sys.exit(1)
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        # Parse metadata from filename: r1_s0_71451d74.json
        m = re.match(r"r(\d+)_s(\d+)_([a-f0-9]+)\.json", fname)
        if m:
            data["_round_number"] = int(m.group(1))
            data["_seed_index"] = int(m.group(2))
            data["_round_id"] = m.group(3)
        all_gt.append(data)
    return all_gt


def find_sample(gt_data, round_num, seed_idx):
    """Find a specific GT sample by round number and seed index."""
    for g in gt_data:
        if g.get("_round_number") == round_num and g.get("_seed_index") == seed_idx:
            return g
    return None


# ---------------------------------------------------------------------------
# Prediction generation: UNet Conditional
# ---------------------------------------------------------------------------

def predict_unet_cond(gt_sample, use_obs=True, temperature=1.0):
    """
    Generate (H, W, 6) prediction tensor using UNet conditional model.
    Matches eval_cnn.py pipeline: model inference → bayesian_blend.
    """
    from train_cnn import (
        encode_initial_grid, encode_obs_channels,
        load_model_from_checkpoint, get_checkpoint_dir,
        PROB_FLOOR,
    )
    from eval_cnn import load_observations_list
    from astar_cnn import bayesian_blend

    ckpt_dir = get_checkpoint_dir("unet_cond")
    # Prefer cnn_latest.pt, fallback to highest epoch
    ckpt_path = os.path.join(ckpt_dir, "cnn_latest.pt")
    if not os.path.isfile(ckpt_path):
        # Find highest epoch checkpoint
        import glob
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "cnn_epoch_*.pt")))
        if not ckpts:
            print(f"ERROR: No UNet checkpoints found in {ckpt_dir}")
            sys.exit(1)
        ckpt_path = ckpts[-1]

    print(f"  Loading UNet model from {os.path.basename(ckpt_path)}")
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    model.eval()

    H, W = gt_sample["height"], gt_sample["width"]
    initial_grid = gt_sample["initial_grid"]
    rid = gt_sample.get("_round_id", "")
    seed = gt_sample.get("_seed_index", 0)

    features = encode_initial_grid(initial_grid, W, H)  # (14, H, W)

    # Load observations
    seed_obs = []
    if use_obs and rid:
        all_obs = load_observations_list(rid)
        if all_obs:
            seed_obs = [o for o in all_obs if o.get("seed_index") == seed]
            if seed_obs:
                print(f"  Using {len(seed_obs)} viewport observations")
            else:
                print(f"  No observations found for this seed")

    obs_feat = encode_obs_channels(seed_obs, W, H)  # (7, H, W)
    x = np.concatenate([features, obs_feat], axis=0)  # (21, H, W)
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = model(x).squeeze(0)  # (6, H, W)
        if temperature != 1.0:
            logits = torch.log(torch.clamp(probs, min=1e-8))
            logits = logits / temperature
            probs = torch.softmax(logits, dim=0)
        pred = probs.permute(1, 2, 0).cpu().numpy()  # (H, W, 6)

    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    # Bayesian blend with observations (matches eval_cnn.py pipeline)
    # if seed_obs:
    #     pred = bayesian_blend(pred, seed_obs, initial_grid, W, H)

    # Mountains are static — hard-set to 1.0 for mountain class
    for y in range(H):
        for x_i in range(W):
            if initial_grid[y][x_i] == 5:  # mountain
                pred[y, x_i, :] = PROB_FLOOR
                pred[y, x_i, 5] = 1.0 - (PROB_FLOOR * 5)

    return pred  # (H, W, 6)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def prob_to_rgb(prob_map, base_color):
    """
    Convert (H, W) probability map to (H, W, 3) RGB image.
    probability=0 → black, probability=1 → full base_color.
    """
    H, W = prob_map.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = prob_map * base_color[c]
    return np.clip(img, 0, 1)


def plot_comparison(pred, gt, round_num, seed_idx, method_name, save_path=None):
    """
    Plot a 2×6 grid: top row = prediction, bottom row = ground truth.
    Each column is one class, colored by class identity, brightness by probability.
    """
    fig, axes = plt.subplots(2, 6, figsize=(20, 7.5))
    fig.suptitle(
        f"Round {round_num}, Seed {seed_idx} — {method_name}\n"
        f"Top: Prediction | Bottom: Ground Truth",
        fontsize=14, fontweight="bold", y=1.0,
    )

    for cls_idx in range(6):
        pred_probs = pred[:, :, cls_idx]   # (H, W)
        gt_probs = gt[:, :, cls_idx]       # (H, W)
        base_color = CLASS_BASE_COLORS[cls_idx]

        # Prediction (top row)
        ax_pred = axes[0, cls_idx]
        img_pred = prob_to_rgb(pred_probs, base_color)
        ax_pred.imshow(img_pred, interpolation="nearest")
        ax_pred.set_title(f"Pred: {CLASS_NAMES[cls_idx]}", fontsize=9)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

        # Ground truth (bottom row)
        ax_gt = axes[1, cls_idx]
        img_gt = prob_to_rgb(gt_probs, base_color)
        ax_gt.imshow(img_gt, interpolation="nearest")
        ax_gt.set_title(f"GT: {CLASS_NAMES[cls_idx]}", fontsize=9)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_score(pred, gt):
    """Competition score: 100 * exp(-3 * wKL)."""
    pred_c = np.clip(pred, 1e-8, None)
    gt_c = np.clip(gt, 1e-8, None)
    kl = (gt_c * (np.log(gt_c) - np.log(pred_c))).sum(axis=-1)    # (H, W)
    ent = -(gt_c * np.log(gt_c)).sum(axis=-1)                     # (H, W)
    total_ent = ent.sum()
    if total_ent < 1e-12:
        return 100.0, 0.0
    wkl = (kl * ent).sum() / total_ent
    return 100.0 * np.exp(-3.0 * wkl), wkl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative analysis: prediction vs ground truth")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number to visualize")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed index to visualize")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all available GT samples")
    parser.add_argument("--no-obs", action="store_true",
                        help="Disable observation weighting")
    parser.add_argument("--temperature", "-t", type=float, default=1.0,
                        help="Softmax temperature (<1 = sharper, >1 = smoother)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save plots to this directory instead of showing")
    args = parser.parse_args()

    if not args.all and args.round is None:
        parser.error("Specify --round and --seed, or use --all")

    gt_data = load_ground_truth()
    if not gt_data:
        print("ERROR: No ground truth files found.")
        sys.exit(1)
    print(f"Loaded {len(gt_data)} ground truth samples")

    use_obs = not args.no_obs
    temperature = args.temperature
    method_label = "UNet Cond"
    if temperature != 1.0:
        method_label += f" (T={temperature})"

    # Determine which samples to process
    if args.all:
        samples = [g for g in gt_data
                   if g.get("initial_grid") and g.get("ground_truth")]
    else:
        seed = args.seed if args.seed is not None else 0
        sample = find_sample(gt_data, args.round, seed)
        if sample is None:
            print(f"ERROR: No GT found for round {args.round}, seed {seed}")
            available = [(g.get("_round_number"), g.get("_seed_index"))
                         for g in gt_data]
            print(f"Available: {available}")
            sys.exit(1)
        samples = [sample]

    # Save directory
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Process each sample
    for i, gt_sample in enumerate(samples):
        rnum = gt_sample.get("_round_number", "?")
        seed = gt_sample.get("_seed_index", "?")
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(samples)}] Round {rnum}, Seed {seed} — {method_label}")
        print(f"{'='*60}")

        gt_tensor = np.array(gt_sample["ground_truth"], dtype=np.float32)

        pred = predict_unet_cond(gt_sample, use_obs=use_obs, temperature=temperature)

        score, wkl = compute_score(pred, gt_tensor)
        print(f"  Score: {score:.2f} | wKL: {wkl:.6f}")

        save_path = None
        if save_dir:
            save_path = os.path.join(
                save_dir, f"unet_cond_r{rnum}_s{seed}.png")

        plot_comparison(pred, gt_tensor, rnum, seed, method_label,
                        save_path=save_path)


if __name__ == "__main__":
    main()
