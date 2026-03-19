"""
Astar Island — CNN Checkpoint Evaluation

Evaluates a trained checkpoint on the validation quadrant using KL divergence
(the same metric used for competition scoring).

Produces:
- Per-seed KL divergence
- Per-class KL breakdown
- Simulated competition score: 100 * exp(-3 * weighted_kl)
- Comparison with a uniform (1/6) baseline and prior-based baseline
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reuse shared code from train_cnn (same dir)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import (
    _load_dotenv, QuickCNN, encode_initial_grid, terrain_to_class,
    quadrant_masks, load_local_data, load_checkpoint,
    CHECKPOINT_DIR, DEVICE, NUM_CLASSES, PROB_FLOOR, VAL_QUADRANT,
)

_load_dotenv()


# ---------------------------------------------------------------------------
# Prior-based baseline (same as astar_cnn.py / astar_baseline.py)
# ---------------------------------------------------------------------------

def build_prior_prediction(initial_grid, width, height):
    pred = np.full((height, width, NUM_CLASSES), PROB_FLOOR, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            if cell == 10:
                pred[y][x][0] = 0.95
            elif cell == 5:
                pred[y][x][5] = 0.95
            elif cell == 4:
                pred[y][x] = [0.06, 0.04, 0.01, 0.03, 0.84, 0.02]
            elif cell == 1:
                pred[y][x] = [0.13, 0.40, 0.02, 0.25, 0.10, 0.10]
            elif cell == 2:
                pred[y][x] = [0.12, 0.15, 0.35, 0.22, 0.05, 0.11]
            elif cell in (0, 11):
                pred[y][x] = [0.58, 0.12, 0.02, 0.05, 0.18, 0.05]
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def kl_per_pixel(pred, target):
    """
    KL(target || pred) per pixel.
    pred, target: (H, W, 6) numpy arrays (probabilities)
    Returns: (H, W) array of per-pixel KL values
    """
    pred = np.clip(pred, 1e-8, None)
    target = np.clip(target, 1e-8, None)
    return (target * (np.log(target) - np.log(pred))).sum(axis=-1)


def entropy_per_pixel(target):
    """Shannon entropy per pixel. target: (H, W, 6)"""
    t = np.clip(target, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def weighted_kl(pred, target):
    """
    Entropy-weighted KL divergence (competition metric).
    Higher entropy pixels (more uncertain) count more.
    """
    kl = kl_per_pixel(pred, target)          # (H, W)
    ent = entropy_per_pixel(target)          # (H, W)
    total_entropy = ent.sum()
    if total_entropy < 1e-12:
        return 0.0
    return (kl * ent).sum() / total_entropy


def competition_score(pred, target):
    """100 * exp(-3 * weighted_kl)"""
    wkl = weighted_kl(pred, target)
    return 100.0 * np.exp(-3.0 * wkl), wkl


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def find_checkpoint(path_arg=None):
    """Resolve which checkpoint to evaluate."""
    if path_arg and os.path.isfile(path_arg):
        return path_arg
    # Default: cnn_latest.pt
    latest = os.path.join(CHECKPOINT_DIR, "cnn_latest.pt")
    if os.path.isfile(latest):
        return latest
    # Fallback: highest epoch checkpoint
    from train_cnn import latest_checkpoint
    ckpt = latest_checkpoint()
    if ckpt:
        return ckpt
    return None


def evaluate(ckpt_path, all_data, val_quadrant):
    model = QuickCNN().to(DEVICE)
    ckpt = load_checkpoint(ckpt_path, model)
    epoch = ckpt.get("epoch", "?")
    meta = ckpt.get("metadata", {})
    print(f"  Epoch: {epoch}")
    print(f"  Train loss: {ckpt.get('train_loss', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?')}")
    if meta:
        print(f"  Trained on rounds: {meta.get('rounds', '?')}")
        print(f"  Training time: {meta.get('total_training_time_s', '?')}s")

    model.eval()

    CLASS_NAMES = ["Ocean", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

    results = []

    for data in all_data:
        initial_grid = data.get("initial_grid")
        ground_truth = data.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            continue

        width = data["width"]
        height = data["height"]
        rnd = data.get("_round_number", "?")
        seed = data.get("_seed_index", "?")
        api_score = data.get("score", None)

        gt = np.array(ground_truth, dtype=np.float32)  # (H, W, 6)
        features = encode_initial_grid(initial_grid, width, height)  # (14, H, W)

        _, val_mask = quadrant_masks(height, width, val_quadrant)

        # --- CNN prediction ---
        with torch.no_grad():
            x = torch.tensor(features).unsqueeze(0).to(DEVICE)  # (1, 14, H, W)
            probs = model(x).squeeze(0)  # (6, H, W)
            cnn_pred = probs.permute(1, 2, 0).cpu().numpy()  # (H, W, 6)
        cnn_pred = np.maximum(cnn_pred, PROB_FLOOR)
        cnn_pred = cnn_pred / cnn_pred.sum(axis=-1, keepdims=True)

        # --- Baselines ---
        uniform_pred = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
        prior_pred = build_prior_prediction(initial_grid, width, height)

        # --- Full-map metrics ---
        cnn_score_full, cnn_wkl_full = competition_score(cnn_pred, gt)
        uni_score_full, uni_wkl_full = competition_score(uniform_pred, gt)
        pri_score_full, pri_wkl_full = competition_score(prior_pred, gt)

        # --- Val-quadrant-only metrics ---
        gt_val = gt[val_mask]                    # (V, 6)
        cnn_val = cnn_pred[val_mask]             # (V, 6)
        uni_val = uniform_pred[val_mask]
        pri_val = prior_pred[val_mask]

        # Per-pixel KL on val
        cnn_kl_val = kl_per_pixel(cnn_val.reshape(-1, 1, NUM_CLASSES),
                                  gt_val.reshape(-1, 1, NUM_CLASSES)).mean()
        uni_kl_val = kl_per_pixel(uni_val.reshape(-1, 1, NUM_CLASSES),
                                  gt_val.reshape(-1, 1, NUM_CLASSES)).mean()
        pri_kl_val = kl_per_pixel(pri_val.reshape(-1, 1, NUM_CLASSES),
                                  gt_val.reshape(-1, 1, NUM_CLASSES)).mean()

        # Per-class KL breakdown on val quadrant
        cnn_class_kl = []
        for c in range(NUM_CLASSES):
            # Pixels where ground truth says this class is dominant
            dominant = gt_val.argmax(axis=-1) == c
            if dominant.sum() > 0:
                kl_c = kl_per_pixel(
                    cnn_val[dominant].reshape(-1, 1, NUM_CLASSES),
                    gt_val[dominant].reshape(-1, 1, NUM_CLASSES),
                ).mean()
            else:
                kl_c = float("nan")
            cnn_class_kl.append(kl_c)

        results.append({
            "round": rnd, "seed": seed,
            "api_score": api_score,
            "cnn_score_full": cnn_score_full,
            "cnn_wkl_full": cnn_wkl_full,
            "pri_score_full": pri_score_full,
            "uni_score_full": uni_score_full,
            "cnn_kl_val": cnn_kl_val,
            "uni_kl_val": uni_kl_val,
            "pri_kl_val": pri_kl_val,
            "cnn_class_kl": cnn_class_kl,
        })

    # --- Print summary ---
    print(f"\n{'='*80}")
    print(f"  EVALUATION RESULTS  (val quadrant = {val_quadrant})")
    print(f"{'='*80}")

    header = (f"  {'Round':>5} {'Seed':>4} | {'API':>6} | "
              f"{'CNN':>7} {'Prior':>7} {'Unif':>7} | "
              f"{'CNN val':>8} {'Pri val':>8} {'Uni val':>8}")
    print(header)
    print(f"  {'':>5} {'':>4} | {'Score':>6} | "
          f"{'Score':>7} {'Score':>7} {'Score':>7} | "
          f"{'KL':>8} {'KL':>8} {'KL':>8}")
    print("  " + "-" * 76)

    for r in results:
        api = f"{r['api_score']:.1f}" if r['api_score'] is not None else "  n/a"
        print(f"  R{r['round']:>4} S{r['seed']:>3} | {api:>6} | "
              f"{r['cnn_score_full']:>7.2f} {r['pri_score_full']:>7.2f} {r['uni_score_full']:>7.2f} | "
              f"{r['cnn_kl_val']:>8.5f} {r['pri_kl_val']:>8.5f} {r['uni_kl_val']:>8.5f}")

    # Averages
    if results:
        avg_cnn = np.mean([r["cnn_score_full"] for r in results])
        avg_pri = np.mean([r["pri_score_full"] for r in results])
        avg_uni = np.mean([r["uni_score_full"] for r in results])
        avg_cnn_kl = np.mean([r["cnn_kl_val"] for r in results])
        avg_pri_kl = np.mean([r["pri_kl_val"] for r in results])
        avg_uni_kl = np.mean([r["uni_kl_val"] for r in results])
        print("  " + "-" * 76)
        print(f"  {'AVG':>10} | {'':>6} | "
              f"{avg_cnn:>7.2f} {avg_pri:>7.2f} {avg_uni:>7.2f} | "
              f"{avg_cnn_kl:>8.5f} {avg_pri_kl:>8.5f} {avg_uni_kl:>8.5f}")

    # Per-class breakdown
    print(f"\n  Per-class KL (CNN, val quadrant, averaged across seeds):")
    print(f"  {'Class':>12} | {'KL':>10} | {'Pixels':>7}")
    print("  " + "-" * 36)
    for c in range(NUM_CLASSES):
        vals = [r["cnn_class_kl"][c] for r in results if not np.isnan(r["cnn_class_kl"][c])]
        if vals:
            avg = np.mean(vals)
            print(f"  {CLASS_NAMES[c]:>12} | {avg:>10.5f} | {len(vals):>7}")
        else:
            print(f"  {CLASS_NAMES[c]:>12} | {'  n/a':>10} |")

    print()


def main():
    ckpt_path = find_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)
    if not ckpt_path:
        print("ERROR: No checkpoint found. Run train_cnn.py first.")
        print(f"  Looked in: {CHECKPOINT_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("  Astar Island — CNN Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {DEVICE}")

    print(f"\n--- Loading ground truth ---")
    all_data = load_local_data()

    if not all_data:
        print("No data to evaluate on.")
        return

    evaluate(ckpt_path, all_data, VAL_QUADRANT)


if __name__ == "__main__":
    main()
