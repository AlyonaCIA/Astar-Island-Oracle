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
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reuse shared code from train_cnn (same dir)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import (
    _load_dotenv, encode_initial_grid,
    quadrant_masks, load_local_data, load_model_from_checkpoint,
    get_checkpoint_dir, latest_checkpoint, MODEL_REGISTRY,
    CHECKPOINT_DIR, DEVICE, NUM_CLASSES, PROB_FLOOR, VAL_QUADRANT,
    SCRIPT_DIR as TRAIN_SCRIPT_DIR,
)
from astar_cnn import bayesian_blend, terrain_to_class

_load_dotenv()

DATA_DIR = os.path.join(TRAIN_SCRIPT_DIR, "data")


# ---------------------------------------------------------------------------
# Viewport mask loading
# ---------------------------------------------------------------------------

def load_observations_list(round_id_short):
    """
    Load cached observations for a round.
    Handles both old format (bare list) and new format (wrapped dict with metadata).
    Returns list of observation dicts, or None if file not found.
    """
    obs_path = os.path.join(DATA_DIR, f"observations_{round_id_short}.json")
    if not os.path.isfile(obs_path):
        return None

    with open(obs_path) as f:
        raw = json.load(f)

    # New format: {"round_id": ..., "observations": [...]}
    if isinstance(raw, dict):
        return raw.get("observations", [])
    # Old format: bare list
    return raw


def load_viewport_masks(round_id_short, height, width, seeds_count=5):
    """
    Load cached observations for a round and build per-seed boolean masks.
    Returns dict: {seed_index: (H, W) bool ndarray} where True = observed pixel.
    Returns None if no observations file found for this round.
    """
    observations = load_observations_list(round_id_short)
    if observations is None:
        return None

    masks = {}
    for obs in observations:
        sid = obs["seed_index"]
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        if sid not in masks:
            masks[sid] = np.zeros((height, width), dtype=bool)
        # Mark observed pixels (clamp to map bounds)
        y_end = min(vy + vh, height)
        x_end = min(vx + vw, width)
        masks[sid][vy:y_end, vx:x_end] = True

    return masks


def find_all_observation_files():
    """Return dict mapping round_id_short → observations file path."""
    pattern = os.path.join(DATA_DIR, "observations_*.json")
    result = {}
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        # observations_76909e29.json → 76909e29
        short_id = fname.replace("observations_", "").replace(".json", "")
        result[short_id] = path
    return result


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


def _masked_weighted_kl(pred_pixels, target_pixels):
    """
    Entropy-weighted KL for a flat array of masked pixels.
    pred_pixels, target_pixels: (P, 6) numpy arrays.
    """
    pred_pixels = np.clip(pred_pixels, 1e-8, None)
    target_pixels = np.clip(target_pixels, 1e-8, None)
    kl = (target_pixels * (np.log(target_pixels) - np.log(pred_pixels))).sum(axis=-1)  # (P,)
    ent = -(target_pixels * np.log(target_pixels)).sum(axis=-1)  # (P,)
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

def find_checkpoint(path_arg=None, arch=None):
    """Resolve which checkpoint to evaluate."""
    if path_arg and os.path.isfile(path_arg):
        return path_arg
    # Use arch-specific directory if requested
    ckpt_dir = get_checkpoint_dir(arch) if arch else CHECKPOINT_DIR
    latest = os.path.join(ckpt_dir, "cnn_latest.pt")
    if os.path.isfile(latest):
        return latest
    # Fallback: highest epoch checkpoint
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt:
        return ckpt
    return None


def evaluate(ckpt_path, all_data, val_quadrant, use_viewports=False):
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    epoch = ckpt.get("epoch", "?")
    meta = ckpt.get("metadata", {})
    arch = ckpt.get("model_arch") or meta.get("model_arch", "quick")
    print(f"  Architecture: {arch}")
    print(f"  Epoch: {epoch}")
    print(f"  Train loss: {ckpt.get('train_loss', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?')}")
    if meta:
        print(f"  Trained on rounds: {meta.get('rounds', '?')}")
        print(f"  Training time: {meta.get('total_training_time_s', '?')}s")

    # Pre-load all available observation files
    obs_index = find_all_observation_files()
    if obs_index:
        print(f"  Observations found for rounds: {list(obs_index.keys())}")
    else:
        print("  No observation files found (blended eval will be skipped)")
    if use_viewports and not obs_index:
        print("  WARNING: --viewports enabled but no observation files found")

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
        round_id = data.get("_round_id", "")
        round_id_short = round_id[:8] if round_id else ""

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

        # --- Bayesian blended prediction (CNN + observations) ---
        blend_pred = None
        seed_obs = None
        if round_id_short in obs_index:
            all_obs = load_observations_list(round_id_short)
            if all_obs:
                seed_obs = [o for o in all_obs if o["seed_index"] == seed]
                if seed_obs:
                    blend_pred = bayesian_blend(
                        cnn_pred, seed_obs, initial_grid, width, height
                    )

        # --- Full-map metrics ---
        cnn_score_full, cnn_wkl_full = competition_score(cnn_pred, gt)
        uni_score_full, uni_wkl_full = competition_score(uniform_pred, gt)
        pri_score_full, pri_wkl_full = competition_score(prior_pred, gt)
        if blend_pred is not None:
            bln_score_full, bln_wkl_full = competition_score(blend_pred, gt)
        else:
            bln_score_full = bln_wkl_full = None

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
        if blend_pred is not None:
            bln_val = blend_pred[val_mask]
            bln_kl_val = kl_per_pixel(bln_val.reshape(-1, 1, NUM_CLASSES),
                                      gt_val.reshape(-1, 1, NUM_CLASSES)).mean()
        else:
            bln_kl_val = None

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

        # --- Viewport-only metrics ---
        vp_cnn_score = vp_pri_score = vp_uni_score = vp_bln_score = None
        vp_cnn_wkl = vp_pri_wkl = vp_uni_wkl = vp_bln_wkl = None
        vp_pixels = 0
        if use_viewports and round_id_short in obs_index:
            vp_masks = load_viewport_masks(round_id_short, height, width)
            if vp_masks and seed in vp_masks:
                vp_mask = vp_masks[seed]
                vp_pixels = int(vp_mask.sum())
                if vp_pixels > 0:
                    gt_vp = gt[vp_mask]                    # (P, 6)
                    cnn_vp = cnn_pred[vp_mask]             # (P, 6)
                    uni_vp = uniform_pred[vp_mask]
                    pri_vp = prior_pred[vp_mask]
                    vp_cnn_wkl = _masked_weighted_kl(cnn_vp, gt_vp)
                    vp_pri_wkl = _masked_weighted_kl(pri_vp, gt_vp)
                    vp_uni_wkl = _masked_weighted_kl(uni_vp, gt_vp)
                    vp_cnn_score = 100.0 * np.exp(-3.0 * vp_cnn_wkl)
                    vp_pri_score = 100.0 * np.exp(-3.0 * vp_pri_wkl)
                    vp_uni_score = 100.0 * np.exp(-3.0 * vp_uni_wkl)
                    if blend_pred is not None:
                        bln_vp = blend_pred[vp_mask]
                        vp_bln_wkl = _masked_weighted_kl(bln_vp, gt_vp)
                        vp_bln_score = 100.0 * np.exp(-3.0 * vp_bln_wkl)

        results.append({
            "round": rnd, "seed": seed,
            "api_score": api_score,
            "cnn_score_full": cnn_score_full,
            "cnn_wkl_full": cnn_wkl_full,
            "bln_score_full": bln_score_full,
            "bln_wkl_full": bln_wkl_full,
            "pri_score_full": pri_score_full,
            "uni_score_full": uni_score_full,
            "cnn_kl_val": cnn_kl_val,
            "bln_kl_val": bln_kl_val,
            "uni_kl_val": uni_kl_val,
            "pri_kl_val": pri_kl_val,
            "cnn_class_kl": cnn_class_kl,
            "vp_cnn_score": vp_cnn_score,
            "vp_bln_score": vp_bln_score,
            "vp_pri_score": vp_pri_score,
            "vp_uni_score": vp_uni_score,
            "vp_cnn_wkl": vp_cnn_wkl,
            "vp_bln_wkl": vp_bln_wkl,
            "vp_pri_wkl": vp_pri_wkl,
            "vp_uni_wkl": vp_uni_wkl,
            "vp_pixels": vp_pixels,
        })

    # --- Print summary ---
    has_blend = any(r["bln_score_full"] is not None for r in results)

    print(f"\n{'='*90}")
    print(f"  EVALUATION RESULTS  (val quadrant = {val_quadrant})")
    print(f"{'='*90}")

    bln_hdr = f" {'Blend':>7}" if has_blend else ""
    bln_hdr2 = f" {'Score':>7}" if has_blend else ""
    bln_vhdr = f" {'Bln val':>8}" if has_blend else ""
    bln_vhdr2 = f" {'KL':>8}" if has_blend else ""

    header = (f"  {'Round':>5} {'Seed':>4} | {'API':>6} | "
              f"{'CNN':>7}{bln_hdr} {'Prior':>7} {'Unif':>7} | "
              f"{'CNN val':>8}{bln_vhdr} {'Pri val':>8} {'Uni val':>8}")
    print(header)
    print(f"  {'':>5} {'':>4} | {'Score':>6} | "
          f"{'Score':>7}{bln_hdr2} {'Score':>7} {'Score':>7} | "
          f"{'KL':>8}{bln_vhdr2} {'KL':>8} {'KL':>8}")
    print("  " + "-" * (76 + (16 if has_blend else 0)))

    for r in results:
        api = f"{r['api_score']:.1f}" if r['api_score'] is not None else "  n/a"
        bln_s = f" {r['bln_score_full']:>7.2f}" if r['bln_score_full'] is not None else (" " + "    n/a") if has_blend else ""
        bln_v = f" {r['bln_kl_val']:>8.5f}" if r['bln_kl_val'] is not None else (" " + "     n/a") if has_blend else ""
        print(f"  R{r['round']:>4} S{r['seed']:>3} | {api:>6} | "
              f"{r['cnn_score_full']:>7.2f}{bln_s} {r['pri_score_full']:>7.2f} {r['uni_score_full']:>7.2f} | "
              f"{r['cnn_kl_val']:>8.5f}{bln_v} {r['pri_kl_val']:>8.5f} {r['uni_kl_val']:>8.5f}")

    # Averages
    if results:
        avg_cnn = np.mean([r["cnn_score_full"] for r in results])
        avg_pri = np.mean([r["pri_score_full"] for r in results])
        avg_uni = np.mean([r["uni_score_full"] for r in results])
        avg_cnn_kl = np.mean([r["cnn_kl_val"] for r in results])
        avg_pri_kl = np.mean([r["pri_kl_val"] for r in results])
        avg_uni_kl = np.mean([r["uni_kl_val"] for r in results])

        bln_results = [r for r in results if r["bln_score_full"] is not None]
        if bln_results:
            avg_bln = np.mean([r["bln_score_full"] for r in bln_results])
            avg_bln_kl = np.mean([r["bln_kl_val"] for r in bln_results])
            bln_avg_s = f" {avg_bln:>7.2f}"
            bln_avg_v = f" {avg_bln_kl:>8.5f}"
        else:
            bln_avg_s = (" " + "    n/a") if has_blend else ""
            bln_avg_v = (" " + "     n/a") if has_blend else ""

        print("  " + "-" * (76 + (16 if has_blend else 0)))
        print(f"  {'AVG':>10} | {'':>6} | "
              f"{avg_cnn:>7.2f}{bln_avg_s} {avg_pri:>7.2f} {avg_uni:>7.2f} | "
              f"{avg_cnn_kl:>8.5f}{bln_avg_v} {avg_pri_kl:>8.5f} {avg_uni_kl:>8.5f}")

    # --- Viewport results ---
    vp_results = [r for r in results if r["vp_cnn_score"] is not None]
    if vp_results:
        has_vp_blend = any(r["vp_bln_score"] is not None for r in vp_results)
        bvh = f" {'Blend':>7}" if has_vp_blend else ""
        bvh2 = f" {'Score':>7}" if has_vp_blend else ""
        bvw = f" {'Blend':>10}" if has_vp_blend else ""
        bvw2 = f" {'WKL':>10}" if has_vp_blend else ""
        vp_sep = 82 + (18 if has_vp_blend else 0)

        print(f"\n{'='*max(80, vp_sep+4)}")
        print(f"  VIEWPORT-ONLY RESULTS  (pixels you actually observed)")
        print(f"{'='*max(80, vp_sep+4)}")
        print(f"  {'Round':>5} {'Seed':>4} | {'Pixels':>6} | "
              f"{'CNN':>7}{bvh} {'Prior':>7} {'Unif':>7} | "
              f"{'CNN':>10}{bvw} {'Prior':>10} {'Unif':>10}")
        print(f"  {'':>5} {'':>4} | {'':>6} | "
              f"{'Score':>7}{bvh2} {'Score':>7} {'Score':>7} | "
              f"{'WKL':>10}{bvw2} {'WKL':>10} {'WKL':>10}")
        print("  " + "-" * vp_sep)
        for r in vp_results:
            bvs = f" {r['vp_bln_score']:>7.2f}" if r['vp_bln_score'] is not None else (" " + "    n/a") if has_vp_blend else ""
            bvwk = f" {r['vp_bln_wkl']:>10.5f}" if r['vp_bln_wkl'] is not None else (" " + "       n/a") if has_vp_blend else ""
            print(f"  R{r['round']:>4} S{r['seed']:>3} | {r['vp_pixels']:>6} | "
                  f"{r['vp_cnn_score']:>7.2f}{bvs} {r['vp_pri_score']:>7.2f} {r['vp_uni_score']:>7.2f} | "
                  f"{r['vp_cnn_wkl']:>10.5f}{bvwk} {r['vp_pri_wkl']:>10.5f} {r['vp_uni_wkl']:>10.5f}")
        if len(vp_results) > 1:
            avg_vp_cnn = np.mean([r["vp_cnn_score"] for r in vp_results])
            avg_vp_pri = np.mean([r["vp_pri_score"] for r in vp_results])
            avg_vp_uni = np.mean([r["vp_uni_score"] for r in vp_results])
            avg_vp_cnn_wkl = np.mean([r["vp_cnn_wkl"] for r in vp_results])
            avg_vp_pri_wkl = np.mean([r["vp_pri_wkl"] for r in vp_results])
            avg_vp_uni_wkl = np.mean([r["vp_uni_wkl"] for r in vp_results])
            avg_vp_pix = np.mean([r["vp_pixels"] for r in vp_results])
            vp_bln_sub = [r for r in vp_results if r["vp_bln_score"] is not None]
            if vp_bln_sub:
                avg_vp_bln = np.mean([r["vp_bln_score"] for r in vp_bln_sub])
                avg_vp_bln_wkl = np.mean([r["vp_bln_wkl"] for r in vp_bln_sub])
                bva_s = f" {avg_vp_bln:>7.2f}"
                bva_w = f" {avg_vp_bln_wkl:>10.5f}"
            else:
                bva_s = (" " + "    n/a") if has_vp_blend else ""
                bva_w = (" " + "       n/a") if has_vp_blend else ""
            print("  " + "-" * vp_sep)
            print(f"  {'AVG':>10} | {avg_vp_pix:>6.0f} | "
                  f"{avg_vp_cnn:>7.2f}{bva_s} {avg_vp_pri:>7.2f} {avg_vp_uni:>7.2f} | "
                  f"{avg_vp_cnn_wkl:>10.5f}{bva_w} {avg_vp_pri_wkl:>10.5f} {avg_vp_uni_wkl:>10.5f}")

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
    parser = argparse.ArgumentParser(description="Astar Island — CNN Evaluation")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to checkpoint file (default: latest)")
    parser.add_argument("--arch", choices=list(MODEL_REGISTRY.keys()), default=None,
                        help="Architecture to find checkpoint for (default: auto-detect)")
    parser.add_argument("--viewports", action="store_true",
                        help="Also evaluate on viewport-only pixels from cached observations")
    parser.add_argument("--val-quadrant", type=int, default=VAL_QUADRANT,
                        choices=[0, 1, 2, 3],
                        help=f"Validation quadrant (default: {VAL_QUADRANT})")
    args = parser.parse_args()

    ckpt_path = find_checkpoint(args.checkpoint, arch=args.arch)
    if not ckpt_path:
        ckpt_dir = get_checkpoint_dir(args.arch) if args.arch else CHECKPOINT_DIR
        print("ERROR: No checkpoint found. Run train_cnn.py first.")
        print(f"  Looked in: {ckpt_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Astar Island — CNN Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {DEVICE}")
    if args.viewports:
        print(f"  Viewport evaluation: ENABLED")

    print(f"\n--- Loading ground truth ---")
    all_data = load_local_data()

    if not all_data:
        print("No data to evaluate on.")
        return

    evaluate(ckpt_path, all_data, args.val_quadrant, use_viewports=args.viewports)


if __name__ == "__main__":
    main()
