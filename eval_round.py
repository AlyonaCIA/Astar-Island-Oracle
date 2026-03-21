"""
Astar Island — Round-Specific Evaluation & Score Diagnosis

Compares your ACTUAL SUBMITTED prediction against:
- The current (possibly updated) checkpoint re-inference
- Prior-based baseline
- Uniform baseline

Shows per-pixel KL breakdown to explain score discrepancies.

Usage:
    python eval_round.py 16                          # evaluate round 16
    python eval_round.py 16 --arch unet_cond         # specific architecture
    python eval_round.py 16 --detailed               # per-pixel breakdown
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import (
    _load_dotenv, encode_initial_grid,
    load_model_from_checkpoint, get_checkpoint_dir,
    latest_checkpoint, MODEL_REGISTRY,
    DEVICE, NUM_CLASSES, PROB_FLOOR,
    encode_obs_channels,
)
from eval_cnn import (
    kl_per_pixel, entropy_per_pixel, weighted_kl, competition_score,
    build_prior_prediction, load_observations_list,
)
from astar_cnn import bayesian_blend

_load_dotenv()

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GT_DIR = os.path.join(DATA_DIR, "ground_truth")

CLASS_NAMES = ["Ocean", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _run_model(model, features, width, height, obs_features=None, temperature=1.0):
    """Run model inference with optional temperature scaling."""
    with torch.no_grad():
        if obs_features is not None:
            x = np.concatenate([features, obs_features], axis=0)
        else:
            x = features
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)
        probs = model(x).squeeze(0)  # (6, H, W)

        # Manual temperature scaling on logits (reverse softmax, scale, re-softmax)
        if temperature != 1.0:
            logits = torch.log(torch.clamp(probs, min=1e-8))
            logits = logits / temperature
            probs = torch.softmax(logits, dim=0)

        probs = probs.permute(1, 2, 0).cpu().numpy()  # (H, W, 6)
    probs = np.maximum(probs, PROB_FLOOR)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def _apply_hard_override(probs, raw_observations, height, width):
    """Apply hard override: observed pixels → 95% for observed class."""
    probs = probs.copy()
    for obs in raw_observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        obs_grid = obs["grid"]
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    actual_code = obs_grid[dy][dx]
                    actual_class = TERRAIN_TO_CLASS.get(actual_code, 0)
                    probs[gy, gx, :] = PROB_FLOOR
                    probs[gy, gx, actual_class] = 1.0 - (PROB_FLOOR * 5)
    return probs


TERRAIN_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
                 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}


def find_round_files(round_number):
    """Find all ground truth files for a round."""
    pattern = os.path.join(GT_DIR, f"r{round_number}_s*_*.json")
    files = sorted(glob.glob(pattern))
    return files


def load_round_data(round_number):
    """Load all seed data for a round, including stored predictions."""
    files = find_round_files(round_number)
    if not files:
        return [], None

    data = []
    round_id_short = None
    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract seed and round_id_short from filename: r16_s0_8f664aed.json
        parts = fname.replace(".json", "").split("_")
        seed_idx = int(parts[1][1:])  # s0 → 0
        rid_short = parts[2]
        round_id_short = rid_short

        with open(fpath) as f:
            record = json.load(f)
        record["_seed_index"] = seed_idx
        record["_round_id_short"] = rid_short
        data.append(record)

    return data, round_id_short


def find_checkpoint(arch):
    """Find the latest checkpoint for the given architecture."""
    ckpt_dir = get_checkpoint_dir(arch)
    latest = os.path.join(ckpt_dir, "cnn_latest.pt")
    if os.path.isfile(latest):
        return latest
    ckpt = latest_checkpoint(ckpt_dir)
    return ckpt


def evaluate_round(round_number, arch="unet_cond", detailed=False):
    arch_label = arch.upper()
    print("=" * 80)
    print(f"  ROUND {round_number} — SCORE DIAGNOSIS  (arch={arch_label})")
    print("=" * 80)

    # Load data
    seeds_data, round_id_short = load_round_data(round_number)
    if not seeds_data:
        print(f"ERROR: No ground truth files found for round {round_number}.")
        print(f"  Run: python fetch_ground_truth.py {round_number}")
        return

    print(f"  Seeds found: {len(seeds_data)}")
    print(f"  Round ID: {round_id_short}")

    # Load observations
    obs_list = None
    if round_id_short:
        obs_list = load_observations_list(round_id_short)
    if obs_list:
        seeds_with_obs = set(o["seed_index"] for o in obs_list)
        print(f"  Observations: {len(obs_list)} viewports for seeds {sorted(seeds_with_obs)}")
    else:
        print("  Observations: NONE (no observation file found)")

    # Load model checkpoint
    ckpt_path = find_checkpoint(arch)
    model = None
    if ckpt_path:
        print(f"  Checkpoint: {ckpt_path}")
        model, ckpt = load_model_from_checkpoint(ckpt_path)
        model.eval()
        ckpt_arch = ckpt.get("model_arch") or ckpt.get("metadata", {}).get("model_arch", "quick")
        ckpt_epoch = ckpt.get("epoch", "?")
        print(f"  Architecture: {ckpt_arch}, Epoch: {ckpt_epoch}")
    else:
        print(f"  WARNING: No checkpoint found for --arch {arch}")

    print()

    # =========================================================================
    # Per-seed analysis
    # =========================================================================
    results = []
    for data in seeds_data:
        seed = data["_seed_index"]
        gt = np.array(data["ground_truth"], dtype=np.float32)  # (H, W, 6)
        initial_grid = data["initial_grid"]
        width, height = data["width"], data["height"]
        api_score = data.get("score")

        # --- Submitted prediction (what was actually sent to the API) ---
        submitted_pred = None
        submitted_score = None
        submitted_wkl = None
        if data.get("prediction") is not None:
            submitted_pred = np.array(data["prediction"], dtype=np.float32)
            submitted_score, submitted_wkl = competition_score(submitted_pred, gt)

        # --- Current checkpoint (re-inference — this IS what gets submitted now) ---
        cnn_score = cnn_wkl = None
        cnn_pred = None
        features = encode_initial_grid(initial_grid, width, height)
        seed_obs = [o for o in (obs_list or []) if o["seed_index"] == seed]
        obs_feat = encode_obs_channels(seed_obs, width, height) if seed_obs else None
        if arch != "unet_cond":
            obs_feat = None
        if model is not None:
            cnn_pred = _run_model(model, features, width, height, obs_features=obs_feat)
            cnn_score, cnn_wkl = competition_score(cnn_pred, gt)

        # --- Bayesian blend ---
        blend_pred = None
        blend_score = blend_wkl = None
        if cnn_pred is not None and seed_obs:
            blend_pred = bayesian_blend(cnn_pred, seed_obs, initial_grid, width, height)
            blend_score, blend_wkl = competition_score(blend_pred, gt)

        # --- Mountain override (static terrain → 1.0 for mountain class) ---
        if cnn_pred is not None:
            for y in range(height):
                for x_i in range(width):
                    if initial_grid[y][x_i] == 5:  # mountain
                        cnn_pred[y, x_i, :] = PROB_FLOOR
                        cnn_pred[y, x_i, 5] = 1.0 - (PROB_FLOOR * 5)
                        if blend_pred is not None:
                            blend_pred[y, x_i, :] = PROB_FLOOR
                            blend_pred[y, x_i, 5] = 1.0 - (PROB_FLOOR * 5)
            cnn_score, cnn_wkl = competition_score(cnn_pred, gt)
            if blend_pred is not None:
                blend_score, blend_wkl = competition_score(blend_pred, gt)

        # --- Baselines ---
        prior_pred = build_prior_prediction(initial_grid, width, height)
        prior_score, prior_wkl = competition_score(prior_pred, gt)

        uniform_pred = np.full((height, width, NUM_CLASSES), 1.0/NUM_CLASSES, dtype=np.float32)
        uniform_score, uniform_wkl = competition_score(uniform_pred, gt)

        results.append({
            "seed": seed,
            "api_score": api_score,
            "submitted_score": submitted_score,
            "submitted_wkl": submitted_wkl,
            "cnn_score": cnn_score,
            "cnn_wkl": cnn_wkl,
            "blend_score": blend_score,
            "blend_wkl": blend_wkl,
            "prior_score": prior_score,
            "prior_wkl": prior_wkl,
            "uniform_score": uniform_score,
            "uniform_wkl": uniform_wkl,
            "gt": gt,
            "submitted_pred": submitted_pred,
            "cnn_pred": cnn_pred,
            "initial_grid": initial_grid,
        })

    # =========================================================================
    # Print comparison table
    # =========================================================================
    print(f"{'='*80}")
    print(f"  SCORE COMPARISON — Round {round_number}")
    print(f"{'='*80}")
    print(f"  {'Seed':>4} | {'API':>7} | {'Submitted':>9} | {arch_label:>9} | {'Blend':>7} | {'Prior':>7} | {'Uniform':>7}")
    print(f"  {'':>4} | {'Score':>7} | {'(re-eval)':>9} | {'(current)':>9} | {'Score':>7} | {'Score':>7} | {'Score':>7}")
    print("  " + "-" * 70)

    for r in results:
        api = f"{r['api_score']:.2f}" if r['api_score'] is not None else "  n/a"
        sub = f"{r['submitted_score']:.2f}" if r['submitted_score'] is not None else "  n/a"
        cnn = f"{r['cnn_score']:.2f}" if r['cnn_score'] is not None else "  n/a"
        bln = f"{r['blend_score']:.2f}" if r['blend_score'] is not None else "  n/a"
        pri = f"{r['prior_score']:.2f}"
        uni = f"{r['uniform_score']:.2f}"
        print(f"  S{r['seed']:>3} | {api:>7} | {sub:>9} | {cnn:>9} | {bln:>7} | {pri:>7} | {uni:>7}")

    # Averages
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.mean(vals) if vals else None

    print("  " + "-" * 70)
    api_avg = avg("api_score")
    sub_avg = avg("submitted_score")
    cnn_avg = avg("cnn_score")
    bln_avg = avg("blend_score")
    pri_avg = avg("prior_score")
    uni_avg = avg("uniform_score")
    print(f"  {'AVG':>4} | {api_avg:>7.2f} | {sub_avg:>9.2f} | "
          f"{cnn_avg if cnn_avg else 0:>9.2f} | "
          f"{bln_avg if bln_avg else 0:>7.2f} | {pri_avg:>7.2f} | {uni_avg:>7.2f}")

    # =========================================================================
    # WKL breakdown
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  WEIGHTED KL BREAKDOWN")
    print(f"{'='*80}")
    print(f"  {'Seed':>4} | {'Submitted':>9} | {arch_label:>9} | {'Blend':>9} | {'Prior':>9}")
    print(f"  {'':>4} | {'wKL':>9} | {'wKL':>9} | {'wKL':>9} | {'wKL':>9}")
    print("  " + "-" * 50)
    for r in results:
        sub_w = f"{r['submitted_wkl']:.6f}" if r['submitted_wkl'] is not None else "    n/a"
        cnn_w = f"{r['cnn_wkl']:.6f}" if r['cnn_wkl'] is not None else "    n/a"
        bln_w = f"{r['blend_wkl']:.6f}" if r['blend_wkl'] is not None else "    n/a"
        pri_w = f"{r['prior_wkl']:.6f}"
        print(f"  S{r['seed']:>3} | {sub_w:>9} | {cnn_w:>9} | {bln_w:>9} | {pri_w:>9}")

    # =========================================================================
    # Detailed per-pixel analysis of submitted prediction
    # =========================================================================
    if detailed:
        for r in results:
            if r["submitted_pred"] is None or r["cnn_pred"] is None:
                continue
            print(f"\n{'='*90}")
            print(f"  DETAILED PIXEL ANALYSIS — Seed {r['seed']}")
            print(f"{'='*90}")

            gt = r["gt"]
            sub = r["submitted_pred"]
            cnn = r["cnn_pred"]
            initial_grid = r["initial_grid"]

            # Per-pixel KL for submitted vs CNN
            sub_kl = kl_per_pixel(sub, gt)  # (H, W)
            cnn_kl = kl_per_pixel(cnn, gt)  # (H, W)
            ent = entropy_per_pixel(gt)      # (H, W)

            # Weighted contribution per pixel
            total_ent = ent.sum()
            sub_contrib = (sub_kl * ent) / total_ent  # contribution to wKL
            cnn_contrib = (cnn_kl * ent) / total_ent

            # Difference: positive = submitted is worse
            diff = sub_contrib - cnn_contrib

            # Top 20 pixels where submitted prediction is worse than CNN
            h, w = gt.shape[:2]
            flat_diff = diff.flatten()
            worst_idx = np.argsort(flat_diff)[::-1][:20]

            print(f"\n  Top 20 pixels where SUBMITTED is worse than {arch_label} (no override):")
            print(f"  {'(y,x)':>7} | {'Terrain':>10} | {'GT top class':>12} | "
                  f"{'Sub KL':>8} | {arch_label+' KL':>8} | {'Entropy':>8} | {'Damage':>8}")
            print("  " + "-" * 80)

            mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
            for idx in worst_idx:
                y, x = divmod(idx, w)
                if diff[y, x] <= 0:
                    break
                terrain_code = initial_grid[y][x]
                terrain_name = TERRAIN_NAMES.get(terrain_code, f"?{terrain_code}")
                gt_top = CLASS_NAMES[gt[y, x].argmax()]
                gt_probs = gt[y, x]
                sub_probs = sub[y, x]
                cnn_probs = cnn[y, x]

                print(f"  ({y:>2},{x:>2}) | {terrain_name:>10} | {gt_top:>12} | "
                      f"{sub_kl[y,x]:>8.4f} | {cnn_kl[y,x]:>8.4f} | {ent[y,x]:>8.4f} | "
                      f"{diff[y,x]:>8.6f}")
                if detailed:
                    # Show probability comparison
                    print(f"           GT:  [{', '.join(f'{p:.3f}' for p in gt_probs)}]")
                    print(f"           Sub: [{', '.join(f'{p:.3f}' for p in sub_probs)}]")
                    print(f"           {arch_label}: [{', '.join(f'{p:.3f}' for p in cnn_probs)}]")

            # Summary: how much KL comes from overridden vs non-overridden pixels
            if obs_list:
                seed_obs = [o for o in obs_list if o["seed_index"] == r["seed"]]
                if seed_obs:
                    obs_mask = np.zeros((h, w), dtype=bool)
                    for obs in seed_obs:
                        vp = obs["viewport"]
                        vx, vy = vp["x"], vp["y"]
                        vw, vh = vp["w"], vp["h"]
                        y_end = min(vy + vh, h)
                        x_end = min(vx + vw, w)
                        obs_mask[vy:y_end, vx:x_end] = True

                    obs_count = obs_mask.sum()
                    unobs_count = (~obs_mask).sum()

                    sub_kl_obs = (sub_kl[obs_mask] * ent[obs_mask]).sum() / total_ent if total_ent > 0 else 0
                    sub_kl_unobs = (sub_kl[~obs_mask] * ent[~obs_mask]).sum() / total_ent if total_ent > 0 else 0
                    cnn_kl_obs = (cnn_kl[obs_mask] * ent[obs_mask]).sum() / total_ent if total_ent > 0 else 0
                    cnn_kl_unobs = (cnn_kl[~obs_mask] * ent[~obs_mask]).sum() / total_ent if total_ent > 0 else 0

                    print(f"\n  KL SPLIT: Observed vs Unobserved pixels")
                    print(f"  {'Region':>12} | {'Pixels':>6} | {'Sub wKL':>10} | {arch_label+' wKL':>10} | {'Diff':>10}")
                    print("  " + "-" * 55)
                    print(f"  {'Observed':>12} | {obs_count:>6} | {sub_kl_obs:>10.6f} | {cnn_kl_obs:>10.6f} | {sub_kl_obs - cnn_kl_obs:>+10.6f}")
                    print(f"  {'Unobserved':>12} | {unobs_count:>6} | {sub_kl_unobs:>10.6f} | {cnn_kl_unobs:>10.6f} | {sub_kl_unobs - cnn_kl_unobs:>+10.6f}")
                    print(f"  {'TOTAL':>12} | {obs_count+unobs_count:>6} | {sub_kl_obs+sub_kl_unobs:>10.6f} | {cnn_kl_obs+cnn_kl_unobs:>10.6f} | {(sub_kl_obs+sub_kl_unobs)-(cnn_kl_obs+cnn_kl_unobs):>+10.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Astar Island — Round-Specific Score Diagnosis")
    parser.add_argument("round", type=int, help="Round number to evaluate")
    parser.add_argument("--arch", default="unet_cond",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Architecture (default: unet_cond)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show per-pixel KL breakdown for worst offenders")
    args = parser.parse_args()

    evaluate_round(args.round, arch=args.arch, detailed=args.detailed)


if __name__ == "__main__":
    main()
