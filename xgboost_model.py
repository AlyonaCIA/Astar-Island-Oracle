"""
Astar Island — XGBoost Direct Predictor
========================================

Standalone XGBoost model that predicts the 6-class terrain probability
distribution directly from the initial grid, using hand-crafted spatial
features (terrain one-hot, neighborhood densities at multiple radii,
position, coastal flags).

Trains 6 independent XGBRegressor models (one per output class) on
per-pixel features extracted from ground truth data.

USAGE:
  python xgboost_model.py train [--n-estimators 500] [--max-depth 8] [--lr 0.05] [--reset]
  python xgboost_model.py evaluate [--n-estimators 500] [--max-depth 8] [--lr 0.05]
"""

import os
import json
import time
import argparse
import numpy as np
import xgboost as xgb
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(".")
LOCAL_GT_DIR = BASE_DIR / "data" / "ground_truth"
CKPT_DIR = BASE_DIR / "checkpoints_xgb"

# Terrain encoding (same as vertex_ai_mc2.py)
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CODE_TO_IDX = {c: i for i, c in enumerate(TERRAIN_CODES)}
NUM_TERRAIN = 8
SUBMIT_CLASSES = 6
PROB_FLOOR = 0.01

# XGBoost defaults
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 8
XGB_LR = 0.05
XGB_RADII = [1, 2, 3, 5, 7]


# =========================
# DATA LOADING
# =========================
def load_ground_truth():
    d = str(LOCAL_GT_DIR)
    if not os.path.isdir(d):
        print(f"ERROR: {d} not found")
        return []
    data = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(d, f)) as fh:
            dd = json.load(fh)
        parts = f.replace(".json", "").split("_")
        if len(parts) >= 3:
            dd["_round_number"] = int(parts[0][1:])
            dd["_seed_index"] = int(parts[1][1:])
            dd["_round_id"] = parts[2]
        data.append(dd)
    print(f"Loaded {len(data)} ground truth samples")
    return data


# =========================
# FEATURE EXTRACTION
# =========================
def _extract_pixel_features(grid, h, w):
    """Extract per-pixel features from an initial grid.

    Features (52 total):
      - 8  terrain one-hot
      - 1  y position (normalized)
      - 1  x position (normalized)
      - 1  distance from center
      - 1  coastal flag (land adjacent to ocean)
      - 40 neighborhood terrain densities (8 types x 5 radii)
    """
    grid_arr = np.array(grid, dtype=np.int32)
    idx_arr = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            idx_arr[y, x] = CODE_TO_IDX.get(grid_arr[y, x], 0)

    # Binary masks for each terrain type
    masks = np.zeros((NUM_TERRAIN, h, w), dtype=np.float32)
    for c in range(NUM_TERRAIN):
        masks[c] = (idx_arr == c).astype(np.float32)

    # Neighbor density at various radii
    neigh_feats = []
    for r in XGB_RADII:
        counts = np.zeros((NUM_TERRAIN, h, w), dtype=np.float32)
        total = np.zeros((h, w), dtype=np.float32)
        padded_masks = np.pad(masks, ((0, 0), (r, r), (r, r)), mode='constant')
        padded_ones = np.pad(np.ones((h, w), dtype=np.float32), r, mode='constant')
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                counts += padded_masks[:, r + dy:r + dy + h, r + dx:r + dx + w]
                total += padded_ones[r + dy:r + dy + h, r + dx:r + dx + w]
        safe_total = np.maximum(total, 1.0)
        norm_counts = counts / safe_total[None, :, :]
        neigh_feats.append(norm_counts.transpose(1, 2, 0))  # (h, w, 8)

    # Position features
    y_norm = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    x_norm = np.arange(w, dtype=np.float32)[None, :] / max(w - 1, 1)
    y_grid = np.broadcast_to(y_norm, (h, w)).copy()
    x_grid = np.broadcast_to(x_norm, (h, w)).copy()
    dist_center = np.sqrt((y_grid - 0.5) ** 2 + (x_grid - 0.5) ** 2)

    # Coastal flag (land cell adjacent to ocean)
    ocean_idx = CODE_TO_IDX[10]
    ocean = masks[ocean_idx]
    padded_ocean = np.pad(ocean, 1, mode='constant')
    coastal = np.zeros((h, w), dtype=np.float32)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal = np.maximum(
            coastal,
            padded_ocean[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
    coastal *= (1.0 - ocean)  # exclude ocean cells

    # Assemble
    feats = [
        masks.transpose(1, 2, 0),       # (h, w, 8)
        y_grid[:, :, None],             # (h, w, 1)
        x_grid[:, :, None],             # (h, w, 1)
        dist_center[:, :, None],        # (h, w, 1)
        coastal[:, :, None],            # (h, w, 1)
    ]
    for nc in neigh_feats:
        feats.append(nc)                # (h, w, 8) per radius
    return np.concatenate(feats, axis=-1)  # (h, w, n_feat)


# =========================
# DATASET
# =========================
def build_dataset(gt_data, held_out=None):
    """Build per-pixel (X, Y) arrays from ground truth samples."""
    all_X, all_Y = [], []
    for g in gt_data:
        rid = g.get("_round_id", "")[:8]
        if held_out and rid in held_out:
            continue
        ig, gt = g.get("initial_grid"), g.get("ground_truth")
        if ig is None or gt is None:
            continue
        h, w = g["height"], g["width"]
        feats = _extract_pixel_features(ig, h, w)
        targets = np.array(gt, dtype=np.float32)
        all_X.append(feats.reshape(-1, feats.shape[-1]))
        all_Y.append(targets.reshape(-1, SUBMIT_CLASSES))
    if not all_X:
        return np.zeros((0, 52), dtype=np.float32), \
               np.zeros((0, SUBMIT_CLASSES), dtype=np.float32)
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    print(f"  Dataset: {X.shape[0]} pixels, {X.shape[1]} features")
    return X, Y


# =========================
# SCORING
# =========================
def entropy_px(t):
    t = np.clip(t, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def kl_px(p, t):
    p = np.clip(p, 1e-8, None)
    t = np.clip(t, 1e-8, None)
    return (t * (np.log(t) - np.log(p))).sum(axis=-1)


def weighted_kl(p, t):
    kl = kl_px(p, t)
    ent = entropy_px(t)
    s = ent.sum()
    return (kl * ent).sum() / s if s > 1e-12 else 0.0


def comp_score(p, t):
    wkl = weighted_kl(p, t)
    return 100.0 * np.exp(-3.0 * wkl), wkl


# =========================
# PREDICT
# =========================
def predict(models, grid, h, w):
    """Predict 6-class probability map using XGBoost ensemble."""
    feats = _extract_pixel_features(grid, h, w)
    X = feats.reshape(-1, feats.shape[-1])
    pred = np.zeros((X.shape[0], SUBMIT_CLASSES), dtype=np.float32)
    for c, m in enumerate(models):
        pred[:, c] = m.predict(X).clip(PROB_FLOOR, None)
    pred = pred.reshape(h, w, SUBMIT_CLASSES)
    pred = np.maximum(pred, PROB_FLOOR)
    return (pred / pred.sum(axis=-1, keepdims=True)).astype(np.float32)


# =========================
# TRAIN
# =========================
def train(gt_data, n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
          lr=XGB_LR, reset=False):
    print(f"\n{'='*60}")
    print(f"  XGBoost Training (all data)")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={lr}")
    print(f"{'='*60}")

    ckpt_path = CKPT_DIR / "xgb_class_0.json"
    if not reset and ckpt_path.exists():
        print(f"  Checkpoints exist in {CKPT_DIR}/. Use --reset to retrain.")
        return load_models()

    X, Y = build_dataset(gt_data)
    if len(X) == 0:
        print("ERROR: no data")
        return None

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    t0 = time.time()

    for c in range(SUBMIT_CLASSES):
        print(f"  Training class {c} regressor...")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            objective='reg:squarederror',
            n_jobs=-1,
            verbosity=1,
        )
        model.fit(X, Y[:, c])
        model.save_model(str(CKPT_DIR / f"xgb_class_{c}.json"))
        models.append(model)

    print(f"  All {SUBMIT_CLASSES} classes trained in {time.time()-t0:.1f}s")
    print(f"  Saved to {CKPT_DIR}/")
    return models


def load_models():
    models = []
    for c in range(SUBMIT_CLASSES):
        p = CKPT_DIR / f"xgb_class_{c}.json"
        if not p.exists():
            print(f"  Missing checkpoint: {p}")
            return None
        m = xgb.XGBRegressor()
        m.load_model(str(p))
        models.append(m)
    print(f"  Loaded {len(models)} XGBoost models from {CKPT_DIR}/")
    return models


# =========================
# EVALUATE (round k-fold CV)
# =========================
def evaluate(gt_data, n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
             lr=XGB_LR):
    rids = sorted(set(
        g.get("_round_id", "")[:8] for g in gt_data if g.get("_round_id")))
    if len(rids) < 2:
        print("ERROR: need >= 2 rounds for CV")
        return

    print(f"\n{'='*60}")
    print(f"  XGBoost — {len(rids)}-fold Round CV")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={lr}")
    print(f"{'='*60}")

    scores, wkls = [], []

    for fi, vr in enumerate(rids):
        print(f"\n  --- Fold {fi+1}/{len(rids)}: holdout {vr} ---")
        Xt, Yt = build_dataset(gt_data, held_out={vr})
        if len(Xt) == 0:
            print("    Skip (no training data)")
            continue

        # Train 6 per-class regressors
        models = []
        t0 = time.time()
        for c in range(SUBMIT_CLASSES):
            m = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                objective='reg:squarederror',
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(Xt, Yt[:, c])
            models.append(m)
        print(f"    Trained {SUBMIT_CLASSES} models in {time.time()-t0:.1f}s")

        # Score on holdout round
        for gs in [g for g in gt_data if g.get("_round_id", "")[:8] == vr]:
            ig, gt = gs.get("initial_grid"), gs.get("ground_truth")
            if ig is None or gt is None:
                continue
            pred = predict(models, ig, gs["height"], gs["width"])
            gtarr = np.array(gt, dtype=np.float32)
            s, w = comp_score(pred, gtarr)
            scores.append(s)
            wkls.append(w)
            print(f"    {vr} seed {gs.get('_seed_index', 0)}: "
                  f"score={s:.2f}  wKL={w:.6f}")

    if scores:
        print(f"\n{'='*60}")
        print(f"  XGBoost CV Results")
        print(f"  Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f} "
              f"[{np.min(scores):.2f}, {np.max(scores):.2f}]")
        print(f"  wKL:   {np.mean(wkls):.6f} +/- {np.std(wkls):.6f}")
        print(f"{'='*60}")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Astar Island — XGBoost predictor")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("train", help="Train on all ground truth data")
    p.add_argument("--n-estimators", type=int, default=XGB_N_ESTIMATORS)
    p.add_argument("--max-depth", type=int, default=XGB_MAX_DEPTH)
    p.add_argument("--lr", type=float, default=XGB_LR)
    p.add_argument("--reset", action="store_true",
                   help="Retrain even if checkpoints exist")

    p = sub.add_parser("evaluate", help="Round k-fold cross-validation")
    p.add_argument("--n-estimators", type=int, default=XGB_N_ESTIMATORS)
    p.add_argument("--max-depth", type=int, default=XGB_MAX_DEPTH)
    p.add_argument("--lr", type=float, default=XGB_LR)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return

    gt = load_ground_truth()
    if not gt:
        print("No ground truth data found. Place JSON files in ground_truth/")
        return

    if args.cmd == "train":
        train(gt, args.n_estimators, args.max_depth, args.lr, args.reset)
    elif args.cmd == "evaluate":
        evaluate(gt, args.n_estimators, args.max_depth, args.lr)

    print("\nDone!")


if __name__ == "__main__":
    main()
