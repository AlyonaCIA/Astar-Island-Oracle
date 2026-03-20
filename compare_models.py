"""
Astar Island — Model Comparison Script

Trains all registered architectures (quick, quick3, unet) from scratch on the
same data using 4-fold quadrant cross-validation, then evaluates each on the
held-out quadrant and prints a side-by-side comparison table.

Usage:
  python compare_models.py                    # train 300 epochs each, evaluate
  python compare_models.py --epochs 100       # fewer epochs for quick test
  python compare_models.py --eval-only        # skip training, just evaluate existing checkpoints
  python compare_models.py --fetch            # fetch ground truth from API first
  python compare_models.py --models quick unet  # only train/eval specific models
"""

import os
import sys
import time
import json
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import (
    _load_dotenv,
    encode_initial_grid,
    quadrant_masks, load_local_data, fetch_ground_truth,
    load_checkpoint, load_model_from_checkpoint,
    make_model, get_checkpoint_dir, latest_checkpoint,
    build_fullmap_datasets, kl_divergence_loss,
    save_checkpoint, _clear_checkpoints,
    MODEL_REGISTRY,
    DEVICE, NUM_CLASSES, PROB_FLOOR, VAL_QUADRANT,
)

_load_dotenv()


# ---------------------------------------------------------------------------
# Evaluation metrics (same as eval_cnn.py)
# ---------------------------------------------------------------------------

def kl_per_pixel(pred, target):
    pred = np.clip(pred, 1e-8, None)
    target = np.clip(target, 1e-8, None)
    return (target * (np.log(target) - np.log(pred))).sum(axis=-1)


def entropy_per_pixel(target):
    t = np.clip(target, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def weighted_kl(pred, target):
    kl = kl_per_pixel(pred, target)
    ent = entropy_per_pixel(target)
    total_entropy = ent.sum()
    if total_entropy < 1e-12:
        return 0.0
    return (kl * ent).sum() / total_entropy


def competition_score(pred, target):
    wkl = weighted_kl(pred, target)
    return 100.0 * np.exp(-3.0 * wkl), wkl


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
# Training a single model (simplified loop for comparison)
# ---------------------------------------------------------------------------

def train_single_model(arch, all_data, epochs, lr, batch_size, reset=False):
    """Train one architecture with 4-fold cross-validation. Returns ckpt path."""
    ckpt_dir = get_checkpoint_dir(arch)

    print(f"\n{'='*60}")
    print(f"  Training: {arch}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    features_list, targets_list, _, _, meta = \
        build_fullmap_datasets(all_data, val_quadrant=0)

    if not features_list:
        print("  ERROR: No usable data.")
        return None

    X = torch.tensor(np.stack(features_list)).to(DEVICE)
    Y = torch.tensor(np.stack(targets_list)).to(DEVICE)
    N, _, H, W = X.shape

    fold_masks = []
    for q in range(4):
        t_np, v_np = quadrant_masks(H, W, val_quadrant=q)
        fold_masks.append((
            torch.tensor(t_np, device=DEVICE),
            torch.tensor(v_np, device=DEVICE),
        ))

    model = make_model(arch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 1
    best_val = float("inf")
    training_start = time.time()

    # Resume if not reset
    ckpt_path = None if reset else latest_checkpoint(ckpt_dir)
    if reset:
        _clear_checkpoints(ckpt_dir)
        print(f"  Training from scratch (reset)")
    elif ckpt_path:
        print(f"  Resuming from {os.path.basename(ckpt_path)}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_loss", float("inf"))
        if start_epoch > epochs:
            print(f"  Already trained {start_epoch - 1} epochs (target: {epochs}). Skipping.")
            final = os.path.join(ckpt_dir, "cnn_latest.pt")
            if os.path.isfile(final):
                return final
            return ckpt_path
    else:
        print(f"  Training from scratch")

    print(f"  Epochs: {start_epoch}→{epochs}, LR={lr}, Device={DEVICE}")

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ckpt_every = 25
    metadata = {
        "model_arch": arch,
        "num_maps": N,
        "map_size": f"{H}x{W}",
        "cross_validation": "4-fold quadrant",
        "rounds": [m["round"] for m in meta],
        "lr": lr,
        "start_time": datetime.datetime.now().isoformat(),
    }

    os.makedirs(ckpt_dir, exist_ok=True)
    history_path = os.path.join(ckpt_dir, "training_history.json")
    history = []
    if not reset and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    last_epoch = start_epoch
    avg_train = 0.0
    val_loss = float("inf")

    try:
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            model.train()
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            total_train_batches = 0

            for t_hw, v_hw in fold_masks:
                for X_b, Y_b in loader:
                    B = X_b.shape[0]
                    Tm_b = t_hw.unsqueeze(0).expand(B, -1, -1)
                    optimizer.zero_grad()
                    pred = model(X_b)
                    loss = kl_divergence_loss(pred, Y_b, mask=Tm_b)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    total_train_batches += 1

                with torch.no_grad():
                    V_full = v_hw.unsqueeze(0).expand(N, -1, -1)
                    pred_all = model(X)
                    epoch_val_loss += kl_divergence_loss(pred_all, Y, mask=V_full).item()

            avg_train = epoch_train_loss / max(total_train_batches, 1)
            val_loss = epoch_val_loss / 4

            if val_loss < best_val:
                best_val = val_loss
                marker = " *"
            else:
                marker = ""

            elapsed = time.time() - training_start
            if epoch % 25 == 0 or epoch == epochs:
                print(f"  [{arch}] Epoch {epoch:4d}/{epochs} | "
                      f"train_kl={avg_train:.6f} | val_kl={val_loss:.6f}{marker} | {elapsed:.0f}s")

            history.append({
                "epoch": epoch, "train_kl": avg_train, "val_kl": val_loss,
                "best_val_kl": best_val, "elapsed_s": round(elapsed, 1),
            })

            if epoch % ckpt_every == 0 or epoch == epochs:
                metadata["total_epochs"] = epoch
                metadata["best_val_loss"] = best_val
                save_checkpoint(model, optimizer, epoch, avg_train, val_loss,
                                metadata, ckpt_dir=ckpt_dir, arch=arch)
                with open(history_path, "w") as f:
                    json.dump(history, f)

    except KeyboardInterrupt:
        print(f"\n  [{arch}] Interrupted at epoch {last_epoch}")

    # Save final
    final_path = os.path.join(ckpt_dir, "cnn_latest.pt")
    metadata["end_time"] = datetime.datetime.now().isoformat()
    metadata["total_epochs"] = last_epoch
    metadata["best_val_loss"] = best_val
    metadata["total_training_time_s"] = time.time() - training_start
    torch.save({
        "epoch": last_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train,
        "val_loss": val_loss,
        "metadata": metadata,
        "model_arch": arch,
    }, final_path)
    with open(history_path, "w") as f:
        json.dump(history, f)

    total_time = time.time() - training_start
    print(f"  [{arch}] Done — best_val_kl={best_val:.6f}, time={total_time:.0f}s")
    return final_path


# ---------------------------------------------------------------------------
# Evaluation of a single model
# ---------------------------------------------------------------------------

def evaluate_model(ckpt_path, all_data, val_quadrant):
    """Evaluate a checkpoint. Returns dict of metrics."""
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    arch = ckpt.get("model_arch") or ckpt.get("metadata", {}).get("model_arch", "quick")
    epoch = ckpt.get("epoch", "?")
    model.eval()

    results = []
    for data in all_data:
        initial_grid = data.get("initial_grid")
        ground_truth = data.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            continue

        width = data["width"]
        height = data["height"]
        gt = np.array(ground_truth, dtype=np.float32)
        features = encode_initial_grid(initial_grid, width, height)
        _, val_mask = quadrant_masks(height, width, val_quadrant)

        with torch.no_grad():
            x = torch.tensor(features).unsqueeze(0).to(DEVICE)
            probs = model(x).squeeze(0).permute(1, 2, 0).cpu().numpy()
        cnn_pred = np.maximum(probs, PROB_FLOOR)
        cnn_pred = cnn_pred / cnn_pred.sum(axis=-1, keepdims=True)

        # Full-map score
        score_full, wkl_full = competition_score(cnn_pred, gt)

        # Val-only score (unseen quadrant)
        gt_val = gt.copy()
        pred_val = cnn_pred.copy()
        # Zero out train region for val-only metric
        train_mask = ~val_mask
        gt_val[train_mask] = 1.0 / NUM_CLASSES
        pred_val[train_mask] = 1.0 / NUM_CLASSES
        score_val, wkl_val = competition_score(pred_val, gt_val)

        results.append({
            "round": data.get("_round_number", "?"),
            "seed": data.get("_seed_index", "?"),
            "score_full": score_full,
            "wkl_full": wkl_full,
            "score_val": score_val,
            "wkl_val": wkl_val,
        })

    if not results:
        return {"arch": arch, "epoch": epoch, "avg_score_full": 0, "avg_wkl_full": 999,
                "avg_score_val": 0, "avg_wkl_val": 999, "n_samples": 0, "per_seed": []}

    return {
        "arch": arch,
        "epoch": epoch,
        "avg_score_full": np.mean([r["score_full"] for r in results]),
        "avg_wkl_full": np.mean([r["wkl_full"] for r in results]),
        "avg_score_val": np.mean([r["score_val"] for r in results]),
        "avg_wkl_val": np.mean([r["wkl_val"] for r in results]),
        "n_samples": len(results),
        "per_seed": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Astar Island — Compare Model Architectures")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per model (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing checkpoints and train from scratch")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch/update ground truth from API first")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which models to compare (default: all)")
    parser.add_argument("--val-quadrant", type=int, default=VAL_QUADRANT,
                        choices=[0, 1, 2, 3],
                        help="Validation quadrant (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Astar Island — Model Comparison")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Models: {args.models}")
    print(f"  Val quadrant: {args.val_quadrant}")
    if not args.eval_only:
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    # Load data
    if args.fetch:
        print(f"\n--- Fetching ground truth from API ---")
        all_data = fetch_ground_truth()
    else:
        print(f"\n--- Loading local ground truth ---")
        all_data = load_local_data()

    if not all_data:
        print("No data available. Run with --fetch to download ground truth.")
        return

    # --- Training phase ---
    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"  TRAINING PHASE  ({len(args.models)} models x {args.epochs} epochs)")
        print(f"{'='*60}")

        for arch in args.models:
            train_single_model(arch, all_data, args.epochs, args.lr,
                               args.batch_size, reset=args.reset)

    # --- Evaluation phase ---
    print(f"\n{'='*60}")
    print(f"  EVALUATION PHASE")
    print(f"{'='*60}")

    all_results = {}

    # Baselines first
    print(f"\n--- Baselines ---")
    baseline_results = []
    for data in all_data:
        initial_grid = data.get("initial_grid")
        ground_truth = data.get("ground_truth")
        if initial_grid is None or ground_truth is None:
            continue
        width, height = data["width"], data["height"]
        gt = np.array(ground_truth, dtype=np.float32)

        uniform_pred = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES)
        prior_pred = build_prior_prediction(initial_grid, width, height)

        uni_score, uni_wkl = competition_score(uniform_pred, gt)
        pri_score, pri_wkl = competition_score(prior_pred, gt)

        baseline_results.append({
            "round": data.get("_round_number", "?"),
            "seed": data.get("_seed_index", "?"),
            "uni_score": uni_score, "uni_wkl": uni_wkl,
            "pri_score": pri_score, "pri_wkl": pri_wkl,
        })

    all_results["uniform"] = {
        "arch": "uniform", "epoch": "-",
        "avg_score_full": np.mean([r["uni_score"] for r in baseline_results]),
        "avg_wkl_full": np.mean([r["uni_wkl"] for r in baseline_results]),
        "n_samples": len(baseline_results),
    }
    all_results["prior"] = {
        "arch": "prior", "epoch": "-",
        "avg_score_full": np.mean([r["pri_score"] for r in baseline_results]),
        "avg_wkl_full": np.mean([r["pri_wkl"] for r in baseline_results]),
        "n_samples": len(baseline_results),
    }

    # Evaluate each model
    for arch in args.models:
        ckpt_dir = get_checkpoint_dir(arch)
        ckpt_path = os.path.join(ckpt_dir, "cnn_latest.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = latest_checkpoint(ckpt_dir)
        if not ckpt_path:
            print(f"  {arch}: no checkpoint found, skipping")
            continue
        print(f"\n--- Evaluating: {arch} ({os.path.basename(ckpt_path)}) ---")
        res = evaluate_model(ckpt_path, all_data, args.val_quadrant)
        all_results[arch] = res

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON RESULTS  (val quadrant = {args.val_quadrant})")
    print(f"{'='*70}")
    print(f"  {'Model':<12} | {'Epoch':>6} | {'Score (full)':>13} | "
          f"{'WKL (full)':>11} | {'Score (val)':>12} | {'WKL (val)':>10} | {'N':>3}")
    print("  " + "-" * 66)

    for name in ["uniform", "prior"] + args.models:
        r = all_results.get(name)
        if not r:
            continue
        epoch_s = str(r.get("epoch", "-"))
        s_full = f"{r['avg_score_full']:>10.2f}" if "avg_score_full" in r else "      n/a"
        w_full = f"{r['avg_wkl_full']:>8.5f}" if "avg_wkl_full" in r else "     n/a"
        s_val = f"{r.get('avg_score_val', 0):>9.2f}" if "avg_score_val" in r else "     n/a"
        w_val = f"{r.get('avg_wkl_val', 0):>7.5f}" if "avg_wkl_val" in r else "    n/a"
        n = r.get("n_samples", 0)
        print(f"  {name:<12} | {epoch_s:>6} | {s_full:>13} | {w_full:>11} | "
              f"{s_val:>12} | {w_val:>10} | {n:>3}")

    # Per-seed detail for CNN models
    for arch in args.models:
        r = all_results.get(arch)
        if not r or not r.get("per_seed"):
            continue
        print(f"\n  --- {arch} per-seed detail ---")
        print(f"  {'Round':>5} {'Seed':>4} | {'Score':>8} | {'WKL':>10} | {'Val Score':>10} | {'Val WKL':>10}")
        print("  " + "-" * 52)
        for s in r["per_seed"]:
            print(f"  R{s['round']:>4} S{s['seed']:>3} | "
                  f"{s['score_full']:>8.2f} | {s['wkl_full']:>10.5f} | "
                  f"{s['score_val']:>10.2f} | {s['wkl_val']:>10.5f}")

    # Best model
    best_name = None
    best_score = -1
    for name in args.models:
        r = all_results.get(name)
        if r and r.get("avg_score_full", 0) > best_score:
            best_score = r["avg_score_full"]
            best_name = name
    if best_name:
        print(f"\n  Best model: {best_name} (avg full-map score: {best_score:.2f})")

    print()


if __name__ == "__main__":
    main()
