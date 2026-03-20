"""
Astar Island — Quick Check: MC Dynamics Model

Fast sanity check of the Monte Carlo dynamics approach:
  1. Fetch fresh ground truth + round data from the API
  2. Train DynamicsCNN on ALL replay transitions for 1000 epochs
  3. For each GT sample, run 1000 MC rollouts and score against ground truth
  4. Print detailed timing breakdown to identify bottlenecks

Usage:
    python quick_check.py
    python quick_check.py --epochs 500 --rollouts 512
"""

import os
import sys
import time
import json
import logging
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("quick_check")

# ---------------------------------------------------------------------------
# .env loader + API setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv():
    env_path = os.path.join(SCRIPT_DIR, ".env")
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

BASE_URL = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("ASTAR_TOKEN")
if not TOKEN:
    log.error("Set ASTAR_TOKEN in .env or as environment variable.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

# ---------------------------------------------------------------------------
# Import from monte_carlo_cond (reuse model + helpers)
# ---------------------------------------------------------------------------
from monte_carlo_cond import (
    DynamicsCNN,
    build_transition_dataset, load_all_replays,
    rollout_trajectories, aggregate_predictions_fast,
    entropy_per_pixel, kl_per_pixel, competition_score,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GT_DIR = os.path.join(DATA_DIR, "ground_truth")
REPLAY_DIR = os.path.join(SCRIPT_DIR, "simulation_replays")


# ===================================================================
# STEP 0 — Fetch fresh data from API
# ===================================================================

def fetch_rounds():
    """GET /rounds — list all rounds."""
    resp = session.get(f"{BASE_URL}/rounds")
    resp.raise_for_status()
    return resp.json()


def fetch_round_details(round_id):
    """GET /rounds/{round_id} — initial states + dimensions."""
    resp = session.get(f"{BASE_URL}/rounds/{round_id}")
    resp.raise_for_status()
    return resp.json()


def fetch_analysis(round_id, seed_index):
    """GET /analysis/{round_id}/{seed_index} — ground truth comparison."""
    resp = session.get(f"{BASE_URL}/analysis/{round_id}/{seed_index}")
    resp.raise_for_status()
    return resp.json()


def refresh_ground_truth():
    """
    Re-fetch ground truth for all completed/scoring rounds from the API.
    Saves to data/ground_truth/ in the same format the codebase expects.
    Returns list of GT dicts.
    """
    os.makedirs(GT_DIR, exist_ok=True)

    log.info("Fetching rounds list from API...")
    t0 = time.time()
    rounds = fetch_rounds()
    completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
    log.info(f"  Found {len(completed)} completed/scoring rounds "
             f"(of {len(rounds)} total) in {time.time()-t0:.1f}s")

    all_gt = []
    fetched = 0
    skipped = 0

    for rnd in completed:
        round_id = rnd["id"]
        round_num = rnd["round_number"]
        seeds_count = rnd.get("seeds_count", 5)
        round_key = round_id[:8]

        # Also fetch round details for initial_states
        try:
            detail = fetch_round_details(round_id)
            time.sleep(1.0)
        except Exception as e:
            log.warning(f"  Could not fetch details for round #{round_num}: {e}")
            detail = None

        for seed_idx in range(seeds_count):
            fname = f"r{round_num}_s{seed_idx}_{round_key}.json"
            fpath = os.path.join(GT_DIR, fname)

            # Always re-fetch to ensure freshness
            try:
                analysis = fetch_analysis(round_id, seed_idx)
                time.sleep(1.0)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    skipped += 1
                    continue
                log.warning(f"  Analysis fetch failed for r{round_num} s{seed_idx}: {e}")
                skipped += 1
                continue
            except Exception as e:
                log.warning(f"  Analysis fetch failed for r{round_num} s{seed_idx}: {e}")
                skipped += 1
                continue

            # Build GT record
            gt_record = {
                "ground_truth": analysis.get("ground_truth"),
                "initial_grid": analysis.get("initial_grid"),
                "width": analysis.get("width", rnd.get("map_width", 40)),
                "height": analysis.get("height", rnd.get("map_height", 40)),
                "score": analysis.get("score"),
                "prediction": analysis.get("prediction"),
            }

            # If initial_grid missing from analysis, get from round details
            if gt_record["initial_grid"] is None and detail:
                init_states = detail.get("initial_states", [])
                if seed_idx < len(init_states):
                    gt_record["initial_grid"] = init_states[seed_idx].get("grid")

            with open(fpath, "w") as f:
                json.dump(gt_record, f)
            fetched += 1

            # Also save round details for offline use
            if detail:
                round_path = os.path.join(DATA_DIR, f"round_{round_key}.json")
                with open(round_path, "w") as f:
                    json.dump(detail, f)

            # Build the in-memory record
            gt_record["_round_number"] = round_num
            gt_record["_seed_index"] = seed_idx
            gt_record["_round_id"] = round_key
            all_gt.append(gt_record)

    log.info(f"  Fetched {fetched} GT samples, skipped {skipped} "
             f"in {time.time()-t0:.1f}s")
    return all_gt


def load_local_ground_truth():
    """Load GT from disk (fallback if API fails)."""
    if not os.path.isdir(GT_DIR):
        return []
    all_gt = []
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
        all_gt.append(data)
    return all_gt


# ===================================================================
# STEP 1 — Train dynamics on all data
# ===================================================================

def train_dynamics_all(replays, epochs=1000, lr=1e-3, batch_size=16):
    """
    Train DynamicsCNN on all replay transitions. No holdout, no checkpoints.
    Returns the trained model.
    """
    log.info("Building transition dataset from all replays...")
    t0 = time.time()
    X, Y, meta = build_transition_dataset(replays)
    dt_dataset = time.time() - t0
    log.info(f"  Dataset: {len(X)} transitions, built in {dt_dataset:.1f}s")

    log.info(f"  Transferring to {DEVICE}...")
    t0 = time.time()
    X_t = torch.tensor(X).to(DEVICE)
    Y_t = torch.tensor(Y).to(DEVICE)
    dt_transfer = time.time() - t0
    log.info(f"  Transfer: {dt_transfer:.1f}s")

    model = DynamicsCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model params: {n_params:,}")
    log.info(f"  Batches/epoch: {len(loader)}")

    log.info(f"\n{'='*60}")
    log.info(f"  Training DynamicsCNN — {epochs} epochs, lr={lr}")
    log.info(f"{'='*60}")

    training_start = time.time()
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
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
        loss_history.append(avg_loss)
        epoch_dt = time.time() - epoch_t0

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            elapsed = time.time() - training_start
            lr_now = optimizer.param_groups[0]["lr"]
            log.info(f"  Epoch {epoch:5d}/{epochs} | loss={avg_loss:.6f} | "
                     f"epoch_time={epoch_dt:.2f}s | total={elapsed:.0f}s | lr={lr_now:.1e}")

    total_train_time = time.time() - training_start
    log.info(f"\n  Training complete in {total_train_time:.1f}s")
    log.info(f"  Final loss: {loss_history[-1]:.6f}")
    log.info(f"  Best loss:  {min(loss_history):.6f} (epoch {loss_history.index(min(loss_history))+1})")
    log.info(f"  Avg epoch time: {total_train_time/epochs:.2f}s")

    return model, total_train_time


# ===================================================================
# STEP 2 — Evaluate against ground truth
# ===================================================================

def evaluate_against_gt(model, gt_data, K=1000, T=50):
    """
    For each GT sample, run MC rollouts and compare to ground truth tensor.
    Returns detailed per-sample results.
    """
    results = []
    total_rollout_time = 0.0
    total_aggregate_time = 0.0
    total_scoring_time = 0.0

    valid_samples = [g for g in gt_data
                     if g.get("initial_grid") and g.get("ground_truth")]
    log.info(f"\n{'='*60}")
    log.info(f"  Evaluating {len(valid_samples)} GT samples")
    log.info(f"  K={K} rollouts, T={T} steps (no observation weighting)")
    log.info(f"{'='*60}")

    for i, gt in enumerate(valid_samples):
        rid = gt.get("_round_id", "?")
        seed = gt.get("_seed_index", "?")
        rnum = gt.get("_round_number", "?")
        height = gt["height"]
        width = gt["width"]
        initial_grid = gt["initial_grid"]
        gt_tensor = np.array(gt["ground_truth"], dtype=np.float32)  # (H, W, 6)

        log.info(f"\n  [{i+1}/{len(valid_samples)}] Round {rnum} seed {seed} "
                 f"({rid}) — {width}x{height}")

        # --- Rollouts ---
        t0 = time.time()
        trajectories = rollout_trajectories(model, initial_grid, height, width, K, T)
        dt_rollout = time.time() - t0
        total_rollout_time += dt_rollout
        log.info(f"    Rollouts ({K}x{T} steps): {dt_rollout:.2f}s "
                 f"({dt_rollout/T*1000:.1f}ms/step, "
                 f"{K*T*height*width/dt_rollout/1e6:.1f}M cells/s)")

        # --- Aggregation (uniform weights — no obs) ---
        t0 = time.time()
        weights = np.ones(K, dtype=np.float64) / K
        pred = aggregate_predictions_fast(trajectories, weights, height, width)
        dt_aggregate = time.time() - t0
        total_aggregate_time += dt_aggregate
        log.info(f"    Aggregation: {dt_aggregate:.2f}s")

        # --- Scoring ---
        t0 = time.time()
        score, wkl = competition_score(pred, gt_tensor)
        dt_score = time.time() - t0
        total_scoring_time += dt_score

        # Per-class accuracy analysis
        pred_argmax = pred.argmax(axis=-1)
        gt_argmax = gt_tensor.argmax(axis=-1)
        accuracy = (pred_argmax == gt_argmax).mean() * 100

        # Entropy stats
        ent = entropy_per_pixel(gt_tensor)
        mean_entropy = ent.mean()
        high_ent_mask = ent > 0.5
        n_high_ent = high_ent_mask.sum()

        # KL on high-entropy cells only
        kl = kl_per_pixel(pred, gt_tensor)
        kl_high_ent = kl[high_ent_mask].mean() if n_high_ent > 0 else 0.0

        log.info(f"    Score: {score:.2f} | wKL: {wkl:.6f} | "
                 f"argmax_acc: {accuracy:.1f}%")
        log.info(f"    GT entropy: mean={mean_entropy:.4f}, "
                 f"high-ent cells: {n_high_ent}/{height*width}")
        log.info(f"    KL on high-ent cells: {kl_high_ent:.6f}")

        results.append({
            "round": rnum,
            "seed": seed,
            "round_id": rid,
            "score": score,
            "wkl": wkl,
            "accuracy": accuracy,
            "mean_entropy": float(mean_entropy),
            "n_high_entropy": int(n_high_ent),
            "kl_high_entropy": float(kl_high_ent),
            "dt_rollout": dt_rollout,
            "dt_aggregate": dt_aggregate,
            "dt_score": dt_score,
        })

    return results, total_rollout_time, total_aggregate_time, total_scoring_time


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick MC dynamics check")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs (default: 1000)")
    parser.add_argument("--rollouts", "-K", type=int, default=1000,
                        help="MC rollouts per sample (default: 1000)")
    parser.add_argument("--steps", "-T", type=int, default=50,
                        help="Rollout steps (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API fetch, use local GT only")
    args = parser.parse_args()

    wall_start = time.time()

    log.info("=" * 60)
    log.info("  Astar Island — Quick MC Check")
    log.info("=" * 60)
    log.info(f"  Device:       {DEVICE}")
    log.info(f"  Epochs:       {args.epochs}")
    log.info(f"  Rollouts (K): {args.rollouts}")
    log.info(f"  Steps (T):    {args.steps}")
    log.info(f"  LR:           {args.lr}")
    log.info(f"  Batch:        {args.batch}")
    log.info("")

    # ── STEP 0: Fetch fresh data ──
    log.info("─" * 60)
    log.info("  STEP 0: Fetching fresh data from API")
    log.info("─" * 60)

    t0 = time.time()
    if not args.skip_fetch:
        try:
            gt_data = refresh_ground_truth()
            log.info(f"  API fetch: {len(gt_data)} GT samples in {time.time()-t0:.1f}s")
        except Exception as e:
            log.warning(f"  API fetch failed: {e}")
            log.info("  Falling back to local ground truth...")
            gt_data = load_local_ground_truth()
    else:
        log.info("  --skip-fetch: using local ground truth")
        gt_data = load_local_ground_truth()

    dt_fetch = time.time() - t0
    valid_gt = [g for g in gt_data if g.get("initial_grid") and g.get("ground_truth")]
    log.info(f"  Valid GT samples: {len(valid_gt)} (of {len(gt_data)} total)")

    if not valid_gt:
        log.error("No valid GT samples found. Cannot evaluate.")
        return

    # Show what rounds we have
    rounds_seen = {}
    for g in valid_gt:
        rn = g.get("_round_number", "?")
        rounds_seen.setdefault(rn, 0)
        rounds_seen[rn] += 1
    log.info(f"  Rounds: {dict(sorted(rounds_seen.items()))}")

    # ── STEP 1: Load replays + train ──
    log.info("")
    log.info("─" * 60)
    log.info("  STEP 1: Train dynamics model on all replays")
    log.info("─" * 60)

    t0 = time.time()
    replays = load_all_replays()
    dt_load_replays = time.time() - t0
    log.info(f"  Loaded {len(replays)} replays in {dt_load_replays:.1f}s")

    if not replays:
        log.error("No replays found. Cannot train.")
        return

    model, dt_train = train_dynamics_all(
        replays, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    # ── STEP 2: Evaluate ──
    log.info("")
    log.info("─" * 60)
    log.info("  STEP 2: MC evaluation against ground truth")
    log.info("─" * 60)

    results, dt_rollouts_total, dt_agg_total, dt_score_total = evaluate_against_gt(
        model, valid_gt, K=args.rollouts, T=args.steps)

    # ── SUMMARY ──
    wall_total = time.time() - wall_start
    scores = [r["score"] for r in results]
    wkls = [r["wkl"] for r in results]
    accs = [r["accuracy"] for r in results]

    log.info("")
    log.info("=" * 60)
    log.info("  RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"  Samples evaluated:  {len(results)}")
    log.info(f"  Score:  {np.mean(scores):.2f} +/- {np.std(scores):.2f}  "
             f"(min={np.min(scores):.2f}, max={np.max(scores):.2f})")
    log.info(f"  wKL:    {np.mean(wkls):.6f} +/- {np.std(wkls):.6f}")
    log.info(f"  Argmax accuracy: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")

    # Per-round breakdown
    round_scores = {}
    for r in results:
        round_scores.setdefault(r["round"], []).append(r["score"])
    log.info("\n  Per-round scores:")
    for rn in sorted(round_scores.keys()):
        s = round_scores[rn]
        log.info(f"    Round {rn}: {np.mean(s):.2f} +/- {np.std(s):.2f} "
                 f"({len(s)} seeds)")

    log.info("")
    log.info("=" * 60)
    log.info("  TIMING BREAKDOWN")
    log.info("=" * 60)
    log.info(f"  API fetch:          {dt_fetch:8.1f}s")
    log.info(f"  Load replays:       {dt_load_replays:8.1f}s")
    log.info(f"  Training ({args.epochs} ep):  {dt_train:8.1f}s  "
             f"({dt_train/args.epochs:.2f}s/epoch)")
    log.info(f"  Rollouts total:     {dt_rollouts_total:8.1f}s  "
             f"({dt_rollouts_total/len(results):.2f}s/sample)")
    log.info(f"  Aggregation total:  {dt_agg_total:8.1f}s  "
             f"({dt_agg_total/len(results):.2f}s/sample)")
    log.info(f"  Scoring total:      {dt_score_total:8.1f}s")
    log.info("  ────────────────────────────────")
    log.info(f"  Wall clock total:   {wall_total:8.1f}s  "
             f"({wall_total/60:.1f} min)")

    # Bottleneck analysis
    phases = {
        "API fetch": dt_fetch,
        "Load replays": dt_load_replays,
        f"Training ({args.epochs}ep)": dt_train,
        "MC rollouts": dt_rollouts_total,
        "Aggregation": dt_agg_total,
        "Scoring": dt_score_total,
    }
    sorted_phases = sorted(phases.items(), key=lambda x: x[1], reverse=True)
    log.info("\n  Bottleneck ranking:")
    for rank, (name, dt) in enumerate(sorted_phases, 1):
        pct = dt / wall_total * 100
        bar = "#" * int(pct / 2)
        log.info(f"    {rank}. {name:25s} {dt:7.1f}s ({pct:5.1f}%) {bar}")

    log.info(f"\n{'='*60}")
    log.info("  Done.")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
