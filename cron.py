"""
Astar Island — Automated Pipeline (Cron)

Runs continuously, checking every 100 minutes for active rounds.

Pipeline per round:
  1. Check for active round + budget
  2. Submit prior-based fallback (guarantees a score)
  3. Query viewports (deterministic/stochastic detection strategy)
  4. Submit UNet predictions (overwrites fallback)
  5. Log observations to disk
  6. Wait 10 minutes for ground truth availability
  7. Fetch ground truth from completed rounds
  8. Retrain UNet (up to 4,000 new epochs from last checkpoint)
  9. Sleep 100 minutes, repeat

Usage:
    python cron.py              # run forever
    python cron.py --once       # run one cycle and exit
"""

import os
import sys
import time
import json
import datetime
import logging
import traceback

import torch

# Import project modules (they auto-load .env and configure API sessions)
import astar_cnn
import train_cnn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCH = "unet_aug"                       # MiniUNet with augmented data (dropout=0.1)
POLL_INTERVAL_S = 100 * 60             # 100 minutes between checks (~160 min round cadence)
GT_WAIT_S = 10 * 60                    # 10 minutes wait for ground truth
MAX_TRAIN_EPOCHS = 4_000               # max new epochs per training cycle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "cron_state.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "cron.log")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("cron")

# ---------------------------------------------------------------------------
# State persistence — tracks which rounds have been processed
# ---------------------------------------------------------------------------


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed_rounds": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_active_round_safe():
    """Fetch the currently active round, or None if there is none."""
    try:
        resp = astar_cnn.session.get(f"{astar_cnn.BASE_URL}/rounds")
        resp.raise_for_status()
        time.sleep(1.0)
        rounds = resp.json()
        return next((r for r in rounds if r["status"] == "active"), None)
    except Exception as e:
        log.error(f"Failed to fetch rounds: {e}")
        return None


def get_budget_safe():
    """Fetch budget dict, or None on failure."""
    try:
        return astar_cnn.check_budget(verbose=False)
    except Exception:
        return None


def load_unet_model():
    """Load the latest UNet checkpoint. Returns (model, epoch) or (None, None)."""
    ckpt_dir = train_cnn.get_checkpoint_dir(ARCH)
    ckpt_path = train_cnn.latest_checkpoint(ckpt_dir)
    if not ckpt_path:
        return None, None

    model = astar_cnn.make_model(ARCH).to(astar_cnn.DEVICE)
    ckpt = train_cnn.load_checkpoint(ckpt_path, model)
    model.eval()
    return model, ckpt.get("epoch", 0)


def get_current_epoch():
    """Return the epoch number of the latest UNet checkpoint, or 0."""
    ckpt_dir = train_cnn.get_checkpoint_dir(ARCH)
    ckpt_path = train_cnn.latest_checkpoint(ckpt_dir)
    if not ckpt_path:
        return 0
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Pipeline: check active round → query → submit
# ---------------------------------------------------------------------------


def run_pipeline():
    """
    Check for an active round and, if budget is unspent, run the full
    query-and-submit pipeline.  Returns True if the pipeline actually ran.
    """
    state = load_state()

    # --- 1. Check for active round ---
    log.info("Checking for active rounds...")
    active = get_active_round_safe()
    if not active:
        log.info("No active round found.")
        return False

    round_id = active["id"]
    round_num = active["round_number"]
    round_key = round_id[:8]
    log.info(f"Active round #{round_num} ({round_key}…)")
    log.info(f"  Closes at: {active.get('closes_at', 'unknown')}")

    # --- 2. Already processed? ---
    if round_key in state["processed_rounds"]:
        prev = state["processed_rounds"][round_key]
        log.info(
            f"Round #{round_num} already processed "
            f"at {prev.get('processed_at', '?')}. Skipping."
        )
        return False

    # --- 3. Check budget ---
    budget = get_budget_safe()
    if budget:
        used, total = budget["queries_used"], budget["queries_max"]
        log.info(f"Budget: {used}/{total} queries used")
        has_budget = used < total
    else:
        log.warning("Could not check budget. Will attempt pipeline anyway.")
        has_budget = True

    # --- 4. Fetch round details ---
    log.info("Fetching round details...")
    try:
        detail = astar_cnn.get_round_details(round_id)
    except Exception as e:
        log.error(f"Failed to get round details: {e}")
        return False

    width = detail["map_width"]
    height = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]
    log.info(f"  Map: {width}x{height}, Seeds: {seeds_count}")

    # Cache round data for offline use
    astar_cnn._save_round_data(round_id, detail)

    # --- 5. Submit prior-based fallback (guarantees a score) ---
    log.info("--- Submitting prior-based fallback ---")
    try:
        astar_cnn.submit_fallback(
            round_id, seeds_count, initial_states, width, height
        )
    except Exception as e:
        log.error(f"Fallback submission failed: {e}")

    # --- 6. Collect observations if budget available ---
    observations = []
    if has_budget:
        log.info("--- Collecting observations (stochastic strategy) ---")
        try:
            observations = astar_cnn.collect_observations(
                round_id, seeds_count, initial_states, width, height
            )
            log.info(f"Collected {len(observations)} observations")
        except Exception as e:
            log.error(f"Observation collection failed: {e}")
            log.error(traceback.format_exc())
    else:
        log.info("Budget already spent. Skipping observation collection.")

    # --- 7. Load UNet and submit predictions ---
    log.info("--- Loading UNet checkpoint ---")
    model, epoch = load_unet_model()

    if model is not None:
        log.info(f"Loaded UNet (epoch {epoch}). Submitting predictions...")
        encoded_grids = {}
        for seed_idx in range(seeds_count):
            encoded_grids[seed_idx] = astar_cnn.encode_initial_grid(
                initial_states[seed_idx]["grid"], width, height
            )
        try:
            astar_cnn.submit_cnn_predictions(
                round_id, model, encoded_grids, initial_states,
                seeds_count, width, height,
            )
            log.info("UNet predictions submitted successfully.")
        except Exception as e:
            log.error(f"UNet submission failed: {e}")
            log.info("Prior-based fallback still stands.")
    else:
        log.warning(
            f"No UNet checkpoint found in "
            f"{train_cnn.get_checkpoint_dir(ARCH)}. Fallback stands."
        )

    # --- 8. Mark as processed ---
    state["processed_rounds"][round_key] = {
        "round_number": round_num,
        "round_id": round_id,
        "processed_at": datetime.datetime.now().isoformat(),
        "n_observations": len(observations),
        "model_epoch": epoch,
        "status": "submitted",
    }
    save_state(state)
    log.info(f"Round #{round_num} marked as processed.\n")
    return True


# ---------------------------------------------------------------------------
# Retrain: fetch ground truth → train up to MAX_TRAIN_EPOCHS new epochs
# ---------------------------------------------------------------------------


def retrain():
    """Fetch the latest ground truth and retrain the UNet."""

    # 1. Fetch / refresh ground truth
    log.info("--- Fetching ground truth from API ---")
    try:
        all_data = train_cnn.fetch_ground_truth()
    except Exception as e:
        log.error(f"Ground truth fetch failed: {e}")
        log.info("Falling back to locally cached ground truth...")
        all_data = train_cnn.load_local_data()

    if not all_data:
        log.warning("No training data available. Skipping retraining.")
        return

    # 2. Compute target epoch = current + MAX_TRAIN_EPOCHS
    current_epoch = get_current_epoch()
    target_epoch = current_epoch + MAX_TRAIN_EPOCHS
    log.info(
        f"Current epoch: {current_epoch} → "
        f"will train up to epoch {target_epoch} "
        f"(max {MAX_TRAIN_EPOCHS} new epochs)"
    )

    # 3. Temporarily override the global EPOCHS and run training
    original_epochs = train_cnn.EPOCHS
    train_cnn.EPOCHS = target_epoch

    try:
        train_cnn.train(all_data, reset=False, forever=False, arch=ARCH)
    except KeyboardInterrupt:
        raise  # propagate to outer loop
    except Exception as e:
        log.error(f"Training error: {e}")
        log.error(traceback.format_exc())
    finally:
        train_cnn.EPOCHS = original_epochs

    final_epoch = get_current_epoch()
    log.info(f"Retraining complete. Final epoch: {final_epoch}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    once = "--once" in sys.argv

    log.info("=" * 60)
    log.info("  Astar Island — Automated Pipeline (Cron)")
    log.info("=" * 60)
    log.info(f"  Architecture:  {ARCH}")
    log.info(f"  Device:        {astar_cnn.DEVICE}")
    log.info(f"  Poll interval: {POLL_INTERVAL_S // 60} min")
    log.info(f"  GT wait:       {GT_WAIT_S // 60} min")
    log.info(f"  Max epochs/cycle: {MAX_TRAIN_EPOCHS}")
    log.info(f"  Log file:      {LOG_FILE}")
    log.info(f"  State file:    {STATE_FILE}")

    _, epoch = load_unet_model()
    if epoch is not None:
        log.info(f"  Latest UNet:   epoch {epoch}")
    else:
        log.info("  Latest UNet:   (none — fallback until first training)")

    if once:
        log.info("  Mode:          --once (single cycle)")
    log.info("")

    while True:
        try:
            submitted = run_pipeline()

            if submitted:
                # Wait for ground truth to become available
                log.info(
                    f"Waiting {GT_WAIT_S // 60} minutes for ground truth "
                    f"before retraining..."
                )
                time.sleep(GT_WAIT_S)
                retrain()

        except KeyboardInterrupt:
            log.info("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            log.error(f"Unexpected error in main loop: {e}")
            log.error(traceback.format_exc())

        if once:
            log.info("Single cycle complete (--once). Exiting.")
            break

        log.info(f"Next check in {POLL_INTERVAL_S // 60} minutes...")
        try:
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.info("\nInterrupted during sleep. Exiting.")
            break


if __name__ == "__main__":
    main()
