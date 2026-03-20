"""
Astar Island — Baseline Observation + Submission Script

Strategy:
1. Fetch active round and initial states
2. Use queries to observe simulation outcomes across seeds
3. Build per-cell probability estimates from initial terrain + observations
4. Apply probability floor and submit for all 5 seeds
"""

import os
import sys
import json
import time
import numpy as np
import requests


def _load_dotenv():
    """Load .env file from script directory if it exists."""
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

# --- Configuration ---
BASE_URL = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("ASTAR_TOKEN")

if not TOKEN:
    print("ERROR: Set ASTAR_TOKEN in .env file or as environment variable.")
    print("  Copy .env.example to .env and fill in your JWT token.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
NUM_CLASSES = 6
PROB_FLOOR = 0.01  # Never assign 0 probability


# --- Helpers ---

def get_active_round():
    """Fetch all rounds and return the active one, if any."""
    resp = session.get(f"{BASE_URL}/rounds")
    resp.raise_for_status()
    rounds = resp.json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round found. Available rounds:")
        for r in rounds:
            print(f"  Round {r['round_number']} — status: {r['status']}")
        sys.exit(0)
    return active


def get_round_details(round_id):
    """Fetch full round details including initial states."""
    resp = session.get(f"{BASE_URL}/rounds/{round_id}")
    resp.raise_for_status()
    return resp.json()


def check_budget(verbose=True):
    """Check remaining query budget for the active round."""
    resp = session.get(f"{BASE_URL}/budget")
    if resp.status_code == 200:
        data = resp.json()
        if verbose:
            print(f"Budget: {data['queries_used']}/{data['queries_max']} queries used")
        return data
    return None


def simulate(round_id, seed_index, vx, vy, vw=15, vh=15):
    """Run one simulation query and return the viewport result."""
    resp = session.post(f"{BASE_URL}/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": min(vw, 15),
        "viewport_h": min(vh, 15),
    })
    resp.raise_for_status()
    return resp.json()


def terrain_to_class(cell_value):
    """Map internal terrain code to prediction class index."""
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(cell_value, 0)


def build_initial_prediction(initial_grid, width, height):
    """
    Build a baseline prediction from the initial terrain grid.
    
    Static cells (ocean, mountain, forest) get high confidence.
    Dynamic cells (settlements, ports, plains) get spread probabilities
    reflecting that they might change during simulation.
    """
    pred = np.full((height, width, NUM_CLASSES), PROB_FLOOR)

    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]

            if cell == 10:  # Ocean — almost certain to stay empty
                pred[y][x][0] = 0.95
            elif cell == 5:  # Mountain — never changes
                pred[y][x][5] = 0.95
            elif cell == 4:  # Forest — mostly static, small chance of settlement
                #                  empty  settl  port   ruin   forest mount
                pred[y][x] = [0.06, 0.04, 0.01, 0.03, 0.84, 0.02]
            elif cell == 1:  # Settlement — could survive, become ruin, etc.
                pred[y][x] = [0.13, 0.40, 0.02, 0.25, 0.10, 0.10]
            elif cell == 2:  # Port — similar to settlement
                pred[y][x] = [0.12, 0.15, 0.35, 0.22, 0.05, 0.11]
            elif cell in (0, 11):  # Empty/Plains — could get settled or forested
                pred[y][x] = [0.58, 0.12, 0.02, 0.05, 0.18, 0.05]

    # Enforce floor and normalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def update_prediction_with_observation(pred, obs_grid, viewport, obs_count_grid):
    """
    Update prediction using an observation from a simulation query.
    Accumulates class counts, to be normalized later.
    """
    vx, vy = viewport["x"], viewport["y"]
    vh, vw = len(obs_grid), len(obs_grid[0])

    for dy in range(vh):
        for dx in range(vw):
            y, x = vy + dy, vx + dx
            cls = terrain_to_class(obs_grid[dy][dx])
            pred[y][x][cls] += 1.0
            obs_count_grid[y][x] += 1


def finalize_prediction(pred, obs_count_grid, initial_pred):
    """
    Blend observation-based counts with initial prediction.
    Cells with more observations rely more on observed data.
    """
    height, width = pred.shape[:2]
    final = np.copy(initial_pred)

    for y in range(height):
        for x in range(width):
            n = obs_count_grid[y][x]
            if n > 0:
                # Observation-based distribution (add small pseudocount)
                obs_dist = pred[y][x] + PROB_FLOOR
                obs_dist = obs_dist / obs_dist.sum()
                # Blend: more observations → more weight on observed
                alpha = n / (n + 2.0)  # simple blending weight
                final[y][x] = alpha * obs_dist + (1 - alpha) * initial_pred[y][x]

    # Final floor + normalize
    final = np.maximum(final, PROB_FLOOR)
    final = final / final.sum(axis=-1, keepdims=True)
    return final


def submit_prediction(round_id, seed_index, prediction):
    """Submit prediction tensor for one seed."""
    resp = session.post(f"{BASE_URL}/submit", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    resp.raise_for_status()
    return resp.json()


def plan_viewports(width, height, num_queries, vw=15, vh=15):
    """
    Generate viewport positions to cover as much of the map as possible.
    Returns a list of (vx, vy) positions.
    """
    viewports = []
    # Tile the map with overlapping viewports
    y_positions = list(range(0, height, vh - 2))  # slight overlap
    x_positions = list(range(0, width, vw - 2))

    for vy in y_positions:
        for vx in x_positions:
            viewports.append((vx, vy))

    # Center viewport for good coverage of middle
    cx, cy = (width - vw) // 2, (height - vh) // 2
    viewports.insert(0, (cx, cy))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in viewports:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique[:num_queries]


def _save_observations(observations, round_id):
    """Save observations to disk for offline analysis."""
    if not observations:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"observations_{round_id[:8]}.json")
    with open(path, "w") as f:
        json.dump(observations, f)
    print(f"  Saved {len(observations)} observations to {path}")


# --- Main ---

def main():
    print("=" * 50)
    print("  Astar Island — Baseline Submission")
    print("=" * 50)

    # Step 1: Get active round
    active = get_active_round()
    round_id = active["id"]
    print(f"\nActive round #{active['round_number']} — {active['map_width']}x{active['map_height']}")
    print(f"  Closes at: {active.get('closes_at', 'unknown')}")

    # Step 2: Get round details
    detail = get_round_details(round_id)
    width = detail["map_width"]
    height = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]
    print(f"  Seeds: {seeds_count}, Size: {width}x{height}")

    # Step 3: Check budget
    budget = check_budget()
    if budget:
        remaining = budget["queries_max"] - budget["queries_used"]
    else:
        remaining = 50
    print(f"  Remaining queries: {remaining}")

    if remaining <= 0:
        print("\nNo queries remaining. Submitting from initial state only.")
        for seed_idx in range(seeds_count):
            grid = initial_states[seed_idx]["grid"]
            pred = build_initial_prediction(grid, width, height)
            result = submit_prediction(round_id, seed_idx, pred)
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        print("\nDone!")
        return

    # Step 4: Plan query strategy
    # Prepare viewports to cover the map
    max_viewports = max(1, remaining // seeds_count)
    all_viewports = plan_viewports(width, height, max_viewports)
    print(f"\n--- Query Strategy: ~{max_viewports} queries per seed, {len(all_viewports)} viewports planned ---")

    # Step 5: Observe and build predictions
    all_observations = []  # collect for saving
    queries_used_total = remaining  # track dynamically
    for seed_idx in range(seeds_count):
        # Recalculate budget for remaining seeds
        seeds_left = seeds_count - seed_idx
        budget_now = check_budget(verbose=False)
        if budget_now:
            queries_used_total = budget_now["queries_max"] - budget_now["queries_used"]
        queries_for_this_seed = max(1, queries_used_total // seeds_left)

        print(f"\n--- Seed {seed_idx} ({queries_for_this_seed} queries planned, {queries_used_total} remaining) ---")
        grid = initial_states[seed_idx]["grid"]
        initial_pred = build_initial_prediction(grid, width, height)

        # Accumulator for observations
        obs_pred = np.zeros((height, width, NUM_CLASSES))
        obs_count = np.zeros((height, width))

        # Run queries for this seed
        viewports_to_use = all_viewports[:queries_for_this_seed]
        for i, (vx, vy) in enumerate(viewports_to_use):
            try:
                result = simulate(round_id, seed_idx, vx, vy)
                update_prediction_with_observation(
                    obs_pred, result["grid"], result["viewport"], obs_count
                )
                all_observations.append({
                    "seed_index": seed_idx,
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                })
                used = result.get("queries_used", "?")
                max_q = result.get("queries_max", "?")
                print(f"  Query {i+1}: viewport ({vx},{vy}) — budget {used}/{max_q}")

                # Respect rate limit (5 req/s max, use 1s for safety)
                time.sleep(1.0)

                # Stop if budget exhausted
                if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                    print("  Budget exhausted!")
                    _save_observations(all_observations, round_id)
                    break

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"  Rate limited or budget exhausted at query {i+1}")
                    _save_observations(all_observations, round_id)
                    break
                raise

        # Finalize prediction for this seed
        final_pred = finalize_prediction(obs_pred, obs_count, initial_pred)

        # Submit
        try:
            result = submit_prediction(round_id, seed_idx, final_pred)
            print(f"  Submitted: {result.get('status', 'unknown')}")
        except requests.HTTPError as e:
            print(f"  Submit FAILED for seed {seed_idx}: {e}")
            if e.response is not None:
                print(f"    Response: {e.response.text[:200]}")
            continue

        observed_pct = (obs_count > 0).sum() / (width * height) * 100
        print(f"  Cells observed: {observed_pct:.1f}% of map")

    # Save all collected observations for offline analysis
    _save_observations(all_observations, round_id)

    print("\n" + "=" * 50)
    print("All seeds submitted! Check results at app.ainm.no")
    print("=" * 50)


if __name__ == "__main__":
    main()
