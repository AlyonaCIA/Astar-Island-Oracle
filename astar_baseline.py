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


def compute_tile_grid(width, height, max_tile=15):
    """
    Non-overlapping tile partition covering the entire map.
    For a 40x40 map: 9 tiles (3x3) at x=[0,15,30], y=[0,15,30].
    Returns list of (x, y, w, h) tuples.
    """
    x_specs = []
    x = 0
    while x < width:
        w = min(max_tile, width - x)
        x_specs.append((x, w))
        x += w
    y_specs = []
    y = 0
    while y < height:
        h = min(max_tile, height - y)
        y_specs.append((y, h))
        y += h
    return [(tx, ty, tw, th) for (ty, th) in y_specs for (tx, tw) in x_specs]


def score_tile(initial_grid, tile, width, height):
    """
    Score a tile's priority for query ordering.
    Higher = more dynamic content = query first.
    """
    tx, ty, tw, th = tile
    score = 0.0
    n_settlements = 0
    for dy in range(th):
        for dx in range(tw):
            cell = initial_grid[ty + dy][tx + dx]
            if cell == 1:
                score += 5.0
                n_settlements += 1
            elif cell == 2:
                score += 6.0
                n_settlements += 1
            elif cell == 4:
                score += 0.3
            elif cell in (0, 11):
                score += 0.5
    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            cell = initial_grid[y][x]
            if cell not in (10, 5):
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if 0 <= ny < height and 0 <= nx < width:
                        if initial_grid[ny][nx] == 10:
                            score += 0.3
                            break
    if n_settlements >= 2:
        score += n_settlements * 1.0
    return score


def plan_viewports(width, height, num_queries, vw=15, vh=15):
    """Legacy wrapper — returns (vx, vy) list from the non-overlapping grid."""
    tiles = compute_tile_grid(width, height, max_tile=min(vw, vh))
    return [(tx, ty) for (tx, ty, tw, th) in tiles[:num_queries]]


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

    # Step 4: Build tile queue and query (stochastic — each query gives different results)
    tiles = compute_tile_grid(width, height)
    n_tiles = len(tiles)

    # Prepare per-seed accumulators
    seed_initial_preds = []
    seed_obs_preds = []
    seed_obs_counts = []
    for s in range(seeds_count):
        grid = initial_states[s]["grid"]
        seed_initial_preds.append(build_initial_prediction(grid, width, height))
        seed_obs_preds.append(np.zeros((height, width, NUM_CLASSES)))
        seed_obs_counts.append(np.zeros((height, width)))

    all_observations = []

    # Per-seed priority queues, sorted by dynamic content
    seed_queues = []
    for s in range(seeds_count):
        grid = initial_states[s]["grid"]
        scored = [(score_tile(grid, t, width, height), t) for t in tiles]
        scored.sort(reverse=True)
        seed_queues.append(scored)

    query_limit = remaining
    queries_done = 0

    # Step 5: Cover all tiles via round-robin, then re-query dynamic tiles
    total_needed = n_tiles * seeds_count
    print(f"\n--- Strategy (STOCHASTIC): {n_tiles} tiles/seed × {seeds_count} seeds = "
          f"{total_needed} for full coverage  (budget {query_limit}) ---")

    # Coverage pass: round-robin across seeds
    while queries_done < query_limit:
        queried_any = False
        for s in range(seeds_count):
            if queries_done >= query_limit:
                break
            if not seed_queues[s]:
                continue
            _, (tx, ty, tw, th) = seed_queues[s].pop(0)
            try:
                result = simulate(round_id, s, tx, ty, tw, th)
                update_prediction_with_observation(
                    seed_obs_preds[s], result["grid"], result["viewport"],
                    seed_obs_counts[s]
                )
                all_observations.append({
                    "seed_index": s,
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                })
                queries_done += 1
                queried_any = True
                tiles_left = len(seed_queues[s])
                used = result.get("queries_used", "?")
                max_q = result.get("queries_max", "?")
                print(f"  Seed {s} tile {n_tiles - tiles_left}/{n_tiles}: "
                      f"({tx},{ty}) {tw}x{th}  budget {used}/{max_q}")
                time.sleep(1.0)
                if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                    print("  Budget exhausted!")
                    break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"  Rate limited at seed {s}")
                    break
                raise
        if not queried_any:
            break

    # Extra pass: re-query most dynamic tiles with remaining budget
    if queries_done < query_limit:
        print(f"  Full coverage done ({queries_done} queries). "
              f"Using {query_limit - queries_done} extra queries on dynamic tiles.")
        extra_targets = []
        for s in range(seeds_count):
            grid = initial_states[s]["grid"]
            for obs in all_observations:
                if obs["seed_index"] != s:
                    continue
                vp = obs["viewport"]
                obs_grid = obs["grid"]
                changes = 0
                total = 0
                for dy in range(len(obs_grid)):
                    for dx in range(len(obs_grid[0])):
                        oy, ox = vp["y"] + dy, vp["x"] + dx
                        if 0 <= oy < height and 0 <= ox < width:
                            if terrain_to_class(grid[oy][ox]) != terrain_to_class(obs_grid[dy][dx]):
                                changes += 1
                            total += 1
                if total > 0:
                    extra_targets.append((changes / total, s,
                                          (vp["x"], vp["y"], vp["w"], vp["h"])))
        extra_targets.sort(reverse=True)

        for change_rate, s, (tx, ty, tw, th) in extra_targets:
            if queries_done >= query_limit:
                break
            try:
                result = simulate(round_id, s, tx, ty, tw, th)
                update_prediction_with_observation(
                    seed_obs_preds[s], result["grid"], result["viewport"],
                    seed_obs_counts[s]
                )
                all_observations.append({
                    "seed_index": s,
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                })
                queries_done += 1
                used = result.get("queries_used", "?")
                max_q = result.get("queries_max", "?")
                print(f"  Extra: seed {s} ({tx},{ty}) {tw}x{th}  "
                      f"change_rate={change_rate:.2f}  budget {used}/{max_q}")
                time.sleep(1.0)
                if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                    break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    break
                raise

    # Step 6: Finalize and submit predictions
    for seed_idx in range(seeds_count):
        final_pred = finalize_prediction(
            seed_obs_preds[seed_idx], seed_obs_counts[seed_idx],
            seed_initial_preds[seed_idx]
        )
        try:
            result = submit_prediction(round_id, seed_idx, final_pred)
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except requests.HTTPError as e:
            print(f"  Submit FAILED for seed {seed_idx}: {e}")
            if e.response is not None:
                print(f"    Response: {e.response.text[:200]}")
            continue

        observed_pct = (seed_obs_counts[seed_idx] > 0).sum() / (width * height) * 100
        print(f"  Cells observed: {observed_pct:.1f}% of map")

    # Save all collected observations for offline analysis
    _save_observations(all_observations, round_id)

    print(f"\n  Total queries: {queries_done}")
    print("=" * 50)
    print("All seeds submitted! Check results at app.ainm.no")
    print("=" * 50)


if __name__ == "__main__":
    main()
