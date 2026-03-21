"""
Fetch ground truth for a specific round from the Astar Island API.

Usage:
    python fetch_ground_truth.py 10
    python fetch_ground_truth.py 10 --seeds 5
"""

import os
import sys
import json
import time
import argparse
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
GT_DIR = os.path.join(DATA_DIR, "ground_truth")

# ---------------------------------------------------------------------------
# .env loader + API setup
# ---------------------------------------------------------------------------

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
    print("ERROR: Set ASTAR_TOKEN in .env or as environment variable.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"
API_TIMEOUT = 30


def fetch_rounds():
    resp = session.get(f"{BASE_URL}/rounds", timeout=API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_round_details(round_id):
    resp = session.get(f"{BASE_URL}/rounds/{round_id}", timeout=API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_analysis(round_id, seed_index):
    resp = session.get(f"{BASE_URL}/analysis/{round_id}/{seed_index}",
                       timeout=API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ground truth for a specific round")
    parser.add_argument("round", type=int, help="Round number (e.g. 10)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds (auto-detected if omitted)")
    args = parser.parse_args()

    os.makedirs(GT_DIR, exist_ok=True)

    # Find the round by round_number
    print(f"Fetching rounds list...")
    rounds = fetch_rounds()
    time.sleep(1.0)

    match = None
    for r in rounds:
        if r.get("round_number") == args.round:
            match = r
            break

    if not match:
        print(f"ERROR: Round {args.round} not found. "
              f"Available: {sorted(r['round_number'] for r in rounds)}")
        sys.exit(1)

    round_id = match["id"]
    round_key = round_id[:8]
    status = match.get("status", "unknown")
    seeds_count = args.seeds or match.get("seeds_count", 5)

    print(f"Round {args.round} ({round_key}) — status: {status}, "
          f"seeds: {seeds_count}")

    # Fetch round details (initial states)
    print(f"Fetching round details...")
    try:
        detail = fetch_round_details(round_id)
    except Exception as e:
        print(f"WARNING: Could not fetch round details: {e}")
        detail = None
    time.sleep(1.0)

    # Save round details
    if detail:
        round_path = os.path.join(DATA_DIR, f"round_{round_key}.json")
        with open(round_path, "w") as f:
            json.dump(detail, f)
        print(f"  Saved round details to {round_path}")

    # Fetch ground truth for each seed
    fetched = 0
    for seed_idx in range(seeds_count):
        fname = f"r{args.round}_s{seed_idx}_{round_key}.json"
        fpath = os.path.join(GT_DIR, fname)

        print(f"  Fetching seed {seed_idx}...", end=" ", flush=True)
        try:
            analysis = fetch_analysis(round_id, seed_idx)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                print(f"skipped (HTTP 400 — not available yet)")
            else:
                print(f"failed: {e}")
            time.sleep(1.0)
            continue
        except Exception as e:
            print(f"failed: {e}")
            time.sleep(1.0)
            continue
        time.sleep(1.0)

        gt_record = {
            "ground_truth": analysis.get("ground_truth"),
            "initial_grid": analysis.get("initial_grid"),
            "width": analysis.get("width", match.get("map_width", 40)),
            "height": analysis.get("height", match.get("map_height", 40)),
            "score": analysis.get("score"),
            "prediction": analysis.get("prediction"),
        }

        # Fill initial_grid from round details if missing
        if gt_record["initial_grid"] is None and detail:
            init_states = detail.get("initial_states", [])
            if seed_idx < len(init_states):
                gt_record["initial_grid"] = init_states[seed_idx].get("grid")

        has_gt = gt_record["ground_truth"] is not None
        has_grid = gt_record["initial_grid"] is not None

        with open(fpath, "w") as f:
            json.dump(gt_record, f)
        fetched += 1
        print(f"OK → {fname} (gt={'yes' if has_gt else 'NO'}, "
              f"grid={'yes' if has_grid else 'NO'})")

    print(f"\nDone. Fetched {fetched}/{seeds_count} seeds to {GT_DIR}")


if __name__ == "__main__":
    main()
