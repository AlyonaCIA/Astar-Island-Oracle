"""
Astar Island — Quick Uniform Fallback Submission

Submits a safe uniform-ish distribution for all seeds immediately.
No simulation queries used. Scores low (~1-5) but guarantees a non-zero
score on the board and takes seconds to run.

Use this to get on the scoreboard fast, then improve with the other scripts.

IMPORTANT: Uses 1/6 per class (≈0.1667) which is safely above 0.0.
Never assigns 0.0 or 1.0 to any class.
"""

import os
import sys
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

BASE_URL = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("ASTAR_TOKEN")

if not TOKEN:
    print("ERROR: Set ASTAR_TOKEN in .env file or as environment variable.")
    print("  Copy .env.example to .env and fill in your JWT token.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

NUM_CLASSES = 6
PROB_FLOOR = 0.01


def main():
    print("=" * 50)
    print("  Astar Island — Quick Uniform Submission")
    print("=" * 50)

    # Get active round
    resp = session.get(f"{BASE_URL}/rounds")
    resp.raise_for_status()
    rounds = resp.json()
    active = next((r for r in rounds if r["status"] == "active"), None)

    if not active:
        print("No active round found. Available rounds:")
        for r in rounds:
            print(f"  Round {r['round_number']} — status: {r['status']}")
        sys.exit(0)

    round_id = active["id"]
    print(f"\nActive round #{active['round_number']} — {active['map_width']}x{active['map_height']}")

    # Get round details for dimensions and seed count
    resp = session.get(f"{BASE_URL}/rounds/{round_id}")
    resp.raise_for_status()
    detail = resp.json()
    width = detail["map_width"]
    height = detail["map_height"]
    seeds_count = detail["seeds_count"]

    # Build uniform prediction: 1/6 per class for every cell
    # This is safe — no zeros, no ones, sums to 1.0
    uniform = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES)

    # Enforce floor and renormalize (redundant here but safe habit)
    uniform = np.maximum(uniform, PROB_FLOOR)
    uniform = uniform / uniform.sum(axis=-1, keepdims=True)

    # Verify before submission
    assert uniform.shape == (height, width, NUM_CLASSES)
    assert np.allclose(uniform.sum(axis=-1), 1.0)
    assert uniform.min() >= PROB_FLOOR
    print(f"  Prediction: {height}x{width}x{NUM_CLASSES}, all cells = {uniform[0][0].tolist()}")

    # Submit for all seeds
    prediction_list = uniform.tolist()
    for seed_idx in range(seeds_count):
        try:
            resp = session.post(f"{BASE_URL}/submit", json={
                "round_id": round_id,
                "seed_index": seed_idx,
                "prediction": prediction_list,
            })
            resp.raise_for_status()
            result = resp.json()
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except requests.HTTPError as e:
            print(f"  Seed {seed_idx} FAILED: {e}")
            if e.response is not None:
                print(f"    {e.response.text[:200]}")

    print(f"\nDone! All {seeds_count} seeds submitted with uniform distribution.")
    print("Score will be low (~1-5) but non-zero. Use other scripts to improve.")


if __name__ == "__main__":
    main()
