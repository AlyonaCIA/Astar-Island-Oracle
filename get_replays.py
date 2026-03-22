"""
Download simulation replays from the Astar Island API.

Fetches 5 replays (seed_index 0-4) for each completed round
and stores them in the replays/ folder. Each file includes a
timestamp so re-running always adds new files without overwriting.

Usage:
    python get_replays.py              # all completed rounds
    python get_replays.py --round 5    # specific round number
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPLAYS_DIR = os.path.join(SCRIPT_DIR, "replays")

SEEDS_PER_ROUND = 5

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

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_rounds():
    resp = session.get(f"{BASE_URL}/rounds", timeout=API_TIMEOUT)
    resp.raise_for_status()
    time.sleep(random.uniform(3, 6))
    return resp.json()


def fetch_replay(round_id, seed_index):
    resp = session.post(
        f"{BASE_URL}/replay",
        json={"round_id": round_id, "seed_index": seed_index},
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    time.sleep(random.uniform(3, 6))
    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def replay_filename(round_number, seed_index, timestamp):
    return f"r{round_number}_s{seed_index}_{timestamp}.json"


def download_replays(target_round=None):
    os.makedirs(REPLAYS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Fetching round list...")
    rounds = fetch_rounds()
    completed = [r for r in rounds if r["status"] == "completed"]
    completed.sort(key=lambda r: r["round_number"])

    if target_round is not None:
        completed = [r for r in completed if r["round_number"] == target_round]
        if not completed:
            print(f"ERROR: Round {target_round} not found or not completed.")
            sys.exit(1)

    print(f"Found {len(completed)} completed round(s).")

    total = len(completed) * SEEDS_PER_ROUND
    downloaded = 0
    errors = 0

    for rnd in completed:
        round_id = rnd["id"]
        round_number = rnd["round_number"]
        print(f"\n--- Round {round_number} (id={round_id}) ---")

        for seed_idx in range(SEEDS_PER_ROUND):
            fname = replay_filename(round_number, seed_idx, timestamp)
            fpath = os.path.join(REPLAYS_DIR, fname)

            try:
                print(f"  Downloading {fname} ...", end=" ", flush=True)
                data = fetch_replay(round_id, seed_idx)
                with open(fpath, "w") as f:
                    json.dump(data, f)
                print("OK")
                downloaded += 1
            except requests.HTTPError as e:
                print(f"FAILED ({e.response.status_code})")
                errors += 1
            except Exception as e:
                print(f"FAILED ({e})")
                errors += 1

    print(f"\nDone. Downloaded: {downloaded}, Errors: {errors} (Total: {total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Astar Island replays")
    parser.add_argument("--round", type=int, default=None,
                        help="Download only this round number")
    args = parser.parse_args()

    download_replays(target_round=args.round)
