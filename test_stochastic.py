"""
Quick test: Query the same viewport on the same seed twice to verify
that each /simulate call is stochastic (different outcome each time).

Usage:
    python test_stochastic.py
"""

import os
import sys
import json
import requests


def _load_dotenv():
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
    print("ERROR: Set ASTAR_TOKEN in .env or environment.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

TERRAIN_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
                 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}


def main():
    # Find active round
    rounds = session.get(f"{BASE_URL}/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round. This test requires an active round.")
        print("Available rounds:")
        for r in rounds:
            print(f"  Round {r['round_number']} — {r['status']}")
        sys.exit(0)

    round_id = active["id"]
    print(f"Active round #{active['round_number']} ({active['map_width']}x{active['map_height']})")

    # Check budget
    budget = session.get(f"{BASE_URL}/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} used ({remaining} remaining)")

    if remaining < 2:
        print("Need at least 2 queries remaining. Aborting.")
        sys.exit(1)

    print(f"\n*** This will use 2 queries from your budget! ***")
    confirm = input("Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # Query the same viewport on the same seed twice
    seed_idx = 0
    vx, vy, vw, vh = 10, 10, 15, 15
    print(f"\nQuerying seed {seed_idx}, viewport ({vx},{vy}) {vw}x{vh} — twice:")

    import time
    results = []
    for i in range(2):
        resp = session.post(f"{BASE_URL}/simulate", json={
            "round_id": round_id,
            "seed_index": seed_idx,
            "viewport_x": vx, "viewport_y": vy,
            "viewport_w": vw, "viewport_h": vh,
        })
        resp.raise_for_status()
        results.append(resp.json())
        used = resp.json().get("queries_used", "?")
        print(f"  Query {i+1}: OK (budget {used}/{budget['queries_max']})")
        time.sleep(1.0)

    # Compare the two grids
    g1 = results[0]["grid"]
    g2 = results[1]["grid"]
    h, w = len(g1), len(g1[0])

    identical = 0
    different = 0
    diff_cells = []
    for y in range(h):
        for x in range(w):
            if g1[y][x] == g2[y][x]:
                identical += 1
            else:
                different += 1
                if len(diff_cells) < 20:
                    diff_cells.append((vx + x, vy + y, g1[y][x], g2[y][x]))

    total = h * w
    print(f"\n{'=' * 60}")
    print(f"  Results:")
    print(f"    Viewport cells: {total}")
    print(f"    Identical:      {identical} ({identical/total*100:.1f}%)")
    print(f"    Different:      {different} ({different/total*100:.1f}%)")
    print(f"{'=' * 60}")

    if different == 0:
        print("\n  → Both queries returned IDENTICAL grids.")
        print("    This means either:")
        print("    a) The simulation is deterministic for this seed, OR")
        print("    b) This viewport has only static terrain (ocean/mountain/forest)")
    else:
        print(f"\n  → The grids DIFFER in {different} cells!")
        print("    This confirms each /simulate call is a different stochastic run.")
        print("\n  Sample differences (first 20):")
        print(f"    {'(x,y)':<10s} {'Query 1':<15s} {'Query 2':<15s}")
        for x, y, v1, v2 in diff_cells:
            n1 = TERRAIN_NAMES.get(v1, f"?{v1}")
            n2 = TERRAIN_NAMES.get(v2, f"?{v2}")
            print(f"    ({x},{y}){'':<5s} {n1:<15s} {n2:<15s}")

    # Settlement comparison
    s1 = {(s["x"], s["y"]): s for s in results[0].get("settlements", [])}
    s2 = {(s["x"], s["y"]): s for s in results[1].get("settlements", [])}
    all_pos = sorted(set(s1.keys()) | set(s2.keys()))
    if all_pos:
        print(f"\n  Settlement comparison ({len(all_pos)} unique positions):")
        print(f"    {'(x,y)':<10s} {'Q1 alive':<10s} {'Q2 alive':<10s} {'Q1 pop':<8s} {'Q2 pop':<8s}")
        for pos in all_pos[:15]:
            a1 = s1.get(pos, {}).get("alive", "—")
            a2 = s2.get(pos, {}).get("alive", "—")
            p1 = s1.get(pos, {}).get("population", "—")
            p2 = s2.get(pos, {}).get("population", "—")
            p1_str = f"{p1:.1f}" if isinstance(p1, (int, float)) else str(p1)
            p2_str = f"{p2:.1f}" if isinstance(p2, (int, float)) else str(p2)
            print(f"    {str(pos):<10s} {str(a1):<10s} {str(a2):<10s} {p1_str:<8s} {p2_str:<8s}")


if __name__ == "__main__":
    main()
