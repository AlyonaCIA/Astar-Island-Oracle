#!/usr/bin/env python3
"""
Data analysis script to verify two key assumptions:

1. STOCHASTIC CHECK: Overlapping viewports on the same seed produce different
   results (the platform does NOT use a fixed simulation seed per query).

2. REPLAY vs OBSERVATION: Compare simulation replay final frames with
   observation viewport data to see how they relate.
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict


DATA_DIR = "data"
REPLAY_DIR = "simulation_replays"


def load_observations():
    """Load all observation files, returning {round_id_prefix: [obs_list]}."""
    obs_by_round = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "observations_*.json"))):
        with open(path) as f:
            raw = json.load(f)
        obs_list = raw.get("observations", raw) if isinstance(raw, dict) else raw
        prefix = os.path.basename(path).replace("observations_", "").replace(".json", "")
        obs_by_round[prefix] = obs_list
    return obs_by_round


def load_round_info():
    """Load round data files, returning {round_id_prefix: round_data}."""
    rounds = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "round_*.json"))):
        with open(path) as f:
            data = json.load(f)
        prefix = data["id"][:8]
        rounds[prefix] = data
    return rounds


def load_replays():
    """Load all replay files, returning {(round_number, seed_index): replay_data}."""
    replays = {}
    for path in sorted(glob.glob(os.path.join(REPLAY_DIR, "r*s*.json"))):
        with open(path) as f:
            data = json.load(f)
        fname = os.path.basename(path).replace(".json", "")
        # Parse r{N}s{M} format
        parts = fname.split("s")
        round_num = int(parts[0][1:])
        seed_idx = int(parts[1]) - 1  # replay files are 1-indexed
        replays[(round_num, seed_idx)] = data
    return replays


# ─────────────────────────────────────────────────────────────────────
# Analysis 1: Stochastic viewport check
# ─────────────────────────────────────────────────────────────────────

def check_stochastic_viewports(obs_by_round):
    """
    Find overlapping viewports on the same seed within the same round.
    Compare cell values in the overlapping region to check if they differ.
    """
    print("=" * 70)
    print("ANALYSIS 1: STOCHASTIC CHECK — Do overlapping viewports differ?")
    print("=" * 70)

    total_pairs = 0
    pairs_with_differences = 0
    total_overlap_cells = 0
    total_differing_cells = 0

    for round_prefix, obs_list in sorted(obs_by_round.items()):
        # Group observations by seed
        by_seed = defaultdict(list)
        for obs in obs_list:
            by_seed[obs["seed_index"]].append(obs)

        round_pairs = 0
        round_diffs = 0

        for seed_idx in sorted(by_seed.keys()):
            seed_obs = by_seed[seed_idx]
            if len(seed_obs) < 2:
                continue

            # Check all pairs for overlaps
            for i in range(len(seed_obs)):
                for j in range(i + 1, len(seed_obs)):
                    vp_a = seed_obs[i]["viewport"]
                    vp_b = seed_obs[j]["viewport"]
                    grid_a = seed_obs[i]["grid"]
                    grid_b = seed_obs[j]["grid"]

                    # Compute overlap region
                    ox1 = max(vp_a["x"], vp_b["x"])
                    oy1 = max(vp_a["y"], vp_b["y"])
                    ox2 = min(vp_a["x"] + vp_a["w"], vp_b["x"] + vp_b["w"])
                    oy2 = min(vp_a["y"] + vp_a["h"], vp_b["y"] + vp_b["h"])

                    if ox1 >= ox2 or oy1 >= oy2:
                        continue  # no overlap

                    overlap_cells = 0
                    diff_cells = 0

                    for y in range(oy1, oy2):
                        for x in range(ox1, ox2):
                            val_a = grid_a[y - vp_a["y"]][x - vp_a["x"]]
                            val_b = grid_b[y - vp_b["y"]][x - vp_b["x"]]
                            overlap_cells += 1
                            if val_a != val_b:
                                diff_cells += 1

                    total_pairs += 1
                    total_overlap_cells += overlap_cells
                    total_differing_cells += diff_cells
                    round_pairs += 1

                    if diff_cells > 0:
                        pairs_with_differences += 1
                        round_diffs += 1

        if round_pairs > 0:
            print(f"\n  Round {round_prefix}:")
            print(f"    Overlapping pairs found: {round_pairs}")
            print(f"    Pairs with differences:  {round_diffs}")

            # Show detailed breakdown per seed
            for seed_idx in sorted(by_seed.keys()):
                seed_obs = by_seed[seed_idx]
                seed_pairs = 0
                seed_diffs = 0
                seed_overlap = 0
                seed_diff_cells = 0

                for i in range(len(seed_obs)):
                    for j in range(i + 1, len(seed_obs)):
                        vp_a = seed_obs[i]["viewport"]
                        vp_b = seed_obs[j]["viewport"]
                        grid_a = seed_obs[i]["grid"]
                        grid_b = seed_obs[j]["grid"]

                        ox1 = max(vp_a["x"], vp_b["x"])
                        oy1 = max(vp_a["y"], vp_b["y"])
                        ox2 = min(vp_a["x"] + vp_a["w"], vp_b["x"] + vp_b["w"])
                        oy2 = min(vp_a["y"] + vp_a["h"], vp_b["y"] + vp_b["h"])

                        if ox1 >= ox2 or oy1 >= oy2:
                            continue

                        n_overlap = (ox2 - ox1) * (oy2 - oy1)
                        n_diff = 0
                        for y in range(oy1, oy2):
                            for x in range(ox1, ox2):
                                val_a = grid_a[y - vp_a["y"]][x - vp_a["x"]]
                                val_b = grid_b[y - vp_b["y"]][x - vp_b["x"]]
                                if val_a != val_b:
                                    n_diff += 1

                        seed_pairs += 1
                        seed_overlap += n_overlap
                        seed_diff_cells += n_diff
                        if n_diff > 0:
                            seed_diffs += 1

                if seed_pairs > 0:
                    pct = seed_diff_cells / seed_overlap * 100 if seed_overlap > 0 else 0
                    print(f"      Seed {seed_idx}: {seed_pairs} pairs, "
                          f"{seed_diffs} with diffs, "
                          f"{seed_diff_cells}/{seed_overlap} cells differ ({pct:.1f}%)")

    print(f"\n  SUMMARY:")
    print(f"    Total overlapping viewport pairs: {total_pairs}")
    print(f"    Pairs with at least 1 difference: {pairs_with_differences}")
    if total_overlap_cells > 0:
        pct = total_differing_cells / total_overlap_cells * 100
        print(f"    Total overlapping cells:          {total_overlap_cells}")
        print(f"    Cells that differ:                {total_differing_cells} ({pct:.1f}%)")
    if total_pairs > 0:
        verdict = "YES — simulations are STOCHASTIC" if pairs_with_differences > 0 else "NO — simulations appear DETERMINISTIC"
        print(f"\n  VERDICT: {verdict}")
    else:
        print("\n  No overlapping viewport pairs found in the data.")


# ─────────────────────────────────────────────────────────────────────
# Analysis 2: Replay final frame vs observation comparison
# ─────────────────────────────────────────────────────────────────────

def compare_replays_vs_observations(obs_by_round, round_info, replays):
    """
    For each round+seed that has both a replay and observations,
    compare the replay's final frame grid with observation viewport grids.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: REPLAY FINAL FRAME vs OBSERVATION VIEWPORTS")
    print("=" * 70)

    # Build round_id_prefix → round_number mapping
    prefix_to_round_num = {}
    for prefix, rdata in round_info.items():
        rnum = rdata.get("round_number")
        if rnum is not None:
            prefix_to_round_num[prefix] = rnum

    print(f"\n  Round ID → Number mapping:")
    for prefix, rnum in sorted(prefix_to_round_num.items(), key=lambda x: x[1]):
        has_obs = prefix in obs_by_round
        has_replay = any((rnum, si) in replays for si in range(5))
        print(f"    {prefix} → round {rnum}  (obs: {'yes' if has_obs else 'no'}, "
              f"replays: {'yes' if has_replay else 'no'})")

    total_comparisons = 0
    total_cells_compared = 0
    total_cells_matching = 0
    total_cells_static_match = 0
    total_cells_static = 0
    total_cells_dynamic = 0
    total_cells_dynamic_match = 0

    # Static terrain codes (ocean=10, mountain=5)
    STATIC_CODES = {10, 5}

    for prefix in sorted(obs_by_round.keys()):
        if prefix not in prefix_to_round_num:
            continue
        round_num = prefix_to_round_num[prefix]
        obs_list = obs_by_round[prefix]

        # Get initial states for this round
        rdata = round_info.get(prefix)
        if not rdata:
            continue
        initial_states = rdata.get("initial_states", [])

        by_seed = defaultdict(list)
        for obs in obs_list:
            by_seed[obs["seed_index"]].append(obs)

        print(f"\n  Round {round_num} (id: {prefix}):")

        for seed_idx in sorted(by_seed.keys()):
            replay_key = (round_num, seed_idx)
            if replay_key not in replays:
                continue

            replay = replays[replay_key]
            final_grid = replay["frames"][-1]["grid"]
            initial_grid = initial_states[seed_idx]["grid"] if seed_idx < len(initial_states) else None

            seed_observations = by_seed[seed_idx]
            seed_cells = 0
            seed_match = 0
            seed_static = 0
            seed_static_match = 0
            seed_dynamic = 0
            seed_dynamic_match = 0

            for obs in seed_observations:
                vp = obs["viewport"]
                obs_grid = obs["grid"]

                for dy in range(vp["h"]):
                    for dx in range(vp["w"]):
                        y, x = vp["y"] + dy, vp["x"] + dx
                        if y >= len(final_grid) or x >= len(final_grid[0]):
                            continue

                        obs_val = obs_grid[dy][dx]
                        replay_val = final_grid[y][x]
                        is_static = initial_grid and initial_grid[y][x] in STATIC_CODES

                        seed_cells += 1
                        if obs_val == replay_val:
                            seed_match += 1

                        if is_static:
                            seed_static += 1
                            if obs_val == replay_val:
                                seed_static_match += 1
                        else:
                            seed_dynamic += 1
                            if obs_val == replay_val:
                                seed_dynamic_match += 1

                total_comparisons += 1

            total_cells_compared += seed_cells
            total_cells_matching += seed_match
            total_cells_static += seed_static
            total_cells_static_match += seed_static_match
            total_cells_dynamic += seed_dynamic
            total_cells_dynamic_match += seed_dynamic_match

            overall_pct = seed_match / seed_cells * 100 if seed_cells > 0 else 0
            static_pct = seed_static_match / seed_static * 100 if seed_static > 0 else 0
            dynamic_pct = seed_dynamic_match / seed_dynamic * 100 if seed_dynamic > 0 else 0

            print(f"    Seed {seed_idx}: {len(seed_observations)} viewports, "
                  f"{seed_cells} cells compared")
            print(f"      Overall match:  {seed_match}/{seed_cells} ({overall_pct:.1f}%)")
            print(f"      Static cells:   {seed_static_match}/{seed_static} ({static_pct:.1f}%) "
                  f"— ocean/mountain should always match")
            print(f"      Dynamic cells:  {seed_dynamic_match}/{seed_dynamic} ({dynamic_pct:.1f}%) "
                  f"— differ = different simulation seeds")

    print(f"\n  SUMMARY:")
    if total_cells_compared > 0:
        pct_all = total_cells_matching / total_cells_compared * 100
        pct_static = total_cells_static_match / total_cells_static * 100 if total_cells_static > 0 else 0
        pct_dynamic = total_cells_dynamic_match / total_cells_dynamic * 100 if total_cells_dynamic > 0 else 0
        print(f"    Total cells compared:   {total_cells_compared}")
        print(f"    Overall match rate:     {pct_all:.1f}%")
        print(f"    Static cell match:      {pct_static:.1f}% (expect ~100%)")
        print(f"    Dynamic cell match:     {pct_dynamic:.1f}% (expect <100% if stochastic)")

        if pct_static > 99.0 and pct_dynamic < 95.0:
            print(f"\n  VERDICT: Replays and observations are INDEPENDENT stochastic runs.")
            print(f"           Static terrain is consistent, dynamic terrain diverges.")
        elif pct_all > 99.0:
            print(f"\n  VERDICT: Replays and observations appear to use the SAME simulation seed.")
        else:
            print(f"\n  VERDICT: Mixed results — check the per-seed breakdown above.")
    else:
        print("    No matching replay+observation pairs found.")


# ─────────────────────────────────────────────────────────────────────
# Analysis 3: Per-cell value distribution from overlapping observations
# ─────────────────────────────────────────────────────────────────────

def show_overlap_distributions(obs_by_round):
    """
    For cells observed multiple times (same seed), show the distribution
    of values seen — demonstrates stochastic variety.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PER-CELL VALUE DISTRIBUTIONS FROM REPEATED OBSERVATIONS")
    print("=" * 70)

    # Terrain code names for readable output
    CODE_NAMES = {
        0: "empty", 1: "settlement", 2: "port", 3: "ruin",
        4: "forest", 5: "mountain", 10: "ocean", 11: "plains"
    }

    for round_prefix, obs_list in sorted(obs_by_round.items()):
        by_seed = defaultdict(list)
        for obs in obs_list:
            by_seed[obs["seed_index"]].append(obs)

        has_multi = False
        for seed_idx in sorted(by_seed.keys()):
            seed_obs = by_seed[seed_idx]
            if len(seed_obs) < 2:
                continue

            # Build per-cell observation counts
            cell_values = defaultdict(list)  # (y, x) → [val1, val2, ...]
            for obs in seed_obs:
                vp = obs["viewport"]
                for dy in range(vp["h"]):
                    for dx in range(vp["w"]):
                        y, x = vp["y"] + dy, vp["x"] + dx
                        cell_values[(y, x)].append(obs["grid"][dy][dx])

            # Find cells observed multiple times with varying values
            multi_obs = {k: v for k, v in cell_values.items() if len(v) >= 2}
            varying = {k: v for k, v in multi_obs.items() if len(set(v)) > 1}

            if not multi_obs:
                continue

            if not has_multi:
                print(f"\n  Round {round_prefix}:")
                has_multi = True

            n_varying = len(varying)
            n_multi = len(multi_obs)
            print(f"    Seed {seed_idx}: {n_multi} cells observed 2+ times, "
                  f"{n_varying} show variation ({n_varying/n_multi*100:.1f}%)")

            # Show top 5 most variable cells
            if varying:
                examples = sorted(varying.items(),
                                  key=lambda kv: len(set(kv[1])), reverse=True)[:5]
                for (y, x), vals in examples:
                    counts = defaultdict(int)
                    for v in vals:
                        counts[v] += 1
                    dist_str = ", ".join(
                        f"{CODE_NAMES.get(v, f'code_{v}')}:{c}"
                        for v, c in sorted(counts.items(), key=lambda x: -x[1])
                    )
                    print(f"        Cell ({x},{y}): {len(vals)} obs → {dist_str}")

        if not has_multi:
            print(f"\n  Round {round_prefix}: no overlapping observations found")


def main():
    print("Loading data...")
    obs_by_round = load_observations()
    round_info = load_round_info()
    replays = load_replays()

    print(f"  Observation files: {len(obs_by_round)}")
    print(f"  Round info files:  {len(round_info)}")
    print(f"  Replay files:      {len(replays)}")

    check_stochastic_viewports(obs_by_round)
    compare_replays_vs_observations(obs_by_round, round_info, replays)
    show_overlap_distributions(obs_by_round)


if __name__ == "__main__":
    main()
