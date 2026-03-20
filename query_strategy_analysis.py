"""
Astar Island — Query Strategy Analysis

Simulates different viewport query strategies offline against ground truth data
and compares their effectiveness under two regimes:
  A) DETERMINISTIC — each /simulate call for the same seed gives the same result
  B) STOCHASTIC    — each /simulate call is a different random run

Uses the actual scoring metric (entropy-weighted KL divergence) to measure
how well each strategy enables predictions.

Deterministic strategies (re-querying same tile = waste):
  det_full_coverage   — 9 tiles, 1 per seed, no repeats
  det_skip_static     — skip all-ocean/mountain tiles, cover more seeds

Stochastic strategies (re-querying = more distribution samples):
  stoch_cover_then_x1 — full coverage, then 1 extra pass on dynamic tiles
  stoch_cover_then_x3 — full coverage (skip static), re-query dynamic tiles 3×
  stoch_all_dynamic    — skip static tiles entirely, maximize samples on dynamic

Both modes:
  uniform_prior       — no queries at all, just terrain-based priors
  old_overlap          — old overlapping grid with out-of-bounds issues
  new_priority_extra   — non-overlapping partition + dynamic repeats
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import load_local_data, terrain_to_class, NUM_CLASSES, PROB_FLOOR

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "ground_truth")


# ---------------------------------------------------------------------------
# Scoring (standalone — no torch dependency)
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
    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * wkl))), wkl


# ---------------------------------------------------------------------------
# Prior prediction (same as astar_baseline / astar_cnn)
# ---------------------------------------------------------------------------

def build_prior(initial_grid, width, height):
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
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def sample_observation(ground_truth, tx, ty, tw, th):
    """
    Simulate one stochastic observation by sampling a class per cell from the
    ground truth distribution.  This mimics what a real /simulate call returns.
    """
    obs = np.zeros((th, tw), dtype=np.int32)
    for dy in range(th):
        for dx in range(tw):
            probs = ground_truth[ty + dy][tx + dx]
            obs[dy, dx] = np.random.choice(NUM_CLASSES, p=np.array(probs) / np.sum(probs))
    return obs


def blend_observations(prior, observations, width, height):
    """
    Blend a list of sampled observations into the prior.
    Each observation is (obs_grid, tx, ty, tw, th).
    Uses the same logic as astar_baseline: count observations, then blend.
    """
    obs_counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    obs_n = np.zeros((height, width), dtype=np.float32)

    for obs_grid, tx, ty, tw, th in observations:
        for dy in range(th):
            for dx in range(tw):
                y, x = ty + dy, tx + dx
                cls = obs_grid[dy, dx]
                obs_counts[y, x, cls] += 1.0
                obs_n[y, x] += 1.0

    result = np.copy(prior)
    for y in range(height):
        for x in range(width):
            n = obs_n[y, x]
            if n > 0:
                obs_dist = obs_counts[y, x] + PROB_FLOOR
                obs_dist /= obs_dist.sum()
                # More observations → more weight on observed data
                alpha = n / (n + 2.0)
                result[y, x] = alpha * obs_dist + (1 - alpha) * prior[y, x]

    result = np.maximum(result, PROB_FLOOR)
    result /= result.sum(axis=-1, keepdims=True)
    return result


# ---------------------------------------------------------------------------
# Tile generation strategies
# ---------------------------------------------------------------------------

def compute_tile_grid(width, height, max_tile=15):
    """Non-overlapping partition. 9 tiles for 40x40."""
    x_specs, y_specs = [], []
    x = 0
    while x < width:
        w = min(max_tile, width - x)
        x_specs.append((x, w))
        x += w
    y = 0
    while y < height:
        h = min(max_tile, height - y)
        y_specs.append((y, h))
        y += h
    return [(tx, ty, tw, th) for (ty, th) in y_specs for (tx, tw) in x_specs]


def old_overlap_viewports(width, height, num_queries, vw=15, vh=15):
    """Old strategy: overlapping grid with step=13, center tile first."""
    viewports = []
    y_positions = list(range(0, height, vh - 2))  # step=13
    x_positions = list(range(0, width, vw - 2))
    for vy in y_positions:
        for vx in x_positions:
            viewports.append((vx, vy, vw, vh))
    cx, cy = (width - vw) // 2, (height - vh) // 2
    viewports.insert(0, (cx, cy, vw, vh))
    seen = set()
    unique = []
    for v in viewports:
        key = (v[0], v[1])
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique[:num_queries]


def score_tile(initial_grid, tile, width, height):
    tx, ty, tw, th = tile
    score = 0.0
    n_settlements = 0
    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            if y >= height or x >= width:
                continue
            cell = initial_grid[y][x]
            if cell == 1:
                score += 5.0; n_settlements += 1
            elif cell == 2:
                score += 6.0; n_settlements += 1
            elif cell == 4:
                score += 0.3
            elif cell in (0, 11):
                score += 0.5
    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            if y >= height or x >= width:
                continue
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


# ---------------------------------------------------------------------------
# Observation blending (simulates what the baseline does with real queries,
# but here we sample "observations" from the ground truth distribution)
# ---------------------------------------------------------------------------

def deterministic_observation(ground_truth, tx, ty, tw, th):
    """
    Simulate a deterministic observation: return the argmax class per cell.
    In a deterministic world, each cell always resolves to its most likely class.
    """
    obs = np.zeros((th, tw), dtype=np.int32)
    for dy in range(th):
        for dx in range(tw):
            probs = np.array(ground_truth[ty + dy][tx + dx])
            obs[dy, dx] = np.argmax(probs)
    return obs


def tile_static_fraction(initial_grid, tile, height, width):
    """Fraction of tile cells that are fully static (ocean or mountain)."""
    tx, ty, tw, th = tile
    static = 0
    for dy in range(th):
        for dx in range(tw):
            cell = initial_grid[ty + dy][tx + dx]
            if cell in (10, 5):  # ocean or mountain
                static += 1
    return static / (tw * th)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_uniform_prior(initial_grid, ground_truth, width, height, budget,
                           deterministic=False):
    """No queries — just the prior."""
    prior = build_prior(initial_grid, width, height)
    return prior, 0, {"coverage": 0.0}


def strategy_old_overlap(initial_grid, ground_truth, width, height, budget,
                         deterministic=False):
    """Old overlapping viewport strategy."""
    prior = build_prior(initial_grid, width, height)
    viewports = old_overlap_viewports(width, height, budget)
    obs_fn = deterministic_observation if deterministic else sample_observation
    observations = []
    covered = set()
    for tx, ty, tw, th in viewports:
        tw_actual = min(tw, width - tx)
        th_actual = min(th, height - ty)
        if tw_actual <= 0 or th_actual <= 0:
            continue
        obs = obs_fn(ground_truth, tx, ty, tw_actual, th_actual)
        observations.append((obs, tx, ty, tw_actual, th_actual))
        for dy in range(th_actual):
            for dx in range(tw_actual):
                covered.add((ty + dy, tx + dx))
    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    return pred, len(observations), {"coverage": coverage}


# --- Deterministic strategies ---

def strategy_det_full_coverage(initial_grid, ground_truth, width, height, budget,
                               deterministic=False):
    """Deterministic: cover all 9 tiles once, extra queries wasted."""
    prior = build_prior(initial_grid, width, height)
    tiles = compute_tile_grid(width, height)
    scored = [(score_tile(initial_grid, t, width, height), t) for t in tiles]
    scored.sort(reverse=True)
    observations = []
    covered = set()
    for _, (tx, ty, tw, th) in scored[:budget]:
        obs = deterministic_observation(ground_truth, tx, ty, tw, th)
        observations.append((obs, tx, ty, tw, th))
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))
    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    return pred, len(observations), {"coverage": coverage}


def strategy_det_skip_static(initial_grid, ground_truth, width, height, budget,
                             deterministic=False):
    """
    Deterministic: skip tiles that are >90% static (ocean/mountain) — those
    are perfectly predicted by the prior. Spread saved queries to other seeds
    (simulated here as extra coverage of the same seed's dynamic tiles).
    """
    prior = build_prior(initial_grid, width, height)
    tiles = compute_tile_grid(width, height)
    # Split into dynamic and static
    dynamic_tiles = []
    static_tiles = []
    for t in tiles:
        sf = tile_static_fraction(initial_grid, t, height, width)
        sc = score_tile(initial_grid, t, width, height)
        if sf > 0.90:
            static_tiles.append((sc, t))
        else:
            dynamic_tiles.append((sc, t))
    dynamic_tiles.sort(reverse=True)
    static_tiles.sort(reverse=True)

    observations = []
    covered = set()
    queries = 0
    # Cover dynamic tiles first
    for _, (tx, ty, tw, th) in dynamic_tiles:
        if queries >= budget:
            break
        obs = deterministic_observation(ground_truth, tx, ty, tw, th)
        observations.append((obs, tx, ty, tw, th))
        queries += 1
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))
    # Then static if budget remains (re-querying dynamic = waste in det mode)
    for _, (tx, ty, tw, th) in static_tiles:
        if queries >= budget:
            break
        obs = deterministic_observation(ground_truth, tx, ty, tw, th)
        observations.append((obs, tx, ty, tw, th))
        queries += 1
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))

    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    skipped = len(static_tiles)
    return pred, queries, {"coverage": coverage,
                           "static_tiles_skipped": skipped}


# --- Stochastic strategies ---

def strategy_stoch_cover_then_x1(initial_grid, ground_truth, width, height,
                                  budget, deterministic=False):
    """Stochastic: full coverage (9 tiles) + 1 extra pass on most dynamic."""
    prior = build_prior(initial_grid, width, height)
    tiles = compute_tile_grid(width, height)
    scored = [(score_tile(initial_grid, t, width, height), t) for t in tiles]
    scored.sort(reverse=True)
    observations = []
    covered = set()
    tile_dynamics = []

    # First pass: cover all
    n_first = min(budget, len(scored))
    for _, (tx, ty, tw, th) in scored[:n_first]:
        obs = sample_observation(ground_truth, tx, ty, tw, th)
        observations.append((obs, tx, ty, tw, th))
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))
        changes = sum(1 for dy in range(th) for dx in range(tw)
                      if terrain_to_class(initial_grid[ty+dy][tx+dx]) != obs[dy, dx])
        tile_dynamics.append((changes / (tw * th), tx, ty, tw, th))

    # Extra pass: re-query most dynamic
    remaining = budget - n_first
    if remaining > 0:
        tile_dynamics.sort(reverse=True)
        for _, tx, ty, tw, th in tile_dynamics[:remaining]:
            obs = sample_observation(ground_truth, tx, ty, tw, th)
            observations.append((obs, tx, ty, tw, th))

    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    return pred, len(observations), {"coverage": coverage,
                                      "extra_queries": max(0, remaining)}


def strategy_stoch_skip_static_x3(initial_grid, ground_truth, width, height,
                                   budget, deterministic=False):
    """
    Stochastic: skip >90% static tiles (prior handles them), spend ALL
    saved queries on dynamic tiles — each gets ~3× observations.
    """
    prior = build_prior(initial_grid, width, height)
    tiles = compute_tile_grid(width, height)
    dynamic_tiles = []
    for t in tiles:
        sf = tile_static_fraction(initial_grid, t, height, width)
        sc = score_tile(initial_grid, t, width, height)
        if sf <= 0.90:
            dynamic_tiles.append((sc, t))
    dynamic_tiles.sort(reverse=True)
    n_dynamic = len(dynamic_tiles)

    observations = []
    covered = set()
    queries = 0

    if n_dynamic == 0:
        pred = build_prior(initial_grid, width, height)
        return pred, 0, {"coverage": 0.0, "samples_per_tile": 0}

    # Distribute budget across dynamic tiles round-robin
    repeats_per_tile = max(1, budget // n_dynamic)
    leftover = budget - repeats_per_tile * n_dynamic

    for idx, (_, (tx, ty, tw, th)) in enumerate(dynamic_tiles):
        n_obs = repeats_per_tile + (1 if idx < leftover else 0)
        for _ in range(n_obs):
            if queries >= budget:
                break
            obs = sample_observation(ground_truth, tx, ty, tw, th)
            observations.append((obs, tx, ty, tw, th))
            queries += 1
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))

    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    return pred, queries, {"coverage": coverage,
                           "samples_per_tile": repeats_per_tile,
                           "dynamic_tiles": n_dynamic}


def strategy_stoch_all_dynamic(initial_grid, ground_truth, width, height,
                                budget, deterministic=False):
    """
    Stochastic: rank ALL tiles by dynamism, query top tiles multiple times.
    No static skipping — lets the scoring naturally sort tiles.
    """
    prior = build_prior(initial_grid, width, height)
    tiles = compute_tile_grid(width, height)
    scored = [(score_tile(initial_grid, t, width, height), t) for t in tiles]
    scored.sort(reverse=True)

    observations = []
    covered = set()
    queries = 0

    # Round-robin: keep cycling through tiles in priority order
    tile_idx = 0
    while queries < budget:
        _, (tx, ty, tw, th) = scored[tile_idx % len(scored)]
        obs = sample_observation(ground_truth, tx, ty, tw, th)
        observations.append((obs, tx, ty, tw, th))
        queries += 1
        for dy in range(th):
            for dx in range(tw):
                covered.add((ty + dy, tx + dx))
        tile_idx += 1

    pred = blend_observations(prior, observations, width, height)
    coverage = len(covered) / (width * height) * 100
    return pred, queries, {"coverage": coverage}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

BASELINE_STRATEGIES = [
    ("uniform_prior",       strategy_uniform_prior),
    ("old_overlap",         strategy_old_overlap),
]

DET_STRATEGIES = [
    ("det_full_coverage",   strategy_det_full_coverage),
    ("det_skip_static",     strategy_det_skip_static),
]

STOCH_STRATEGIES = [
    ("stoch_cover_then_x1",  strategy_stoch_cover_then_x1),
    ("stoch_skip_static_x3", strategy_stoch_skip_static_x3),
    ("stoch_all_dynamic",    strategy_stoch_all_dynamic),
]


def analyze_single_seed(data, budget, n_trials=20):
    """
    Run all strategies on one seed for both deterministic and stochastic
    regimes, averaged over n_trials Monte Carlo trials.
    Returns dict: strategy_name → {score, wkl, queries, coverage, ...}
    """
    initial_grid = data["initial_grid"]
    ground_truth = np.array(data["ground_truth"], dtype=np.float64)
    width = data["width"]
    height = data["height"]

    all_strategies = (BASELINE_STRATEGIES + DET_STRATEGIES + STOCH_STRATEGIES)

    results = {}
    for name, strategy_fn in all_strategies:
        is_det = name.startswith("det_")
        scores, wkls = [], []
        queries_used = 0
        meta = {}
        for _ in range(n_trials):
            pred, q, m = strategy_fn(initial_grid, ground_truth,
                                     width, height, budget,
                                     deterministic=is_det)
            sc, wk = competition_score(pred, ground_truth)
            scores.append(sc)
            wkls.append(wk)
            queries_used = q
            meta = m
        results[name] = {
            "score_mean": np.mean(scores),
            "score_std": np.std(scores),
            "wkl_mean": np.mean(wkls),
            "wkl_std": np.std(wkls),
            "queries": queries_used,
            **meta,
        }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare viewport query strategies")
    parser.add_argument("--budget", type=int, default=10,
                        help="Queries per seed to simulate (default: 10 = 50/5 seeds)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Monte Carlo trials per strategy/seed (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    all_strategies = BASELINE_STRATEGIES + DET_STRATEGIES + STOCH_STRATEGIES
    all_names = [n for n, _ in all_strategies]

    print("=" * 70)
    print("  Astar Island — Query Strategy Comparison (Dual-Mode)")
    print("=" * 70)
    print(f"  Budget per seed: {args.budget} queries")
    print(f"    (incl. 2 detection queries → {args.budget - 2} effective)")
    print(f"  Monte Carlo trials: {args.trials}")
    print(f"  Random seed: {args.seed}")

    all_data = load_local_data()
    if not all_data:
        print("\nNo ground truth data found. Run: python train_cnn.py --fetch")
        sys.exit(1)

    print(f"\n  Ground truth samples: {len(all_data)}")

    # Group by round
    rounds = {}
    for d in all_data:
        rnum = d.get("_round_number", "?")
        if rnum not in rounds:
            rounds[rnum] = []
        rounds[rnum].append(d)

    # Collect results across all seeds
    strategy_scores = {name: [] for name in all_names}
    strategy_wkls = {name: [] for name in all_names}

    for rnum in sorted(rounds.keys()):
        seeds = rounds[rnum]
        print(f"\n{'─' * 70}")
        print(f"  Round {rnum} — {len(seeds)} seeds")
        print(f"{'─' * 70}")

        for data in seeds:
            sid = data.get("_seed_index", "?")
            results = analyze_single_seed(data, args.budget, args.trials)

            # Print deterministic strategies
            print(f"\n  Seed {sid} — DETERMINISTIC regime:")
            print(f"    {'Strategy':<25s} {'Score':>8s} {'±Std':>6s}"
                  f"  {'wKL':>8s}  {'Queries':>7s}  {'Coverage':>8s}")
            print(f"    {'─' * 66}")
            for name, _ in (BASELINE_STRATEGIES + DET_STRATEGIES):
                r = results[name]
                cov = r.get("coverage", 0)
                cov_str = f"{cov:5.1f}%"
                print(f"    {name:<25s} {r['score_mean']:8.2f} {r['score_std']:6.2f}"
                      f"  {r['wkl_mean']:8.4f}  {r['queries']:>7d}  {cov_str:>8s}")
                strategy_scores[name].append(r["score_mean"])
                strategy_wkls[name].append(r["wkl_mean"])

            # Print stochastic strategies
            print(f"\n  Seed {sid} — STOCHASTIC regime:")
            print(f"    {'Strategy':<25s} {'Score':>8s} {'±Std':>6s}"
                  f"  {'wKL':>8s}  {'Queries':>7s}  {'Coverage':>8s}")
            print(f"    {'─' * 66}")
            for name, _ in (BASELINE_STRATEGIES + STOCH_STRATEGIES):
                r = results[name]
                cov = r.get("coverage", 0)
                extra = r.get("extra_queries", 0)
                cov_str = f"{cov:5.1f}%"
                if extra > 0:
                    cov_str += f" +{extra}x"
                # Skip baselines that were already printed (just reference them)
                if name in [n for n, _ in BASELINE_STRATEGIES]:
                    # Already collected above for det, skip collecting again
                    pass
                else:
                    strategy_scores[name].append(r["score_mean"])
                    strategy_wkls[name].append(r["wkl_mean"])
                print(f"    {name:<25s} {r['score_mean']:8.2f} {r['score_std']:6.2f}"
                      f"  {r['wkl_mean']:8.4f}  {r['queries']:>7d}  {cov_str:>8s}")

    # Summary
    n_seeds = sum(len(s) for s in rounds.values())
    print(f"\n{'=' * 70}")
    print(f"  OVERALL AVERAGES (across {n_seeds} seeds)")
    print(f"{'=' * 70}")

    prior_avg = np.mean(strategy_scores["uniform_prior"])

    print(f"\n  Baselines:")
    print(f"  {'Strategy':<25s} {'Avg Score':>10s}  {'Avg wKL':>10s}  {'Δ vs prior':>10s}")
    print(f"  {'─' * 60}")
    for name, _ in BASELINE_STRATEGIES:
        avg_score = np.mean(strategy_scores[name])
        avg_wkl = np.mean(strategy_wkls[name])
        delta = avg_score - prior_avg
        delta_str = f"{delta:+.2f}" if name != "uniform_prior" else "—"
        print(f"  {name:<25s} {avg_score:10.2f}  {avg_wkl:10.4f}  {delta_str:>10s}")

    print(f"\n  Deterministic strategies (if /simulate is deterministic):")
    print(f"  {'Strategy':<25s} {'Avg Score':>10s}  {'Avg wKL':>10s}  {'Δ vs prior':>10s}")
    print(f"  {'─' * 60}")
    for name, _ in DET_STRATEGIES:
        avg_score = np.mean(strategy_scores[name])
        avg_wkl = np.mean(strategy_wkls[name])
        delta = avg_score - prior_avg
        print(f"  {name:<25s} {avg_score:10.2f}  {avg_wkl:10.4f}  {delta:>+10.2f}")

    print(f"\n  Stochastic strategies (if /simulate is stochastic):")
    print(f"  {'Strategy':<25s} {'Avg Score':>10s}  {'Avg wKL':>10s}  {'Δ vs prior':>10s}")
    print(f"  {'─' * 60}")
    for name, _ in STOCH_STRATEGIES:
        avg_score = np.mean(strategy_scores[name])
        avg_wkl = np.mean(strategy_wkls[name])
        delta = avg_score - prior_avg
        print(f"  {name:<25s} {avg_score:10.2f}  {avg_wkl:10.4f}  {delta:>+10.2f}")

    # Tile geometry summary
    print(f"\n{'─' * 70}")
    print("  Tile geometry (40×40 map):")
    tiles = compute_tile_grid(40, 40)
    print(f"    Non-overlapping tiles: {len(tiles)}")
    for tx, ty, tw, th in tiles:
        print(f"      ({tx:2d},{ty:2d}) {tw:2d}×{th:2d}")
    total_needed = len(tiles) * 5
    print(f"    Total for 5 seeds: {total_needed} queries "
          f"(leaves {50 - total_needed} extra from budget of 50)")

    old_vps = old_overlap_viewports(40, 40, 10)
    covered_old = set()
    for tx, ty, tw, th in old_vps:
        tw_a = min(tw, 40 - tx)
        th_a = min(th, 40 - ty)
        for dy in range(max(0, th_a)):
            for dx in range(max(0, tw_a)):
                covered_old.add((ty + dy, tx + dx))
    print(f"\n    Old strategy (10 queries): {len(old_vps)} viewports, "
          f"{len(covered_old)} unique cells = {len(covered_old)/1600*100:.1f}% coverage")

    new_tiles = tiles[:10]
    covered_new = set()
    for tx, ty, tw, th in new_tiles:
        for dy in range(th):
            for dx in range(tw):
                covered_new.add((ty + dy, tx + dx))
    print(f"    New strategy (9 tiles):    {len(new_tiles)} tiles, "
          f"{len(covered_new)} unique cells = {len(covered_new)/1600*100:.1f}% coverage")


if __name__ == "__main__":
    main()
