#!/usr/bin/env python3
"""
Astar Island empirical-Bayes baseline with crafted map features.

This workspace is intentionally separate from the original repo.
It learns feature-conditioned priors from cached completed rounds,
then updates those priors using live or cached viewport observations.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from cnn_prior import LocalCnnPrior

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(SCRIPT_DIR, "ground_truth")
BASE_URL = "https://api.ainm.no/astar-island"
NUM_CLASSES = 6
PROB_FLOOR = 0.01
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _load_dotenv() -> None:
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


@dataclass
class Sample:
    round_number: int
    seed_index: int
    round_id: str
    initial_grid: List[List[int]]
    ground_truth: np.ndarray
    width: int
    height: int
    submitted_prediction: Optional[np.ndarray]


@dataclass
class FeatureBundle:
    specific: List[List[Tuple[int, ...]]]
    coarse: List[List[Tuple[int, ...]]]
    marginals: List[np.ndarray]


class CnnPrior:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        try:
            self.local = LocalCnnPrior(checkpoint_path=checkpoint_path)
            self.available = True
        except Exception:
            self.local = None
            self.available = False

    def predict(self, sample: Sample, observations: Sequence[dict]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("CNN prior unavailable")
        seed_obs = [obs for obs in observations if obs["seed_index"] == sample.seed_index]
        probs = self.local.predict(sample.initial_grid, seed_obs, sample.width, sample.height)
        for y in range(sample.height):
            for x_idx in range(sample.width):
                if sample.initial_grid[y][x_idx] == 5:
                    probs[y, x_idx, :] = PROB_FLOOR
                    probs[y, x_idx, 5] = 1.0 - PROB_FLOOR * (NUM_CLASSES - 1)
                elif sample.initial_grid[y][x_idx] == 10:
                    probs[y, x_idx, :] = PROB_FLOOR
                    probs[y, x_idx, 0] = 1.0 - PROB_FLOOR * (NUM_CLASSES - 1)
        return normalize_probs(probs)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.maximum(probs, PROB_FLOOR)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def competition_score(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = np.clip(pred, 1e-8, None)
    gt = np.clip(gt, 1e-8, None)
    kl = (gt * (np.log(gt) - np.log(pred))).sum(axis=-1)
    ent = -(gt * np.log(gt)).sum(axis=-1)
    total_ent = ent.sum()
    if total_ent < 1e-12:
        return 100.0, 0.0
    wkl = float((kl * ent).sum() / total_ent)
    return float(100.0 * math.exp(-3.0 * wkl)), wkl


def blend_with_cnn(
    bayes_pred: np.ndarray,
    cnn_pred: np.ndarray,
    hits: np.ndarray,
    round_summary: Optional[np.ndarray],
) -> np.ndarray:
    observed_mask = hits > 0
    obs_density = float(observed_mask.mean())
    change_rate = float(round_summary[-5]) if round_summary is not None and round_summary.size >= 5 else 0.0
    base_bayes = 0.08 + 0.18 * min(change_rate * 2.0, 1.0) + 0.10 * min(obs_density * 2.0, 1.0)
    base_bayes = min(max(base_bayes, 0.08), 0.32)

    weight = np.full(hits.shape, base_bayes, dtype=np.float32)
    weight[observed_mask] = np.clip(base_bayes + 0.12 + 0.08 * np.log1p(hits[observed_mask]), 0.18, 0.45)

    blended = (1.0 - weight[..., None]) * cnn_pred + weight[..., None] * bayes_pred
    return normalize_probs(blended)


def load_samples() -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(glob.glob(os.path.join(GT_DIR, "r*_s*_*.json"))):
        fname = os.path.basename(path)
        parts = fname[:-5].split("_")
        round_number = int(parts[0][1:])
        seed_index = int(parts[1][1:])
        round_id = parts[2]
        with open(path) as f:
            data = json.load(f)
        if data.get("ground_truth") is None or data.get("initial_grid") is None:
            continue
        submitted = data.get("prediction")
        submitted_pred = None
        if submitted is not None:
            submitted_pred = np.array(submitted, dtype=np.float32)
        samples.append(
            Sample(
                round_number=round_number,
                seed_index=seed_index,
                round_id=round_id,
                initial_grid=data["initial_grid"],
                ground_truth=np.array(data["ground_truth"], dtype=np.float32),
                width=data["width"],
                height=data["height"],
                submitted_prediction=submitted_pred,
            )
        )
    return samples


def load_observations(round_id: str) -> List[dict]:
    path = os.path.join(SCRIPT_DIR, f"observations_{round_id[:8]}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("observations", [])
    return raw


def save_observations(round_id: str, round_number: int, observations: Sequence[dict]) -> None:
    path = os.path.join(SCRIPT_DIR, f"observations_{round_id[:8]}.json")
    payload = {
        "round_id": round_id,
        "round_number": round_number,
        "observations": list(observations),
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def nearest_distance_grid(positions: Sequence[Tuple[int, int]], height: int, width: int, cap: int = 9) -> np.ndarray:
    dist = np.full((height, width), cap, dtype=np.int16)
    if not positions:
        return dist
    for y in range(height):
        for x in range(width):
            best = cap
            for py, px in positions:
                d = abs(y - py) + abs(x - px)
                if d < best:
                    best = d
                if best == 0:
                    break
            dist[y, x] = min(best, cap)
    return dist


def bin_distance(d: int) -> int:
    if d == 0:
        return 0
    if d == 1:
        return 1
    if d == 2:
        return 2
    if d <= 4:
        return 3
    if d <= 6:
        return 4
    return 5


def extract_features(grid: List[List[int]]) -> FeatureBundle:
    height = len(grid)
    width = len(grid[0])
    settlements = [(y, x) for y in range(height) for x in range(width) if grid[y][x] in (1, 2)]
    ports = [(y, x) for y in range(height) for x in range(width) if grid[y][x] == 2]
    forests = [(y, x) for y in range(height) for x in range(width) if grid[y][x] == 4]

    d_settlement = nearest_distance_grid(settlements, height, width)
    d_port = nearest_distance_grid(ports, height, width)
    d_forest = nearest_distance_grid(forests, height, width)

    marginals = [np.zeros((height, width), dtype=np.int16) for _ in range(10)]
    specific: List[List[Tuple[int, ...]]] = []
    coarse: List[List[Tuple[int, ...]]] = []

    for y in range(height):
        specific_row = []
        coarse_row = []
        for x in range(width):
            cell = grid[y][x]
            ocean_n = 0
            forest_n = 0
            mountain_n = 0
            settle_n = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        v = grid[ny][nx]
                        ocean_n += int(v == 10)
                        forest_n += int(v == 4)
                        mountain_n += int(v == 5)
                        settle_n += int(v in (1, 2))

            coast = int(ocean_n > 0 and cell not in (10, 5))
            local_sett = 0
            for ny in range(max(0, y - 2), min(height, y + 3)):
                for nx in range(max(0, x - 2), min(width, x + 3)):
                    local_sett += int(grid[ny][nx] in (1, 2))

            frontier = int(cell in (0, 11, 4) and d_settlement[y, x] <= 3)
            dist_s = bin_distance(int(d_settlement[y, x]))
            dist_p = bin_distance(int(d_port[y, x]))
            dist_f = bin_distance(int(d_forest[y, x]))

            values = (
                cell,
                coast,
                min(ocean_n, 3),
                min(forest_n, 3),
                min(mountain_n, 3),
                min(settle_n, 3),
                dist_s,
                dist_p,
                min(local_sett, 4),
                frontier,
            )
            for i, value in enumerate(values):
                marginals[i][y, x] = value

            specific_key = values + (dist_f,)
            coarse_key = (cell, coast, dist_s, dist_p, min(local_sett, 4), frontier)
            specific_row.append(specific_key)
            coarse_row.append(coarse_key)

        specific.append(specific_row)
        coarse.append(coarse_row)

    return FeatureBundle(specific=specific, coarse=coarse, marginals=marginals)


def observations_to_counts(observations: Sequence[dict], height: int = 40, width: int = 40) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    by_seed: Dict[int, List[dict]] = {}
    for obs in observations:
        by_seed.setdefault(obs["seed_index"], []).append(obs)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for seed_index, seed_obs in by_seed.items():
        counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
        hits = np.zeros((height, width), dtype=np.float32)
        for obs in seed_obs:
            viewport = obs["viewport"]
            vx, vy = viewport["x"], viewport["y"]
            for dy, row in enumerate(obs["grid"]):
                for dx, value in enumerate(row):
                    gy, gx = vy + dy, vx + dx
                    if 0 <= gy < height and 0 <= gx < width:
                        counts[gy, gx, TERRAIN_TO_CLASS[value]] += 1.0
                        hits[gy, gx] += 1.0
        out[seed_index] = (counts, hits)
    return out


def _normalize_count_vector(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)
    return (counts / total).astype(np.float32)


def compute_round_summary(
    samples_for_round: Sequence[Sample],
    observations: Sequence[dict],
    feature_lookup: Dict[Tuple[int, int, str], FeatureBundle],
) -> np.ndarray:
    if not samples_for_round:
        return np.zeros(1, dtype=np.float32)

    obs_counts = observations_to_counts(observations, samples_for_round[0].height, samples_for_round[0].width)

    overall = np.zeros(NUM_CLASSES, dtype=np.float64)
    settle_init = np.zeros(NUM_CLASSES, dtype=np.float64)
    coast_land = np.zeros(NUM_CLASSES, dtype=np.float64)
    frontier_land = np.zeros(NUM_CLASSES, dtype=np.float64)
    forest_init = np.zeros(NUM_CLASSES, dtype=np.float64)
    changed = 0.0
    total_hits = 0.0
    settle_hits = 0.0
    coast_hits = 0.0
    frontier_hits = 0.0
    forest_hits = 0.0

    for sample in samples_for_round:
        features = feature_lookup[(sample.round_number, sample.seed_index, sample.round_id)]
        counts, hits = obs_counts.get(
            sample.seed_index,
            (
                np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32),
                np.zeros((sample.height, sample.width), dtype=np.float32),
            ),
        )
        ys, xs = np.where(hits > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cnt = counts[y, x]
            init_cell = sample.initial_grid[y][x]
            overall += cnt
            total_hits += hits[y, x]
            changed += cnt.sum() - cnt[TERRAIN_TO_CLASS[init_cell]]

            if init_cell in (1, 2):
                settle_init += cnt
                settle_hits += hits[y, x]
            if features.marginals[1][y, x] == 1 and init_cell not in (10, 5):
                coast_land += cnt
                coast_hits += hits[y, x]
            if features.marginals[9][y, x] == 1:
                frontier_land += cnt
                frontier_hits += hits[y, x]
            if init_cell == 4:
                forest_init += cnt
                forest_hits += hits[y, x]

    change_rate = 0.0 if total_hits <= 0 else changed / total_hits
    settle_rate = 0.0 if settle_hits <= 0 else (settle_init[1] + settle_init[2] + settle_init[3]) / settle_hits
    frontier_rate = 0.0 if frontier_hits <= 0 else (frontier_land[1] + frontier_land[2] + frontier_land[3]) / frontier_hits
    coast_port_rate = 0.0 if coast_hits <= 0 else coast_land[2] / coast_hits
    forest_persist_rate = 0.0 if forest_hits <= 0 else forest_init[4] / forest_hits

    summary = np.concatenate(
        [
            _normalize_count_vector(overall),
            _normalize_count_vector(settle_init),
            _normalize_count_vector(coast_land),
            _normalize_count_vector(frontier_land),
            _normalize_count_vector(forest_init),
            np.array(
                [
                    change_rate,
                    settle_rate,
                    frontier_rate,
                    coast_port_rate,
                    forest_persist_rate,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return summary.astype(np.float32)


class EmpiricalBayesOracle:
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)
        self.feature_cache: Dict[Tuple[int, int, str], FeatureBundle] = {}
        self.global_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.global_cells = 0
        self.specific_sum: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
        )
        self.specific_n: collections.Counter = collections.Counter()
        self.coarse_sum: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
        )
        self.coarse_n: collections.Counter = collections.Counter()
        self.marginal_sum: List[Dict[int, np.ndarray]] = [
            collections.defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64)) for _ in range(10)
        ]
        self.marginal_n: List[collections.Counter] = [collections.Counter() for _ in range(10)]
        self.round_summary: Dict[int, np.ndarray] = {}
        self.round_specific_mean: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = {}
        self.round_coarse_mean: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = {}

    def features_for(self, sample: Sample) -> FeatureBundle:
        key = (sample.round_number, sample.seed_index, sample.round_id)
        if key not in self.feature_cache:
            self.feature_cache[key] = extract_features(sample.initial_grid)
        return self.feature_cache[key]

    def fit(self, exclude_round: Optional[int] = None) -> None:
        self.global_sum.fill(0.0)
        self.global_cells = 0
        self.specific_sum.clear()
        self.specific_n.clear()
        self.coarse_sum.clear()
        self.coarse_n.clear()
        for i in range(10):
            self.marginal_sum[i].clear()
            self.marginal_n[i].clear()
        self.round_summary = {}
        self.round_specific_mean = {}
        self.round_coarse_mean = {}

        for sample in self.samples:
            if exclude_round is not None and sample.round_number == exclude_round:
                continue
            features = self.features_for(sample)
            gt = sample.ground_truth
            self.global_sum += gt.sum(axis=(0, 1))
            self.global_cells += sample.height * sample.width
            for y in range(sample.height):
                for x in range(sample.width):
                    p = gt[y, x]
                    specific_key = features.specific[y][x]
                    coarse_key = features.coarse[y][x]
                    self.specific_sum[specific_key] += p
                    self.specific_n[specific_key] += 1
                    self.coarse_sum[coarse_key] += p
                    self.coarse_n[coarse_key] += 1
                    for i in range(10):
                        value = int(features.marginals[i][y, x])
                        self.marginal_sum[i][value] += p
                        self.marginal_n[i][value] += 1

        self.global_mean = self.global_sum / max(self.global_sum.sum(), 1.0)

        rounds_present = sorted({sample.round_number for sample in self.samples if exclude_round != sample.round_number})
        for round_number in rounds_present:
            round_samples = [s for s in self.samples if s.round_number == round_number]
            specific_sum: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
                lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
            )
            specific_n: collections.Counter = collections.Counter()
            coarse_sum: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
                lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
            )
            coarse_n: collections.Counter = collections.Counter()
            for sample in round_samples:
                features = self.features_for(sample)
                for y in range(sample.height):
                    for x in range(sample.width):
                        specific_key = features.specific[y][x]
                        coarse_key = features.coarse[y][x]
                        p = sample.ground_truth[y, x]
                        specific_sum[specific_key] += p
                        specific_n[specific_key] += 1
                        coarse_sum[coarse_key] += p
                        coarse_n[coarse_key] += 1
            self.round_specific_mean[round_number] = {
                key: value / specific_n[key] for key, value in specific_sum.items()
            }
            self.round_coarse_mean[round_number] = {
                key: value / coarse_n[key] for key, value in coarse_sum.items()
            }
            round_id = round_samples[0].round_id
            observations = load_observations(round_id)
            if observations:
                self.round_summary[round_number] = compute_round_summary(
                    round_samples,
                    observations,
                    self.feature_cache,
                )

    def prior_mean_for_cell(self, features: FeatureBundle, y: int, x: int) -> np.ndarray:
        specific_key = features.specific[y][x]
        coarse_key = features.coarse[y][x]
        pieces: List[np.ndarray] = []
        weights: List[float] = []

        pieces.append(self.global_mean)
        weights.append(0.5)

        if self.coarse_n[coarse_key] > 0:
            pieces.append(self.coarse_sum[coarse_key] / self.coarse_n[coarse_key])
            weights.append(2.5)
        if self.specific_n[specific_key] >= 8:
            pieces.append(self.specific_sum[specific_key] / self.specific_n[specific_key])
            weights.append(3.5)

        feature_weights = [3.0, 0.8, 0.7, 1.0, 0.4, 1.4, 2.2, 1.2, 1.2, 1.0]
        for i, weight in enumerate(feature_weights):
            value = int(features.marginals[i][y, x])
            if self.marginal_n[i][value] > 0:
                pieces.append(self.marginal_sum[i][value] / self.marginal_n[i][value])
                weights.append(weight)

        prior = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_weight = 0.0
        for piece, weight in zip(pieces, weights):
            prior += weight * piece
            total_weight += weight
        prior = prior / max(total_weight, 1e-8)
        prior = np.maximum(prior, 1e-6)
        prior = prior / prior.sum()
        return prior

    def regime_prior_for_cell(
        self,
        round_summary: Optional[np.ndarray],
        coarse_key: Tuple[int, ...],
        specific_key: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if round_summary is None or not self.round_summary:
            return None

        weighted = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_weight = 0.0
        for round_number, summary in self.round_summary.items():
            dist = float(np.linalg.norm(round_summary - summary))
            weight = math.exp(-dist / 0.75)
            if specific_key in self.round_specific_mean.get(round_number, {}):
                weighted += 1.5 * weight * self.round_specific_mean[round_number][specific_key]
                total_weight += 1.5 * weight
            elif coarse_key in self.round_coarse_mean.get(round_number, {}):
                weighted += 1.0 * weight * self.round_coarse_mean[round_number][coarse_key]
                total_weight += 1.0 * weight
        if total_weight <= 1e-8:
            return None
        regime = weighted / total_weight
        regime = np.maximum(regime, 1e-6)
        regime = regime / regime.sum()
        return regime

    def build_round_updates(
        self,
        samples_for_round: Sequence[Sample],
        obs_counts_by_seed: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Dict[Tuple[int, ...], np.ndarray], Dict[Tuple[int, ...], np.ndarray]]:
        coarse_counts: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
        )
        specific_counts: Dict[Tuple[int, ...], np.ndarray] = collections.defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=np.float64)
        )
        for sample in samples_for_round:
            if sample.seed_index not in obs_counts_by_seed:
                continue
            features = self.features_for(sample)
            counts, hits = obs_counts_by_seed[sample.seed_index]
            ys, xs = np.where(hits > 0)
            for y, x in zip(ys.tolist(), xs.tolist()):
                specific_counts[features.specific[y][x]] += counts[y, x]
                specific_counts[features.coarse[y][x]] += 0.0
                coarse_counts[features.coarse[y][x]] += counts[y, x]
        return coarse_counts, specific_counts

    def predict_sample(
        self,
        sample: Sample,
        observations: Sequence[dict],
        round_coarse_counts: Dict[Tuple[int, ...], np.ndarray],
        round_specific_counts: Dict[Tuple[int, ...], np.ndarray],
        round_summary: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        features = self.features_for(sample)
        counts, hits = observations_to_counts(observations, sample.height, sample.width).get(
            sample.seed_index,
            (
                np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32),
                np.zeros((sample.height, sample.width), dtype=np.float32),
            ),
        )
        pred = np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32)

        for y in range(sample.height):
            for x in range(sample.width):
                init_cell = sample.initial_grid[y][x]
                if init_cell == 5:
                    forced = np.full(NUM_CLASSES, PROB_FLOOR, dtype=np.float32)
                    forced[5] = 1.0 - PROB_FLOOR * (NUM_CLASSES - 1)
                    pred[y, x] = forced
                    continue
                if init_cell == 10:
                    forced = np.full(NUM_CLASSES, PROB_FLOOR, dtype=np.float32)
                    forced[0] = 1.0 - PROB_FLOOR * (NUM_CLASSES - 1)
                    pred[y, x] = forced
                    continue

                prior_mean = self.prior_mean_for_cell(features, y, x)
                coarse_key = features.coarse[y][x]
                specific_key = features.specific[y][x]

                round_mean = prior_mean.copy()
                regime_mean = self.regime_prior_for_cell(round_summary, coarse_key, specific_key)
                if regime_mean is not None:
                    round_mean = (5.0 * round_mean + 4.0 * regime_mean) / 9.0
                coarse_obs = round_coarse_counts.get(coarse_key)
                if coarse_obs is not None and coarse_obs.sum() > 0:
                    round_mean = (8.0 * round_mean + coarse_obs) / (8.0 + coarse_obs.sum())
                specific_obs = round_specific_counts.get(specific_key)
                if specific_obs is not None and specific_obs.sum() > 0:
                    round_mean = (6.0 * round_mean + specific_obs) / (6.0 + specific_obs.sum())

                posterior = 7.0 * round_mean
                if hits[y, x] > 0:
                    posterior = posterior + counts[y, x]
                pred[y, x] = posterior / posterior.sum()

        return normalize_probs(pred)


def compute_settlement_viewports(initial_grid: List[List[int]], width: int, height: int, max_tile: int = 15) -> List[Tuple[int, int, int, int]]:
    settlements = []
    for y in range(height):
        for x in range(width):
            if initial_grid[y][x] in (1, 2):
                settlements.append((y, x))

    tw = min(max_tile, width)
    th = min(max_tile, height)
    if not settlements:
        return [(max(0, (width - tw) // 2), max(0, (height - th) // 2), tw, th)]

    uncovered = set(range(len(settlements)))
    viewports: List[Tuple[int, int, int, int]] = []
    max_x = max(0, width - tw)
    max_y = max(0, height - th)

    while uncovered:
        best_vp = None
        best_count = 0
        for vy in range(max_y + 1):
            for vx in range(max_x + 1):
                count = sum(
                    1
                    for i in uncovered
                    if vy <= settlements[i][0] < vy + th and vx <= settlements[i][1] < vx + tw
                )
                if count > best_count:
                    best_count = count
                    best_vp = (vx, vy, tw, th)
        if best_vp is None or best_count == 0:
            break
        viewports.append(best_vp)
        vx, vy, vw, vh = best_vp
        uncovered = {
            i
            for i in uncovered
            if not (vy <= settlements[i][0] < vy + vh and vx <= settlements[i][1] < vx + vw)
        }

    def _count(vp: Tuple[int, int, int, int]) -> int:
        vx, vy, vw, vh = vp
        return sum(1 for sy, sx in settlements if vy <= sy < vy + vh and vx <= sx < vx + vw)

    viewports.sort(key=_count, reverse=True)
    return viewports


def score_tile(initial_grid: List[List[int]], tile: Tuple[int, int, int, int], width: int, height: int) -> float:
    tx, ty, tw, th = tile
    score = 0.0
    n_settlements = 0
    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            cell = initial_grid[y][x]
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
    if n_settlements >= 2:
        score += n_settlements
    return score


def make_session() -> requests.Session:
    token = os.environ.get("ASTAR_TOKEN")
    if not token:
        raise RuntimeError("ASTAR_TOKEN missing from environment or .env")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def api_get(session: requests.Session, path: str) -> dict:
    resp = session.get(f"{BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(session: requests.Session, path: str, payload: dict) -> dict:
    resp = session.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_active_round(session: requests.Session) -> dict:
    rounds = api_get(session, "/rounds")
    active = next((r for r in rounds if r["status"] == "active"), None)
    if active is None:
        raise RuntimeError("No active round")
    return active


def collect_live_observations(session: requests.Session, round_detail: dict) -> List[dict]:
    round_id = round_detail["id"]
    width = round_detail["map_width"]
    height = round_detail["map_height"]
    seeds_count = round_detail["seeds_count"]
    initial_states = round_detail["initial_states"]

    budget = api_get(session, "/budget")
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        return []

    per_seed_plans = [
        compute_settlement_viewports(initial_states[s]["grid"], width, height) for s in range(seeds_count)
    ]
    observations: List[dict] = []
    max_len = max(len(vps) for vps in per_seed_plans)

    def _query(seed_index: int, vp: Tuple[int, int, int, int]) -> bool:
        vx, vy, vw, vh = vp
        result = api_post(
            session,
            "/simulate",
            {
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": vx,
                "viewport_y": vy,
                "viewport_w": vw,
                "viewport_h": vh,
            },
        )
        observations.append(
            {"seed_index": seed_index, "viewport": result["viewport"], "grid": result["grid"]}
        )
        return result.get("queries_used", 0) >= result.get("queries_max", 50)

    queries_done = 0
    for idx in range(max_len):
        for seed_index in range(seeds_count):
            if queries_done >= remaining:
                save_observations(round_id, round_detail["round_number"], observations)
                return observations
            if idx >= len(per_seed_plans[seed_index]):
                continue
            exhausted = _query(seed_index, per_seed_plans[seed_index][idx])
            queries_done += 1
            if exhausted:
                save_observations(round_id, round_detail["round_number"], observations)
                return observations
            time.sleep(1.0)

    candidates: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
    for seed_index in range(seeds_count):
        for vp in per_seed_plans[seed_index]:
            score = score_tile(initial_states[seed_index]["grid"], vp, width, height)
            candidates.append((score, seed_index, vp))
    candidates.sort(reverse=True)

    cidx = 0
    while queries_done < remaining and candidates:
        _, seed_index, vp = candidates[cidx % len(candidates)]
        exhausted = _query(seed_index, vp)
        queries_done += 1
        cidx += 1
        if exhausted:
            break
        time.sleep(1.0)

    save_observations(round_id, round_detail["round_number"], observations)
    return observations


def submit_predictions(session: requests.Session, round_detail: dict, predictions: Dict[int, np.ndarray]) -> None:
    for seed_index, pred in sorted(predictions.items()):
        api_post(
            session,
            "/submit",
            {
                "round_id": round_detail["id"],
                "seed_index": seed_index,
                "prediction": pred.tolist(),
            },
        )
        time.sleep(1.0)


def crossval(args: argparse.Namespace) -> None:
    samples = load_samples()
    cnn_prior = CnnPrior(args.cnn_checkpoint) if args.hybrid_cnn else None
    obs_rounds = sorted(
        {
            sample.round_number
            for sample in samples
            if os.path.exists(os.path.join(SCRIPT_DIR, f"observations_{sample.round_id[:8]}.json"))
        }
    )
    oracle = EmpiricalBayesOracle(samples)
    results = []
    for round_number in obs_rounds:
        oracle.fit(exclude_round=round_number)
        holdout_samples = [s for s in samples if s.round_number == round_number]
        round_obs = load_observations(holdout_samples[0].round_id)
        obs_counts = observations_to_counts(round_obs)
        round_coarse, round_specific = oracle.build_round_updates(holdout_samples, obs_counts)
        round_summary = compute_round_summary(holdout_samples, round_obs, oracle.feature_cache)
        scores = []
        hybrid_scores = []
        submitted = []
        for sample in holdout_samples:
            pred = oracle.predict_sample(sample, round_obs, round_coarse, round_specific, round_summary)
            score_val, _ = competition_score(pred, sample.ground_truth)
            scores.append(score_val)
            if cnn_prior is not None and cnn_prior.available:
                cnn_pred = cnn_prior.predict(sample, round_obs)
                _, hits = observations_to_counts(round_obs).get(
                    sample.seed_index,
                    (
                        np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32),
                        np.zeros((sample.height, sample.width), dtype=np.float32),
                    ),
                )
                hybrid = blend_with_cnn(pred, cnn_pred, hits, round_summary)
                hybrid_score, _ = competition_score(hybrid, sample.ground_truth)
                hybrid_scores.append(hybrid_score)
            if sample.submitted_prediction is not None:
                sub_score, _ = competition_score(sample.submitted_prediction, sample.ground_truth)
                submitted.append(sub_score)
        avg_score = float(np.mean(scores))
        avg_sub = float(np.mean(submitted)) if submitted else float("nan")
        avg_hybrid = float(np.mean(hybrid_scores)) if hybrid_scores else float("nan")
        results.append(avg_score)
        if hybrid_scores:
            print(f"round {round_number:2d}: bayes={avg_score:6.2f} hybrid={avg_hybrid:6.2f} submitted={avg_sub:6.2f}")
        else:
            print(f"round {round_number:2d}: bayes={avg_score:6.2f} submitted={avg_sub:6.2f}")
    if results:
        print(f"mean bayes score: {float(np.mean(results)):.2f}")
        if args.hybrid_cnn and cnn_prior is not None and cnn_prior.available:
            print("hybrid-cnn enabled")


def round_eval(args: argparse.Namespace) -> None:
    samples = load_samples()
    cnn_prior = CnnPrior(args.cnn_checkpoint) if args.hybrid_cnn else None
    round_samples = [s for s in samples if s.round_number == args.round]
    if not round_samples:
        raise RuntimeError(f"Round {args.round} not found in cached ground truth")
    oracle = EmpiricalBayesOracle(samples)
    oracle.fit(exclude_round=args.exclude_round and args.round or None)
    round_obs = load_observations(round_samples[0].round_id)
    if not round_obs:
        print(f"No cached observations for round {args.round}.")
        return
    obs_counts = observations_to_counts(round_obs)
    round_coarse, round_specific = oracle.build_round_updates(round_samples, obs_counts)
    round_summary = compute_round_summary(round_samples, round_obs, oracle.feature_cache)
    bayes_scores = []
    hybrid_scores = []
    submitted_scores = []
    for sample in round_samples:
        pred = oracle.predict_sample(sample, round_obs, round_coarse, round_specific, round_summary)
        bayes_score, _ = competition_score(pred, sample.ground_truth)
        bayes_scores.append(bayes_score)
        hybrid_score = float("nan")
        if cnn_prior is not None and cnn_prior.available:
            cnn_pred = cnn_prior.predict(sample, round_obs)
            _, hits = obs_counts.get(
                sample.seed_index,
                (
                    np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32),
                    np.zeros((sample.height, sample.width), dtype=np.float32),
                ),
            )
            hybrid = blend_with_cnn(pred, cnn_pred, hits, round_summary)
            hybrid_score, _ = competition_score(hybrid, sample.ground_truth)
            hybrid_scores.append(hybrid_score)
        submitted_score = float("nan")
        if sample.submitted_prediction is not None:
            submitted_score, _ = competition_score(sample.submitted_prediction, sample.ground_truth)
            submitted_scores.append(submitted_score)
        if not math.isnan(hybrid_score):
            print(
                f"seed {sample.seed_index}: bayes={bayes_score:6.2f} "
                f"hybrid={hybrid_score:6.2f} submitted={submitted_score:6.2f}"
            )
        else:
            print(f"seed {sample.seed_index}: bayes={bayes_score:6.2f} submitted={submitted_score:6.2f}")
    print(f"avg bayes={float(np.mean(bayes_scores)):.2f}")
    if hybrid_scores:
        print(f"avg hybrid={float(np.mean(hybrid_scores)):.2f}")
    if submitted_scores:
        print(f"avg submitted={float(np.mean(submitted_scores)):.2f}")


def live(args: argparse.Namespace) -> None:
    samples = load_samples()
    oracle = EmpiricalBayesOracle(samples)
    oracle.fit()
    cnn_prior = CnnPrior(args.cnn_checkpoint) if args.hybrid_cnn else None
    session = make_session()
    active = get_active_round(session)
    round_detail = api_get(session, f"/rounds/{active['id']}")

    observations = collect_live_observations(session, round_detail)
    round_id = round_detail["id"][:8]
    if args.reuse_only and not observations:
        observations = load_observations(round_id)

    obs_counts = observations_to_counts(observations)
    synthetic_samples = []
    for seed_index, state in enumerate(round_detail["initial_states"]):
        synthetic_samples.append(
            Sample(
                round_number=round_detail["round_number"],
                seed_index=seed_index,
                round_id=round_detail["id"][:8],
                initial_grid=state["grid"],
                ground_truth=np.zeros((round_detail["map_height"], round_detail["map_width"], NUM_CLASSES), dtype=np.float32),
                width=round_detail["map_width"],
                height=round_detail["map_height"],
                submitted_prediction=None,
            )
        )
    round_coarse, round_specific = oracle.build_round_updates(synthetic_samples, obs_counts)
    round_summary = compute_round_summary(synthetic_samples, observations, oracle.feature_cache)
    predictions = {}
    for sample in synthetic_samples:
        bayes_pred = oracle.predict_sample(
            sample, observations, round_coarse, round_specific, round_summary
        )
        if cnn_prior is not None and cnn_prior.available:
            cnn_pred = cnn_prior.predict(sample, observations)
            _, hits = obs_counts.get(
                sample.seed_index,
                (
                    np.zeros((sample.height, sample.width, NUM_CLASSES), dtype=np.float32),
                    np.zeros((sample.height, sample.width), dtype=np.float32),
                ),
            )
            predictions[sample.seed_index] = blend_with_cnn(bayes_pred, cnn_pred, hits, round_summary)
        else:
            predictions[sample.seed_index] = bayes_pred

    if args.no_submit:
        for seed_index, pred in predictions.items():
            out = os.path.join(SCRIPT_DIR, f"prediction_r{round_detail['round_number']}_s{seed_index}.json")
            with open(out, "w") as f:
                json.dump(pred.tolist(), f)
            print(f"saved {out}")
        return

    submit_predictions(session, round_detail, predictions)
    print(f"submitted Bayesian predictions for round {round_detail['round_number']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island empirical-Bayes workspace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cross = sub.add_parser("crossval", help="Leave-one-observed-round-out evaluation")
    p_cross.add_argument("--hybrid-cnn", action="store_true", help="Blend Bayesian output with the refreshed CNN prior")
    p_cross.add_argument("--cnn-checkpoint", default=None, help="Path to local CNN checkpoint for hybrid mode")
    p_cross.set_defaults(func=crossval)

    p_round = sub.add_parser("round-eval", help="Evaluate one completed round with cached observations")
    p_round.add_argument("round", type=int)
    p_round.add_argument("--exclude-round", action="store_true", help="Exclude the evaluated round from training")
    p_round.add_argument("--hybrid-cnn", action="store_true", help="Blend Bayesian output with the refreshed CNN prior")
    p_round.add_argument("--cnn-checkpoint", default=None, help="Path to local CNN checkpoint for hybrid mode")
    p_round.set_defaults(func=round_eval)

    p_live = sub.add_parser("live", help="Query active round and submit Bayesian predictions")
    p_live.add_argument("--no-submit", action="store_true", help="Write predictions locally instead of submitting")
    p_live.add_argument("--reuse-only", action="store_true", help="Reuse cached observations if budget is already spent")
    p_live.add_argument("--hybrid-cnn", action="store_true", help="Blend Bayesian output with the refreshed CNN prior")
    p_live.add_argument("--cnn-checkpoint", default=None, help="Path to local CNN checkpoint for hybrid mode")
    p_live.set_defaults(func=live)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
