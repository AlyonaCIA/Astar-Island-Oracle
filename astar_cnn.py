"""
Astar Island — CNN-based Prediction Script

Strategy:
1. Fetch active round and initial states
2. IMMEDIATELY submit prior-based fallback (guarantees a score even if CNN is slow)
3. Use queries to observe simulation outcomes (training data)
4. Save observations to disk for offline training
5. Train a small CNN: initial features → final terrain class probabilities
6. Predict full map for all 5 seeds and resubmit (overwrites fallback)

Time limit: If the CNN pipeline doesn't finish within TIME_LIMIT_MINUTES,
the fallback submission from step 2 still stands.
"""

import os
import sys
import time
import json
import numpy as np
import requests

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"WARNING: PyTorch not available ({e}). Will submit fallback only.")


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
TIME_LIMIT_MINUTES = float(os.environ.get("ASTAR_TIME_LIMIT", "120"))  # default 2h
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if not TOKEN:
    print("ERROR: Set ASTAR_TOKEN in .env file or as environment variable.")
    print("  Copy .env.example to .env and fill in your JWT token.")
    sys.exit(1)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

NUM_CLASSES = 6
PROB_FLOOR = 1e-6
if TORCH_AVAILABLE:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]  # 8 one-hot channels


# --- Deadline ---

DEADLINE = None  # set in main()


def time_remaining():
    """Seconds remaining before deadline. Negative = past deadline."""
    if DEADLINE is None:
        return float("inf")
    return DEADLINE - time.time()


def past_deadline():
    return time_remaining() <= 0


# --- API Helpers ---

def get_active_round():
    resp = session.get(f"{BASE_URL}/rounds")
    resp.raise_for_status()
    time.sleep(1.0)
    rounds = resp.json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round found. Available rounds:")
        for r in rounds:
            print(f"  Round {r['round_number']} — status: {r['status']}")
        sys.exit(0)
    return active


def get_round_details(round_id):
    resp = session.get(f"{BASE_URL}/rounds/{round_id}")
    resp.raise_for_status()
    time.sleep(1.0)
    return resp.json()


def check_budget(verbose=True):
    resp = session.get(f"{BASE_URL}/budget")
    time.sleep(1.0)
    if resp.status_code == 200:
        data = resp.json()
        if verbose:
            print(f"Budget: {data['queries_used']}/{data['queries_max']} queries used")
        return data
    return None


def simulate(round_id, seed_index, vx, vy, vw=15, vh=15):
    resp = session.post(f"{BASE_URL}/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": min(vw, 15),
        "viewport_h": min(vh, 15),
    })
    resp.raise_for_status()
    time.sleep(1.0)
    return resp.json()


def submit_prediction(round_id, seed_index, prediction):
    resp = session.post(f"{BASE_URL}/submit", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    resp.raise_for_status()
    time.sleep(1.0)
    return resp.json()


def terrain_to_class(cell_value):
    """Map internal terrain code to prediction class index (0-5)."""
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(cell_value, 0)


# --- Feature Encoding ---

def encode_initial_grid(initial_grid, width, height):
    """
    Encode initial terrain grid into a 14-channel feature tensor.

    Channels 0-7:  One-hot encoding for each terrain type
                   [Empty, Settlement, Port, Ruin, Forest, Mountain, Ocean, Plains]
    Channels 8-13: Neighbor class counts (how many of each prediction class
                   are in the 8 surrounding cells), normalized by 8.
                   This gives the CNN local context without needing large kernels.

    Returns: numpy array of shape (14, height, width)
    """
    features = np.zeros((14, height, width), dtype=np.float32)

    # Channels 0-7: one-hot terrain type
    code_to_channel = {code: i for i, code in enumerate(TERRAIN_CODES)}
    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            ch = code_to_channel.get(cell, 0)
            features[ch, y, x] = 1.0

    # Channels 8-13: neighbor class counts (normalized)
    class_grid = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            class_grid[y, x] = terrain_to_class(initial_grid[y][x])

    for y in range(height):
        for x in range(width):
            counts = np.zeros(NUM_CLASSES, dtype=np.float32)
            n = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        counts[class_grid[ny, nx]] += 1.0
                        n += 1
            if n > 0:
                counts /= n
            features[8:14, y, x] = counts

    return features


# --- CNN Models ---

OBS_CHANNELS = 7  # 6 class frequencies + 1 coverage


def encode_obs_channels(observations, width, height):
    """
    Encode viewport observations into 7 extra input channels.

    Channels 0-5: observed class frequency at each pixel (count[c] / total_obs)
    Channel 6:    log(1 + total_observations) coverage indicator

    Unobserved pixels are all zeros, so the network learns to fall back
    to terrain-only prediction when no observations are available.

    Returns: numpy array of shape (7, height, width)
    """
    obs_counts = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    obs_hits = np.zeros((height, width), dtype=np.float32)
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        grid = obs["grid"]
        for dy in range(len(grid)):
            for dx in range(len(grid[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    cls = mapping.get(grid[dy][dx], 0)
                    obs_counts[cls, gy, gx] += 1.0
                    obs_hits[gy, gx] += 1.0

    # Normalize counts to frequencies
    mask = obs_hits > 0
    for c in range(NUM_CLASSES):
        obs_counts[c][mask] /= obs_hits[mask]

    # Coverage channel
    coverage = np.log1p(obs_hits)[np.newaxis, :, :]  # (1, H, W)

    return np.concatenate([obs_counts, coverage], axis=0)  # (7, H, W)


class MiniUNet(nn.Module):
    """Small U-Net for 40x40 maps. Encoder-decoder with skip connections."""
    def __init__(self, dropout=0.2, in_channels=14):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(32, 6, kernel_size=1)

    def forward(self, x, temperature=1.0):
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.out_conv(d1)

        if pad_h or pad_w:
            logits = logits[:, :, :H, :W]
        if temperature != 1.0:
            logits = logits/temperature

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


MODEL_REGISTRY = {
    "unet_cond": lambda **kw: MiniUNet(dropout=kw.get('dropout', 0.1),
                                        in_channels=14 + OBS_CHANNELS),
}

CHECKPOINT_DIR_MAP = {
    "unet_cond": os.path.join(SCRIPT_DIR, "checkpoints_unet_cond"),
}

MODEL_ARCH = "unet_cond"


def make_model(arch="unet_cond", **kwargs):
    cls = MODEL_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY)}")
    return cls(**kwargs)


# --- Training Data Collection ---

def collect_observations(round_id, seeds_count, initial_states, width, height):
    """
    Settlement-focused viewport query strategy: capture all settlements
    with multiple observations for better distribution estimates.

    Phase 1 — SETTLEMENT COVERAGE:
        Place viewports via greedy set-cover to ensure every settlement and
        port is observed on every seed. Viewports sorted by settlement
        density, queried round-robin across seeds.

    Phase 2 — MULTI-OBSERVE (remaining budget):
        Re-query the most dynamic/settlement-dense viewports across all seeds.
        Each query is a fresh stochastic simulation, so multiple observations
        build richer empirical frequency distributions — exactly what the
        unet_cond model leverages.
    """
    budget = check_budget()
    if not budget:
        return []
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        print("No queries remaining — will train on priors only.")
        return []

    observations = []
    seed_obs = [[] for _ in range(seeds_count)]
    queries_done = 0
    query_limit = remaining

    # Build settlement-focused viewports per seed
    seed_viewports = []
    for s in range(seeds_count):
        grid = initial_states[s]["grid"]
        vps = compute_settlement_viewports(grid, width, height)
        seed_viewports.append(vps)
        n_settle = sum(1 for y in range(height) for x in range(width)
                       if grid[y][x] in (1, 2))
        print(f"  Seed {s}: {n_settle} settlements/ports → {len(vps)} viewports")

    def _do_query(seed, tx, ty, tw, th, phase_label):
        """Execute a single query and record the observation. Returns True if budget exhausted."""
        nonlocal queries_done
        try:
            result = simulate(round_id, seed, tx, ty, tw, th)
            vp = result["viewport"]
            obs = {"seed_index": seed, "viewport": vp, "grid": result["grid"]}
            observations.append(obs)
            seed_obs[seed].append(obs)
            queries_done += 1

            used = result.get("queries_used", "?")
            max_q = result.get("queries_max", "?")
            print(f"  {phase_label}: seed {seed} ({vp['x']},{vp['y']}) "
                  f"{vp['w']}x{vp['h']}  budget {used}/{max_q}")

            if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                return True  # budget exhausted
            return False
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                return True  # treat rate limit as budget stop
            raise

    # Compute budget allocation
    total_settlement_vps = sum(len(vps) for vps in seed_viewports)
    max_vps_per_seed = max((len(vps) for vps in seed_viewports), default=0)
    print(f"  Strategy: settlement-focused (capture all settlements → multi-observe)")
    print(f"  Budget: {remaining} queries")
    print(f"  Phase 1: SETTLEMENT COVERAGE ({total_settlement_vps} viewports "
          f"across {seeds_count} seeds)")

    # ── Phase 1: SETTLEMENT COVERAGE ──
    # Round-robin: query viewport 0 on all seeds, then viewport 1, etc.
    for vp_idx in range(max_vps_per_seed):
        for s in range(seeds_count):
            if queries_done >= query_limit or past_deadline():
                break
            if vp_idx >= len(seed_viewports[s]):
                continue
            tx, ty, tw, th = seed_viewports[s][vp_idx]
            exhausted = _do_query(s, tx, ty, tw, th,
                                  f"Settle {vp_idx+1}/{len(seed_viewports[s])}")
            if exhausted:
                _save_observations(observations, round_id)
                return observations
        if queries_done >= query_limit or past_deadline():
            break

    # Report coverage after Phase 1
    for s in range(seeds_count):
        covered = np.zeros((height, width), dtype=bool)
        for obs in seed_obs[s]:
            vp = obs["viewport"]
            covered[vp["y"]:vp["y"]+vp["h"], vp["x"]:vp["x"]+vp["w"]] = True
        pct = covered.sum() / (width * height) * 100
        grid = initial_states[s]["grid"]
        n_total = sum(1 for y in range(height) for x in range(width)
                      if grid[y][x] in (1, 2))
        n_covered = sum(1 for y in range(height) for x in range(width)
                        if grid[y][x] in (1, 2) and covered[y, x])
        print(f"  Seed {s}: {len(seed_obs[s])} queries, {pct:.0f}% map, "
              f"settlements: {n_covered}/{n_total}")

    # ── Build cross-seed dynamism from Phase 1 observations ──
    cross_seed_dynamism = np.zeros((height, width), dtype=np.float32)
    for s in range(seeds_count):
        grid = initial_states[s]["grid"]
        dyn = build_observed_dynamism_heatmap(seed_obs[s], grid, width, height)
        cross_seed_dynamism += dyn

    # ── Phase 2: MULTI-OBSERVE — re-query settlement viewports ──
    resample_budget = query_limit - queries_done
    if resample_budget > 0 and not past_deadline():
        print(f"  Phase 2: MULTI-OBSERVE ({resample_budget} queries on settlement viewports)")

        # Rank all settlement viewports by terrain score + observed dynamism
        candidates = []
        for s in range(seeds_count):
            for vp in seed_viewports[s]:
                tx, ty, tw, th = vp
                base_score = score_tile(initial_states[s]["grid"], vp, width, height)
                dyn_bonus = cross_seed_dynamism[ty:ty+th, tx:tx+tw].sum()
                combined = base_score + 2.0 * dyn_bonus
                candidates.append((combined, s, vp))
        candidates.sort(reverse=True)

        # Cycle through top candidates
        resample_idx = 0
        while queries_done < query_limit and not past_deadline():
            if not candidates:
                break
            score, s, (tx, ty, tw, th) = candidates[resample_idx % len(candidates)]
            resample_idx += 1
            exhausted = _do_query(s, tx, ty, tw, th,
                                  f"Multi-obs {resample_idx} (score={score:.1f})")
            if exhausted:
                break

    _save_observations(observations, round_id)
    return observations


def _save_observations(observations, round_id, round_number=None):
    """Save observations to disk so they can be reused for offline training."""
    if not observations:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"observations_{round_id[:8]}.json")
    payload = {
        "round_id": round_id,
        "round_number": round_number,
        "observations": observations,
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"  Saved {len(observations)} observations to {path}")


def _load_observations(round_id):
    """Load previously saved observations for a round, if available."""
    path = os.path.join(DATA_DIR, f"observations_{round_id[:8]}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("observations", [])
    return raw


def _save_round_data(round_id, detail):
    """Save round details (initial states, dimensions) for offline use."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"round_{round_id[:8]}.json")
    with open(path, "w") as f:
        json.dump(detail, f)
    print(f"  Cached round data to {path}")


def compute_tile_grid(width, height, max_tile=15):
    """
    Full-size viewport grid covering the entire map.
    Every tile is max_tile × max_tile (no wasted viewport area).
    Edge tiles are shifted inward so no pixel falls outside the map,
    creating natural overlap instead of smaller edge tiles.

    For a 40×40 map with max_tile=15: 9 tiles (3×3), all 15×15.
      x=[0, 12, 25]  y=[0, 12, 25]  — overlap at edges instead of shrinkage.
    Returns list of (x, y, w, h) tuples.
    """
    def axis_positions(length, tile_size):
        if length <= tile_size:
            return [(0, min(length, tile_size))]
        n_tiles = -(-length // tile_size)  # ceil division
        max_start = length - tile_size
        positions = []
        for i in range(n_tiles):
            start = round(i * max_start / (n_tiles - 1)) if n_tiles > 1 else 0
            positions.append((start, tile_size))
        return positions

    x_specs = axis_positions(width, max_tile)
    y_specs = axis_positions(height, max_tile)
    return [(tx, ty, tw, th) for (ty, th) in y_specs for (tx, tw) in x_specs]


def compute_settlement_viewports(initial_grid, width, height, max_tile=15):
    """
    Greedy set-cover: place max_tile × max_tile viewports to capture ALL
    settlements (1) and ports (2) on the initial grid.
    Returns viewports sorted by settlement density (most settlements first),
    so the most informative viewports are queried first if budget is limited.
    Falls back to a single center viewport if no settlements exist.
    """
    settlements = []
    for y in range(height):
        for x in range(width):
            if initial_grid[y][x] in (1, 2):
                settlements.append((y, x))

    tw = min(max_tile, width)
    th = min(max_tile, height)

    if not settlements:
        cx = max(0, (width - tw) // 2)
        cy = max(0, (height - th) // 2)
        return [(cx, cy, tw, th)]

    max_x = max(0, width - tw)
    max_y = max(0, height - th)

    uncovered = set(range(len(settlements)))
    viewports = []

    while uncovered:
        best_vp = None
        best_count = 0

        for vy in range(max_y + 1):
            for vx in range(max_x + 1):
                count = sum(1 for i in uncovered
                            if vy <= settlements[i][0] < vy + th
                            and vx <= settlements[i][1] < vx + tw)
                if count > best_count:
                    best_count = count
                    best_vp = (vx, vy, tw, th)

        if best_vp is None or best_count == 0:
            break

        viewports.append(best_vp)
        vx, vy, vw, vh = best_vp
        uncovered = {i for i in uncovered
                     if not (vy <= settlements[i][0] < vy + vh
                             and vx <= settlements[i][1] < vx + vw)}

    def _count_settlements(vp):
        vx, vy, vw, vh = vp
        return sum(1 for sy, sx in settlements
                   if vy <= sy < vy + vh and vx <= sx < vx + vw)

    viewports.sort(key=_count_settlements, reverse=True)
    return viewports


def score_tile(initial_grid, tile, width, height):
    """
    Score a tile's priority based on initial terrain content.
    Higher = more dynamic/interesting content = should be queried first.
    """
    tx, ty, tw, th = tile
    score = 0.0
    n_settlements = 0

    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            cell = initial_grid[y][x]
            if cell == 1:       # Settlement — most dynamic
                score += 5.0
                n_settlements += 1
            elif cell == 2:     # Port — dynamic + trade
                score += 6.0
                n_settlements += 1
            elif cell == 4:     # Forest — mostly static but supports settlements
                score += 0.3
            elif cell in (0, 11):  # Plains/Empty — expansion potential
                score += 0.5
            # Ocean (10) and Mountain (5): 0 — completely static

    # Bonus for coastal land near settlements (potential port development)
    for dy in range(th):
        for dx in range(tw):
            y, x = ty + dy, tx + dx
            cell = initial_grid[y][x]
            if cell not in (10, 5):  # land cell
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if 0 <= ny < height and 0 <= nx < width:
                        if initial_grid[ny][nx] == 10:  # adjacent to ocean
                            score += 0.3
                            break

    # Bonus for settlement clusters (lots of interaction = conflict/trade)
    if n_settlements >= 2:
        score += n_settlements * 1.0

    return score


def build_interest_heatmap(initial_grid, width, height):
    """
    Build a per-cell interest score from initial terrain.
    Higher values = more likely to be dynamic / contribute to scoring entropy.
    Used by the greedy viewport placement algorithm.
    """
    interest = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            if cell == 1:         # Settlement — highly dynamic
                interest[y, x] = 5.0
            elif cell == 2:       # Port — dynamic + trade
                interest[y, x] = 6.0
            elif cell == 4:       # Forest — mostly static, supports settlements
                interest[y, x] = 0.3
            elif cell in (0, 11): # Plains — expansion potential
                interest[y, x] = 0.5
            # Ocean (10) and Mountain (5): 0 — completely static

    # Coastal bonus — land cells adjacent to ocean have port/expansion potential
    for y in range(height):
        for x in range(width):
            if interest[y, x] > 0:
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if 0 <= ny < height and 0 <= nx < width:
                        if initial_grid[ny][nx] == 10:
                            interest[y, x] += 0.3
                            break

    # Settlement proximity bonus — cells near settlements gain interest
    # because settlements expand, raid, and create ruins in their vicinity
    settlement_positions = []
    for y in range(height):
        for x in range(width):
            if initial_grid[y][x] in (1, 2):
                settlement_positions.append((y, x))

    for y in range(height):
        for x in range(width):
            if interest[y, x] > 0 and initial_grid[y][x] not in (1, 2):
                for sy, sx in settlement_positions:
                    dist = abs(y - sy) + abs(x - sx)  # Manhattan distance
                    if dist <= 5:
                        interest[y, x] += max(0, (5 - dist)) * 0.3

    return interest


def compute_greedy_viewports(interest, width, height, n_viewports,
                             max_tile=15, min_spacing=5, min_score=15.0):
    """
    Greedily place viewports to maximize total covered interest.

    At each step, picks the 15×15 viewport position that maximizes
    *uncovered* interest, then marks those cells as covered.
    Stops when the best remaining viewport has interest below min_score
    (skips predominantly static tiles like ocean/mountain).
    Returns list of (x, y, w, h) tuples.
    """
    # Summed area table for fast viewport scoring
    remaining = interest.copy()
    selected = []

    max_x = max(0, width - max_tile)
    max_y = max(0, height - max_tile)

    for _ in range(n_viewports):
        # Build SAT from remaining interest
        sat = np.zeros((height + 1, width + 1), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                sat[y+1, x+1] = remaining[y, x] + sat[y, x+1] + sat[y+1, x] - sat[y, x]

        best_score = -1
        best_pos = (0, 0)
        for vy in range(max_y + 1):
            for vx in range(max_x + 1):
                score = (sat[vy+max_tile, vx+max_tile] - sat[vy, vx+max_tile]
                         - sat[vy+max_tile, vx] + sat[vy, vx])
                if score > best_score:
                    # Check minimum spacing from already selected viewports
                    cx, cy = vx + max_tile / 2.0, vy + max_tile / 2.0
                    too_close = False
                    for sx, sy, _, _ in selected:
                        scx, scy = sx + max_tile / 2.0, sy + max_tile / 2.0
                        if abs(cx - scx) < min_spacing and abs(cy - scy) < min_spacing:
                            too_close = True
                            break
                    if not too_close:
                        best_score = score
                        best_pos = (vx, vy)

        if best_score <= min_score:
            break
        vx, vy = best_pos
        selected.append((vx, vy, max_tile, max_tile))
        # Zero out the covered area so next viewport picks uncovered regions
        remaining[vy:vy+max_tile, vx:vx+max_tile] = 0

    return selected


def compute_obs_change_rate(obs, initial_grid, width, height):
    """Fraction of cells in an observation viewport that changed from initial state."""
    vp = obs["viewport"]
    obs_grid = obs["grid"]
    changes = 0
    total = 0
    for dy in range(len(obs_grid)):
        for dx in range(len(obs_grid[0])):
            oy, ox = vp["y"] + dy, vp["x"] + dx
            if 0 <= oy < height and 0 <= ox < width:
                if terrain_to_class(initial_grid[oy][ox]) != terrain_to_class(obs_grid[dy][dx]):
                    changes += 1
                total += 1
    return changes / max(total, 1)


def build_observed_dynamism_heatmap(seed_observations, initial_grid, width, height):
    """
    Build a per-cell dynamism heatmap from observations.
    Cells that changed from initial state get a high score.
    Cells near changed cells get a distance-decayed bonus.
    Used to guide re-sampling and cross-seed intelligence.
    """
    dynamism = np.zeros((height, width), dtype=np.float32)
    for obs in seed_observations:
        vp = obs["viewport"]
        obs_grid = obs["grid"]
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                oy, ox = vp["y"] + dy, vp["x"] + dx
                if 0 <= oy < height and 0 <= ox < width:
                    if terrain_to_class(initial_grid[oy][ox]) != terrain_to_class(obs_grid[dy][dx]):
                        dynamism[oy, ox] += 1.0
    return dynamism


def plan_viewports(width, height, num_queries, vw=15, vh=15):
    """Legacy wrapper — returns (vx, vy) list from the tile grid."""
    tiles = compute_tile_grid(width, height, max_tile=min(vw, vh))
    return [(tx, ty) for (tx, ty, tw, th) in tiles[:num_queries]]


# --- Training ---

def train_unet_live(observations, initial_states, encoded_grids, width, height, epochs=80, lr=1e-3):
    """Trains the U-Net on full 40x40 maps using a masked loss for observed regions."""
    model = make_model("unet_cond").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss(reduction='none') # 'none' allows us to apply the observation mask

    # Group observations by seed
    obs_by_seed = {}
    for obs in observations:
        sid = obs["seed_index"]
        obs_by_seed.setdefault(sid, []).append(obs)

    min_remaining = 60
    model.train()

    for epoch in range(epochs):
        if time_remaining() < min_remaining:
            print(f"  Stopping training at epoch {epoch+1}/{epochs} — deadline approaching")
            break

        total_loss = 0.0
        n_seeds = 0

        # Train one full map (seed) at a time
        for seed_idx, seed_obs in obs_by_seed.items():
            if not seed_obs:
                continue

            # 1. Prepare Inputs
            features = encoded_grids[seed_idx]
            obs_features = encode_obs_channels(seed_obs, width, height)
            x = np.concatenate([features, obs_features], axis=0) # (21, 40, 40)
            X_t = torch.tensor(x).unsqueeze(0).to(DEVICE)        # (1, 21, 40, 40)

            # 2. Prepare Targets & Mask
            target_grid = np.zeros((height, width), dtype=np.int64)
            mask_grid = np.zeros((height, width), dtype=np.float32)

            for obs in seed_obs:
                vp = obs["viewport"]
                vx, vy = vp["x"], vp["y"]
                grid = obs["grid"]
                for dy in range(len(grid)):
                    for dx in range(len(grid[0])):
                        gy, gx = vy + dy, vx + dx
                        if 0 <= gy < height and 0 <= gx < width:
                            target_grid[gy, gx] = terrain_to_class(grid[dy][dx])
                            mask_grid[gy, gx] = 1.0 # Mark this pixel as "observed"

            # Add static cells to mask
            initial_grid = initial_states[seed_idx]["grid"]
            for y in range(height):
                for x in range(width):
                    cell = initial_grid[y][x]
                    if cell == 10:  # Ocean
                        target_grid[y, x] = 0
                        mask_grid[y, x] = 1.0
                    elif cell == 5: # Mountain
                        target_grid[y, x] = 5
                        mask_grid[y, x] = 1.0

            Y_t = torch.tensor(target_grid).unsqueeze(0).to(DEVICE) 
            Mask_t = torch.tensor(mask_grid).unsqueeze(0).to(DEVICE)

            # 3. Forward Pass & Masked Loss
            optimizer.zero_grad()
            probs = model(X_t, temperature=1.0) 
            
            log_probs = torch.log(probs.clamp(min=1e-8))
            raw_loss = loss_fn(log_probs, Y_t) 
            
            masked_loss = (raw_loss * Mask_t).sum() / Mask_t.sum().clamp(min=1.0)
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            n_seeds += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_seeds, 1)
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} ({time_remaining():.0f}s left)")

    model.eval()
    return model

# --- Checkpoint loading ---

def load_pretrained_checkpoint():
    """Load the latest pretrained checkpoint from train_cnn.py, if available."""
    ckpt_dir = CHECKPOINT_DIR_MAP.get(MODEL_ARCH, CHECKPOINT_DIR)
    latest = os.path.join(ckpt_dir, "cnn_latest.pt")
    if os.path.isfile(latest):
        path = latest
    else:
        # Fall back to highest epoch checkpoint
        if not os.path.isdir(ckpt_dir):
            return None
        files = [f for f in os.listdir(ckpt_dir)
                 if f.startswith("cnn_epoch_") and f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda f: int(f.replace("cnn_epoch_", "").replace(".pt", "")),
                   reverse=True)
        path = os.path.join(ckpt_dir, files[0])

    print(f"  Loading pretrained checkpoint: {os.path.basename(path)}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("model_arch") or ckpt.get("metadata", {}).get("model_arch", "quick")
    model = make_model(arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"  Checkpoint epoch={epoch}, val_loss={val_loss}, arch={arch}")
    return model


# --- Prediction ---

def predict_full_map(model, features, width, height, obs_features=None):
    """
    Run the CNN to produce (H, W, 6) probability predictions.
    Observations are already encoded in obs_features (input channels).
    No hard override — the model output is submitted directly.
    """
    with torch.no_grad():
        if obs_features is not None:
            x = np.concatenate([features, obs_features], axis=0)
        else:
            x = features
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)

        probs = model(x, temperature=0.85)
        probs = probs.squeeze(0)
        probs = probs.permute(1, 2, 0).cpu().numpy()

    return probs




# --- Bayesian Blending ---

def bayesian_blend(cnn_pred, observations, initial_grid, width, height,
                   strength=5.0):
    """
    Blend CNN predictions with empirical observation counts.

    For observed pixels:
        posterior[c] ∝ strength * cnn_pred[c] + obs_count[c]
    For unobserved pixels:
        prediction unchanged (CNN only).

    Args:
        cnn_pred: (H, W, 6) CNN probability predictions
        observations: list of {"seed_index": int, "viewport": {...}, "grid": [[...]]}
                      filtered to a single seed
        initial_grid: the initial terrain grid for this seed
        width, height: map dimensions
        strength: pseudo-count weight for CNN prior (higher = trust CNN more)

    Returns:
        (H, W, 6) blended probability predictions
    """
    # Build empirical counts per pixel from observations
    obs_counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
    obs_hits = np.zeros((height, width), dtype=np.float32)

    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        obs_grid = obs["grid"]
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    cls = mapping.get(obs_grid[dy][dx], 0)
                    obs_counts[gy, gx, cls] += 1.0
                    obs_hits[gy, gx] += 1.0

    # Blend: for observed pixels where CNN is uncertain, compute posterior
    blended = cnn_pred.copy()
    observed_mask = obs_hits > 0

    if observed_mask.any():
        # Compute per-pixel entropy of CNN predictions (nat)
        p = cnn_pred[observed_mask]                          # (P, 6)
        cnn_entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)  # (P,)
        max_entropy = np.log(NUM_CLASSES)                    # ln(6) ≈ 1.79
        # Only blend pixels where CNN entropy exceeds threshold
        entropy_thresh = 0.3 * max_entropy                   # ~0.54 nat
        uncertain = cnn_entropy > entropy_thresh             # (P,) bool

        if uncertain.any():
            prior_counts = strength * p[uncertain]           # (U, 6)
            empirical = obs_counts[observed_mask][uncertain]  # (U, 6)
            posterior = prior_counts + empirical
            posterior = np.maximum(posterior, 1e-8)
            posterior = posterior / posterior.sum(axis=-1, keepdims=True)
            posterior = np.maximum(posterior, PROB_FLOOR)
            posterior = posterior / posterior.sum(axis=-1, keepdims=True)
            # Write back only the uncertain observed pixels
            idx = np.where(observed_mask)
            uy = idx[0][uncertain]
            ux = idx[1][uncertain]
            blended[uy, ux] = posterior

    return blended


# --- Fallback ---

def build_prior_prediction(initial_grid, width, height):
    """Fallback prior-based prediction when no training data is available."""
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
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


# --- Submission helpers ---

def submit_fallback(round_id, seeds_count, initial_states, width, height):
    """
    Immediately submit prior-based predictions for all seeds.
    Guarantees a non-zero score even if the CNN pipeline runs out of time.
    Resubmitting later overwrites these — only the last submission counts.
    """
    print("\n--- Submitting prior-based fallback (safe score) ---")
    for seed_idx in range(seeds_count):
        grid = initial_states[seed_idx]["grid"]
        prediction = build_prior_prediction(grid, width, height)
        try:
            result = submit_prediction(round_id, seed_idx, prediction)
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except requests.HTTPError as e:
            print(f"  Seed {seed_idx} fallback FAILED: {e}")


def submit_cnn_predictions(round_id, model, encoded_grids, initial_states,
                           seeds_count, width, height,
                           observations=None, arch=None, **_kwargs):
    """Submit UNet predictions for all seeds (overwrites fallback).

    Observations are encoded as input channels for the unet_cond model.
    The raw CNN output is submitted directly (no post-processing).
    """
    print(f"\n--- Submitting UNet predictions (arch={arch or MODEL_ARCH}) ---")

    # Group observations by seed
    obs_by_seed = {}
    if observations:
        for obs in observations:
            sid = obs["seed_index"]
            obs_by_seed.setdefault(sid, []).append(obs)

    for seed_idx in range(seeds_count):
        if past_deadline():
            print(f"  Deadline reached at seed {seed_idx}, keeping fallback for remaining seeds")
            break
        if seed_idx not in encoded_grids:
            encoded_grids[seed_idx] = encode_initial_grid(
                initial_states[seed_idx]["grid"], width, height
            )
        features = encoded_grids[seed_idx]

        # Build observation channels
        seed_obs = obs_by_seed.get(seed_idx, [])
        obs_feat = encode_obs_channels(seed_obs, width, height)

        prediction = predict_full_map(model, features, width, height,
                                      obs_features=obs_feat)

        try:
            result = submit_prediction(round_id, seed_idx, prediction)
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except requests.HTTPError as e:
            print(f"  Seed {seed_idx} submit FAILED: {e}")
            if e.response is not None:
                print(f"    {e.response.text[:200]}")


# --- Main ---

def main():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT_MINUTES * 60

    print("=" * 50)
    print("  Astar Island — CNN Prediction")
    print("=" * 50)
    print(f"  Device: {DEVICE or 'N/A (no PyTorch)'}")
    print(f"  PyTorch: {'available' if TORCH_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Time limit: {TIME_LIMIT_MINUTES:.0f} min (deadline in {time_remaining():.0f}s)")

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

    # Cache round data to disk for offline analysis
    _save_round_data(round_id, detail)

    # Step 3: Submit fallback IMMEDIATELY (guarantees a score)
    submit_fallback(round_id, seeds_count, initial_states, width, height)

    if not TORCH_AVAILABLE:
        print("\nPyTorch unavailable — fallback submitted, skipping CNN pipeline.")
        return

    if past_deadline():
        print("\nDeadline already reached. Fallback submitted, exiting.")
        return

    # --- CNN pipeline (any error here is non-fatal — fallback already submitted) ---
    try:
        # Step 4: Collect observations (or load previously saved ones if budget spent)
        print(f"\n--- Collecting observations ({time_remaining():.0f}s remaining) ---")
        observations = collect_observations(
            round_id, seeds_count, initial_states, width, height
        )
        if not observations:
            observations = _load_observations(round_id)
            if observations:
                print(f"  Loaded {len(observations)} previously saved observations")

        if past_deadline():
            print("\nDeadline reached after observations. Fallback submission stands.")
            return

        # Step 5: Load pretrained unet_cond checkpoint
        print(f"\n--- Loading pretrained checkpoint (arch={MODEL_ARCH}) ---")
        model = load_pretrained_checkpoint()

        if model is None:
            print(f"\n  WARNING: No checkpoint found. Fallback submission stands.")
            return

        # Step 6: Encode grids and submit predictions
        encoded_grids = {}
        for seed_idx in range(seeds_count):
            encoded_grids[seed_idx] = encode_initial_grid(
                initial_states[seed_idx]["grid"], width, height
            )
        submit_cnn_predictions(
            round_id, model, encoded_grids, initial_states,
            seeds_count, width, height,
            observations=observations,
            arch=MODEL_ARCH,
        )
    except Exception as e:
        print(f"\nERROR in CNN pipeline: {e}")
        print("Fallback submission still stands — you will get a score.")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Done in {elapsed:.0f}s. Check results at app.ainm.no")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
