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
PROB_FLOOR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
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

class QuickCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1)

    def forward(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        logits = self.out_conv(x)

        # Convert logits to probabilities with safety floor
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class QuickCNN3(nn.Module):
    """QuickCNN with one additional hidden conv layer (receptive field 7x7)."""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.drop3 = nn.Dropout2d(p=dropout)
        self.out_conv = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1)

    def forward(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        x = self.drop3(F.relu(self.conv3(x)))
        logits = self.out_conv(x)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


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

    def forward(self, x):
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

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


MODEL_REGISTRY = {
    "quick": QuickCNN,
    "quick3": QuickCNN3,
    "unet": MiniUNet,
    "unet_aug": lambda **kw: MiniUNet(dropout=kw.get('dropout', 0.1)),
    "unet_obs": lambda **kw: MiniUNet(dropout=kw.get('dropout', 0.1),
                                       in_channels=14 + OBS_CHANNELS),
}

CHECKPOINT_DIR_MAP = {
    "quick": os.path.join(SCRIPT_DIR, "checkpoints"),
    "quick3": os.path.join(SCRIPT_DIR, "checkpoints_quick3"),
    "unet": os.path.join(SCRIPT_DIR, "checkpoints_unet"),
    "unet_aug": os.path.join(SCRIPT_DIR, "checkpoints_unet_aug"),
    "unet_obs": os.path.join(SCRIPT_DIR, "checkpoints_unet_obs"),
}

MODEL_ARCH = os.environ.get("ASTAR_MODEL", "quick")


def make_model(arch="quick", **kwargs):
    cls = MODEL_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: {list(MODEL_REGISTRY)}")
    return cls(**kwargs)


# --- Training Data Collection ---

def collect_observations(round_id, seeds_count, initial_states, width, height):
    """
    Query the simulator using a full-size viewport grid with priority ordering.
    Assumes stochastic simulation: covers all tiles via round-robin, then
    spends remaining budget on dynamic re-queries and interest-centred viewports.
    """
    budget = check_budget()
    if not budget:
        return []
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        print("No queries remaining — will train on priors only.")
        return []

    # --- 1. Compute non-overlapping tile partition ---
    tiles = compute_tile_grid(width, height)
    n_tiles = len(tiles)

    # --- 2. Per-seed priority queues ---
    seed_queues = []
    for s in range(seeds_count):
        grid = initial_states[s]["grid"]
        scored = [(score_tile(grid, t, width, height), t) for t in tiles]
        scored.sort(reverse=True)
        seed_queues.append(scored)

    observations = []
    seed_obs = [[] for _ in range(seeds_count)]
    queries_done = 0

    query_limit = remaining
    total_needed = n_tiles * seeds_count
    print(f"  Strategy (STOCHASTIC): {n_tiles} tiles/seed × {seeds_count} seeds = "
          f"{total_needed} for full coverage  (budget {remaining})")

    # --- 4. Round-robin with re-ranking ---
    round_num = 0

    while queries_done < query_limit:
        if past_deadline():
            print(f"  Deadline reached after {queries_done} queries")
            break

        # Re-rank remaining tiles after each full round (except first)
        if round_num > 0:
            for s in range(seeds_count):
                if not seed_queues[s]:
                    continue
                grid = initial_states[s]["grid"]
                rescored = [
                    (rescore_tile(t, grid, seed_obs[s], width, height, sc), t)
                    for sc, t in seed_queues[s]
                ]
                rescored.sort(reverse=True)
                seed_queues[s] = rescored

        queried_any = False
        for s in range(seeds_count):
            if past_deadline() or queries_done >= query_limit:
                break
            if not seed_queues[s]:
                continue

            _, (tx, ty, tw, th) = seed_queues[s].pop(0)
            try:
                result = simulate(round_id, s, tx, ty, tw, th)
                vp = result["viewport"]
                obs = {"seed_index": s, "viewport": vp, "grid": result["grid"]}
                observations.append(obs)
                seed_obs[s].append(obs)
                queries_done += 1
                queried_any = True

                tiles_left = len(seed_queues[s])
                used = result.get("queries_used", "?")
                max_q = result.get("queries_max", "?")
                print(f"  Seed {s} tile {n_tiles - tiles_left}/{n_tiles}: "
                      f"({vp['x']},{vp['y']}) {vp['w']}x{vp['h']}  "
                      f"budget {used}/{max_q}")

                if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                    print("  Budget exhausted!")
                    _save_observations(observations, round_id)
                    return observations

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"  Rate limited at seed {s}")
                    _save_observations(observations, round_id)
                    return observations
                raise

        if not queried_any:
            break
        round_num += 1

    # --- 4. Extra queries on most dynamic tiles ---
    if queries_done < query_limit and not past_deadline():
        extra_budget = query_limit - queries_done
        print(f"  Full coverage done ({queries_done} queries). "
              f"Using {extra_budget} extra queries on dynamic tiles.")

        # Score each observed tile by change rate from initial state
        tile_dynamics = []  # (change_rate, seed, tile_xywh)
        for s in range(seeds_count):
            grid = initial_states[s]["grid"]
            for obs in seed_obs[s]:
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
                    tile_dynamics.append((changes / total, s,
                                          (vp["x"], vp["y"], vp["w"], vp["h"])))
        tile_dynamics.sort(reverse=True)

        # Also generate interest-centered extra viewports (always 15×15)
        extra_interest = []
        for s in range(seeds_count):
            grid = initial_states[s]["grid"]
            queried_tiles = [(o["viewport"]["x"], o["viewport"]["y"],
                              o["viewport"]["w"], o["viewport"]["h"])
                             for o in seed_obs[s]]
            interest_vps = compute_interest_viewports(
                grid, width, height, existing_tiles=queried_tiles,
                n_extra=max(2, extra_budget // seeds_count)
            )
            for ivp in interest_vps:
                # Score by initial interest (approximate)
                sc = score_tile(grid, ivp, width, height)
                extra_interest.append((sc, s, ivp))
        extra_interest.sort(reverse=True)

        # Interleave: prioritise observed-dynamic re-queries, then interest viewports
        extra_targets = []
        di, ii = 0, 0
        while len(extra_targets) < extra_budget:
            added = False
            if di < len(tile_dynamics):
                extra_targets.append(tile_dynamics[di])
                di += 1
                added = True
            if ii < len(extra_interest) and len(extra_targets) < extra_budget:
                extra_targets.append(extra_interest[ii])
                ii += 1
                added = True
            if not added:
                break

        for score_val, s, (tx, ty, tw, th) in extra_targets:
            if queries_done >= query_limit or past_deadline():
                break
            try:
                result = simulate(round_id, s, tx, ty, tw, th)
                vp = result["viewport"]
                obs = {"seed_index": s, "viewport": vp, "grid": result["grid"]}
                observations.append(obs)
                seed_obs[s].append(obs)
                queries_done += 1

                used = result.get("queries_used", "?")
                max_q = result.get("queries_max", "?")
                print(f"  Extra: seed {s} ({vp['x']},{vp['y']}) {vp['w']}x{vp['h']}  "
                      f"score={score_val:.2f}  budget {used}/{max_q}")

                if isinstance(used, int) and isinstance(max_q, int) and used >= max_q:
                    break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    break
                raise

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


def rescore_tile(tile, initial_grid, seed_observations, width, height,
                 base_score):
    """
    Re-score an unqueried tile after observing other tiles for the same seed.
    Boosts priority if nearby observed tiles show dynamic activity (cells that
    changed from their initial state).
    """
    if not seed_observations:
        return base_score

    tx, ty, tw, th = tile
    tile_cx = tx + tw / 2.0
    tile_cy = ty + th / 2.0
    bonus = 0.0

    for obs in seed_observations:
        vp = obs["viewport"]
        obs_grid = obs["grid"]
        obs_cx = vp["x"] + vp["w"] / 2.0
        obs_cy = vp["y"] + vp["h"] / 2.0

        dist = ((tile_cx - obs_cx) ** 2 + (tile_cy - obs_cy) ** 2) ** 0.5
        if dist > 30:
            continue
        proximity = max(0.0, 1.0 - dist / 30.0)

        # Count cells that changed from initial state
        changes = 0
        total = 0
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                oy, ox = vp["y"] + dy, vp["x"] + dx
                if 0 <= oy < height and 0 <= ox < width:
                    initial_cls = terrain_to_class(initial_grid[oy][ox])
                    observed_cls = terrain_to_class(obs_grid[dy][dx])
                    total += 1
                    if initial_cls != observed_cls:
                        changes += 1

        if total > 0:
            change_rate = changes / total
            bonus += change_rate * proximity * 5.0

    return base_score + bonus


def compute_interest_viewports(initial_grid, width, height, max_tile=15,
                               existing_tiles=None, n_extra=5):
    """
    Generate extra viewport positions centred on the most dynamic/interesting
    map regions.  Each viewport is always max_tile×max_tile, clamped to bounds.

    Ranks every possible viewport position by an interest heatmap built from
    the initial terrain, then returns the top-n positions that are most
    different from the already-queried tiles.
    """
    # Build per-cell interest score
    interest = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            if cell == 1:        # Settlement
                interest[y, x] = 5.0
            elif cell == 2:      # Port
                interest[y, x] = 6.0
            elif cell == 4:      # Forest
                interest[y, x] = 0.3
            elif cell in (0, 11):  # Plains
                interest[y, x] = 0.5
            # Ocean/Mountain stay 0

    # Coastal bonus
    for y in range(height):
        for x in range(width):
            if interest[y, x] > 0:
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if 0 <= ny < height and 0 <= nx < width:
                        if initial_grid[ny][nx] == 10:
                            interest[y, x] += 0.3
                            break

    # Summed area table for fast viewport scoring
    sat = np.zeros((height + 1, width + 1), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            sat[y+1, x+1] = interest[y, x] + sat[y, x+1] + sat[y+1, x] - sat[y, x]

    def viewport_score(vx, vy, vw, vh):
        return (sat[vy+vh, vx+vw] - sat[vy, vx+vw] - sat[vy+vh, vx] + sat[vy, vx])

    # Existing tile centers for diversity penalty
    existing_centers = []
    if existing_tiles:
        for (ex, ey, ew, eh) in existing_tiles:
            existing_centers.append((ex + ew / 2.0, ey + eh / 2.0))

    max_x = width - max_tile
    max_y = height - max_tile
    candidates = []
    # Sample every possible position (stride 1 on a 40×40 map is only ~676 positions)
    for vy in range(max(0, max_y) + 1):
        for vx in range(max(0, max_x) + 1):
            score = viewport_score(vx, vy, max_tile, max_tile)
            # Diversity: penalise overlap with existing tile centers
            cx, cy = vx + max_tile / 2.0, vy + max_tile / 2.0
            for ecx, ecy in existing_centers:
                dist = ((cx - ecx)**2 + (cy - ecy)**2) ** 0.5
                if dist < max_tile:
                    score *= (0.3 + 0.7 * dist / max_tile)  # mild penalty
            candidates.append((score, vx, vy))

    candidates.sort(reverse=True)

    # Deduplicate: skip candidates too close to already selected ones
    selected = []
    for score, vx, vy in candidates:
        if len(selected) >= n_extra:
            break
        cx, cy = vx + max_tile / 2.0, vy + max_tile / 2.0
        too_close = False
        for sx, sy in selected:
            if abs(cx - (sx + max_tile/2.0)) < 3 and abs(cy - (sy + max_tile/2.0)) < 3:
                too_close = True
                break
        if not too_close:
            selected.append((vx, vy))

    return [(vx, vy, max_tile, max_tile) for vx, vy in selected]


def plan_viewports(width, height, num_queries, vw=15, vh=15):
    """Legacy wrapper — returns (vx, vy) list from the tile grid."""
    tiles = compute_tile_grid(width, height, max_tile=min(vw, vh))
    return [(tx, ty) for (tx, ty, tw, th) in tiles[:num_queries]]


# --- Training ---

def build_training_data(observations, initial_states, width, height):
    """
    Build pixel-level training pairs from observations.

    For each observed cell: input = 14-channel feature vector at that cell,
    target = observed class index.

    Also generates "static" training samples from cells we know won't change
    (ocean, mountain) across the full map to anchor the model.
    """
    # Pre-encode all initial grids
    encoded_grids = {}
    for obs in observations:
        si = obs["seed_index"]
        if si not in encoded_grids:
            encoded_grids[si] = encode_initial_grid(
                initial_states[si]["grid"], width, height
            )

    # Collect pixel samples from observations
    X_pixels = []  # (14,) feature vectors
    y_pixels = []  # class index

    for obs in observations:
        si = obs["seed_index"]
        features = encoded_grids[si]
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        obs_grid = obs["grid"]
        vh, vw = len(obs_grid), len(obs_grid[0])

        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    cls = terrain_to_class(obs_grid[dy][dx])
                    X_pixels.append(features[:, y, x])
                    y_pixels.append(cls)

    # Add static cell samples from ALL seeds (ocean, mountain — known outcomes)
    for si in range(len(initial_states)):
        if si not in encoded_grids:
            encoded_grids[si] = encode_initial_grid(
                initial_states[si]["grid"], width, height
            )
        features = encoded_grids[si]
        grid = initial_states[si]["grid"]
        for y in range(height):
            for x in range(width):
                cell = grid[y][x]
                if cell == 10:  # Ocean → class 0
                    X_pixels.append(features[:, y, x])
                    y_pixels.append(0)
                elif cell == 5:  # Mountain → class 5
                    X_pixels.append(features[:, y, x])
                    y_pixels.append(5)

    X = np.array(X_pixels, dtype=np.float32)
    y = np.array(y_pixels, dtype=np.int64)
    print(f"Training data: {len(X)} pixel samples ({len(observations)} viewports + static cells)")
    return X, y, encoded_grids


def train_model(X, y, epochs=80, lr=1e-3, batch_size=512):
    """Train the CNN on pixel-level data. Stops early if deadline approached."""
    model = make_model(MODEL_ARCH).to(DEVICE)

    X_t = torch.tensor(X).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    y_t = torch.tensor(y).to(DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Reserve 60s for prediction + submission
    min_remaining = 60

    model.train()
    for epoch in range(epochs):
        if time_remaining() < min_remaining:
            print(f"  Stopping training at epoch {epoch+1}/{epochs} — deadline approaching")
            break

        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            x = F.relu(model.conv1(X_batch))
            x = F.relu(model.conv2(x))
            logits = model.out_conv(x)
            logits = logits.squeeze(-1).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            remaining = time_remaining()
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} ({remaining:.0f}s left)")

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
    Run the trained CNN on the full feature map to get per-cell probabilities.

    Input features: (C, H, W) — 14 channels for standard, 21 for obs-conditioned
    obs_features: optional (7, H, W) observation channels to concatenate
    Output: (H, W, 6) numpy array of class probabilities
    """
    with torch.no_grad():
        if obs_features is not None:
            x = np.concatenate([features, obs_features], axis=0)  # (21, H, W)
        else:
            x = features
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)
        probs = model(x)  # (1, 6, H, W)
        probs = probs.squeeze(0)  # (6, H, W)
        # Transpose to (H, W, 6) for submission
        probs = probs.permute(1, 2, 0).cpu().numpy()

    # Safety: enforce floor and renormalize
    probs = np.maximum(probs, PROB_FLOOR)
    probs = probs / probs.sum(axis=-1, keepdims=True)
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
                           observations=None, arch=None):
    """Submit CNN-based predictions for all seeds (overwrites fallback).

    For unet_obs architecture, observations are encoded as extra input channels.
    """
    is_obs_model = (arch == "unet_obs")
    label = "obs-conditioned CNN" if is_obs_model and observations else "CNN"
    print(f"\n--- Submitting {label} predictions ---")

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

        # Build observation channels for obs-conditioned model
        obs_feat = None
        if is_obs_model:
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
        # Step 4: Always collect observations (saved to disk for future training)
        print(f"\n--- Collecting observations ({time_remaining():.0f}s remaining) ---")
        observations = collect_observations(
            round_id, seeds_count, initial_states, width, height
        )

        if past_deadline():
            print("\nDeadline reached after observations. Fallback submission stands.")
            return

        # Step 5: Try loading pretrained checkpoint
        print(f"\n--- Checking for pretrained checkpoint ---")
        model = load_pretrained_checkpoint()

        if model is not None:
            # Pretrained model found — use it directly for predictions
            # If we have observations, blend them with CNN predictions
            print(f"\n--- Submitting pretrained CNN predictions ---")
            encoded_grids = {}
            for seed_idx in range(seeds_count):
                encoded_grids[seed_idx] = encode_initial_grid(
                    initial_states[seed_idx]["grid"], width, height
                )
            submit_cnn_predictions(
                round_id, model, encoded_grids, initial_states,
                seeds_count, width, height,
            )
        else:
            # No checkpoint — train from observations collected above
            print(f"\n--- No pretrained checkpoint found, training from observations ---")

            if not observations:
                print("\nNo observations collected. Fallback submission stands.")
                return

            # Step 6: Build training data and train CNN
            print(f"\n--- Building training data ({time_remaining():.0f}s remaining) ---")
            X, y, encoded_grids = build_training_data(
                observations, initial_states, width, height
            )

            if past_deadline():
                print("\nDeadline reached before training. Fallback submission stands.")
                return

            print(f"\n--- Training CNN ({time_remaining():.0f}s remaining) ---")
            model = train_model(X, y)

            if past_deadline():
                print("\nDeadline reached after training. Fallback submission stands.")
                return

            # Step 7: Predict and resubmit (overwrites the fallback)
            submit_cnn_predictions(
                round_id, model, encoded_grids, initial_states,
                seeds_count, width, height
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
