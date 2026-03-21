"""
Astar Island — Better Monte Carlo
==================================

Improved dynamics model addressing error propagation & long-range effects:
  - Multi-horizon conditioning: Δ ∈ {1, 2, 5, 10}
  - Unrolled training with multi-step loss
  - Scheduled sampling (teacher forcing → model-predicted mix)
  - Mixed-stride rollout at inference to reduce compounding errors

USAGE:
  python better_monte_carlo.py train [--epochs 300] [--reset]
  python better_monte_carlo.py evaluate [--rollouts 200]
"""

import os, sys, json, time, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# ─── CONFIG ───
BASE_DIR = Path(".")
LOCAL_REPLAY_DIR = BASE_DIR / "simulation_replays"
LOCAL_GT_DIR     = BASE_DIR / "ground_truth"
LOCAL_OBS_DIR    = BASE_DIR / "observations"
CKPT_DIR         = BASE_DIR / "checkpoints_better_mc"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
CODE_TO_IDX = {c: i for i, c in enumerate(TERRAIN_CODES)}
IDX_TO_CODE = {i: c for c, i in CODE_TO_IDX.items()}
NUM_TERRAIN = 8
SUBMIT_CLASSES = 6
TERRAIN_TO_SUBMIT = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
IDX8_TO_SUBMIT6 = [TERRAIN_TO_SUBMIT[IDX_TO_CODE[i]] for i in range(NUM_TERRAIN)]

PROB_FLOOR = 0.01
MAP_H, MAP_W = 40, 40
MAX_TIMESTEP = 50

# Horizons to train on
HORIZONS = [1, 2, 5, 10]
HORIZON_WEIGHTS = {1: 1.0, 2: 0.7, 5: 0.5, 10: 0.3}

# Training
EPOCHS     = 300
LR         = 3e-4
BATCH      = 16
HIDDEN     = 64
CKPT_EVERY = 25

# Unrolled training
UNROLL_H       = 5       # unroll up to 5 steps during training
UNROLL_MIN_EP  = 30      # start unrolling after this many epochs
SS_START       = 0.0     # scheduled sampling: start frac of model-predicted
SS_END         = 0.5     # scheduled sampling: end frac
SS_RAMP_EP     = 200     # ramp over this many epochs

# Inference
DEFAULT_K      = 200
DEFAULT_T      = 50
INFER_STRIDE   = 10

NOISE_PROB = 0.3
NOISE_MAG  = 0.15


# ─── DATA LOADING ───
def load_all_replays():
    replays = []
    d = str(LOCAL_REPLAY_DIR)
    if not os.path.isdir(d):
        # Try data/ subfolder
        d = str(BASE_DIR / "data" / "simulation_replays")
    if not os.path.isdir(d):
        print(f"ERROR: replay dir not found"); return replays
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                replays.append(json.load(fh))
    print(f"Loaded {len(replays)} replays")
    return replays


def load_ground_truth():
    d = str(LOCAL_GT_DIR)
    if not os.path.isdir(d):
        d = str(BASE_DIR / "data" / "ground_truth")
    if not os.path.isdir(d):
        print(f"ERROR: GT dir not found"); return []
    data = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(d, f)) as fh:
            dd = json.load(fh)
        parts = f.replace(".json", "").split("_")
        if len(parts) >= 3:
            dd["_round_number"] = int(parts[0][1:])
            dd["_seed_index"] = int(parts[1][1:])
            dd["_round_id"] = parts[2]
        data.append(dd)
    print(f"Loaded {len(data)} ground truth samples")
    return data


def load_observations(round_id_short):
    path = LOCAL_OBS_DIR / f"observations_{round_id_short}.json"
    if not path.is_file():
        path = BASE_DIR / "data" / "observations" / f"observations_{round_id_short}.json"
    if not path.is_file():
        return []
    with open(path) as f:
        raw = json.load(f)
    return raw.get("observations", []) if isinstance(raw, dict) else raw


# ─── ENCODING ───
def grid_to_onehot(grid, h, w):
    oh = np.zeros((NUM_TERRAIN, h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            oh[CODE_TO_IDX.get(grid[y][x], 0), y, x] = 1.0
    return oh


def grid_to_class_indices(grid, h, w):
    idx = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            idx[y, x] = CODE_TO_IDX.get(grid[y][x], 0)
    return idx


# ─── AUGMENTATION ───
def augment_pair(x, y):
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    x = torch.rot90(x, k, [1, 2])
    y = torch.rot90(y, k, [0, 1])
    if flip:
        x = torch.flip(x, [2])
        y = torch.flip(y, [1])
    return x, y


def augment_grid(x, y):
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    x = torch.rot90(x, k, [1, 2])
    y = torch.rot90(y, k, [1, 2])
    if flip:
        x = torch.flip(x, [2])
        y = torch.flip(y, [2])
    return x, y


def add_noise(x, prob=NOISE_PROB, mag=NOISE_MAG):
    mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < prob).float()
    noise = torch.randn_like(x) * mag
    return F.softmax((x + noise * mask) / 0.1, dim=1)


# ─── FiLM CONDITIONING (timestep + horizon) ───
class DualFiLM(nn.Module):
    """FiLM conditioning from both timestep t and horizon delta."""
    def __init__(self, n_channels, max_t=MAX_TIMESTEP + 1, max_delta=max(HORIZONS) + 1):
        super().__init__()
        self.t_embed = nn.Embedding(max_t, 32)
        self.d_embed = nn.Embedding(max_delta, 32)
        self.fc = nn.Linear(64, n_channels * 2)

    def forward(self, t, delta):
        e = F.gelu(torch.cat([self.t_embed(t), self.d_embed(delta)], dim=1))
        params = self.fc(e)
        scale, shift = params.chunk(2, dim=1)
        return scale[:, :, None, None], shift[:, :, None, None]


# ─── MODEL: MultiHorizonDynamics ───
class MultiHorizonDynamics(nn.Module):
    """
    Dilated-CNN dynamics conditioned on both timestep and prediction horizon.
    Learns s_t -> s_{t+Δ} for multiple Δ values simultaneously.
    Identity skip connection biases toward "no change".
    """

    def __init__(self, hidden=HIDDEN, skip_init=5.0):
        super().__init__()
        self.hidden = hidden

        self.conv1 = nn.Conv2d(NUM_TERRAIN, hidden, 3, padding=1, dilation=1)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.bn2   = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4)
        self.bn3   = nn.BatchNorm2d(hidden)
        self.conv4 = nn.Conv2d(hidden, hidden, 3, padding=8, dilation=8)
        self.bn4   = nn.BatchNorm2d(hidden)

        self.film1 = DualFiLM(hidden)
        self.film2 = DualFiLM(hidden)
        self.film3 = DualFiLM(hidden)
        self.film4 = DualFiLM(hidden)

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, NUM_TERRAIN, 1),
        )

        self.skip_gain = nn.Parameter(torch.tensor(skip_init))

    def forward(self, x, t, delta):
        """
        x: (B, NUM_TERRAIN, H, W) one-hot input
        t: (B,) timestep
        delta: (B,) horizon
        """
        h = self.conv1(x)
        h = self.bn1(h)
        s, b = self.film1(t, delta)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv2(h)
        h = self.bn2(h)
        s, b = self.film2(t, delta)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv3(h)
        h = self.bn3(h)
        s, b = self.film3(t, delta)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv4(h)
        h = self.bn4(h)
        s, b = self.film4(t, delta)
        h = F.gelu(h * (1 + s) + b)

        logits = self.head(h) + self.skip_gain * x
        return logits

    def predict_probs(self, x, t, delta):
        return F.softmax(self.forward(x, t, delta), dim=1)


# ─── MULTI-HORIZON DATASET ───
def build_multi_horizon_dataset(replays):
    """Build training pairs for all horizons: (x_t, t, delta, y_{t+delta})."""
    Xs, Ts, Ds, Ys = [], [], [], []
    for r in replays:
        h, w, frames = r["height"], r["width"], r["frames"]
        n = len(frames)
        for delta in HORIZONS:
            for i in range(n - delta):
                Xs.append(grid_to_onehot(frames[i]["grid"], h, w))
                Ts.append(frames[i]["step"])
                Ds.append(delta)
                Ys.append(grid_to_class_indices(frames[i + delta]["grid"], h, w))
    if not Xs:
        return None
    X = np.stack(Xs)
    T = np.array(Ts, dtype=np.int64)
    D = np.array(Ds, dtype=np.int64)
    Y = np.stack(Ys)
    for delta in HORIZONS:
        cnt = (D == delta).sum()
        print(f"  Horizon Δ={delta:2d}: {cnt} pairs")
    print(f"  Total: {len(X)} pairs from {len(replays)} replays")
    return X, T, D, Y


def build_sequence_segments(replays, seg_len=6):
    """Build contiguous frame segments for unrolled training.
    Each segment: list of (onehot, timestep, class_idx) for seg_len consecutive frames.
    """
    segments = []
    for r in replays:
        h, w, frames = r["height"], r["width"], r["frames"]
        n = len(frames)
        for start in range(n - seg_len + 1):
            seg = []
            for j in range(seg_len):
                f = frames[start + j]
                oh = grid_to_onehot(f["grid"], h, w)
                ci = grid_to_class_indices(f["grid"], h, w)
                seg.append((oh, f["step"], ci))
            segments.append(seg)
    print(f"  Sequence segments (len={seg_len}): {len(segments)}")
    return segments


# ─── LOSS ───
def compute_class_weights(Y, nc=NUM_TERRAIN, smooth=0.1):
    counts = np.bincount(Y.flatten(), minlength=nc).astype(np.float64)
    counts = np.maximum(counts, 1)
    w = 1.0 / (counts / counts.sum() + smooth)
    w = w / w.sum() * nc
    return torch.tensor(w, dtype=torch.float32)


# ─── CHECKPOINT ───
def latest_ckpt(d, pfx="mhdyn"):
    d = str(d)
    os.makedirs(d, exist_ok=True)
    pat = f"{pfx}_epoch_"
    fs = [f for f in os.listdir(d) if f.startswith(pat) and f.endswith(".pt")]
    if not fs:
        return None
    fs.sort(key=lambda f: int(f.replace(pat, "").replace(".pt", "")))
    return os.path.join(d, fs[-1])


def save_ckpt(model, opt, epoch, tl, meta, d, pfx="mhdyn"):
    d = str(d)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{pfx}_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": tl, "metadata": meta}, p)
    return p


def load_ckpt(path, model, opt=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    if opt and "optimizer_state_dict" in ck:
        opt.load_state_dict(ck["optimizer_state_dict"])
    return ck


# ─── TRAINING ───
def train(replays, epochs=EPOCHS, lr=LR, bs=BATCH, reset=False):
    print(f"\n{'='*60}")
    print(f"  MultiHorizonDynamics Training")
    print(f"  horizons={HORIZONS}, unroll_H={UNROLL_H}")
    print(f"  scheduled sampling: {SS_START} -> {SS_END} over {SS_RAMP_EP} epochs")
    print(f"{'='*60}")

    # Phase 1 data: multi-horizon pairs
    dataset = build_multi_horizon_dataset(replays)
    if dataset is None:
        print("ERROR: no data"); return
    X, T, D, Y = dataset
    cw = compute_class_weights(Y).to(DEVICE)

    # Phase 2 data: sequence segments for unrolled training
    segments = build_sequence_segments(replays, seg_len=UNROLL_H + 1)

    # Model
    model = MultiHorizonDynamics().to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Params: {npar:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr * 0.01)

    start_ep = 1
    if not reset:
        cp = latest_ckpt(CKPT_DIR)
        if cp:
            print(f"  Resuming from {os.path.basename(cp)}")
            start_ep = load_ckpt(cp, model, opt)["epoch"] + 1

    # Tensors for multi-horizon training
    X_d = torch.tensor(X, device=DEVICE)
    T_d = torch.tensor(T, device=DEVICE)
    D_d = torch.tensor(D, device=DEVICE)
    Y_d = torch.tensor(Y, device=DEVICE)

    n = len(X_d)
    meta = {"model": "MultiHorizonDynamics", "horizons": HORIZONS,
            "unroll": UNROLL_H, "n_pairs": n, "n_segs": len(segments)}

    t0 = time.time()
    best_loss = float("inf")

    for ep in range(start_ep, epochs + 1):
        model.train()

        # ── Phase 1: multi-horizon supervised loss ──
        perm = torch.randperm(n, device=DEVICE)
        ep_loss_mh, cnt_mh = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, tb, db, yb = X_d[idx], T_d[idx], D_d[idx], Y_d[idx]

            # Augment
            ax, ay = [], []
            for j in range(xb.shape[0]):
                a, b = augment_pair(xb[j], yb[j])
                ax.append(a); ay.append(b)
            xb = add_noise(torch.stack(ax))
            yb = torch.stack(ay)

            # Per-horizon weight
            hw = torch.tensor([HORIZON_WEIGHTS.get(d.item(), 0.3) for d in db],
                              device=DEVICE, dtype=torch.float32)

            opt.zero_grad()
            logits = model(xb, tb, db)
            loss_per = F.cross_entropy(logits, yb, weight=cw, reduction='none')
            # loss_per: (B, H, W) -> mean over spatial, weighted by horizon
            loss = (loss_per.mean(dim=(1, 2)) * hw).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss_mh += loss.item(); cnt_mh += 1

        # ── Phase 2: unrolled training with scheduled sampling ──
        ep_loss_ur = 0.0
        cnt_ur = 0
        do_unroll = ep >= UNROLL_MIN_EP and len(segments) > 0

        if do_unroll:
            # Scheduled sampling ratio
            progress = min(1.0, (ep - UNROLL_MIN_EP) / max(SS_RAMP_EP, 1))
            ss_frac = SS_START + (SS_END - SS_START) * progress

            random.shuffle(segments)
            n_seg = min(len(segments), n // bs)  # don't do more unroll than MH
            for si in range(0, n_seg, bs):
                batch_segs = segments[si:si + bs]
                actual_bs = len(batch_segs)

                # Load first frame
                frames_oh = []  # list of (B, C, H, W) tensors
                frames_t = []   # list of (B,) timestep tensors
                frames_y = []   # list of (B, H, W) class index tensors
                for step_j in range(UNROLL_H + 1):
                    oh_list = [torch.tensor(seg[step_j][0], device=DEVICE) for seg in batch_segs]
                    t_list = [seg[step_j][1] for seg in batch_segs]
                    ci_list = [torch.tensor(seg[step_j][2], device=DEVICE) for seg in batch_segs]
                    frames_oh.append(torch.stack(oh_list))
                    frames_t.append(torch.tensor(t_list, dtype=torch.long, device=DEVICE))
                    frames_y.append(torch.stack(ci_list))

                # Apply consistent augmentation (same rotation/flip for whole sequence)
                k_rot = random.randint(0, 3)
                do_flip = random.random() > 0.5
                for j in range(UNROLL_H + 1):
                    frames_oh[j] = torch.rot90(frames_oh[j], k_rot, [2, 3])
                    frames_y[j] = torch.rot90(frames_y[j], k_rot, [1, 2])
                    if do_flip:
                        frames_oh[j] = torch.flip(frames_oh[j], [3])
                        frames_y[j] = torch.flip(frames_y[j], [2])

                opt.zero_grad()
                total_loss = torch.tensor(0.0, device=DEVICE)
                cur_input = frames_oh[0]

                for step_j in range(UNROLL_H):
                    t_cur = frames_t[step_j]
                    delta_1 = torch.ones(actual_bs, dtype=torch.long, device=DEVICE)
                    target = frames_y[step_j + 1]

                    logits = model(cur_input, t_cur, delta_1)
                    step_loss = F.cross_entropy(logits, target, weight=cw)

                    # Decay weight for further steps
                    step_w = max(0.3, 1.0 - 0.15 * step_j)
                    total_loss = total_loss + step_w * step_loss

                    # Prepare next input: scheduled sampling
                    with torch.no_grad():
                        probs = F.softmax(logits.detach(), dim=1)
                        pred_idx = torch.multinomial(
                            probs.permute(0, 2, 3, 1).reshape(-1, NUM_TERRAIN), 1
                        ).squeeze(1).reshape(actual_bs, MAP_H, MAP_W)
                        pred_oh = torch.zeros_like(cur_input)
                        pred_oh.scatter_(1, pred_idx.unsqueeze(1), 1.0)

                    if random.random() < ss_frac:
                        cur_input = pred_oh  # use model prediction
                    else:
                        cur_input = frames_oh[step_j + 1]  # teacher forcing

                total_loss = total_loss / UNROLL_H
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss_ur += total_loss.item(); cnt_ur += 1

        sched.step()

        tl_mh = ep_loss_mh / max(cnt_mh, 1)
        tl_ur = ep_loss_ur / max(cnt_ur, 1) if cnt_ur > 0 else 0.0
        tl_total = tl_mh + tl_ur

        if tl_total < best_loss:
            best_loss = tl_total; mk = " *"
        else:
            mk = ""

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            ur_str = f" | unroll={tl_ur:.4f}" if do_unroll else ""
            ss_str = f" ss={ss_frac:.2f}" if do_unroll else ""
            print(f"  Ep {ep:4d}/{epochs} | mh_loss={tl_mh:.4f}{ur_str}{mk} | "
                  f"skip={model.skip_gain.item():.2f}{ss_str} | "
                  f"lr={opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s")

        if ep % CKPT_EVERY == 0:
            meta["best_loss"] = best_loss
            p = save_ckpt(model, opt, ep, tl_total, meta, CKPT_DIR)
            print(f"    -> {os.path.basename(p)}")

    # Save final
    meta["total_epochs"] = ep; meta["best_loss"] = best_loss
    fp = str(CKPT_DIR / "mhdyn_latest.pt")
    torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": tl_total, "metadata": meta}, fp)
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Saved: {fp}")


# ─── MC ROLLOUT (mixed-stride) ───
def rollout_trajectories(model, initial_grid, h, w, K=DEFAULT_K,
                         T=DEFAULT_T, temperature=1.0):
    """
    Roll out K trajectories using mixed strides to reduce error accumulation.
    Strategy: alternate between Δ=10 jumps and Δ=1 refinements.
    Primary: 5 jumps of Δ=10 (covers 50 steps).
    """
    model.eval()
    s0 = grid_to_class_indices(initial_grid, h, w)

    # Schedule: mostly stride-10 jumps, with occasional stride-1 refinement
    schedule = []
    t_cur = 0
    while t_cur < T:
        remaining = T - t_cur
        if remaining >= 10:
            schedule.append((t_cur, 10))
            t_cur += 10
        elif remaining >= 5:
            schedule.append((t_cur, 5))
            t_cur += 5
        elif remaining >= 2:
            schedule.append((t_cur, 2))
            t_cur += 2
        else:
            schedule.append((t_cur, 1))
            t_cur += 1

    n_steps = len(schedule)
    traj = np.zeros((K, n_steps + 1, h, w), dtype=np.int64)
    traj[:, 0] = s0

    chunk = min(K, 64)
    with torch.no_grad():
        for cs in range(0, K, chunk):
            ce = min(cs + chunk, K)
            B = ce - cs
            cur = torch.tensor(s0, device=DEVICE).unsqueeze(0).expand(B, -1, -1).clone()

            for si, (t_start, delta) in enumerate(schedule):
                oh = torch.zeros(B, NUM_TERRAIN, h, w, device=DEVICE)
                oh.scatter_(1, cur.unsqueeze(1), 1.0)

                t_t = torch.full((B,), t_start, dtype=torch.long, device=DEVICE)
                d_t = torch.full((B,), delta, dtype=torch.long, device=DEVICE)

                logits = model(oh, t_t, d_t)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                flat = probs.permute(0, 2, 3, 1).reshape(-1, NUM_TERRAIN)
                cur = torch.multinomial(flat, 1).squeeze(1).reshape(B, h, w)
                traj[cs:ce, si + 1] = cur.cpu().numpy()

    return traj


# ─── OBSERVATION SCORING & AGGREGATION ───
_STATIC = {CODE_TO_IDX[10], CODE_TO_IDX[11], CODE_TO_IDX[5]}


def aggregate_obs(observations, h, w):
    counts = np.zeros((h, w, NUM_TERRAIN), dtype=np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        g = obs["grid"]
        for dy in range(len(g)):
            for dx in range(len(g[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < h and 0 <= gx < w:
                    counts[gy, gx, CODE_TO_IDX.get(g[dy][dx], 0)] += 1.0
                    total[gy, gx] += 1.0
    probs = np.zeros_like(counts)
    m = total > 0
    for c in range(NUM_TERRAIN):
        probs[:, :, c][m] = counts[:, :, c][m] / total[m]
    return probs, total


def score_rollouts(traj, obs_probs, obs_total, init_idx, eps=1e-6):
    K = traj.shape[0]
    observed = obs_total > 0
    dynamic = np.ones_like(observed)
    for s in _STATIC:
        dynamic &= (init_idx != s)
    mask = observed & dynamic
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(K, dtype=np.float64)
    final = traj[:, -1]
    fc = final[:, ys, xs]
    sp = obs_probs[ys, xs, :]
    lw = np.zeros(K, dtype=np.float64)
    for k in range(K):
        lw[k] = np.log(sp[np.arange(len(ys)), fc[k]] + eps).sum()
    return lw / len(ys)


def normalize_weights(lw, beta=1.0):
    lw = beta * lw
    lw -= lw.max()
    w = np.exp(lw)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones_like(w) / len(w)


def aggregate_preds(traj, weights, h, w):
    mapping = np.array(IDX8_TO_SUBMIT6, dtype=np.int64)
    final = mapping[traj[:, -1]]
    pred = np.zeros((h, w, SUBMIT_CLASSES), dtype=np.float64)
    for k in range(len(weights)):
        for c in range(SUBMIT_CLASSES):
            pred[:, :, c] += weights[k] * (final[k] == c)
    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    return pred / pred.sum(axis=-1, keepdims=True)


def blend_obs(pred, observations, h, w, kappa=3.0):
    if not observations:
        return pred, 0
    oc = np.zeros((h, w, SUBMIT_CLASSES), dtype=np.float64)
    ot = np.zeros((h, w), dtype=np.float64)
    for obs in observations:
        vp = obs["viewport"]
        vx, vy = vp["x"], vp["y"]
        g = obs["grid"]
        for dy in range(len(g)):
            for dx in range(len(g[0])):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < h and 0 <= gx < w:
                    oc[gy, gx, TERRAIN_TO_SUBMIT.get(g[dy][dx], 0)] += 1.0
                    ot[gy, gx] += 1.0
    m = ot > 0
    n = int(m.sum())
    if n == 0:
        return pred, 0
    emp = np.zeros_like(oc)
    for c in range(SUBMIT_CLASSES):
        emp[:, :, c][m] = oc[:, :, c][m] / ot[m]
    alpha = np.zeros((h, w, 1), dtype=np.float64)
    alpha[m, 0] = ot[m] / (ot[m] + kappa)
    bl = alpha * emp + (1 - alpha) * pred.astype(np.float64)
    bl = np.maximum(bl, PROB_FLOOR).astype(np.float32)
    return bl / bl.sum(axis=-1, keepdims=True), n


# ─── METRICS ───
def entropy_px(t):
    t = np.clip(t, 1e-8, None)
    return -(t * np.log(t)).sum(axis=-1)


def kl_px(p, t):
    p, t = np.clip(p, 1e-8, None), np.clip(t, 1e-8, None)
    return (t * (np.log(t) - np.log(p))).sum(axis=-1)


def weighted_kl(p, t):
    kl, ent = kl_px(p, t), entropy_px(t)
    s = ent.sum()
    return (kl * ent).sum() / s if s > 1e-12 else 0.0


def comp_score(p, t):
    wkl = weighted_kl(p, t)
    return 100.0 * np.exp(-3.0 * wkl), wkl


# ─── EVALUATION ───
def evaluate(replays, gt_data, K=DEFAULT_K, T=DEFAULT_T, use_obs=True):
    """
    Train on all data, evaluate on all ground truth (no k-fold).
    This tests the model's ability to fit the training distribution.
    """
    print(f"\n{'='*60}")
    print(f"  Better MC Evaluation (train-all, eval-all)")
    print(f"  K={K}, T={T}, obs={'ON' if use_obs else 'OFF'}")
    print(f"{'='*60}")

    # Load or train model
    model = MultiHorizonDynamics().to(DEVICE)
    cp = latest_ckpt(CKPT_DIR)
    if cp:
        print(f"  Loading checkpoint: {os.path.basename(cp)}")
        load_ckpt(cp, model)
    else:
        print("  No checkpoint found, training first...")
        train(replays, epochs=EPOCHS)
        cp = latest_ckpt(CKPT_DIR)
        if cp:
            load_ckpt(cp, model)
        else:
            print("ERROR: training failed"); return

    model.eval()

    # Group GT by round
    gb = {}
    for g in gt_data:
        rid = g.get("_round_id", "")[:8]
        gb.setdefault(rid, []).append(g)

    scores = []
    wkls = []

    for rid in sorted(gb):
        obs_all = load_observations(rid) if use_obs else []

        for gs in gb[rid]:
            ig, gt = gs.get("initial_grid"), gs.get("ground_truth")
            if ig is None or gt is None:
                continue
            seed = gs.get("_seed_index", 0)
            h, w = gs["height"], gs["width"]
            gtarr = np.array(gt, dtype=np.float32)

            so = [o for o in obs_all if o.get("seed_index") == seed] if use_obs else []

            print(f"\n  Round {rid} seed {seed}:")

            # MC rollout
            t0 = time.time()
            traj = rollout_trajectories(model, ig, h, w, K, T, temperature=1.0)
            t_roll = time.time() - t0
            print(f"    Rollout: {K} trajectories in {t_roll:.1f}s")

            # Weight by observations
            if so:
                op, ot = aggregate_obs(so, h, w)
                ii = grid_to_class_indices(ig, h, w)
                lw = score_rollouts(traj, op, ot, ii)
                wt = normalize_weights(lw, beta=1.0)
                ess = 1.0 / (wt ** 2).sum()
                print(f"    ESS: {ess:.1f}/{K}")
            else:
                wt = np.ones(K, dtype=np.float64) / K

            pred = aggregate_preds(traj, wt, h, w)

            # Blend observations
            if so:
                pred, n_blend = blend_obs(pred, so, h, w, kappa=3.0)
                print(f"    Blended {n_blend} observed cells")

            pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
            pred = pred / pred.sum(axis=-1, keepdims=True)

            s, wkl = comp_score(pred, gtarr)
            scores.append(s)
            wkls.append(wkl)
            print(f"    Score: {s:.2f}  wKL: {wkl:.6f}")

    print(f"\n{'='*60}")
    if scores:
        print(f"  Results ({len(scores)} samples):")
        print(f"    Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f} "
              f"[{np.min(scores):.2f}, {np.max(scores):.2f}]")
        print(f"    wKL:   {np.mean(wkls):.6f} +/- {np.std(wkls):.6f}")
    else:
        print("  No samples evaluated")
    print(f"{'='*60}")


# ─── CLI ───
def main():
    ap = argparse.ArgumentParser(description="Better Monte Carlo: Multi-Horizon Dynamics")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--reset", action="store_true")

    p = sub.add_parser("evaluate")
    p.add_argument("-K", "--rollouts", type=int, default=DEFAULT_K)
    p.add_argument("-T", "--steps", type=int, default=DEFAULT_T)
    p.add_argument("--no-obs", action="store_true")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); return

    if args.cmd == "train":
        r = load_all_replays()
        if r:
            train(r, epochs=args.epochs, reset=args.reset)
    elif args.cmd == "evaluate":
        r = load_all_replays()
        g = load_ground_truth()
        if r and g:
            evaluate(r, g, K=args.rollouts, T=args.steps,
                     use_obs=not args.no_obs)

    print("\nDone!")


if __name__ == "__main__":
    main()
