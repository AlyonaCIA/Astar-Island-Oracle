"""
Astar Island - Monte Carlo v2: Lean Dilated-CNN Dynamics

Key design decisions:
  - Dilated CNN instead of Transformer: 4 dilated conv layers give ~31x31
    receptive field at a fraction of the cost of self-attention on 1600 tokens.
  - Stride-10 only: Train on (frame_t, frame_{t+10}) pairs (~2,050 pairs).
    Every example shows meaningful change. Inference = 5 autoregressive steps.
  - Identity skip connection: Most cells don't change. Skip bias means
    the model only needs to learn the DELTA.
  - FiLM conditioning: Timestep information modulates conv features via
    scale+shift, telling the model WHERE in the simulation timeline we are.
  - Input noise: Scheduled noise injection for robustness to autoregressive
    error accumulation during rollout.
  - DirectPredictor: Separate model trained on 50 GT distributions. Provides
    baseline prediction; ensembled with MC rollouts.

Data budget:
  - Dynamics: ~2,050 stride-10 pairs from 50 replays. With 8x augmentation = ~16K.
  - Direct: 50 GT samples (10 rounds x 5 seeds). With 8x augmentation = 400.

Usage:
  python monte_carlo_2.py train-dynamics [--epochs 300] [--reset]
  python monte_carlo_2.py train-direct [--epochs 500] [--reset]
  python monte_carlo_2.py evaluate [--rollouts 256] [--stride 10]
  python monte_carlo_2.py evaluate-direct
"""

import os, sys, json, time, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.join(SCRIPT_DIR, "simulation_replays")
GT_DIR     = os.path.join(SCRIPT_DIR, "data", "ground_truth")
OBS_DIR    = os.path.join(SCRIPT_DIR, "data")
CKPT_DIR_DYN    = os.path.join(SCRIPT_DIR, "checkpoints_mc2_dyn")
CKPT_DIR_DIRECT = os.path.join(SCRIPT_DIR, "checkpoints_mc2_direct")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Training hyperparameters
TRAIN_STRIDE   = 10           # only stride used for dynamics training
DYN_EPOCHS     = 300
DYN_LR         = 3e-4
DYN_BATCH      = 32
DYN_HIDDEN     = 64
DIRECT_EPOCHS  = 500
DIRECT_LR      = 1e-3
DIRECT_BATCH   = 8
CKPT_EVERY     = 25

# MC inference
DEFAULT_K      = 256
DEFAULT_T      = 50
INFER_STRIDE   = 10           # 5 autoregressive steps

# Noise schedule
NOISE_PROB = 0.3
NOISE_MAG  = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_replays():
    replays = []
    if not os.path.isdir(REPLAY_DIR):
        print(f"ERROR: {REPLAY_DIR} not found"); return replays
    for f in sorted(os.listdir(REPLAY_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(REPLAY_DIR, f)) as fh:
                replays.append(json.load(fh))
    print(f"Loaded {len(replays)} replays")
    return replays


def load_ground_truth():
    if not os.path.isdir(GT_DIR):
        print(f"ERROR: {GT_DIR} not found"); return []
    data = []
    for f in sorted(os.listdir(GT_DIR)):
        if not f.endswith(".json"): continue
        with open(os.path.join(GT_DIR, f)) as fh:
            d = json.load(fh)
        parts = f.replace(".json", "").split("_")
        if len(parts) >= 3:
            d["_round_number"] = int(parts[0][1:])
            d["_seed_index"] = int(parts[1][1:])
            d["_round_id"] = parts[2]
        data.append(d)
    print(f"Loaded {len(data)} ground truth samples")
    return data


def load_observations(round_id_short):
    path = os.path.join(OBS_DIR, f"observations_{round_id_short}.json")
    if not os.path.isfile(path): return []
    with open(path) as f:
        raw = json.load(f)
    return raw.get("observations", []) if isinstance(raw, dict) else raw


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FiLM Conditioning (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Generate scale+shift for each conv channel from a scalar timestep."""
    def __init__(self, n_channels, max_t=MAX_TIMESTEP + 1):
        super().__init__()
        self.embed = nn.Embedding(max_t, 64)
        self.fc = nn.Linear(64, n_channels * 2)

    def forward(self, t):
        """t: (B,) int -> scale (B,C,1,1), shift (B,C,1,1)"""
        e = F.gelu(self.embed(t))              # (B, 64)
        params = self.fc(e)                     # (B, 2C)
        scale, shift = params.chunk(2, dim=1)   # (B, C) each
        return scale[:, :, None, None], shift[:, :, None, None]


# ---------------------------------------------------------------------------
# Model 1: DilatedDynamics
# ---------------------------------------------------------------------------

class DilatedDynamics(nn.Module):
    """
    Dilated-CNN dynamics model.

    4 dilated conv layers with increasing dilation -> ~31x31 receptive field.
    FiLM conditioning from timestep modulates each layer.
    Identity skip connection biases toward "no change."
    """

    def __init__(self, hidden=DYN_HIDDEN, skip_init=5.0):
        super().__init__()
        self.hidden = hidden

        # Dilated conv stack: dilation 1, 2, 4, 8 -> RF ~31x31
        self.conv1 = nn.Conv2d(NUM_TERRAIN, hidden, 3, padding=1,  dilation=1)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden,       3, padding=2,  dilation=2)
        self.bn2   = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, hidden,       3, padding=4,  dilation=4)
        self.bn3   = nn.BatchNorm2d(hidden)
        self.conv4 = nn.Conv2d(hidden, hidden,       3, padding=8,  dilation=8)
        self.bn4   = nn.BatchNorm2d(hidden)

        # FiLM for each layer
        self.film1 = FiLMGenerator(hidden)
        self.film2 = FiLMGenerator(hidden)
        self.film3 = FiLMGenerator(hidden)
        self.film4 = FiLMGenerator(hidden)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, NUM_TERRAIN, 1),
        )

        # Identity skip: output = delta + gain * input
        self.skip_gain = nn.Parameter(torch.tensor(skip_init))

    def forward(self, x, t=None):
        """
        x: (B, 8, H, W) one-hot terrain
        t: (B,) timestep (0..50). Defaults to 0 if None.
        """
        B = x.shape[0]
        if t is None:
            t = torch.zeros(B, dtype=torch.long, device=x.device)

        h = self.conv1(x)
        h = self.bn1(h)
        s, b = self.film1(t)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv2(h)
        h = self.bn2(h)
        s, b = self.film2(t)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv3(h)
        h = self.bn3(h)
        s, b = self.film3(t)
        h = F.gelu(h * (1 + s) + b)

        h = self.conv4(h)
        h = self.bn4(h)
        s, b = self.film4(t)
        h = F.gelu(h * (1 + s) + b)

        delta = self.head(h)
        logits = delta + self.skip_gain * x
        return logits

    def predict_probs(self, x, t=None):
        return F.softmax(self.forward(x, t), dim=1)


# ---------------------------------------------------------------------------
# Model 2: DirectPredictor
# ---------------------------------------------------------------------------

class DirectPredictor(nn.Module):
    """
    Dilated-CNN that predicts 6-class distribution directly from initial grid.
    No skip connection (output classes differ from input classes).
    """

    def __init__(self, hidden=DYN_HIDDEN):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_TERRAIN, hidden, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.bn2   = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4)
        self.bn3   = nn.BatchNorm2d(hidden)
        self.conv4 = nn.Conv2d(hidden, hidden, 3, padding=8, dilation=8)
        self.bn4   = nn.BatchNorm2d(hidden)
        self.conv5 = nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4)
        self.bn5   = nn.BatchNorm2d(hidden)
        self.conv6 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.bn6   = nn.BatchNorm2d(hidden)

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, SUBMIT_CLASSES, 1),
        )

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        h = F.gelu(self.bn4(self.conv4(h)))
        h = F.gelu(self.bn5(self.conv5(h)))
        h = F.gelu(self.bn6(self.conv6(h)))
        return self.head(h)

    def predict_probs(self, x):
        return F.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Augmentation & noise
# ---------------------------------------------------------------------------

def augment_pair(x, y):
    """Consistent rotation+flip on (C,H,W) input and (H,W) target."""
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    x = torch.rot90(x, k, [1, 2])
    y = torch.rot90(y, k, [0, 1])
    if flip:
        x = torch.flip(x, [2])
        y = torch.flip(y, [1])
    return x, y


def augment_grid(x, y):
    """Consistent rotation+flip on two (C,H,W) tensors."""
    k = random.randint(0, 3)
    flip = random.random() > 0.5
    x = torch.rot90(x, k, [1, 2])
    y = torch.rot90(y, k, [1, 2])
    if flip:
        x = torch.flip(x, [2])
        y = torch.flip(y, [2])
    return x, y


def add_noise(x, prob=NOISE_PROB, mag=NOISE_MAG):
    """Inject noise into one-hot inputs. Simulates autoregressive error."""
    mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < prob).float()
    noise = torch.randn_like(x) * mag
    return F.softmax((x + noise * mask) / 0.1, dim=1)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_stride_dataset(replays, stride=TRAIN_STRIDE, held_out=None):
    """Extract (frame_t, timestep, frame_{t+stride}) triples."""
    Xs, Ts, Ys = [], [], []
    for r in replays:
        rid = r["round_id"][:8]
        if held_out and rid in held_out:
            continue
        h, w, frames = r["height"], r["width"], r["frames"]
        for i in range(len(frames) - stride):
            Xs.append(grid_to_onehot(frames[i]["grid"], h, w))
            Ts.append(frames[i]["step"])
            Ys.append(grid_to_class_indices(frames[i + stride]["grid"], h, w))
    if not Xs:
        return (np.zeros((0, NUM_TERRAIN, MAP_H, MAP_W), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, MAP_H, MAP_W), dtype=np.int64))
    X = np.stack(Xs); T = np.array(Ts, dtype=np.int64); Y = np.stack(Ys)
    n_rounds = len(set(r["round_id"][:8] for r in replays
                       if not (held_out and r["round_id"][:8] in held_out)))
    print(f"  Stride-{stride}: {len(X)} pairs from {n_rounds} rounds")
    return X, T, Y


def build_direct_dataset(gt_data, held_out=None):
    """Initial grid -> ground truth distribution pairs."""
    Xs, Ys = [], []
    for g in gt_data:
        rid = g.get("_round_id", "")[:8]
        if held_out and rid in held_out:
            continue
        ig, gt = g.get("initial_grid"), g.get("ground_truth")
        if ig is None or gt is None:
            continue
        Xs.append(grid_to_onehot(ig, g["height"], g["width"]))
        Ys.append(np.array(gt, dtype=np.float32).transpose(2, 0, 1))
    if not Xs:
        return (np.zeros((0, NUM_TERRAIN, MAP_H, MAP_W), dtype=np.float32),
                np.zeros((0, SUBMIT_CLASSES, MAP_H, MAP_W), dtype=np.float32))
    print(f"  Direct: {len(Xs)} samples")
    return np.stack(Xs), np.stack(Ys)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_class_weights(Y, nc=NUM_TERRAIN, smooth=0.1):
    counts = np.bincount(Y.flatten(), minlength=nc).astype(np.float64)
    counts = np.maximum(counts, 1)
    w = 1.0 / (counts / counts.sum() + smooth)
    w = w / w.sum() * nc
    return torch.tensor(w, dtype=torch.float32)


def kl_loss(logits, target, eps=1e-8):
    """KL(target || pred) for distribution targets."""
    lp = F.log_softmax(logits, dim=1)
    t = target.clamp(min=eps)
    return (t * (t.log() - lp)).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_ckpt(d, pfx):
    os.makedirs(d, exist_ok=True)
    pat = f"{pfx}_epoch_"
    fs = [f for f in os.listdir(d) if f.startswith(pat) and f.endswith(".pt")]
    if not fs:
        return None
    fs.sort(key=lambda f: int(f.replace(pat, "").replace(".pt", "")))
    return os.path.join(d, fs[-1])


def save_ckpt(model, opt, epoch, tl, vl, meta, d, pfx):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{pfx}_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": tl, "val_loss": vl, "metadata": meta}, p)
    return p


def load_ckpt(path, model, opt=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    if opt and "optimizer_state_dict" in ck:
        opt.load_state_dict(ck["optimizer_state_dict"])
    return ck


# ---------------------------------------------------------------------------
# Training: DilatedDynamics
# ---------------------------------------------------------------------------

def train_dynamics(replays, epochs=DYN_EPOCHS, lr=DYN_LR, bs=DYN_BATCH,
                   cv="all", reset=False):
    if cv == "round_kfold":
        rids = sorted(set(r["round_id"][:8] for r in replays))
        print(f"\n{'='*60}")
        print(f"  DilatedDynamics -- {len(rids)}-fold Round CV")
        print(f"  stride={TRAIN_STRIDE}")
        print(f"{'='*60}")
        folds = []
        for i, vr in enumerate(rids):
            print(f"\n  --- Fold {i+1}/{len(rids)}: holdout {vr} ---")
            Xt, Tt, Yt = build_stride_dataset(replays, held_out={vr})
            Xv, Tv, Yv = build_stride_dataset(
                [r for r in replays if r["round_id"][:8] == vr])
            cw = compute_class_weights(Yt).to(DEVICE)
            v = _train_dyn_loop(Xt, Tt, Yt, Xv, Tv, Yv, epochs, lr, bs, cw,
                                reset=True, save=False, tag=f"fold{i+1}")
            folds.append(v)
        print(f"\n  CV mean: {np.mean(folds):.6f} +/- {np.std(folds):.6f}")
        return

    print(f"\n{'='*60}")
    print(f"  DilatedDynamics Training (all data, stride={TRAIN_STRIDE})")
    print(f"{'='*60}")
    X, T, Y = build_stride_dataset(replays)
    cw = compute_class_weights(Y).to(DEVICE)
    _train_dyn_loop(X, T, Y, None, None, None, epochs, lr, bs, cw,
                    reset=reset, save=True, tag="all")


def _train_dyn_loop(Xt, Tt, Yt, Xv, Tv, Yv, epochs, lr, bs, cw,
                    reset=False, save=True, tag=""):
    Xt_d = torch.tensor(Xt, device=DEVICE)
    Tt_d = torch.tensor(Tt, device=DEVICE)
    Yt_d = torch.tensor(Yt, device=DEVICE)
    has_val = Xv is not None and len(Xv) > 0
    if has_val:
        Xv_d = torch.tensor(Xv, device=DEVICE)
        Tv_d = torch.tensor(Tv, device=DEVICE)
        Yv_d = torch.tensor(Yv, device=DEVICE)

    model = DilatedDynamics().to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Params: {npar:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr * 0.01)

    start = 1
    if save and not reset:
        cp = latest_ckpt(CKPT_DIR_DYN, "dyn")
        if cp:
            print(f"  Resuming from {os.path.basename(cp)}")
            start = load_ckpt(cp, model, opt)["epoch"] + 1

    loader = DataLoader(TensorDataset(Xt_d, Tt_d, Yt_d), bs, shuffle=True)
    nb = len(loader)
    best_val, no_imp = float("inf"), 0
    t0 = time.time()
    meta = {"tag": tag, "n_train": len(Xt), "model": "DilatedDynamics",
            "stride": TRAIN_STRIDE}

    for ep in range(start, epochs + 1):
        model.train()
        el, cnt = 0.0, 0
        for xb, tb, yb in loader:
            ax, ay = [], []
            for i in range(xb.shape[0]):
                a, b = augment_pair(xb[i], yb[i])
                ax.append(a); ay.append(b)
            xb = add_noise(torch.stack(ax))
            yb = torch.stack(ay)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb, tb), yb, weight=cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item(); cnt += 1
        sched.step()
        tl = el / max(cnt, 1)

        vl = tl
        if has_val:
            model.eval()
            with torch.no_grad():
                vs = []
                for i in range(0, len(Xv_d), bs):
                    vs.append(F.cross_entropy(
                        model(Xv_d[i:i+bs], Tv_d[i:i+bs]),
                        Yv_d[i:i+bs], weight=cw).item())
                vl = np.mean(vs)

        if vl < best_val:
            best_val = vl; no_imp = 0; mk = " *"
        else:
            no_imp += 1; mk = ""

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  Ep {ep:4d}/{epochs} | train={tl:.6f} | "
                  f"val={vl:.6f}{mk} | skip={model.skip_gain.item():.2f} | "
                  f"lr={opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s")

        if save and ep % CKPT_EVERY == 0:
            meta["best_val"] = best_val
            p = save_ckpt(model, opt, ep, tl, vl, meta, CKPT_DIR_DYN, "dyn")
            print(f"    -> {os.path.basename(p)}")

        if has_val and no_imp >= 150:
            print(f"  Early stop at epoch {ep}"); break

    if save:
        meta["total_epochs"] = ep; meta["best_val"] = best_val
        fp = os.path.join(CKPT_DIR_DYN, "dyn_latest.pt")
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": tl, "val_loss": vl, "metadata": meta}, fp)
    print(f"  Best val CE: {best_val:.6f}")
    return best_val


# ---------------------------------------------------------------------------
# Training: DirectPredictor
# ---------------------------------------------------------------------------

def train_direct(gt_data, epochs=DIRECT_EPOCHS, lr=DIRECT_LR, bs=DIRECT_BATCH,
                 cv="all", reset=False):
    if cv == "round_kfold":
        rids = sorted(set(g.get("_round_id", "")[:8] for g in gt_data
                          if g.get("_round_id")))
        print(f"\n{'='*60}")
        print(f"  DirectPredictor -- {len(rids)}-fold Round CV")
        print(f"{'='*60}")
        folds = []
        for i, vr in enumerate(rids):
            print(f"\n  --- Fold {i+1}/{len(rids)}: holdout {vr} ---")
            Xt, Yt = build_direct_dataset(gt_data, held_out={vr})
            Xv, Yv = build_direct_dataset(
                [g for g in gt_data if g.get("_round_id", "")[:8] == vr])
            if len(Xt) == 0 or len(Xv) == 0:
                print("    Skip (no data)"); continue
            v = _train_direct_loop(Xt, Yt, Xv, Yv, epochs, lr, bs,
                                   reset=True, save=False, tag=f"fold{i+1}")
            folds.append(v)
        if folds:
            print(f"\n  CV mean KL: {np.mean(folds):.6f} +/- {np.std(folds):.6f}")
        return

    print(f"\n{'='*60}")
    print(f"  DirectPredictor Training (all data)")
    print(f"{'='*60}")
    X, Y = build_direct_dataset(gt_data)
    if len(X) == 0:
        print("ERROR: no data"); return
    _train_direct_loop(X, Y, None, None, epochs, lr, bs,
                       reset=reset, save=True, tag="all")


def _train_direct_loop(Xt, Yt, Xv, Yv, epochs, lr, bs,
                       reset=False, save=True, tag=""):
    Xt_d = torch.tensor(Xt, device=DEVICE)
    Yt_d = torch.tensor(Yt, device=DEVICE)
    has_val = Xv is not None and len(Xv) > 0
    if has_val:
        Xv_d = torch.tensor(Xv, device=DEVICE)
        Yv_d = torch.tensor(Yv, device=DEVICE)

    model = DirectPredictor().to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Params: {npar:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, lr * 0.01)

    start = 1
    if save and not reset:
        cp = latest_ckpt(CKPT_DIR_DIRECT, "direct")
        if cp:
            print(f"  Resuming from {os.path.basename(cp)}")
            start = load_ckpt(cp, model, opt)["epoch"] + 1

    loader = DataLoader(TensorDataset(Xt_d, Yt_d), bs, shuffle=True)
    best_val, no_imp = float("inf"), 0
    t0 = time.time()
    meta = {"tag": tag, "n_train": len(Xt), "model": "DirectPredictor"}

    for ep in range(start, epochs + 1):
        model.train()
        el, cnt = 0.0, 0
        for xb, yb in loader:
            ax, ay = [], []
            for i in range(xb.shape[0]):
                a, b = augment_grid(xb[i], yb[i])
                ax.append(a); ay.append(b)
            xb, yb = torch.stack(ax), torch.stack(ay)
            opt.zero_grad()
            loss = kl_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item(); cnt += 1
        sched.step()
        tl = el / max(cnt, 1)

        vl = tl
        if has_val:
            model.eval()
            with torch.no_grad():
                vs = []
                for i in range(0, len(Xv_d), bs):
                    vs.append(kl_loss(
                        model(Xv_d[i:i+bs]), Yv_d[i:i+bs]).item())
                vl = np.mean(vs)

        if vl < best_val:
            best_val = vl; no_imp = 0; mk = " *"
        else:
            no_imp += 1; mk = ""

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  Ep {ep:4d}/{epochs} | train={tl:.6f} | "
                  f"val={vl:.6f}{mk} | lr={opt.param_groups[0]['lr']:.1e} | "
                  f"{time.time()-t0:.0f}s")

        if save and ep % CKPT_EVERY == 0:
            meta["best_val"] = best_val
            p = save_ckpt(model, opt, ep, tl, vl, meta, CKPT_DIR_DIRECT, "direct")
            print(f"    -> {os.path.basename(p)}")

        if has_val and no_imp >= 200:
            print(f"  Early stop at epoch {ep}"); break

    if save:
        meta["total_epochs"] = ep; meta["best_val"] = best_val
        fp = os.path.join(CKPT_DIR_DIRECT, "direct_latest.pt")
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": tl, "val_loss": vl, "metadata": meta}, fp)
    print(f"  Best val KL: {best_val:.6f}")
    return best_val


# ---------------------------------------------------------------------------
# MC Rollout Engine
# ---------------------------------------------------------------------------

def rollout_trajectories(model, initial_grid, h, w, K=DEFAULT_K,
                         T=DEFAULT_T, stride=INFER_STRIDE, temperature=1.0):
    """
    K stochastic rollouts using stride-N steps.
    stride=10, T=50 -> 5 autoregressive steps.
    Returns (K, n_checkpoints+1, H, W) int64.
    """
    model.eval()
    s0 = grid_to_class_indices(initial_grid, h, w)

    times = list(range(0, T, stride))
    n_jumps = len(times)
    traj = np.zeros((K, n_jumps + 1, h, w), dtype=np.int64)
    traj[:, 0] = s0

    chunk = min(K, 64)
    with torch.no_grad():
        for cs in range(0, K, chunk):
            ce = min(cs + chunk, K)
            B = ce - cs
            cur = torch.tensor(s0, device=DEVICE).unsqueeze(0).expand(B, -1, -1).clone()

            for si, t_start in enumerate(times):
                oh = torch.zeros(B, NUM_TERRAIN, h, w, device=DEVICE)
                oh.scatter_(1, cur.unsqueeze(1), 1.0)
                t_t = torch.full((B,), t_start, dtype=torch.long, device=DEVICE)

                logits = model(oh, t_t)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                flat = probs.permute(0, 2, 3, 1).reshape(-1, NUM_TERRAIN)
                cur = torch.multinomial(flat, 1).squeeze(1).reshape(B, h, w)
                traj[cs:ce, si + 1] = cur.cpu().numpy()

    return traj


# ---------------------------------------------------------------------------
# Observation scoring & aggregation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def predict_direct(model, grid, h, w):
    model.eval()
    x = torch.tensor(grid_to_onehot(grid, h, w), device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        p = model.predict_probs(x)
    p = p[0].cpu().numpy().transpose(1, 2, 0)
    p = np.maximum(p, PROB_FLOOR).astype(np.float32)
    return p / p.sum(axis=-1, keepdims=True)


def predict_round(dyn, direct, grid, h, w, obs, K=DEFAULT_K, T=DEFAULT_T,
                  stride=INFER_STRIDE, temp=1.0, beta=1.0, kappa=3.0,
                  ens_w=0.5):
    pred_d = predict_direct(direct, grid, h, w) if direct else None

    pred_mc = None
    if dyn is not None:
        n_steps = T // stride
        print(f"    MC: {K} rollouts x {n_steps} steps (stride={stride})")
        t0 = time.time()
        traj = rollout_trajectories(dyn, grid, h, w, K, T, stride, temp)
        print(f"    Rollouts: {time.time()-t0:.1f}s")

        if obs:
            op, ot = aggregate_obs(obs, h, w)
            ii = grid_to_class_indices(grid, h, w)
            lw = score_rollouts(traj, op, ot, ii)
            wt = normalize_weights(lw, beta)
            ess = 1.0 / (wt ** 2).sum()
            print(f"    ESS: {ess:.1f}/{K}")
        else:
            wt = np.ones(K, dtype=np.float64) / K
        pred_mc = aggregate_preds(traj, wt, h, w)

    if pred_d is not None and pred_mc is not None:
        pred = (ens_w * pred_d + (1 - ens_w) * pred_mc).astype(np.float32)
    elif pred_d is not None:
        pred = pred_d
    elif pred_mc is not None:
        pred = pred_mc
    else:
        raise RuntimeError("No model")

    if obs:
        pred, n = blend_obs(pred, obs, h, w, kappa)
        print(f"    Blended {n} observed cells")

    pred = np.maximum(pred, PROB_FLOOR).astype(np.float32)
    return pred / pred.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full evaluation -- round k-fold
# ---------------------------------------------------------------------------

def evaluate_all(replays, gt_data, K=DEFAULT_K, T=DEFAULT_T,
                 stride=INFER_STRIDE, dyn_ep=DYN_EPOCHS, dir_ep=DIRECT_EPOCHS,
                 use_obs=True):
    rb = {}
    for r in replays:
        rb.setdefault(r["round_id"][:8], []).append(r)
    gb = {}
    for g in gt_data:
        rid = g.get("_round_id", "")[:8]
        gb.setdefault(rid, []).append(g)

    common = sorted(set(rb) & set(gb))
    if len(common) < 2:
        print("ERROR: need >= 2 rounds with replays + GT"); return

    print(f"\n{'='*60}")
    print(f"  MC v2 Eval -- {len(common)}-fold CV")
    print(f"  K={K}, stride={stride}, obs={'ON' if use_obs else 'OFF'}")
    print(f"{'='*60}")

    res = {"direct": [], "mc": [], "ensemble": []}

    for fi, vr in enumerate(common):
        print(f"\n  === Fold {fi+1}/{len(common)}: holdout {vr} ===")

        # Train DirectPredictor on non-holdout
        print(f"  Training DirectPredictor...")
        Xdt, Ydt = build_direct_dataset(gt_data, held_out={vr})
        dm = None
        if len(Xdt) > 0:
            dm = DirectPredictor().to(DEVICE)
            od = torch.optim.AdamW(dm.parameters(), lr=DIRECT_LR, weight_decay=1e-4)
            dl = DataLoader(TensorDataset(
                torch.tensor(Xdt, device=DEVICE),
                torch.tensor(Ydt, device=DEVICE)), DIRECT_BATCH, shuffle=True)
            t0 = time.time()
            for ep in range(1, dir_ep + 1):
                dm.train()
                el, nb = 0.0, 0
                for xb, yb in dl:
                    ax, ay = [], []
                    for i in range(xb.shape[0]):
                        a, b = augment_grid(xb[i], yb[i])
                        ax.append(a); ay.append(b)
                    xb, yb = torch.stack(ax), torch.stack(ay)
                    od.zero_grad()
                    loss = kl_loss(dm(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
                    od.step()
                    el += loss.item(); nb += 1
                if ep % 100 == 0 or ep == 1:
                    print(f"    Direct ep {ep}/{dir_ep} loss={el/max(nb,1):.6f} "
                          f"({time.time()-t0:.0f}s)")

        # Train DilatedDynamics on non-holdout
        print(f"  Training DilatedDynamics...")
        tr = [r for r in replays if r["round_id"][:8] != vr]
        Xm, Tm, Ym = build_stride_dataset(tr)
        cw = compute_class_weights(Ym).to(DEVICE)
        dyn = DilatedDynamics().to(DEVICE)
        om = torch.optim.AdamW(dyn.parameters(), lr=DYN_LR, weight_decay=1e-4)
        dlm = DataLoader(TensorDataset(
            torch.tensor(Xm, device=DEVICE),
            torch.tensor(Tm, device=DEVICE),
            torch.tensor(Ym, device=DEVICE)), DYN_BATCH, shuffle=True)
        t0 = time.time()
        for ep in range(1, dyn_ep + 1):
            dyn.train()
            el, nb = 0.0, 0
            for xb, tb, yb in dlm:
                ax, ay = [], []
                for i in range(xb.shape[0]):
                    a, b = augment_pair(xb[i], yb[i])
                    ax.append(a); ay.append(b)
                xb = add_noise(torch.stack(ax))
                yb = torch.stack(ay)
                om.zero_grad()
                loss = F.cross_entropy(dyn(xb, tb), yb, weight=cw)
                loss.backward()
                nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
                om.step()
                el += loss.item(); nb += 1
            if ep % 50 == 0 or ep == 1:
                print(f"    Dyn ep {ep}/{dyn_ep} loss={el/max(nb,1):.6f} "
                      f"skip={dyn.skip_gain.item():.2f} ({time.time()-t0:.0f}s)")

        # Evaluate on holdout
        obs_all = load_observations(vr)
        for gs in gb[vr]:
            ig, gt = gs.get("initial_grid"), gs.get("ground_truth")
            if ig is None or gt is None:
                continue
            seed = gs.get("_seed_index", 0)
            h, w = gs["height"], gs["width"]
            gtarr = np.array(gt, dtype=np.float32)
            so = [o for o in obs_all if o.get("seed_index") == seed] if use_obs else []

            print(f"\n  Round {vr} seed {seed}:")
            if dm:
                pd = predict_direct(dm, ig, h, w)
                sd, wd = comp_score(pd, gtarr)
                res["direct"].append(sd)
                print(f"    Direct:   {sd:.2f}  wKL={wd:.6f}")

            pm = predict_round(dyn, None, ig, h, w, so, K, T, stride, ens_w=0.0)
            sm, wm = comp_score(pm, gtarr)
            res["mc"].append(sm)
            print(f"    MC only:  {sm:.2f}  wKL={wm:.6f}")

            if dm:
                pe = predict_round(dyn, dm, ig, h, w, so, K, T, stride, ens_w=0.5)
                se, we = comp_score(pe, gtarr)
                res["ensemble"].append(se)
                print(f"    Ensemble: {se:.2f}  wKL={we:.6f}")

    print(f"\n{'='*60}")
    print(f"  Results (stride={stride})")
    for m, s in res.items():
        if s:
            print(f"  {m:10s}: {np.mean(s):.2f} +/- {np.std(s):.2f} "
                  f"[{np.min(s):.2f}, {np.max(s):.2f}]")
    print(f"{'='*60}")


def evaluate_direct_only(gt_data, epochs=DIRECT_EPOCHS):
    rids = sorted(set(g.get("_round_id", "")[:8] for g in gt_data
                      if g.get("_round_id")))
    if len(rids) < 2:
        print("ERROR: need >= 2 rounds"); return
    print(f"\n{'='*60}")
    print(f"  DirectPredictor Only -- {len(rids)}-fold CV")
    print(f"{'='*60}")
    scores, wkls = [], []
    for fi, vr in enumerate(rids):
        print(f"\n  --- Fold {fi+1}/{len(rids)}: holdout {vr} ---")
        Xt, Yt = build_direct_dataset(gt_data, held_out={vr})
        Xv, Yv = build_direct_dataset(
            [g for g in gt_data if g.get("_round_id", "")[:8] == vr])
        if len(Xt) == 0 or len(Xv) == 0:
            continue
        m = DirectPredictor().to(DEVICE)
        o = torch.optim.AdamW(m.parameters(), lr=DIRECT_LR, weight_decay=1e-4)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, epochs, DIRECT_LR*0.01)
        dl = DataLoader(TensorDataset(
            torch.tensor(Xt, device=DEVICE),
            torch.tensor(Yt, device=DEVICE)), DIRECT_BATCH, shuffle=True)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            m.train()
            el, nb = 0.0, 0
            for xb, yb in dl:
                ax, ay = [], []
                for i in range(xb.shape[0]):
                    a, b = augment_grid(xb[i], yb[i])
                    ax.append(a); ay.append(b)
                xb, yb = torch.stack(ax), torch.stack(ay)
                o.zero_grad()
                loss = kl_loss(m(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o.step()
                el += loss.item(); nb += 1
            sc.step()
            if ep % 100 == 0 or ep == 1:
                print(f"    Ep {ep}/{epochs} loss={el/max(nb,1):.6f} "
                      f"({time.time()-t0:.0f}s)")
        m.eval()
        for gs in [g for g in gt_data if g.get("_round_id", "")[:8] == vr]:
            ig, gt = gs.get("initial_grid"), gs.get("ground_truth")
            if ig is None or gt is None:
                continue
            p = predict_direct(m, ig, gs["height"], gs["width"])
            s, w = comp_score(p, np.array(gt, dtype=np.float32))
            scores.append(s); wkls.append(w)
            print(f"    {vr} s{gs.get('_seed_index',0)}: "
                  f"score={s:.2f} wKL={w:.6f}")
    if scores:
        print(f"\n  Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f} "
              f"[{np.min(scores):.2f}, {np.max(scores):.2f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MC v2: Lean Dilated-CNN Dynamics")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("train-dynamics")
    p.add_argument("--cv", choices=["all", "round_kfold"], default="all")
    p.add_argument("--epochs", type=int, default=DYN_EPOCHS)
    p.add_argument("--reset", action="store_true")

    p = sub.add_parser("train-direct")
    p.add_argument("--cv", choices=["all", "round_kfold"], default="all")
    p.add_argument("--epochs", type=int, default=DIRECT_EPOCHS)
    p.add_argument("--reset", action="store_true")

    p = sub.add_parser("evaluate")
    p.add_argument("-K", "--rollouts", type=int, default=DEFAULT_K)
    p.add_argument("-T", "--steps", type=int, default=DEFAULT_T)
    p.add_argument("--stride", type=int, default=INFER_STRIDE)
    p.add_argument("--dyn-epochs", type=int, default=DYN_EPOCHS)
    p.add_argument("--direct-epochs", type=int, default=DIRECT_EPOCHS)
    p.add_argument("--no-obs", action="store_true")

    p = sub.add_parser("evaluate-direct")
    p.add_argument("--epochs", type=int, default=DIRECT_EPOCHS)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); return
    print(f"Device: {DEVICE}")

    if args.cmd == "train-dynamics":
        r = load_all_replays()
        if r:
            train_dynamics(r, args.epochs, reset=args.reset, cv=args.cv)
    elif args.cmd == "train-direct":
        g = load_ground_truth()
        if g:
            train_direct(g, args.epochs, reset=args.reset, cv=args.cv)
    elif args.cmd == "evaluate":
        r, g = load_all_replays(), load_ground_truth()
        if r and g:
            evaluate_all(r, g, args.rollouts, args.steps, args.stride,
                         args.dyn_epochs, args.direct_epochs,
                         not args.no_obs)
    elif args.cmd == "evaluate-direct":
        g = load_ground_truth()
        if g:
            evaluate_direct_only(g, args.epochs)


if __name__ == "__main__":
    main()
