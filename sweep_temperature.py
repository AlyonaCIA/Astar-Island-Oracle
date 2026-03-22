"""
Sweep softmax temperature on round 22 and report scores per seed + average.

Usage:
    python sweep_temperature.py
    python sweep_temperature.py --round 21
"""

import os, sys, json, re, glob, argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_cnn import (
    encode_initial_grid, encode_obs_channels,
    load_model_from_checkpoint, get_checkpoint_dir, PROB_FLOOR,
)
from eval_cnn import load_observations_list

GT_DIR = os.path.join(SCRIPT_DIR, "data", "ground_truth")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_round_samples(round_num):
    samples = []
    for fname in sorted(os.listdir(GT_DIR)):
        m = re.match(r"r(\d+)_s(\d+)_([a-f0-9]+)\.json", fname)
        if not m or int(m.group(1)) != round_num:
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        data["_round_number"] = int(m.group(1))
        data["_seed_index"] = int(m.group(2))
        data["_round_id"] = m.group(3)
        if data.get("initial_grid") and data.get("ground_truth"):
            samples.append(data)
    return samples


def compute_score(pred, gt):
    pred_c = np.clip(pred, 1e-8, None)
    gt_c = np.clip(gt, 1e-8, None)
    kl = (gt_c * (np.log(gt_c) - np.log(pred_c))).sum(axis=-1)
    ent = -(gt_c * np.log(gt_c)).sum(axis=-1)
    total_ent = ent.sum()
    if total_ent < 1e-12:
        return 100.0, 0.0
    wkl = (kl * ent).sum() / total_ent
    return 100.0 * np.exp(-3.0 * wkl), wkl


def predict(model, gt_sample, temperature):
    H, W = gt_sample["height"], gt_sample["width"]
    initial_grid = gt_sample["initial_grid"]
    rid = gt_sample.get("_round_id", "")
    seed = gt_sample.get("_seed_index", 0)

    features = encode_initial_grid(initial_grid, W, H)
    seed_obs = []
    if rid:
        all_obs = load_observations_list(rid)
        if all_obs:
            seed_obs = [o for o in all_obs if o.get("seed_index") == seed]

    obs_feat = encode_obs_channels(seed_obs, W, H)
    x = np.concatenate([features, obs_feat], axis=0)
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = model(x).squeeze(0)  # (6, H, W)
        if temperature != 1.0:
            logits = torch.log(torch.clamp(probs, min=1e-8))
            logits = logits / temperature
            probs = torch.softmax(logits, dim=0)
        pred = probs.permute(1, 2, 0).cpu().numpy()

    pred = np.maximum(pred, PROB_FLOOR)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    for y in range(H):
        for xi in range(W):
            if initial_grid[y][xi] == 5:
                pred[y, xi, :] = PROB_FLOOR
                pred[y, xi, 5] = 1.0 - (PROB_FLOOR * 5)

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=22)
    args = parser.parse_args()

    samples = load_round_samples(args.round)
    if not samples:
        print(f"No GT data for round {args.round}. Run fetch_ground_truth.py first.")
        sys.exit(1)
    print(f"Round {args.round}: {len(samples)} seeds\n")

    # Load model once
    ckpt_dir = get_checkpoint_dir("unet_cond")
    ckpt_path = os.path.join(ckpt_dir, "cnn_latest.pt")
    if not os.path.isfile(ckpt_path):
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "cnn_epoch_*.pt")))
        if not ckpts:
            print("No checkpoint found"); sys.exit(1)
        ckpt_path = ckpts[-1]
    model, _ = load_model_from_checkpoint(ckpt_path)
    model.eval()
    print(f"Model: {os.path.basename(ckpt_path)}\n")

    temperatures = [round(t * 0.01, 2) for t in range(5, 101)]  # 0.05 to 1.00 step 0.01

    # header
    seed_ids = [s["_seed_index"] for s in samples]
    header = f"{'Temp':>6}" + "".join(f"  S{s} score" for s in seed_ids) + "     AVG    AVG wKL"
    print(header)
    print("-" * len(header))

    best_avg = -1
    best_temp = None

    for temp in temperatures:
        scores, wkls = [], []
        for sample in samples:
            gt = np.array(sample["ground_truth"], dtype=np.float32)
            pred = predict(model, sample, temp)
            score, wkl = compute_score(pred, gt)
            scores.append(score)
            wkls.append(wkl)

        avg_score = np.mean(scores)
        avg_wkl = np.mean(wkls)
        row = f"{temp:>6.2f}"
        for sc in scores:
            row += f"  {sc:>8.2f}"
        row += f"   {avg_score:>6.2f}   {avg_wkl:>7.5f}"
        if avg_score > best_avg:
            best_avg = avg_score
            best_temp = temp
            row += "  *"
        print(row)

    print("-" * len(header))
    print(f"\nBest temperature: {best_temp:.2f}  (avg score {best_avg:.2f})")


if __name__ == "__main__":
    main()
