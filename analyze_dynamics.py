"""Quick analysis of grid dynamics at various strides."""
import json, os, numpy as np

replay_dir = "simulation_replays"
replays = sorted(os.listdir(replay_dir))

# Per-step change rates
step_changes = []
for fname in replays[:10]:
    d = json.load(open(os.path.join(replay_dir, fname)))
    frames = d["frames"]
    for i in range(len(frames) - 1):
        g1 = np.array(frames[i]["grid"])
        g2 = np.array(frames[i + 1]["grid"])
        pct = (g1 != g2).sum() / g1.size * 100
        step_changes.append(pct)
c = np.array(step_changes)
print(f"Stride-1 changes: mean={c.mean():.2f}% median={np.median(c):.2f}% max={c.max():.2f}%")

# Stride-10 changes
for stride in [5, 10, 25, 50]:
    sc = []
    for fname in replays[:10]:
        d = json.load(open(os.path.join(replay_dir, fname)))
        frames = d["frames"]
        for i in range(0, len(frames) - stride, stride):
            g1 = np.array(frames[i]["grid"])
            g2 = np.array(frames[i + stride]["grid"])
            pct = (g1 != g2).sum() / g1.size * 100
            sc.append(pct)
    sc = np.array(sc)
    print(f"Stride-{stride:2d} changes: mean={sc.mean():.2f}% median={np.median(sc):.2f}% max={sc.max():.2f}%")

# Settlement lifecycle: how many alive per step?
print("\n=== Settlement counts over time ===")
for fname in replays[:3]:
    d = json.load(open(os.path.join(replay_dir, fname)))
    counts = []
    for f in d["frames"]:
        alive = sum(1 for s in f["settlements"] if s["alive"])
        counts.append(alive)
    print(f"  {fname[:30]}: start={counts[0]}, mid={counts[25]}, end={counts[-1]}")

# Replay settlement data: what properties are tracked?
d = json.load(open(os.path.join(replay_dir, replays[0])))
f0 = d["frames"][0]
f50 = d["frames"][-1]
print(f"\nStep 0: {len(f0['settlements'])} settlements")
print(f"Step 50: {len(f50['settlements'])} settlements")

# Compare replay settlement data vs inference-time settlement data
print("\n=== Replay settlement (full info) ===")
print(json.dumps(f0["settlements"][0], indent=2))

# Count training pairs at different strides
total_pairs = {}
for stride in [1, 2, 5, 10, 25, 50]:
    n = 0
    for fname in replays:
        d = json.load(open(os.path.join(replay_dir, fname)))
        nf = len(d["frames"])
        for i in range(nf - stride):
            n += 1
    total_pairs[stride] = n
    print(f"Stride {stride:2d}: {n:6d} training pairs")
