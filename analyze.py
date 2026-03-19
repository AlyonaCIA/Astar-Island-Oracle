"""
Astar Island — Training Analysis & Plots

Reads checkpoints/training_history.json (produced by train_cnn.py) and
generates plots of training/validation KL divergence vs epochs and time.

Usage:
    python analyze.py                         # default history file
    python analyze.py path/to/history.json    # custom file
    python analyze.py --no-show               # save PNGs without opening windows
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HISTORY = os.path.join(SCRIPT_DIR, "checkpoints", "training_history.json")


def load_history(path):
    if not os.path.isfile(path):
        print(f"ERROR: History file not found: {path}")
        print("  Run train_cnn.py first to generate training history.")
        sys.exit(1)
    with open(path) as f:
        history = json.load(f)
    print(f"Loaded {len(history)} entries from {path}")
    return history


def plot_kl_vs_epoch(history, save_dir, show):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_kl = [h["train_kl"] for h in history]
    val_kl = [h["val_kl"] for h in history]
    best_val = [h.get("best_val_kl", h["val_kl"]) for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_kl, label="Train KL", alpha=0.8, linewidth=1.2)
    ax.plot(epochs, val_kl, label="Val KL", alpha=0.8, linewidth=1.2)
    ax.plot(epochs, best_val, label="Best Val KL", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "kl_vs_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_kl_vs_time(history, save_dir, show):
    import matplotlib.pyplot as plt

    times = [h["elapsed_s"] / 60.0 for h in history]  # minutes
    train_kl = [h["train_kl"] for h in history]
    val_kl = [h["val_kl"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, train_kl, label="Train KL", alpha=0.8, linewidth=1.2)
    ax.plot(times, val_kl, label="Val KL", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence vs Training Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "kl_vs_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_kl_log_scale(history, save_dir, show):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_kl = [h["train_kl"] for h in history]
    val_kl = [h["val_kl"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, train_kl, label="Train KL", alpha=0.8, linewidth=1.2)
    ax.semilogy(epochs, val_kl, label="Val KL", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence (log scale)")
    ax.set_title("KL Divergence vs Epoch (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    path = os.path.join(save_dir, "kl_vs_epoch_log.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_simulated_score(history, save_dir, show):
    """Plot simulated competition score: 100 * exp(-3 * val_kl)"""
    import matplotlib.pyplot as plt
    import math

    epochs = [h["epoch"] for h in history]
    scores = [100.0 * math.exp(-3.0 * h["val_kl"]) for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, scores, label="Simulated Score", color="green", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Simulated Competition Score vs Epoch")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "score_vs_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(history):
    import math

    if not history:
        return

    first = history[0]
    last = history[-1]
    best_entry = min(history, key=lambda h: h["val_kl"])

    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Epochs:      {first['epoch']} → {last['epoch']} ({len(history)} entries)")
    print(f"  Duration:    {last['elapsed_s']:.0f}s ({last['elapsed_s']/60:.1f} min)")
    print()
    print(f"  First epoch:  train_kl={first['train_kl']:.6f}  val_kl={first['val_kl']:.6f}")
    print(f"  Last epoch:   train_kl={last['train_kl']:.6f}  val_kl={last['val_kl']:.6f}")
    print(f"  Best val_kl:  {best_entry['val_kl']:.6f} (epoch {best_entry['epoch']})")
    print()

    first_score = 100.0 * math.exp(-3.0 * first["val_kl"])
    last_score = 100.0 * math.exp(-3.0 * last["val_kl"])
    best_score = 100.0 * math.exp(-3.0 * best_entry["val_kl"])
    print(f"  Simulated scores (100 × exp(-3 × val_kl)):")
    print(f"    First:  {first_score:.2f}")
    print(f"    Last:   {last_score:.2f}")
    print(f"    Best:   {best_score:.2f} (epoch {best_entry['epoch']})")

    # Overfitting check
    gap = last["val_kl"] - last["train_kl"]
    if gap > 0.05:
        print(f"\n  ⚠ Possible overfitting: val-train gap = {gap:.4f}")
    elif last["val_kl"] < last["train_kl"]:
        print(f"\n  ℹ Val loss below train loss — model may still be underfitting")
    print()


def main():
    parser = argparse.ArgumentParser(description="Astar Island — Training Analysis")
    parser.add_argument("history", nargs="?", default=DEFAULT_HISTORY,
                        help="Path to training_history.json")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots as PNG without displaying them")
    args = parser.parse_args()

    history = load_history(args.history)

    if not history:
        print("History file is empty.")
        return

    # Save plots next to the history file
    save_dir = os.path.dirname(os.path.abspath(args.history))
    show = not args.no_show

    print_summary(history)

    print("Generating plots...")
    plot_kl_vs_epoch(history, save_dir, show)
    plot_kl_vs_time(history, save_dir, show)
    plot_kl_log_scale(history, save_dir, show)
    plot_simulated_score(history, save_dir, show)

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
