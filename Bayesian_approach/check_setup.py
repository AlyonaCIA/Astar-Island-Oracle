#!/usr/bin/env python3
"""Lightweight workspace validation for Bayesian_approach."""

from __future__ import annotations

import glob
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    problems = []

    gt_files = glob.glob(os.path.join(SCRIPT_DIR, "ground_truth", "r*_s*_*.json"))
    if not gt_files:
        problems.append("No ground truth files found under ground_truth/")

    ckpt = os.path.join(SCRIPT_DIR, "checkpoints", "cnn_latest.pt")
    if not os.path.exists(ckpt):
        problems.append("Missing CNN checkpoint at checkpoints/cnn_latest.pt")

    env_path = os.path.join(SCRIPT_DIR, ".env")
    if not os.path.exists(env_path):
        problems.append("Missing .env file")
    else:
        text = open(env_path).read()
        if "ASTAR_TOKEN=" not in text:
            problems.append(".env exists but ASTAR_TOKEN is missing")

    if problems:
        print("Workspace check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Workspace check passed.")
    print(f"Ground truth files: {len(gt_files)}")
    print(f"Checkpoint: {ckpt}")
    print(f"Env file: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
