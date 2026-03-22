#!/usr/bin/env python3
"""
Minimal local CNN prior for Bayesian_approach.

This is a self-contained copy of the pieces needed to load the refreshed
`unet_cond` checkpoint without depending on the original repo.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "cnn_latest.pt")
NUM_CLASSES = 6
PROB_FLOOR = 1e-6
TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
OBS_CHANNELS = 7


def terrain_to_class(cell_value: int) -> int:
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(cell_value, 0)


def encode_initial_grid(initial_grid: Sequence[Sequence[int]], width: int, height: int) -> np.ndarray:
    features = np.zeros((14, height, width), dtype=np.float32)
    code_to_channel = {code: i for i, code in enumerate(TERRAIN_CODES)}
    class_grid = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            cell = initial_grid[y][x]
            features[code_to_channel.get(cell, 0), y, x] = 1.0
            class_grid[y, x] = terrain_to_class(cell)

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


def encode_obs_channels(observations, width: int, height: int) -> np.ndarray:
    obs_counts = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    obs_hits = np.zeros((height, width), dtype=np.float32)

    for obs in observations:
        viewport = obs["viewport"]
        vx, vy = viewport["x"], viewport["y"]
        for dy, row in enumerate(obs["grid"]):
            for dx, value in enumerate(row):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < height and 0 <= gx < width:
                    obs_counts[terrain_to_class(value), gy, gx] += 1.0
                    obs_hits[gy, gx] += 1.0

    mask = obs_hits > 0
    for c in range(NUM_CLASSES):
        obs_counts[c][mask] /= obs_hits[mask]
    coverage = np.log1p(obs_hits)[np.newaxis, :, :]
    return np.concatenate([obs_counts, coverage], axis=0)


class MiniUNet(nn.Module):
    def __init__(self, dropout: float = 0.1, in_channels: int = 14 + OBS_CHANNELS):
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
        self.out_conv = nn.Conv2d(32, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        _, _, height, width = x.shape
        pad_h = (2 - height % 2) % 2
        pad_w = (2 - width % 2) % 2
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
            logits = logits[:, :, :height, :width]
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=PROB_FLOOR)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class LocalCnnPrior:
    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self._load()

    def _load(self) -> MiniUNet:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model = MiniUNet().to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def predict(self, initial_grid, observations, width: int, height: int) -> np.ndarray:
        features = encode_initial_grid(initial_grid, width, height)
        obs_features = encode_obs_channels(observations, width, height)
        x = np.concatenate([features, obs_features], axis=0)
        with torch.no_grad():
            probs = (
                self.model(torch.tensor(x).unsqueeze(0).to(self.device))
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
        probs = np.maximum(probs, PROB_FLOOR)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs
