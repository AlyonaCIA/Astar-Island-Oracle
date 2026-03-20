"""
Astar Island — Observation Viewport Visualizer

Reads saved observations and round data, plots a 5-panel figure (one per seed)
showing the 40×40 initial terrain grid with queried viewport tiles overlaid.

Usage:
    python observations_viz.py                          # auto-detect latest
    python observations_viz.py data/observations_76909e29.json
    python observations_viz.py data/observations_76909e29.json data/round_76909e29.json
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Terrain code → class index (same mapping as the rest of the codebase)
TERRAIN_CLASS_MAP = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Class names and colors
CLASS_NAMES = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
CLASS_COLORS = [
    "#d4c89a",  # 0 Empty/Plains  — sandy tan
    "#e04040",  # 1 Settlement    — red
    "#4080e0",  # 2 Port          — blue
    "#a0522d",  # 3 Ruin          — brown
    "#228b22",  # 4 Forest        — green
    "#808080",  # 5 Mountain      — grey
]

# Separate initial-terrain colors (more specific than class colors)
INIT_COLORS = {
    10: "#1a5276",  # Ocean       — dark blue
    11: "#d4c89a",  # Plains      — sandy tan
    0:  "#f5f0e1",  # Empty       — off-white
    1:  "#e04040",  # Settlement  — red
    2:  "#4080e0",  # Port        — blue
    3:  "#a0522d",  # Ruin        — brown
    4:  "#228b22",  # Forest      — green
    5:  "#808080",  # Mountain    — grey
}

# For the viewport overlay: one color per query pass
QUERY_CMAP = plt.cm.tab10


def load_observations(obs_path):
    with open(obs_path) as f:
        return json.load(f)


def load_round_data(round_path):
    with open(round_path) as f:
        return json.load(f)


def find_latest_files():
    """Auto-detect the most recent observation and matching round file."""
    obs_files = sorted(glob.glob(os.path.join(DATA_DIR, "observations_*.json")))
    if not obs_files:
        print(f"No observation files found in {DATA_DIR}")
        sys.exit(1)
    obs_path = obs_files[-1]
    # Extract round key (e.g. "76909e29") from filename
    key = os.path.basename(obs_path).replace("observations_", "").replace(".json", "")
    round_path = os.path.join(DATA_DIR, f"round_{key}.json")
    if not os.path.exists(round_path):
        round_path = None
    return obs_path, round_path


def terrain_grid_to_rgb(grid, width, height):
    """Convert a terrain grid (codes) to an RGB image for plotting."""
    img = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            code = grid[y][x]
            hex_color = INIT_COLORS.get(code, "#f5f0e1")
            r = int(hex_color[1:3], 16) / 255
            g = int(hex_color[3:5], 16) / 255
            b = int(hex_color[5:7], 16) / 255
            img[y, x] = [r, g, b]
    return img


def observed_class_grid(observations, seed_idx, width, height):
    """Build a heatmap of how many times each cell was observed for a seed."""
    counts = np.zeros((height, width), dtype=int)
    for obs in observations:
        if obs["seed_index"] != seed_idx:
            continue
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        obs_grid = obs["grid"]
        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0])):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    counts[y, x] += 1
    return counts


def plot_observations(observations, round_data=None, width=40, height=40):
    seeds_count = 5
    if round_data:
        width = round_data.get("map_width", width)
        height = round_data.get("map_height", height)
        seeds_count = round_data.get("seeds_count", 5)
        initial_states = round_data.get("initial_states", [])
    else:
        initial_states = []

    # Group observations by seed
    seed_obs = {s: [] for s in range(seeds_count)}
    for obs in observations:
        s = obs["seed_index"]
        if s in seed_obs:
            seed_obs[s].append(obs)

    fig, axes = plt.subplots(1, seeds_count, figsize=(4.5 * seeds_count, 5.5),
                             constrained_layout=True)
    if seeds_count == 1:
        axes = [axes]

    round_key = ""
    if round_data:
        round_key = f" — Round #{round_data.get('round_number', '?')}"

    fig.suptitle(
        f"Viewport Query Coverage{round_key}\n"
        f"{len(observations)} queries across {seeds_count} seeds  "
        f"({width}×{height} map)",
        fontsize=13, fontweight="bold",
    )

    for s in range(seeds_count):
        ax = axes[s]

        # Background: initial terrain or blank grid
        if s < len(initial_states) and "grid" in initial_states[s]:
            bg = terrain_grid_to_rgb(initial_states[s]["grid"], width, height)
            ax.imshow(bg, origin="upper", extent=[0, width, height, 0],
                      interpolation="nearest", alpha=0.6)
        else:
            ax.set_facecolor("#f0ece0")

        # Observation count heatmap (semi-transparent overlay)
        counts = observed_class_grid(observations, s, width, height)
        if counts.max() > 0:
            cmap_heat = plt.cm.YlOrRd
            masked = np.ma.masked_where(counts == 0, counts)
            ax.imshow(masked, origin="upper", extent=[0, width, height, 0],
                      cmap=cmap_heat, alpha=0.35, vmin=0,
                      vmax=max(counts.max(), 1), interpolation="nearest")

        # Draw viewport rectangles with query order
        obs_list = seed_obs[s]
        for i, obs in enumerate(obs_list):
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            color = QUERY_CMAP(i / max(len(obs_list) - 1, 1))
            rect = patches.Rectangle(
                (vx, vy), vw, vh,
                linewidth=2, edgecolor=color, facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)
            # Query order label in corner of tile
            ax.text(
                vx + 0.5, vy + 0.5, str(i + 1),
                fontsize=7, fontweight="bold",
                color="white", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc=color, alpha=0.85,
                          edgecolor="none"),
            )

        # Coverage stats
        total_cells = width * height
        observed_cells = (counts > 0).sum()
        multi_observed = (counts > 1).sum()
        pct = observed_cells / total_cells * 100

        ax.set_title(
            f"Seed {s}  ({len(obs_list)} queries)\n"
            f"{observed_cells}/{total_cells} cells ({pct:.0f}%) covered"
            + (f", {multi_observed} multi-obs" if multi_observed else ""),
            fontsize=9,
        )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks(range(0, width + 1, 5))
        ax.set_yticks(range(0, height + 1, 5))
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_aspect("equal")

    # Legend for terrain colors
    legend_elements = []
    for code, hex_c in sorted(INIT_COLORS.items()):
        name = {10: "Ocean", 11: "Plains", 0: "Empty", 1: "Settlement",
                2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}[code]
        r = int(hex_c[1:3], 16) / 255
        g = int(hex_c[3:5], 16) / 255
        b = int(hex_c[5:7], 16) / 255
        legend_elements.append(
            patches.Patch(facecolor=(r, g, b), edgecolor="grey",
                          label=f"{name} ({code})")
        )
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=len(legend_elements), fontsize=7, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Also print a summary table
    print("\n  Query distribution per seed:")
    print(f"  {'Seed':>4}  {'Queries':>7}  {'Coverage':>8}  {'Multi-obs':>9}  Tile positions")
    for s in range(seeds_count):
        counts = observed_class_grid(observations, s, width, height)
        obs_list = seed_obs[s]
        tiles = [(o["viewport"]["x"], o["viewport"]["y"],
                  o["viewport"]["w"], o["viewport"]["h"]) for o in obs_list]
        unique_tiles = set(tiles)
        pct = (counts > 0).sum() / (width * height) * 100
        multi = (counts > 1).sum()
        tile_strs = [f"({x},{y}){w}x{h}" for x, y, w, h in unique_tiles]
        print(f"  {s:>4}  {len(obs_list):>7}  {pct:>7.1f}%  {multi:>9}  {', '.join(tile_strs)}")

    plt.savefig(os.path.join(SCRIPT_DIR, "observations_viz.png"),
                dpi=150, bbox_inches="tight")
    print(f"\n  Saved: observations_viz.png")
    plt.show()


def main():
    args = sys.argv[1:]

    if len(args) >= 1:
        obs_path = args[0]
        round_path = args[1] if len(args) >= 2 else None
        if round_path is None:
            key = os.path.basename(obs_path).replace("observations_", "").replace(".json", "")
            candidate = os.path.join(DATA_DIR, f"round_{key}.json")
            if os.path.exists(candidate):
                round_path = candidate
    else:
        obs_path, round_path = find_latest_files()

    print(f"  Observations: {obs_path}")
    print(f"  Round data:   {round_path or '(not found — using defaults)'}")

    observations = load_observations(obs_path)
    round_data = load_round_data(round_path) if round_path else None

    plot_observations(observations, round_data)


if __name__ == "__main__":
    main()
