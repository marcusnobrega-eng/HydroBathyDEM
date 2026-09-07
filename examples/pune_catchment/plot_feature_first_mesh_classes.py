"""Render a readable class map for a feature-first HydroPol mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLOURS = {
    "river": "#0B81A2",
    "floodplain": "#8CC5E3",
    "urban": "#E6A0A1",
    "rural": "#ECECEC",
}
LABELS = {
    "river": "Mapped river core\n30 m along-flow; ≤90 m across",
    "floodplain": "HAND floodplain\n40 m along-flow; 90 m across",
    "urban": "Urban / impervious\n45 m target width",
    "rural": "Rural background\n120 m target width",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    mesh = gpd.read_file(args.mesh, layer="mesh")
    fig, ax = plt.subplots(figsize=(9.0, 7.2), layout="constrained")
    for feature_class in ("rural", "urban", "floodplain", "river"):
        subset = mesh[mesh.feature_class == feature_class]
        if not subset.empty:
            subset.plot(ax=ax, color=COLOURS[feature_class], edgecolor="#262626", linewidth=0.34)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Pune feature-first mesh: cell classes", fontname="Helvetica", fontsize=15, pad=8)
    handles = [Patch(facecolor=COLOURS[item], edgecolor="#262626", label=LABELS[item]) for item in ("river", "floodplain", "urban", "rural")]
    legend = ax.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9, handlelength=1.3, columnspacing=1.8)
    for text in legend.get_texts():
        text.set_fontname("Helvetica")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
