#!/usr/bin/env python3
"""
Figure 1A (paper) — single-row pipeline schematic.

Pipeline overview reproduced from the paper.  Self-contained: no
external data dependencies.

Usage:
    python figures/pipeline_schematic.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from fig_utils import apply_style

OUT = Path(__file__).parent / "pipeline_schematic.pdf"

STEP_COLORS = [
    "#3F51B5",
    "#00897B",
    "#F57C00",
    "#7B1FA2",
    "#1565C0",
]

STEPS = [
    ("Online\ntapping",            "N participants\ntap freely"),
    ("Pool\nresponses",            "Aggregate across\nlisteners"),
    ("KDE\nestimation",            "Gaussian kernel\n\u03c3 = 80 ms"),
    ("Consensus\nannotations",     "Peak extraction +\nnear-isochronous fit"),
    ("Validated beat\nannotation", "Click-track\nevaluation"),
]


def draw(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.65, 1.15)
    ax.axis("off")

    n = len(STEPS)
    box_w, box_h = 1.65, 0.74
    spacing = (10.0 - n * box_w) / (n - 1) + box_w
    start_x = (10.0 - (n - 1) * spacing) / 2

    for i, ((label, sublabel), color) in enumerate(zip(STEPS, STEP_COLORS)):
        cx = start_x + i * spacing
        cy = 0.5
        box = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.04", facecolor=color, edgecolor="white",
            linewidth=1.6, alpha=0.92, zorder=5,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.02, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white",
                zorder=6, linespacing=1.05)
        ax.text(cx, cy - box_h / 2 - 0.20, sublabel, ha="center", va="top",
                fontsize=8.0, color="#444444", linespacing=1.15,
                style="italic")
        if i < n - 1:
            x_start = cx + box_w / 2 + 0.05
            x_end = start_x + (i + 1) * spacing - box_w / 2 - 0.05
            ax.annotate(
                "", xy=(x_end, cy), xytext=(x_start, cy),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.22,head_length=0.16",
                    color="#888888", lw=2.0,
                    connectionstyle="arc3,rad=0",
                ),
                zorder=4,
            )


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 1.10))
    draw(ax)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight",
                pad_inches=0.02, dpi=300)
    print(f"Saved \u2192 {OUT}")
    print(f"Saved \u2192 {OUT.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    main()
