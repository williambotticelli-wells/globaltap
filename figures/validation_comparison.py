#!/usr/bin/env python3
"""
Supplementary figure — per-stem perceptual validation across corpus groups.

Reads the included per-stem metrics CSV (``data/per_stem_metrics.csv``)
so the figure can be regenerated without raw trial-level data.
Each panel shows per-stem mean Crowd vs Reference ratings for one
corpus group, with grouped means and SE error bars.  The mixed-corpora
panel also overlays per-source corpus means so the heterogeneous mix
is legible.

Usage:
    python figures/validation_comparison.py [--wide]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig_utils import apply_style, PER_STEM_CSV, MIXED_CORPUS_LABELS

OUT = Path(__file__).parent / "validation_comparison.pdf"

CORPUS_ORDER = ["ballroom", "benchmark100", "mirex", "globalmood40"]
CORPUS_TITLES = {
    "ballroom":     "Ballroom",
    "benchmark100": "Mixed corpora",
    "mirex":        "MIREX",
    "globalmood40": "GlobalMood",
}
HAS_REF = {
    "ballroom":     True,
    "benchmark100": True,
    "mirex":        True,
    "globalmood40": False,
}
CROWD_COLOR = "#2166AC"
REF_COLOR = "#4DAF4A"


def load_data():
    df = pd.read_csv(PER_STEM_CSV).rename(columns={"_stem": "stem"})
    df["crowd"] = pd.to_numeric(df["crowd_rating"], errors="coerce")
    df["ref"] = pd.to_numeric(df["ref_rating"], errors="coerce")
    return df


def _draw_paired_panel(
    ax, sub, has_ref, corpus_group, *,
    mean_marker_size=8, scatter_size=10,
    mean_label_fontsize=7, title_fontsize=9,
    per_corpus_label_fontsize=6.5, per_corpus_min_sep=0.20,
    label_x=1.18, leader_x_start=0.97,
    show_per_corpus_labels=True,
):
    """One Crowd-vs-Reference paired panel for a corpus group."""
    crowd_means = sub["crowd"].dropna()
    n = len(crowd_means)

    if has_ref:
        paired = sub.dropna(subset=["crowd", "ref"]).copy()
        c = paired["crowd"].values
        g = paired["ref"].values

        for ci, gi in zip(c, g):
            ax.plot([0, 1], [ci, gi], color="#888888", alpha=0.15,
                    linewidth=0.5, zorder=1)
            ax.scatter([0, 1], [ci, gi], s=scatter_size, alpha=0.2,
                       color="#888888", zorder=2)

        if corpus_group == "benchmark100" and show_per_corpus_labels:
            _plot_mixed_corpus_means(
                ax, sub,
                label_fontsize=per_corpus_label_fontsize,
                min_sep=per_corpus_min_sep,
                label_x=label_x,
                leader_x_start=leader_x_start,
            )

        for i, (vals, color) in enumerate([(c, CROWD_COLOR), (g, REF_COLOR)]):
            m = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(
                i, m, yerr=se, fmt="o", color=color,
                markersize=mean_marker_size, capsize=4, capthick=1.5,
                linewidth=1.5, zorder=10,
                markeredgecolor="white", markeredgewidth=0.8,
            )
            if corpus_group == "benchmark100" and show_per_corpus_labels:
                ax.text(i, 1.5, f"{m:.2f}", ha="center", va="bottom",
                        fontsize=mean_label_fontsize, fontweight="bold",
                        color=color, zorder=11)
            else:
                ax.text(i, m - se - 0.22, f"{m:.2f}", ha="center", va="top",
                        fontsize=mean_label_fontsize, fontweight="bold",
                        color=color, zorder=11,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.7, pad=1))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Crowd", "Ref"])
        ax.set_xlim(
            -0.6,
            2.05 if (corpus_group == "benchmark100" and
                     show_per_corpus_labels) else 1.5,
        )
    else:
        vals = crowd_means.values
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(0, m, yerr=se, fmt="o", color=CROWD_COLOR,
                    markersize=mean_marker_size, capsize=4, capthick=1.5,
                    linewidth=1.5, zorder=10,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.scatter([0] * len(vals), vals, s=scatter_size, alpha=0.25,
                   color=CROWD_COLOR, zorder=2)
        ax.text(0, m - se - 0.22, f"{m:.2f}", ha="center", va="top",
                fontsize=mean_label_fontsize, fontweight="bold",
                color=CROWD_COLOR, zorder=11,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1))
        ax.set_xticks([0])
        ax.set_xticklabels(["Crowd"])
        ax.set_xlim(-0.7, 0.7)

    ax.set_title(f"{CORPUS_TITLES[corpus_group]} ($n={n}$)",
                 fontweight="bold", fontsize=title_fontsize)


def _plot_mixed_corpus_means(
    ax, sub, *,
    label_fontsize=6.5, min_sep=0.20,
    label_x=1.18, leader_x_start=0.97,
):
    """Overlay per-source corpus means on the mixed-corpora paired panel."""
    means = (
        sub.dropna(subset=["crowd", "ref"])
        .groupby("corpus")[["crowd", "ref"]]
        .mean()
        .dropna()
        .sort_values("ref", ascending=False)
    )
    n = len(means)
    if n == 0:
        return
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for (corpus_name, row), color in zip(means.iterrows(), colors):
        y = [row["crowd"], row["ref"]]
        ax.plot([0.06, 0.94], y, color=color, alpha=0.75,
                linewidth=1.0, zorder=6)
        ax.scatter([0.06, 0.94], y, s=22, color=color, edgecolors="white",
                   linewidths=0.45, zorder=7)

    target_y = means["ref"].values.astype(float)
    label_y = target_y.copy()

    y_min, y_max = ax.get_ylim() if ax.get_ylim()[1] > 2 else (1.3, 7.3)
    for _ in range(80):
        moved = False
        for i in range(len(label_y) - 1):
            gap = label_y[i] - label_y[i + 1]
            if gap < min_sep:
                shift = (min_sep - gap) / 2
                label_y[i] += shift
                label_y[i + 1] -= shift
                moved = True
        if not moved:
            break
    label_y = np.clip(label_y, y_min + 0.1, y_max - 0.15)

    leader_x_end = label_x - 0.03
    for (corpus_name, _row), color, ty, ly in zip(
        means.iterrows(), colors, target_y, label_y,
    ):
        ax.plot([leader_x_start, leader_x_end], [ty, ly],
                color=color, alpha=0.55, linewidth=0.5, zorder=5,
                clip_on=False)
        label = MIXED_CORPUS_LABELS.get(corpus_name, corpus_name)
        ax.text(label_x, ly, label, fontsize=label_fontsize, va="center",
                ha="left", color=color, zorder=8, clip_on=False,
                fontweight="medium")


def _plot_wide(df, out_path):
    fig, axes = plt.subplots(
        1, 4, figsize=(9.6, 3.6), sharey=True,
        gridspec_kw={"width_ratios": [1.3, 3.4, 1.1, 1.0]},
    )
    for ax, cg in zip(axes, CORPUS_ORDER):
        sub = df[df["corpus_group"] == cg]
        _draw_paired_panel(ax, sub, HAS_REF[cg], cg)
    axes[0].set_ylabel("Perceptual rating (1\u20137)")
    axes[0].set_ylim(1.3, 7.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"Saved \u2192 {out_path}")
    plt.close()


def _plot_singlecol(df, out_path):
    fig = plt.figure(figsize=(3.4, 4.6))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 0.85],
        hspace=0.55, wspace=0.30,
        left=0.13, right=0.97, top=0.95, bottom=0.10,
    )

    ax_top = fig.add_subplot(gs[0, :])
    sub = df[df["corpus_group"] == "benchmark100"]
    _draw_paired_panel(
        ax_top, sub, HAS_REF["benchmark100"], "benchmark100",
        mean_marker_size=6.5, scatter_size=8,
        mean_label_fontsize=7, title_fontsize=8,
        per_corpus_label_fontsize=6, per_corpus_min_sep=0.32,
        label_x=1.15, leader_x_start=0.97,
    )
    ax_top.set_ylabel("Rating (1\u20137)", fontsize=7.5)
    ax_top.set_ylim(1.3, 7.3)
    ax_top.tick_params(axis="both", labelsize=7)

    bottom_corpora = ["ballroom", "mirex", "globalmood40"]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, cg in zip(bottom_axes, bottom_corpora):
        sub = df[df["corpus_group"] == cg]
        _draw_paired_panel(
            ax, sub, HAS_REF[cg], cg,
            mean_marker_size=6, scatter_size=7,
            mean_label_fontsize=6.5, title_fontsize=7.5,
            show_per_corpus_labels=False,
        )
        ax.tick_params(axis="both", labelsize=7)

    bottom_axes[0].set_ylabel("Rating (1\u20137)", fontsize=7.5)
    for ax in bottom_axes:
        ax.set_ylim(1.3, 7.3)
    for ax in bottom_axes[1:]:
        ax.set_yticklabels([])

    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"Saved \u2192 {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wide", action="store_true",
        help="Regenerate full-width 1\u00d74 layout (default: single column).",
    )
    args = parser.parse_args()

    apply_style()
    df = load_data()
    if args.wide:
        _plot_wide(df, OUT)
    else:
        _plot_singlecol(df, OUT)


if __name__ == "__main__":
    main()
