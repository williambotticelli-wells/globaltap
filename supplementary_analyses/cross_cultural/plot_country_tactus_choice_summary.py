#!/usr/bin/env python3
"""Plot the aggregate period-relation summary (tall stacked layout).

One panel per period category, all sharing the country axis, with 95%
cluster-bootstrap error bars (see `run_country_tactus_choice_bootstrap.py`)
and a dashed line marking each category's mean across the ten panels.

Only the period/tactus-choice breakdown is plotted here. The test of overall
circular-phase differences (backed by `panel_tests/omnibus_panel_tests.csv`)
found no panel-level heterogeneity (permutation p=.152), so a per-country bar
chart of phase point estimates alone would visually suggest country-to-country
differences that the test does not support; it is reported as a sentence in
the text instead of a figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
CSV = HERE / "panel_tests" / "country_tactus_choice_proportions_with_ci.csv"
ORDER = ["AU", "BR", "EG", "FR", "JP", "KR", "MX", "PL", "US", "ZA"]
CATEGORIES = [
    ("double_time", "Double rate", "#0072B2"),
    ("matched", "Matched rate", "#009E73"),
    ("half_time", "Half rate", "#D55E00"),
    ("other_unclear", "Other/unclear", "#999999"),
]


def darken(color: str, factor: float = 0.5) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def main() -> None:
    df = pd.read_csv(CSV).set_index("listener_country").loc[ORDER]
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(4, 1, figsize=(4.6, 7.6), sharex=True)
    for ax, (cat, label, color) in zip(axes, CATEGORIES):
        point = df[f"{cat}_prop"].to_numpy()
        err = [point - df[f"{cat}_lo95"].to_numpy(),
               df[f"{cat}_hi95"].to_numpy() - point]
        ax.bar(x, point, width=0.72, yerr=err, color=color,
               capsize=2.5, error_kw={"elinewidth": 0.9, "capthick": 0.9})
        mean = point.mean()
        ax.axhline(mean, color=darken(color), linestyle=(0, (5, 3)),
                   linewidth=0.9, alpha=0.9, zorder=3)
        ax.annotate(f"{mean:.2f}".lstrip("0"), xy=(1.005, mean),
                    xycoords=("axes fraction", "data"), fontsize=6.5,
                    color=darken(color), va="center", ha="left",
                    annotation_clip=False)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, max(df[f"{cat}_hi95"].max() * 1.15, 0.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
    axes[-1].set(xticks=x, xticklabels=ORDER)
    axes[-1].tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Period relative to other panels (95% cluster bootstrap CI)",
        fontsize=9.5, y=0.995,
    )
    fig.text(0.5, 0.973, "dashed lines: mean across panels", ha="center",
             va="top", fontsize=7.5, color="0.35")
    fig.supylabel("Proportion of trials", fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0, 0.97, 0.955))

    fig.savefig(HERE / "country_tactus_choice_summary.png", dpi=240,
                bbox_inches="tight")
    fig.savefig(HERE / "country_tactus_choice_summary.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(HERE / "country_tactus_choice_summary.png")


if __name__ == "__main__":
    main()
