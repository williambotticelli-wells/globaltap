#!/usr/bin/env python3
"""Single-column versions of the period-relation summary for the main paper.

Same data and cluster-bootstrap intervals as
`plot_country_tactus_choice_summary.py`, sized for one ISMIR column so the
figure can sit in the Results section. Two layouts, because column space is
the binding constraint:

    country_tactus_choice_maintext.pdf         2x2, all four categories
    country_tactus_choice_maintext_2panel.pdf  1x2, the two categories the
                                               text discusses

The 2x2 layout is Fig. 3 of the paper.
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
CATEGORIES = {
    "double_time": ("Double rate", "#0072B2"),
    "half_time": ("Half rate", "#D55E00"),
    "matched": ("Matched rate", "#009E73"),
    "other_unclear": ("Other/unclear", "#999999"),
}


def darken(color: str, factor: float = 0.5) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def draw(df, cats, nrows, ncols, figsize, suffix) -> None:
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    grid = np.atleast_2d(axes)
    for ax, cat in zip(grid.ravel(), cats):
        label, color = CATEGORIES[cat]
        point = df[f"{cat}_prop"].to_numpy()
        err = [point - df[f"{cat}_lo95"].to_numpy(),
               df[f"{cat}_hi95"].to_numpy() - point]
        ax.bar(x, point, width=0.78, yerr=err, color=color,
               capsize=1.2, error_kw={"elinewidth": 0.55, "capthick": 0.55})
        ax.axhline(point.mean(), color=darken(color), linestyle=(0, (4, 2.5)),
                   linewidth=0.7, alpha=0.9, zorder=3)
        ax.set_title(label, fontsize=6.5, pad=2)
        ax.set_ylim(0, max(df[f"{cat}_hi95"].max() * 1.12, 0.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)
        ax.tick_params(axis="y", labelsize=5.5, length=2, width=0.5, pad=1)
        ax.tick_params(axis="x", length=2, width=0.5, pad=1)
    for ax in grid[-1]:
        ax.set(xticks=x, xticklabels=ORDER)
        ax.tick_params(axis="x", labelsize=5.5)
    fig.supylabel("Proportion of trials", fontsize=6.5, x=0.005)
    fig.tight_layout(pad=0.25, h_pad=0.7, w_pad=1.0, rect=(0.02, 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"country_tactus_choice_maintext{suffix}.{ext}",
                    dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(HERE / f"country_tactus_choice_maintext{suffix}.pdf")


def main() -> None:
    df = pd.read_csv(CSV).set_index("listener_country").loc[ORDER]
    draw(df, ["matched", "double_time", "half_time", "other_unclear"],
         2, 2, (3.35, 2.35), "")
    draw(df, ["double_time", "half_time"], 1, 2, (3.35, 1.35), "_2panel")


if __name__ == "__main__":
    main()
