#!/usr/bin/env python3
"""Reproduce the complete metrical-preference summary table."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

HERE = Path(__file__).resolve().parent
DATA = HERE / "metrical_preference_per_excerpt.csv"
ORDER = ["tactus_match", "half_time", "double_time", "other_flip"]
LABELS = {
    "tactus_match": "Matched rate",
    "half_time": "Crowd half rate",
    "double_time": "Crowd double rate",
    "other_flip": "Non-integer",
}


def main() -> None:
    data = pd.read_csv(DATA)
    if len(data) != 180:
        raise ValueError("Expected all 180 expert-referenced excerpts")

    rng = np.random.default_rng(20260718)
    records = []
    for metrical_class in ORDER:
        values = data.loc[
            data["metrical_class"] == metrical_class,
            "rating_delta",
        ].to_numpy()
        medians = np.median(
            rng.choice(values, size=(20_000, len(values)), replace=True),
            axis=1,
        )
        ci_low, ci_high = np.quantile(medians, [0.025, 0.975])
        records.append(
            {
                "metrical_class": metrical_class,
                "label": LABELS[metrical_class],
                "n_excerpts": len(values),
                "n_crowd_rating_higher": int(np.sum(values > 0)),
                "mean_rating_delta": float(np.mean(values)),
                "median_rating_delta": float(np.median(values)),
                "median_ci95_low": float(ci_low),
                "median_ci95_high": float(ci_high),
                "min_rating_delta": float(np.min(values)),
                "max_rating_delta": float(np.max(values)),
            }
        )

    pd.DataFrame(records).to_csv(
        HERE / "metrical_preference_summary.csv",
        index=False,
    )
    alternatives = [
        data.loc[data["metrical_class"] == name, "rating_delta"].to_numpy()
        for name in ORDER[1:]
    ]
    omnibus_p = float(kruskal(*alternatives).pvalue)
    (HERE / "metrical_preference_statistics.json").write_text(
        json.dumps(
            {
                "n_expert_referenced_excerpts": len(data),
                "kruskal_wallis_alternative_classes_p": omnibus_p,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: summarized 180 excerpts; alternative-class p={omnibus_p:.6f}")


if __name__ == "__main__":
    main()
