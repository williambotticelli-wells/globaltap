#!/usr/bin/env python3
"""Cluster-bootstrap CIs for the per-panel period-relation proportions.

Writes `panel_tests/country_tactus_choice_proportions_with_ci.csv`, the
input for the period-relation figures. The point
estimates equal the pooled per-panel rates in
`panel_tests/country_tactus_choice_proportions.csv`; this script adds 95%
bootstrap intervals.

The bootstrap resamples participants (clusters), not trials: one
participant contributes many trials, and the inferential tests on this
data (`run_panel_tests.py`) already treat the participant as the unit of
replication. Resampling trials directly would understate the uncertainty.

Requires the raw tapping export (see the repository README). The included
CSV lets the figure be regenerated without it.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from pipeline_subset_core import GLOBALTAP_ROOT, load_and_clean_trials  # noqa: E402
from run_globaltap_phase_periodicity_nested import build_trial_metrics  # noqa: E402

DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
ORDER = ["AU", "BR", "EG", "FR", "JP", "KR", "MX", "PL", "US", "ZA"]
CATEGORIES = ["double_time", "matched", "half_time", "other_unclear"]
N_BOOT = 10000
SEED = 20260728


def cluster_bootstrap_ci(
    participant_counts: np.ndarray,
    participant_totals: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap over participants; returns (point, lo95, hi95)."""
    n = len(participant_totals)
    observed = participant_counts.sum(axis=0) / participant_totals.sum()
    boot = np.empty((n_boot, participant_counts.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = participant_counts[idx].sum(axis=0) / participant_totals[idx].sum()
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return observed, lo, hi


def main() -> None:
    cleaned = load_and_clean_trials(DEFAULT_INPUT)
    base = cleaned.rename(columns={"stem": "stim_name", "clean_taps": "taps"})
    trials = build_trial_metrics(base)

    one_hot = pd.get_dummies(trials["tactus_choice"]).reindex(
        columns=["double_time", "matched", "half_time", "other", "unclear"],
        fill_value=0,
    )
    trials = pd.concat([trials.reset_index(drop=True), one_hot], axis=1)
    trials["other_unclear"] = trials["other"] + trials["unclear"]

    rng = np.random.default_rng(SEED)
    rows = []
    for country in ORDER:
        sub = trials[trials["listener_country"] == country]
        per_participant = sub.groupby("participant_id")[CATEGORIES].sum()
        counts = per_participant.to_numpy(float)
        totals = counts.sum(axis=1)
        point, lo, hi = cluster_bootstrap_ci(counts, totals, rng)
        rows.append(
            {
                "listener_country": country,
                "n_participants": len(per_participant),
                "n_trials": int(totals.sum()),
                **{f"{cat}_prop": point[i] for i, cat in enumerate(CATEGORIES)},
                **{f"{cat}_lo95": lo[i] for i, cat in enumerate(CATEGORIES)},
                **{f"{cat}_hi95": hi[i] for i, cat in enumerate(CATEGORIES)},
            }
        )
    out_csv = HERE / "panel_tests" / "country_tactus_choice_proportions_with_ci.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
