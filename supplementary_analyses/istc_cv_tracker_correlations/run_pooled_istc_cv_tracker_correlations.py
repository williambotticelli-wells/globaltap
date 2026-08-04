#!/usr/bin/env python3
r"""Pooled ISTC/CV vs. tracker-agreement correlations, full 746-participant cohort.

This script computes the manuscript's ISTC/CV-vs-tracker-agreement
correlations (reported in \S3.4) from the complete 746-participant
benchmark release. It recomputes the correlations from:

1. Pooled ISTC per excerpt, using the paper's exact bin-counting definition
   (100 ms bins, 50 ms steps, max unique-participant count per 1-s window,
   averaged over the [0,30]s window), computed on the full 746-participant
   raw_tapping_data/merged/globalmood_200_merged.csv -- the same function
   already used for the per-country ISTC breakdown
   (cross_cultural/run_country_istc_cv_descriptives.py::istc_for_group), here
   applied with all countries pooled together (the paper's original unit of
   analysis).
2. Pooled final-grid CV (SD/mean of consensus inter-beat intervals) from the
   corrected 746-participant crowd_gs consensus
   (convergence/final_grid/full_746/*_crowd_gs.txt).
3. Per-stem tracker F-measure against that same corrected consensus
   (beat_metrics/per_stem_beat_metrics.csv),
   whose aggregate values (madmom F=.890 over all 200 excerpts, F=.897 over
   the 198 excerpts both trackers can be scored on, Beat This! F=.841) match
   the manuscript's §3.4 numbers, confirming this is the right cohort
   (median 117 tappers/stem, matching the manuscript's stated participant
   count).

Verified output (Spearman rho, two-sided p, n = number of excerpts with both
values defined):

    ISTC vs. madmom F      : rho=.395  p=7.4e-9   n=200
    ISTC vs. Beat This! F  : rho=.266  p=1.5e-4   n=198
    CV   vs. madmom F      : rho=-.184 p=.0092    n=200
    CV   vs. Beat This! F  : rho=-.243 p=5.8e-4   n=198

These are the values reported in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CROSS_CULTURAL_DIR = HERE.parent / "cross_cultural"
sys.path.insert(0, str(CROSS_CULTURAL_DIR))
sys.path.insert(0, str(HERE.parent))

from run_country_istc_cv_descriptives import istc_for_group  # noqa: E402
from pipeline_subset_core import GLOBALTAP_ROOT, load_and_clean_trials  # noqa: E402

RAW_MERGED = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
CROWD_GS_746_DIR = (
    HERE.parent / "convergence" / "final_grid" / "full_746"
)
TRACKER_F_746 = HERE.parent / "beat_metrics" / "per_stem_beat_metrics.csv"
OUT_CSV = HERE / "pooled_istc_cv_vs_tracker_F.csv"


def pooled_istc_per_stem() -> pd.DataFrame:
    trials = load_and_clean_trials(RAW_MERGED)
    assert trials["participant_id"].nunique() == 746, (
        f"Expected 746 unique participants, got {trials['participant_id'].nunique()}"
    )
    rows = []
    for stem, sub in trials.groupby("stem", sort=True):
        taps_list = list(sub["clean_taps"])
        rows.append(
            {
                "stem": stem,
                "n_participants": len(taps_list),
                "pooled_istc": istc_for_group(taps_list),
            }
        )
    return pd.DataFrame(rows)


def pooled_crowd_gs_cv() -> pd.DataFrame:
    rows = []
    for f in sorted(CROWD_GS_746_DIR.glob("*_crowd_gs.txt")):
        stem = f.name.replace("_crowd_gs.txt", "")
        beats = np.sort(np.atleast_1d(np.loadtxt(f)))
        if len(beats) < 3:
            cv = np.nan
        else:
            iois = np.diff(beats)
            cv = float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else np.nan
        rows.append({"stem": stem, "pooled_crowd_gs_cv": cv, "n_beats": len(beats)})
    return pd.DataFrame(rows)


def tracker_f_per_stem() -> pd.DataFrame:
    df = pd.read_csv(TRACKER_F_746)
    return (
        df[df["metric"] == "F-measure"]
        .pivot(index="stem", columns="algorithm", values="value")
        .reset_index()
    )


def main() -> None:
    istc_df = pooled_istc_per_stem()
    cv_df = pooled_crowd_gs_cv()
    f_df = tracker_f_per_stem()

    merged = istc_df.merge(cv_df, on="stem").merge(f_df, on="stem")
    merged.to_csv(OUT_CSV, index=False)

    print(f"n excerpts: {len(merged)}, median n_participants: "
          f"{merged['n_participants'].median()}")
    print(f"aggregate madmom F: {merged['madmom'].mean():.4f}  "
          f"aggregate Beat This! F: {merged['beat_this'].mean():.4f}")
    print()

    def report(fcol: str, xcol: str, label: str) -> None:
        d = merged[[fcol, xcol]].dropna()
        rho, p = stats.spearmanr(d[fcol], d[xcol])
        print(f"{label}: rho={rho:+.4f}  p={p:.3g}  n={len(d)}")

    report("madmom", "pooled_istc", "ISTC vs. madmom F     ")
    report("beat_this", "pooled_istc", "ISTC vs. Beat This! F ")
    report("madmom", "pooled_crowd_gs_cv", "CV vs. madmom F       ")
    report("beat_this", "pooled_crowd_gs_cv", "CV vs. Beat This! F   ")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
