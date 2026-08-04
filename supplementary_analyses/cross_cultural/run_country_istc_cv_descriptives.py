#!/usr/bin/env python3
"""Per-country ISTC and raw-tapping descriptives.

Only ``istc_for_group`` below is used elsewhere in this repository (imported
by ``../istc_cv_tracker_correlations/``, which backs the paper's pooled
ISTC/CV-vs-tracker-agreement result). ISTC and the raw descriptives use only
the Stage-2 MAD-cleaned per-trial taps (already loaded in seconds by
``load_and_clean_trials``); no optimizer call is required.

This script can also be run standalone for a per-country ISTC/tapping
descriptive breakdown, which the paper does not otherwise report.

ISTC is reported two ways per country, exactly analogous to how the paper
already reports it "within-corpus" because raw counts are bounded by panel
size:
  - raw_istc: the paper's exact bin-counting definition (max unique
    participants per second, averaged over 0-30s), computed using only that
    country's own participants.
  - istc_fraction: raw_istc divided by that country's own available N for
    the excerpt, i.e. the *proportion* of the country's own panel that
    agreed -- comparable across countries of different sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pipeline_subset_core import GLOBALTAP_ROOT, load_and_clean_trials  # noqa: E402


DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
DEFAULT_OUT = HERE / "country_istc_cv_descriptives"


def istc_for_group(trial_taps: list[np.ndarray]) -> float:
    """Paper's exact bin-counting ISTC definition, restricted to one group."""
    bin_ms, step_ms = 100.0, 50.0
    t0_ms, t1_ms = 0.0, 30000.0
    centers_ms = np.arange(t0_ms, t1_ms + step_ms, step_ms)
    half = bin_ms / 2.0
    centers_s = centers_ms / 1000.0
    sec_edges = np.arange(0.0, 30.0 + 1e-9, 1.0)

    pooled, pids = [], []
    for pid, taps in enumerate(trial_taps):
        for t in taps:
            pooled.append(t * 1000.0)
            pids.append(pid)
    if not pooled:
        return np.nan
    pooled = np.array(pooled)
    pids = np.array(pids)

    counts = np.zeros(len(centers_ms))
    for i, c in enumerate(centers_ms):
        mask = (pooled >= c - half) & (pooled <= c + half)
        if np.any(mask):
            counts[i] = np.unique(pids[mask]).size

    maxes = []
    for k in range(len(sec_edges) - 1):
        lo, hi = sec_edges[k], sec_edges[k + 1]
        mask = (centers_s >= lo) & (centers_s < hi)
        if np.any(mask):
            maxes.append(counts[mask].max())
    return float(np.mean(maxes)) if maxes else np.nan


def raw_trial_descriptives(taps: np.ndarray) -> dict[str, float]:
    n = len(taps)
    if n < 2:
        return {"n_taps": n, "median_ioi_s": np.nan, "ioi_cv": np.nan}
    iois = np.diff(taps)
    return {
        "n_taps": n,
        "median_ioi_s": float(np.median(iois)),
        "ioi_cv": float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_and_clean_trials(args.input)
    countries = sorted(trials["listener_country"].unique())

    # --- ISTC per country, per excerpt ---
    istc_rows = []
    for (stem, country), sub in trials.groupby(["stem", "listener_country"], sort=True):
        taps_list = list(sub["clean_taps"])
        n_participants = len(taps_list)
        raw = istc_for_group(taps_list)
        istc_rows.append(
            {
                "stem": stem,
                "listener_country": country,
                "n_participants": n_participants,
                "raw_istc": raw,
                "istc_fraction": raw / n_participants if n_participants else np.nan,
            }
        )
    istc_df = pd.DataFrame(istc_rows)
    istc_df.to_csv(args.out_dir / "country_istc_per_stem.csv.gz", index=False)

    istc_summary = (
        istc_df.groupby("listener_country")
        .agg(
            n_stems=("stem", "nunique"),
            median_n_participants=("n_participants", "median"),
            mean_raw_istc=("raw_istc", "mean"),
            mean_istc_fraction=("istc_fraction", "mean"),
        )
        .reset_index()
        .sort_values("mean_istc_fraction", ascending=False)
    )
    istc_summary.to_csv(args.out_dir / "country_istc_summary.csv", index=False)

    # --- Raw per-trial tapping descriptives per country ---
    descr_rows = []
    for _, row in trials.iterrows():
        d = raw_trial_descriptives(row["clean_taps"])
        d["stem"] = row["stem"]
        d["listener_country"] = row["listener_country"]
        d["participant_id"] = row["participant_id"]
        descr_rows.append(d)
    descr_df = pd.DataFrame(descr_rows)
    descr_df.to_csv(args.out_dir / "country_trial_descriptives.csv.gz", index=False)

    descr_summary = (
        descr_df.groupby("listener_country")
        .agg(
            n_trials=("stem", "size"),
            n_participants=("participant_id", "nunique"),
            mean_n_taps=("n_taps", "mean"),
            median_ioi_s=("median_ioi_s", "median"),
            mean_ioi_cv=("ioi_cv", "mean"),
        )
        .reset_index()
        .sort_values("listener_country")
    )
    descr_summary.to_csv(args.out_dir / "country_trial_descriptives_summary.csv", index=False)

    print("\nISTC per country (fraction of own panel agreeing, higher = more coherent):")
    print(istc_summary.to_string(index=False))
    print("\nRaw trial descriptives per country:")
    print(descr_summary.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
