#!/usr/bin/env python3
"""Rebuild Stage-2 KDE peaks from the released per-tap tables in ``data/taps/``.

This runs from a fresh clone: it needs only the released taps, not the audio or
the raw session exports. Cleaning, kernel density estimation, and peak
picking are imported from the Stage-2 module, so the code path is the one used
for the paper.

    python scripts/kde_peaks_from_taps.py --dataset globalmood200 --verify

With ``--verify`` the recomputed peaks are compared against the included
``data/stage_inputs/<dataset>/cleaned_kde_peaks.csv`` file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from repp_gt_alignment_pipeline import (  # noqa: E402
    clean_taps_trial,
    gaussian_kde,
    kde_peaks,
)

# Stage-2 settings fixed for the paper (config/*.json, METHOD_PARAMETERS.md).
WINDOW = (0.0, 30.0)
KERNEL_MS = 80.0
GRID_DT = 0.005
PROMINENCE = 0.10
MIN_DISTANCE_S = 0.15
MAD_FACTOR = 3.5
PEAK_DECIMALS = 3  # the released file stores peak times to the millisecond


def stem_peaks(trials: pd.core.groupby.DataFrameGroupBy) -> np.ndarray:
    pool = []
    for _, trial in trials:
        cleaned, _ = clean_taps_trial(
            trial["tap_time_s"].to_numpy(dtype=float),
            WINDOW[0],
            WINDOW[1],
            None,
            0.0,
            MAD_FACTOR,
            skip_tempo_gate=True,
        )
        if cleaned.size:
            pool.append(cleaned)
    if not pool:
        return np.array([], dtype=float)
    grid, density = gaussian_kde(
        np.concatenate(pool), WINDOW[0], WINDOW[1], KERNEL_MS, GRID_DT
    )
    peaks = kde_peaks(grid, density, GRID_DT, PROMINENCE, MIN_DISTANCE_S, 0.0)
    return np.round(np.sort(peaks), PEAK_DECIMALS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="globalmood200", choices=["globalmood200"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    taps = pd.read_csv(ROOT / f"data/taps/{args.dataset}_taps.csv.gz")
    rows = []
    computed: dict[str, np.ndarray] = {}
    for stem, sub in taps.groupby("stem_id", sort=True):
        peaks = stem_peaks(sub.groupby("participant", sort=False))
        computed[stem] = peaks
        rows.extend(
            {
                "stimulus_id": stem,
                "peak_index": i,
                "peak_time_s": t,
                "window_t0": WINDOW[0],
                "window_t1": WINDOW[1],
                "kw_ms": KERNEL_MS,
                "kde_dt": GRID_DT,
                "peak_prominence": PROMINENCE,
                "peak_distance_s": MIN_DISTANCE_S,
                "peak_width_s": 0.0,
                "cleaning": "mad_only",
            }
            for i, t in enumerate(peaks)
        )
    out = args.out or ROOT / f"outputs/{args.dataset}_kde_peaks_from_taps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(computed)} excerpts, {len(rows):,} peaks -> {out}")

    if not args.verify:
        return
    published = pd.read_csv(
        ROOT / f"data/stage_inputs/{args.dataset}/cleaned_kde_peaks.csv"
    )
    reference = {
        stem: np.sort(group["peak_time_s"].to_numpy(dtype=float))
        for stem, group in published.groupby("stimulus_id")
    }
    identical = same_count = differing = 0
    for stem, peaks in computed.items():
        want = reference.get(stem)
        if want is None or peaks.size != want.size:
            differing += 1
            continue
        gap = float(np.max(np.abs(peaks - want))) if peaks.size else 0.0
        if gap <= 0.001:
            identical += 1
        elif gap <= 0.0051:
            same_count += 1
        else:
            differing += 1
    print(f"  peak times equal to released file  : {identical}")
    print(f"  one 5 ms grid step from released    : {same_count}")
    print(f"  differing peak count or > one step : {differing}")


if __name__ == "__main__":
    main()
