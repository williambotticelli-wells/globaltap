#!/usr/bin/env python3
"""Bounded sensitivity analysis for active final-grid optimizer heuristics."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUPPLEMENTARY = HERE.parent
COMPANION = SUPPLEMENTARY.parent
GLOBALTAP = COMPANION.parent
DATA_ROOT = GLOBALTAP.parent
sys.path.insert(0, str(SUPPLEMENTARY))
sys.path.insert(0, str(COMPANION))

from lib.gt_loader import load_gt_for_stem  # noqa: E402
from pipeline_subset_core import (  # noqa: E402
    classify_period_ratio,
    compare_grids,
    consensus_from_peaks,
    load_released_consensus,
    median_period,
    trim_beats,
)


DEFAULT_PEAKS = COMPANION / "data/stage_inputs/canonical240/cleaned_kde_peaks.csv"
DEFAULT_CANONICAL = GLOBALTAP / "pipeline/outputs/canonical240/crowd_gs"
DEFAULT_GT_MAP = COMPANION / "config/gt_source_map_canonical240.json"
DEFAULT_OUT = HERE / "results"

VARIANTS = {
    "T_steps_80": ({"T_steps": 80}, 0.10),
    "T_steps_160": ({"T_steps": 160}, 0.10),
    "phi_steps_160": ({"phi_steps": 160}, 0.10),
    "phi_steps_320": ({"phi_steps": 320}, 0.10),
    "peak_weight_025": ({"peak_weight": 0.25}, 0.10),
    "peak_weight_075": ({"peak_weight": 0.75}, 0.10),
    "snap_weight_025": ({"snap_weight": 0.25}, 0.10),
    "snap_weight_065": ({"snap_weight": 0.65}, 0.10),
    "snap_max_008": ({"snap_max_frac": 0.08}, 0.10),
    "snap_max_016": ({"snap_max_frac": 0.16}, 0.10),
    "bimodal_008": ({}, 0.08),
    "bimodal_012": ({}, 0.12),
    "ioi_min_023": ({"ioi_min_s": 0.23}, 0.10),
    "ioi_min_033": ({"ioi_min_s": 0.33}, 0.10),
}

_PEAKS: dict[str, np.ndarray] = {}
_CANONICAL: dict[str, np.ndarray] = {}
_GT: dict[str, np.ndarray] = {}


def f_measure(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = trim_beats(reference)
    estimate = trim_beats(estimate)
    if len(reference) < 2 or len(estimate) < 2:
        return np.nan
    return float(
        mir_eval.beat.f_measure(
            reference, estimate, f_measure_threshold=0.07
        )
    )


def _run(task: tuple[str, str]):
    stem, variant = task
    overrides, threshold = VARIANTS[variant]
    beats = consensus_from_peaks(
        stem,
        _PEAKS[stem],
        optimizer_overrides=overrides,
        bimodal_threshold=threshold,
    )
    canonical = _CANONICAL[stem]
    gt = _GT.get(stem, np.array([], dtype=float))
    metrics = compare_grids(canonical, beats)
    gt_period = median_period(trim_beats(gt))
    estimate_period = median_period(trim_beats(beats))
    gt_ratio = estimate_period / gt_period if gt_period > 0 else np.nan
    return {
        "stem": stem,
        "variant": variant,
        "f_vs_canonical": metrics["f_measure"],
        "period_relation_vs_canonical": metrics["period_relation"],
        "phase_offset_vs_canonical": metrics["phase_offset_fraction"],
        "f_vs_gt": f_measure(gt, beats),
        "canonical_f_vs_gt": f_measure(gt, canonical),
        "period_relation_vs_gt": classify_period_ratio(gt_ratio),
        "n_beats": len(beats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--gt-map", type=Path, default=DEFAULT_GT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--variants", nargs="*", default=list(VARIANTS)
    )
    parser.add_argument(
        "--stems", nargs="*", default=None, help="Optional smoke-test subset."
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    peaks = pd.read_csv(args.peaks, low_memory=False)
    global _PEAKS, _CANONICAL, _GT
    _PEAKS = {
        str(stem): np.sort(sub["peak_time_s"].to_numpy(float))
        for stem, sub in peaks.groupby("stimulus_id")
    }
    _CANONICAL = load_released_consensus(args.canonical_dir)
    gt_map = json.loads(args.gt_map.read_text())
    for entry in gt_map.values():
        path = entry.get("path")
        if isinstance(path, str) and path.startswith("${PROJECT_ROOT}/data/"):
            entry["path"] = path.removeprefix("${PROJECT_ROOT}/data/")
    _GT = {}
    for stem in sorted(_PEAKS):
        loaded = load_gt_for_stem(stem, gt_map, DATA_ROOT)
        if loaded is not None and len(loaded):
            _GT[stem] = np.sort(np.asarray(loaded[:, 0], dtype=float))

    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    stems = sorted(set(_PEAKS) & set(_CANONICAL))
    if args.stems:
        stems = [stem for stem in args.stems if stem in stems]
    tasks = [
        (stem, variant)
        for variant in args.variants
        for stem in stems
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
    ) as executor:
        rows = list(executor.map(_run, tasks, chunksize=1))
    per_stem = pd.DataFrame(rows)
    per_stem.to_csv(args.out_dir / "optimizer_sensitivity_per_stem.csv.gz", index=False)

    summary_rows = []
    for variant, sub in per_stem.groupby("variant"):
        summary_rows.append(
            {
                "variant": variant,
                "n_stems": sub["stem"].nunique(),
                "mean_f_vs_canonical": sub["f_vs_canonical"].mean(),
                "matched_rate_vs_canonical": (
                    sub["period_relation_vs_canonical"] == "matched_rate"
                ).mean(),
                "median_abs_phase_vs_canonical": sub[
                    "phase_offset_vs_canonical"
                ].abs().median(),
                "mean_f_vs_gt": sub["f_vs_gt"].mean(),
                "mean_canonical_f_vs_gt": sub["canonical_f_vs_gt"].mean(),
                "mean_delta_f_vs_gt": (
                    sub["f_vs_gt"] - sub["canonical_f_vs_gt"]
                ).mean(),
                "matched_rate_vs_gt": (
                    sub["period_relation_vs_gt"] == "matched_rate"
                ).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("variant")
    summary.to_csv(args.out_dir / "optimizer_sensitivity_summary.csv", index=False)
    lines = [
        "# Final-grid optimizer sensitivity",
        "",
        "Each row changes one parameter from the paper setting while holding "
        "the Stage-2 peaks and all other settings fixed. “Same level” is the "
        "proportion of excerpts that keep the same metrical level as the "
        "paper's grid.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.variant}: F={row.mean_f_vs_canonical:.3f}, "
            f"same level={row.matched_rate_vs_canonical:.3f}, "
            f"change in F against expert references="
            f"{row.mean_delta_f_vs_gt:+.3f}."
        )
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
