#!/usr/bin/env python3
"""Final-grid reliability using the GlobalTap pipeline.

This analysis complements KDE/peak convergence by rerunning Stage 2
and Stage 4 on participant subsets.  It never overwrites released annotations.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_subset_core import (
    GLOBALTAP_ROOT,
    compare_grids,
    consensus_from_trials,
    load_and_clean_trials,
    load_released_consensus,
)


DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "final_grid"
DEFAULT_RELEASED = (
    GLOBALTAP_ROOT / "companion_repo/outputs/globalmood200/crowd_gs"
)

_TRIALS: dict[str, list[np.ndarray]] = {}
_FULL: dict[str, np.ndarray] = {}


def _run_consensus_task(task: tuple[str, str, int, int, list[int]]):
    stem, group, n_tappers, rep, indices = task
    beats = consensus_from_trials(stem, [_TRIALS[stem][i] for i in indices])
    metrics = compare_grids(_FULL[stem], beats)
    return {
        "stem": stem,
        "group": group,
        "n_tappers": n_tappers,
        "rep": rep,
        "n_beats": len(beats),
        "beats": beats,
        **metrics,
    }


def _run_full_task(stem: str):
    beats = consensus_from_trials(stem, _TRIALS[stem])
    return stem, beats


def _pool(workers: int):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
    )


def generate_final_full(
    out_dir: Path,
    workers: int,
    released_dir: Path,
) -> dict[str, np.ndarray]:
    full_dir = out_dir / "full_746"
    full_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(_TRIALS)
    outputs = load_released_consensus(full_dir)
    if set(outputs) != set(stems):
        with _pool(workers) as executor:
            rows = list(executor.map(_run_full_task, stems, chunksize=1))
        outputs = {}
        for stem, beats in rows:
            outputs[stem] = beats
            np.savetxt(full_dir / f"{stem}_crowd_gs.txt", beats, fmt="%.9f")

    released = load_released_consensus(released_dir)
    comparisons = []
    for stem in sorted(set(outputs) & set(released)):
        comparisons.append(
            {
                "stem": stem,
                "n_final_trials": len(_TRIALS[stem]),
                **compare_grids(released[stem], outputs[stem]),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        out_dir / "final_746_vs_released_consensus.csv", index=False
    )
    return outputs


def run_reliability(
    out_dir: Path,
    n_grid: list[int],
    reps: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []
    pair_rows: list[dict] = []

    for n_tappers in n_grid:
        tasks = []
        for stem in sorted(_TRIALS):
            available = len(_TRIALS[stem])
            if 2 * n_tappers > available:
                continue
            for rep in range(reps):
                selected = rng.choice(
                    available, size=2 * n_tappers, replace=False
                ).tolist()
                tasks.append((stem, "A", n_tappers, rep, selected[:n_tappers]))
                tasks.append((stem, "B", n_tappers, rep, selected[n_tappers:]))
        with _pool(workers) as executor:
            results = list(executor.map(_run_consensus_task, tasks, chunksize=1))

        lookup: dict[tuple[str, int], dict[str, dict]] = {}
        for result in results:
            beats = result.pop("beats")
            key = (result["stem"], result["rep"])
            lookup.setdefault(key, {})[result["group"]] = {
                "beats": beats,
                **result,
            }
            all_rows.append(result)

        for (stem, rep), groups in lookup.items():
            if {"A", "B"} - set(groups):
                continue
            pair_rows.append(
                {
                    "comparison": "independent_n_vs_n",
                    "stem": stem,
                    "n_tappers": n_tappers,
                    "rep": rep,
                    **compare_grids(groups["A"]["beats"], groups["B"]["beats"]),
                }
            )
        pd.DataFrame(all_rows).to_csv(
            out_dir / "subset_vs_full_metrics.csv.gz", index=False
        )
        pd.DataFrame(pair_rows).to_csv(
            out_dir / "independent_subset_metrics.csv.gz", index=False
        )
        print(f"completed N={n_tappers}: {len(results)} subset grids", flush=True)

    subset = pd.DataFrame(all_rows)
    subset["comparison"] = "subsample_vs_full"
    independent = pd.DataFrame(pair_rows)
    combined = pd.concat([subset, independent], ignore_index=True, sort=False)
    summary_rows = []
    for (comparison, n_tappers), sub in combined.groupby(
        ["comparison", "n_tappers"]
    ):
        by_stem = sub.groupby("stem")
        summary_rows.append(
            {
                "comparison": comparison,
                "n_tappers": int(n_tappers),
                "n_stems": int(sub["stem"].nunique()),
                "n_repetitions": int(sub["rep"].nunique()),
                "mean_f_measure": float(by_stem["f_measure"].mean().mean()),
                "median_f_measure": float(by_stem["f_measure"].mean().median()),
                "matched_rate_proportion": float(
                    (sub["period_relation"] == "matched_rate").mean()
                ),
                "median_abs_phase_offset_fraction": float(
                    sub["phase_offset_fraction"].abs().median()
                ),
                "median_abs_matched_offset_ms": float(
                    sub["median_abs_matched_offset_ms"].median()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["comparison", "n_tappers"]
    )
    summary.to_csv(out_dir / "final_grid_summary_by_n.csv", index=False)
    return summary


def write_summary(
    out_dir: Path,
    summary: pd.DataFrame,
    full: dict[str, np.ndarray],
) -> None:
    lines = [
        "# Final consensus-grid reliability",
        "",
        f"- Excerpts: {len(full)}.",
        "- Participant subsets use the paper's per-trial MAD cleaning, 5-ms KDE "
        "and peak extraction, and Stage-4 optimizer.",
        "- At each N, one disjoint A/B split was evaluated per excerpt. The "
        "KDE/peak convergence analysis tests more repeated subsets.",
        "",
        "## Subset results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.comparison}, N={row.n_tappers}: "
            f"mean F={row.mean_f_measure:.3f}, "
            f"matched-rate={row.matched_rate_proportion:.3f}, "
            f"median |phase|={row.median_abs_phase_offset_fraction:.3f} cycles."
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--released-dir", type=Path, default=DEFAULT_RELEASED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--n-grid", type=int, nargs="+", default=[10, 20, 30, 50]
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional stem subset for smoke tests.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_and_clean_trials(args.input)
    global _TRIALS, _FULL
    _TRIALS = {
        str(stem): list(sub["clean_taps"])
        for stem, sub in trials.groupby("stem", sort=True)
    }
    if args.stems:
        _TRIALS = {stem: _TRIALS[stem] for stem in args.stems if stem in _TRIALS}
    _FULL = generate_final_full(args.out_dir, args.workers, args.released_dir)
    summary = run_reliability(
        args.out_dir,
        args.n_grid,
        args.reps,
        args.seed,
        args.workers,
    )
    write_summary(args.out_dir, summary, _FULL)
    print(args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
