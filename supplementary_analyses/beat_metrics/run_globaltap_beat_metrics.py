#!/usr/bin/env python3
"""Evaluate algorithm agreement with GlobalTap under standard beat metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mir_eval import beat
from scipy.stats import false_discovery_control, wilcoxon


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CROWD = REPO_ROOT / "data/annotations/globalmood200"
DEFAULT_MADMOM = REPO_ROOT / "data/stage_inputs/globalmood200/algorithm_beats.csv"
DEFAULT_BEAT_THIS = (
    REPO_ROOT / "data/stage_inputs/globalmood200/beat_this_algorithm_beats.csv"
)
DEFAULT_META = REPO_ROOT / "data/per_stem_metrics.csv"
DEFAULT_OUT = HERE


def algorithm_beats(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    df["stem"] = df["wav"].astype(str).str.removesuffix(".wav")
    return {
        stem: np.sort(sub["beat_time_s"].dropna().to_numpy(float))
        for stem, sub in df.groupby("stem")
    }


def load_reference(path: Path, t0: float, t1: float) -> dict[str, np.ndarray]:
    refs = {}
    for pattern in ("*.beats", "*_crowd_gs.txt"):
        for file in sorted(path.glob(pattern)):
            stem = file.name.removesuffix(".beats").removesuffix("_crowd_gs.txt")
            values = np.atleast_1d(np.loadtxt(file, dtype=float))
            refs.setdefault(stem, values[(values >= t0) & (values <= t1)])
    return refs


def evaluate_one(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    if len(reference) < 2 or len(estimate) < 2:
        return {}
    return {key: float(value) for key, value in beat.evaluate(reference, estimate).items()}


def paired_tests(per_stem: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in sorted(per_stem["metric"].unique()):
        wide = per_stem[per_stem["metric"] == metric].pivot(
            index="stem", columns="algorithm", values="value"
        ).dropna()
        if {"madmom", "beat_this"} - set(wide.columns) or len(wide) < 5:
            continue
        diff = wide["beat_this"] - wide["madmom"]
        stat, p = wilcoxon(diff, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "n_stems": len(diff),
                "madmom_mean": float(wide["madmom"].mean()),
                "beat_this_mean": float(wide["beat_this"].mean()),
                "beat_this_minus_madmom_mean": float(diff.mean()),
                "beat_this_minus_madmom_median": float(diff.median()),
                "wilcoxon_stat": float(stat),
                "p_two_sided": float(p),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q"] = false_discovery_control(result["p_two_sided"].to_numpy(), method="bh")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowd-dir", type=Path, default=DEFAULT_CROWD)
    parser.add_argument("--madmom", type=Path, default=DEFAULT_MADMOM)
    parser.add_argument("--beat-this", type=Path, default=DEFAULT_BEAT_THIS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--t0", type=float, default=10.0)
    parser.add_argument("--t1", type=float, default=25.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    references = load_reference(args.crowd_dir, args.t0, args.t1)
    algorithms = {
        "madmom": algorithm_beats(args.madmom),
        "beat_this": algorithm_beats(args.beat_this),
    }
    rows = []
    for stem, reference in references.items():
        for algorithm, beats_by_stem in algorithms.items():
            estimate = beats_by_stem.get(stem, np.array([]))
            estimate = estimate[(estimate >= args.t0) & (estimate <= args.t1)]
            for metric, value in evaluate_one(reference, estimate).items():
                rows.append(
                    {
                        "stem": stem,
                        "algorithm": algorithm,
                        "metric": metric,
                        "value": value,
                        "n_crowd_beats": len(reference),
                        "n_algorithm_beats": len(estimate),
                    }
                )
    per_stem = pd.DataFrame(rows)
    if args.metadata.exists():
        meta = pd.read_csv(args.metadata, low_memory=False)
        keep = [c for c in ["_stem", "source_country", "group_b5", "ISTC"] if c in meta]
        per_stem = per_stem.merge(
            meta[keep].drop_duplicates("_stem").rename(columns={"_stem": "stem"}),
            on="stem",
            how="left",
        )
    per_stem.to_csv(args.out_dir / "per_stem_beat_metrics.csv", index=False)

    tests = paired_tests(per_stem)
    tests.to_csv(args.out_dir / "paired_algorithm_metric_tests.csv", index=False)
    f_rows = per_stem[per_stem["metric"] == "F-measure"]
    paired_stems = set(
        f_rows.groupby("stem")["algorithm"].nunique().loc[lambda x: x == 2].index
    )
    exclusions = pd.DataFrame(
        [
            {
                "stem": stem,
                "reason": "fewer than two Beat This! beats in the 10--25 s evaluation window",
            }
            for stem in sorted(set(references) - paired_stems)
        ]
    )
    exclusions.to_csv(args.out_dir / "paired_metric_exclusions.csv", index=False)
    key_metrics = [
        "F-measure",
        "Correct Metric Level Continuous",
        "Correct Metric Level Total",
        "Any Metric Level Continuous",
        "Any Metric Level Total",
        "Information gain",
    ]
    lines = [
        "# Beat metrics beyond F-measure",
        "",
        f"- Evaluation window: {args.t0:g}--{args.t1:g} s.",
        "- The tapping-derived GlobalTap consensus is the `mir_eval` reference "
        "and each tracker's beat sequence is the estimate.",
        f"- GlobalTap consensus stems: {len(references)}.",
        f"- Paired algorithm comparisons include {len(paired_stems)} stems. "
        f"{len(exclusions)} are excluded because Beat This! returned fewer than two "
        "beats in the evaluation window.",
        "- Metrics computed with `mir_eval` 0.8.2.",
        "- Paired two-sided Wilcoxon tests compare algorithms stem by stem. "
        "BH-FDR is applied across all metric comparisons.",
        "",
        "## Key results",
        "",
        "`delta` below is Beat This! minus madmom.",
        "",
    ]
    for row in tests[tests["metric"].isin(key_metrics)].itertuples(index=False):
        lines.append(
            f"- {row.metric}: madmom={row.madmom_mean:.3f}, "
            f"Beat This!={row.beat_this_mean:.3f}, "
            f"delta={row.beat_this_minus_madmom_mean:+.3f}, q={row.fdr_q:.4g}."
        )
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
