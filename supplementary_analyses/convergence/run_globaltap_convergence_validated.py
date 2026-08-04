#!/usr/bin/env python3
"""Empirical GlobalTap consensus convergence using the final 746-person merge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_subset_core import (  # noqa: E402
    CANONICAL,
    GLOBALTAP_ROOT,
    clean_taps_trial,
    parse_analysis_taps,
)

DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "kde_peak"


def false_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(["", "false", "0", "nan", "<na>", "none"])


def parse_taps(value: object, duration_s: float) -> np.ndarray:
    taps = parse_analysis_taps(value)
    cleaned, _ = clean_taps_trial(
        taps,
        0.0,
        duration_s,
        None,
        0.30,
        CANONICAL["mad_factor"],
        skip_tempo_gate=True,
        tempo_ref_label="none_mad_only",
    )
    return cleaned


def load_trials(path: Path, duration_s: float) -> dict[str, list[np.ndarray]]:
    cols = [
        "participant_id",
        "_stem",
        "failed",
        "async_post_trial_failed",
        "complete",
        "analysis",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    mask = df["participant_id"].notna() & df["_stem"].notna()
    for col in ["failed", "async_post_trial_failed"]:
        if col in df:
            mask &= false_like_mask(df[col])
    if "complete" in df:
        complete = df["complete"].astype("string").str.strip().str.lower()
        mask &= complete.isin(["true", "1"])
    df = df[mask].copy()
    df["taps"] = df["analysis"].map(lambda x: parse_taps(x, duration_s))
    df = df[df["taps"].map(len) >= 3].copy()
    if df.duplicated(["participant_id", "_stem"]).any():
        raise ValueError("Duplicate participant/stimulus trials in final merge")
    return {
        str(stem): list(sub["taps"])
        for stem, sub in df.groupby("_stem")
    }


def participant_densities(
    trials: list[np.ndarray], duration_s: float, dt_s: float, bandwidth_s: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, duration_s + dt_s, dt_s)
    grid = edges[:-1] + dt_s / 2
    rows = []
    for taps in trials:
        hist, _ = np.histogram(taps, bins=edges)
        hist = hist.astype(float)
        if hist.sum() == 0:
            continue
        hist /= hist.sum()
        rows.append(gaussian_filter1d(hist, bandwidth_s / dt_s, mode="constant"))
    return grid, np.asarray(rows)


def peaks(density: np.ndarray, grid: np.ndarray, distance_s: float, prominence: float) -> np.ndarray:
    if len(density) == 0 or density.max() <= 0:
        return np.array([])
    idx, _ = find_peaks(
        density,
        distance=max(1, int(round(distance_s / (grid[1] - grid[0])))),
        prominence=prominence * density.max(),
    )
    return grid[idx]


def greedy_match(reference: np.ndarray, estimate: np.ndarray, tolerance_s: float) -> list[float]:
    if len(reference) == 0 or len(estimate) == 0:
        return []
    distances = np.abs(estimate[:, None] - reference[None, :])
    offsets = []
    used_e: set[int] = set()
    used_r: set[int] = set()
    while True:
        masked = distances.copy()
        if used_e:
            masked[list(used_e), :] = np.inf
        if used_r:
            masked[:, list(used_r)] = np.inf
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        if not np.isfinite(masked[i, j]) or masked[i, j] > tolerance_s:
            break
        used_e.add(int(i))
        used_r.add(int(j))
        offsets.append(float(estimate[i] - reference[j]))
    return offsets


def compare(
    reference_density: np.ndarray,
    estimate_density: np.ndarray,
    grid: np.ndarray,
    eval_t0: float,
    eval_t1: float,
    peak_distance_s: float,
    prominence: float,
    tolerance_s: float,
) -> dict[str, float]:
    mask = (grid >= eval_t0) & (grid <= eval_t1)
    density_r = (
        float(np.corrcoef(reference_density[mask], estimate_density[mask])[0, 1])
        if np.std(reference_density[mask]) > 0 and np.std(estimate_density[mask]) > 0
        else np.nan
    )
    reference_peaks = peaks(reference_density, grid, peak_distance_s, prominence)
    estimate_peaks = peaks(estimate_density, grid, peak_distance_s, prominence)
    reference_peaks = reference_peaks[
        (reference_peaks >= eval_t0) & (reference_peaks <= eval_t1)
    ]
    estimate_peaks = estimate_peaks[
        (estimate_peaks >= eval_t0) & (estimate_peaks <= eval_t1)
    ]
    offsets = greedy_match(reference_peaks, estimate_peaks, tolerance_s)
    precision = len(offsets) / len(estimate_peaks) if len(estimate_peaks) else np.nan
    recall = len(offsets) / len(reference_peaks) if len(reference_peaks) else np.nan
    f_measure = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else 0.0
    )
    return {
        "density_r": density_r,
        "peak_f_measure": float(f_measure),
        "median_abs_peak_offset_ms": (
            float(np.median(np.abs(offsets)) * 1000.0) if offsets else np.nan
        ),
    }


def bootstrap_stem_ci(
    values: pd.DataFrame, metric: str, rng: np.random.Generator, draws: int = 2000
) -> tuple[float, float]:
    per_stem = values.groupby("stem")[metric].mean().dropna().to_numpy()
    if len(per_stem) < 2:
        return np.nan, np.nan
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(per_stem, size=len(per_stem), replace=True).mean()
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[5, 10, 15, 20, 30, 40, 50, 70, 90])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--eval-t0", type=float, default=10.0)
    parser.add_argument("--eval-t1", type=float, default=25.0)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--bandwidth-ms", type=float, default=80.0)
    parser.add_argument("--peak-distance-ms", type=float, default=150.0)
    parser.add_argument("--peak-prominence", type=float, default=0.10)
    parser.add_argument("--tolerance-ms", type=float, default=70.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    trials = load_trials(args.input, args.duration)
    rows = []
    availability = []
    for stem_i, (stem, stem_trials) in enumerate(sorted(trials.items()), start=1):
        grid, matrix = participant_densities(
            stem_trials,
            args.duration,
            args.dt_ms / 1000.0,
            args.bandwidth_ms / 1000.0,
        )
        n_available = len(matrix)
        availability.append({"stem": stem, "n_available": n_available})
        full_density = matrix.mean(axis=0)
        for n in args.n_grid:
            if n > n_available:
                continue
            for rep in range(args.reps):
                selected = rng.choice(n_available, size=n, replace=False)
                sample_density = matrix[selected].mean(axis=0)
                metrics = compare(
                    full_density,
                    sample_density,
                    grid,
                    args.eval_t0,
                    args.eval_t1,
                    args.peak_distance_ms / 1000.0,
                    args.peak_prominence,
                    args.tolerance_ms / 1000.0,
                )
                rows.append(
                    {
                        "comparison": "subsample_vs_full",
                        "stem": stem,
                        "n_tappers": n,
                        "rep": rep,
                        "n_available": n_available,
                        **metrics,
                    }
                )
                if 2 * n <= n_available:
                    disjoint = rng.choice(n_available, size=2 * n, replace=False)
                    a = matrix[disjoint[:n]].mean(axis=0)
                    b = matrix[disjoint[n:]].mean(axis=0)
                    metrics = compare(
                        a,
                        b,
                        grid,
                        args.eval_t0,
                        args.eval_t1,
                        args.peak_distance_ms / 1000.0,
                        args.peak_prominence,
                        args.tolerance_ms / 1000.0,
                    )
                    rows.append(
                        {
                            "comparison": "independent_n_vs_n",
                            "stem": stem,
                            "n_tappers": n,
                            "rep": rep,
                            "n_available": n_available,
                            **metrics,
                        }
                    )
        if stem_i % 25 == 0:
            print(f"processed {stem_i}/{len(trials)} stems", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "per_stem_repetition_metrics.csv.gz", index=False)
    availability_df = pd.DataFrame(availability)
    availability_df.to_csv(args.out_dir / "tappers_available_by_stem.csv", index=False)

    summary_rows = []
    for (comparison, n), sub in results.groupby(["comparison", "n_tappers"]):
        row = {
            "comparison": comparison,
            "n_tappers": n,
            "n_stems": int(sub["stem"].nunique()),
        }
        for metric in ["density_r", "peak_f_measure", "median_abs_peak_offset_ms"]:
            row[f"mean_{metric}"] = float(sub.groupby("stem")[metric].mean().mean())
            row[f"median_{metric}"] = float(sub.groupby("stem")[metric].mean().median())
            low, high = bootstrap_stem_ci(sub, metric, rng)
            row[f"mean_{metric}_ci_low"] = low
            row[f"mean_{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["comparison", "n_tappers"])
    summary.to_csv(args.out_dir / "summary_by_n_tappers.csv", index=False)

    primary = summary[summary["comparison"] == "subsample_vs_full"]
    knee_f90 = primary.loc[primary["mean_peak_f_measure"] >= 0.90, "n_tappers"]
    knee_r90 = primary.loc[primary["mean_density_r"] >= 0.90, "n_tappers"]
    knee_f90 = int(knee_f90.min()) if len(knee_f90) else None
    knee_r90 = int(knee_r90.min()) if len(knee_r90) else None

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for comparison, style in [
        ("subsample_vs_full", "-o"),
        ("independent_n_vs_n", "--s"),
    ]:
        sub = summary[summary["comparison"] == comparison]
        label = comparison.replace("_", " ")
        axes[0].plot(sub["n_tappers"], sub["mean_density_r"], style, label=label)
        axes[1].plot(sub["n_tappers"], sub["mean_peak_f_measure"], style, label=label)
    axes[0].axhline(0.90, color="0.5", linestyle=":", linewidth=0.8)
    axes[1].axhline(0.90, color="0.5", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Mean KDE correlation")
    axes[1].set_ylabel("Mean peak F-measure")
    for axis in axes:
        axis.set_xlabel("Tappers per consensus")
        axis.set_ylim(0, 1.02)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "convergence_curve.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "convergence_curve.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# GlobalTap consensus convergence",
        "",
        f"- Stems: {len(availability_df)}.",
        f"- Median usable tappers per stem: {availability_df['n_available'].median():.0f} "
        f"(range {availability_df['n_available'].min()}--{availability_df['n_available'].max()}).",
        "- Each tapper contributes a normalized smoothed tap density, so prolific "
        "tappers do not receive more weight.",
        "- The primary curve compares random N-tapper consensuses with the full "
        "consensus. Because those subsamples overlap with the full pool, which "
        "can inflate agreement, an independent-groups curve is also reported.",
        "- Confidence intervals bootstrap stems, not repeated subsamples.",
        f"- Smallest tested N with mean peak F-measure >= .90: {knee_f90}.",
        f"- Smallest tested N with mean KDE correlation >= .90: {knee_r90}.",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
