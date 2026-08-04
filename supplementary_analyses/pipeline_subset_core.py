#!/usr/bin/env python3
"""Shared exact-pipeline helpers for participant-subset analyses.

The functions imported below are the same Stage-2 cleaning/KDE functions and
Stage-4 crowd-grid implementation used by the canonical GlobalTap pipeline.
This module does not modify the released annotations; it only creates
temporary consensus grids for reliability and panel-comparison analyses.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMPANION_ROOT = Path(__file__).resolve().parents[1]
GLOBALTAP_ROOT = COMPANION_ROOT.parent
OPTIMIZATION_DIR = COMPANION_ROOT / "optimization"
SCRIPTS_DIR = COMPANION_ROOT / "scripts"

sys.path.insert(0, str(COMPANION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import run_pipeline as cascade  # noqa: E402
from optimization_regularize import optimize_beats as _optimize_beats  # noqa: E402
from optimization_regularize import optimize_beats_grid_search  # noqa: E402
from repp_gt_alignment_pipeline import (  # noqa: E402
    clean_taps_trial,
    gaussian_kde,
    kde_peaks,
)


CANONICAL = {
    "duration_s": 30.0,
    "stim_offset_ms": 2000.0,
    "mad_factor": 3.5,
    "kde_bandwidth_ms": 80.0,
    "kde_dt_s": 0.005,
    "peak_prominence": 0.10,
    "peak_distance_s": 0.15,
    "peak_width_s": 0.0,
    "eval_t0_s": 10.0,
    "eval_t1_s": 25.0,
}


@functools.wraps(_optimize_beats)
def _grid_optimize(*args, **kwargs):
    kwargs.setdefault("method", "grid_search")
    return _optimize_beats(*args, **kwargs)


def configure_canonical_pipeline() -> None:
    """Reset mutable cascade settings to the canonical values."""
    cascade.W0 = 0.0
    cascade.W1 = 30.0
    cascade.KW_MS = 80.0
    cascade.KDE_DT = 0.005
    cascade.IOI_MIN = 0.28
    cascade.IOI_MAX = 1.25
    cascade.BIMODAL_THRESHOLD = 0.10
    cascade.HIGH_CV_THRESHOLD = 0.06
    cascade.optimize_beats = _grid_optimize


def _false_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(["", "false", "0", "nan", "<na>", "none"])


def parse_analysis_taps(value: object) -> np.ndarray:
    """Parse REPP-aligned taps and convert them to music-time seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return np.array([], dtype=float)
    if payload.get("failed") is True:
        return np.array([], dtype=float)
    extracted = payload.get("extracted_onsets")
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            return np.array([], dtype=float)
    if not isinstance(extracted, dict):
        return np.array([], dtype=float)
    try:
        taps = np.asarray(extracted.get("resp_onsets_aligned", []), dtype=float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    taps = (taps[np.isfinite(taps)] - CANONICAL["stim_offset_ms"]) / 1000.0
    return np.sort(
        taps[
            (taps >= 0.0)
            & (taps <= CANONICAL["duration_s"])
        ]
    )


def load_and_clean_trials(path: Path) -> pd.DataFrame:
    """Load the controlled raw merge and apply the exact canonical MAD filter."""
    columns = [
        "participant_id",
        "_country_panel",
        "_stem",
        "failed",
        "async_post_trial_failed",
        "complete",
        "analysis",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in columns, low_memory=False)
    mask = df["participant_id"].notna() & df["_stem"].notna()
    if "_country_panel" in df:
        mask &= df["_country_panel"].notna()
    for column in ["failed", "async_post_trial_failed"]:
        if column in df:
            mask &= _false_like_mask(df[column])
    if "complete" in df:
        complete = df["complete"].astype("string").str.strip().str.lower()
        mask &= complete.isin(["true", "1"])
    df = df[mask].copy()
    df["raw_taps"] = df["analysis"].map(parse_analysis_taps)
    df = df[df["raw_taps"].map(len) >= 3].copy()
    if df.duplicated(["participant_id", "_stem"]).any():
        raise ValueError("Duplicate participant/stimulus trials in final merge")

    def clean(taps: np.ndarray) -> np.ndarray:
        values, _ = clean_taps_trial(
            taps,
            0.0,
            CANONICAL["duration_s"],
            None,
            0.30,
            CANONICAL["mad_factor"],
            skip_tempo_gate=True,
            tempo_ref_label="none_mad_only",
        )
        return values

    df["clean_taps"] = df["raw_taps"].map(clean)
    df = df[df["clean_taps"].map(len) >= 2].copy()
    return df.rename(
        columns={
            "_country_panel": "listener_country",
            "_stem": "stem",
        }
    )


def peaks_from_trials(trials: Iterable[np.ndarray]) -> np.ndarray:
    """Pool cleaned trial taps and run the exact canonical KDE/peak extractor."""
    arrays = [np.asarray(taps, dtype=float) for taps in trials if len(taps)]
    if not arrays:
        return np.array([], dtype=float)
    pooled = np.concatenate(arrays)
    grid, density = gaussian_kde(
        pooled,
        0.0,
        CANONICAL["duration_s"],
        CANONICAL["kde_bandwidth_ms"],
        CANONICAL["kde_dt_s"],
    )
    return kde_peaks(
        grid,
        density,
        CANONICAL["kde_dt_s"],
        CANONICAL["peak_prominence"],
        CANONICAL["peak_distance_s"],
        CANONICAL["peak_width_s"],
    )


def consensus_from_trials(stem: str, trials: Iterable[np.ndarray]) -> np.ndarray:
    """Run the canonical Stage-4 crowd path on a participant subset."""
    configure_canonical_pipeline()
    peaks = peaks_from_trials(trials)
    if len(peaks) < 3:
        return np.array([], dtype=float)
    result = cascade.run_cascade(stem, peaks, np.array([], dtype=float), 15.0)
    return np.sort(np.asarray(result.get("crowd_beats", []), dtype=float))


def diagnostics_from_trials(stem: str, trials: Iterable[np.ndarray]) -> dict:
    """Run the canonical Stage-4 crowd path and return the full diagnostics dict
    (including bimodal_score, P_std, cv_std), not just crowd_beats.

    Identical computation to consensus_from_trials -- same peaks_from_trials
    call, same cascade.run_cascade call -- this only changes what is returned.
    """
    configure_canonical_pipeline()
    peaks = peaks_from_trials(trials)
    if len(peaks) < 3:
        return {"stem": stem, "crowd_beats": np.array([], dtype=float)}
    result = cascade.run_cascade(stem, peaks, np.array([], dtype=float), 15.0)
    result["crowd_beats"] = np.sort(np.asarray(result.get("crowd_beats", []), dtype=float))
    return result


def consensus_from_peaks(
    stem: str,
    peaks: np.ndarray,
    optimizer_overrides: dict[str, float | int | bool] | None = None,
    bimodal_threshold: float = 0.10,
) -> np.ndarray:
    """Run Stage 4 with bounded analysis-only optimizer overrides."""
    configure_canonical_pipeline()
    overrides = dict(optimizer_overrides or {})

    def configured_grid(raw_peaks, tg, yk, w0, w1, **_kwargs):
        return optimize_beats_grid_search(
            raw_peaks,
            tg,
            yk,
            w0,
            w1,
            **overrides,
        )

    cascade.optimize_beats = configured_grid
    cascade.BIMODAL_THRESHOLD = bimodal_threshold
    values = np.sort(np.asarray(peaks, dtype=float))
    if len(values) < 3:
        return np.array([], dtype=float)
    result = cascade.run_cascade(
        stem, values, np.array([], dtype=float), 15.0
    )
    return np.sort(np.asarray(result.get("crowd_beats", []), dtype=float))


def trim_beats(beats: np.ndarray, t0: float = 10.0, t1: float = 25.0) -> np.ndarray:
    values = np.sort(np.asarray(beats, dtype=float))
    return values[(values >= t0) & (values <= t1)]


def median_period(beats: np.ndarray) -> float:
    values = np.sort(np.asarray(beats, dtype=float))
    return float(np.median(np.diff(values))) if len(values) >= 2 else np.nan


def classify_period_ratio(ratio: float) -> str:
    if not np.isfinite(ratio) or ratio <= 0:
        return "unclear"
    if 0.35 <= ratio <= 0.65:
        return "double_rate"
    if 0.80 <= ratio <= 1.25:
        return "matched_rate"
    if 1.75 <= ratio <= 2.25:
        return "half_rate"
    return "other"


def matched_offsets(
    reference: np.ndarray, estimate: np.ndarray, tolerance_s: float = 0.07
) -> np.ndarray:
    """Greedy one-to-one signed offsets, estimate minus reference."""
    ref = np.sort(np.asarray(reference, dtype=float))
    est = np.sort(np.asarray(estimate, dtype=float))
    if not len(ref) or not len(est):
        return np.array([], dtype=float)
    distances = np.abs(est[:, None] - ref[None, :])
    offsets: list[float] = []
    used_est: set[int] = set()
    used_ref: set[int] = set()
    while True:
        masked = distances.copy()
        if used_est:
            masked[list(used_est), :] = np.inf
        if used_ref:
            masked[:, list(used_ref)] = np.inf
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        if not np.isfinite(masked[i, j]) or masked[i, j] > tolerance_s:
            break
        used_est.add(int(i))
        used_ref.add(int(j))
        offsets.append(float(est[i] - ref[j]))
    return np.asarray(offsets, dtype=float)


def phase_offset_fraction(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Circular phase offset for grids at the same metrical rate."""
    ref = trim_beats(reference)
    est = trim_beats(estimate)
    period = median_period(ref)
    ratio = median_period(est) / period if period > 0 else np.nan
    if classify_period_ratio(ratio) != "matched_rate" or not len(ref) or not len(est):
        return np.nan
    nearest = np.asarray(
        [value - ref[int(np.argmin(np.abs(ref - value)))] for value in est],
        dtype=float,
    )
    fractions = nearest / period
    vector = np.mean(np.exp(2j * np.pi * fractions))
    return float(np.angle(vector) / (2 * np.pi))


def compare_grids(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | str]:
    """Compare final grids in timing, metrical level, and phase."""
    import mir_eval

    ref = trim_beats(reference)
    est = trim_beats(estimate)
    ref_period = median_period(ref)
    est_period = median_period(est)
    ratio = est_period / ref_period if ref_period > 0 else np.nan
    offsets = matched_offsets(ref, est)
    return {
        "f_measure": (
            float(mir_eval.beat.f_measure(ref, est, f_measure_threshold=0.07))
            if len(ref) and len(est)
            else np.nan
        ),
        "reference_period_s": ref_period,
        "estimate_period_s": est_period,
        "period_ratio": ratio,
        "period_relation": classify_period_ratio(ratio),
        "phase_offset_fraction": phase_offset_fraction(ref, est),
        "median_abs_matched_offset_ms": (
            float(np.median(np.abs(offsets)) * 1000.0) if len(offsets) else np.nan
        ),
    }


def load_released_consensus(directory: Path) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*_crowd_gs.txt")):
        stem = path.name.removesuffix("_crowd_gs.txt")
        outputs[stem] = np.atleast_1d(np.loadtxt(path, dtype=float))
    return outputs
