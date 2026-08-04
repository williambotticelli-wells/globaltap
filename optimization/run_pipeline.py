#!/usr/bin/env python3
"""
Auto-detection beat-grid optimizer used by the canonical crowd_gs pipeline.

Exposed entry point:

    run_cascade(stem, peaks, mm_beats, istc_val) -> dict

For each stem it runs a decision cascade that automatically detects and
corrects two failure modes:

  1. Phase bimodality   -> mono-modal split
  2. Irregular spacing  -> grid extension on the chosen result

The function returns a diagnostics dict whose ``crowd_beats`` entry is the
canonical optimized beat sequence written by ``pipeline/4_run_crowd_gs.py``;
this module is imported by the Stage 4 driver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from optimization_regularize import (
    optimize_beats,
    build_kde_interpolant,
    _confident_grid_extension,
    _peak_coverage,
)
from detect_multimodal import bimodal_score, split_peaks_bimodal


# ── Config ──────────────────────────────────────────────────────────────────
# These constants are overwritten by pipeline/4_run_crowd_gs.py from the
# canonical config; the defaults below match config/canonical240.json.

KW_MS = 80.0
KDE_DT = 0.005
DEFAULT_LAMBDAS = (2.0, 0.3, 0.3)
BALANCED_LAMBDAS = (1.0, 1.0, 1.0)
W0, W1 = 0.0, 30.0
IOI_MIN = 0.28
IOI_MAX = 1.25

BIMODAL_THRESHOLD = 0.10
HIGH_CV_THRESHOLD = 0.06
FALLBACK_CV_THRESHOLD = 0.15


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_kde(peaks: np.ndarray, w0: float, w1: float):
    tg = np.arange(w0, w1 + KDE_DT, KDE_DT)
    bw = KW_MS / 1000.0
    yk = np.zeros_like(tg)
    p_win = peaks[(peaks >= w0) & (peaks <= w1)]
    for pk in p_win:
        yk += np.exp(-0.5 * ((tg - pk) / bw) ** 2)
    n = max(1, len(p_win))
    yk /= bw * np.sqrt(2 * np.pi) * n
    return tg, yk


def ioi_cv(beats: np.ndarray) -> float:
    if beats.size < 2:
        return float("nan")
    iois = np.diff(beats)
    return float(np.std(iois) / (np.mean(iois) + 1e-12))


def run_opt(peaks, w0, w1, lambdas=DEFAULT_LAMBDAS, max_iter=500):
    p_win = peaks[(peaks >= w0) & (peaks <= w1)]
    if len(p_win) < 3:
        return None
    tg, yk = build_kde(peaks, w0, w1)
    r = optimize_beats(
        p_win, tg, yk, w0, w1, lambdas=lambdas, max_iter=max_iter
    )
    if r.get("beats") is not None and len(r["beats"]) >= 3:
        return r
    return None


def apply_ge(result, peaks, w0, w1):
    if result is None:
        return result
    tg, yk = build_kde(peaks, w0, w1)
    kde_fn = build_kde_interpolant(tg, yk)
    p_win = peaks[(peaks >= w0) & (peaks <= w1)]
    return _confident_grid_extension(
        result.copy(), kde_fn, w0, w1, input_peaks=p_win
    )


def score_result(result, peaks, w0, w1):
    """Composite score: coverage + isochrony + KDE density."""
    if result is None:
        return -1e6
    beats = np.sort(result["beats"])
    b_win = beats[(beats >= w0) & (beats <= w1)]
    p_win = peaks[(peaks >= w0) & (peaks <= w1)]
    if len(b_win) < 3:
        return -1e6

    P = result.get("P", float(np.median(np.diff(b_win))))
    tol = 0.12 * P + 0.03
    coverage = (
        _peak_coverage(b_win, p_win, tol_s=tol) if len(p_win) > 0 else 0.0
    )
    cv = ioi_cv(b_win)
    isochrony = max(0.0, 1.0 - cv) if not np.isnan(cv) else 0.0

    tg, yk = build_kde(peaks, w0, w1)
    kde_fn = build_kde_interpolant(tg, yk)
    mean_kde = float(np.mean([float(kde_fn(xi)) for xi in b_win]))

    return 2.0 * coverage + 1.0 * isochrony + 0.5 * mean_kde


def best_of(
    candidates: List[Tuple[str, Optional[Dict]]], peaks, w0, w1
):
    """Pick the highest-scoring candidate. Returns (label, result, score)."""
    best_label, best_res, best_s = None, None, -1e6
    for label, res in candidates:
        if res is None:
            continue
        s = score_result(res, peaks, w0, w1)
        if s > best_s:
            best_label, best_res, best_s = label, res, s
    return best_label, best_res, best_s


# ── Decision cascade ────────────────────────────────────────────────────────

def run_cascade(
    stem: str,
    peaks: np.ndarray,
    mm_beats: np.ndarray,
    istc_val: float,
) -> Dict:
    """Run the full optimizer cascade for one stem.

    Returns a diagnostics dict whose ``crowd_beats`` entry is the
    canonical optimized beat sequence. ``mm_beats`` and ``istc_val`` are
    accepted for backwards-compatibility with the Stage 4 caller but are
    not used by the canonical crowd_gs path.
    """
    diag = {"stem": stem, "istc": istc_val, "path": [], "scores": {}}

    std_result = run_opt(peaks, W0, W1)
    if std_result is None:
        diag["path"].append("FAIL:no_std_result")
        return {**diag, "crowd_beats": np.array([])}

    std_beats = np.sort(std_result["beats"])
    P_std = std_result["P"]
    cv_std = ioi_cv(std_beats)
    diag["P_std"] = round(P_std, 4)
    diag["cv_std"] = round(cv_std, 5)

    bscore = bimodal_score(peaks, std_beats, min_gap_frac=BIMODAL_THRESHOLD)
    diag["bimodal_score"] = round(bscore, 4)

    chosen_label = "std"
    chosen_result = std_result

    is_bimodal = bscore >= BIMODAL_THRESHOLD
    force_mono = (not is_bimodal) and (cv_std > FALLBACK_CV_THRESHOLD)

    if is_bimodal or force_mono:
        diag["path"].append("bimodal" if is_bimodal else "bimodal_fallback")
        groupA, groupB, was_split, _phase_delta = split_peaks_bimodal(
            peaks, std_beats, min_gap_frac=BIMODAL_THRESHOLD * 0.8
        )
        if was_split and len(groupA) >= 3 and len(groupB) >= 3:
            resA = run_opt(groupA, W0, W1)
            resB = run_opt(groupB, W0, W1)
            mono_label, mono_res, mono_score = best_of(
                [("monoA", resA), ("monoB", resB)], peaks, W0, W1
            )
            std_score = score_result(std_result, peaks, W0, W1)
            diag["scores"]["monoA"] = (
                round(score_result(resA, peaks, W0, W1), 4) if resA else None
            )
            diag["scores"]["monoB"] = (
                round(score_result(resB, peaks, W0, W1), 4) if resB else None
            )
            diag["scores"]["std"] = round(std_score, 4)

            if mono_res is not None and mono_score > std_score:
                chosen_label = mono_label
                chosen_result = mono_res
                diag["path"].append(f"chose_{mono_label}")
            else:
                diag["path"].append("bimodal_no_improvement")

    cv_chosen = ioi_cv(np.sort(chosen_result["beats"]))
    if cv_chosen > HIGH_CV_THRESHOLD:
        diag["path"].append("high_cv_balanced_lambdas")
        balanced_result = run_opt(peaks, W0, W1, lambdas=BALANCED_LAMBDAS)
        if balanced_result is not None:
            bal_score = score_result(balanced_result, peaks, W0, W1)
            cur_score = score_result(chosen_result, peaks, W0, W1)
            diag["scores"]["balanced"] = round(bal_score, 4)
            if bal_score > cur_score:
                chosen_label = "balanced"
                chosen_result = balanced_result
                diag["path"].append("chose_balanced")

    pre_ge_result = chosen_result
    ge_result = apply_ge(chosen_result, peaks, W0, W1)
    ge_activated = (
        ge_result is not None
        and ge_result.get("metadata", {})
                     .get("grid_extension", {})
                     .get("activated", False)
    )
    diag["ge_activated"] = ge_activated

    if ge_activated:
        ge_score = score_result(ge_result, peaks, W0, W1)
        pre_score = score_result(pre_ge_result, peaks, W0, W1)
        diag["scores"]["pre_ge"] = round(pre_score, 4)
        diag["scores"]["post_ge"] = round(ge_score, 4)
        if ge_score >= pre_score * 0.95:
            chosen_result = ge_result
            chosen_label += "+GE"
            diag["path"].append("GE_applied")
        else:
            diag["path"].append("GE_rejected_worse")
    else:
        ge_reason = (
            (ge_result or {})
            .get("metadata", {})
            .get("grid_extension", {})
            .get("reason", "unknown")
        )
        diag["path"].append(f"GE_not_activated:{ge_reason}")

    crowd_beats = np.sort(chosen_result["beats"])
    diag["crowd_label"] = chosen_label
    diag["crowd_P"] = round(
        float(np.median(np.diff(crowd_beats))) if len(crowd_beats) > 1 else 0, 4
    )
    diag["crowd_cv"] = round(ioi_cv(crowd_beats), 5)
    diag["crowd_n_beats"] = len(crowd_beats)
    diag["crowd_beats"] = crowd_beats
    return diag
