#!/usr/bin/env python3
"""
detect_multimodal.py — Phase-bimodal detection utility for the beat-tracking pipeline.

When participants tap at two distinct phase offsets within the same beat period P,
the cleaned KDE peaks are interleaved:  [A₀, B₀, A₁, B₁, ...].
The IPIs alternate between δ (the phase gap between groups) and P−δ.

This produces "uneven spacing" in the optimizer output because the optimizer
fits a single regular grid to a signal that has two peaks per beat period, each
at a slightly different phase.  This module detects the bimodality in the IPI
distribution of cleaned KDE peaks and routes the stem to TWO separate mono-modal
optimizations (one per phase group), so the better-fitting one can be selected.

Usage (standalone audit)
------------------------
  cd ${PROJECT_ROOT}/analysis_pipeline
  python optimization/detect_multimodal.py

  # Or from a script:
  from optimization.detect_multimodal import detect_multimodal_stems, split_peaks_bimodal

Integration with run_pipeline_v2.py
------------------------------------
  After calling optimize_beats(), pass (peaks, result["beats"]) to
  split_peaks_bimodal().  If was_split=True, run optimize_beats() separately
  on groupA and groupB and store both results alongside the main result.

  Suggested JSON key layout (per stem in the pipeline output):
    result["mono_modal"] = {
        "detected":  True,
        "phase_delta_s": float,
        "groupA_n_peaks": int,
        "groupB_n_peaks": int,
        "mono_A": { ...optimize_beats() output for groupA... },
        "mono_B": { ...optimize_beats() output for groupB... },
    }
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ── Adjust to wherever the pipeline lives ──────────────────────────────────────
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "optimization"))


# ── Core algorithm (also imported by gen_validation_audio.py) ─────────────────

def split_peaks_bimodal(
    peaks: np.ndarray,
    opt_beats: np.ndarray,
    min_gap_frac: float = 0.12,
    min_group_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Split cleaned KDE peaks into two phase-consistent groups using IPI analysis.

    Parameters
    ----------
    peaks : sorted array of KDE peak times (seconds) within the analysis window.
    opt_beats : sorted array of optimizer beat times — used only to derive P.
    min_gap_frac : minimum gap in the sub-period IPI distribution (as a fraction
                   of P) required to declare bimodality.  Default 0.12 (~12%).
    min_group_size : minimum number of peaks required in each group.

    Returns
    -------
    groupA : peaks in phase cluster A
    groupB : peaks in phase cluster B
    was_split : True if clear bimodality was found
    phase_delta_s : estimated phase separation δ between the two groups (seconds);
                    0.0 if was_split=False.

    Algorithm
    ---------
    1. Compute all inter-peak intervals (IPIs).
    2. Keep only "sub-period" IPIs (< 0.9 × P).
    3. Find the largest jump in the sorted sub-period IPI sequence.
       If that jump ≥ min_gap_frac × P, two IPI clusters exist: δ and P−δ.
    4. Walk the peak sequence: consecutive peaks with IPI ≤ threshold form
       (groupA, groupB) pairs.

    Applicability
    -------------
    Works for any dataset processed by the pipeline — just pass the cleaned
    KDE peaks for the stem and the optimizer's beat array.  The algorithm is
    agnostic to corpus, tempo, or number of participants.
    """
    peaks = np.sort(peaks)
    if len(peaks) < 8 or opt_beats.size < 2:
        return peaks, np.array([], dtype=float), False, 0.0

    P = float(np.median(np.diff(opt_beats)))
    if P <= 0:
        return peaks, np.array([], dtype=float), False, 0.0

    IPIs = np.diff(peaks)
    sub_IPIs = IPIs[IPIs < 0.9 * P]
    if len(sub_IPIs) < 3:
        return peaks, np.array([], dtype=float), False, 0.0

    sorted_sub = np.sort(sub_IPIs)
    diffs_sub = np.diff(sorted_sub)
    if len(diffs_sub) == 0 or float(np.max(diffs_sub)) < min_gap_frac * P:
        return peaks, np.array([], dtype=float), False, 0.0

    gap_idx = int(np.argmax(diffs_sub))
    threshold = 0.5 * (sorted_sub[gap_idx] + sorted_sub[gap_idx + 1])
    phase_delta = float(threshold)

    is_short = IPIs <= threshold
    labels = np.zeros(len(peaks), dtype=int)   # 0 = group A, 1 = group B
    in_pair = False
    for i in range(len(peaks) - 1):
        if is_short[i] and not in_pair:
            labels[i]     = 0
            labels[i + 1] = 1
            in_pair = True
        elif in_pair:
            in_pair = False

    groupA = peaks[labels == 0]
    groupB = peaks[labels == 1]

    if len(groupA) < min_group_size or len(groupB) < min_group_size:
        return peaks, np.array([], dtype=float), False, 0.0

    return groupA, groupB, True, phase_delta


def bimodal_score(peaks: np.ndarray, opt_beats: np.ndarray,
                  min_gap_frac: float = 0.12) -> float:
    """
    Return the bimodality score for a stem: the largest sub-period IPI gap
    as a fraction of P.  Values ≥ min_gap_frac indicate likely bimodality.

    Useful for sorting stems by severity or setting adaptive thresholds.
    """
    peaks = np.sort(peaks)
    if len(peaks) < 8 or opt_beats.size < 2:
        return 0.0
    P = float(np.median(np.diff(opt_beats)))
    if P <= 0:
        return 0.0
    IPIs = np.diff(peaks)
    sub_IPIs = IPIs[IPIs < 0.9 * P]
    if len(sub_IPIs) < 3:
        return 0.0
    diffs_sub = np.diff(np.sort(sub_IPIs))
    if len(diffs_sub) == 0:
        return 0.0
    return float(np.max(diffs_sub)) / P


# ── Standalone audit ───────────────────────────────────────────────────────────

def detect_multimodal_stems(
    summary_csv: Path,
    peaks_csvs: dict[str, Path],
    min_gap_frac: float = 0.12,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan all stems in summary_csv for phase bimodality.

    Parameters
    ----------
    summary_csv : path to pipeline_v2/summary.csv (or any summary CSV with
                  columns: stimulus_id, corpus, peaks_source, period_s, beats).
    peaks_csvs : mapping corpus_key → path to cleaned_kde_peaks.csv, e.g.
                 {"ballroom": ..., "ballroom_2x": ..., "mirex": ..., "mirex_2x": ...}
    min_gap_frac : bimodality detection threshold (fraction of P).

    Returns
    -------
    DataFrame with columns:
      stimulus_id, corpus, P_s, n_peaks, bimodal_score, is_bimodal,
      phase_delta_s, groupA_n, groupB_n
    """
    summary = pd.read_csv(summary_csv, low_memory=False)

    # Load all peaks CSVs into memory once
    all_peaks: dict[str, pd.DataFrame] = {}
    for key, path in peaks_csvs.items():
        if path.exists():
            all_peaks[key] = pd.read_csv(path, low_memory=False)
        else:
            print(f"  [WARN] peaks CSV not found: {path}")

    records = []
    for _, row in summary.iterrows():
        stem   = str(row["stem"])
        corpus = str(row.get("corpus", "ballroom"))
        src    = str(row.get("peaks_source", "1x"))
        key    = f"{corpus}_2x" if src == "2x" else corpus

        pdf = all_peaks.get(key)
        if pdf is None:
            continue
        sub = pdf[pdf["stimulus_id"].astype(str) == stem]
        if sub.empty:
            continue
        peaks = np.sort(sub["peak_time_s"].astype(float).values)

        # Reconstruct approximate optimizer beats from period + window
        P = float(row.get("P_opt", row.get("period_s", 0.8)))
        W0 = 0.0
        W1 = 30.0
        opt_beats = np.arange(W0, W1, P)

        score = bimodal_score(peaks, opt_beats, min_gap_frac)
        is_bi = score >= min_gap_frac
        phase_delta, gA_n, gB_n = 0.0, 0, 0

        if is_bi:
            gA, gB, split_ok, phase_delta = split_peaks_bimodal(
                peaks, opt_beats, min_gap_frac
            )
            if split_ok:
                gA_n, gB_n = len(gA), len(gB)
            else:
                is_bi = False

        records.append({
            "stimulus_id":   stem,
            "stem":          stem,
            "corpus":        corpus,
            "peaks_source":  src,
            "P_s":           round(P, 4),
            "n_peaks":       len(peaks),
            "bimodal_score": round(score, 4),
            "is_bimodal":    is_bi,
            "phase_delta_s": round(phase_delta, 4),
            "groupA_n":      gA_n,
            "groupB_n":      gB_n,
        })

    df = pd.DataFrame(records).sort_values("bimodal_score", ascending=False)

    if verbose:
        flagged = df[df["is_bimodal"]]
        print(f"\n{'─'*60}")
        print(f"Phase-bimodal stems detected: {len(flagged)} / {len(df)}")
        print(f"{'─'*60}")
        if len(flagged):
            print(flagged[["stimulus_id", "P_s", "bimodal_score",
                            "phase_delta_s", "groupA_n", "groupB_n"]].to_string(index=False))
        else:
            print("  None — all stems are unimodal at this threshold.")
        print(f"{'─'*60}\n")

    return df


if __name__ == "__main__":
    PIPELINE_SUMMARY = PIPELINE_ROOT / "outputs" / "pipeline_v2" / "summary.csv"
    PEAKS = {
        "ballroom_2x": PIPELINE_ROOT / "outputs/upstream_sweeps/opt_U2_loose1_ballroom_2x/cleaned_kde_peaks.csv",
        "ballroom":    PIPELINE_ROOT / "outputs/upstream_sweeps/opt_U2_loose1_ballroom/cleaned_kde_peaks.csv",
        "mirex_2x":    PIPELINE_ROOT / "outputs/upstream_sweeps/opt_U2_loose1_mirex_2x/cleaned_kde_peaks.csv",
        "mirex":       PIPELINE_ROOT / "outputs/upstream_sweeps/opt_U2_loose1_mirex/cleaned_kde_peaks.csv",
    }

    df = detect_multimodal_stems(PIPELINE_SUMMARY, PEAKS, min_gap_frac=0.12)

    out = PIPELINE_ROOT / "outputs" / "pipeline_v2" / "multimodal_audit.csv"
    df.to_csv(out, index=False)
    print(f"Full audit saved → {out}")
