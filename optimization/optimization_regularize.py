#!/usr/bin/env python3
"""
Optimization-based beat regularization (Track B).

Given raw KDE peaks and a KDE density curve from pooled taps, find beat times
that jointly optimize:
  - ISI regularity (beats evenly spaced at the dominant period)
  - KDE density support (beats at high-density regions)
  - Proximity to observed peaks (beats near raw KDE peaks)

Uses scipy.optimize.minimize (SLSQP) with nonlinear constraints.
Lives alongside (never replaces) the heuristic in single_pulse_regularize_kde_peaks.py.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Period estimation via autocorrelation of the peak train
# ---------------------------------------------------------------------------

def peak_train_acf(
    peaks: np.ndarray,
    w0: float,
    w1: float,
    dt: float = 0.005,
    max_lag_s: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autocorrelation of binary peak train → lag vs correlation.

    Returns (lags_s, acf_values) where lags_s is in seconds.
    """
    n = int(round((w1 - w0) / dt))
    x = np.zeros(n, dtype=np.float64)
    for p in peaks:
        idx = int(round((p - w0) / dt))
        if 0 <= idx < n:
            x[idx] = 1.0

    max_lag_samples = min(int(round(max_lag_s / dt)), n - 1)
    acf = np.correlate(x, x, mode="full")
    center = len(acf) // 2
    acf = acf[center : center + max_lag_samples + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    lags_s = np.arange(len(acf)) * dt
    return lags_s, acf


def estimate_period_acf(
    peaks: np.ndarray,
    w0: float,
    w1: float,
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
    dt: float = 0.005,
) -> Tuple[float, List[float]]:
    """Estimate dominant period from ACF of the peak train.

    Returns (best_period, candidate_periods) where candidates are sorted
    by descending ACF strength.
    """
    lags, acf = peak_train_acf(peaks, w0, w1, dt=dt)
    min_idx = max(1, int(round(ioi_min_s / dt)))
    max_idx = min(len(acf) - 1, int(round(ioi_max_s / dt)))

    if max_idx <= min_idx:
        ioi = np.diff(np.sort(peaks))
        ioi = ioi[(ioi >= ioi_min_s) & (ioi <= ioi_max_s)]
        return float(np.median(ioi)) if ioi.size > 0 else 0.5, []

    acf_band = acf[min_idx : max_idx + 1]
    peak_idx, props = find_peaks(acf_band, prominence=0.05, distance=max(1, int(0.05 / dt)))
    if len(peak_idx) == 0:
        best_idx = np.argmax(acf_band)
        return lags[min_idx + best_idx], []

    strengths = acf_band[peak_idx]
    order = np.argsort(-strengths)
    candidates = [float(lags[min_idx + peak_idx[i]]) for i in order]
    return candidates[0], candidates


def estimate_period_median_ioi(
    peaks: np.ndarray,
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
) -> float:
    """Fallback: median IOI in the valid range."""
    ioi = np.diff(np.sort(peaks))
    valid = ioi[(ioi >= ioi_min_s) & (ioi <= ioi_max_s)]
    if valid.size == 0:
        return float(np.median(ioi)) if ioi.size > 0 else 0.5
    return float(np.median(valid))


def refine_period_ls(
    peaks: np.ndarray, P_init: float, phi_init: float, tol_frac: float = 0.15
) -> Tuple[float, float]:
    """Least-squares refinement of period and phase.

    Fit peaks ≈ phi + k*P for integer k, keeping only peaks within tol_frac*P
    of the nearest grid slot.
    """
    P, phi = P_init, phi_init
    for _ in range(3):
        k = np.round((peaks - phi) / P)
        residuals = peaks - (phi + k * P)
        mask = np.abs(residuals) <= tol_frac * P
        if mask.sum() < 3:
            break
        k_sel = k[mask]
        p_sel = peaks[mask]
        A = np.column_stack([np.ones(len(k_sel)), k_sel])
        result = np.linalg.lstsq(A, p_sel, rcond=None)
        phi, P = result[0]
        if P < 0.1:
            break
    return float(P), float(phi)


def estimate_period(
    peaks: np.ndarray,
    w0: float,
    w1: float,
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
) -> Tuple[float, float, List[float]]:
    """Full period estimation: ACF → LS refinement → harmonic check.

    Returns (P, phi, candidate_periods).
    """
    if peaks.size < 3:
        P = estimate_period_median_ioi(peaks, ioi_min_s, ioi_max_s)
        return P, float(peaks[0]) if peaks.size > 0 else w0, [P]

    P_acf, candidates = estimate_period_acf(peaks, w0, w1, ioi_min_s, ioi_max_s)
    if not candidates:
        candidates = [P_acf]

    best_P, best_phi, best_frac = P_acf, w0, 0.0

    for P_cand in candidates[:5]:
        for mult in [0.5, 1.0, 2.0]:
            P_try = P_cand * mult
            if P_try < ioi_min_s or P_try > ioi_max_s:
                continue

            # Phase search
            n_phase = 48
            best_phase_frac = 0.0
            best_phase_phi = 0.0
            for j in range(n_phase):
                phi_try = w0 + j * P_try / n_phase
                k = np.round((peaks - phi_try) / P_try)
                residuals = np.abs(peaks - (phi_try + k * P_try))
                frac = np.mean(residuals < 0.15 * P_try)
                if frac > best_phase_frac:
                    best_phase_frac = frac
                    best_phase_phi = phi_try

            P_ref, phi_ref = refine_period_ls(peaks, P_try, best_phase_phi)
            if P_ref < ioi_min_s or P_ref > ioi_max_s:
                continue

            k = np.round((peaks - phi_ref) / P_ref)
            residuals = np.abs(peaks - (phi_ref + k * P_ref))
            frac = np.mean(residuals < 0.12 * P_ref + 0.035)

            if frac > best_frac:
                best_frac = frac
                best_P = P_ref
                best_phi = phi_ref

    return best_P, best_phi, candidates


# ---------------------------------------------------------------------------
# Smoothed ISI from raw peaks
# ---------------------------------------------------------------------------

def compute_smoothed_isi(
    peaks: np.ndarray,
    P_global: float,
    w0: float,
    w1: float,
    window_s: float = 6.0,
    max_deviation: float = 0.25,
) -> interp1d:
    """Compute smoothed ISI function from local median IOI of raw peaks.

    Returns an interpolation function ISI(t) -> local target period.
    The result is clamped to [P*(1-max_deviation), P*(1+max_deviation)].
    """
    peaks_s = np.sort(peaks)
    if peaks_s.size < 3:
        return interp1d([w0, w1], [P_global, P_global], fill_value=P_global, bounds_error=False)

    sample_t = np.linspace(w0, w1, max(10, int((w1 - w0) / 0.5)))
    local_isi = np.full(len(sample_t), P_global)

    ioi_all = np.diff(peaks_s)

    for i, t in enumerate(sample_t):
        mask = (peaks_s >= t - window_s / 2) & (peaks_s <= t + window_s / 2)
        local_peaks = peaks_s[mask]
        if local_peaks.size >= 3:
            ioi = np.diff(local_peaks)
            valid = ioi[(ioi >= P_global * 0.4) & (ioi <= P_global * 2.5)]
            if valid.size >= 2:
                local_isi[i] = float(np.median(valid))

    # Smooth with simple moving average
    kernel_size = max(3, len(sample_t) // 8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(local_isi, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")[:len(sample_t)]

    # Clamp
    lo = P_global * (1 - max_deviation)
    hi = P_global * (1 + max_deviation)
    smoothed = np.clip(smoothed, lo, hi)

    return interp1d(sample_t, smoothed, fill_value=(smoothed[0], smoothed[-1]), bounds_error=False)


# ---------------------------------------------------------------------------
# Objective function and constraints
# ---------------------------------------------------------------------------

def build_kde_interpolant(
    tg: np.ndarray, yk: np.ndarray
) -> interp1d:
    """Normalized KDE as a continuous interpolant."""
    yk_n = yk / (np.max(yk) + 1e-12)
    return interp1d(tg, yk_n, fill_value=0.0, bounds_error=False)


def objective(
    x: np.ndarray,
    isi_fn: interp1d,
    kde_fn: interp1d,
    raw_peaks: np.ndarray,
    lambdas: Tuple[float, float, float],
    anchor_weights: Optional[np.ndarray] = None,
    madmom_beats: Optional[np.ndarray] = None,
    madmom_weight: float = 0.0,
) -> float:
    """Three-or-four-term objective for beat placement.

    J(x) = λ1*C_isi + λ2*C_kde + λ3*C_peaks [+ madmom_weight*C_madmom]

    When madmom_beats is provided and madmom_weight > 0, a fourth term pulls
    beats toward nearest Madmom beat. The weight is typically set adaptively
    based on tapping coherence (high weight when crowd signal is weak).

    x must be sorted ascending.
    """
    lam1, lam2, lam3 = lambdas
    m = len(x)
    if m < 2:
        return 1e6

    # Term 1: ISI regularity — penalize deviation from smoothed ISI
    ioi = np.diff(x)
    target_isi = np.array([float(isi_fn(x[i])) for i in range(1, m)])
    c_isi = float(np.mean((ioi - target_isi) ** 2))

    # Term 2: KDE density — reward beats at high-density positions
    kde_vals = np.array([float(kde_fn(xi)) for xi in x])
    c_kde = -float(np.mean(kde_vals))

    # Term 3: Proximity to observed peaks — pull toward nearest raw peak
    if raw_peaks.size > 0:
        dists_sq = np.array([
            float(np.min((xi - raw_peaks) ** 2)) for xi in x
        ])
        if anchor_weights is not None and len(anchor_weights) == len(raw_peaks):
            nearest_idx = np.array([int(np.argmin(np.abs(xi - raw_peaks))) for xi in x])
            w = anchor_weights[nearest_idx]
            c_peaks = float(np.mean(w * dists_sq))
        else:
            c_peaks = float(np.mean(dists_sq))
    else:
        c_peaks = 0.0

    cost = lam1 * c_isi + lam2 * c_kde + lam3 * c_peaks

    # Term 4 (optional): Madmom proximity — pull toward algorithmic beats
    if madmom_beats is not None and madmom_beats.size > 0 and madmom_weight > 0:
        mad_dists_sq = np.array([
            float(np.min((xi - madmom_beats) ** 2)) for xi in x
        ])
        cost += madmom_weight * float(np.mean(mad_dists_sq))

    return cost


def objective_gradient(
    x: np.ndarray,
    isi_fn: interp1d,
    kde_fn: interp1d,
    raw_peaks: np.ndarray,
    lambdas: Tuple[float, float, float],
    anchor_weights: Optional[np.ndarray] = None,
    madmom_beats: Optional[np.ndarray] = None,
    madmom_weight: float = 0.0,
) -> np.ndarray:
    """Numerical gradient of the objective (central differences)."""
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = objective(x, isi_fn, kde_fn, raw_peaks, lambdas, anchor_weights,
                   madmom_beats, madmom_weight)
    for i in range(len(x)):
        x_up = x.copy()
        x_up[i] += eps
        grad[i] = (objective(x_up, isi_fn, kde_fn, raw_peaks, lambdas, anchor_weights,
                             madmom_beats, madmom_weight) - f0) / eps
    return grad


def build_constraints(
    m: int, isi_fn: interp1d, P_global: float, w0: float, w1: float,
    isi_deviation: float = 0.25, min_ioi_frac: float = 0.50, max_ioi_frac: float = 2.0,
):
    """Build SLSQP constraints for beat optimization.

    - Each IOI within isi_deviation of the local smoothed ISI
    - min IOI >= min_ioi_frac * P
    - max IOI <= max_ioi_frac * P
    - beats within [w0, w1]
    """
    constraints = []

    # Ordering + minimum IOI
    min_ioi = min_ioi_frac * P_global
    for i in range(m - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i: x[i + 1] - x[i] - min_ioi,
        })

    # Maximum IOI
    max_ioi = max_ioi_frac * P_global
    for i in range(m - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i: max_ioi - (x[i + 1] - x[i]),
        })

    # Bounds
    bounds = [(w0, w1)] * m

    return constraints, bounds


# ---------------------------------------------------------------------------
# Anchor weights from KDE density
# ---------------------------------------------------------------------------

def compute_anchor_weights(
    raw_peaks: np.ndarray,
    kde_fn: interp1d,
    threshold: float = 0.4,
    high_weight: float = 2.0,
    low_weight: float = 1.0,
) -> np.ndarray:
    """Upweight peaks at high-KDE positions (anchor points)."""
    kde_vals = np.array([float(kde_fn(p)) for p in raw_peaks])
    weights = np.where(kde_vals > threshold, high_weight, low_weight)
    return weights


# ---------------------------------------------------------------------------
# Grid-search beat optimizer (replaces free SLSQP for isochrony-first mode)
# ---------------------------------------------------------------------------

def _grid_beats(T: float, phi: float, w0: float, w1: float) -> np.ndarray:
    """Generate a perfect isochronous grid φ + k·T within [w0, w1]."""
    k_min = int(np.ceil((w0 - phi) / T))
    k_max = int(np.floor((w1 - phi) / T))
    beats = phi + np.arange(k_min, k_max + 1) * T
    return beats[(beats >= w0) & (beats <= w1)]


def _grid_score(
    T: float,
    phi: float,
    w0: float,
    w1: float,
    kde_fn: interp1d,
    raw_peaks: np.ndarray,
    peak_weight: float = 0.5,
) -> float:
    """Score a (T, φ) grid by: mean KDE density + peak_weight * coverage fraction."""
    beats = _grid_beats(T, phi, w0, w1)
    if len(beats) < 2:
        return -1.0
    kde_vals = np.array([float(kde_fn(b)) for b in beats])
    mean_kde = float(np.mean(kde_vals))
    if raw_peaks.size > 0 and peak_weight > 0:
        tol = 0.12 * T + 0.02
        cov = _peak_coverage(beats, raw_peaks, tol_s=tol)
    else:
        cov = 0.0
    return mean_kde + peak_weight * cov


def optimize_beats_grid_search(
    raw_peaks: np.ndarray,
    tg: np.ndarray,
    yk: np.ndarray,
    w0: float,
    w1: float,
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
    T_steps: int = 120,
    phi_steps: int = 240,
    snap_max_frac: float = 0.12,
    snap_weight: float = 0.45,
    peak_weight: float = 0.5,
    refine_ls: bool = True,
) -> Dict:
    """Isochrony-first beat estimator via 2-D grid search over (period, phase).

    Algorithm
    ---------
    1. Build a dense grid of candidate periods T ∈ [ioi_min_s, ioi_max_s] and
       phases φ ∈ [w0, w0 + T).
    2. For each (T, φ) score the regular grid by mean KDE density + coverage
       of raw KDE peaks.  Pick the (T*, φ*) with the highest score.
    3. Optionally refine (T*, φ*) via least-squares fitting on peaks that fall
       within ±0.15*T of the winning grid.
    4. Apply bounded per-beat snapping: each beat shifts toward its nearest
       raw KDE peak by snap_weight * distance, capped at snap_max_frac * T.
       Only one IOI-enforcement pass is done afterwards (median-based).
    5. Post-process with doublet removal, gap filling, and edge extension
       (same as optimize_beats).

    Parameters
    ----------
    raw_peaks    : KDE peak times (seconds)
    tg, yk       : KDE time grid and density values
    w0, w1       : analysis window
    ioi_min_s    : minimum allowed IOI (seconds)
    ioi_max_s    : maximum allowed IOI (seconds)
    T_steps      : number of period values to try
    phi_steps    : number of phase values per period
    snap_max_frac: max per-beat shift toward nearest peak, as fraction of T
                   (default 0.12 → ±12 % of T, ~36–150 ms at typical tempos)
    snap_weight  : how strongly to attract each beat toward the nearest peak
                   (0 = pure grid, 1 = fully snap; default 0.45)
    peak_weight  : weight of peak coverage term in grid scoring (default 0.5)
    refine_ls    : refine (T*, φ*) by least-squares before snapping (default True)

    Returns
    -------
    dict with keys: beats, P, phi, cost, cost_terms, n_beats, candidates_tried,
                    metadata  (same schema as optimize_beats)
    """
    peaks = np.sort(np.unique(raw_peaks))
    peaks = peaks[(peaks >= w0) & (peaks <= w1)]

    kde_fn = build_kde_interpolant(tg, yk)

    if peaks.size < 3:
        return {
            "beats": peaks.copy(),
            "P": 0.5, "phi": w0,
            "cost": float("inf"), "cost_terms": {},
            "n_beats": len(peaks), "candidates_tried": [],
            "metadata": {"status": "insufficient_peaks", "method": "grid_search"},
        }

    # ── 1. 2-D grid search ─────────────────────────────────────────────────
    T_vals = np.linspace(ioi_min_s, ioi_max_s, T_steps)
    best_score = -1.0
    best_T = float(np.median(np.diff(peaks)))
    best_phi = w0

    for T in T_vals:
        phi_vals = w0 + np.arange(phi_steps) * (T / phi_steps)
        for phi in phi_vals:
            s = _grid_score(T, phi, w0, w1, kde_fn, peaks, peak_weight=peak_weight)
            if s > best_score:
                best_score = s
                best_T = T
                best_phi = phi

    # ── 2. Least-squares refinement ────────────────────────────────────────
    if refine_ls and peaks.size >= 3:
        T_ref, phi_ref = refine_period_ls(peaks, best_T, best_phi, tol_frac=0.15)
        if ioi_min_s <= T_ref <= ioi_max_s:
            best_T, best_phi = T_ref, phi_ref
        # Micro-refine phase by dense search around best_T
        phi_fine = best_phi + np.linspace(-best_T * 0.1, best_T * 0.1, 80)
        phi_fine = phi_fine[(phi_fine >= w0) & (phi_fine < w0 + best_T)]
        if len(phi_fine) > 0:
            scores_fine = [
                _grid_score(best_T, ph, w0, w1, kde_fn, peaks, peak_weight=peak_weight)
                for ph in phi_fine
            ]
            best_phi = float(phi_fine[int(np.argmax(scores_fine))])

    # ── 3. Build the winning grid ───────────────────────────────────────────
    grid = _grid_beats(best_T, best_phi, w0, w1)
    if len(grid) < 3:
        return {
            "beats": peaks.copy(),
            "P": best_T, "phi": best_phi,
            "cost": float("inf"), "cost_terms": {},
            "n_beats": len(peaks), "candidates_tried": [],
            "metadata": {"status": "grid_too_short", "method": "grid_search"},
        }

    # ── 4. Bounded per-beat snapping toward nearest KDE peak ───────────────
    snapped = grid.copy()
    max_shift = snap_max_frac * best_T
    if peaks.size > 0 and snap_weight > 0:
        for i, b in enumerate(grid):
            dists = np.abs(peaks - b)
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])
            if nearest_dist <= max_shift:
                delta = (peaks[nearest_idx] - b) * snap_weight
                # Hard-cap at max_shift
                delta = float(np.clip(delta, -max_shift, max_shift))
                snapped[i] = b + delta

        # Enforce that no IOI deviates more than snap_max_frac from median
        # (one relaxation pass: if a snapped beat creates a bad IOI, revert)
        iois = np.diff(snapped)
        median_ioi = float(np.median(iois))
        for i in range(len(snapped) - 1):
            if abs(iois[i] - median_ioi) / (median_ioi + 1e-9) > snap_max_frac * 2:
                snapped[i + 1] = grid[i + 1]  # revert this beat

    snapped = np.sort(snapped)

    # ── 5. Compute cost terms for metadata ─────────────────────────────────
    ioi_s = np.diff(snapped)
    c_isi = float(np.std(ioi_s) / (np.mean(ioi_s) + 1e-12))  # CV (lower = better)
    c_kde = -float(np.mean([float(kde_fn(b)) for b in snapped]))
    c_peaks = float(np.mean([float(np.min((b - peaks) ** 2)) for b in snapped])) \
              if peaks.size > 0 else 0.0
    coverage = _peak_coverage(snapped, peaks, tol_s=0.12 * best_T + 0.02)

    result: Dict = {
        "beats": snapped,
        "P": best_T,
        "phi": best_phi,
        "cost": c_isi,
        "cost_terms": {"c_isi_cv": c_isi, "c_kde": c_kde, "c_peaks": c_peaks},
        "n_beats": len(snapped),
        "candidates_tried": [{"T": best_T, "phi": best_phi, "score": best_score}],
        "metadata": {
            "status": "optimized",
            "method": "grid_search",
            "T_steps": T_steps,
            "phi_steps": phi_steps,
            "snap_max_frac": snap_max_frac,
            "snap_weight": snap_weight,
            "best_grid_score": best_score,
            "ioi_cv": c_isi,
            "coverage": coverage,
        },
    }

    # ── 6. Same post-processing pipeline as optimize_beats ─────────────────
    result = _remove_doublets(result, kde_fn)
    result = metrical_subdivide(result, peaks, kde_fn, w0, w1, ioi_min_s)
    result = _remove_doublets(result, kde_fn)
    result = _fill_gaps(result, w0, w1, ioi_min_s)

    return result


# ---------------------------------------------------------------------------
# Initial guess strategies
# ---------------------------------------------------------------------------

def initial_guess_grid(
    P: float, phi: float, w0: float, w1: float
) -> np.ndarray:
    """Uniform grid at estimated period and phase."""
    k_min = int(np.ceil((w0 - phi) / P))
    k_max = int(np.floor((w1 - phi) / P))
    beats = phi + np.arange(k_min, k_max + 1) * P
    beats = beats[(beats >= w0) & (beats <= w1)]
    return np.sort(beats)


def initial_guess_peaks(
    raw_peaks: np.ndarray, P: float, w0: float, w1: float,
    ioi_min_frac: float = 0.5,
) -> np.ndarray:
    """Start from raw peaks, removing those too close together."""
    peaks = np.sort(raw_peaks)
    peaks = peaks[(peaks >= w0) & (peaks <= w1)]
    if peaks.size == 0:
        return peaks

    min_gap = P * ioi_min_frac
    selected = [peaks[0]]
    for p in peaks[1:]:
        if p - selected[-1] >= min_gap:
            selected.append(p)
    return np.array(selected)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def _peak_coverage(x_opt: np.ndarray, raw_peaks: np.ndarray, tol_s: float = 0.08) -> float:
    """Fraction of raw KDE peaks that have an optimized beat within tol_s."""
    if len(raw_peaks) == 0 or len(x_opt) == 0:
        return 0.0
    covered = 0
    for p in raw_peaks:
        if np.min(np.abs(x_opt - p)) <= tol_s:
            covered += 1
    return covered / len(raw_peaks)


def optimize_beats(
    raw_peaks: np.ndarray,
    tg: np.ndarray,
    yk: np.ndarray,
    w0: float,
    w1: float,
    lambdas: Tuple[float, float, float] = (2.0, 0.3, 0.3),
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
    isi_deviation: float = 0.25,
    min_ioi_frac: float = 0.50,
    max_ioi_frac: float = 2.0,
    anchor_threshold: float = 0.4,
    anchor_high_weight: float = 2.0,
    max_iter: int = 500,
    madmom_beats: Optional[np.ndarray] = None,
    madmom_weight: float = 0.0,
    external_init: Optional[Tuple[float, float]] = None,
    method: str = "slsqp",
    grid_snap_max_frac: float = 0.12,
    grid_snap_weight: float = 0.45,
) -> Dict:
    """Run the full optimization pipeline for one stem.

    Parameters
    ----------
    method : "slsqp" (default) or "grid_search"
        "slsqp"       — original free optimization (individual beat positions)
        "grid_search" — isochrony-first: search (T, φ) space, then apply bounded
                        per-beat snapping toward KDE peaks.  Guarantees near-
                        perfect IOI regularity similar to Madmom.
    grid_snap_max_frac : max per-beat shift in grid_search mode (fraction of T)
    grid_snap_weight   : attraction strength toward nearest KDE peak in grid mode

    Returns dict with keys: beats, P, phi, cost, cost_terms, n_beats,
    candidates_tried, metadata.
    """
    if method == "grid_search":
        return optimize_beats_grid_search(
            raw_peaks, tg, yk, w0, w1,
            ioi_min_s=ioi_min_s,
            ioi_max_s=ioi_max_s,
            snap_max_frac=grid_snap_max_frac,
            snap_weight=grid_snap_weight,
        )

    peaks = np.sort(np.unique(raw_peaks))
    peaks = peaks[(peaks >= w0) & (peaks <= w1)]

    if peaks.size < 3:
        return {
            "beats": peaks.copy(),
            "P": 0.5,
            "phi": w0,
            "cost": float("inf"),
            "cost_terms": {},
            "n_beats": len(peaks),
            "candidates_tried": [],
            "metadata": {"status": "insufficient_peaks"},
        }

    # Period estimation
    P_best, phi_best, candidates = estimate_period(peaks, w0, w1, ioi_min_s, ioi_max_s)

    # Build KDE interpolant
    kde_fn = build_kde_interpolant(tg, yk)

    # Anchor weights
    anchor_w = compute_anchor_weights(peaks, kde_fn, threshold=anchor_threshold,
                                       high_weight=anchor_high_weight)

    # Try multiple period candidates (best-of-N)
    all_candidates = [P_best]
    for c in candidates:
        for mult in [0.5, 1.0, 2.0]:
            p = c * mult
            if ioi_min_s <= p <= ioi_max_s and not any(abs(p - a) < 0.02 for a in all_candidates):
                all_candidates.append(p)
    # Also explicitly add P_best/2 and P_best/3 if in range
    for div in [2, 3]:
        p = P_best / div
        if ioi_min_s <= p <= ioi_max_s and not any(abs(p - a) < 0.02 for a in all_candidates):
            all_candidates.append(p)
    all_candidates = all_candidates[:8]

    ext_P: Optional[float] = None
    ext_phi: Optional[float] = None
    if external_init is not None:
        ext_P, ext_phi = float(external_init[0]), float(external_init[1])
        if ioi_min_s <= ext_P <= ioi_max_s:
            merged = [ext_P]
            for a in all_candidates:
                if abs(a - ext_P) >= 0.02:
                    merged.append(a)
            all_candidates = merged[:8]

    best_result = None
    best_score = float("-inf")
    tried = []

    for P_cand in all_candidates:
        if ext_P is not None and abs(P_cand - ext_P) < 0.02 and ext_phi is not None:
            init_phi_ls = ext_phi
        else:
            init_phi_ls = phi_best if abs(P_cand - P_best) < 1e-9 else w0
        P_ref, phi_ref = refine_period_ls(peaks, P_cand, init_phi_ls)
        if P_ref < ioi_min_s or P_ref > ioi_max_s:
            P_ref = P_cand

        isi_fn = compute_smoothed_isi(peaks, P_ref, w0, w1, max_deviation=isi_deviation)

        for init_label, x0 in [
            ("grid", initial_guess_grid(P_ref, phi_ref, w0, w1)),
            ("peaks", initial_guess_peaks(peaks, P_ref, w0, w1)),
        ]:
            if x0.size < 3:
                continue

            m = len(x0)
            constraints, bounds = build_constraints(
                m, isi_fn, P_ref, w0, w1,
                isi_deviation=isi_deviation,
                min_ioi_frac=min_ioi_frac,
                max_ioi_frac=max_ioi_frac,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = minimize(
                        objective,
                        x0,
                        args=(isi_fn, kde_fn, peaks, lambdas, anchor_w,
                              madmom_beats, madmom_weight),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": max_iter, "ftol": 1e-9},
                    )
                except Exception:
                    continue

            if not (result.success or result.fun < 1e5):
                continue

            cost = float(result.fun)
            x_opt = np.sort(result.x)

            ioi_opt = np.diff(x_opt)
            tgt = np.array([float(isi_fn(x_opt[j])) for j in range(1, len(x_opt))])
            c_isi = float(np.mean((ioi_opt - tgt) ** 2))
            c_kde = -float(np.mean([float(kde_fn(xi)) for xi in x_opt]))
            dists_sq = np.array([float(np.min((xi - peaks) ** 2)) for xi in x_opt])
            c_peaks = float(np.mean(dists_sq))

            # Composite score: maximize peak coverage + KDE density + isochrony,
            # penalize raw cost. This prevents half-time solutions from winning
            # just because they have fewer terms in the mean.
            coverage = _peak_coverage(x_opt, peaks, tol_s=0.12 * P_ref + 0.03)
            ioi_cv = float(np.std(ioi_opt) / (np.mean(ioi_opt) + 1e-12))
            isochrony = max(0.0, 1.0 - ioi_cv)
            mean_kde = float(np.mean([float(kde_fn(xi)) for xi in x_opt]))
            score = (
                2.0 * coverage      # strongly reward covering raw KDE peaks
                + 1.0 * isochrony   # reward even spacing
                + 0.5 * mean_kde    # reward KDE density
                - 0.3 * cost        # penalize high optimization cost
            )

            tried.append({
                "P": P_ref, "init": init_label, "n_beats": m,
                "cost": cost, "score": score, "coverage": coverage,
                "isochrony": isochrony, "success": result.success,
            })

            if score > best_score:
                best_score = score
                best_result = {
                    "beats": x_opt,
                    "P": P_ref,
                    "phi": phi_ref,
                    "cost": cost,
                    "cost_terms": {
                        "c_isi": c_isi, "c_kde": c_kde, "c_peaks": c_peaks,
                        "lambdas": lambdas,
                    },
                    "n_beats": len(x_opt),
                    "candidates_tried": tried,
                    "metadata": {
                        "status": "optimized",
                        "solver_success": result.success,
                        "solver_message": result.message,
                        "n_iter": int(result.nit) if hasattr(result, "nit") else -1,
                        "score": best_score,
                        "coverage": coverage,
                        "isochrony": isochrony,
                    },
                }

    if best_result is None:
        return {
            "beats": peaks.copy(),
            "P": P_best,
            "phi": phi_best,
            "cost": float("inf"),
            "cost_terms": {},
            "n_beats": len(peaks),
            "candidates_tried": tried,
            "metadata": {"status": "fallback_raw_peaks"},
        }

    # --- Post-optimization cleanup pipeline ---
    best_result = _remove_doublets(best_result, kde_fn)
    best_result = metrical_subdivide(best_result, peaks, kde_fn, w0, w1, ioi_min_s)
    best_result = _remove_doublets(best_result, kde_fn)
    best_result = _fill_gaps(best_result, w0, w1, ioi_min_s)

    final_cov = _peak_coverage(best_result["beats"], peaks, tol_s=0.12 * best_result["P"] + 0.03)
    best_result["metadata"] = dict(best_result.get("metadata", {}))
    best_result["metadata"]["final_coverage"] = final_cov

    return best_result


# ---------------------------------------------------------------------------
# Confident grid extension: find stable regions and back-project the best grid
# ---------------------------------------------------------------------------

def _fit_grid_from_region(region_beats: np.ndarray, w0: float, w1: float,
                          ioi_min_s: float = 0.25, ioi_max_s: float = 1.5
                          ) -> Optional[Tuple[float, float, np.ndarray]]:
    """Fit P and phi from a confident region via least-squares, then generate
    a full-window grid.  Returns (P, phi, grid) or None if infeasible."""
    from numpy.linalg import lstsq

    if len(region_beats) < 4:
        return None

    P_init = float(np.median(np.diff(region_beats)))
    if P_init < ioi_min_s or P_init > ioi_max_s:
        return None

    phi = region_beats[0]
    P = P_init

    for _ in range(5):
        k = np.round((region_beats - phi) / P).astype(int)
        if len(np.unique(k)) < len(k) * 0.8:
            break
        A = np.column_stack([k, np.ones(len(k))])
        sol, _, _, _ = lstsq(A, region_beats, rcond=None)
        P_new, phi_new = float(sol[0]), float(sol[1])
        if P_new < ioi_min_s or P_new > ioi_max_s:
            break
        if abs(P_new - P) < 1e-6 and abs(phi_new - phi) < 1e-6:
            P, phi = P_new, phi_new
            break
        P, phi = P_new, phi_new

    k_min = int(np.floor((w0 - phi) / P))
    k_max = int(np.ceil((w1 - phi) / P))
    grid = phi + np.arange(k_min, k_max + 1) * P
    grid = grid[(grid >= w0) & (grid <= w1)]

    if len(grid) < 3:
        return None
    return P, phi, grid


def _confident_grid_extension(
    result: Dict,
    kde_fn: interp1d,
    w0: float,
    w1: float,
    input_peaks: Optional[np.ndarray] = None,
    min_region_beats: int = 6,
    max_cv_thresh: float = 0.15,
    min_global_cv: float = 0.03,
) -> Dict:
    """Find the most isochronous consecutive run of optimizer beats, fit P and
    phi precisely via least-squares, then stamp phi + k*P wall-to-wall across
    [w0, w1], completely replacing the optimizer output.

    This is a hard override: the confident region is the authority and the pure
    grid is the output.  KDE density is NOT used for placement or scoring.

    Algorithm
    ---------
    1. Skip if the optimizer output is already isochronous (CV <= min_global_cv).
    2. Score every consecutive window of size >= min_region_beats using
           score = n_beats * exp(-cv / 0.05)
       which rewards length while strongly penalising high CV.  Discard windows
       with CV > max_cv_thresh.  Pick the highest-scoring window.
    3. Require the winning region's CV to be < 0.75 × global_cv (meaningfully
       more isochronous than the full sequence).
    4. Fit P and phi on that region via iterative least-squares
       (_fit_grid_from_region).
    5. Accept the resulting grid only if it covers raw input_peaks at least
       75 % as well as the original optimizer beats do (safety guard against
       half-time or otherwise mismatched grids).
    6. Stamp grid = {phi + k*P : k ∈ ℤ} ∩ [w0, w1] as the new beat sequence.

    NOTE: this function is intentionally NOT called inside optimize_beats().
    Apply it as an explicit post-processing step in test / batch scripts so
    that the v14 reference pipeline remains unmodified.

    Parameters
    ----------
    result           : optimizer result dict (must have 'beats' key)
    kde_fn           : interpolated KDE (kept for caller convenience; not used here)
    w0, w1           : analysis window boundaries
    input_peaks      : raw KDE peaks that fed the optimizer — used for the
                       coverage acceptance gate.  If None, gate is skipped.
    min_region_beats : minimum beats in a confident region (default 6)
    max_cv_thresh    : upper CV limit for candidate regions (default 0.15)
    min_global_cv    : skip if optimizer output is already this isochronous
                       (default 0.03 — very permissive, almost always try)
    """
    beats = np.sort(result["beats"])

    if len(beats) < min_region_beats:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "insufficient_beats",
            "n_beats": len(beats),
        }
        return result

    iois = np.diff(beats)
    global_cv = float(np.std(iois) / (np.mean(iois) + 1e-12))

    if global_cv <= min_global_cv:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "already_isochronous",
            "global_cv": global_cv,
        }
        return result

    # --- Score every consecutive window: reward length, penalise CV ---
    # score = n * exp(-cv / 0.05): a 12-beat window at CV=0.05 scores higher
    # than a 6-beat window at CV=0.02, which is the desired behavior.
    n_beats = len(beats)
    best_score = -1.0
    best_start = -1
    best_end = -1
    best_region_cv = float("inf")

    for n in range(min_region_beats, n_beats + 1):
        for start in range(n_beats - n + 1):
            end = start + n - 1
            seg_iois = np.diff(beats[start : end + 1])
            seg_cv = float(np.std(seg_iois) / (np.mean(seg_iois) + 1e-12))
            if seg_cv > max_cv_thresh:
                continue
            score = n * float(np.exp(-seg_cv / 0.05))
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
                best_region_cv = seg_cv

    if best_start < 0:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "no_confident_region",
            "global_cv": global_cv,
        }
        return result

    # --- Gate 1: region must be meaningfully more isochronous than the full sequence ---
    # For very uneven sequences (global_cv > 0.10) we relax the threshold so that a
    # moderately good region (e.g. CV=0.07 vs global=0.13) still qualifies.
    gate1_ratio = 0.75 if global_cv <= 0.10 else 0.85
    if best_region_cv >= global_cv * gate1_ratio:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "region_not_meaningfully_better",
            "global_cv": global_cv,
            "best_region_cv": best_region_cv,
            "gate1_ratio_used": gate1_ratio,
        }
        return result

    region_beats = beats[best_start : best_end + 1]

    # --- Fit P and phi on the confident region via iterative least-squares ---
    fit = _fit_grid_from_region(region_beats, w0, w1)
    if fit is None:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "fit_failed",
            "global_cv": global_cv,
            "best_region_cv": best_region_cv,
        }
        return result

    P, phi, grid = fit

    if len(grid) < 3:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "grid_too_short",
        }
        return result

    # --- Gate 2a: fit quality — reject if residuals are too large ---
    # A poor P/phi fit means the region wasn't truly isochronous; don't project it.
    k_indices = np.round((region_beats - phi) / P).astype(int)
    residuals = np.abs(region_beats - (phi + k_indices * P))
    fit_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_fit_rmse = 0.035  # 35ms — reject fits worse than this
    if fit_rmse > max_fit_rmse:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["grid_extension"] = {
            "activated": False,
            "reason": "fit_quality_too_low",
            "global_cv": global_cv,
            "best_region_cv": best_region_cv,
            "fit_rmse": fit_rmse,
            "max_fit_rmse": max_fit_rmse,
        }
        return result

    # --- Gate 2b: grid must cover input_peaks at least as well as original ---
    # This guards against half-time or otherwise mismatched P values.
    if input_peaks is not None and len(input_peaks) > 0:
        tol = 0.12 * P + 0.030
        orig_cov = _peak_coverage(beats, input_peaks, tol_s=tol)
        grid_cov = _peak_coverage(grid, input_peaks, tol_s=tol)
        if grid_cov < orig_cov * 0.75:
            result = dict(result)
            result["metadata"] = dict(result.get("metadata", {}))
            result["metadata"]["grid_extension"] = {
                "activated": False,
                "reason": "grid_coverage_too_low",
                "global_cv": global_cv,
                "P_fitted": P,
                "fit_rmse": fit_rmse,
                "orig_coverage": orig_cov,
                "grid_coverage": grid_cov,
            }
            return result
    else:
        orig_cov = grid_cov = None

    grid_iois = np.diff(grid)
    grid_cv = float(np.std(grid_iois) / (np.mean(grid_iois) + 1e-12))

    result = dict(result)
    result["beats"] = grid
    result["n_beats"] = len(grid)
    result["P"] = P
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"]["grid_extension"] = {
        "activated": True,
        "confident_region": [float(region_beats[0]), float(region_beats[-1])],
        "confident_region_indices": [int(best_start), int(best_end)],
        "confident_n_beats": int(best_end - best_start + 1),
        "confident_cv": best_region_cv,
        "gate1_ratio_used": gate1_ratio,
        "fit_rmse": fit_rmse,
        "global_cv_before": global_cv,
        "global_cv_after": grid_cv,
        "P_fitted": P,
        "phi_fitted": phi,
        "n_grid_beats": len(grid),
        "orig_coverage": orig_cov,
        "grid_coverage": grid_cov,
    }
    return result


# ---------------------------------------------------------------------------
# Final isochrony enforcement
# ---------------------------------------------------------------------------

def _enforce_isochrony(result: Dict, w0: float, w1: float, max_shift_frac: float = 0.05) -> Dict:
    """Single-pass isochrony nudge with proportional cap on movement.

    Each interior beat is nudged toward the midpoint of its neighbors,
    but never by more than max_shift_frac * P (default 5% of period).
    For P=0.3s this is 15ms; for P=0.7s this is 35ms.
    """
    beats = np.sort(result["beats"])
    if len(beats) < 4:
        return result

    P = float(np.median(np.diff(beats)))
    max_shift = max_shift_frac * P
    x = beats.copy()
    n = len(x)

    x_new = x.copy()
    for i in range(1, n - 1):
        midpoint = (x[i - 1] + x[i + 1]) / 2.0
        delta = midpoint - x[i]
        delta = np.clip(delta, -max_shift, max_shift)
        x_new[i] = x[i] + delta

    x_new = np.clip(x_new, w0, w1)
    x_new = np.sort(x_new)

    ioi_final = np.diff(x_new)
    if np.any(ioi_final < 0.5 * P):
        return result

    result = dict(result)
    result["beats"] = x_new
    ioi_cv = float(np.std(ioi_final) / (np.mean(ioi_final) + 1e-12))
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"]["isochrony_cv_after"] = ioi_cv
    result["metadata"]["max_shift_applied_ms"] = float(np.max(np.abs(x_new - beats)) * 1000)
    return result


# ---------------------------------------------------------------------------
# Doublet removal: merge beats that are too close together
# ---------------------------------------------------------------------------

def _remove_doublets(result: Dict, kde_fn: interp1d) -> Dict:
    """Remove beats that are too close together (doublets).

    Two-pass approach:
    1. Remove any beat pair closer than 0.60 * median_IOI
    2. After removal, snap remaining beats toward an isochronous grid
       to restore even spacing
    """
    beats = np.sort(result["beats"])
    if len(beats) < 3:
        return result

    median_ioi = float(np.median(np.diff(beats)))
    threshold = 0.60 * median_ioi
    removed = 0

    keep = np.ones(len(beats), dtype=bool)
    i = 0
    while i < len(beats) - 1:
        if not keep[i]:
            i += 1
            continue
        gap = beats[i + 1] - beats[i]
        if gap < threshold:
            kde_a = float(kde_fn(beats[i]))
            kde_b = float(kde_fn(beats[i + 1]))
            if kde_a >= kde_b:
                keep[i + 1] = False
            else:
                keep[i] = False
            removed += 1
        i += 1

    if removed == 0:
        return result

    cleaned = beats[keep]

    # After removing doublets, do a light grid-snap pass to restore isochrony
    if len(cleaned) >= 3:
        P_clean = float(np.median(np.diff(cleaned)))
        for _ in range(5):
            new_beats = cleaned.copy()
            for j in range(1, len(new_beats) - 1):
                target = (new_beats[j - 1] + new_beats[j + 1]) / 2.0
                new_beats[j] = new_beats[j] + 0.3 * (target - new_beats[j])
            if np.max(np.abs(new_beats - cleaned)) < 1e-4:
                break
            cleaned = np.sort(new_beats)

    result = dict(result)
    result["beats"] = cleaned
    result["n_beats"] = len(cleaned)
    result["P"] = float(np.median(np.diff(cleaned))) if len(cleaned) >= 2 else result["P"]
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"]["doublets_removed"] = removed
    return result


# ---------------------------------------------------------------------------
# Gap filling: insert beats in large gaps to maintain isochrony
# ---------------------------------------------------------------------------

def _fill_gaps(result: Dict, w0: float, w1: float, ioi_min_s: float = 0.28) -> Dict:
    """Fill gaps that are larger than ~1.4x the median IOI.

    Isochrony-first: any gap that could fit another beat gets one.
    Uses the stem's own median IOI as the spacing floor (not hard-coded min).
    Also extends edges if the first/last beat is far from the window boundary.
    """
    beats = np.sort(result["beats"])
    if len(beats) < 3:
        return result

    P = float(np.median(np.diff(beats)))
    gap_threshold = 1.4 * P
    # Use the stem's actual spacing as floor, not the global ioi_min_s
    local_min = 0.75 * P
    filled = list(beats)
    insertions = 0

    i = 0
    while i < len(filled) - 1:
        gap = filled[i + 1] - filled[i]
        if gap > gap_threshold:
            n_insert = max(1, round(gap / P) - 1)
            spacing = gap / (n_insert + 1)
            if spacing >= local_min:
                new_beats = [filled[i] + (k + 1) * spacing for k in range(n_insert)]
                for nb in reversed(new_beats):
                    filled.insert(i + 1, nb)
                    insertions += 1
                i += n_insert + 1
            else:
                i += 1
        else:
            i += 1

    # Extend left edge
    t = filled[0] - P
    while t >= w0:
        filled.insert(0, t)
        insertions += 1
        t -= P

    # Extend right edge
    t = filled[-1] + P
    while t <= w1:
        filled.append(t)
        insertions += 1
        t += P

    if insertions == 0:
        return result

    filled = np.sort(np.array(filled))

    result = dict(result)
    result["beats"] = filled
    result["n_beats"] = len(filled)
    result["P"] = float(np.median(np.diff(filled))) if len(filled) >= 2 else result["P"]
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"]["gaps_filled"] = insertions
    return result


# ---------------------------------------------------------------------------
# Metrical subdivision: fix half-time / quarter-time outputs
# ---------------------------------------------------------------------------

def _kde_support_at_midpoints(beats: np.ndarray, kde_fn: interp1d) -> float:
    """Mean normalized KDE value at midpoints between consecutive beats."""
    if len(beats) < 2:
        return 0.0
    midpoints = (beats[:-1] + beats[1:]) / 2.0
    vals = np.array([float(kde_fn(m)) for m in midpoints])
    return float(np.mean(vals))


def _subdivide_isochronous(beats: np.ndarray, factor: int, w0: float, w1: float) -> np.ndarray:
    """Insert evenly-spaced beats between existing beats (factor=2 → halve spacing)."""
    if len(beats) < 2 or factor < 2:
        return beats
    new_beats = []
    for i in range(len(beats) - 1):
        new_beats.append(beats[i])
        gap = beats[i + 1] - beats[i]
        for k in range(1, factor):
            new_beats.append(beats[i] + k * gap / factor)
    new_beats.append(beats[-1])

    # Extend edges if there's room for more beats at the same spacing
    P_sub = np.median(np.diff(beats)) / factor
    # Extend left
    t = new_beats[0] - P_sub
    while t >= w0:
        new_beats.insert(0, t)
        t -= P_sub
    # Extend right
    t = new_beats[-1] + P_sub
    while t <= w1:
        new_beats.append(t)
        t += P_sub

    return np.sort(np.array(new_beats))


def _isochrony_score(beats: np.ndarray) -> float:
    """1 - IOI_CV. Higher = more isochronous."""
    if len(beats) < 2:
        return 0.0
    ioi = np.diff(beats)
    cv = float(np.std(ioi) / (np.mean(ioi) + 1e-12))
    return max(0.0, 1.0 - cv)


def metrical_subdivide(
    result: Dict,
    raw_peaks: np.ndarray,
    kde_fn: interp1d,
    w0: float,
    w1: float,
    ioi_min_s: float = 0.28,
) -> Dict:
    """Ultra-conservative half-time fix: only subdivide by 2 when raw KDE peaks
    overwhelmingly sit at midpoints between optimized beats.

    Requires >55% of midpoints to have a raw KDE peak within tight tolerance.
    Never subdivides by 3x or 4x — those should be caught by period estimation.
    """
    beats = np.sort(result["beats"])
    if len(beats) < 4 or len(raw_peaks) < 4:
        return result

    P_opt = float(np.median(np.diff(beats)))
    P_sub = P_opt / 2.0

    if P_sub < ioi_min_s:
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["subdivision"] = "no_subdivision"
        return result

    # Count raw KDE peaks near midpoints (tight tolerance)
    midpoints = (beats[:-1] + beats[1:]) / 2.0
    tol = 0.18 * P_opt  # tight: 18% of the beat spacing
    hits = 0
    for m in midpoints:
        if np.min(np.abs(m - raw_peaks)) <= tol:
            hits += 1
    midpoint_hit_frac = hits / len(midpoints)

    # Also check KDE density at midpoints vs at beats
    kde_at_beats = np.mean([float(kde_fn(b)) for b in beats])
    kde_at_mids = np.mean([float(kde_fn(m)) for m in midpoints])
    kde_ratio = kde_at_mids / (kde_at_beats + 1e-12)

    # Only subdivide if BOTH conditions met:
    # 1) >55% of midpoints have a raw KDE peak nearby
    # 2) KDE density at midpoints is at least 40% of density at beats
    should_subdivide = midpoint_hit_frac > 0.55 and kde_ratio > 0.40

    result = dict(result)
    result["metadata"] = dict(result.get("metadata", {}))

    if should_subdivide:
        subdivided = _subdivide_isochronous(beats, 2, w0, w1)
        subdivided = _equalize_beats(subdivided, kde_fn, raw_peaks, w0, w1, P_sub)
        result["beats"] = subdivided
        result["P"] = P_sub
        result["n_beats"] = len(subdivided)
        result["metadata"]["subdivision"] = "subdivide_2x"
        result["metadata"]["midpoint_hit_frac"] = midpoint_hit_frac
        result["metadata"]["kde_ratio"] = kde_ratio
    else:
        result["metadata"]["subdivision"] = "no_subdivision"
        result["metadata"]["midpoint_hit_frac"] = midpoint_hit_frac
        result["metadata"]["kde_ratio"] = kde_ratio

    return result


def _equalize_beats(
    beats: np.ndarray,
    kde_fn: interp1d,
    raw_peaks: np.ndarray,
    w0: float,
    w1: float,
    P_target: float,
    alpha: float = 0.3,
) -> np.ndarray:
    """Nudge beats toward perfect isochrony while respecting KDE/peak anchors.

    Iteratively blend each beat toward the position that would make its
    left and right IOIs equal to P_target, with weight alpha (rest stays put).
    Beats near strong KDE peaks are anchored more firmly.
    """
    x = beats.copy()
    n = len(x)
    if n < 3:
        return x

    kde_vals = np.array([float(kde_fn(xi)) for xi in x])
    # Anchor strength: high KDE = don't move much
    anchor = np.clip(kde_vals / (np.max(kde_vals) + 1e-12), 0, 1)

    for iteration in range(10):
        x_new = x.copy()
        for i in range(1, n - 1):
            # Target position from neighbors
            target_left = x[i - 1] + P_target
            target_right = x[i + 1] - P_target
            target = (target_left + target_right) / 2.0

            # Blend: strong anchors move less
            move_weight = alpha * (1.0 - 0.5 * anchor[i])
            x_new[i] = x[i] + move_weight * (target - x[i])

        # Keep within bounds and sorted
        x_new = np.clip(x_new, w0, w1)
        x_new = np.sort(x_new)

        if np.max(np.abs(x_new - x)) < 1e-4:
            break
        x = x_new

    return x
