#!/usr/bin/env python3
"""
Build listening-validation audio: per-stem excerpts with clicks at KDE / GT / madmom times,
plus concatenated triple files (6 permutations) for PsyNet trials.

Outputs (default --out_dir = next-steps/tapping-validation/data/validation_audio):
  {stem}_kde.wav, {stem}_gt.wav, {stem}_madmom.wav   — single-condition excerpts
  triple_{stem}_{perm}.wav                             — three excerpts + gaps (order = perm; default gap 1.0s)
  triple_{stem}_meta.json                            — perm index -> [cond1, cond2, cond3]

Music time window [t0_music, t1_music] is the same for all conditions.
WAV files are raw music (no REPP silence prepended); music_time = WAV_time.
stim_offset_s is used only for tap-time conversion, not for WAV extraction.

KDE sonification: optional **metrical-majority** filter (``--kde-metrical-filter``) fits one
global tempo+phase grid and thins peaks; it can sound wrong if the grid is misestimated.
**Default is raw** ``cleaned_kde_peaks.csv`` times; enable the filter only when you accept that risk.

Run from analysis_pipeline with venv:
  cd analysis_pipeline
  .venv/bin/python scripts/build_validation_listening_stims.py \\
    --manifest ../next-steps/tapping-validation/data/stimulus_manifest.csv

  # KDE-only sonifications for every stem in the cleaned_kde CSVs (ballroom + mirex),
  # same [t0,t1] window for all (default 10–25 s music time) — for offline QC / plots:
  .venv/bin/python scripts/build_validation_listening_stims.py \\
    --stems-from-peaks --kde-only \\
    --out_dir ../next-steps/tapping-validation/data/kde_sonification_eval

  # Optional: KDE times from GT-free pattern completion (see kde_pattern_completion_peaks.py):
  .venv/bin/python scripts/build_validation_listening_stims.py \\
    --manifest ../next-steps/tapping-validation/data/stimulus_manifest.csv \\
    --kde_peaks_csv outputs/pattern_completion/validation_manifest_pattern_peaks.csv \\
    --out_dir ../next-steps/tapping-validation/data/validation_audio_pattern_kde

Requires: soundfile, numpy, pandas; WAVs under audio/ballroom_wav and audio/mirex.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_SCRIPT_DIR = Path(__file__).resolve().parent
_AP_ROOT = _SCRIPT_DIR.parent

_spec = importlib.util.spec_from_file_location("_r", _SCRIPT_DIR / "repp_gt_alignment_pipeline.py")
_r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r)

# Copy minimal sonification helpers (match playhead_kde_gt_video.py)
def make_click_track(
    sr: int,
    n_samples: int,
    peak_times_wav: np.ndarray,
    tone_hz: float = 1100.0,
    click_ms: float = 12.0,
    amp: float = 0.9,
) -> np.ndarray:
    x = np.zeros(n_samples, dtype=np.float32)
    click_len = max(1, int(round(click_ms / 1000.0 * sr)))
    t = np.arange(click_len, dtype=np.float32) / sr
    burst = np.sin(2 * np.pi * tone_hz * t).astype(np.float32)
    burst *= np.hanning(click_len).astype(np.float32)
    burst *= float(amp)
    for pt in peak_times_wav:
        i0 = int(round(pt * sr))
        if i0 < 0 or i0 >= n_samples:
            continue
        i1 = min(n_samples, i0 + click_len)
        x[i0:i1] += burst[: (i1 - i0)]
    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx > 1.0:
        x /= mx
    return x


def mix_50_50(audio: np.ndarray, clicks: np.ndarray, music_ratio: float = 0.35) -> np.ndarray:
    a = audio.astype(np.float32)
    c = clicks.astype(np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    a_mx = float(np.max(np.abs(a))) if a.size else 1.0
    c_mx = float(np.max(np.abs(c))) if c.size else 1.0
    if a_mx > 0:
        a = a / a_mx
    if c_mx > 0:
        c = c / c_mx
    click_ratio = 1.0 - music_ratio
    y = music_ratio * a + click_ratio * c
    y_mx = float(np.max(np.abs(y))) if y.size else 0.0
    if y_mx > 0:
        y = y / max(1.0, y_mx)
    return y


def load_gt_beats_music(
    dataset: str,
    stem: str,
    t0: float,
    t1: float,
    beats_dir: Path,
    mirex_dir: Path,
    mirex_line: int,
) -> np.ndarray:
    if dataset == "ballroom":
        p = beats_dir / f"{stem}.beats"
        if not p.exists():
            return np.array([], dtype=float)
        b = _r.load_ballroom_beats(p)
        return np.sort(b[(b >= t0) & (b <= t1)].astype(float))
    m = re.match(r"train(\d+)$", stem, re.I)
    if not m:
        return np.array([], dtype=float)
    p = mirex_dir / f"train{m.group(1)}.txt"
    if not p.exists():
        return np.array([], dtype=float)
    b = _r.load_mirex_reference_line(p, t0, t1, line_index=mirex_line)
    return np.sort(np.asarray(b, dtype=float))


def load_kde_peaks_music(peaks_csv: Path, stem: str, t0: float, t1: float) -> np.ndarray:
    """Load peak times in music seconds for ``stem``, clipped to ``[t0, t1]``.

    Peaks CSV rows are tagged with the KDE analysis window ``(window_t0, window_t1)``.
    The listening excerpt ``(t0, t1)`` is often a sub-window (e.g. 10–25 s).

    1. Prefer rows whose window **contains** ``[t0, t1]`` (tightest span if several).
    2. Else exact legacy match on ``(window_t0, window_t1) == (t0, t1)``.
    3. Else **ignore window columns** and use all peaks for this stem, then clip —
       covers short WAVs / partial KDE spans so sonification does not silently drop
       every click.
    """
    df = pd.read_csv(peaks_csv, low_memory=False)
    df = df[df["stimulus_id"].astype(str) == stem]
    if df.empty:
        return np.array([], dtype=float)
    if "window_t0" in df.columns and "window_t1" in df.columns:
        eps = 0.05
        wt0 = df["window_t0"].astype(float)
        wt1 = df["window_t1"].astype(float)
        contain = (wt0 <= t0 + eps) & (wt1 >= t1 - eps)
        df_c = df[contain]
        if df_c.empty:
            exact = ((wt0 - t0).abs() < 0.02) & ((wt1 - t1).abs() < 0.02)
            df_c = df[exact]
        if df_c.empty:
            df_c = df
        else:
            # Multiple containing windows (e.g. re-runs): prefer the tightest span.
            groups = list(df_c.groupby(["window_t0", "window_t1"], sort=False))
            if len(groups) > 1:

                def _span(key: tuple) -> float:
                    a, b = key
                    return float(b) - float(a)

                groups.sort(key=lambda ng: _span(ng[0]))
            df_c = groups[0][1]
        df = df_c
    if df.empty:
        return np.array([], dtype=float)
    pt = np.unique(df["peak_time_s"].astype(float).values)
    return np.sort(pt[(pt >= t0) & (pt <= t1)])


def _dist_to_grid(peaks: np.ndarray, period: float, phase: float) -> np.ndarray:
    """Absolute time error between each peak and nearest point on grid phase + n*period."""
    k = np.round((peaks - phase) / period)
    nearest = phase + k * period
    return np.abs(peaks - nearest)


def _candidate_periods_iois(iois: np.ndarray, ioi_min: float, ioi_max: float) -> list[float]:
    """Derive plausible beat periods from observed inter-peak intervals (handles half/double tempo)."""
    iois = np.asarray(iois, dtype=float)
    iois = iois[(iois >= ioi_min * 0.35) & (iois <= ioi_max * 2.5)]
    if iois.size == 0:
        return []
    cand: set[float] = set()
    for q in (20.0, 50.0, 80.0):
        base = float(np.percentile(iois, q))
        for mult in (0.5, 1.0, 2.0):
            T = base * mult
            if ioi_min <= T <= ioi_max:
                cand.add(T)
    med = float(np.median(iois))
    for mult in (0.5, 1.0, 2.0):
        T = med * mult
        if ioi_min <= T <= ioi_max:
            cand.add(T)
    return sorted(cand)


def filter_kde_peaks_metrical_majority(
    peaks_music: np.ndarray,
    *,
    tol_ms: float = 40.0,
    ioi_min_s: float = 0.28,
    ioi_max_s: float = 1.25,
    phase_grid_steps: int = 36,
    min_fraction_on_grid: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """
    Reduce "double-time" KDE sonification by keeping peaks that agree with a single global
    pulse grid (one tempo + phase) that best explains the majority of peaks.

    1. Propose candidate beat periods from IOI percentiles (incl. half/double).
    2. For each (period, phase), count peaks within ``tol_ms`` of grid ``phase + n*period``.
    3. Pick the best-scoring grid; keep at most one peak per grid index (closest to the line).

    If peaks are too few or no grid reaches ``min_fraction_on_grid``, returns the original peaks.
    """
    t = np.sort(np.asarray(peaks_music, dtype=float))
    meta: dict = {"applied": False, "reason": "", "n_in": int(t.size), "n_out": int(t.size)}
    if t.size < 3:
        meta["reason"] = "too_few_peaks"
        return t, meta

    tol = tol_ms / 1000.0
    iois = np.diff(t)
    periods = _candidate_periods_iois(iois, ioi_min_s, ioi_max_s)
    if not periods:
        T0 = float(np.median(iois)) if iois.size else (t[-1] - t[0]) / max(len(t) - 1, 1)
        T0 = float(np.clip(T0, ioi_min_s, ioi_max_s))
        periods = [T0]

    best_score = -1
    best_T, best_phi = periods[0], 0.0

    for T in periods:
        if T <= 1e-9:
            continue
        # Uniform phases on [0, T): best grid is periodic in phase
        for j in range(phase_grid_steps):
            phi = j * (T / phase_grid_steps)
            d = _dist_to_grid(t, T, phi)
            score = int(np.sum(d <= tol))
            if score > best_score:
                best_score = score
                best_T, best_phi = T, phi

    if best_score < max(3, min_fraction_on_grid * t.size):
        meta["reason"] = "low_grid_consensus"
        return t, meta

    d = _dist_to_grid(t, best_T, best_phi)
    on_grid = d <= tol
    t_g = t[on_grid]
    k = np.round((t_g - best_phi) / best_T).astype(np.int64)
    d_g = d[on_grid]

    order = np.argsort(k)
    t_g, k, d_g = t_g[order], k[order], d_g[order]
    keep_idx: list[int] = []
    i = 0
    n = t_g.size
    while i < n:
        j = i + 1
        while j < n and k[j] == k[i]:
            j += 1
        seg = slice(i, j)
        rel = np.arange(i, j)
        keep_idx.append(rel[np.argmin(d_g[seg])])
        i = j
    out = np.sort(t_g[np.array(keep_idx, dtype=int)])
    meta.update(
        {
            "applied": True,
            "reason": "ok",
            "n_out": int(out.size),
            "period_s": best_T,
            "phase_s": best_phi,
            "grid_tol_ms": tol_ms,
            "on_grid_score": best_score,
        }
    )
    if out.size == 0:
        meta["reason"] = "empty_after_thin"
        meta["applied"] = False
        return t, meta
    return out, meta


def load_madmom_music(madmom_csv: Path, stem: str, t0: float, t1: float) -> np.ndarray:
    df = pd.read_csv(madmom_csv, low_memory=False)
    df = df[df["stimulus_id"].astype(str) == stem]
    if df.empty:
        return np.array([], dtype=float)
    bt = df["beat_time_s_music"].astype(float).values
    return np.sort(bt[(bt >= t0) & (bt <= t1)])


def music_to_wav_times(times_music: np.ndarray, stim_offset_s: float) -> np.ndarray:
    return np.asarray(times_music, dtype=float) + float(stim_offset_s)


def extract_segment(y: np.ndarray, sr: int, wav_t0: float, wav_t1: float) -> np.ndarray:
    i0 = max(0, int(round(wav_t0 * sr)))
    i1 = min(len(y), int(round(wav_t1 * sr)))
    return y[i0:i1].astype(np.float32)


def effective_music_excerpt_bounds(
    n_samples: int, sr: int, t0_music: float, t1_music: float
) -> tuple[float, float, float]:
    """Return ``(t0_use, t1_use, music_end)`` with ``t1`` clipped to how much music fits in the WAV."""
    dur_wav = n_samples / float(sr)
    music_end = dur_wav
    t1_use = float(min(t1_music, music_end))
    t0_use = float(t0_music)
    if t0_use >= t1_use:
        t0_use = max(0.0, t1_use - 0.5)
    return t0_use, t1_use, music_end


def build_segment_with_clicks(
    y_full: np.ndarray,
    sr: int,
    t0_music: float,
    t1_music: float,
    event_times_music: np.ndarray,
) -> np.ndarray:
    """Extract ``[t0_music, t1_music]`` from WAV; clip ``t1`` to available audio.

    WAV files are raw music (no REPP silence); music_time = WAV_time.
    """
    t0_use, t1_use, _ = effective_music_excerpt_bounds(
        len(y_full), sr, t0_music, t1_music
    )
    seg = extract_segment(y_full, sr, t0_use, t1_use)
    n = seg.size
    ev = event_times_music[(event_times_music >= t0_use) & (event_times_music <= t1_use)]
    ev_wav_local = ev - t0_use
    ev_sample = music_to_wav_times(ev_wav_local, 0.0)
    clicks = make_click_track(sr, n, ev_sample)
    return mix_50_50(seg, clicks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_AP_ROOT.parent / "next-steps/tapping-validation/data/stimulus_manifest.csv",
    )
    ap.add_argument(
        "--stems-from-peaks",
        action="store_true",
        help=(
            "Ignore manifest rows; process every unique stimulus_id in --peaks-ballroom and "
            "--peaks-mirex (skips stems with missing WAV). Use --t0-music-default / "
            "--t1-music-default for the excerpt window (default 10–25 s)."
        ),
    )
    ap.add_argument(
        "--t0-music-default",
        type=float,
        default=10.0,
        help="Music start (s) when using --stems-from-peaks (ignored for manifest mode).",
    )
    ap.add_argument(
        "--t1-music-default",
        type=float,
        default=25.0,
        help="Music end (s) when using --stems-from-peaks (ignored for manifest mode).",
    )
    ap.add_argument(
        "--kde-only",
        action="store_true",
        help="Write only {stem}_kde.wav per stem (no GT, madmom, or triple files).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_AP_ROOT.parent / "next-steps/tapping-validation/data/validation_audio",
    )
    ap.add_argument("--stim_offset_s", type=float, default=2.0)
    ap.add_argument(
        "--no-triple",
        action="store_true",
        help="Skip generation of triple (concatenated) WAV permutations — only write per-condition WAVs.",
    )
    ap.add_argument(
        "--gap_s",
        type=float,
        default=1.0,
        help="Silence between versions in triple file (seconds). Try 0.75–1.25; increase if separation feels unclear.",
    )
    ap.add_argument("--wav_ballroom", type=Path, default=_AP_ROOT / "audio/ballroom_wav")
    ap.add_argument("--wav_mirex", type=Path, default=_AP_ROOT / "audio/mirex")
    ap.add_argument("--beats_dir", type=Path, default=_AP_ROOT.parent / "ballroom/BallroomAnnotations-master")
    ap.add_argument("--mirex_dir", type=Path, default=_AP_ROOT / "mirex")
    ap.add_argument(
        "--peaks_ballroom",
        type=Path,
        default=_AP_ROOT / "outputs/repp_gt_mad_only_crossverify/ballroom/cleaned_kde_peaks.csv",
    )
    ap.add_argument(
        "--peaks_mirex",
        type=Path,
        default=_AP_ROOT / "outputs/cleaning_gt_comparison/mirex_mad_only/cleaned_kde_peaks.csv",
    )
    ap.add_argument(
        "--kde_peaks_csv",
        type=Path,
        default=None,
        help=(
            "If set, load KDE click times from this single CSV for every stem (stimulus_id + window), "
            "ignoring --peaks_ballroom / --peaks_mirex for the KDE condition only. "
            "Use e.g. kde_pattern_completion_peaks.py output."
        ),
    )
    ap.add_argument(
        "--madmom_ballroom",
        type=Path,
        default=_AP_ROOT / "outputs/paper_run_locked/ballroom_algo_offset0/madmom_beats.csv",
    )
    ap.add_argument(
        "--madmom_mirex",
        type=Path,
        default=_AP_ROOT / "outputs/paper_run_locked/mirex_algo_offset0/madmom_beats.csv",
    )
    ap.add_argument(
        "--kde-metrical-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For KDE clicks only: project peaks onto a single global pulse grid and thin to "
            "one click per step (can fix double-clicks but often sounds wrong if tempo/phase "
            "is mis-fit). Default: off (raw cleaned_kde_peaks.csv)."
        ),
    )
    ap.add_argument(
        "--kde-metrical-tol-ms",
        type=float,
        default=40.0,
        help="Half-width (ms) for snapping peaks to the chosen metrical grid when --kde-metrical-filter.",
    )
    ap.add_argument(
        "--kde-metrical-min-ioi-s",
        type=float,
        default=0.28,
        help="Minimum allowed beat period (s) when searching grids (~214 BPM).",
    )
    ap.add_argument(
        "--kde-metrical-max-ioi-s",
        type=float,
        default=1.25,
        help="Maximum allowed beat period (s) when searching grids (~48 BPM).",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    perms = list(permutations(["kde", "gt", "madmom"], 3))
    perm_list = [list(p) for p in perms]

    if args.stems_from_peaks:
        jobs: list[tuple[str, str, float, float]] = []
        t0_d, t1_d = float(args.t0_music_default), float(args.t1_music_default)
        for peaks_csv, corpus in (
            (args.peaks_ballroom, "ballroom"),
            (args.peaks_mirex, "mirex"),
        ):
            df = pd.read_csv(peaks_csv, low_memory=False)
            for stem in sorted(df["stimulus_id"].astype(str).unique()):
                jobs.append((stem, corpus, t0_d, t1_d))
        print(f"[stems-from-peaks] {len(jobs)} stems (ballroom then mirex, unique per CSV)")
    else:
        manifest = pd.read_csv(args.manifest)
        jobs = [
            (
                str(row["stimulus_id"]).strip(),
                str(row["corpus"]).strip().lower(),
                float(row["t0_music"]),
                float(row["t1_music"]),
            )
            for _, row in manifest.iterrows()
        ]

    kde_peaks_override = Path(args.kde_peaks_csv).resolve() if args.kde_peaks_csv else None
    kde_qc_rows: list[dict] = []

    for stem, corpus, t0, t1 in jobs:
        if corpus == "ballroom":
            wav_path = args.wav_ballroom / f"{stem}.wav"
            peaks_csv = args.peaks_ballroom
            madmom_csv = args.madmom_ballroom
        else:
            wav_path = args.wav_mirex / f"{stem}.wav"
            peaks_csv = args.peaks_mirex
            madmom_csv = args.madmom_mirex

        if not wav_path.exists():
            print(f"[skip] missing WAV: {wav_path}")
            continue

        y, sr = sf.read(str(wav_path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)

        kde_src = kde_peaks_override if kde_peaks_override is not None else peaks_csv
        kde_m = load_kde_peaks_music(kde_src, stem, t0, t1)
        if args.kde_metrical_filter and kde_m.size:
            kde_m, kde_meta = filter_kde_peaks_metrical_majority(
                kde_m,
                tol_ms=float(args.kde_metrical_tol_ms),
                ioi_min_s=float(args.kde_metrical_min_ioi_s),
                ioi_max_s=float(args.kde_metrical_max_ioi_s),
            )
            if kde_meta.get("applied"):
                print(
                    f"[kde-metrical] {stem}: {kde_meta['n_in']} -> {kde_meta['n_out']} peaks "
                    f"(T≈{kde_meta.get('period_s', 0):.3f}s, tol={kde_meta.get('grid_tol_ms')}ms)"
                )
            elif kde_meta.get("reason") not in ("too_few_peaks",):
                print(f"[kde-metrical] {stem}: skipped ({kde_meta.get('reason')}), using raw peaks")
        kde_seg = build_segment_with_clicks(y, sr, t0, t1, kde_m)
        if args.kde_only:
            t0u, t1u, music_end = effective_music_excerpt_bounds(
                len(y), sr, t0, t1
            )
            n_eff = int(np.sum((kde_m >= t0u) & (kde_m <= t1u))) if kde_m.size else 0
            kde_qc_rows.append(
                {
                    "stimulus_id": stem,
                    "corpus": corpus,
                    "t0_music_requested": t0,
                    "t1_music_requested": t1,
                    "t0_music_in_wav": t0u,
                    "t1_music_in_wav": t1u,
                    "music_end_s": music_end,
                    "n_peaks_loaded_for_request": int(kde_m.size),
                    "n_peaks_in_excerpt_audio": n_eff,
                    "peaks_csv": str(kde_src),
                }
            )
            if kde_m.size == 0:
                print(f"[WARN] {stem}: 0 peaks in CSV for requested window [{t0},{t1}] — silent clicks")
            elif n_eff == 0:
                print(
                    f"[WARN] {stem}: {kde_m.size} peaks loaded but none fall in played excerpt "
                    f"[{t0u:.2f},{t1u:.2f}] (requested [{t0},{t1}], music_end={music_end:.2f}s)"
                )
            out_w = out_dir / f"{stem}_kde.wav"
            sf.write(str(out_w), kde_seg, sr)
            print(f"[OK] {out_w.name}")
            continue

        gt_m = load_gt_beats_music(corpus, stem, t0, t1, args.beats_dir, args.mirex_dir, 0)
        mm_m = load_madmom_music(madmom_csv, stem, t0, t1)

        segs = {
            "kde": kde_seg,
            "gt": build_segment_with_clicks(y, sr, t0, t1, gt_m),
            "madmom": build_segment_with_clicks(y, sr, t0, t1, mm_m),
        }

        for k, seg in segs.items():
            out_w = out_dir / f"{stem}_{k}.wav"
            sf.write(str(out_w), seg, sr)
            print(f"[OK] {out_w.name}")

        if not args.no_triple:
            gap_n = int(round(args.gap_s * sr))
            gap = np.zeros(gap_n, dtype=np.float32)
            meta = {"stimulus_id": stem, "corpus": corpus, "t0_music": t0, "t1_music": t1, "permutations": {}}

            for perm_id, order in enumerate(perm_list):
                parts = [segs[order[0]], gap, segs[order[1]], gap, segs[order[2]]]
                triple = np.concatenate(parts).astype(np.float32)
                tw = out_dir / f"triple_{stem}_{perm_id}.wav"
                sf.write(str(tw), triple, sr)
                meta["permutations"][str(perm_id)] = {
                    "order": order,
                    "section_1": order[0],
                    "section_2": order[1],
                    "section_3": order[2],
                }
                print(f"[OK] {tw.name}")

            with open(out_dir / f"triple_{stem}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    if args.kde_only:
        idx = pd.DataFrame(
            [{"stimulus_id": s, "corpus": c, "t0_music": t0, "t1_music": t1} for s, c, t0, t1 in jobs]
        )
        idx.to_csv(out_dir / "kde_only_manifest.csv", index=False)
        print(f"[OK] {out_dir / 'kde_only_manifest.csv'}")
        if kde_qc_rows:
            qc_path = out_dir / "kde_sonify_peak_qc.csv"
            pd.DataFrame(kde_qc_rows).to_csv(qc_path, index=False)
            print(f"[OK] {qc_path}")

    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
