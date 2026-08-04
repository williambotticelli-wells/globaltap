#!/usr/bin/env python3
"""
REPP tap cleaning, ±hit-window hit/miss vs reference annotation beats,
overlaid KDE + reference vertical lines, sonification.

**Reference annotations**
  - ballroom: .beats | mirex: **one** annotator row from trainN.txt
    (--mirex_gt_annotator)
  - global: no reference annotation for hit/miss

**Cleaning (per trial)**
  1) Tempo gate (optional): median tap IOI → BPM; drop trial if
     |BPM_trial - BPM_ref|/BPM_ref > --tempo_tol.
     BPM_ref = stimulus_metadata (--tempo_gate_bpm_from metadata) or
     60/median(IBI) from reference beats
     (--tempo_gate_bpm_from gt_beats).
  2) Outlier taps: remove tap if adjacent IOI > mad_factor × MAD(IOIs)
     (within trial).

**Hit/miss**: tap within ±hit_tol_ms of nearest reference beat → hit;
  else miss. Compare uncleaned vs cleaned tap streams.

Outputs under --out_dir:
  cleaned_trials.csv              — per trial: n_taps raw/cleaned, tempo_ok, etc.
  hit_miss_per_trial.csv
  hit_miss_summary_per_stimulus.csv
  kde_raster_hit_summary.csv      — peak counts for each KDE variant
  figures/<stem>_kde_over_gt.png      — 4 small KDE panels (legacy)
  figures/<stem>_alignment_clear.png  — readable: metrics box + overlaid KDEs + GT ticks + error hist
  repp_vs_gt_alignment_stages.csv     — tap + KDE alignment per stage (see REPP_CLEANING_AND_ALIGNMENT.md)
  cleaned_kde_peaks.csv               — **primary:** KDE peak times after MAD (± optional tempo gate) cleaning
  (optional) flattened_taps.csv       — run scripts/export_flattened_taps.py same args — 1 row/tap for IOI / phase work
  figures/summary_alignment_heatmap.png
  ALIGNMENT_SUMMARY.md
  sonify/<stem>_reference_beats.wav, _repp_kde_raw.wav, _repp_kde_clean.wav, _stereo_referenceL_reppCleanR.wav

Example (Ballroom, when export-compat CSV exists):
  cd analysis_pipeline && .venv/bin/python scripts/repp_gt_alignment_pipeline.py \\
    --dataset ballroom \\
    --csv path/to/BallroomTap_EXPORT_COMPAT.csv \\
    --beats_dir ../ballroom/BallroomAnnotations-master \\
    --wav_dir audio/ballroom_wav \\
    --metadata_csv ../PROJECT/stimulus_metadata.csv \\
    --out_dir outputs/ballroom_repp_gt \\
    --t0 0 --t1 30 --stim_offset_s 2 --only_nonfailed

Example (MIREX):
  ... --dataset mirex --mirex_dir mirex --wav_dir audio/mirex \\
    --csv merged/merged_mirex/TapTrialMusic_MERGED_EXPORT_COMPAT.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

# --- I/O ---

def safe_json_list(x: Any) -> List[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isnan(x))):
        return []
    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        if not s:
            return []
        try:
            vals = json.loads(s)
        except Exception:
            return []
    if not isinstance(vals, list):
        return []
    out = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def load_ballroom_beats(beats_path: Path) -> np.ndarray:
    t = []
    with open(beats_path) as f:
        for line in f:
            parts = line.split()
            if parts:
                try:
                    t.append(float(parts[0]))
                except ValueError:
                    pass
    return np.array(sorted(t), dtype=float)


def load_mirex_pooled(txt_path: Path, t0: float, t1: float) -> np.ndarray:
    """All beat times from all annotators in window (dense; do not use for ±50ms hit/miss)."""
    all_t: List[float] = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for x in line.split("\t"):
                x = x.strip()
                if not x:
                    continue
                try:
                    v = float(x)
                except ValueError:
                    continue
                if t0 <= v <= t1:
                    all_t.append(v)
    return np.array(sorted(all_t), dtype=float)


def load_mirex_reference_line(txt_path: Path, t0: float, t1: float, line_index: int = 0) -> np.ndarray:
    """One annotator row = beat grid for hit/miss vs REPP (same role as single .beats file)."""
    lines = [ln.strip() for ln in open(txt_path) if ln.strip()]
    if line_index >= len(lines):
        return np.array([], dtype=float)
    times = [float(x) for x in lines[line_index].split("\t") if x.strip()]
    t = np.array(times, dtype=float)
    return t[(t >= t0) & (t <= t1)]


def audio_stem(name: str) -> str:
    # Some raw audio filenames carry a trailing space before the extension
    # (e.g. "10-TunFcFByq28 .wav"). Strip the stem as well so that it matches
    # the tapping CSV _stem column and the GT source map.
    return Path(str(name).strip()).stem.strip()


def gaussian_kde(taps_s: np.ndarray, t0: float, t1: float, kw_ms: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    kw_s = kw_ms / 1000.0
    x = np.arange(t0, t1 + 1e-12, dt, dtype=float)
    taps = np.asarray(taps_s, dtype=float)
    taps = taps[np.isfinite(taps)]
    taps = taps[(taps >= t0) & (taps <= t1)]
    if taps.size == 0:
        return x, np.zeros_like(x)
    dx = (x[:, None] - taps[None, :]) / kw_s
    y = np.exp(-0.5 * dx * dx) / (kw_s * math.sqrt(2 * math.pi))
    return x, y.sum(axis=1)


def kde_peaks(
    t_grid: np.ndarray,
    y: np.ndarray,
    kde_dt: float,
    prominence: float,
    min_dist_s: float,
    min_width_s: float = 0.0,
) -> np.ndarray:
    if find_peaks is None or y.size < 4:
        return np.array([], dtype=float)
    yn = y / (np.max(y) + 1e-12)
    md = max(1, int(round(min_dist_s / kde_dt)))
    kwargs: Dict[str, Any] = {"prominence": prominence, "distance": md}
    if float(min_width_s) > 0:
        kwargs["width"] = max(1, int(round(float(min_width_s) / kde_dt)))
    idx, _ = find_peaks(yn, **kwargs)
    return t_grid[np.asarray(idx, dtype=int)]


def taps_ms_to_s(taps_ms: List[float], stim_offset_s: float) -> np.ndarray:
    return np.array(taps_ms, dtype=float) / 1000.0 - stim_offset_s


def median_bpm_from_taps_s(taps: np.ndarray, t0: float, t1: float) -> Optional[float]:
    t = taps[(taps >= t0) & (taps <= t1)]
    t = np.sort(t)
    if t.size < 2:
        return None
    ioi = np.diff(t)
    ioi = ioi[ioi > 0.05]
    if ioi.size < 1:
        return None
    med = float(np.median(ioi))
    if med <= 0:
        return None
    return 60.0 / med


def bpm_from_annotated_beats(beats_s: np.ndarray) -> Optional[float]:
    """Reference tempo (BPM) from expert beat times in the analysis window: 60 / median(IBI)."""
    b = np.sort(np.asarray(beats_s, dtype=float))
    b = b[np.isfinite(b)]
    if b.size < 2:
        return None
    ioi = np.diff(b)
    ioi = ioi[ioi > 0.02]
    if ioi.size < 1:
        return None
    med = float(np.median(ioi))
    if med <= 0:
        return None
    return 60.0 / med


def clean_taps_trial(
    taps_s: np.ndarray,
    t0: float,
    t1: float,
    bpm_gt: Optional[float],
    tempo_rel_tol: float,
    mad_factor: float,
    skip_tempo_gate: bool = False,
    tempo_ref_label: str = "metadata",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns cleaned taps in music seconds and info dict."""
    info: Dict[str, Any] = {"tempo_ok": True, "n_in": int(taps_s.size), "n_out": 0, "reason": ""}
    t = taps_s[np.isfinite(taps_s)]
    t = t[(t >= t0) & (t <= t1)]
    t = np.sort(np.unique(t))
    if t.size < 2:
        info["n_out"] = int(t.size)
        return t, info

    if not skip_tempo_gate and bpm_gt and bpm_gt > 0:
        bpm_tr = median_bpm_from_taps_s(t, t0, t1)
        if bpm_tr is not None:
            if abs(bpm_tr - bpm_gt) / bpm_gt > tempo_rel_tol:
                info["tempo_ok"] = False
                info["reason"] = f"bpm_trial={bpm_tr:.1f} vs ref={bpm_gt:.1f} [{tempo_ref_label}]"
                return np.array([], dtype=float), info

    # MAD outlier removal on interior taps (by IOI violation)
    for _ in range(3):
        if t.size < 4:
            break
        ioi = np.diff(t)
        med = np.median(ioi)
        mad = np.median(np.abs(ioi - med)) + 1e-9
        if mad < 1e-6:
            break
        bad = np.zeros(len(t), dtype=bool)
        for i in range(1, len(t) - 1):
            left_ioi = t[i] - t[i - 1]
            right_ioi = t[i + 1] - t[i]
            if left_ioi > med + mad_factor * mad * 1.4826 or right_ioi > med + mad_factor * mad * 1.4826:
                bad[i] = True
        if not bad.any():
            break
        t = t[~bad]
    info["n_out"] = int(t.size)
    return t, info


def hit_miss(taps_s: np.ndarray, beats_s: np.ndarray, tol_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: hit_mask (bool per tap), nearest_beat_s, asynchrony_s (tap - beat) for hits only nan for miss.
    """
    taps = np.sort(np.asarray(taps_s, dtype=float))
    beats = np.sort(np.asarray(beats_s, dtype=float))
    hit = np.zeros(len(taps), dtype=bool)
    nearest = np.full(len(taps), np.nan)
    async_s = np.full(len(taps), np.nan)
    if beats.size == 0:
        return hit, nearest, async_s
    for i, tt in enumerate(taps):
        j = np.searchsorted(beats, tt)
        cand = []
        if j > 0:
            cand.append(beats[j - 1])
        if j < len(beats):
            cand.append(beats[j])
        if not cand:
            continue
        b = min(cand, key=lambda x: abs(tt - x))
        d = abs(tt - b)
        nearest[i] = b
        if d <= tol_s:
            hit[i] = True
            async_s[i] = tt - b
    return hit, nearest, async_s


def tap_vs_gt_alignment(
    taps_s: np.ndarray, beats_s: np.ndarray, tol_s: float
) -> Dict[str, Any]:
    """Pooled taps vs GT beats: hit rate and asynchrony on hits (ms)."""
    taps = np.asarray(taps_s, dtype=float)
    beats = np.asarray(beats_s, dtype=float)
    out: Dict[str, Any] = {
        "n_taps": int(taps.size),
        "hit_rate": np.nan,
        "mean_async_ms_signed_on_hits": np.nan,
        "mean_abs_async_ms_on_hits": np.nan,
        "std_async_ms_on_hits": np.nan,
        "rms_async_ms_on_hits": np.nan,
    }
    if taps.size == 0 or beats.size == 0:
        return out
    hit, _, async_s = hit_miss(taps, beats, tol_s)
    ah = async_s[hit] * 1000.0
    out["hit_rate"] = float(np.mean(hit))
    if ah.size:
        out["mean_async_ms_signed_on_hits"] = float(np.mean(ah))
        out["mean_abs_async_ms_on_hits"] = float(np.mean(np.abs(ah)))
        out["std_async_ms_on_hits"] = float(np.std(ah, ddof=1)) if ah.size > 1 else 0.0
        out["rms_async_ms_on_hits"] = float(np.sqrt(np.mean(ah ** 2)))
    return out


def peaks_vs_gt_alignment(peaks_s: np.ndarray, beats_s: np.ndarray, tol_s: float) -> Dict[str, Any]:
    """Each KDE peak → nearest GT beat; report % within tol and |offset| stats (ms)."""
    peaks = np.sort(np.asarray(peaks_s, dtype=float))
    beats = np.sort(np.asarray(beats_s, dtype=float))
    tol_ms = tol_s * 1000.0
    out = {
        "n_peaks": int(peaks.size),
        "pct_peaks_within_tol_of_beat": np.nan,
        "mean_abs_peak_to_beat_ms": np.nan,
        "median_abs_peak_to_beat_ms": np.nan,
    }
    if peaks.size == 0 or beats.size == 0:
        return out
    dist_ms = []
    for p in peaks:
        j = int(np.searchsorted(beats, p))
        cand = []
        if j > 0:
            cand.append(abs(p - beats[j - 1]) * 1000.0)
        if j < len(beats):
            cand.append(abs(p - beats[j]) * 1000.0)
        dist_ms.append(min(cand) if cand else 1e9)
    dist_ms = np.array(dist_ms, dtype=float)
    out["pct_peaks_within_tol_of_beat"] = float(100.0 * np.mean(dist_ms <= tol_ms))
    out["mean_abs_peak_to_beat_ms"] = float(np.mean(dist_ms))
    out["median_abs_peak_to_beat_ms"] = float(np.median(dist_ms))
    return out


def make_clicks(sr: int, n: int, times_s: np.ndarray, hz: float = 1000.0, click_ms: float = 10.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    L = max(1, int(round(click_ms / 1000 * sr)))
    burst = (np.sin(2 * np.pi * hz * np.arange(L) / sr) * np.hanning(L)).astype(np.float32) * 0.9
    for pt in times_s:
        i0 = int(round(float(pt) * sr))
        if 0 <= i0 < n:
            i1 = min(n, i0 + L)
            x[i0:i1] += burst[: i1 - i0]
    m = float(np.max(np.abs(x))) if x.size else 1.0
    if m > 1:
        x /= m
    return x


def mix_audio_clicks(audio: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    a = audio.astype(np.float32).ravel() if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
    c = clicks.astype(np.float32)
    if np.max(np.abs(a)) > 0:
        a = a / np.max(np.abs(a))
    if np.max(np.abs(c)) > 0:
        c = c / np.max(np.abs(c))
    y = 0.5 * a + 0.5 * c
    y /= max(1.0, float(np.max(np.abs(y))))
    return y


def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def find_wav(wav_dir: Path, stem: str) -> Optional[Path]:
    for p in [wav_dir / f"{stem}.wav", wav_dir / f"{stem}.WAV"]:
        if p.exists():
            return p
    hits = list(wav_dir.rglob(f"{stem}.wav"))
    return hits[0] if hits else None


def wav_duration_seconds(wav_path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        return float(sf.info(str(wav_path)).duration)
    except Exception:
        return None


def clip_times(arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
    a = np.sort(np.asarray(arr, dtype=float))
    a = a[np.isfinite(a)]
    return a[(a >= t0) & (a <= t1)]


def build_stem_kde_and_metrics_windows(
    stems: List[str],
    wav_dir: Path,
    kde_full_wav: bool,
    fallback_t0: float,
    fallback_t1: float,
    metrics_t0: Optional[float],
    metrics_t1: Optional[float],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """KDE = tap pool + smoothing grid; metrics = GT / hit-miss / peak–GT eval / figure xlim."""
    stem_kde: Dict[str, Tuple[float, float]] = {}
    for stem in stems:
        if kde_full_wav:
            wp = find_wav(wav_dir, stem)
            d = wav_duration_seconds(wp) if wp else None
            if d is None or not math.isfinite(d) or d <= 0.05:
                stem_kde[stem] = (fallback_t0, fallback_t1)
            else:
                stem_kde[stem] = (0.0, float(d))
        else:
            stem_kde[stem] = (fallback_t0, fallback_t1)
    stem_metrics: Dict[str, Tuple[float, float]] = {}
    for stem, (k0, k1) in stem_kde.items():
        if metrics_t0 is not None and metrics_t1 is not None:
            stem_metrics[stem] = (float(metrics_t0), float(metrics_t1))
        else:
            stem_metrics[stem] = (k0, k1)
    return stem_kde, stem_metrics


def main():
    ap = argparse.ArgumentParser(description="REPP vs GT: clean, hit/miss, KDE, sonify")
    ap.add_argument("--dataset", choices=["ballroom", "mirex", "global", "benchmark100"], required=True)
    ap.add_argument("--csv", type=Path, required=True, help="Export-compat CSV")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--metadata_csv", type=Path, default=None, help="stimulus_metadata.csv for BPM")
    ap.add_argument("--beats_dir", type=Path, default=None, help="Ballroom .beats directory")
    ap.add_argument("--mirex_dir", type=Path, default=None)
    # --- benchmark100 support (additive; does not affect other datasets) ---
    ap.add_argument(
        "--benchmark100_gt_map",
        type=Path,
        default=None,
        help="Path to gt_source_map_benchmark100.json (see pipeline_benchmark_validation/config/).",
    )
    ap.add_argument(
        "--benchmark100_data_root",
        type=Path,
        default=Path("${PROJECT_ROOT}"),
        help="Project root used to resolve relative reference-annotator paths in the GT map.",
    )
    ap.add_argument(
        "--mirex_gt_annotator",
        type=int,
        default=0,
        help="MIREX .txt row index (0=first annotator) used as beat GT for hit/miss + vlines",
    )
    ap.add_argument("--wav_dir", type=Path, required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=35.0)
    ap.add_argument(
        "--kde_full_wav",
        action="store_true",
        help=(
            "Pool taps and fit KDE on full WAV length per stem (music axis [0, duration]). "
            "Missing WAV falls back to --t0/--t1. Use with --metrics_t0/--metrics_t1 to evaluate "
            "GT alignment on a shorter excerpt (e.g. 10–25 s) while peaks cover the whole file."
        ),
    )
    ap.add_argument(
        "--metrics_t0",
        type=float,
        default=None,
        help="Optional excerpt for GT, hit/miss, KDE→GT stages, figure xlim (music s). Pair with --metrics_t1.",
    )
    ap.add_argument(
        "--metrics_t1",
        type=float,
        default=None,
        help="Optional excerpt end (music s). When omitted, metrics window matches KDE window per stem.",
    )
    ap.add_argument("--stim_offset_s", type=float, default=2.0)
    ap.add_argument("--hit_tol_ms", type=float, default=50.0)
    ap.add_argument("--tempo_tol", type=float, default=0.30, help="Max rel BPM deviation vs metadata")
    ap.add_argument(
        "--no_tempo_gate",
        action="store_true",
        help="Skip tempo gate entirely; keep only MAD outlier removal.",
    )
    ap.add_argument(
        "--tempo_gate_bpm_from",
        choices=["metadata", "gt_beats"],
        default="metadata",
        help="Reference BPM for tempo gate: stimulus_metadata bpm, or 60/median(IBI) from beat GT in [t0,t1] "
        "(falls back to metadata if <2 beats). Ignored with --no_tempo_gate.",
    )
    ap.add_argument("--mad_factor", type=float, default=3.5)
    ap.add_argument("--only_nonfailed", action="store_true")
    ap.add_argument("--failed_col", default="analysis_failed")
    ap.add_argument("--kw_ms", type=float, default=80.0)
    ap.add_argument("--kde_dt", type=float, default=0.005)
    ap.add_argument("--peak_prominence", type=float, default=0.2)
    ap.add_argument("--peak_distance_s", type=float, default=0.15)
    ap.add_argument(
        "--peak_width_s",
        type=float,
        default=0.0,
        help="Optional minimum peak width in seconds for scipy.signal.find_peaks (0 disables width filter).",
    )
    ap.add_argument("--skip_sonify", action="store_true")
    ap.add_argument("--skip_figures", action="store_true",
                    help="Skip per-stem overlay PNGs and summary figures when testing parameter combinations.")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument(
        "--sonify_outlier_max_pct",
        type=float,
        default=42.0,
        help="Extra sonify in sonify_outliers/ when %% KDE clean peaks within hit_tol falls below this",
    )
    ap.add_argument(
        "--sonify_outlier_regress_pts",
        type=float,
        default=12.0,
        help="Also sonify if clean KDE peak %% within tol drops by this many points vs raw KDE peaks",
    )
    args = ap.parse_args()
    if (args.metrics_t0 is None) ^ (args.metrics_t1 is None):
        raise SystemExit("[ERROR] Pass both --metrics_t0 and --metrics_t1, or neither")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    son_dir = out / "sonify"
    fig_dir.mkdir(exist_ok=True)
    son_dir.mkdir(exist_ok=True)

    meta_bpm: Dict[str, float] = {}
    if args.metadata_csv and Path(args.metadata_csv).exists():
        mdf = pd.read_csv(args.metadata_csv)
        for _, r in mdf.iterrows():
            ds = str(r.get("dataset", "")).lower()
            if ds != args.dataset:
                continue
            sid = str(r.get("stimulus_id", ""))
            b = r.get("bpm", "")
            try:
                if b != "" and not (isinstance(b, float) and math.isnan(b)):
                    meta_bpm[sid] = float(b)
            except Exception:
                pass

    df = pd.read_csv(args.csv, low_memory=False)
    if args.only_nonfailed and args.failed_col in df.columns:
        df = df.loc[df[args.failed_col].eq(False)].copy()
    if "participant_id" not in df.columns or "audio_filename" not in df.columns:
        raise SystemExit("CSV needs participant_id, audio_filename")
    if "resp_onsets_aligned" not in df.columns:
        raise SystemExit("CSV needs resp_onsets_aligned")

    df["_stem"] = df["audio_filename"].apply(audio_stem)
    hit_tol_s = args.hit_tol_ms / 1000.0

    stems_list = sorted({str(s) for s in df["_stem"].unique()})
    stem_kde, stem_metrics = build_stem_kde_and_metrics_windows(
        stems_list,
        Path(args.wav_dir),
        bool(args.kde_full_wav),
        float(args.t0),
        float(args.t1),
        args.metrics_t0,
        args.metrics_t1,
    )

    # GT map (clipped to per-stem metrics window)
    gt_map: Dict[str, np.ndarray] = {}
    if args.dataset == "ballroom":
        if not args.beats_dir:
            raise SystemExit("--beats_dir required for ballroom")
        bd = Path(args.beats_dir)
        for stem in stems_list:
            m0, m1 = stem_metrics[stem]
            p = bd / f"{stem}.beats"
            if p.exists():
                t = load_ballroom_beats(p)
                gt_map[stem] = t[(t >= m0) & (t <= m1)]
    elif args.dataset == "mirex":
        if not args.mirex_dir:
            raise SystemExit("--mirex_dir required for mirex")
        md = Path(args.mirex_dir)
        for stem in stems_list:
            m = re.match(r"train(\d+)$", stem, re.I)
            if not m:
                continue
            p = md / f"train{m.group(1)}.txt"
            if p.exists():
                m0, m1 = stem_metrics[stem]
                gt_map[stem] = load_mirex_reference_line(
                    p, m0, m1, line_index=args.mirex_gt_annotator
                )
    elif args.dataset == "benchmark100":
        if not args.benchmark100_gt_map:
            raise SystemExit("--benchmark100_gt_map required for benchmark100")
        # Import the shared GT loader without touching the repo layout.
        _pipeline_root = (
            Path(__file__).resolve().parent.parent / "pipeline_benchmark_validation"
        )
        if str(_pipeline_root) not in sys.path:
            sys.path.insert(0, str(_pipeline_root))
        from lib.gt_loader import load_gt_for_stem  # type: ignore
        source_map = json.loads(Path(args.benchmark100_gt_map).read_text())
        data_root = Path(args.benchmark100_data_root)
        for stem in stems_list:
            m0, m1 = stem_metrics[stem]
            if stem not in source_map:
                continue
            beats = load_gt_for_stem(stem, source_map, data_root)
            if beats is None or beats.size == 0:
                continue
            t = beats[:, 0]
            gt_map[stem] = t[(t >= m0) & (t <= m1)]

    rows_trial = []
    rows_hit_trial = []
    by_stem: Dict[str, Dict[str, List]] = {}

    for stem in stems_list:
        by_stem[stem] = {"raw": [], "clean": []}

    for idx, row in df.iterrows():
        stem = str(row["_stem"])
        k0, k1 = stem_kde[stem]
        m0, m1 = stem_metrics[stem]
        taps_ms = safe_json_list(row.get("resp_onsets_aligned"))
        taps_s = taps_ms_to_s(taps_ms, args.stim_offset_s)
        beats_w = gt_map.get(stem, np.array([]))
        if args.no_tempo_gate:
            bpm_ref = None
            ref_src = "none_mad_only"
        elif args.tempo_gate_bpm_from == "gt_beats":
            bpm_ref = bpm_from_annotated_beats(beats_w)
            if bpm_ref is not None and bpm_ref > 0:
                ref_src = "gt_beats"
            else:
                bpm_ref = meta_bpm.get(stem)
                ref_src = "metadata_fallback"
        else:
            bpm_ref = meta_bpm.get(stem)
            ref_src = "metadata"

        t_clean, cinfo = clean_taps_trial(
            taps_s,
            k0,
            k1,
            bpm_ref,
            args.tempo_tol,
            args.mad_factor,
            skip_tempo_gate=bool(args.no_tempo_gate),
            tempo_ref_label=ref_src,
        )
        rows_trial.append({
            "row_idx": idx,
            "participant_id": row.get("participant_id"),
            "stimulus_id": stem,
            "n_taps_raw": int(np.sum((taps_s >= k0) & (taps_s <= k1))),
            "n_taps_clean": cinfo["n_out"],
            "tempo_ok": cinfo["tempo_ok"],
            "clean_reason": cinfo.get("reason", ""),
            "tempo_ref_bpm": float(bpm_ref) if bpm_ref else "",
            "tempo_ref_source": ref_src,
        })

        beats = gt_map.get(stem, np.array([]))
        if args.dataset == "global":
            beats = np.array([])

        tr_kde = taps_s[(taps_s >= k0) & (taps_s <= k1)]
        by_stem[stem]["raw"].extend(tr_kde.tolist())
        by_stem[stem]["clean"].extend(t_clean.tolist())

        tr = taps_s[(taps_s >= m0) & (taps_s <= m1)]
        t_clean_m = t_clean[(t_clean >= m0) & (t_clean <= m1)]
        for label, tvec in [("raw", tr), ("clean", t_clean_m)]:
            hit, _, async_s = hit_miss(tvec, beats, hit_tol_s)
            n_hit = int(np.sum(hit))
            n_tot = int(len(tvec))
            async_hit = async_s[hit]
            rows_hit_trial.append({
                "row_idx": idx,
                "participant_id": row.get("participant_id"),
                "stimulus_id": stem,
                "variant": label,
                "n_taps": n_tot,
                "n_hits": n_hit,
                "hit_rate": n_hit / n_tot if n_tot else 0.0,
                "mean_async_ms_hits": float(np.mean(async_hit) * 1000) if async_hit.size else np.nan,
                "std_async_ms_hits": float(np.std(async_hit, ddof=1) * 1000) if async_hit.size > 1 else (0.0 if async_hit.size == 1 else np.nan),
            })

    for stem in by_stem:
        d = by_stem[stem]
        m0, m1 = stem_metrics[stem]
        beats = gt_map.get(stem, np.array([]))
        raw_all = np.array(d["raw"], dtype=float)
        clean_all = np.array(d["clean"], dtype=float)
        raw_m = raw_all[(raw_all >= m0) & (raw_all <= m1)]
        clean_m = clean_all[(clean_all >= m0) & (clean_all <= m1)]
        if beats.size and raw_m.size:
            h, _, _ = hit_miss(raw_m, beats, hit_tol_s)
            d["hits_raw"] = raw_m[h].tolist()
        else:
            d["hits_raw"] = []
        if beats.size and clean_m.size:
            h, _, _ = hit_miss(clean_m, beats, hit_tol_s)
            d["hits_clean"] = clean_m[h].tolist()
        else:
            d["hits_clean"] = []

    pd.DataFrame(rows_trial).to_csv(out / "cleaned_trials.csv", index=False)
    pd.DataFrame(rows_hit_trial).to_csv(out / "hit_miss_per_trial.csv", index=False)

    # Per-stimulus summary
    summ = []
    kde_rows = []
    all_stage_rows: List[Dict[str, Any]] = []
    peak_clean_rows: List[Dict[str, Any]] = []
    hmt = pd.DataFrame(rows_hit_trial)
    for stem in sorted(by_stem.keys()):
        k0, k1 = stem_kde[stem]
        m0, m1 = stem_metrics[stem]
        beats = gt_map.get(stem, np.array([]))
        for variant in ["raw", "clean"]:
            sub = hmt[(hmt["stimulus_id"] == stem) & (hmt["variant"] == variant)]
            summ.append({
                "stimulus_id": stem,
                "variant": variant,
                "trials": len(sub),
                "mean_hit_rate": float(sub["hit_rate"].mean()) if len(sub) else np.nan,
                "mean_async_ms_hits": float(
                    np.nanmean(sub["mean_async_ms_hits"].astype(float))
                ) if len(sub) else np.nan,
                "n_gt_beats_window": int(beats.size),
            })

        d = by_stem[stem]
        variants = [
            ("kde_raw", np.array(d["raw"], dtype=float)),
            ("kde_clean", np.array(d["clean"], dtype=float)),
            ("kde_hits_raw", np.array(d["hits_raw"], dtype=float)),
            ("kde_hits_clean", np.array(d["hits_clean"], dtype=float)),
        ]
        pk_store: Dict[str, np.ndarray] = {}
        for name, taps in variants:
            tg, y = gaussian_kde(taps, k0, k1, args.kw_ms, args.kde_dt)
            pk = kde_peaks(
                tg, y, args.kde_dt, args.peak_prominence, args.peak_distance_s, args.peak_width_s
            )
            pk_store[name] = pk
            kde_rows.append({"stimulus_id": stem, "kde_variant": name, "n_taps_pooled": int(taps.size), "n_kde_peaks": int(pk.size)})

        pk_cln_final = pk_store.get("kde_clean", np.array([], dtype=float))
        for k_idx, pt in enumerate(pk_cln_final):
            peak_clean_rows.append({
                "stimulus_id": stem,
                "peak_index": k_idx,
                "peak_time_s": float(pt),
                "window_t0": k0,
                "window_t1": k1,
                "kw_ms": args.kw_ms,
                "kde_dt": args.kde_dt,
                "peak_prominence": args.peak_prominence,
                "peak_distance_s": args.peak_distance_s,
                "peak_width_s": args.peak_width_s,
                "cleaning": "mad_only" if args.no_tempo_gate else "tempo_gate_plus_mad",
            })

        raw_m_pool = clip_times(np.array(d["raw"], dtype=float), m0, m1)
        clean_m_pool = clip_times(np.array(d["clean"], dtype=float), m0, m1)
        if beats.size:
            raw_a = tap_vs_gt_alignment(raw_m_pool, beats, hit_tol_s)
            cl_a = tap_vs_gt_alignment(clean_m_pool, beats, hit_tol_s)
            for stage_name, a in [
                ("1_pooled_raw_taps_vs_gt", raw_a),
                ("2_pooled_cleaned_taps_vs_gt", cl_a),
            ]:
                all_stage_rows.append({"stimulus_id": stem, "stage": stage_name, **a})
            pk_raw = pk_store.get("kde_raw", np.array([]))
            pk_cln = pk_store.get("kde_clean", np.array([]))
            pk_hr = pk_store.get("kde_hits_raw", np.array([]))
            pk_hc = pk_store.get("kde_hits_clean", np.array([]))
            for stage_name, pk in [
                ("3_kde_peaks_from_raw_taps", pk_raw),
                ("4_kde_peaks_from_clean_taps", pk_cln),
                ("5_kde_peaks_from_hits_raw_pool", pk_hr),
                ("6_kde_peaks_from_hits_clean_pool", pk_hc),
            ]:
                pv = peaks_vs_gt_alignment(clip_times(pk, m0, m1), beats, hit_tol_s)
                all_stage_rows.append({"stimulus_id": stem, "stage": stage_name, **pv})

        # --- Readable single-stimulus figure ---
        # When --skip_figures is set we bypass all per-stem plotting AND the
        # sonify step (both only consume CPU, and no other CSVs depend on them).
        if args.skip_figures:
            continue
        fig_c = plt.figure(figsize=(11, 9), dpi=args.dpi)
        gs = fig_c.add_gridspec(3, 1, height_ratios=[1.15, 2.2, 1.35], hspace=0.32)
        ax0 = fig_c.add_subplot(gs[0, 0])
        ax0.axis("off")
        if beats.size:
            lines = [
                f"{stem}  |  GT beats: {beats.size}  |  hit window: ±{args.hit_tol_ms:.0f} ms",
                "",
                "Stage 1 — All REPP taps (raw):     hit_rate=%.1f%%  |  mean|async| on hits=%.1f ms  |  RMS=%.1f ms"
                % (
                    raw_a.get("hit_rate", 0) * 100,
                    _num(raw_a.get("mean_abs_async_ms_on_hits")),
                    _num(raw_a.get("rms_async_ms_on_hits")),
                ),
                "Stage 2 — After cleaning:          hit_rate=%.1f%%  |  mean|async|=%.1f ms  |  RMS=%.1f ms"
                % (
                    cl_a.get("hit_rate", 0) * 100,
                    _num(cl_a.get("mean_abs_async_ms_on_hits")),
                    _num(cl_a.get("rms_async_ms_on_hits")),
                ),
                "Stage 3 — KDE peaks (raw pool):    %.0f%% peaks within ±%.0f ms of a beat  |  median |Δ|=%.1f ms"
                % (
                    _num(peaks_vs_gt_alignment(clip_times(pk_raw, m0, m1), beats, hit_tol_s)["pct_peaks_within_tol_of_beat"]),
                    args.hit_tol_ms,
                    _num(peaks_vs_gt_alignment(clip_times(pk_raw, m0, m1), beats, hit_tol_s)["median_abs_peak_to_beat_ms"]),
                ),
                "Stage 4 — KDE peaks (clean pool):  %.0f%%  |  median |Δ|=%.1f ms"
                % (
                    _num(peaks_vs_gt_alignment(clip_times(pk_cln, m0, m1), beats, hit_tol_s)["pct_peaks_within_tol_of_beat"]),
                    _num(peaks_vs_gt_alignment(clip_times(pk_cln, m0, m1), beats, hit_tol_s)["median_abs_peak_to_beat_ms"]),
                ),
                "Stage 5 — KDE on hits-only (raw):   %.0f%% peaks @ beat ±%.0f ms"
                % (_num(peaks_vs_gt_alignment(clip_times(pk_hr, m0, m1), beats, hit_tol_s)["pct_peaks_within_tol_of_beat"]), args.hit_tol_ms),
                "Stage 6 — KDE on hits-only (clean): %.0f%%"
                % (_num(peaks_vs_gt_alignment(clip_times(pk_hc, m0, m1), beats, hit_tol_s)["pct_peaks_within_tol_of_beat"])),
                "",
                (
                    "Cleaning: MAD-only on IOIs (no whole-trial BPM drop). See REPP_CLEANING_AND_ALIGNMENT.md"
                    if args.no_tempo_gate
                    else "Cleaning: (A) tempo gate vs metadata BPM  (B) MAD on IOIs. See REPP_CLEANING_AND_ALIGNMENT.md"
                ),
            ]
        else:
            lines = [stem, "No GT in window — alignment stages skipped"]
        ax0.text(0.02, 0.98, "\n".join(lines), transform=ax0.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace")

        ax1 = fig_c.add_subplot(gs[1, 0])
        raw_t = np.array(d["raw"], dtype=float)
        cl_t = np.array(d["clean"], dtype=float)
        tg1, y1 = gaussian_kde(raw_t, k0, k1, args.kw_ms, args.kde_dt)
        tg2, y2 = gaussian_kde(cl_t, k0, k1, args.kw_ms, args.kde_dt)
        y1n = y1 / (np.max(y1) + 1e-12)
        y2n = y2 / (np.max(y2) + 1e-12)
        ax1.fill_between(tg1, 0, y1n, alpha=0.35, color="C0", label="KDE raw taps")
        ax1.plot(tg2, y2n, color="C1", lw=2.2, label="KDE cleaned taps")
        if beats.size:
            yb = -0.08
            ax1.scatter(beats, np.full_like(beats, yb), marker="|", s=220, c="darkred", linewidths=2, label="GT beat", zorder=6)
        ax1.set_ylim(min(-0.12, float(np.min(y1n)) * 0.5 if y1n.size else -0.12), 1.08)
        ax1.set_xlim(m0, m1)
        ax1.set_ylabel("norm KDE")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Overlaid KDEs + GT beats (ticks at bottom)")

        ax2 = fig_c.add_subplot(gs[2, 0], sharex=ax1)
        if beats.size and raw_m_pool.size:
            _, _, async_r = hit_miss(raw_m_pool, beats, hit_tol_s)
            _, _, async_c = hit_miss(clean_m_pool, beats, hit_tol_s)
            er = np.abs(async_r[np.isfinite(async_r)]) * 1000.0
            ec = np.abs(async_c[np.isfinite(async_c)]) * 1000.0
            if er.size or ec.size:
                mx = max(50.0, float(np.nanmax(np.r_[er, ec])) * 1.1 if (er.size or ec.size) else 50)
                bins = np.linspace(0, min(mx, args.hit_tol_ms * 3), 40)
                if er.size:
                    ax2.hist(er, bins=bins, alpha=0.5, color="C0", label=f"|async| hits raw (n={er.size})", density=True)
                if ec.size:
                    ax2.hist(ec, bins=bins, alpha=0.5, color="C1", label=f"|async| hits clean (n={ec.size})", density=True)
                ax2.axvline(args.hit_tol_ms, color="k", ls="--", lw=1, label="hit cutoff")
        ax2.set_xlabel("|Asynchrony| on hits only — distance tap → nearest GT beat (ms)")
        ax2.set_ylabel("density")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig_c.savefig(fig_dir / f"{stem}_alignment_clear.png", bbox_inches="tight")
        plt.close(fig_c)

        # KDE + peak times (raw vs cleaned) on one axis
        pk_r = pk_store.get("kde_raw", np.array([], dtype=float))
        pk_c = pk_store.get("kde_clean", np.array([], dtype=float))
        fig_pk = plt.figure(figsize=(12, 4.2), dpi=args.dpi)
        axp = fig_pk.add_subplot(1, 1, 1)
        axp.fill_between(tg1, 0, y1n, alpha=0.35, color="C0", label="KDE (raw taps)")
        axp.plot(tg2, y2n, color="C1", lw=2.0, label="KDE (clean taps)")
        ymax = float(max(1.0, np.max(y1n) if y1n.size else 1.0, np.max(y2n) if y2n.size else 1.0))
        for p in pk_r:
            axp.axvline(p, color="C0", alpha=0.55, lw=1.0, ymin=0.72, ymax=1.0)
        for p in pk_c:
            axp.axvline(p, color="C3", alpha=0.9, lw=1.4, ymin=0.82, ymax=1.0)
        axp.scatter(pk_r, np.full_like(pk_r, ymax * 0.98), c="C0", s=22, zorder=5, label=f"peaks raw (n={len(pk_r)})")
        axp.scatter(pk_c, np.full_like(pk_c, ymax * 0.92), c="C3", marker="^", s=28, zorder=5, label=f"peaks clean (n={len(pk_c)})")
        if beats.size:
            axp.scatter(beats, np.full_like(beats, -0.06 * ymax), marker="|", s=180, c="darkred", linewidths=1.5, label="GT beat")
            axp.set_ylim(-0.12 * ymax, ymax * 1.05)
        else:
            axp.set_ylim(0, ymax * 1.05)
        axp.set_xlim(m0, m1)
        axp.set_xlabel("Time (s)")
        axp.set_ylabel("norm KDE + peak markers")
        axp.set_title(f"{stem} | KDE overlaid; ticks=KDE peaks (blue raw, red clean); | = GT")
        axp.legend(loc="upper right", fontsize=8, ncol=2)
        axp.grid(True, alpha=0.25)
        fig_pk.savefig(fig_dir / f"{stem}_kde_peaks_raw_vs_clean.png", bbox_inches="tight")
        plt.close(fig_pk)

        # Figure (legacy 4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=args.dpi)
        fig.suptitle(
            f"{stem} | KDE [{k0:.1f}–{k1:.1f}s] | plot/GT excerpt [{m0:.1f}–{m1:.1f}s] | {args.dataset}"
        )

        panels = [
            (axes[0, 0], np.array(d["raw"], dtype=float), "KDE all taps (uncleaned)"),
            (axes[0, 1], np.array(d["clean"], dtype=float), "KDE cleaned"),
            (axes[1, 0], np.array(d["hits_raw"], dtype=float), "KDE hits only (uncleaned pool)"),
            (axes[1, 1], np.array(d["hits_clean"], dtype=float), "KDE hits only (cleaned pool)"),
        ]
        for ax, taps, title in panels:
            tg, y = gaussian_kde(taps, k0, k1, args.kw_ms, args.kde_dt)
            yn = y / (np.max(y) + 1e-12) if y.size else y
            ax.plot(tg, yn, lw=1.5)
            for b in beats:
                ax.axvline(b, color="0.35", lw=0.7, alpha=0.45)
            ax.set_ylabel("norm KDE")
            ax.set_title(title + f" | n={taps.size}")
            ax.set_xlim(m0, m1)
            ax.grid(True, alpha=0.25)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 1].set_xlabel("Time (s)")
        fig.savefig(fig_dir / f"{stem}_kde_over_gt.png", bbox_inches="tight")
        plt.close(fig)

        # Sonify
        if args.skip_sonify:
            continue
        wav_p = find_wav(args.wav_dir, stem)
        if wav_p is None:
            continue
        try:
            import soundfile as sf
        except ImportError:
            break
        audio, sr = sf.read(str(wav_p))
        n = len(audio) if audio.ndim == 1 else audio.shape[0]
        mono = audio.mean(axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)

        beats_full = gt_map.get(stem, np.array([]))
        if args.dataset == "ballroom" and args.beats_dir:
            p = Path(args.beats_dir) / f"{stem}.beats"
            if p.exists():
                beats_full = load_ballroom_beats(p)
        elif args.dataset == "mirex" and args.mirex_dir:
            m = re.match(r"train(\d+)$", stem, re.I)
            if m:
                p = Path(args.mirex_dir) / f"train{m.group(1)}.txt"
                if p.exists():
                    lines = [ln.strip() for ln in open(p) if ln.strip()]
                    idx = min(args.mirex_gt_annotator, len(lines) - 1)
                    if lines:
                        beats_full = np.array(
                            [float(x) for x in lines[idx].split("\t") if x.strip()],
                            dtype=float,
                        )

        gt_click = make_clicks(sr, n, beats_full, hz=800.0)
        sf.write(str(son_dir / f"{stem}_gt_beats.wav"), mix_audio_clicks(mono, gt_click), sr)

        tg, y = gaussian_kde(np.array(d["raw"], dtype=float), k0, k1, args.kw_ms, args.kde_dt)
        pk_raw = kde_peaks(
            tg, y, args.kde_dt, args.peak_prominence, args.peak_distance_s, args.peak_width_s
        )
        tg2, y2 = gaussian_kde(np.array(d["clean"], dtype=float), k0, k1, args.kw_ms, args.kde_dt)
        pk_clean = kde_peaks(
            tg2, y2, args.kde_dt, args.peak_prominence, args.peak_distance_s, args.peak_width_s
        )

        sf.write(
            str(son_dir / f"{stem}_repp_kde_raw.wav"),
            mix_audio_clicks(mono, make_clicks(sr, n, pk_raw, hz=1100.0)),
            sr,
        )
        sf.write(
            str(son_dir / f"{stem}_repp_kde_clean.wav"),
            mix_audio_clicks(mono, make_clicks(sr, n, pk_clean, hz=1300.0)),
            sr,
        )
        L = mix_audio_clicks(mono, gt_click)
        R = mix_audio_clicks(mono, make_clicks(sr, n, pk_clean, hz=1300.0))
        if L.shape[0] == R.shape[0]:
            sf.write(str(son_dir / f"{stem}_stereo_gtL_reppCleanR.wav"), np.column_stack([L, R]), sr)

    pd.DataFrame(summ).to_csv(out / "hit_miss_summary_per_stimulus.csv", index=False)
    pd.DataFrame(kde_rows).to_csv(out / "kde_raster_hit_summary.csv", index=False)
    if peak_clean_rows:
        pd.DataFrame(peak_clean_rows).to_csv(out / "cleaned_kde_peaks.csv", index=False)

    if all_stage_rows:
        stg = pd.DataFrame(all_stage_rows)
        stg.to_csv(out / "repp_vs_gt_alignment_stages.csv", index=False)

        def _sort_stem(s: str) -> Tuple[int, str]:
            m = re.search(r"(\d+)$", str(s))
            return (int(m.group(1)), str(s)) if m else (999, str(s))

        if args.skip_figures:
            fig_h = None
        else:
            fig_h, axes_h = plt.subplots(1, 2, figsize=(14, 6), dpi=args.dpi)
        s12 = stg[stg["stage"].isin(["1_pooled_raw_taps_vs_gt", "2_pooled_cleaned_taps_vs_gt"])]
        if len(s12) and fig_h is not None:
            p12 = s12.pivot_table(index="stimulus_id", columns="stage", values="hit_rate", aggfunc="first")
            p12 = p12.reindex(sorted(p12.index, key=_sort_stem))
            im0 = axes_h[0].imshow(p12.T.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
            axes_h[0].set_yticks(range(len(p12.columns)))
            axes_h[0].set_yticklabels(["S1 raw taps", "S2 cleaned"], fontsize=9)
            axes_h[0].set_xticks(range(len(p12.index)))
            axes_h[0].set_xticklabels(list(p12.index), rotation=90, fontsize=7)
            axes_h[0].set_title("Hit rate (tap within ±%.0f ms of GT beat)" % args.hit_tol_ms)
            plt.colorbar(im0, ax=axes_h[0], fraction=0.046)
        s36 = stg[stg["stage"].str.match(r"^[3456]_")]
        if len(s36) and fig_h is not None:
            p36 = s36.pivot_table(
                index="stimulus_id", columns="stage", values="pct_peaks_within_tol_of_beat", aggfunc="first"
            )
            p36 = p36.reindex(sorted(p36.index, key=_sort_stem))
            im1 = axes_h[1].imshow(p36.T.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
            axes_h[1].set_yticks(range(len(p36.columns)))
            axes_h[1].set_yticklabels([c[:20] for c in p36.columns], fontsize=7)
            axes_h[1].set_xticks(range(len(p36.index)))
            axes_h[1].set_xticklabels(list(p36.index), rotation=90, fontsize=7)
            axes_h[1].set_title("KDE peaks within ±%.0f ms of GT (%%)" % args.hit_tol_ms)
            plt.colorbar(im1, ax=axes_h[1], fraction=0.046)
        if fig_h is not None:
            fig_h.tight_layout()
            fig_h.savefig(fig_dir / "summary_alignment_heatmap.png", bbox_inches="tight")
            plt.close(fig_h)

        order_st = [
            "1_pooled_raw_taps_vs_gt",
            "2_pooled_cleaned_taps_vs_gt",
            "3_kde_peaks_from_raw_taps",
            "4_kde_peaks_from_clean_taps",
            "5_kde_peaks_from_hits_raw_pool",
            "6_kde_peaks_from_hits_clean_pool",
        ]
        lines_md = [
            "# REPP vs GT — mean across stimuli\n\n",
            "| Stage | Mean across tracks |\n",
            "|-------|--------------------|\n",
        ]
        for st in order_st:
            sub = stg[stg["stage"] == st]
            if sub.empty:
                continue
            if st.startswith("1") or st.startswith("2"):
                hr = 100 * float(sub["hit_rate"].mean())
                ma = float(sub["mean_abs_async_ms_on_hits"].mean())
                ms = float(sub["mean_async_ms_signed_on_hits"].mean())
                sd = float(sub["std_async_ms_on_hits"].mean())
                lines_md.append(
                    f"| {st} | **{hr:.1f}%** hit rate; mean|async| on hits ≈ **{ma:.2f} ms**; "
                    f"mean signed ≈ **{ms:.2f} ms**; SD(signed) on hits ≈ **{sd:.2f} ms** (means across tracks) |\n"
                )
            else:
                pc = float(sub["pct_peaks_within_tol_of_beat"].mean())
                md = float(sub["median_abs_peak_to_beat_ms"].mean())
                lines_md.append(
                    f"| {st} | **{pc:.1f}%** KDE peaks within tol; mean median|peak−beat| ≈ **{md:.2f} ms** |\n"
                )
        (out / "ALIGNMENT_SUMMARY.md").write_text("".join(lines_md), encoding="utf-8")

        # --- Cleaning effectiveness narrative + outlier sonify ---
        ct = pd.read_csv(out / "cleaned_trials.csv")
        removed = (ct["n_taps_raw"].astype(int) - ct["n_taps_clean"].astype(int)).clip(lower=0)
        eff_lines = [
            "# Cleaning effectiveness (this run)\n\n",
            "## Per-trial MAD cleaning\n\n",
            f"- Trials: **{len(ct)}**\n",
            f"- Mean taps removed per trial (raw − clean): **{removed.mean():.2f}** "
            f"(median **{removed.median():.0f}**)\n",
            f"- Share of trials with ≥1 tap removed: **{100 * (removed > 0).mean():.1f}%**\n\n",
        ]
        s1 = stg[stg["stage"] == "1_pooled_raw_taps_vs_gt"]
        s2 = stg[stg["stage"] == "2_pooled_cleaned_taps_vs_gt"]
        s3 = stg[stg["stage"] == "3_kde_peaks_from_raw_taps"]
        s4 = stg[stg["stage"] == "4_kde_peaks_from_clean_taps"]
        if len(s1) and len(s2):
            dhr = float(s2["hit_rate"].mean() - s1["hit_rate"].mean())
            eff_lines.append("## Pooled taps vs GT (mean across stimuli)\n\n")
            eff_lines.append(
                f"- Mean hit rate **raw**: **{100 * s1['hit_rate'].mean():.1f}%** → **clean**: **{100 * s2['hit_rate'].mean():.1f}%** "
                f"(Δ **{100 * dhr:+.1f}** pp)\n"
            )
            eff_lines.append(
                f"- Mean |asynchrony| on hits (raw vs clean): "
                f"**{s1['mean_abs_async_ms_on_hits'].mean():.1f}** ms → **{s2['mean_abs_async_ms_on_hits'].mean():.1f}** ms\n\n"
            )
            eff_lines.append(
                "*Interpretation*: Positive Δ hit rate means pooled taps land closer to annotated beats after removing "
                "IOI-outlier taps. If Δ is small, either taps were already clean or MAD removed mostly on-beat noise. "
                "Large negative Δ is rare (check per-stimulus figures).\n\n"
            )
        if len(s3) and len(s4):
            dpk = float(s4["pct_peaks_within_tol_of_beat"].mean() - s3["pct_peaks_within_tol_of_beat"].mean())
            eff_lines.append("## KDE peaks vs GT\n\n")
            eff_lines.append(
                f"- Mean % of KDE peaks within ±{args.hit_tol_ms:.0f} ms of a beat — **raw**: **{s3['pct_peaks_within_tol_of_beat'].mean():.1f}%** "
                f"→ **clean**: **{s4['pct_peaks_within_tol_of_beat'].mean():.1f}%** (Δ **{dpk:+.1f}** pp)\n\n"
            )
            eff_lines.append(
                "Peaks summarize where the group *collectively* taps; MAD cleaning often **sharpens** the KDE so peaks "
                "align better with GT when sporadic off-beat taps inflated side lobes.\n"
            )
        (out / "CLEANING_EFFECTIVENESS.md").write_text("".join(eff_lines), encoding="utf-8")

        piv = stg[stg["stage"].isin(["3_kde_peaks_from_raw_taps", "4_kde_peaks_from_clean_taps"])].pivot_table(
            index="stimulus_id", columns="stage", values="pct_peaks_within_tol_of_beat", aggfunc="first"
        )
        if len(piv.columns) >= 2:
            c3 = "3_kde_peaks_from_raw_taps"
            c4 = "4_kde_peaks_from_clean_taps"
            if c3 in piv.columns and c4 in piv.columns:
                reg = piv[c3] - piv[c4]
                bad = (piv[c4] < args.sonify_outlier_max_pct) | (reg > args.sonify_outlier_regress_pts)
                bad_stems = list(piv.index[bad.fillna(False)])
                son_bad = out / "sonify_outliers"
                son_bad.mkdir(exist_ok=True)
                meta = [
                    "# Outlier sonification\n\n",
                    f"Stimuli flagged: KDE clean peaks within ±{args.hit_tol_ms:.0f} ms of GT **below {args.sonify_outlier_max_pct}%** "
                    f"**or** raw-minus-clean peak-alignment (percentage points) **greater than {args.sonify_outlier_regress_pts}**.\n\n",
                ]
                if not bad_stems:
                    meta.append("*No stimuli met these criteria for this run.*\n\n")
                try:
                    import soundfile as sf
                except ImportError:
                    sf = None
                for stem in bad_stems:
                    meta.append(
                        f"- **{stem}** — KDE peaks within tol: raw **{piv.loc[stem, c3]:.1f}%**, clean **{piv.loc[stem, c4]:.1f}%**\n"
                    )
                    if args.skip_sonify or sf is None:
                        continue
                    wav_p = find_wav(args.wav_dir, stem)
                    if wav_p is None:
                        meta.append(f"  - *no local WAV under wav_dir*\n")
                        continue
                    d = by_stem.get(stem)
                    if not d:
                        continue
                    ok0, ok1 = stem_kde[stem]
                    audio, sr = sf.read(str(wav_p))
                    n = len(audio) if audio.ndim == 1 else audio.shape[0]
                    mono = audio.mean(axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)
                    beats_full = gt_map.get(stem, np.array([]))
                    if args.dataset == "ballroom" and args.beats_dir:
                        p = Path(args.beats_dir) / f"{stem}.beats"
                        if p.exists():
                            beats_full = load_ballroom_beats(p)
                    elif args.dataset == "mirex" and args.mirex_dir:
                        m = re.match(r"train(\d+)$", stem, re.I)
                        if m:
                            p = Path(args.mirex_dir) / f"train{m.group(1)}.txt"
                            if p.exists():
                                lines = [ln.strip() for ln in open(p) if ln.strip()]
                                idx = min(args.mirex_gt_annotator, len(lines) - 1)
                                if lines:
                                    beats_full = np.array(
                                        [float(x) for x in lines[idx].split("\t") if x.strip()], dtype=float
                                    )
                    tg, y = gaussian_kde(np.array(d["raw"], dtype=float), ok0, ok1, args.kw_ms, args.kde_dt)
                    pk_r = kde_peaks(
                        tg, y, args.kde_dt, args.peak_prominence, args.peak_distance_s, args.peak_width_s
                    )
                    tg2, y2 = gaussian_kde(np.array(d["clean"], dtype=float), ok0, ok1, args.kw_ms, args.kde_dt)
                    pk_c = kde_peaks(
                        tg2, y2, args.kde_dt, args.peak_prominence, args.peak_distance_s, args.peak_width_s
                    )
                    gt_c = make_clicks(sr, n, beats_full, hz=750.0)
                    raw_c = make_clicks(sr, n, pk_r, hz=950.0)
                    cln_c = make_clicks(sr, n, pk_c, hz=1550.0)
                    L = mix_audio_clicks(mono, gt_c)
                    R = 0.5 * mix_audio_clicks(mono, raw_c) + 0.5 * mix_audio_clicks(mono, cln_c)
                    R = R / max(1e-6, float(np.max(np.abs(R))))
                    if L.shape[0] == R.shape[0]:
                        sf.write(str(son_bad / f"{stem}_OUTLIER_stereo_gtL_rawVsCleanR.wav"), np.column_stack([L, R]), sr)
                (son_bad / "README_OUTLIERS.md").write_text("".join(meta), encoding="utf-8")

    readme = out / "README_PIPELINE.txt"
    readme.write_text(
        f"dataset={args.dataset}\n"
        f"cleaning={'mad_only' if args.no_tempo_gate else 'tempo_gate_plus_mad'}\n"
        f"hit_tol={args.hit_tol_ms}ms | tempo_tol={args.tempo_tol} | stim_offset={args.stim_offset_s}s\n"
        f"kde: kw_ms={args.kw_ms} | kde_dt={args.kde_dt}\n"
        f"peaks: prominence={args.peak_prominence} | distance_s={args.peak_distance_s} | width_s={args.peak_width_s}\n"
        f"MIREX GT = annotator row {args.mirex_gt_annotator} in trainN.txt (ballroom: .beats)\n"
        f"See CLEANING_EFFECTIVENESS.md, ALIGNMENT_SUMMARY.md, repp_vs_gt_alignment_stages.csv\n"
        f"Figures: *_alignment_clear.png, *_kde_peaks_raw_vs_clean.png, summary_alignment_heatmap.png\n"
        f"Sonify outliers: sonify_outliers/ (if GT available)\n",
        encoding="utf-8",
    )
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
