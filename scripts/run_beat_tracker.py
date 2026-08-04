#!/usr/bin/env python3
"""
Run computational beat tracking (librosa, madmom) on stimulus WAVs and compare to human KDE consensus.

Outputs:
- algorithm_beats.csv: wav, beat_time_s, window_start, window_end, algorithm
- comparison_vs_human.csv: per-track F1, precision, recall vs consensus (if --peaks_csv)
- comparison_report.md: summary + correlations with ISTC/IOI (if --istc_csv, --ioi_csv)
- f1_vs_istc.png, f1_vs_ioi.png: scatter plots (if coherence CSVs provided)

Usage:
  # Global: both algorithms vs consensus + ISTC/IOI
  python scripts/run_beat_tracker.py --wav_dir audio/reconstructed-demucs --out_dir outputs/beat_tracker_global \\
    --peaks_csv outputs/merged_global_tap_all/plots_0_50/all_peaks.csv \\
    --istc_csv outputs/merged_global_tap_all/coherence_0_50/istc_per_track.csv \\
    --ioi_csv outputs/merged_global_tap_all/coherence_0_50/ioi_ratio_summary_per_track.csv \\
    --algorithm both --t0 0 --t1 50

Requires: pip install librosa soundfile (madmom optional)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\nRun: pip install librosa soundfile"
    ) from None


def run_librosa_beats(wav_path: Path, t0: float, t1: float, sr: int = 22050) -> tuple[float, np.ndarray]:
    """Run librosa beat_track, return (tempo_bpm, beat_times_s)."""
    y, file_sr = librosa.load(str(wav_path), sr=sr, mono=True)
    tempo, beats_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    beat_times = np.asarray(beats_frames)
    beat_times = beat_times[(beat_times >= t0) & (beat_times <= t1)]
    return float(tempo), beat_times


def run_madmom_beats(wav_path: Path, t0: float, t1: float) -> np.ndarray:
    """Run madmom DBN beat tracker if available. Returns beat times in seconds."""
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    except ImportError:
        return np.array([])

    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(str(wav_path))
    beats = proc(act)
    beat_times = np.array(beats)
    beat_times = beat_times[(beat_times >= t0) & (beat_times <= t1)]
    return beat_times


def compute_alignment_metrics(
    human_peaks: np.ndarray,
    algo_beats: np.ndarray,
    tolerance_s: float = 0.05,
) -> dict:
    """Compute F-measure–style alignment: % of human peaks matched by algo, % of algo beats matched."""
    if len(human_peaks) == 0 or len(algo_beats) == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "mean_offset_s": np.nan}

    human_matched = 0
    for hp in human_peaks:
        if np.any(np.abs(algo_beats - hp) <= tolerance_s):
            human_matched += 1
    recall = human_matched / len(human_peaks)

    algo_matched = 0
    for ab in algo_beats:
        if np.any(np.abs(human_peaks - ab) <= tolerance_s):
            algo_matched += 1
    precision = algo_matched / len(algo_beats) if len(algo_beats) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Mean temporal offset (algo - human) for beats within tolerance
    offsets = []
    for ab in algo_beats:
        d = np.abs(human_peaks - ab)
        if np.min(d) <= tolerance_s:
            idx = np.argmin(d)
            offsets.append(ab - human_peaks[idx])
    mean_offset = np.mean(offsets) if offsets else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_offset_s": mean_offset,
        "n_human": len(human_peaks),
        "n_algo": len(algo_beats),
    }


def main():
    ap = argparse.ArgumentParser(description="Run beat tracker and optionally compare to human consensus")
    ap.add_argument("--wav_dir", default="", help="Folder containing stimulus WAVs (non-recursive glob)")
    ap.add_argument("--wav_list", default="", help="Text file with one absolute WAV path per line; overrides --wav_dir")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of --wav_dir")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--peaks_csv", default="", help="Human consensus all_peaks.csv for comparison")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--stim_offset_s", type=float, default=2.0,
                    help="Subtract from human peaks to convert tap time (0=marker) to WAV time (0=music)")
    ap.add_argument("--algorithm", choices=["librosa", "madmom", "both"], default="librosa")
    ap.add_argument("--tolerance_s", type=float, default=0.05, help="Beat match tolerance (seconds)")
    ap.add_argument("--istc_csv", default="", help="ISTC per-track CSV (group_key, istc_mean_max_per_sec) for correlation")
    ap.add_argument("--ioi_csv", default="", help="IOI-ratio summary CSV (group_key, mean_ratio, pct_near_1) for correlation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wav_list:
        list_path = Path(args.wav_list)
        if not list_path.exists():
            raise SystemExit(f"--wav_list file not found: {list_path}")
        wavs = [Path(p.strip()) for p in list_path.read_text().splitlines() if p.strip()]
        wavs = [w for w in wavs if w.exists()]
        if not wavs:
            raise SystemExit(f"--wav_list contained no existing files")
    else:
        if not args.wav_dir:
            raise SystemExit("Must supply --wav_dir or --wav_list")
        wav_dir = Path(args.wav_dir).resolve()
        if args.recursive:
            wavs = sorted(wav_dir.rglob("*.wav"))
        else:
            wavs = sorted(wav_dir.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"No WAVs found in {wav_dir}")

    human_peaks_df = None
    if args.peaks_csv and Path(args.peaks_csv).exists():
        human_peaks_df = pd.read_csv(args.peaks_csv)
        if "wav" not in human_peaks_df.columns or "peak_time_s" not in human_peaks_df.columns:
            human_peaks_df = None
        elif "window_start" in human_peaks_df.columns and "window_end" in human_peaks_df.columns:
            human_peaks_df = human_peaks_df[
                (np.isclose(human_peaks_df["window_start"], args.t0))
                & (np.isclose(human_peaks_df["window_end"], args.t1))
            ]

    rows = []
    comparison_rows = []
    beats_by_wav = {}  # wav_name -> {"librosa": arr, "madmom": arr}

    for wav_path in wavs:
        wav_name = wav_path.name
        human_peaks = np.array([])
        if human_peaks_df is not None:
            hp = human_peaks_df[human_peaks_df["wav"] == wav_name]
            if not hp.empty:
                human_peaks_arr = hp["peak_time_s"].values
                human_peaks = human_peaks_arr[(human_peaks_arr >= args.t0) & (human_peaks_arr <= args.t1)]
                # Convert tap time (0=marker) to WAV time (0=music) for comparison with algo beats
                if args.stim_offset_s != 0:
                    human_peaks = human_peaks - args.stim_offset_s
                    human_peaks = human_peaks[human_peaks >= 0]

        if args.algorithm in ("librosa", "both"):
            tempo, beat_times = run_librosa_beats(wav_path, args.t0, args.t1)
            beats_by_wav.setdefault(wav_name, {})["librosa"] = beat_times
            for t in beat_times:
                rows.append({
                    "wav": wav_name,
                    "beat_time_s": float(t),
                    "window_start": args.t0,
                    "window_end": args.t1,
                    "algorithm": "librosa",
                    "tempo_bpm": tempo,
                })
            if human_peaks_df is not None and len(human_peaks) > 0:
                m = compute_alignment_metrics(human_peaks, beat_times, args.tolerance_s)
                comparison_rows.append({
                    "wav": wav_name,
                    "algorithm": "librosa",
                    **m,
                })

        if args.algorithm in ("madmom", "both"):
            beat_times = run_madmom_beats(wav_path, args.t0, args.t1)
            beats_by_wav.setdefault(wav_name, {})["madmom"] = beat_times
            for t in beat_times:
                rows.append({
                    "wav": wav_name,
                    "beat_time_s": float(t),
                    "window_start": args.t0,
                    "window_end": args.t1,
                    "algorithm": "madmom",
                })
            if human_peaks_df is not None and len(human_peaks) > 0:
                m = compute_alignment_metrics(human_peaks, beat_times, args.tolerance_s)
                comparison_rows.append({
                    "wav": wav_name,
                    "algorithm": "madmom",
                    **m,
                })

    out_csv = out_dir / "algorithm_beats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} ({len(rows)} beats)")

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_csv = out_dir / "comparison_vs_human.csv"
        comp_df.to_csv(comp_csv, index=False)
        print(f"[OK] Wrote {comp_csv}")

        # Librosa vs madmom algorithm-algorithm comparison (when both run)
        librosa_vs_madmom_rows = []
        if args.algorithm == "both" and "librosa" in beats_by_wav.get(wavs[0].name, {}) and "madmom" in beats_by_wav.get(wavs[0].name, {}):
            for wav_name, beats in beats_by_wav.items():
                lb = beats.get("librosa", np.array([]))
                mm = beats.get("madmom", np.array([]))
                if len(lb) > 0 and len(mm) > 0:
                    m = compute_alignment_metrics(mm, lb, args.tolerance_s)
                    librosa_vs_madmom_rows.append({"wav": wav_name, "f1_librosa_vs_madmom": m["f1"], **m})
            if librosa_vs_madmom_rows:
                lvm_df = pd.DataFrame(librosa_vs_madmom_rows)
                lvm_csv = out_dir / "librosa_vs_madmom.csv"
                lvm_df.to_csv(lvm_csv, index=False)
                print(f"[OK] Wrote {lvm_csv} (mean F1 lib↔mad: {lvm_df['f1'].mean():.3f})")

        # Merge with ISTC / IOI if provided
        merged = comp_df.copy()
        if args.istc_csv and Path(args.istc_csv).exists():
            istc_df = pd.read_csv(args.istc_csv)
            key = "group_key" if "group_key" in istc_df.columns else istc_df.columns[0]
            istc_renamed = istc_df[[key, "istc_mean_max_per_sec"]].rename(columns={key: "wav"})
            merged = merged.merge(istc_renamed, on="wav", how="left")
        if args.ioi_csv and Path(args.ioi_csv).exists():
            ioi_df = pd.read_csv(args.ioi_csv)
            key = "group_key" if "group_key" in ioi_df.columns else ioi_df.columns[0]
            ioi_cols = [key]
            for c in ["mean_ratio", "pct_near_1", "pct_half", "pct_double"]:
                if c in ioi_df.columns:
                    ioi_cols.append(c)
            ioi_renamed = ioi_df[ioi_cols].rename(columns={key: "wav"})
            merged = merged.merge(ioi_renamed, on="wav", how="left")

        merged_csv = out_dir / "comparison_with_coherence.csv"
        merged.to_csv(merged_csv, index=False)
        if "istc_mean_max_per_sec" in merged.columns or "mean_ratio" in merged.columns:
            print(f"[OK] Wrote {merged_csv}")

        # Report
        report = out_dir / "comparison_report.md"
        with open(report, "w") as f:
            f.write("# Beat Tracker vs Human Consensus\n\n")
            f.write(f"Window: {args.t0}-{args.t1}s, tolerance={args.tolerance_s}s\n\n")
            for algo in comp_df["algorithm"].unique():
                sub = comp_df[comp_df["algorithm"] == algo]
                f.write(f"## {algo} vs human consensus\n\n")
                f.write(f"- Mean F1: {sub['f1'].mean():.3f}\n")
                f.write(f"- Mean recall: {sub['recall'].mean():.3f}\n")
                f.write(f"- Mean precision: {sub['precision'].mean():.3f}\n")
                f.write(f"- Mean offset (algo - human): {sub['mean_offset_s'].mean():.4f}s\n\n")
            if librosa_vs_madmom_rows:
                f.write("## Librosa vs madmom (algorithm-algorithm)\n\n")
                f.write(f"- Mean F1: {lvm_df['f1'].mean():.3f}\n\n")
            if "istc_mean_max_per_sec" in merged.columns:
                f.write("## Correlations with ISTC\n\n")
                for algo in merged["algorithm"].unique():
                    sub = merged[merged["algorithm"] == algo].dropna(subset=["f1", "istc_mean_max_per_sec"])
                    if len(sub) >= 5:
                        r = sub["f1"].corr(sub["istc_mean_max_per_sec"])
                        f.write(f"- {algo} F1 vs ISTC: r = {r:.3f}\n")
                f.write("\n")
            if "mean_ratio" in merged.columns:
                f.write("## Correlations with IOI-ratio\n\n")
                for algo in merged["algorithm"].unique():
                    sub = merged[merged["algorithm"] == algo].dropna(subset=["f1", "mean_ratio"])
                    if len(sub) >= 5:
                        r = sub["f1"].corr(sub["mean_ratio"])
                        f.write(f"- {algo} F1 vs mean_ratio: r = {r:.3f}\n")
                    if "pct_near_1" in merged.columns:
                        sub2 = merged[merged["algorithm"] == algo].dropna(subset=["f1", "pct_near_1"])
                        if len(sub2) >= 5:
                            r2 = sub2["f1"].corr(sub2["pct_near_1"])
                            f.write(f"- {algo} F1 vs pct_near_1: r = {r2:.3f}\n")
                f.write("\n")
        print(f"[OK] Wrote {report}")

        # Plots (see also scripts/plot_beat_tracker_results.py for full viz)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            has_plots = False
            if "istc_mean_max_per_sec" in merged.columns:
                fig, ax = plt.subplots(figsize=(7, 5))
                for algo in merged["algorithm"].unique():
                    sub = merged[merged["algorithm"] == algo].dropna(subset=["f1", "istc_mean_max_per_sec"])
                    if len(sub) > 0:
                        ax.scatter(sub["istc_mean_max_per_sec"], sub["f1"], label=algo, alpha=0.7)
                ax.set_xlabel("ISTC (mean max taps/sec)")
                ax.set_ylabel("F1 vs human consensus")
                ax.legend()
                ax.set_title("F1 vs ISTC: algorithm–human agreement vs tap coherence")
                fig.tight_layout()
                fig.savefig(out_dir / "f1_vs_istc.png", dpi=120)
                plt.close(fig)
                has_plots = True
            if "mean_ratio" in merged.columns:
                fig, ax = plt.subplots(figsize=(7, 5))
                for algo in merged["algorithm"].unique():
                    sub = merged[merged["algorithm"] == algo].dropna(subset=["f1", "mean_ratio"])
                    if len(sub) > 0:
                        ax.scatter(sub["mean_ratio"], sub["f1"], label=algo, alpha=0.7)
                ax.set_xlabel("Mean IOI-ratio (participant / consensus)")
                ax.set_ylabel("F1 vs human consensus")
                ax.legend()
                ax.set_title("F1 vs IOI-ratio")
                fig.tight_layout()
                fig.savefig(out_dir / "f1_vs_ioi.png", dpi=120)
                plt.close(fig)
                has_plots = True
            if has_plots:
                print(f"[OK] Wrote {out_dir / 'f1_vs_istc.png'}, {out_dir / 'f1_vs_ioi.png'}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
