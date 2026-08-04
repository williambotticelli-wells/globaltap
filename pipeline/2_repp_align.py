"""Stage 2 — REPP alignment + KDE peak extraction.

Thin wrapper around `scripts/repp_gt_alignment_pipeline.py`. This stage is
pure orchestration — no new DSP. All DSP params (KDE bandwidth, dt, peak
prominence, peak distance, cleaning mode) are read from the config's
``alignment`` block (new) or fall back to the ``optimizer`` block (legacy
configs).

Produces, under `outputs/<dataset>/repp/`:
    cleaned_kde_peaks.csv            ← primary input for Stage 4
    cleaned_trials.csv
    hit_miss_per_trial.csv
    hit_miss_summary_per_stimulus.csv
    repp_vs_gt_alignment_stages.csv  ← human-readable summary tables
    ALIGNMENT_SUMMARY.md
    CLEANING_EFFECTIVENESS.md
    figures/*.png                    ← per-stem overlay figures
    sonify/*.wav                     ← full-track sonification (GT + KDE)

Usage:
    python pipeline/2_repp_align.py --config config/canonical240.json
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(HERE))

from lib.paths import load_config, resolve_config_path  # noqa: E402

REPP_SCRIPT = HERE / "scripts" / "repp_gt_alignment_pipeline.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/canonical240.json")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current).",
    )
    ap.add_argument(
        "--extra", nargs=argparse.REMAINDER,
        help="Extra args appended verbatim to the REPP invocation.",
    )
    ap.add_argument(
        "--skip_sonify", action="store_true",
        help="Skip the full-track sonification step (speeds up reruns).",
    )
    ap.add_argument(
        "--skip_figures", action="store_true",
        help="Skip per-stem overlay PNGs (speeds up parameter tests).",
    )
    ap.add_argument(
        "--out_subdir", default="repp",
        help="Subdir under outputs_root to write to (default: 'repp'). Use a separate subdirectory when testing parameter combinations.",
    )
    # Optional CLI overrides for parameter tests. None means "use config value".
    ap.add_argument("--override_peak_prominence", type=float, default=None)
    ap.add_argument("--override_peak_distance_s", type=float, default=None)
    ap.add_argument("--override_peak_width_s",   type=float, default=None)
    ap.add_argument("--override_tempo_gate", choices=["on","off"], default=None,
                    help="'on' = tempo_gate+MAD cleaning; 'off' = MAD-only.")
    args = ap.parse_args()

    cfg_path = HERE / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    import os
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))

    _out_root_rel = Path(cfg["outputs_root"])
    _out_root_abs = _out_root_rel if _out_root_rel.is_absolute() else data_root / _out_root_rel
    merged_csv = _out_root_abs / "merged" / "TapTrialMusic_EXPORT_COMPAT.csv"
    if not merged_csv.exists():
        raise SystemExit(
            f"Merged CSV not found: {merged_csv}\n"
            f"Run stages/1_merge_exports.py first."
        )

    # Resolve outputs_root relative to data_root if not absolute.
    out_root = Path(cfg["outputs_root"])
    if not out_root.is_absolute():
        out_root = data_root / out_root
    repp_out = out_root / args.out_subdir
    repp_out.mkdir(parents=True, exist_ok=True)

    wav_dir = data_root / cfg["audio_dir"]

    # We keep the full WAV as the KDE window (--kde_full_wav), i.e. 0-30 s
    # per clip. GT / hit-miss / KDE-peak-alignment evaluated on the
    # optimizer window too (0-30 s). Listening excerpt (10-25 s) is
    # produced by Stage 7; Stage 2 runs on the full clip.
    w0 = cfg["optimizer_window"]["w0"]
    w1 = cfg["optimizer_window"]["w1"]
    stim_offset = cfg.get("stim_offset_s", 2.0)

    # Alignment params: prefer new 'alignment' block; fall back to 'optimizer' for legacy configs.
    align = cfg.get("alignment", {})
    opt   = cfg.get("optimizer", {})
    kw_ms      = align.get("kde_bandwidth_ms",  opt.get("kde_bandwidth_ms", 80.0))
    kde_dt     = align.get("kde_dt_s",          opt.get("kde_dt_s", 0.005))
    prom       = align.get("peak_prominence",   0.20)
    dist_s     = align.get("peak_distance_s",   0.15)
    width_s    = align.get("peak_width_s",      0.0)
    tempo_gate = align.get("tempo_gate",        False)

    # CLI overrides used when testing parameter combinations.
    if args.override_peak_prominence is not None: prom       = args.override_peak_prominence
    if args.override_peak_distance_s is not None: dist_s     = args.override_peak_distance_s
    if args.override_peak_width_s    is not None: width_s    = args.override_peak_width_s
    if args.override_tempo_gate      is not None: tempo_gate = (args.override_tempo_gate == "on")

    gt_map_path = resolve_config_path(cfg["gt_source_map"], HERE, data_root)

    cmd = [
        args.python, str(REPP_SCRIPT),
        "--dataset", "benchmark100",
        "--csv", str(merged_csv),
        "--wav_dir", str(wav_dir),
        "--out_dir", str(repp_out),
        "--benchmark100_gt_map", str(gt_map_path),
        "--benchmark100_data_root", str(data_root),
        "--t0", str(w0),
        "--t1", str(w1),
        "--stim_offset_s", str(stim_offset),
        "--kw_ms", str(kw_ms),
        "--kde_dt", str(kde_dt),
        "--peak_prominence", str(prom),
        "--peak_distance_s", str(dist_s),
        "--peak_width_s",    str(width_s),
        "--only_nonfailed",
    ]
    if not tempo_gate:
        cmd.append("--no_tempo_gate")
    if args.skip_sonify:
        cmd.append("--skip_sonify")
    if args.skip_figures:
        cmd.append("--skip_figures")
    if args.extra:
        extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd.extend(extra)

    print("Running:")
    print("  " + " \\\n    ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print(f"\nStage 2 complete. Outputs in: {repp_out}")
    print("Primary artifact: cleaned_kde_peaks.csv (input for Stage 4)")


if __name__ == "__main__":
    main()
