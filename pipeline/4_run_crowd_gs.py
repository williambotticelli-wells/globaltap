#!/usr/bin/env python3
"""
Stage 4: Run the canonical crowd_gs optimizer for every stem.

For each manifest stem this stage computes the canonical beat sequence
over the optimization window:

    crowd_gs   — canonical optimizer cascade
                 (bimodal split → mono / balanced \u03bb → grid-search /
                 isochrony-first optimizer).
                 Input: Stage-2 cleaned KDE peaks; no participant-level
                 regression filter.

The output beat sequence is written to
    outputs/<dataset>/crowd_gs/{stem}_crowd_gs.txt
along with a run-level stage4_summary.csv.

Dependencies (imported at runtime, not at module top):
    optimization.run_pipeline                 (run_cascade, KDE_DT, KW_MS)
    optimization.optimization_regularize      (optimize_beats)
"""
from __future__ import annotations
import argparse, functools, sys, time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent.parent  # companion_repo/
OPT_DIR = HERE / "optimization"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OPT_DIR))

from lib.paths import load_config  # noqa: E402

from run_pipeline import run_cascade, KDE_DT, KW_MS  # noqa: E402
import run_pipeline as _opt_mod  # noqa: E402
from optimization_regularize import optimize_beats as _orig_optimize_beats  # noqa: E402


# ── Grid-search monkeypatch ─────────────────────────────────────────────────

@functools.wraps(_orig_optimize_beats)
def _gs_optimize_beats(*args, **kwargs):
    kwargs.setdefault("method", "grid_search")
    return _orig_optimize_beats(*args, **kwargs)


def _install_gs_patch():
    _opt_mod.optimize_beats = _gs_optimize_beats


def _uninstall_gs_patch():
    _opt_mod.optimize_beats = _orig_optimize_beats


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_peaks_by_stem(peaks_csv: Path) -> dict:
    df = pd.read_csv(peaks_csv)
    out = {}
    for sid, grp in df.groupby("stimulus_id"):
        out[str(sid)] = np.sort(grp["peak_time_s"].values.astype(float))
    return out


def load_madmom_by_stem(algo_csv: Path) -> dict:
    df = pd.read_csv(algo_csv)
    df = df[df["algorithm"].astype(str) == "madmom"]
    out = {}
    for wav, grp in df.groupby("wav"):
        stem = str(wav).removesuffix(".wav")
        out[stem] = np.sort(grp["beat_time_s"].values.astype(float))
    return out


# ── Variant ─────────────────────────────────────────────────────────────────

def run_crowd_gs(stem: str, peaks: np.ndarray, mm_beats: np.ndarray,
                 istc_val: float = 15.0) -> dict:
    """Canonical optimizer cascade with grid-search."""
    _install_gs_patch()
    try:
        diag = run_cascade(stem, peaks, mm_beats, istc_val)
    finally:
        _uninstall_gs_patch()
    return diag


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config" / "canonical240.json"))
    ap.add_argument("--variant", default="crowd_gs", choices=["crowd_gs"],
                    help="Only crowd_gs is supported in the canonical pipeline.")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="Restrict to a subset of stems (for debugging)")
    ap.add_argument("--peaks_override", default=None,
                    help="Override path to cleaned_kde_peaks.csv (e.g. for ablation)")
    ap.add_argument("--out_subdir", default="crowd_gs",
                    help="Output subdirectory under outputs_root (default: crowd_gs)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root_str = str(cfg["data_root"])
    if "${" in data_root_str:
        # PROJECT_ROOT is unset: fall back to repo-local outputs so a fresh
        # clone can run Stage 4 against the included stage_inputs/.
        data_root = HERE
        out_root = HERE / "outputs" / cfg["dataset"]
        print(f"[stage4] PROJECT_ROOT unset; using repo-local out_root={out_root}")
    else:
        data_root = Path(data_root_str)
        out_root = Path(cfg["outputs_root"])
        if not out_root.is_absolute():
            out_root = data_root / out_root

    # Resolve Stage-2/3 inputs. If a full pipeline run lives at
    # ``out_root`` use it; otherwise fall back to the
    # ``data/stage_inputs/<dataset>/`` files included in the repo
    # so Stage 4 can run without raw audio or tapping CSVs.
    included = HERE / "data" / "stage_inputs" / cfg["dataset"]
    peaks_csv = (
        Path(args.peaks_override) if args.peaks_override
        else (out_root / "repp" / "cleaned_kde_peaks.csv")
    )
    algo_csv = out_root / "madmom" / "algorithm_beats.csv"
    if not peaks_csv.exists() and (included / "cleaned_kde_peaks.csv").exists():
        peaks_csv = included / "cleaned_kde_peaks.csv"
    if not algo_csv.exists() and (included / "algorithm_beats.csv").exists():
        algo_csv = included / "algorithm_beats.csv"

    for p in [peaks_csv, algo_csv]:
        if not p.exists():
            raise SystemExit(f"Required input missing: {p}")

    out_dir = out_root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage4] peaks_csv = {peaks_csv}")
    print(f"[stage4] out_dir   = {out_dir}")

    w0 = cfg["optimizer_window"]["w0"]
    w1 = cfg["optimizer_window"]["w1"]

    # Ensure the optimizer module uses the config's IOI / window / KDE constants
    _opt_mod.W0 = w0
    _opt_mod.W1 = w1
    _opt_mod.IOI_MIN = cfg["optimizer"]["ioi_min_s"]
    _opt_mod.IOI_MAX = cfg["optimizer"]["ioi_max_s"]
    _opt_mod.KW_MS = cfg["optimizer"]["kde_bandwidth_ms"]
    _opt_mod.KDE_DT = cfg["optimizer"]["kde_dt_s"]

    print(f"[stage4] Loading inputs \u2026")
    peaks_by = load_peaks_by_stem(peaks_csv)
    mm_by = load_madmom_by_stem(algo_csv)

    stems = sorted(peaks_by.keys())
    if args.stems:
        stems = [s for s in stems if s in args.stems]
    print(f"[stage4] {len(stems)} stems to process "
          f"(peaks={len(peaks_by)} mm={len(mm_by)})")

    if args.dry_run:
        print("[stage4] DRY RUN \u2014 exiting.")
        return

    summary_rows = []
    t_start = time.time()
    for i, stem in enumerate(stems):
        _t0 = time.time()
        peaks = peaks_by.get(stem, np.array([]))
        mm_beats = mm_by.get(stem, np.array([]))
        row = {
            "stem": stem,
            "n_peaks_in": int(peaks.size),
            "n_madmom_in": int(mm_beats.size),
        }

        d = run_crowd_gs(stem, peaks, mm_beats)
        cb = d.get("crowd_beats", np.array([]))
        np.savetxt(out_dir / f"{stem}_crowd_gs.txt", cb)
        row.update({
            "crowd_gs_n":    int(cb.size),
            "crowd_gs_P_ms": round(
                float(np.median(np.diff(cb)) * 1000) if cb.size > 1 else 0, 1),
            "crowd_gs_cv":   round(d.get("crowd_cv", 0), 5),
            "crowd_gs_path": " \u2192 ".join(d.get("path", [])),
        })

        summary_rows.append(row)
        print(f"  [{i+1:3d}/{len(stems)}] {stem:40s} "
              f"cg={row.get('crowd_gs_n','?'):>3} "
              f"({time.time()-_t0:.1f}s)", flush=True)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "stage4_summary.csv", index=False)
    elapsed = time.time() - t_start
    print(f"\n[stage4] Done in {elapsed:.1f}s "
          f"({elapsed/max(len(stems),1):.2f}s/stem)")
    print(f"[stage4] Summary: {out_dir / 'stage4_summary.csv'}")


if __name__ == "__main__":
    main()
