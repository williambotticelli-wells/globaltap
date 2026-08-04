#!/usr/bin/env python3
"""
Stage 3: Run madmom DBN beat tracker on all manifest WAVs (0-30 s window).

Thin wrapper around scripts/run_beat_tracker.py. We call it with
--algorithm madmom and --t0 0 --t1 30, pointing at the audio_dir.

Output:
    outputs/<dataset>/madmom/algorithm_beats.csv
        columns: wav, beat_time_s, window_start, window_end, algorithm

This output feeds Stage 5 (click rendering) and the period/F-measure
comparison in Stage 6.

Usage:
    python pipeline/3_run_madmom.py --config config/canonical240.json
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(PIPELINE))

from lib.paths import load_config  # noqa: E402

RUN_BEAT_TRACKER = PIPELINE / "scripts" / "run_beat_tracker.py"
DEFAULT_VENV_PY = PIPELINE / ".venv" / "bin" / "python"


def build_wav_list(cfg: dict, out_dir: Path) -> Path:
    """Emit a text file containing absolute WAV paths from the stimuli manifest."""
    import os
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    manifest_path = data_root / cfg["stimuli_manifest"]
    manifest = json.loads(manifest_path.read_text())
    lines = []
    missing = []
    for entry in manifest:
        p = Path(entry["audio_path"])
        if not p.exists():
            missing.append(str(p))
        lines.append(str(p))
    if missing:
        print(f"[stage3] WARNING: {len(missing)} manifest WAVs missing (first: {missing[0]})")
    list_path = out_dir / "wav_list.txt"
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PIPELINE / "config" / "canonical240.json"))
    ap.add_argument("--python", default=str(DEFAULT_VENV_PY),
                    help="Python interpreter with madmom installed (default: ./.venv)")
    ap.add_argument("--algorithm", default="madmom", choices=["madmom", "librosa", "both"],
                    help="Beat tracker to run (default: madmom; 'both' adds librosa for comparison)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(cfg["data_root"]) / cfg["outputs_root"] / "madmom"
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = build_wav_list(cfg, out_dir)
    n_in_list = sum(1 for _ in list_path.read_text().splitlines() if _.strip())
    print(f"[stage3] Wrote WAV list ({n_in_list} paths) -> {list_path}")

    t0 = cfg["optimizer_window"]["w0"]
    t1 = cfg["optimizer_window"]["w1"]

    py = args.python
    if not Path(py).exists():
        print(f"[stage3] WARNING: {py} not found, falling back to {sys.executable}")
        py = sys.executable

    cmd = [
        py, str(RUN_BEAT_TRACKER),
        "--wav_list", str(list_path),
        "--out_dir", str(out_dir),
        "--t0", str(t0),
        "--t1", str(t1),
        "--algorithm", args.algorithm,
    ]
    print(f"[stage3] $ {' '.join(cmd)}")
    if args.dry_run:
        return

    t_start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[stage3] FAILED (exit {e.returncode})")

    elapsed = time.time() - t_start

    out_csv = out_dir / "algorithm_beats.csv"
    if not out_csv.exists():
        raise SystemExit(f"[stage3] expected output not produced: {out_csv}")

    import pandas as pd
    df = pd.read_csv(out_csv)
    n_wav = df["wav"].nunique()
    n_beats = len(df)
    per_stem = df.groupby(["wav", "algorithm"]).size().describe()
    print(f"\n[stage3] Done in {elapsed:.1f}s")
    print(f"[stage3] {n_beats} beat events across {n_wav} WAVs")
    print(f"[stage3] Beats/stem — median={per_stem['50%']:.0f}, min={per_stem['min']:.0f}, max={per_stem['max']:.0f}")
    print(f"[stage3] Output: {out_csv}")


if __name__ == "__main__":
    main()
