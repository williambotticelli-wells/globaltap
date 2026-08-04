#!/usr/bin/env python3
"""Stage 3b: Run Beat This! (Foscarin et al., ISMIR 2024) on all manifest WAVs.

Parallel to ``pipeline/3_run_madmom.py`` but uses the Beat This! transformer
model. Output: ``outputs/<dataset>/beat_this/algorithm_beats.csv`` with
columns (``wav``, ``beat_time_s``, ``window_start``, ``window_end``,
``algorithm``).

Together with Stage 3 (Madmom), this stage produces the two F-measure
baselines reported against the pooled crowd consensus in paper §3.4.

Usage:
    python pipeline/3_run_beat_this.py --config config/canonical240.json
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(PIPELINE))

from lib.paths import load_config  # noqa: E402

RUN_BEAT_THIS = PIPELINE / "scripts" / "run_beat_this.py"
DEFAULT_VENV_PY = PIPELINE / ".venv" / "bin" / "python"


def build_wav_list(cfg: dict, out_dir: Path) -> Path:
    """Emit a text file containing absolute WAV paths from the stimuli manifest."""
    import os
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    manifest_path = data_root / cfg["stimuli_manifest"]
    manifest = json.loads(manifest_path.read_text())
    lines, missing = [], []
    for entry in manifest:
        p = Path(entry.get("audio_path") or entry.get("wav_path") or "")
        if not p.is_absolute():
            p = data_root / p
        if not p.exists():
            missing.append(str(p))
            continue
        lines.append(str(p))
    if missing:
        print(f"[stage3bt] WARNING: {len(missing)} manifest WAVs missing (first: {missing[0]})")
    list_path = out_dir / "wav_list.txt"
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PIPELINE / "config" / "canonical240.json"))
    ap.add_argument("--python", default=str(DEFAULT_VENV_PY),
                    help="Python interpreter with beat_this installed (default: ./.venv)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint", default="final0",
                    help="Beat This! checkpoint (default: 'final0', the main paper model)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(cfg["data_root"]) / cfg["outputs_root"] / "beat_this"
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = build_wav_list(cfg, out_dir)
    n_in_list = sum(1 for _ in list_path.read_text().splitlines() if _.strip())
    print(f"[stage3bt] Wrote WAV list ({n_in_list} paths) -> {list_path}")

    t0 = cfg["optimizer_window"]["w0"]
    t1 = cfg["optimizer_window"]["w1"]

    py = args.python
    if not Path(py).exists():
        print(f"[stage3bt] WARNING: {py} not found, falling back to {sys.executable}")
        py = sys.executable

    cmd = [
        py, str(RUN_BEAT_THIS),
        "--wav_list", str(list_path),
        "--out_dir", str(out_dir),
        "--t0", str(t0), "--t1", str(t1),
        "--checkpoint", args.checkpoint,
        "--device", args.device,
    ]
    print(f"[stage3bt] $ {' '.join(cmd)}")
    if args.dry_run:
        return

    t_start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[stage3bt] FAILED (exit {e.returncode})")
    elapsed = time.time() - t_start

    out_csv = out_dir / "algorithm_beats.csv"
    if not out_csv.exists():
        raise SystemExit(f"[stage3bt] expected output not produced: {out_csv}")

    import pandas as pd
    df = pd.read_csv(out_csv)
    n_wav = df["wav"].nunique()
    n_beats = len(df)
    print(f"\n[stage3bt] Done in {elapsed:.1f}s")
    print(f"[stage3bt] {n_beats} beat events across {n_wav} WAVs -> {out_csv}")


if __name__ == "__main__":
    main()
