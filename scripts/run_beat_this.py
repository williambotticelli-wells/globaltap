#!/usr/bin/env python3
"""Run the Beat This! (Foscarin et al., ISMIR 2024) beat tracker on a list of WAVs.

Produces ``algorithm_beats.csv`` with columns
(``wav``, ``beat_time_s``, ``window_start``, ``window_end``, ``algorithm``)
so it drops straight into Stage 5 alongside the Madmom output.

Usage:
    python scripts/run_beat_this.py \
        --wav_list path/to/wav_list.txt \
        --out_dir outputs/canonical240/beat_this \
        --t0 0 --t1 30
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from beat_this.inference import File2Beats
except ImportError as e:
    raise SystemExit(f"Missing beat_this: pip install beat_this ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=30.0)
    ap.add_argument("--checkpoint", default="final0",
                    help="Beat-This checkpoint; 'final0' is the main paper checkpoint")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wavs = [Path(p.strip()) for p in Path(args.wav_list).read_text().splitlines() if p.strip()]
    wavs = [w for w in wavs if w.exists()]
    if not wavs:
        raise SystemExit("No existing WAVs in wav_list")

    print(f"[beat_this] {len(wavs)} WAVs | checkpoint={args.checkpoint} | device={args.device}")
    print(f"[beat_this] Loading model ...")
    t_load = time.time()
    f2b = File2Beats(checkpoint_path=args.checkpoint, device=args.device, dbn=False)
    print(f"[beat_this] Model loaded in {time.time()-t_load:.1f}s")

    rows = []
    per_stem_rows = []
    t_start = time.time()
    for i, wav_path in enumerate(wavs, 1):
        try:
            beats, downbeats = f2b(str(wav_path))
        except Exception as e:
            print(f"[beat_this] ERROR on {wav_path.name}: {e}")
            continue
        beats = np.asarray(beats, dtype=float)
        beats = beats[(beats >= args.t0) & (beats <= args.t1)]
        for t in beats:
            rows.append({
                "wav": wav_path.name,
                "beat_time_s": float(t),
                "window_start": args.t0,
                "window_end": args.t1,
                "algorithm": "beat_this",
            })
        per_stem_rows.append({"wav": wav_path.name, "n_beats": len(beats),
                              "n_downbeats": int(len(np.asarray(downbeats)))})
        if i % 10 == 0 or i == len(wavs):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(wavs) - i) / rate if rate > 0 else 0
            print(f"[beat_this] {i}/{len(wavs)}  ({rate:.2f} wav/s, eta {eta:.0f}s)")

    df = pd.DataFrame(rows)
    out_csv = out_dir / "algorithm_beats.csv"
    df.to_csv(out_csv, index=False)
    pd.DataFrame(per_stem_rows).to_csv(out_dir / "per_stem_beat_counts.csv", index=False)
    print(f"[beat_this] Wrote {len(df)} beat events across {df['wav'].nunique()} wavs -> {out_csv}")
    print(f"[beat_this] Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
