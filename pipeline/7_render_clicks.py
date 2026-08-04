#!/usr/bin/env python3
"""
Stage 7: Render click-track WAVs per stem per listening condition.

For every stem in the manifest and every enabled condition in
``cfg['conditions']`` this stage renders a mono WAV consisting of the
music excerpt (listening_window, default 10-25 s) mixed with clicks at
the condition's beat times, using the canonical DSP (``click_dsp`` block
in config).

The two listening conditions used in the paper's validation experiment
(§3.2) are ``gt`` and ``crowd_gs``; ``madmom`` and ``beat_this`` are
provided for diagnostic re-rendering only and are **not** part of the
listener-rating study.  The DSP helpers are shared with
``scripts/build_validation_listening_stims.py`` so click tracks use the
same DSP across all conditions.

Conditions supported:
    gt        - reference annotator beats from gt_source_map_*.json
                (via lib.gt_loader)
    crowd_gs  - canonical optimizer output, from
                outputs/<dataset>/crowd_gs/<stem>_crowd_gs.txt
    madmom    - madmom DBN baseline (F-measure baseline only)
    beat_this - Beat This! baseline (F-measure baseline only)
    peaks     - raw KDE peaks (diagnostic only)

Outputs:
    outputs/<dataset>/clicks/<condition>/<stem>__<condition>.wav
    outputs/<dataset>/clicks/stage7_summary.csv
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(HERE))

from lib.gt_loader import load_gt_for_stem  # noqa: E402
from lib.paths import load_config, resolve_config_path  # noqa: E402

_SCRIPT_PATH = HERE / "scripts" / "build_validation_listening_stims.py"
_spec = importlib.util.spec_from_file_location("_b", _SCRIPT_PATH)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)


def load_crowd_beats(crowd_dir: Path, stem: str, variant: str = "crowd_gs") -> np.ndarray:
    p = crowd_dir / f"{stem}_{variant}.txt"
    if not p.exists():
        return np.array([], dtype=float)
    vals = []
    for line in p.read_text().splitlines():
        s = line.strip().split()
        if s:
            try: vals.append(float(s[0]))
            except ValueError: continue
    return np.sort(np.array(vals, dtype=float))


def load_madmom_beats(madmom_csv: Path) -> dict:
    if not madmom_csv.exists():
        return {}
    df = pd.read_csv(madmom_csv)
    if "algorithm" in df.columns:
        df = df[df["algorithm"].astype(str) == "madmom"]
    out = {}
    for wav, g in df.groupby("wav"):
        stem = str(wav).removesuffix(".wav").strip()
        out[stem] = np.sort(g["beat_time_s"].values.astype(float))
    return out


def load_kde_peaks(peaks_csv: Path) -> dict:
    if not peaks_csv.exists():
        return {}
    df = pd.read_csv(peaks_csv)
    df["stimulus_id"] = df["stimulus_id"].astype(str).str.strip()
    out = {}
    for stem, g in df.groupby("stimulus_id"):
        out[stem] = np.sort(g["peak_time_s"].values.astype(float))
    return out


def clip_beats(b: np.ndarray, t0: float, t1: float) -> np.ndarray:
    if b.size == 0: return b
    return b[(b >= t0) & (b <= t1)]


def find_wav_for_stem(audio_dir: Path, stem: str) -> Path | None:
    # Support both flat and corpus-subdir layouts; also match trailing-space filenames.
    for pat in [f"{stem}.wav", f"{stem} .wav"]:
        direct = audio_dir / pat
        if direct.exists():
            return direct
        hits = list(audio_dir.rglob(pat))
        if hits:
            return hits[0]
    return None


def render_one(
    stem: str, condition: str, beat_times_music: np.ndarray,
    wav_path: Path, t0: float, t1: float, dsp: dict, out_path: Path,
) -> dict:
    music, sr_file = sf.read(str(wav_path))
    if music.ndim == 2:
        music = music.mean(axis=1)
    sr_target = int(dsp.get("sample_rate", 44100))
    if sr_file != sr_target:
        import soxr
        music = soxr.resample(music, sr_file, sr_target)
    sr = sr_target
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(music), int(round(t1 * sr)))
    excerpt = music[i0:i1].astype(np.float32)
    n_samples = len(excerpt)

    # Convert music-time beat locations to excerpt-local sample times
    beats_excerpt = beat_times_music - t0
    beats_excerpt = beats_excerpt[(beats_excerpt >= 0) & (beats_excerpt * sr < n_samples)]

    clicks = _b.make_click_track(
        sr=sr,
        n_samples=n_samples,
        peak_times_wav=beats_excerpt,
        tone_hz=float(dsp.get("tone_hz", 1100.0)),
        click_ms=float(dsp.get("click_ms", 12.0)),
        amp=float(dsp.get("click_amplitude", 0.9)),
    )
    y = _b.mix_50_50(excerpt, clicks, music_ratio=float(dsp.get("music_ratio", 0.35)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y.astype(np.float32), sr)
    return {
        "stem": stem,
        "condition": condition,
        "n_beats_in_window": int(beats_excerpt.size),
        "out_wav": str(out_path),
        "duration_s": round(n_samples / sr, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config" / "canonical240.json"))
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Override which conditions to render (default: all in cfg['conditions'])")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="Restrict to a subset of stems")
    ap.add_argument("--out_subdir", default="clicks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    import os
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    out_root = Path(cfg["outputs_root"])
    if not out_root.is_absolute():
        out_root = data_root / out_root

    audio_dir = data_root / cfg["audio_dir"]
    gt_map   = json.loads(resolve_config_path(cfg["gt_source_map"], HERE, data_root).read_text())
    manifest = json.loads(resolve_config_path(cfg["stimuli_manifest"], HERE, data_root).read_text())

    t0 = float(cfg["listening_window"]["t0"])
    t1 = float(cfg["listening_window"]["t1"])
    dsp = dict(cfg.get("click_dsp", {}))

    conds_cfg = cfg.get("conditions", [
        {"name": "gt", "source": "annotator"},
        {"name": "crowd_gs", "source": "pipeline"},
    ])
    cond_names = [c["name"] for c in conds_cfg]
    VALID_CONDS = {"gt", "crowd_gs", "madmom", "beat_this", "peaks"}
    if args.conditions:
        cond_names = [c for c in args.conditions if c in VALID_CONDS]

    # Resolve manifest → stems
    def _stem(e): return e.get("stem_id") or e.get("stimulus_id")
    def _corp(e): return e.get("dataset") or e.get("corpus")
    stems_all = [(_stem(e), _corp(e)) for e in manifest]
    if args.stems:
        requested = set(args.stems)
        stems_all = [(s, c) for s, c in stems_all if s in requested]

    # Pre-load condition data once
    crowd_dir = out_root / "crowd_gs"
    madmom_csv = out_root / "madmom" / "algorithm_beats.csv"
    bt_csv     = out_root / "beat_this" / "algorithm_beats.csv"
    peaks_csv  = out_root / "repp" / "cleaned_kde_peaks.csv"
    mm_beats = load_madmom_beats(madmom_csv) if "madmom" in cond_names else {}
    bt_beats = {}
    if "beat_this" in cond_names and bt_csv.exists():
        _btdf = pd.read_csv(bt_csv)
        if "algorithm" in _btdf.columns:
            _btdf = _btdf[_btdf["algorithm"].astype(str) == "beat_this"]
        for _wav, _g in _btdf.groupby("wav"):
            bt_beats[str(_wav).replace(".wav", "")] = _g["beat_time_s"].to_numpy(dtype=float)
    kde_peaks = load_kde_peaks(peaks_csv) if "peaks" in cond_names else {}

    out_dir = out_root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    start = time.time()
    print(f"[stage7] dataset={cfg['dataset']} listening_window=[{t0},{t1}]s "
          f"conditions={cond_names} stems={len(stems_all)}")

    for i, (stem, corpus) in enumerate(stems_all, 1):
        wav = find_wav_for_stem(audio_dir, stem)
        if wav is None:
            skipped.append((stem, "no_audio"))
            continue

        for cond in cond_names:
            try:
                if cond == "gt":
                    arr = load_gt_for_stem(stem, gt_map, data_root)
                    if arr is None or len(arr) == 0:
                        skipped.append((stem, f"no_gt_{cond}"))
                        continue
                    beats = arr[:, 0]
                elif cond == "crowd_gs":
                    beats = load_crowd_beats(crowd_dir, stem, "crowd_gs")
                elif cond == "madmom":
                    beats = mm_beats.get(stem, np.array([], dtype=float))
                elif cond == "beat_this":
                    beats = bt_beats.get(stem, np.array([], dtype=float))
                elif cond == "peaks":
                    beats = kde_peaks.get(stem, np.array([], dtype=float))
                else:
                    skipped.append((stem, f"unknown_cond_{cond}"))
                    continue

                beats = clip_beats(np.asarray(beats, dtype=float), t0, t1)
                out_wav = out_dir / cond / f"{stem}__{cond}.wav"
                row = render_one(stem, cond, beats, wav, t0, t1, dsp, out_wav)
                row["corpus"] = corpus
                rows.append(row)
            except Exception as e:
                skipped.append((stem, f"{cond}_error:{e}"))

        if i % 20 == 0 or i == len(stems_all):
            print(f"  [{i}/{len(stems_all)}] {stem} ({time.time()-start:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage7_summary.csv", index=False)
    if skipped:
        pd.DataFrame(skipped, columns=["stem", "reason"]).to_csv(out_dir / "stage7_skipped.csv", index=False)

    print(f"\n[stage7] Done in {time.time()-start:.1f}s.")
    print(f"[stage7] Rendered {len(df)} WAVs across {df['stem'].nunique() if len(df) else 0} stems / "
          f"{df['condition'].nunique() if len(df) else 0} conditions.")
    print(f"[stage7] Skipped: {len(skipped)}")
    print(f"[stage7] Outputs: {out_dir}")


if __name__ == "__main__":
    main()
