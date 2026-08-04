#!/usr/bin/env python3
"""Compute per-stem ISTC (inter-subject tapping coherence) from raw taps.

ISTC is the paper's *pre-optimization, input-level* reliability measure and
is computed directly from raw per-participant tap onsets (unlike
`cv_pipeline`, which is computed from the optimized Stage-4 output beats).
Definition (Granot et al., 2023, JOC 10.5334/joc.286), as stated in Methods:

    1. Pool taps from every participant who contributed >=5 taps within the
       stem's 0-30 s analysis window.
    2. Slide a 100 ms bin across the window in 50 ms steps; in each bin,
       count the number of *unique participants* who placed a tap.
    3. Within each 1 s of the window, take the maximum unique-participant
       count across that second's bins.
    4. Per-stem ISTC = mean of those per-second maxima.

Input: the per-corpus raw tapping merges (`ballroom_merged.csv`,
`mirex_merged.csv`, `benchmark_merged.csv`, `globalmood_40_merged.csv`),
each with one row per trial and a `resp_onsets_aligned` / `analysis` column
holding that trial's raw tap onsets, `_source_batch`, and `participant_id`.
These raw merges are not included in the public repository (`data/taps/` has
the de-identified benchmark taps). Point `--merged-dir` at a tree that
mirrors `raw_tapping_data/merged/` to recompute the `istc` column of
`data/per_stem_metrics.csv`.

One naming pitfall this script resolves: six of the seven GlobalMood-40
recruitment batches (`cint_export1`, `cint_export2b`, `cint_export3`,
`prolific_batch2`, `prolific_batch3`, `prolific_batch4`) label `stim_name`
with an arbitrary per-deployment index (e.g. "track19") rather than the
canonical YouTube-ID stem name; only the `retap` batch's `stim_name` is
already canonical. `audio_filename`'s basename is the canonical YouTube ID
in every batch (a few rows also have stray whitespace before the
extension), so this script canonicalizes GlobalMood-40 stems from
`audio_filename`, not `stim_name`.

Usage:
    python scripts/compute_istc_per_stem.py \\
        --merged-dir /path/to/raw_tapping_data/merged \\
        --out istc_per_stem.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORPUS_FILES = {
    "ballroom": "ballroom_merged.csv",
    "mirex": "mirex_merged.csv",
    "benchmark100": "benchmark_merged.csv",
    "globalmood40": "globalmood_40_merged.csv",
}


def _basename_no_ext(p: str) -> str:
    return Path(str(p)).stem


def canonicalize_stems(df: pd.DataFrame, corpus_group: str) -> pd.DataFrame:
    """Add a `stim_canonical` column using the naming convention that
    matches each corpus's deployment history."""
    df = df.copy()
    af = df["audio_filename"].astype(str)
    sn = df["stim_name"].astype(str)

    if corpus_group == "ballroom":
        canonical = np.where(
            af.str.contains("ballroom_wav", na=False),
            af.map(_basename_no_ext),
            sn,
        )
    elif corpus_group == "mirex":
        canonical = np.where(
            af.str.contains("/mirex/", na=False),
            af.map(_basename_no_ext),
            sn,
        )
    elif corpus_group == "benchmark100":
        canonical = sn.apply(lambda s: s.split("/", 1)[-1] if "/" in s else s)
    elif corpus_group == "globalmood40":
        # See module docstring: stim_name is only canonical for the "retap"
        # batch; audio_filename's basename is canonical in every batch.
        canonical = af.str.strip().map(_basename_no_ext).str.strip()
    else:
        raise ValueError(corpus_group)

    df["stim_canonical"] = canonical
    return df


def extract_taps_ms(row: pd.Series) -> list[float]:
    """Pull one trial's tap onsets (ms), tolerant of a couple of export
    formats seen across deployment batches."""
    val = row.get("resp_onsets_aligned")
    if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
        try:
            obj = json.loads(val.replace("'", '"'))
            if isinstance(obj, list):
                arr = [float(x) for x in obj]
                if arr and max(abs(x) for x in arr) < 100:
                    arr = [x * 1000.0 for x in arr]
                return arr
        except Exception:
            pass
    a = row.get("analysis")
    if not isinstance(a, str) or not a.strip() or a.strip().lower() == "nan":
        return []
    try:
        obj = json.loads(a)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    eo = obj.get("extracted_onsets")
    if isinstance(eo, str):
        try:
            eo = json.loads(eo)
        except Exception:
            eo = None
    if isinstance(eo, dict):
        for key in ("resp_onsets_aligned", "resp_onsets_detected"):
            v = eo.get(key)
            if isinstance(v, list) and v:
                arr = [float(x) for x in v]
                if arr and max(abs(x) for x in arr) < 100:
                    arr = [x * 1000.0 for x in arr]
                return arr
    return []


def compute_istc_per_stem(
    df_taps: pd.DataFrame,
    grid_ms: np.ndarray,
    bin_ms: float = 100.0,
    step_ms: float = 50.0,
    min_taps_per_pid: int = 5,
    min_pids_per_stem: int = 5,
) -> pd.DataFrame:
    """Per-stem ISTC; see module docstring for the definition."""
    t0_ms, t1_ms = float(grid_ms[0]), float(grid_ms[-1])
    centers = np.arange(t0_ms, t1_ms + step_ms, step_ms)
    half = bin_ms / 2.0
    centers_s = centers / 1000.0
    sec_edges = np.arange(t0_ms / 1000.0, t1_ms / 1000.0 + 1e-9, 1.0)

    rows = []
    for stem, sub in df_taps.groupby("stim_canonical"):
        pid_taps: list[tuple[object, list[float]]] = []
        for pid, part_rows in sub.groupby("participant_uid"):
            all_taps: list[float] = []
            for _, r in part_rows.iterrows():
                all_taps.extend(extract_taps_ms(r))
            all_taps = [t for t in all_taps if t0_ms <= t <= t1_ms]
            if len(all_taps) < min_taps_per_pid:
                continue
            pid_taps.append((pid, all_taps))

        n_part = len(pid_taps)
        if n_part < min_pids_per_stem:
            rows.append({"stim_canonical": stem, "n_participants": n_part, "istc": np.nan})
            continue

        pooled_taps = np.concatenate([np.asarray(ts, dtype=float) for _, ts in pid_taps])
        pooled_pids = np.concatenate([np.full(len(ts), pid) for pid, ts in pid_taps])

        counts_subj = np.zeros(len(centers))
        for i, c in enumerate(centers):
            m = (pooled_taps >= c - half) & (pooled_taps <= c + half)
            if np.any(m):
                counts_subj[i] = np.unique(pooled_pids[m]).size

        max_per_sec = []
        for k in range(len(sec_edges) - 1):
            lo, hi = sec_edges[k], sec_edges[k + 1]
            m = (centers_s >= lo) & (centers_s < hi)
            if np.any(m):
                max_per_sec.append(counts_subj[m].max())

        rows.append({
            "stim_canonical": stem,
            "n_participants": n_part,
            "istc": float(np.mean(max_per_sec)) if max_per_sec else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True,
                         help="Directory containing {ballroom,mirex,benchmark,"
                              "globalmood_40}_merged.csv")
    parser.add_argument("--out", type=Path, default=Path("istc_per_stem.csv"))
    args = parser.parse_args()

    grid_ms = np.arange(0, 30000 + 1, 5, dtype=float)
    frames = []
    for corpus_group, filename in CORPUS_FILES.items():
        path = args.merged_dir / filename
        df = pd.read_csv(path, low_memory=False)
        df["participant_uid"] = df["_source_batch"].astype(str) + "__" + df["participant_id"].astype(str)
        df = canonicalize_stems(df, corpus_group)
        istc = compute_istc_per_stem(df, grid_ms)
        istc["corpus_group"] = corpus_group
        istc = istc.rename(columns={"stim_canonical": "_stem"})
        frames.append(istc)
        print(f"{corpus_group}: {istc['istc'].notna().sum()}/{len(istc)} stems")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
