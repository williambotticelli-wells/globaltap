#!/usr/bin/env python3
"""Build the de-identified per-tap release in ``data/taps/``.

Runs against the raw session exports (not included in this repository) and
writes the public tap tables: one row per tap, plus a trial index. Tap onsets
are parsed with the same routine the published analyses use
(``supplementary_analyses/pipeline_subset_core.parse_analysis_taps``) and
trials are kept with the same usable-trial rule as the reported counts, so the
released onsets are the music-time taps that feed Stage 2 of the pipeline.

Everything that could identify a participant is dropped: the export keeps only
a pseudonymous participant key, the stimulus, the recruitment panel or corpus,
and the tap times.

Usage:
    python scripts/build_tap_release.py --raw-dir /path/to/merged --out-dir data/taps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supplementary_analyses"))
from pipeline_subset_core import parse_analysis_taps  # noqa: E402

DATASETS = {
    "globalmood200": {
        "files": ["globalmood_200_merged.csv"],
        "prefix": "gt",
    },
}
SOURCE_COLS = [
    "participant_id",
    "_stem",
    "_corpus",
    "_country_panel",
    "_source_batch",
    "analysis",
    "failed",
    "analysis_failed",
    "async_post_trial_failed",
]
TAP_DECIMALS = 6  # microseconds; audio onset resolution is ~23 us at 44.1 kHz


def falsey(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(["", "false", "0", "nan", "<na>", "none"])


def usable_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Same filter as the reported trial counts and Stage 1 of the pipeline."""
    mask = pd.Series(True, index=df.index)
    for col in ("failed", "analysis_failed", "async_post_trial_failed"):
        if col in df.columns:
            mask &= falsey(df[col])
    mask &= df["participant_id"].notna() & df["_stem"].notna()
    return df[mask]


def read_trials(raw_dir: Path, files: list[str]) -> pd.DataFrame:
    frames = []
    for name in files:
        for chunk in pd.read_csv(
            raw_dir / name,
            usecols=lambda c: c in SOURCE_COLS,
            low_memory=False,
            chunksize=1500,
        ):
            chunk = usable_trials(chunk)
            if chunk.empty:
                continue
            chunk = chunk.assign(
                taps=[parse_analysis_taps(v) for v in chunk["analysis"].to_numpy()]
            ).drop(columns=["analysis"])
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def build(dataset: str, raw_dir: Path, out_dir: Path) -> None:
    spec = DATASETS[dataset]
    trials = read_trials(raw_dir, spec["files"])

    originals = sorted(trials["participant_id"].astype(str).unique())
    keys = {
        pid: f"{spec['prefix']}-{i:04d}" for i, pid in enumerate(originals, start=1)
    }
    trials["participant"] = trials["participant_id"].astype(str).map(keys)
    trials["n_taps"] = [int(t.size) for t in trials["taps"]]

    taps = pd.DataFrame(
        {
            "participant": np.repeat(trials["participant"], trials["n_taps"]),
            "stem_id": np.repeat(trials["_stem"].astype(str), trials["n_taps"]),
            "tap_time_s": np.round(
                np.concatenate(list(trials["taps"])) if len(trials) else [],
                TAP_DECIMALS,
            ),
        }
    ).sort_values(["stem_id", "participant", "tap_time_s"])

    index = trials.rename(
        columns={
            "_stem": "stem_id",
            "_corpus": "corpus",
            "_country_panel": "country_panel",
            "_source_batch": "source_batch",
        }
    ).reindex(
        columns=[
            "participant",
            "stem_id",
            "corpus",
            "country_panel",
            "source_batch",
            "n_taps",
        ]
    ).sort_values(["stem_id", "participant"])

    out_dir.mkdir(parents=True, exist_ok=True)
    taps.to_csv(out_dir / f"{dataset}_taps.csv.gz", index=False, compression="gzip")
    index.to_csv(out_dir / f"{dataset}_trials.csv.gz", index=False, compression="gzip")

    per_stem = index[index["n_taps"] > 0].groupby("stem_id")["participant"].nunique()
    print(f"[{dataset}]")
    print(f"  participants : {index['participant'].nunique():,}")
    print(f"  excerpts     : {index['stem_id'].nunique():,}")
    print(f"  trials       : {len(index):,}")
    print(f"  taps         : {len(taps):,}")
    print(
        "  tappers/stem : "
        f"min={per_stem.min()} median={int(per_stem.median())} max={per_stem.max()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/taps"))
    ap.add_argument(
        "--dataset", choices=sorted(DATASETS) + ["all"], default="all"
    )
    args = ap.parse_args()
    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        build(name, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
