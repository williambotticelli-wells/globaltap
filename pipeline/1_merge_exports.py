"""Stage 1 — Merge PsyNet TapTrialMusic exports into EXPORT_COMPAT format.

Reads every `tap_csv_sources` listed in the config, parses the nested
`analysis` JSON column, flattens:

    analysis.failed                             -> analysis_failed
    analysis.extracted_onsets.resp_onsets_aligned -> resp_onsets_aligned

Filters at **trial level** on `analysis_failed == False` (uses all valid
trials regardless of participant approval — matches the user directive
"use all valid trials for both").

Emits:
    outputs/<dataset>/merged/TapTrialMusic_EXPORT_COMPAT.csv

This CSV is the canonical input for Stage 2 (REPP alignment).

Usage:
    python pipeline/1_merge_exports.py --config config/canonical240.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(HERE))

from lib.paths import load_config  # noqa: E402


def _parse_analysis(v):
    if pd.isna(v):
        return {"failed": True, "aligned": None, "reason": "missing"}
    try:
        a = json.loads(v) if isinstance(v, str) else v
        eo = a.get("extracted_onsets")
        if isinstance(eo, str):
            eo = json.loads(eo)
        aligned = eo.get("resp_onsets_aligned") if isinstance(eo, dict) else None
        return {
            "failed": bool(a.get("failed", True)),
            "aligned": aligned,
            "reason": a.get("reason", ""),
        }
    except Exception as e:
        return {"failed": True, "aligned": None, "reason": f"parse_error:{e}"}


def process_csv(csv_path: Path, source_label: str) -> pd.DataFrame:
    print(f"  reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    rows:       {len(df)}")
    parsed = df["analysis"].apply(_parse_analysis)
    df["analysis_failed"] = parsed.apply(lambda x: x["failed"])
    df["resp_onsets_aligned"] = parsed.apply(
        lambda x: json.dumps(x["aligned"]) if x["aligned"] is not None else None
    )
    df["analysis_reason"] = parsed.apply(lambda x: x["reason"])
    df["_source_batch"] = source_label
    df["_stem"] = df["audio_filename"].apply(lambda s: Path(str(s).strip()).stem.strip())
    # participant_ids are only unique within a single export; prefix to
    # make them globally unique after merge.
    df["participant_id"] = (
        source_label + ":p" + df["participant_id"].astype(str)
    )
    keep_cols = [
        "participant_id", "audio_filename", "_stem", "stim_name",
        "analysis_failed", "resp_onsets_aligned", "analysis_reason",
        "failed", "async_post_trial_failed",
        "_source_batch", "id", "node_id", "creation_time",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    print(f"    analysis_failed==False: {(out['analysis_failed'] == False).sum()}")  # noqa: E712
    print(f"    participants:           {out['participant_id'].nunique()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/canonical240.json")
    args = ap.parse_args()
    cfg_path = HERE / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    sources = cfg["tap_csv_sources"]
    out_dir = Path(cfg.get("outputs_root", "outputs/canonical240"))
    if not out_dir.is_absolute():
        out_dir = data_root / out_dir
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {cfg['dataset']}")
    frames = []
    for src in sources:
        p = Path(src) if Path(src).is_absolute() else data_root / src
        label = p.parent.parent.parent.parent.name
        frames.append(process_csv(p, label))

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} trials across {combined['participant_id'].nunique()} participants")
    clean = combined[combined["analysis_failed"] == False].copy()  # noqa: E712
    print(f"  after analysis_failed==False filter: {len(clean)} trials  "
          f"({clean['participant_id'].nunique()} participants contributing ≥1 valid trial)")

    out_csv = merged_dir / "TapTrialMusic_EXPORT_COMPAT.csv"
    clean.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(clean)} rows)")

    # Per-stem trial count report
    counts = clean.groupby("_stem").size().sort_values()
    print(f"\nPer-stem trial counts — min={counts.min()}, median={int(counts.median())}, max={counts.max()}")
    if (counts < 10).any():
        print("  ⚠ Stems with <10 valid trials:")
        for s, n in counts[counts < 10].items():
            print(f"    {s}: {n}")


if __name__ == "__main__":
    main()
