#!/usr/bin/env python3
"""
Stage 6: Period audit.

For each stem, compare the dominant period (= median IOI) of:
    - reference annotation (inside 0-30 s)
    - crowd_gs
    - madmom

Report the metric-level relationship of each predictor vs reference:

    "same"       - period ratio within 1.05 (same metrical level)
    "half"       - predictor ~ 2x reference period
    "double"     - predictor ~ 0.5x reference period
    "triplet/2"  - period ratio ~1.5 (binary vs triple relation)
    "other"      - anything else

Flags:
    "clamped_lo"  - crowd_gs period <= IOI_MIN+1 ms (hit the lower bound)
    "clamped_hi"  - crowd_gs period >= IOI_MAX-1 ms (hit the upper bound)

Outputs:
    outputs/<dataset>/compare/period_audit.csv
    outputs/<dataset>/compare/PERIOD_AUDIT.md
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(HERE))
from lib.gt_loader import load_gt_for_stem  # noqa: E402
from lib.paths import load_config, resolve_config_path  # noqa: E402


def period_s(beats: np.ndarray) -> float:
    """Median IOI of a beat sequence (NaN if <2 beats)."""
    if beats.size < 2:
        return float("nan")
    return float(np.median(np.diff(np.sort(beats))))


def metric_relation(p_ref: float, p_pred: float) -> str:
    if not (np.isfinite(p_ref) and np.isfinite(p_pred)) or p_ref <= 0 or p_pred <= 0:
        return "missing"
    r = p_pred / p_ref
    if abs(r - 1.0) <= 0.05:
        return "same"
    if abs(r - 0.5) <= 0.05:
        return "double"  # predictor taps twice per GT beat
    if abs(r - 2.0) <= 0.10:
        return "half"    # predictor taps every other GT beat
    if abs(r - 1.5) <= 0.05 or abs(r - 2.0/3.0) <= 0.05:
        return "triplet/2"
    return "other"


def load_beats(p: Path) -> np.ndarray:
    if not p.exists():
        return np.array([])
    try:
        return np.atleast_1d(np.loadtxt(str(p)))
    except Exception:
        return np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config" / "canonical240.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    import os
    data_root = Path(os.path.expandvars(str(cfg["data_root"])))
    out_root = Path(cfg["outputs_root"])
    if not out_root.is_absolute():
        out_root = data_root / out_root

    gt_map = json.loads(resolve_config_path(cfg["gt_source_map"], HERE, data_root).read_text())
    manifest = json.loads(resolve_config_path(cfg["stimuli_manifest"], HERE, data_root).read_text())
    def _stem(e): return e.get("stem_id") or e.get("stimulus_id")
    def _corp(e): return e.get("dataset") or e.get("corpus")
    stem_to_corpus = {_stem(e): _corp(e) for e in manifest}

    cg_dir = out_root / "crowd_gs"
    mm_csv = out_root / "madmom" / "algorithm_beats.csv"
    mm_df = pd.read_csv(mm_csv) if mm_csv.exists() else pd.DataFrame()
    if not mm_df.empty:
        mm_df["_stem"] = mm_df["wav"].str.replace(".wav", "", regex=False)

    ioi_min = cfg["optimizer"]["ioi_min_s"]
    ioi_max = cfg["optimizer"]["ioi_max_s"]

    rows = []
    for stem, corpus in sorted(stem_to_corpus.items()):
        gt_arr = load_gt_for_stem(stem, gt_map, data_root)
        if gt_arr is None or len(gt_arr) == 0:
            continue
        gt_beats = np.sort(np.asarray(gt_arr[:, 0], dtype=float))
        p_gt = period_s(gt_beats)

        crowd = load_beats(cg_dir / f"{stem}_crowd_gs.txt")
        mm_sub = (mm_df[(mm_df["_stem"] == stem)
                        & (mm_df["algorithm"] == "madmom")]
                  if not mm_df.empty else pd.DataFrame())
        mm_beats = (np.sort(mm_sub["beat_time_s"].values.astype(float))
                    if not mm_sub.empty else np.array([]))

        p_crowd = period_s(crowd)
        p_mm = period_s(mm_beats)

        clamped_lo = bool(np.isfinite(p_crowd)
                          and p_crowd <= ioi_min + 1e-3)
        clamped_hi = bool(np.isfinite(p_crowd)
                          and p_crowd >= ioi_max - 1e-3)

        rows.append({
            "stem": stem, "corpus": corpus,
            "n_gt": len(gt_beats),
            "p_gt_ms":     round(p_gt * 1000, 1),
            "p_crowd_ms":  (round(p_crowd * 1000, 1)
                            if np.isfinite(p_crowd) else None),
            "p_madmom_ms": (round(p_mm * 1000, 1)
                            if np.isfinite(p_mm) else None),
            "rel_crowd_vs_gt":  metric_relation(p_gt, p_crowd),
            "rel_madmom_vs_gt": metric_relation(p_gt, p_mm),
            "clamped_lo": clamped_lo,
            "clamped_hi": clamped_hi,
        })

    df = pd.DataFrame(rows)
    cmp_dir = out_root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cmp_dir / "period_audit.csv", index=False)

    lines = ["# Stage 6 - Period audit\n", f"n stems = {len(df)}\n"]
    for pred in ["crowd", "madmom"]:
        rel_col = f"rel_{pred}_vs_gt"
        lines.append(f"## {pred} vs reference - metric relationships\n")
        vc = df[rel_col].value_counts()
        for k, n in vc.items():
            lines.append(f"- `{k}`: **{n}**")
        lines.append("")

    n_clamped_lo = int(df["clamped_lo"].sum())
    n_clamped_hi = int(df["clamped_hi"].sum())
    lines.append("## Boundary flags\n")
    lines.append(
        f"- Stems hitting IOI_MIN={ioi_min*1000:.0f} ms lower clamp "
        f"(crowd_gs): **{n_clamped_lo}**")
    lines.append(
        f"- Stems hitting IOI_MAX={ioi_max*1000:.0f} ms upper clamp "
        f"(crowd_gs): **{n_clamped_hi}**")
    lines.append("")

    lines.append("## % `same` metrical level per corpus\n```")
    same = df.groupby("corpus").apply(
        lambda g: 100 * (g["rel_crowd_vs_gt"] == "same").mean(),
        include_groups=False,
    ).round(1).to_frame("pct_same_crowd")
    same["pct_same_madmom"] = df.groupby("corpus").apply(
        lambda g: 100 * (g["rel_madmom_vs_gt"] == "same").mean(),
        include_groups=False,
    ).round(1)
    lines.append(same.to_string())
    lines.append("```\n")

    (cmp_dir / "PERIOD_AUDIT.md").write_text("\n".join(lines))

    print(f"Period audit complete. {len(df)} stems audited.")
    print(f"  crowd_gs at same metric level as ref: "
          f"{int((df['rel_crowd_vs_gt']=='same').sum())}/{len(df)}")
    print(f"  madmom   at same metric level as ref: "
          f"{int((df['rel_madmom_vs_gt']=='same').sum())}/{len(df)}")
    print(f"  stems clamping at IOI_MIN: {n_clamped_lo}, "
          f"IOI_MAX: {n_clamped_hi}")
    print(f"  -> {cmp_dir}/period_audit.csv, PERIOD_AUDIT.md")


if __name__ == "__main__":
    main()
