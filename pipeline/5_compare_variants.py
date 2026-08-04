#!/usr/bin/env python3
"""
Stage 5: Compare crowd_gs against baseline beat trackers per stem.

For each stem with a reference annotation, loads the beat sequence from
Stage 4 (crowd_gs) and from any available baselines (madmom, beat_this),
then computes:

- F-measure @ 70 ms (Davies et al. 2009 standard tolerance)
- F-measure @ 120 ms (wider tolerance, separates phase drift from
  tempo errors)
- Phase-aligned F-measure (small \u00b1250 ms shift, only when tempos match)
- Median absolute offset on hits (ms)
- mir_eval continuity scores (CMLc/CMLt/AMLc/AMLt)
- Beat counts in 0\u201330 s and in the listening window

Produces:
    outputs/<dataset>/compare/variant_metrics_per_stem.csv
    outputs/<dataset>/compare/variant_summary_per_corpus.csv
    outputs/<dataset>/compare/stats_wilcoxon.csv
    outputs/<dataset>/compare/stats_wilcoxon_per_corpus.csv
    outputs/<dataset>/compare/SUMMARY.md
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # companion_repo/
sys.path.insert(0, str(HERE))
from lib.gt_loader import load_gt_for_stem  # noqa: E402
from lib.paths import load_config, resolve_config_path  # noqa: E402

try:
    import mir_eval
    HAS_MIREVAL = True
except ImportError:
    HAS_MIREVAL = False

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


TOL_S = 0.07        # standard F-measure tolerance (Davies et al. 2009)
TOL_WIDE_S = 0.12   # wider tolerance to separate phase drift from tempo errors


def mir_continuity(pred: np.ndarray, gt: np.ndarray) -> dict:
    if not HAS_MIREVAL or pred.size < 2 or gt.size < 2:
        return {"CMLc": np.nan, "CMLt": np.nan,
                "AMLc": np.nan, "AMLt": np.nan}
    try:
        ref_t = mir_eval.beat.trim_beats(np.asarray(gt, dtype=float))
        est_t = mir_eval.beat.trim_beats(np.asarray(pred, dtype=float))
        CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(ref_t, est_t)
        return {"CMLc": float(CMLc), "CMLt": float(CMLt),
                "AMLc": float(AMLc), "AMLt": float(AMLt)}
    except Exception:
        return {"CMLc": np.nan, "CMLt": np.nan,
                "AMLc": np.nan, "AMLt": np.nan}


def estimate_phase_offset(pred: np.ndarray, gt: np.ndarray,
                          max_shift_s: float = 0.25) -> float:
    if pred.size < 2 or gt.size < 2:
        return 0.0
    P_pred = float(np.median(np.diff(np.sort(pred))))
    P_gt = float(np.median(np.diff(np.sort(gt))))
    if (P_pred <= 0 or P_gt <= 0
            or max(P_pred, P_gt) / min(P_pred, P_gt) > 1.1):
        return 0.0
    shifts = np.arange(-max_shift_s, max_shift_s + 0.001, 0.005)
    best, best_score = 0.0, -1
    for s in shifts:
        shifted = pred + s
        score = sum(1 for t in gt
                    if np.min(np.abs(shifted - t)) <= TOL_S)
        if score > best_score:
            best_score, best = score, s
    return float(best)


def f_measure(pred: np.ndarray, gt: np.ndarray, tol: float = TOL_S) -> dict:
    if pred.size == 0 or gt.size == 0:
        return {"F": np.nan, "P": np.nan, "R": np.nan,
                "med_abs_off_ms": np.nan, "n_hits": 0}
    hits_gt = 0
    offsets = []
    for t in gt:
        d = np.abs(pred - t)
        if d.size and d.min() <= tol:
            hits_gt += 1
            offsets.append(pred[d.argmin()] - t)
    hits_pred = 0
    for t in pred:
        d = np.abs(gt - t)
        if d.size and d.min() <= tol:
            hits_pred += 1
    R = hits_gt / len(gt)
    P = hits_pred / len(pred) if len(pred) else 0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0
    med_abs = (float(np.median(np.abs(offsets)) * 1000)
               if offsets else np.nan)
    return {"F": F, "P": P, "R": R,
            "med_abs_off_ms": med_abs, "n_hits": len(offsets)}


def load_beats(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        return np.array([])
    try:
        b = np.atleast_1d(np.loadtxt(str(txt_path)))
    except Exception:
        return np.array([])
    return np.asarray(b, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(HERE / "config" / "canonical240.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root_str = str(cfg["data_root"])
    if "${" in os.path.expandvars(data_root_str):
        # PROJECT_ROOT unset: fall back to repo-local outputs so a fresh
        # clone can run Stage 5 against the included stage_inputs/.
        data_root = HERE
        out_root = HERE / "outputs" / cfg["dataset"]
        print(f"[stage5] PROJECT_ROOT unset; using repo-local out_root={out_root}")
    else:
        data_root = Path(os.path.expandvars(data_root_str))
        out_root = Path(cfg["outputs_root"])
        if not out_root.is_absolute():
            out_root = data_root / out_root

    gt_map = json.loads(
        resolve_config_path(cfg["gt_source_map"], HERE, data_root).read_text())
    manifest = json.loads((data_root / cfg["stimuli_manifest"]).read_text())

    def _stem(e): return e.get("stem_id") or e.get("stimulus_id")
    def _corp(e): return e.get("dataset") or e.get("corpus")
    stem_to_corpus = {_stem(e): _corp(e) for e in manifest}

    cg_dir = out_root / "crowd_gs"
    madmom_csv = out_root / "madmom" / "algorithm_beats.csv"
    beat_this_csv = out_root / "beat_this" / "algorithm_beats.csv"
    # Fall back to the stage-3 outputs included in the repo so
    # the F-measure baselines can be reproduced without needing to
    # actually run Madmom / Beat This! locally (paper §3.4).
    included = HERE / "data" / "stage_inputs" / cfg["dataset"]
    if not madmom_csv.exists() and (included / "algorithm_beats.csv").exists():
        madmom_csv = included / "algorithm_beats.csv"
    if not beat_this_csv.exists() and (included / "beat_this_algorithm_beats.csv").exists():
        beat_this_csv = included / "beat_this_algorithm_beats.csv"
    cmp_dir = out_root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    mm_df = pd.read_csv(madmom_csv) if madmom_csv.exists() else pd.DataFrame()
    if not mm_df.empty:
        mm_df["_stem"] = mm_df["wav"].str.replace(".wav", "", regex=False)
    bt_df = pd.read_csv(beat_this_csv) if beat_this_csv.exists() else pd.DataFrame()
    if not bt_df.empty:
        bt_df["_stem"] = bt_df["wav"].str.replace(".wav", "", regex=False)

    stems = sorted(stem_to_corpus.keys())
    t10, t25 = 10.0, 25.0
    rows = []
    for stem in stems:
        corpus = stem_to_corpus[stem]
        gt_arr = load_gt_for_stem(stem, gt_map, data_root)
        if gt_arr is None or len(gt_arr) == 0:
            continue
        gt = np.sort(np.asarray(gt_arr[:, 0], dtype=float))

        variants = {
            "crowd_gs": load_beats(cg_dir / f"{stem}_crowd_gs.txt"),
        }
        if not mm_df.empty:
            sub = mm_df[(mm_df["_stem"] == stem)
                        & (mm_df["algorithm"] == "madmom")]
            variants["madmom"] = np.sort(
                sub["beat_time_s"].values.astype(float))
        if not bt_df.empty:
            sub = bt_df[(bt_df["_stem"] == stem)
                        & (bt_df["algorithm"] == "beat_this")]
            variants["beat_this"] = np.sort(
                sub["beat_time_s"].values.astype(float))

        row = {"stem": stem, "corpus": corpus, "n_gt": len(gt)}
        for variant, beats in variants.items():
            m = f_measure(beats, gt, tol=TOL_S)
            m_wide = f_measure(beats, gt, tol=TOL_WIDE_S)
            phase_shift = estimate_phase_offset(beats, gt)
            m_aligned = (f_measure(beats + phase_shift, gt, tol=TOL_S)
                         if phase_shift != 0.0 else m)
            cont = mir_continuity(beats, gt)
            row[f"{variant}_F"] = m["F"]
            row[f"{variant}_F_wide"] = m_wide["F"]
            row[f"{variant}_F_phase_aligned"] = m_aligned["F"]
            row[f"{variant}_phase_shift_ms"] = round(phase_shift * 1000, 1)
            row[f"{variant}_medoff_ms"] = m["med_abs_off_ms"]
            row[f"{variant}_CMLt"] = cont["CMLt"]
            row[f"{variant}_AMLt"] = cont["AMLt"]
            row[f"{variant}_n_30s"] = (
                int(((beats >= 0) & (beats <= 30)).sum())
                if beats.size else 0)
            row[f"{variant}_n_10_25"] = (
                int(((beats >= t10) & (beats <= t25)).sum())
                if beats.size else 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cmp_dir / "variant_metrics_per_stem.csv", index=False)

    if df.empty:
        note = (cmp_dir / "NO_GT_DATASET.md")
        note.write_text(
            "# No reference annotations in this dataset\n\n"
            f"All {len(stems)} stems in this config lack a reference "
            "annotation (e.g. cross-cultural corpora such as "
            "GlobalMood200). Stage 5 reference-anchored comparisons are "
            "therefore not applicable for this run.\n"
        )
        print(f"[stage5] No annotated stems found; wrote {note}.")
        return

    # corpus summary
    variant_cols = [c for c in df.columns if c.endswith("_F")]
    per_corpus = (df.groupby("corpus")[variant_cols]
                  .agg(["mean", "median"]).round(3))
    per_corpus.to_csv(cmp_dir / "variant_summary_per_corpus.csv")

    # Wilcoxon: paired per-stem crowd_gs vs each baseline, on F & AMLt
    wilcox_rows = []
    baselines = [v for v in ["madmom", "beat_this"] if f"{v}_F" in df.columns]
    for metric in ["F", "AMLt"]:
        ref_col = f"crowd_gs_{metric}"
        if ref_col not in df.columns:
            continue
        for b in baselines:
            b_col = f"{b}_{metric}"
            if b_col not in df.columns:
                continue
            paired = df[[ref_col, b_col]].dropna()
            if len(paired) < 5 or not HAS_SCIPY:
                continue
            try:
                stat, p = wilcoxon(paired[ref_col], paired[b_col],
                                   zero_method="wilcox",
                                   alternative="two-sided")
                delta_median = float(
                    (paired[ref_col] - paired[b_col]).median())
            except Exception:
                continue
            wilcox_rows.append({
                "metric": metric,
                "crowd_vs": b,
                "n_stems": len(paired),
                "median_delta_crowd_minus_baseline": round(delta_median, 4),
                "wilcoxon_W": float(stat),
                "wilcoxon_p_two_sided": float(p),
            })
    per_corpus_wilcox = []
    if "beat_this_F" in df.columns and HAS_SCIPY:
        for corp, g in df.groupby("corpus"):
            paired = g[["crowd_gs_F", "beat_this_F"]].dropna()
            if len(paired) < 5:
                continue
            try:
                stat, p = wilcoxon(paired["crowd_gs_F"],
                                   paired["beat_this_F"],
                                   alternative="two-sided",
                                   zero_method="wilcox")
                per_corpus_wilcox.append({
                    "corpus": corp,
                    "n_stems": len(paired),
                    "crowd_F_mean":  float(paired["crowd_gs_F"].mean()),
                    "beat_this_F_mean": float(paired["beat_this_F"].mean()),
                    "median_delta_crowd_minus_bt": float(
                        (paired["crowd_gs_F"] - paired["beat_this_F"]).median()),
                    "wilcoxon_p": float(p),
                })
            except Exception:
                pass

    if wilcox_rows:
        ws = pd.DataFrame(wilcox_rows).sort_values("wilcoxon_p_two_sided")
        m = len(ws); ws = ws.reset_index(drop=True)
        ws["rank"] = ws.index + 1
        ws["q_bh"] = (ws["wilcoxon_p_two_sided"] * m / ws["rank"]).clip(upper=1.0)
        ws.to_csv(cmp_dir / "stats_wilcoxon.csv", index=False)
    if per_corpus_wilcox:
        pd.DataFrame(per_corpus_wilcox).to_csv(
            cmp_dir / "stats_wilcoxon_per_corpus.csv", index=False)

    # Summary report
    report = []
    report.append("# Stage 5 \u2014 crowd_gs vs baselines\n")
    report.append(f"**n stems with reference**: {len(df)}\n")
    report.append(f"**F tolerance**: {TOL_S*1000:.0f} ms\n\n")
    report.append("## Mean F-measure per variant\n")
    for c in variant_cols:
        report.append(
            f"- `{c.removesuffix('_F')}`: **{df[c].mean():.3f}** "
            f"(median {df[c].median():.3f}, min {df[c].min():.3f})")
    report.append("")
    report.append("## Per-corpus mean F\n```")
    corpus_tbl = df.groupby("corpus")[variant_cols].mean().round(3)
    report.append(corpus_tbl.to_string())
    report.append("```\n")
    if wilcox_rows:
        report.append("## Wilcoxon (crowd_gs vs baselines, paired)\n")
        for r in wilcox_rows:
            report.append(
                f"- {r['metric']} \u2014 crowd vs {r['crowd_vs']}: "
                f"\u0394 = {r['median_delta_crowd_minus_baseline']:+.3f} "
                f"(p={r['wilcoxon_p_two_sided']:.3g})")
        report.append("")
    (cmp_dir / "SUMMARY.md").write_text("\n".join(report))

    print("=" * 60)
    print(f"Mean F per variant (n={len(df)} stems with reference):")
    for c in variant_cols:
        print(f"  {c.removesuffix('_F'):20s}  {df[c].mean():.3f}  "
              f"(median {df[c].median():.3f})")
    if wilcox_rows:
        print("\nWilcoxon (crowd_gs vs baselines, paired):")
        for r in wilcox_rows:
            print(f"  {r['metric']:>5s}  crowd vs {r['crowd_vs']:<10s}  "
                  f"\u0394={r['median_delta_crowd_minus_baseline']:+.3f}  "
                  f"p={r['wilcoxon_p_two_sided']:.3g}")
    print(f"\nOutputs -> {cmp_dir}")


if __name__ == "__main__":
    main()
