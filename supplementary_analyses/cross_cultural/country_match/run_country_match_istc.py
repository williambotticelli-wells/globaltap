#!/usr/bin/env python3
"""Phase 5.2 — Country-matched tapping coherence (ISTC).

Computes inter-subject tapping coherence (ISTC; Granot et al. 2023) for each
excerpt and tapping country using equally sized participant samples, then runs
an excerpt- and tapping-country-fixed-effects model for the five balanced source
countries (US, KR, FR, MX, EG) plus all-10-country descriptive summaries.

ISTC follows the strict participant-count definition:
    - bin tapping events in 100 ms bins with 50 ms step (overlapping)
    - count each participant at most once per bin
    - for each second [0..30], record the maximum participant count
    - ISTC = mean of those per-second maxima

Primary model:
    Equal-N ISTC ~ source/tapper-country match + stem fixed effects
                   + tapping-country fixed effects,
    restricted to the 100 stems balanced across the five source countries.
    Standard errors are clustered by stem.

Descriptive: all 10 countries, own vs other-9 delta, per stem averaged
    across each tapping country's own-country stems.

Usage:
    python3 scripts/country_istc_analysis.py \\
        --config config/globalmood200_cc.json

Companion-repo copy of the pipeline script
`pipeline/scripts/country_istc_analysis.py`. It reads the controlled
trial-level table, which is not part of the public release; the outputs it
produces are in this directory and summarized in
`COUNTRY_MATCH_SUMMARY.md`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent.parent  # GlobalTap_v1.0/pipeline
GT_ROOT = HERE.parent
DEFAULT_CANONICAL_MERGE = (
    GT_ROOT / "raw_tapping_data" / "merged" / "globalmood_200_merged.csv"
)


# ---------- ISTC ----------

def safe_json_list(x) -> list[float]:
    """Parse a string-encoded JSON list of floats, tolerant of ragged input."""
    if x is None:
        return []
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return []
    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return []
        try:
            vals = json.loads(s)
        except Exception:
            return []
    if not isinstance(vals, list):
        return []
    out: list[float] = []
    for v in vals:
        try:
            f = float(v)
            if np.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            continue
    return out


def _bin_centers_ms(t0_s: float, t1_s: float, overlap_ms: float) -> np.ndarray:
    return np.arange(t0_s * 1000, t1_s * 1000 + overlap_ms, overlap_ms)


def _sec_bin_slices(centers_ms: np.ndarray, t0_s: float, t1_s: float) -> list[tuple[int, int]]:
    """Pre-compute [start_idx, stop_idx) slices of centers_ms indices that
    fall into each 1-second window [t0..t1). Returns one slice per second.
    """
    centers_s = centers_ms / 1000.0
    sec_edges = np.arange(t0_s, t1_s + 1e-9, 1.0)
    slices: list[tuple[int, int]] = []
    for i in range(len(sec_edges) - 1):
        lo, hi = sec_edges[i], sec_edges[i + 1]
        mask = (centers_s >= lo) & (centers_s < hi)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            slices.append((int(idxs[0]), int(idxs[-1] + 1)))
    return slices


def participant_bin_presence(
    taps_ms: list[float],
    centers_ms: np.ndarray,
    bin_ms: float,
    stim_offset_ms: float,
) -> np.ndarray:
    """Return binary per-bin presence for one participant's tap sequence."""
    if not taps_ms:
        return np.zeros(centers_ms.size, dtype=np.int32)
    arr = np.asarray(taps_ms, dtype=float) - stim_offset_ms
    half = bin_ms / 2.0
    # Each tap contributes +1 to every bin whose center is within half of it.
    # Vectorised: for each tap find range of bin-indices via searchsorted on centers.
    lo = centers_ms[0]
    step = centers_ms[1] - centers_ms[0] if centers_ms.size > 1 else 1.0
    counts = np.zeros(centers_ms.size, dtype=np.int32)
    for t in arr:
        # bins whose centers c satisfy |c - t| <= half
        lo_c = t - half
        hi_c = t + half
        i0 = int(np.ceil((lo_c - lo) / step))
        i1 = int(np.floor((hi_c - lo) / step))
        if i0 < 0:
            i0 = 0
        if i1 >= counts.size:
            i1 = counts.size - 1
        if i0 <= i1:
            counts[i0:i1 + 1] += 1
    return (counts > 0).astype(np.int32)


def false_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(["", "false", "0", "nan", "<na>", "none"])


def taps_from_raw_analysis(value: object) -> list[float]:
    """Extract aligned response onsets from the final raw PsyNet merge."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if payload.get("failed") is True:
        return []
    extracted = payload.get("extracted_onsets")
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            return []
    if not isinstance(extracted, dict):
        return []
    return safe_json_list(extracted.get("resp_onsets_aligned"))


def istc_from_counts(pooled_counts: np.ndarray, sec_slices: list[tuple[int, int]]) -> float:
    if not sec_slices:
        return 0.0
    maxes = [pooled_counts[a:b].max() for a, b in sec_slices]
    return float(np.mean(maxes))


# ---------- equal-size participant sampling ----------

def cell_istc_subsampled(
    cell_df: pd.DataFrame,
    *,
    m: int,
    n_boot: int,
    centers_ms: np.ndarray,
    sec_slices: list[tuple[int, int]],
    bin_ms: float,
    stim_offset_ms: float,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Return (mean ISTC over n_boot random samples of size m, n_participants).

    Each participant's per-bin counts are cached once; bootstrap
    iterations just sum those cached vectors. Orders of magnitude faster
    than re-binning per iteration.
    """
    g = cell_df.drop_duplicates(subset=["participant_id"], keep="first")
    taps_by_pid: list[list[float]] = [
        safe_json_list(row) for row in g["resp_onsets_aligned"].tolist()
    ]
    taps_by_pid = [t for t in taps_by_pid if len(t) >= 2]
    n = len(taps_by_pid)
    if n < m:
        return float("nan"), n

    # Pre-compute per-participant count vectors (n x n_bins).
    P = np.stack([
        participant_bin_presence(t, centers_ms, bin_ms, stim_offset_ms)
        for t in taps_by_pid
    ])  # (n, n_bins)

    idxs = np.arange(n)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(idxs, size=m, replace=False)
        pooled = P[pick].sum(axis=0)
        vals[b] = istc_from_counts(pooled, sec_slices)
    return float(np.mean(vals)), n


# ---------- Analysis ----------

def build_country_mapping(cfg: dict) -> dict[str, str]:
    mapping = cfg["country_mapping"]
    return dict(mapping)


def _batch_to_country(src: str, mapping: dict[str, str]) -> str | None:
    for key, cc in mapping.items():
        if key in src:
            return cc
    return None


def load_source_country(cfg: dict) -> pd.DataFrame:
    """Map stem_id (YouTube videoID) -> ISO2 source country."""
    root = Path(cfg["data_root"])
    path = root / cfg["source_country_csv"]
    df = pd.read_csv(path)
    df = df.rename(columns={"videoID": "stem_id", "iso2": "source_country"})
    return df[["stem_id", "source_country"]].copy()


def compute_per_cell(
    merged_csv: Path, cfg: dict, out_dir: Path
) -> pd.DataFrame:
    print(f"[istc] loading {merged_csv}")
    df = pd.read_csv(merged_csv, low_memory=False)
    if "resp_onsets_aligned" not in df.columns:
        if "analysis" not in df.columns:
            raise ValueError("Input needs resp_onsets_aligned or raw analysis")
        df["resp_onsets_aligned"] = df["analysis"].map(taps_from_raw_analysis)
    if "_country_panel" in df.columns:
        df["tapping_country"] = df["_country_panel"].astype("string")
    else:
        mapping = build_country_mapping(cfg)
        df["tapping_country"] = df["_source_batch"].apply(
            lambda s: _batch_to_country(str(s), mapping)
        )
    if "stim_name" in df.columns and "_stem" not in df.columns:
        df["_stem"] = df["stim_name"]
    usable = df["participant_id"].notna() & df["_stem"].notna()
    for col in ["failed", "async_post_trial_failed"]:
        if col in df.columns:
            usable &= false_like_mask(df[col])
    if "complete" in df.columns:
        complete = df["complete"].astype("string").str.strip().str.lower()
        usable &= complete.isin(["true", "1"])
    df = df[usable].copy()
    df = df[df["resp_onsets_aligned"].map(lambda x: len(safe_json_list(x)) >= 2)]
    before = len(df)
    df = df[df["tapping_country"].notna()].copy()
    print(f"[istc] mapped {len(df)}/{before} rows to a tapping country")

    # ISTC config
    istc_cfg = cfg["istc"]
    sub_cfg = cfg["subsampling"]
    stim_offset_ms = float(cfg.get("stim_offset_s", 0.0)) * 1000.0
    m = int(sub_cfg["subsample_n"])
    n_boot = int(sub_cfg["n_boot"])
    rng = np.random.default_rng(int(sub_cfg.get("rng_seed", 20260424)))

    stems = sorted(df["_stem"].dropna().unique().tolist())
    countries = cfg["all_tapping_countries"]

    centers_ms = _bin_centers_ms(
        istc_cfg["t0_s"], istc_cfg["t1_s"], istc_cfg["overlap_ms"]
    )
    sec_slices = _sec_bin_slices(
        centers_ms, istc_cfg["t0_s"], istc_cfg["t1_s"]
    )

    rows = []
    import time
    t_start = time.time()
    for i, stem in enumerate(stems):
        if i % 10 == 0:
            rate = (i + 1) / max(time.time() - t_start, 1e-6)
            eta = (len(stems) - i - 1) / max(rate, 1e-6)
            print(f"[istc] stem {i+1}/{len(stems)}: {stem}  "
                  f"(rate={rate:.2f} stems/s, eta={eta:.0f}s)",
                  flush=True)
        for c in countries:
            cell = df[(df["_stem"] == stem) & (df["tapping_country"] == c)]
            if len(cell) == 0:
                rows.append({
                    "stem": stem, "tapping_country": c,
                    "n_participants": 0, "istc_mean": float("nan"),
                })
                continue
            val, n_part = cell_istc_subsampled(
                cell, m=m, n_boot=n_boot,
                centers_ms=centers_ms, sec_slices=sec_slices,
                bin_ms=istc_cfg["bin_ms"],
                stim_offset_ms=stim_offset_ms, rng=rng,
            )
            rows.append({
                "stem": stem, "tapping_country": c,
                "n_participants": n_part, "istc_mean": val,
            })
    per_cell = pd.DataFrame(rows)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "cells" / "istc_per_stem_country.csv", index=False)
    print(f"[istc] wrote {len(per_cell)} excerpt-country combinations")
    return per_cell


def add_source_country(per_cell: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    src_df = load_source_country(cfg)
    # stem (== stim ID in the Prolific deployments) matches videoID in
    # global_pop_200_stimuli.csv
    merged = per_cell.merge(src_df, left_on="stem", right_on="stem_id", how="left")
    merged = merged.drop(columns=["stem_id"])
    n_missing = merged["source_country"].isna().sum()
    if n_missing > 0:
        print(f"[istc] WARNING: {n_missing} combinations missing source_country")
    return merged


# ---------- Phase 5.2 primary: balanced five paired ----------

def wilcoxon_or_nan(diffs: np.ndarray) -> tuple[float, float]:
    """Return (W, p_one_sided for >0). Returns (nan, nan) on degenerate input."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 3 or np.allclose(diffs, 0.0):
        return float("nan"), float("nan")
    try:
        W, p_two = stats.wilcoxon(diffs, zero_method="wilcox", alternative="greater")
        return float(W), float(p_two)
    except Exception:
        return float("nan"), float("nan")


def cohens_dz(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return float("nan")
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def balanced_five_analysis(
    per_cell: pd.DataFrame, cfg: dict, out_dir: Path
) -> pd.DataFrame:
    balanced = cfg["balanced_five"]
    descriptive = cfg["descriptive_five"]
    all_countries = cfg["all_tapping_countries"]

    # Wide frame: rows = stems, cols = ISTC per tapping country
    wide = per_cell.pivot_table(
        index=["stem", "source_country"],
        columns="tapping_country",
        values="istc_mean",
        aggfunc="first",
    ).reset_index()

    # For each country c, compute per-stem Δ_s = ISTC[c,s] − mean(ISTC[other9,s]).
    # Primary test: difference between Δ on own-country stems vs Δ on non-own
    # stems (baseline-adjusted country-matching effect; controls for country-
    # level differences in tapping coherence such as MX > everyone globally).
    country_results = []
    per_country_stems = {}
    for c in all_countries:
        other_cols = [x for x in all_countries if x != c]
        full = wide.copy()
        full["delta_istc"] = full[c] - full[other_cols].mean(axis=1, skipna=True)
        full = full.dropna(subset=["delta_istc"])
        own = full[full["source_country"] == c].copy()
        non_own = full[full["source_country"] != c].copy()
        if len(own) == 0:
            continue
        # Primary: baseline-adjusted per-stem diff for the own-stems is
        # (Δ_own − mean(Δ_non_own)); test "own Δ > non-own Δ" with Mann-Whitney
        # (own-stems are independent of non-own-stems, so unpaired).
        delta_own_vec = own["delta_istc"].values
        delta_non_own_vec = non_own["delta_istc"].values
        if len(delta_non_own_vec) >= 3 and len(delta_own_vec) >= 2:
            try:
                U, p_mwu = stats.mannwhitneyu(
                    delta_own_vec, delta_non_own_vec, alternative="greater"
                )
            except Exception:
                U, p_mwu = float("nan"), float("nan")
        else:
            U, p_mwu = float("nan"), float("nan")
        # Effect size: Cohen's d (between two independent samples) on the Δ values
        if (len(delta_own_vec) > 1 and len(delta_non_own_vec) > 1):
            s1 = delta_own_vec.std(ddof=1); s2 = delta_non_own_vec.std(ddof=1)
            n1, n2 = len(delta_own_vec), len(delta_non_own_vec)
            sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / max(n1+n2-2, 1))
            d_adj = (delta_own_vec.mean() - delta_non_own_vec.mean()) / sp if sp > 0 else float("nan")
        else:
            d_adj = float("nan")

        # Also retain the unadjusted per-country "own − other9" stat (what
        # feedback-1 first suggested) as a sanity number.
        W_raw, p_raw = wilcoxon_or_nan(delta_own_vec)
        dz_raw = cohens_dz(delta_own_vec)

        country_results.append({
            "country": c,
            "role": "balanced" if c in balanced else "descriptive",
            "n_own_stems": len(own),
            "n_non_own_stems": len(non_own),
            "mean_own_istc": float(own[c].mean()),
            "mean_other9_istc": float(own[other_cols].mean(axis=1).mean()),
            "mean_delta_istc_own":    float(delta_own_vec.mean()),
            "mean_delta_istc_non_own": float(delta_non_own_vec.mean()),
            "baseline_adj_delta":     float(delta_own_vec.mean() - delta_non_own_vec.mean()),
            "se_delta_istc":          float(delta_own_vec.std(ddof=1) / np.sqrt(len(delta_own_vec))) if len(delta_own_vec) > 1 else float("nan"),
            "cohens_d_adj":           float(d_adj),
            "mannwhitney_U":          float(U) if np.isfinite(U) else float("nan"),
            "p_adj_one_sided_gt0":    float(p_mwu) if np.isfinite(p_mwu) else float("nan"),
            "cohens_dz_raw":          float(dz_raw),
            "p_raw_one_sided_gt0":    float(p_raw) if np.isfinite(p_raw) else float("nan"),
        })
        per_country_stems[c] = own.copy()

    res = pd.DataFrame(country_results).set_index("country").reindex(all_countries).reset_index()
    res.to_csv(out_dir / "country_delta_summary.csv", index=False)

    # Meta-analysis across the 5 balanced countries.  We combine the
    # baseline-adjusted country-matching effect (own-stem Δ − non-own-stem Δ)
    # via fixed-effect inverse-variance weighting on Cohen's d (unpaired).
    meta_rows = res[res["role"] == "balanced"].copy()
    d_adj = meta_rows["cohens_d_adj"].to_numpy(dtype=float)
    n1 = meta_rows["n_own_stems"].to_numpy(dtype=float)
    n2 = meta_rows["n_non_own_stems"].to_numpy(dtype=float)
    # SE(Cohen's d, independent samples; Hedges & Olkin eq. 10)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d_adj ** 2 / (2 * (n1 + n2)))
    w = 1.0 / (se_d ** 2)
    mu = float(np.sum(w * d_adj) / np.sum(w))
    se_mu = float(np.sqrt(1.0 / np.sum(w)))
    z = mu / se_mu
    p_meta = float(1.0 - stats.norm.cdf(z))  # one-sided > 0
    ci_lo = mu - 1.96 * se_mu
    ci_hi = mu + 1.96 * se_mu

    # Also report the raw within-country d_z for completeness.
    dz_raw = meta_rows["cohens_dz_raw"].to_numpy(dtype=float)
    se_dz_raw = np.sqrt(1.0 / n1 + dz_raw ** 2 / (2 * n1))
    w_raw = 1.0 / (se_dz_raw ** 2)
    mu_raw = float(np.sum(w_raw * dz_raw) / np.sum(w_raw))
    se_mu_raw = float(np.sqrt(1.0 / np.sum(w_raw)))
    p_raw = float(1.0 - stats.norm.cdf(mu_raw / se_mu_raw))

    with open(out_dir / "meta_balanced_five.json", "w") as f:
        json.dump({
            "k_countries": len(meta_rows),
            "primary_adj": {
                "per_country_d_adj": dict(zip(meta_rows["country"].tolist(), d_adj.tolist())),
                "per_country_n_own": dict(zip(meta_rows["country"].tolist(), n1.astype(int).tolist())),
                "per_country_n_non_own": dict(zip(meta_rows["country"].tolist(), n2.astype(int).tolist())),
                "fixed_effect_d": mu,
                "se_d": se_mu,
                "ci95_d": [ci_lo, ci_hi],
                "z": z,
                "p_one_sided_gt0": p_meta,
                "n_countries_with_pos_effect": int((d_adj > 0).sum()),
            },
            "raw_within_country": {
                "fixed_effect_dz": mu_raw,
                "se_dz": se_mu_raw,
                "p_one_sided_gt0": p_raw,
                "per_country_dz": dict(zip(meta_rows["country"].tolist(), dz_raw.tolist())),
                "_note": "ignores country-level baseline ISTC differences (e.g. MX globally higher)",
            },
        }, f, indent=2)

    per_country_dir = out_dir / "per_country_stems"
    per_country_dir.mkdir(exist_ok=True)
    for c, sub in per_country_stems.items():
        sub.to_csv(per_country_dir / f"{c}_stems.csv", index=False)

    return res


def balanced_five_fixed_effect_model(
    per_cell: pd.DataFrame, cfg: dict, out_dir: Path
) -> dict:
    """Primary nested analysis with stem and tapping-country fixed effects."""
    balanced = set(cfg["balanced_five"])
    model_df = per_cell[
        per_cell["source_country"].isin(balanced)
        & per_cell["tapping_country"].isin(cfg["all_tapping_countries"])
    ].dropna(subset=["istc_mean"]).copy()
    model_df["source_match"] = (
        model_df["source_country"] == model_df["tapping_country"]
    ).astype(int)
    fit = smf.ols(
        "istc_mean ~ source_match + C(stem) + C(tapping_country)",
        data=model_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["stem"]})
    term = "source_match"
    effect = float(fit.params[term])
    se = float(fit.bse[term])
    ci = [float(x) for x in fit.conf_int().loc[term]]
    result = {
        "analysis": "balanced_five_stem_and_tapping_country_fixed_effects",
        "n_cells": int(len(model_df)),
        "n_stems": int(model_df["stem"].nunique()),
        "n_tapping_countries": int(model_df["tapping_country"].nunique()),
        "n_source_matched_cells": int(model_df["source_match"].sum()),
        "effect_istc_participants_per_bin": effect,
        "cluster_robust_se": se,
        "ci95": ci,
        "t": float(fit.tvalues[term]),
        "p_two_sided": float(fit.pvalues[term]),
        "formula": "istc_mean ~ source_match + C(stem) + C(tapping_country)",
        "se_cluster": "stem",
    }
    with open(out_dir / "balanced_five_fixed_effect_model.json", "w") as f:
        json.dump(result, f, indent=2)
    model_df.to_csv(out_dir / "balanced_five_model_cells.csv", index=False)
    with open(out_dir / "balanced_five_fixed_effect_model.txt", "w") as f:
        f.write(fit.summary().as_text())
    return result


def expanded_fixed_effect_sensitivities(
    per_cell: pd.DataFrame, cfg: dict, out_dir: Path
) -> dict:
    """Fit match models on all eligible stems and on the full benchmark.

    Source-country matching is identified only for excerpts whose source
    country is one of the ten recruitment countries. The all-200 model includes
    the other excerpts when estimating panel baselines, but those excerpts have
    no matched panel and therefore do not directly identify the match term.
    """

    all_countries = set(cfg["all_tapping_countries"])
    specifications = {
        "match_eligible_110": per_cell[
            per_cell["source_country"].isin(all_countries)
        ].copy(),
        "full_benchmark_200": per_cell.copy(),
    }
    results = {}
    for label, model_df in specifications.items():
        model_df = model_df.dropna(subset=["istc_mean"]).copy()
        model_df["source_match"] = (
            model_df["source_country"] == model_df["tapping_country"]
        ).astype(int)
        fit = smf.ols(
            "istc_mean ~ source_match + C(stem) + C(tapping_country)",
            data=model_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": model_df["stem"]})
        term = "source_match"
        result = {
            "analysis": label,
            "n_cells": int(len(model_df)),
            "n_stems": int(model_df["stem"].nunique()),
            "n_match_eligible_stems": int(
                model_df.loc[
                    model_df["source_country"].isin(all_countries), "stem"
                ].nunique()
            ),
            "n_source_matched_cells": int(model_df["source_match"].sum()),
            "effect_istc_participants_per_bin": float(fit.params[term]),
            "cluster_robust_se": float(fit.bse[term]),
            "ci95": [float(x) for x in fit.conf_int().loc[term]],
            "t": float(fit.tvalues[term]),
            "p_two_sided": float(fit.pvalues[term]),
            "formula": "istc_mean ~ source_match + C(stem) + C(tapping_country)",
            "se_cluster": "stem",
            "identification_note": (
                "The match coefficient is identified by the 110 excerpts whose "
                "source country overlaps a recruitment panel."
            ),
        }
        results[label] = result
        model_df.to_csv(out_dir / f"{label}_model_cells.csv", index=False)
        (out_dir / f"{label}_fixed_effect_model.txt").write_text(
            fit.summary().as_text()
        )

    with open(out_dir / "expanded_fixed_effect_sensitivities.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------- Option B robustness: all-110 stem-weighted ----------

def option_b_robustness(
    per_cell: pd.DataFrame, cfg: dict, out_dir: Path
) -> dict:
    all_countries = cfg["all_tapping_countries"]
    wide = per_cell.pivot_table(
        index=["stem", "source_country"],
        columns="tapping_country",
        values="istc_mean",
        aggfunc="first",
    ).reset_index()
    # Restrict to stems whose source country is one of the 10 tapping countries
    sub = wide[wide["source_country"].isin(all_countries)].copy()
    if len(sub) == 0:
        return {}
    # D_s = ISTC[source_country_tappers, s] - mean(ISTC[other 9 tapping countries, s])
    deltas = []
    for _, row in sub.iterrows():
        c = row["source_country"]
        own = row[c]
        others = [row[x] for x in all_countries if x != c and np.isfinite(row[x])]
        if not np.isfinite(own) or not others:
            continue
        deltas.append(own - float(np.mean(others)))
    deltas = np.array(deltas, dtype=float)
    W, p_one = wilcoxon_or_nan(deltas)
    dz = cohens_dz(deltas)
    out = {
        "n_stems_used": int(deltas.size),
        "mean_delta_istc": float(np.mean(deltas)),
        "median_delta_istc": float(np.median(deltas)),
        "cohens_dz": dz,
        "wilcoxon_W": W,
        "p_one_sided_gt0": p_one,
    }
    with open(out_dir / "option_b_all110.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------- Summary markdown ----------

def write_summary(
    out_dir: Path,
    res: pd.DataFrame,
    cfg: dict,
    opt_b: dict,
    fe_model: dict,
    expanded_models: dict,
) -> None:
    meta = json.loads((out_dir / "meta_balanced_five.json").read_text())
    m_adj = meta["primary_adj"]
    m_raw = meta["raw_within_country"]
    lines = [
        "# Phase 5.2 — Country-matched tapping coherence",
        "",
        "## Design",
        "- **Primary (inferential)**: balanced-five = " + ", ".join(cfg["balanced_five"]) + ".",
        "- **Descriptive**: all 10 tapping countries (underpowered 5: "
            + ", ".join(cfg["descriptive_five"]) + ", n_own=2).",
        "- **ISTC**: 100 ms bins, 50 ms overlap; mean max-per-second tap count; "
            "each participant counted at most once per bin; Granot et al. 2023 JoC.",
        f"- **Participant sampling**: ISTC for each excerpt and panel is averaged "
            f"across {cfg['subsampling']['n_boot']} random samples of "
            f"{cfg['subsampling']['subsample_n']} tappers. Combinations with fewer "
            f"than {cfg['subsampling']['min_tappers_per_cell']} tappers are dropped.",
        "- **Secondary descriptive effect**: *baseline-adjusted* per-stem diff. For each "
            "country c and stem s: Δ_s = ISTC[c,s] − mean_{c′≠c} ISTC[c′,s]. "
            "The country-matching effect is the contrast Δ on own-country "
            "stems − Δ on non-own stems (Mann–Whitney one-sided). This "
            "controls for global country-level differences in tapping "
            "coherence (e.g. MX ISTC is +0.25 higher than other countries "
            "on average, across *all* 200 stems).",
        "",
        "## Balanced-five nested fixed-effects model (primary)",
        f"- Excerpt-panel combinations: {fe_model['n_cells']} across "
            f"{fe_model['n_stems']} excerpts and "
            f"{fe_model['n_tapping_countries']} tapping countries. "
            f"{fe_model['n_source_matched_cells']} combinations are source-matched.",
        f"- Source/tapper-country match effect: "
            f"{fe_model['effect_istc_participants_per_bin']:+.3f} ISTC units "
            f"(cluster-robust SE {fe_model['cluster_robust_se']:.3f}, "
            f"95% CI [{fe_model['ci95'][0]:+.3f}, {fe_model['ci95'][1]:+.3f}], "
            f"two-sided p={fe_model['p_two_sided']:.4f}).",
        "- Model includes stem and tapping-country fixed effects; standard "
            "errors are clustered by stem.",
        "",
        "## Expanded fixed-effects sensitivities",
        (
            "- **All 110 match-eligible excerpts**: effect "
            f"{expanded_models['match_eligible_110']['effect_istc_participants_per_bin']:+.3f}, "
            f"95% CI [{expanded_models['match_eligible_110']['ci95'][0]:+.3f}, "
            f"{expanded_models['match_eligible_110']['ci95'][1]:+.3f}], "
            f"two-sided p={expanded_models['match_eligible_110']['p_two_sided']:.4f}."
        ),
        (
            "- **All 200 excerpts included**: effect "
            f"{expanded_models['full_benchmark_200']['effect_istc_participants_per_bin']:+.3f}, "
            f"95% CI [{expanded_models['full_benchmark_200']['ci95'][0]:+.3f}, "
            f"{expanded_models['full_benchmark_200']['ci95'][1]:+.3f}], "
            f"two-sided p={expanded_models['full_benchmark_200']['p_two_sided']:.4f}. "
            "The match coefficient is still identified only by the 110 excerpts "
            "whose source country has a recruitment panel."
        ),
        "",
        "## Per-country summary",
        "",
        res.round(3).to_markdown(index=False),
        "",
        "## Country-wise standardized-effect meta-analysis (secondary sensitivity)",
        f"- k countries: {meta['k_countries']}",
        f"- Fixed-effect Cohen's d = **{m_adj['fixed_effect_d']:+.3f}** "
            f"(95% CI [{m_adj['ci95_d'][0]:+.3f}, {m_adj['ci95_d'][1]:+.3f}]), "
            f"z = {m_adj['z']:.2f}, p (one-sided > 0) = {m_adj['p_one_sided_gt0']:.4f}.",
        f"- Per-country d_adj: " + ", ".join(
            f"{c}={d:+.3f}" for c, d in m_adj['per_country_d_adj'].items()
        ),
        f"- Countries with positive adjusted effect: "
            f"{m_adj['n_countries_with_pos_effect']}/{meta['k_countries']}",
        "",
        "## Raw within-country effect (for comparison; NOT baseline-adjusted)",
        f"- Fixed-effect d_z = {m_raw['fixed_effect_dz']:+.3f} "
            f"(SE {m_raw['se_dz']:.3f}), p = {m_raw['p_one_sided_gt0']:.4f}",
        "- *This is inflated for MX because MX tappers are globally more "
            "coherent regardless of source country.*",
        "",
    ]
    if opt_b:
        lines += [
            "## Option B robustness (all 110 source-matched stems, stem-weighted)",
            f"- n stems: {opt_b['n_stems_used']}",
            f"- mean ΔISTC = {opt_b['mean_delta_istc']:+.3f}, "
                f"d_z = {opt_b['cohens_dz']:+.3f}, "
                f"Wilcoxon p (one-sided > 0) = {opt_b['p_one_sided_gt0']:.4f}",
            "- Option B is raw (not baseline-adjusted); same caveat applies.",
            "",
        ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/globalmood200_cc.json")
    ap.add_argument(
        "--merged-csv",
        default=str(DEFAULT_CANONICAL_MERGE),
        help="Final raw merge or pipeline-compatible merged trial CSV.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(GT_ROOT / "analysis" / "cross_cultural" / "validated_country_istc"),
    )
    args = ap.parse_args()

    cfg_path = HERE / args.config
    cfg = json.loads(cfg_path.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = Path(args.merged_csv)
    if not merged_csv.exists():
        raise SystemExit(f"Missing merged CSV {merged_csv}.")

    per_cell = compute_per_cell(merged_csv, cfg, out_dir)
    per_cell = add_source_country(per_cell, cfg)
    per_cell.to_csv(out_dir / "cells" / "istc_per_stem_country_labeled.csv", index=False)

    res = balanced_five_analysis(per_cell, cfg, out_dir)
    fe_model = balanced_five_fixed_effect_model(per_cell, cfg, out_dir)
    expanded_models = expanded_fixed_effect_sensitivities(per_cell, cfg, out_dir)
    opt_b = option_b_robustness(per_cell, cfg, out_dir)
    write_summary(out_dir, res, cfg, opt_b, fe_model, expanded_models)

    print()
    print(f"[istc] wrote {out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
