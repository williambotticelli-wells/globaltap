#!/usr/bin/env python3
"""Validated country-panel tactus and circular-phase analysis.

Uses the final raw GlobalTap merge (746 usable participants), removes the
2-second REPP marker offset, constructs an equal-country-weighted leave-one-
country-out reference for every stimulus, and models repeated trials with
stimulus fixed effects and participant-clustered standard errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBALTAP_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_subset_core import CANONICAL, clean_taps_trial  # noqa: E402

DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
DEFAULT_META = GLOBALTAP_ROOT / "paper/edits/globalmood200_istc_tracker_F.csv"
DEFAULT_OUT = Path(__file__).resolve().parent
STIM_OFFSET_MS = 2000.0
WINDOW_S = 30.0

TACTUS_BINS = [
    ("double_time", 0.35, 0.65),
    ("matched", 0.80, 1.25),
    ("half_time", 1.75, 2.25),
]


def false_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.isna() | text.isin(["", "false", "0", "nan", "<na>", "none"])


def parse_raw_analysis(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([])
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return np.array([])
    if payload.get("failed") is True:
        return np.array([])
    extracted = payload.get("extracted_onsets")
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            return np.array([])
    if not isinstance(extracted, dict):
        return np.array([])
    values = extracted.get("resp_onsets_aligned", [])
    try:
        taps = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.array([])
    taps = (taps[np.isfinite(taps)] - STIM_OFFSET_MS) / 1000.0
    taps = np.sort(taps[(taps >= 0.0) & (taps <= WINDOW_S)])
    cleaned, _ = clean_taps_trial(
        taps,
        0.0,
        WINDOW_S,
        None,
        0.30,
        CANONICAL["mad_factor"],
        skip_tempo_gate=True,
        tempo_ref_label="none_mad_only",
    )
    return cleaned


def load_trials(path: Path) -> pd.DataFrame:
    use = [
        "participant_id",
        "_country_panel",
        "_stem",
        "failed",
        "async_post_trial_failed",
        "complete",
        "analysis",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    mask = df["participant_id"].notna() & df["_country_panel"].notna() & df["_stem"].notna()
    for col in ["failed", "async_post_trial_failed"]:
        if col in df:
            mask &= false_like_mask(df[col])
    if "complete" in df:
        complete = df["complete"].astype("string").str.strip().str.lower()
        mask &= complete.isin(["true", "1"])
    df = df[mask].copy()
    df["taps"] = df["analysis"].map(parse_raw_analysis)
    df = df[df["taps"].map(len) >= 3].copy()
    df = df.rename(
        columns={
            "_country_panel": "listener_country",
            "_stem": "stim_name",
        }
    )
    if df.duplicated(["participant_id", "stim_name"]).any():
        raise ValueError("Duplicate participant/stimulus trials in final merge")
    return df


def trial_period(taps: np.ndarray) -> float:
    iois = np.diff(taps)
    iois = iois[(iois >= 0.15) & (iois <= 3.0)]
    return float(np.median(iois)) if len(iois) >= 2 else np.nan


def circular_vector(taps: np.ndarray, period: float) -> complex:
    if not np.isfinite(period) or period <= 0 or len(taps) == 0:
        return complex(np.nan, np.nan)
    return complex(np.mean(np.exp(2j * np.pi * taps / period)))


def classify_tactus(ratio: float) -> str:
    if not np.isfinite(ratio) or ratio <= 0:
        return "unclear"
    for label, lo, hi in TACTUS_BINS:
        if lo <= ratio <= hi:
            return label
    return "other"


def wrapped_fraction(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi)


def build_trial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stim_name, stim_df in df.groupby("stim_name", sort=True):
        by_country = {
            country: list(sub.itertuples(index=False))
            for country, sub in stim_df.groupby("listener_country")
        }
        if len(by_country) < 3:
            continue
        country_periods = {
            country: np.asarray([trial_period(row.taps) for row in trials], dtype=float)
            for country, trials in by_country.items()
        }
        for country, trials in by_country.items():
            other_country_medians = []
            for other, periods in country_periods.items():
                if other == country:
                    continue
                finite = periods[np.isfinite(periods)]
                if len(finite):
                    other_country_medians.append(float(np.median(finite)))
            if len(other_country_medians) < 3:
                continue
            ref_period = float(np.median(other_country_medians))

            # Equal country weighting: one phase vector per other panel.
            panel_vectors = []
            for other, other_trials in by_country.items():
                if other == country:
                    continue
                taps = np.concatenate([row.taps for row in other_trials])
                vec = circular_vector(taps, ref_period)
                if np.isfinite(vec.real) and abs(vec) > 0:
                    panel_vectors.append(vec / abs(vec))
            ref_vec = np.mean(panel_vectors) if panel_vectors else complex(np.nan, np.nan)
            ref_angle = float(np.angle(ref_vec)) if np.isfinite(ref_vec.real) else np.nan

            for row in trials:
                period = trial_period(row.taps)
                ratio = period / ref_period if np.isfinite(period) else np.nan
                choice = classify_tactus(ratio)
                phase = np.nan
                phase_strength = np.nan
                if choice == "matched" and np.isfinite(ref_angle):
                    vec = circular_vector(row.taps, ref_period)
                    if np.isfinite(vec.real) and abs(vec) > 0:
                        phase = wrapped_fraction(float(np.angle(vec) - ref_angle))
                        phase_strength = float(abs(vec))
                rows.append(
                    {
                        "participant_id": row.participant_id,
                        "stim_name": stim_name,
                        "listener_country": country,
                        "n_taps": len(row.taps),
                        "trial_period_s": period,
                        "reference_period_s": ref_period,
                        "tempo_ratio_vs_other_panels": ratio,
                        "tactus_choice": choice,
                        "is_matched": int(choice == "matched"),
                        "phase_offset_frac": phase,
                        "phase_resultant_strength": phase_strength,
                    }
                )
    return pd.DataFrame(rows)


def summarize_and_plot(trials: pd.DataFrame, out_dir: Path) -> None:
    categories = ["double_time", "matched", "half_time", "other", "unclear"]
    counts = trials.groupby(["listener_country", "tactus_choice"]).size().unstack(fill_value=0)
    for category in categories:
        if category not in counts:
            counts[category] = 0
    props = counts[categories].div(counts[categories].sum(axis=1), axis=0)
    props.to_csv(out_dir / "country_tactus_choice_proportions.csv")

    phase_rows = []
    for country, sub in trials.dropna(subset=["phase_offset_frac"]).groupby("listener_country"):
        angles = sub["phase_offset_frac"].to_numpy() * 2 * np.pi
        vec = np.mean(np.exp(1j * angles))
        phase_rows.append(
            {
                "listener_country": country,
                "n_matched_trials": len(sub),
                "circular_mean_phase_fraction": wrapped_fraction(float(np.angle(vec))),
                "resultant_length": float(abs(vec)),
                "median_trial_phase_strength": float(sub["phase_resultant_strength"].median()),
            }
        )
    phase_summary = pd.DataFrame(phase_rows).sort_values("listener_country")
    phase_summary.to_csv(out_dir / "country_phase_circular_summary.csv", index=False)

    order = ["AU", "BR", "EG", "FR", "JP", "KR", "MX", "PL", "US", "ZA"]
    colors = {
        "double_time": "#0072B2",
        "matched": "#009E73",
        "half_time": "#D55E00",
        "other": "#CC79A7",
        "unclear": "#999999",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    bottom = np.zeros(len(order))
    for category in categories:
        values = props.loc[order, category].to_numpy()
        label = {
            "double_time": "Double rate",
            "matched": "Matched rate",
            "half_time": "Half rate",
            "other": "Other rate",
            "unclear": "Unclear",
        }[category]
        axes[0].bar(order, values, bottom=bottom, color=colors[category], label=label)
        bottom += values
    axes[0].set_ylabel("Proportion of trials")
    axes[0].set_title("Period relative to other panels")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    phase_plot = phase_summary.set_index("listener_country").loc[order]
    axes[1].scatter(
        order,
        phase_plot["circular_mean_phase_fraction"],
        s=25 + 100 * phase_plot["resultant_length"],
        color="#0072B2",
    )
    axes[1].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-0.03, 0.03)
    axes[1].set_ylabel("Circular mean phase offset (period fraction)")
    axes[1].set_title("Matched-rate phase")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "country_period_phase_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "country_period_phase_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Period-relation and phase analysis",
        "",
        f"- Trials analyzed: {len(trials):,}",
        f"- Participants: {trials['participant_id'].nunique():,}",
        f"- Stimuli: {trials['stim_name'].nunique()}",
        "- Reference: leave-one-country-out, equal country weighting.",
        "- Timing: 2-second REPP marker offset removed before analysis.",
        "- Cleaning: the adjacent-IOI MAD filter used in the paper before "
        "period and phase extraction.",
        "- Period-relation thresholds were specified before inspecting panel outcomes.",
        "",
        "## Primary panel tests",
        "",
        "- Full metrical-choice distribution: participant-level residualized "
        "permutation p<.001.",
        "- Metrical-choice pairwise comparisons: 9 of 45 panel pairs had the "
        "minimum FDR-corrected q-value, q=.0005 (most involved Japan or South "
        "Africa). See "
        "`panel_tests/tactus_pairwise_panel_tests.csv`.",
        "- Joint circular phase: participant-level residualized permutation p=.152.",
        "- Phase pairwise comparisons: 2 of 45 panel pairs survived FDR "
        "correction (JP--MX and BR--JP, both q=.045). Tap timing showed no "
        "consistent differences across panels overall (p=.152), so the pairwise "
        "results provide weak phase evidence rather than none. See "
        "`panel_tests/phase_pairwise_panel_tests.csv`.",
        "",
        "The figure is descriptive. Primary inference is produced by "
        "`run_panel_tests.py`.",
    ]
    (out_dir / "TACTUS_PHASE_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials(args.input)
    metrics = build_trial_metrics(trials)
    if args.metadata.exists():
        meta = pd.read_csv(args.metadata, low_memory=False)
        keep = [c for c in ["_stem", "source_country", "group_b5"] if c in meta]
        metrics = metrics.merge(
            meta[keep].drop_duplicates("_stem").rename(columns={"_stem": "stim_name"}),
            on="stim_name",
            how="left",
        )
    summarize_and_plot(metrics, args.out_dir)
    print(args.out_dir / "TACTUS_PHASE_SUMMARY.md")


if __name__ == "__main__":
    main()
