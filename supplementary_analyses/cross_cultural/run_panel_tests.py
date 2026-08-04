#!/usr/bin/env python3
"""Participant-clustered panel tests for metrical choice and circular phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


HERE = Path(__file__).resolve().parent
SUPPLEMENTARY = HERE.parent
sys.path.insert(0, str(SUPPLEMENTARY))
sys.path.insert(0, str(HERE))

from pipeline_subset_core import GLOBALTAP_ROOT, load_and_clean_trials  # noqa: E402
from run_globaltap_phase_periodicity_nested import build_trial_metrics  # noqa: E402


DEFAULT_INPUT = GLOBALTAP_ROOT / "raw_tapping_data/merged/globalmood_200_merged.csv"
DEFAULT_META = GLOBALTAP_ROOT / "paper/edits/globalmood200_istc_tracker_F.csv"
DEFAULT_OUT = HERE / "panel_tests"
CATEGORIES = ["double_time", "matched", "half_time", "other", "unclear"]


def residualized_participant_vectors(
    trials: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    values = trials[
        ["participant_id", "listener_country", "stim_name", *value_columns]
    ].dropna(subset=value_columns)
    for column in value_columns:
        values[f"{column}_resid"] = (
            values[column] - values.groupby("stim_name")[column].transform("mean")
        )
    residual_columns = [f"{column}_resid" for column in value_columns]
    return (
        values.groupby(["participant_id", "listener_country"], as_index=False)[
            residual_columns
        ]
        .mean()
        .rename(columns={f"{column}_resid": column for column in value_columns})
    )


def between_group_stat(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    grand = values.mean(axis=0)
    total = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        total += len(group) * float(np.sum((group.mean(axis=0) - grand) ** 2))
    return total


def permutation_test(
    participants: pd.DataFrame,
    value_columns: list[str],
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float, float]:
    values = participants[value_columns].to_numpy(float)
    labels = participants["listener_country"].to_numpy(str)
    observed = between_group_stat(values, labels)
    exceed = 0
    for _ in range(permutations):
        if between_group_stat(values, rng.permutation(labels)) >= observed:
            exceed += 1
    return observed, (exceed + 1) / (permutations + 1)


def pairwise_tests(
    participants: pd.DataFrame,
    value_columns: list[str],
    rng: np.random.Generator,
    permutations: int,
) -> pd.DataFrame:
    countries = sorted(participants["listener_country"].unique())
    rows = []
    for i, country_a in enumerate(countries):
        for country_b in countries[i + 1 :]:
            sub = participants[
                participants["listener_country"].isin([country_a, country_b])
            ]
            values = sub[value_columns].to_numpy(float)
            labels = sub["listener_country"].to_numpy(str)
            observed = between_group_stat(values, labels)
            exceed = 0
            for _ in range(permutations):
                if between_group_stat(values, rng.permutation(labels)) >= observed:
                    exceed += 1
            means = sub.groupby("listener_country")[value_columns].mean()
            row = {
                "country_a": country_a,
                "country_b": country_b,
                "n_a": int((labels == country_a).sum()),
                "n_b": int((labels == country_b).sum()),
                "distance_stat": observed,
                "permutation_p": (exceed + 1) / (permutations + 1),
            }
            for column in value_columns:
                row[f"{column}_a_minus_b"] = float(
                    means.loc[country_a, column] - means.loc[country_b, column]
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    result["fdr_q"] = false_discovery_control(
        result["permutation_p"].to_numpy(), method="bh"
    )
    return result.sort_values("permutation_p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cleaned = load_and_clean_trials(args.input)
    base = cleaned.rename(columns={"stem": "stim_name", "clean_taps": "taps"})
    trials = build_trial_metrics(base)
    meta = pd.read_csv(args.metadata, low_memory=False)
    keep = [
        column
        for column in ["_stem", "source_country", "group_b5"]
        if column in meta
    ]
    trials = trials.merge(
        meta[keep].drop_duplicates("_stem").rename(columns={"_stem": "stim_name"}),
        on="stim_name",
        how="left",
    )

    one_hot = pd.get_dummies(trials["tactus_choice"]).reindex(
        columns=CATEGORIES, fill_value=0
    )
    tactus = pd.concat([trials.reset_index(drop=True), one_hot], axis=1)
    tactus_participants = residualized_participant_vectors(tactus, CATEGORIES)
    tactus_stat, tactus_p = permutation_test(
        tactus_participants, CATEGORIES, rng, args.permutations
    )
    tactus_pairs = pairwise_tests(
        tactus_participants, CATEGORIES, rng, args.permutations
    )
    tactus_pairs.to_csv(args.out_dir / "tactus_pairwise_panel_tests.csv", index=False)

    phase = trials.dropna(subset=["phase_offset_frac"]).copy()
    angles = phase["phase_offset_frac"].to_numpy(float) * 2 * np.pi
    phase["phase_sin"] = np.sin(angles)
    phase["phase_cos"] = np.cos(angles)
    phase_participants = residualized_participant_vectors(
        phase, ["phase_sin", "phase_cos"]
    )
    phase_stat, phase_p = permutation_test(
        phase_participants, ["phase_sin", "phase_cos"], rng, args.permutations
    )
    pd.DataFrame(
        [
            {
                "outcome": "full_metrical_choice",
                "statistic": tactus_stat,
                "permutation_p": tactus_p,
            },
            {
                "outcome": "joint_circular_phase",
                "statistic": phase_stat,
                "permutation_p": phase_p,
            },
        ]
    ).to_csv(args.out_dir / "omnibus_panel_tests.csv", index=False)
    phase_pairs = pairwise_tests(
        phase_participants,
        ["phase_sin", "phase_cos"],
        rng,
        args.permutations,
    )
    phase_pairs.to_csv(args.out_dir / "phase_pairwise_panel_tests.csv", index=False)

    descriptive = (
        trials.groupby(["listener_country", "tactus_choice"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORIES, fill_value=0)
    )
    descriptive = descriptive.div(descriptive.sum(axis=1), axis=0)
    descriptive.to_csv(args.out_dir / "country_tactus_choice_proportions.csv")

    tactus_floor = tactus_pairs["fdr_q"].min()
    n_at_floor = int(
        np.isclose(tactus_pairs["fdr_q"], tactus_floor, atol=1e-9).sum()
    )
    n_pairs = len(tactus_pairs)
    lines = [
        "# Recruitment-panel tests",
        "",
        f"- The analysis includes {len(trials):,} trials, "
        f"{trials['participant_id'].nunique():,} participants, and "
        f"{trials['stim_name'].nunique()} excerpts.",
        "- Per-trial taps were cleaned with the adjacent-IOI MAD filter used "
        "in the paper.",
        "- The tests of overall panel differences remove excerpt-level "
        "differences, average each participant's remaining values, and permute "
        "panel labels across participants.",
        "- Each panel is compared with a reference built from the other nine "
        "panels.",
        f"- Full metrical-choice distribution: statistic={tactus_stat:.4f}, "
        f"permutation p={tactus_p:.4g}.",
        f"- Joint circular phase: statistic={phase_stat:.4f}, "
        f"permutation p={phase_p:.4g}.",
        f"- Metrical-choice pairwise tests: {n_at_floor} of {n_pairs} panel "
        f"pairs have the minimum FDR-corrected q={tactus_floor:.4g}. See "
        "`tactus_pairwise_panel_tests.csv` "
        "for the full pairwise table.",
    ]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
