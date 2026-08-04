# Cross-cultural (country-panel) analyses

This directory contains the exploratory country-panel analyses reported in the
paper (Results &sect;3.5 and supplementary.html &sect;5.2). They ask two
questions. First, do metrical-level tapping (double, matched, or half rate) and
tap timing differ across the ten GlobalTap recruitment panels (Australia,
Brazil, Egypt, France, Japan, Mexico, Poland, South Africa, South Korea, and the
United States)? Second, is tapping coherence higher when an excerpt's source
country matches the panel that tapped it (`country_match/`)?

These are exploratory associations among Prolific recruitment panels, not
measures of cultural identity. Familiarity with individual excerpts was not
measured.

## Scripts (run in order)

- `run_globaltap_phase_periodicity_nested.py` -- for each trial, computes its
  period/phase relative to a leave-one-country-out reference and assigns a
  metrical class. Defines `build_trial_metrics`, reused by the next script.
- `run_panel_tests.py` -- the final participant-clustered, per-stimulus-
  residualized permutation tests of overall metrical-choice and phase
  differences, plus pairwise tests. Writes `panel_tests/`. Source of every
  number the paper and supplement cite for this analysis.
- `run_country_tactus_choice_bootstrap.py` -- participant-clustered
  bootstrap for the per-panel period-relation proportions. Writes
  `panel_tests/country_tactus_choice_proportions_with_ci.csv`. Needs the
  raw tapping export. The CSV is included so the figure can be regenerated
  without it.
- `plot_country_tactus_choice_summary.py` -- tall stacked layout of the same data (one
  panel per period category, 95% cluster-bootstrap error bars, dashed
  mean-across-panels lines) from
  `panel_tests/country_tactus_choice_proportions_with_ci.csv`.
- `plot_country_tactus_choice_maintext.py` -- the same data as a 2x2 grid at
  one-column width. This is Fig. 3 of the paper.
- `country_match/run_country_match_istc.py` -- estimates ISTC for each excerpt
  and panel using equally sized participant samples, then fits the own-country
  fixed-effects model. Needs the raw tapping exports, which are not included.
  Its outputs and `country_match/COUNTRY_MATCH_SUMMARY.md` are included.

`run_country_istc_cv_descriptives.py` is not run standalone here. It is
imported by `../istc_cv_tracker_correlations/` for its ISTC-pooling helper.

## Outputs

- `panel_tests/omnibus_panel_tests.csv` -- the cited $p<.001$ (metrical
  choice) and $p=.152$ (phase) values.
- `panel_tests/country_tactus_choice_proportions.csv` -- per-panel rates
  (e.g. Japan's 7.5%/20.1%, Egypt's 28.1%/9.4%).
- `panel_tests/country_tactus_choice_proportions_with_ci.csv` -- the same
  pooled rates with 95% cluster-bootstrap intervals. The source for Figure
  S3.
- `panel_tests/tactus_pairwise_panel_tests.csv`,
  `panel_tests/phase_pairwise_panel_tests.csv` -- all 45 pairwise panel
  comparisons. Nine of 45 have the minimum FDR-corrected q-value
  (see `TACTUS_PHASE_SUMMARY.md`).
- `country_tactus_choice_summary.{png,pdf}` -- tall stacked layout of the same data.
- `country_tactus_choice_maintext.{png,pdf}` -- paper Fig. 3.
- `TACTUS_PHASE_SUMMARY.md` -- the exact numbers behind the cited result.
- `country_match/` -- the own-country coherence model: the cited
  $+0.066$ ISTC units (95% CI $[-0.030, +0.161]$, $p=.179$), its
  sensitivities, and per-panel descriptives.
