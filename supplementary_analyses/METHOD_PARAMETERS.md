# Method settings and sensitivity tests

The pipeline uses fixed heuristic settings. They were selected during
development using checks on Ballroom, the mixed corpora, and MIREX, then used
unchanged for every reported analysis, including the GlobalTap benchmark.
Because the same validation corpora informed both development and evaluation,
the sensitivity test below checks how much the output depends on these choices.

The config files record the main settings. Stage 4 reads its window and KDE
settings from the config. Several other optimizer values come from matching
Python constants or function defaults, including some values also written in
the JSON for documentation. The table below lists the values the code uses.
The sensitivity script changes them by passing alternative values directly to
the optimizer.

| Setting | Value | What it does and why it was used |
|---|---:|---|
| KDE grid | 5 ms | Computational resolution well below the 70-ms evaluation tolerance. |
| KDE bandwidth | 80 ms | Favored overall in an earlier 50/80/120-ms cross-corpus comparison. A broader 40–160-ms check found relatively stable period and beat-count behavior around 70–100 ms. |
| Trial-level MAD filter | 3.5 scaled MADs, up to 3 passes | In a 100-excerpt test, enabling or disabling MAD cleaning changed post-optimizer mean F-measure by at most 1 percentage point, with no consistent direction across tolerances. A separate check on 10 difficult excerpts produced the same mean agreement at factors 3.5 and 4.5, so 3.5 was not uniquely best. |
| Peak prominence | 0.10 of maximum | Selected with the distance and cleaning settings in a 24-condition development test. The leading prominence values performed similarly. |
| Minimum peak distance | 150 ms | Included in the 24-condition test. This choice had little effect because prominent peaks were already more than 200 ms apart. |
| Period search | 0.28–1.25 s | Bounds the fitted near-isochronous grid to 48–214 BPM. |
| Period/phase search | 120 / 240 steps | Provides fixed finite resolution before local refinement. |
| Density/coverage weight | 1.0 / 0.5 | Balances support at grid beats with coverage of extracted peaks. |
| Peak-match tolerance | 0.12T + 20 ms | Scales matching tolerance with the candidate period while retaining a small fixed allowance. |
| Merge / insert thresholds | 0.6T / 1.4T | Removes close duplicates and fills large gaps in the near-isochronous grid. |
| Bimodal gap threshold | 0.10T | Triggers comparison of two offset phase families when inter-peak intervals separate. |

The rows above summarize the bandwidth comparison, the peak and cleaning test,
and the MAD-filter test. The remaining optimizer values were set during
development.

To test their effect, we changed each final-grid setting to one lower and one
higher value across all 240 validation excerpts. The Stage-2 peaks and all
other settings stayed fixed:

| Setting | Paper setting | Values tested |
|---|---:|---:|
| Period-grid steps | 120 | 80, 160 |
| Phase-grid steps | 240 | 160, 320 |
| Peak-fit weight | 0.5 | 0.25, 0.75 |
| Snap weight | 0.45 | 0.25, 0.65 |
| Snap-max fraction (of T) | 0.12 | 0.08, 0.16 |
| Bimodal gap threshold (of T) | 0.10 | 0.08, 0.12 |
| Minimum IOI bound | 0.28 s | 0.23 s, 0.33 s |

Agreement with the paper's output remained high (mean F-measure .947--1.000,
with the tactus unchanged in 92.1--100% of cases).
Changing the bimodality and snapping settings had almost no effect. Period-grid
resolution, period bounds, and peak fit weight produced the largest changes.
Mean agreement with corpus references changed by -.014 to +.012. Results and
the script are in `parameter_sensitivity/`.

## Final-grid behavior

Two final-grid behaviors are set in the code rather than the config:

Snapping pulls each beat partway toward its nearest peak rather than onto it.
The shift is 45% of the distance to the peak (`snap_weight`), capped at 0.12 of
the period (`snap_max_frac`), so a beat settles between its fitted position and
the peak. Both values are changed in the sensitivity test above, where the
output changed very little (`optimization/optimization_regularize.py`).

The grid score S(T, φ) = D̄ + 0.5C ranks periods and phases within one fit.
The optimizer may then compare the standard fit, two single-phase fits after a
bimodal split, and a balanced-weight refit when CV is high. It chooses among
them using peak coverage, isochrony (1 - IOI CV), and KDE density at the beats
(`score_result` in `optimization/run_pipeline.py`). When Stage 4 runs from the
included `data/stage_inputs/` files, `stage4_summary.csv` records which branch
was used for each excerpt in the `crowd_gs_path` column.
