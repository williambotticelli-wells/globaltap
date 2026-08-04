# Beat metrics beyond F-measure

- Evaluation window: 10--25 s.
- The tapping-derived GlobalTap consensus is the `mir_eval` reference and each tracker's beat sequence is the estimate.
- GlobalTap consensus stems: 200.
- Paired algorithm comparisons include 198 stems. Two are excluded because
  Beat This! returned fewer than two beats in the evaluation window.
- Metrics computed with `mir_eval` 0.8.2.
- Paired two-sided Wilcoxon tests compare algorithms stem by stem. BH-FDR is
  applied across all metric comparisons.

## Key results

`delta` below is Beat This! minus madmom.

- Any Metric Level Continuous: madmom=0.947, Beat This!=0.908, delta=-0.039, q=0.01052.
- Any Metric Level Total: madmom=0.956, Beat This!=0.935, delta=-0.022, q=0.02406.
- Correct Metric Level Continuous: madmom=0.797, Beat This!=0.621, delta=-0.176, q=1.324e-06.
- Correct Metric Level Total: madmom=0.807, Beat This!=0.636, delta=-0.171, q=1.768e-06.
- F-measure: madmom=0.897, Beat This!=0.841, delta=-0.056, q=6.317e-05.
- Tracker-tracker F-measure (madmom vs Beat This!, same 198 paired stems and
  same 10-25 s window): 0.875. Reported in the paper for scale.
  Recomputed from `data/stage_inputs/globalmood200/` with mir_eval 0.8.2.
- Information gain: madmom=0.669, Beat This!=0.643, delta=-0.026, q=2.794e-06.
