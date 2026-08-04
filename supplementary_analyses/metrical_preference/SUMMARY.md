# Metrical-preference analysis

This analysis includes all 180 excerpts with independent expert references.
Rating differences are excerpt-level Crowd minus Reference means. Each
excerpt's `metrical_class` is the ratio of the crowd's median tapping
period to the reference beat period, classified as matched/double/half
rate when that ratio falls within +/-0.15 octaves (about +/-11%) of 1x,
0.5x, or 2x respectively (otherwise non-integer).

| Relation | n | Crowd higher | Median Δ | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Matched rate | 145 | 51 | -0.143 | [-0.231, -0.077] |
| Crowd half rate | 9 | 9 | +0.748 | [+0.231, +0.882] |
| Crowd double rate | 14 | 7 | -0.238 | [-1.231, +0.412] |
| Non-integer | 12 | 7 | +0.393 | [-0.811, +1.030] |

All nine half-rate excerpts combine fast reference beats (median IOIs
0.28–0.37 s, the `gt_median_ioi_s` column) with crowd tapping near
preferred rates (0.57–0.75 s, `crowd_median_ioi_s`).
