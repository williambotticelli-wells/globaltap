# GlobalTap consensus convergence

- Stems: 200.
- Median usable tappers per stem: 117 (range 114--120).
- Each tapper contributes a normalized smoothed tap density, so prolific tappers do not receive more weight.
- The primary curve compares random N-tapper consensuses with the full
  consensus. Because those subsamples overlap with the full pool, which can
  inflate agreement, an independent-groups curve is also reported.
- Confidence intervals bootstrap stems, not repeated subsamples.
- Smallest tested N with mean peak F-measure >= .90: 30.
- Smallest tested N with mean KDE correlation >= .90: 40.
