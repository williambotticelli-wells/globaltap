# Final consensus-grid reliability

- Excerpts: 200.
- Participant subsets use the paper's per-trial MAD cleaning, 5-ms KDE and peak
  extraction, and Stage-4 optimizer.
- At each N, one disjoint A/B split was evaluated per excerpt. The KDE/peak
  convergence analysis tests more repeated subsets.
- The full-pool grids match the released annotations in
  `data/annotations/globalmood200/` exactly.

## Subset results

- independent_n_vs_n, N=10: mean F=0.828, matched-rate=0.790, median |phase|=0.021 cycles.
- independent_n_vs_n, N=20: mean F=0.893, matched-rate=0.850, median |phase|=0.016 cycles.
- independent_n_vs_n, N=30: mean F=0.916, matched-rate=0.895, median |phase|=0.012 cycles.
- independent_n_vs_n, N=50: mean F=0.918, matched-rate=0.885, median |phase|=0.009 cycles.
- subsample_vs_full, N=10: mean F=0.878, matched-rate=0.835, median |phase|=0.014 cycles.
- subsample_vs_full, N=20: mean F=0.923, matched-rate=0.877, median |phase|=0.010 cycles.
- subsample_vs_full, N=30: mean F=0.938, matched-rate=0.910, median |phase|=0.008 cycles.
- subsample_vs_full, N=50: mean F=0.951, matched-rate=0.927, median |phase|=0.005 cycles.
