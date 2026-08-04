# Recruitment-panel tests

- The analysis includes 23,492 trials, 746 participants, and 200 excerpts. One
  of the 23,493 usable
  benchmark trials has fewer than three taps after cleaning, too few for a
  period estimate, so it is excluded here.
- Per-trial taps were cleaned with the adjacent-IOI MAD filter used in the
  paper.
- The tests of overall panel differences remove excerpt-level differences,
  average each participant's remaining values, and permute panel labels across
  participants.
- Each panel is compared with a reference built from the other nine panels.
- Full metrical-choice distribution: statistic=7.8471, permutation p=0.0001.
- Joint circular phase: statistic=1.6975, permutation p=0.1516.
- In the pairwise metrical-choice tests, 9 of 45 panel pairs have the minimum
  FDR-corrected q=0.0005. See `tactus_pairwise_panel_tests.csv` for the full
  table.
