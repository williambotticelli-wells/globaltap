# Consensus beat annotations

The annotations themselves: one file per excerpt, one beat time per line, in
seconds of music time (0 s = first audio sample of the 30 s clip).

```
globalmood200/   200 files — the GlobalTap benchmark (paper §3.4–§3.5)
canonical240/    240 files — the validation set, Crowd condition (paper §3.1–§3.3, §3.6)
```

Each file is the optimized consensus grid that Stage 4 produces from the pooled
tap density, which is what the paper evaluates trackers against and what the
click-track listening stimuli were rendered from. The file contents are
identical to the pipeline's `crowd_gs` output. Only the extension differs.

The benchmark grids come from the full 746-participant pool. Median 57 beats per
excerpt (range 25–105), consistent with 30 s excerpts at listener-chosen
metrical levels.

## How to get back to these files

The chain from taps to annotations is reproducible from this repository:

```bash
python scripts/kde_peaks_from_taps.py            # data/taps/ -> KDE peaks
python pipeline/4_run_crowd_gs.py --config config/globalmood200.json  # peaks -> grids
```

Step 1 recreates
`data/stage_inputs/globalmood200/cleaned_kde_peaks.csv`, matching it excerpt
by excerpt within the 5 ms analysis grid. Step 2 turns those peaks into the
grids in `globalmood200/`. From a fresh clone, these commands reproduce all 200
files.

Evaluating the trackers in `data/stage_inputs/globalmood200/` against these files
over the paper's 10--25 s window reproduces the reported benchmark numbers:
F-measure .897 (madmom) and .841 (Beat This!), CMLt .807/.636, AMLt .956/.935,
information gain .669/.643, tracker--tracker F .875, over the 198 excerpts where
both trackers returned beats. See
[`../../supplementary_analyses/beat_metrics/`](../../supplementary_analyses/beat_metrics/).

Licensed CC BY 4.0. The source audio is not included. See
[`../SOURCES.md`](../SOURCES.md).
