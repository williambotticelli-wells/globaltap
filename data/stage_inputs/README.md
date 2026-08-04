# Included pipeline outputs

This directory contains de-identified outputs from Stages 2–3 of the
pipeline. They come from the runs used in the paper. With these files,
**Stage 4 (`pipeline/4_run_crowd_gs.py`) can be
run from a fresh clone** — no `PROJECT_ROOT` and no raw audio required.

Running Stage 4 from these files reproduces the `cv_pipeline` values in
`data/per_stem_metrics.csv` for all
240 stems — all 100 mixed-corpora, all 80 Ballroom, all 20 MIREX, and all 40
supplementary GlobalMood-40 stems. Differences are limited to the 5-decimal
rounding in the released CSV. This uses the paper's Stage-2
setting `peak_prominence=0.10` (`config/canonical240.json` /
`config/globalmood200.json`).

```
canonical240/   240-stem validation set (paper §3.2–§3.3, §3.6)
globalmood200/  200-stem GlobalTap cross-cultural benchmark (paper §3.4–§3.5)
```

Each subdirectory contains:

| File                                  | Stage | Purpose                                            |
| ------------------------------------- | ----- | -------------------------------------------------- |
| `cleaned_kde_peaks.csv`               | 2     | KDE peaks after MAD-based outlier cleaning (Stage-4 input) |
| `algorithm_beats.csv`                 | 3     | Madmom DBN F-measure baseline beats                |
| `beat_this_algorithm_beats.csv`       | 3b    | Beat This! transformer F-measure baseline beats    |

The benchmark taps behind the GlobalMood200 files are released in
`data/taps/`, and `scripts/kde_peaks_from_taps.py` rebuilds
`cleaned_kde_peaks.csv` from them. Only the benchmark taps are released; the
validation peaks and annotations here are the inputs and grids that match the
rated stimuli (`data/taps/README.md` has more).
All included `cleaned_kde_peaks.csv`, algorithm-beat tables, and optimized
`crowd_gs` beats are already in **music time** (0 s = first audio sample of
the 30 s clip).

`pipeline/4_run_crowd_gs.py` will look in
`${PROJECT_ROOT}/outputs/<dataset>/{repp,madmom}/` first. If that path
is absent (e.g. `PROJECT_ROOT` is unset, as in a fresh clone), it
uses the files here, so running

```bash
python pipeline/4_run_crowd_gs.py --config config/canonical240.json
```

from a fresh clone regenerates the `crowd_gs` beat files without any
external inputs. The resulting `cv_pipeline` values match
`data/per_stem_metrics.csv` for the full 240-stem set. The repository does not
include a second copy of the raw `.txt` beat files.
`data/per_stem_metrics.csv` contains the released per-stem values derived from
them.

The raw audio, original session exports, and reference annotations from
the source corpora are not included in this repo — see
`data/SOURCES.md` for how to fetch them.
