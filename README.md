# GlobalTap — Companion Code Repository (ISMIR 2026)

Source code, configuration, and per-stem data for the ISMIR 2026
submission **"GlobalTap: Globally Diverse Beat Tracking Benchmark via
Crowdsourced Tapping."** This repository contains the pipeline used for the
paper, the de-identified tap data, and scripts supporting the paper's numerical
claims and figures. A few analyses that need the raw audio ship their code and
summary outputs here. A
self-contained HTML
supplement (`supplementary.html`) provides per-corpus rating
distributions, per-source breakdowns of the mixed-corpora panel,
participant demographics, and text descriptions of the §3.6 case studies.

`${PROJECT_ROOT}` is a portable placeholder set by the user via environment
variable (or by editing the dataset config) — see §2. Tap data uses
pseudonymous participant labels. The release contains no platform identifiers,
timestamps, or participant-level demographics.

---

## 0. Dataset summary

| Component | Count |
|---|---|
| Total participants | **1,628** (1,498 tappers + 130 listener raters) |
| Tapping trials | 36,970 |
| Parsed taps | 2,205,932 |
| Released tap onsets ([`data/taps/`](data/taps/)) | 1,201,342 in the 23,493 usable benchmark trials |
| Released consensus annotations ([`data/annotations/`](data/annotations/)) | 440 excerpts (200 benchmark + 240 validation) |
| Listener ratings (validation) | 6,487 (median 15 ratings per stem and condition) |
| GlobalTap benchmark excerpts | **200** (from 59 source countries) |
| Country panels of tappers | **10**: US, KR, FR, MX, EG, AU, BR, JP, PL, ZA (median 66, range 49–111) |
| Validation set excerpts | **240** (80 Ballroom + 100 Mixed corpora + 20 MIREX + 40 GlobalMood) |
| Reference-paired validation excerpts | 200 (Ballroom + Mixed + MIREX) |

Per-panel and per-corpus breakdowns are tabulated in
[`supplementary.html`](supplementary.html) §3 (Demographics) and §2
(Per-source mixed-corpora panel).

---

## 1. Repository layout

```
.
├── README.md                  # This file (architecture + reproduction)
├── supplementary.html         # Self-contained HTML supplement
├── LICENSE                    # MIT for code and CC BY 4.0 for data
├── requirements.txt           # Pinned Python deps
├── run_pipeline.sh            # End-to-end driver
│
├── pipeline/                  # Stage scripts (run in order, config-driven)
│   ├── 0_build_gt_source_map.py   # Resolve annotator reference-beat files
│   ├── 1_merge_exports.py         # Merge tap CSVs and exclude failed analyses
│   ├── 2_repp_align.py            # REPP alignment → cleaned KDE peaks
│   ├── 3_run_madmom.py            # Madmom DBN F-measure baseline
│   ├── 3_run_beat_this.py         # Beat This! transformer F-measure baseline
│   ├── 4_run_crowd_gs.py          # Build the crowd consensus beat grid
│   ├── 5_compare_variants.py      # F-measure + offset comparisons
│   ├── 6_period_audit.py          # Compare reference and crowd periods
│   └── 7_render_clicks.py         # Render click-overlay listening stimuli
│
├── lib/                       # Shared helpers used by the stages
│   ├── gt_loader.py               # Annotator-original beat-file loader
│   ├── manifest.py                # Records inputs and settings for each run
│   ├── manifest_cli.py
│   └── paths.py                   # Config loader + ${PROJECT_ROOT} expansion
│
├── optimization/              # Beat-grid optimizer and automatic branch selection
│   ├── run_pipeline.py            # Main optimizer sequence: run_cascade(...)
│   ├── optimization_regularize.py # Grid-search optimizer + post-processing
│   └── detect_multimodal.py       # Test for two tapping phases
│
├── scripts/                   # Modules the stages call out to
│   ├── repp_gt_alignment_pipeline.py   # REPP audio-onset alignment
│   ├── build_tap_release.py       # Builds data/taps/ from the raw session exports
│   ├── kde_peaks_from_taps.py     # Stage 2 from released taps (verifies the released peaks)
│   ├── build_validation_listening_stims.py  # Click-track DSP
│   ├── run_beat_tracker.py        # Madmom DBNBeatTracker wrapper
│   └── run_beat_this.py           # Beat This! transformer wrapper
│
├── config/                    # Dataset configs (fixed for the paper)
│   ├── canonical240.json          # 240-excerpt validation set (paper §3.1-§3.3, §3.6)
│   ├── gt_source_map_canonical240.json
│   └── globalmood200.json         # GlobalTap cross-cultural benchmark (§3.4-§3.5)
│
├── data/                      # Annotations, tap data, per-stem reproduction inputs
│   ├── annotations/                               # Consensus beat annotations, one file per excerpt
│   │   ├── globalmood200/*.beats                  # 200 benchmark excerpts
│   │   └── canonical240/*.beats                   # 240 validation excerpts (Crowd condition)
│   ├── taps/                                      # De-identified taps, one row per tap (see taps/README.md)
│   │   └── globalmood200_{taps,trials}.csv.gz     # Benchmark: 200 stems, 746 participants
│   ├── per_stem_metrics.csv                       # 240 stems, paper §3.2-§3.3 (rating, ISTC, CV)
│   ├── country_panels.csv                         # Tapping-panel + source-country list (Fig 1)
│   ├── example_stem_panels.npz                    # Pre-computed inputs for the data-panels figure
│   ├── stimuli_index_canonical240.csv             # Per-stem audio source pointers
│   ├── stimuli_index_globalmood200.csv            # Per-stem YouTube IDs (GlobalMood)
│   ├── stage_inputs/                              # Stage-1/2/3 outputs from the validation runs
│   │   ├── canonical240/{cleaned_kde_peaks.csv,algorithm_beats.csv,beat_this_algorithm_beats.csv}
│   │   └── globalmood200/{cleaned_kde_peaks.csv,algorithm_beats.csv,beat_this_algorithm_beats.csv}
│   └── SOURCES.md                                 # Where to download the source audio + annotations
│
├── figures/                   # Paper figure-generation
│   ├── pipeline_schematic.py        # Fig 1A: pipeline schematic
│   ├── globaltap_map.py             # Fig 1B: 59-country world map
│   ├── data_panels.py               # Fig 2: spectrogram → optimized beat grid
│   ├── validation_comparison.py     # Per-corpus rating boxplots
│   ├── render_playhead_videos.py    # Optional local-only case-study renderer
│   └── fig_utils.py
│
├── supplementary_analyses/    # Supplementary analyses
│   ├── cross_cultural/             # Recruitment-panel period/phase result (§3.5)
│   ├── istc_cv_tracker_correlations/  # Pooled ISTC/CV-vs-tracker-agreement correlations
│   ├── convergence/                # Tapper-count convergence curves
│   ├── beat_metrics/               # Beat metrics beyond F-measure
│   ├── metrical_preference/        # Complete 180-excerpt rating analysis
│   ├── parameter_sensitivity/      # Tests of optimizer settings on 240 excerpts
│   └── METHOD_PARAMETERS.md        # Fixed settings and how they were chosen
│
├── experiments/               # PsyNet experiment templates
│   ├── tapping/                    # Online beat-tapping experiment (REPP)
│   └── listening/                  # Click-overlay listener-rating experiment
│
└── audio_demos/               # 8 case-study playhead videos + click-overlay demos
    └── README.md
```

---

## 2. Quick start

```bash
git clone https://github.com/williambotticelli-wells/GlobalTap.git
cd GlobalTap

# Madmom requires CPython 3.10 or 3.11 (it does not build on 3.12+).
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The simplest entry point is **§4 (Reproducing the paper figures)**: the
figure scripts read only from `data/` and from the included
`supplementary_analyses/` result tables, so neither raw audio nor tapping
CSVs are needed to regenerate the figures.

The next-fastest path is to confirm the consensus beats themselves.
The repository includes Stage-2/3 outputs from the validation runs in
`data/stage_inputs/`, so

```bash
python pipeline/4_run_crowd_gs.py --config config/canonical240.json
python pipeline/4_run_crowd_gs.py --config config/globalmood200.json
```

each run Stage 4 **end-to-end from a fresh clone** (~20--25 min single-core),
with no `PROJECT_ROOT` and no audio. When the full data tree is unavailable,
Stage 4 uses the included files in `data/stage_inputs/`. This reproduces the
`cv_pipeline`
values in `data/per_stem_metrics.csv` for the full 240-stem set (all 100
mixed-corpora, 80 Ballroom, 20 MIREX, and 40 supplementary GlobalMood-40
stems) and the GlobalTap benchmark's tracker-agreement F-measures reported
in Results §3.4. See `data/stage_inputs/README.md` for details.

Stage 2 can also be rebuilt from the released taps, again without audio:

```bash
python scripts/kde_peaks_from_taps.py --verify
```

This rebuilds the included Stage-2 results and matches them excerpt for excerpt
within the 5 ms analysis grid, so the consensus annotations can be traced back
to individual taps. See `data/taps/README.md`.

To re-run the pipeline end-to-end from raw inputs (Stages 1–7),
point `PROJECT_ROOT` at a tree that contains the source audio, tapping
CSVs, and reference annotation packs (see `data/SOURCES.md`):

```bash
export PROJECT_ROOT=/path/to/your/data/tree
./run_pipeline.sh config/canonical240.json     # Validation experiment (paper §3.1-§3.3, §3.6)
./run_pipeline.sh config/globalmood200.json    # Cross-cultural benchmark (paper §3.4-§3.5)
```

Pipeline outputs land in `${PROJECT_ROOT}/outputs/<dataset>/`. Every
stage writes a `manifest.json` with input checksums, parameter dump,
and timing.

---

## 3. Pipeline architecture

The config files record the main alignment and optimizer settings. Stage 4
reads its window and KDE settings from the config. Several other optimizer
values come from matching Python constants or function defaults, including
some values also written in the JSON for documentation. The values documented
here match the code and were used for every reported analysis. In a
sensitivity test, each of seven settings was changed one at a time across all
240 validation excerpts. The resulting grids matched
the paper's grids at F=.947--1.000, and the metrical level stayed the same in
92.1--100% of excerpts. See
`supplementary_analyses/METHOD_PARAMETERS.md` for the values tested and the
results.

| Stage | Script                          | Inputs                              | Outputs                                      |
|-------|---------------------------------|-------------------------------------|----------------------------------------------|
| 0     | `0_build_gt_source_map.py`      | Stimuli manifest                    | `config/gt_source_map_<dataset>.json`        |
| 1     | `1_merge_exports.py`            | Tap CSVs                            | `merged/TapTrialMusic_EXPORT_COMPAT.csv`     |
| 2     | `2_repp_align.py`               | Merged CSV + audio                  | `repp/cleaned_kde_peaks.csv`                 |
| 3     | `3_run_madmom.py`               | Audio                               | `madmom/algorithm_beats.csv`                 |
| 3b    | `3_run_beat_this.py`            | Audio                               | `beat_this/algorithm_beats.csv`              |
| 4     | `4_run_crowd_gs.py`             | KDE peaks                           | `crowd_gs/{stem}_crowd_gs.txt`               |
| 5     | `5_compare_variants.py`         | Stage 4 + Stage 0/3/3b references   | `compare/variant_metrics_per_stem.csv`       |
| 6     | `6_period_audit.py`             | Stage 0 + Stage 4                   | `compare/period_audit.csv`                   |
| 7     | `7_render_clicks.py`            | Stages 0, 4, 3 + audio              | `clicks/{cond}/{stem}__{cond}.wav`           |

Stages 3 and 3b run independently and only feed Stage 5 (F-measure
baselines) and Stage 7 (optional click-track stimuli). Neither
algorithm participates in Stage 4 — the paper's `crowd_gs` consensus
is computed from the human tap KDE alone. Stage 3b is opt-in in the driver
because Beat This! is a separate install: name it in the stage list, as in
`./run_pipeline.sh config/canonical240.json 1,2,3,3b,4,5,6,7`.

`scripts/compute_istc_per_stem.py` computes the paper's other reference-free
measure, ISTC, directly from raw taps rather than from any pipeline stage's
output. It recovers the released `istc` column to within 0.2 points on the
10-90 scale (see `data/SOURCES.md` for the verification).

For each candidate period and phase, the optimizer scores the beat grid as
**S(T, φ) = D̄(b) + 0.5 · C(b, p)**, where D̄ is mean KDE density at the
beats and C is peak coverage. The optimizer flags possible phase bimodality when
the largest gap between sorted sub-period inter-peak intervals is at least
0.10T. The scoring and detection code is in
`optimization/optimization_regularize.py`,
`optimization/detect_multimodal.py`, and `optimization/run_pipeline.py`. See
`supplementary_analyses/METHOD_PARAMETERS.md` for the parameter values.

### `clip_offset_s` in `gt_source_map_*.json`

Source corpora time their beats against the full-length track, but
the GlobalTap pipeline operates on 30 s clips. `clip_offset_s` gives the
clip's start time in seconds. `lib/gt_loader.py` subtracts it so all
beat times are measured from the start of the 30 s excerpt
(`t_clip = t_annotator − clip_offset_s`). Each entry also carries a
`row_filter` (e.g.
`"pos>=1"` to drop unlabeled annotator rows) and a `source_label`
identifying the collection of reference beat files supplied by the source
corpus.

---

## 4. Reproducing the paper figures

The `data/` directory includes the per-stem values that
the paper cites, so the figure scripts can be run **without the raw
audio or tapping CSVs**:

```bash
python figures/pipeline_schematic.py     # Fig 1A — schematic
python figures/globaltap_map.py          # Fig 1B — 59-country world map
python figures/data_panels.py            # Fig 2  — spectrogram → optimized beats
python figures/validation_comparison.py  # Supplementary §1 boxplots

# Fig 3 — country tactus choice
python supplementary_analyses/cross_cultural/plot_country_tactus_choice_maintext.py
```

The Fig. 3 script reads the panel-test table it sits next to
(`supplementary_analyses/cross_cultural/panel_tests/country_tactus_choice_proportions_with_ci.csv`).
The other scripts read `data/`.

To rebuild `data/per_stem_metrics.csv` from raw inputs (and reproduce
the click-track WAVs used in the listening experiment), run the full
pipeline (stages 1–7) over `${PROJECT_ROOT}` as shown in §2. That path is
the 240-excerpt validation set: stages 2, 5, 6 and 7 need the reference
annotation map, and the stimulus manifests named by the configs
(`data/sampled_stimuli_canonical240.json`,
`data/sampled_stimuli_globalmood200.json`) live in the raw-data tree rather
than in this repository. The GlobalTap benchmark has no reference
annotations, so it runs stages 1, 3 and 4 only. Its Stage-2 KDE peaks are
rebuilt from the released taps with `scripts/kde_peaks_from_taps.py`. Audio for
the 200 GlobalMood excerpts is fetched from YouTube (one URL per
`stem_id`). Audio and reference annotations for the 240-excerpt
validation set come from publicly distributed corpus packs. See
[`data/SOURCES.md`](data/SOURCES.md) for one-stop download
instructions and per-stem source pointers.

---

## 5. Case-study descriptions

The supplementary HTML identifies eight case-study excerpts spanning phase
offsets, metrical alternatives, bimodal pooled tapping, and high-consensus
baselines. `audio_demos/` ships pre-rendered playhead videos and click-overlay
audio for each case (see `audio_demos/README.md`). Re-rendering instructions
are also included for anyone regenerating them from source.

---

## 6. Software dependencies

Python 3.10+. Key libraries: `numpy`, `pandas`, `scipy`, `matplotlib`,
`librosa`, `madmom`. See `requirements.txt` for pinned versions.
Madmom requires CPython ≤ 3.11.

The pipeline is deterministic given the config and pinned dependencies.
Rerunning it on the same inputs produces the same beat sequences and
per-stem metrics.

---

## 7. License & data release

- **Pipeline source code**: MIT (see `LICENSE`).
- **Beat annotations and per-stem CSVs** (`data/`): CC BY 4.0.
- **Audio**: GlobalTap benchmark source recordings are not included. The
  benchmark links original sources for authorized full-audio access. Eight
  short case-study demonstrations (`audio_demos/`) from validation corpora
  are included.

---

## 8. AI assistance

Large-language-model tools helped with the sensitivity analyses used to select
the fixed pipeline settings. During revision, they also helped design and check
supplementary analyses and edit this documentation. The authors selected the
final settings, verified all code and outputs, and remain responsible for this
repository. See the paper's AI Usage Statement and
`supplementary_analyses/METHOD_PARAMETERS.md`.
