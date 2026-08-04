## Where to obtain the audio and reference annotations

The companion repository does **not** redistribute the music itself.
The 30 s audio excerpts and the beat-reference annotations
are sourced from publicly distributed corpora (or, for GlobalMood,
public YouTube uploads). This document maps every stem in the paper
to its source so anyone can rebuild the audio inputs from scratch.

Stem-level indices:

- `data/stimuli_index_canonical240.csv` — the 240 stems that enter the
  validation experiment (Mixed corpora + Ballroom + MIREX +
  GlobalMood). The `audio_source` field is either an
  `upstream::<corpus>` token (look up the row's corpus below) or a
  direct YouTube URL.
- `data/stimuli_index_globalmood200.csv` — the 200 GlobalMood stems
  in the GlobalTap benchmark. Each `stem_id` is a
  YouTube video ID. `audio_source` resolves to
  `https://www.youtube.com/watch?v=<stem_id>`.

For every stem the pipeline operates on a 30 s clip starting at a
corpus-specific offset. That offset is recorded in
`config/gt_source_map_canonical240.json` (see the `clip_offset_s`
section of the top-level README) so anyone downloading
full-length audio can extract the matching window.

### Reference annotations for the 240-excerpt validation set

| Corpus       | Audio                                          | Reference annotations                                            |
|--------------|------------------------------------------------|------------------------------------------------------------------|
| Ballroom     | ISMIR 2004 Ballroom dance corpus (200 clips).  | Krebs et al. (2013) annotations redistributed in the Beat This! release. |
| Hainsworth   | Hainsworth & Macleod (2003) corpus.            | Beat This! release.                                              |
| RWC          | RWC Pop / Royalty / Classical / Jazz subsets.  | Goto et al. RWC annotations from the `bm30_filt` clip-window collection. |
| Beatles      | Isophonics Beatles corpus.                     | Mauch et al. Isophonics beats (Beat This! release).              |
| Robbie Williams | Di Giorgi et al. Robbie Williams set.       | Beat This! release.                                              |
| Hjerkinn     | Hjerkinn corpus.                               | Beat This! release.                                              |
| Harmonix     | Harmonix Set (Nieto et al. 2019).              | Harmonix Set beats from the Beat This! release.                  |
| Tapcorrect   | TapCorrect / GTZAN-rhythm pack.                | Driedger et al. tap-correct annotations.                         |
| Candombe     | Candombe corpus (Rocamora et al.).             | Candombe beat annotations.                                       |
| GTZAN-rhythm | GTZAN with Marchand & Peeters annotations.     | Marchand & Peeters beats from the Beat This! release.            |
| ASAP         | ASAP solo-piano performances.                  | ASAP score-aligned beats.                                        |
| Carnatic     | CompMusic Carnatic music collection.           | Srinivasamurthy et al. (2014) beats.                             |
| Cretan       | Holzapfel Cretan-leaping-dance set.            | Holzapfel & Stylianou (2011) beats.                              |
| MIREX        | MIREX 2006 beat-tracking set (40 clips).       | Pooled lab-tapper consensus distributed with the MIREX 2006 set. |

`source_label = "beat_this_upstream"` rows in
`config/gt_source_map_canonical240.json` mean the reference beats are
those distributed in the
[Beat This!](https://github.com/CPJKU/beat_this) release. Everything
else lists its own pack name in the `source_label` field.

### Audio for GlobalMood (200-stem benchmark)

Each `stem_id` is the YouTube video ID. To rebuild the 30 s clips:

```bash
yt-dlp -x --audio-format wav -o "audio/%(id)s.%(ext)s" "<stem-url>"
ffmpeg -ss 30 -t 30 -i audio/<stem-id>.wav -ac 1 -ar 44100 \
       audio/globalmood_30s/<stem-id>.wav
```

(Start times for GlobalMood clips are 30 s into each track, matching
the GlobalMood release, see the original GlobalMood paper for the
recruitment-driven excerpt selection.)

### Tapping data

De-identified tap onsets for the 200-excerpt benchmark are in `data/taps/`.
Raw session exports with platform identifiers and per-session metadata are not
included (see the paper's Ethics Statement).

Prolific recruited participants for the eleven benchmark exports and the
listener-rating study. The 40-excerpt GlobalMood validation set also included
three Cint exports, which contributed 58 of its 296 tappers. The batch names
appear in `scripts/compute_istc_per_stem.py`. The paper's Methods and Ethics
Statement mention only Prolific. Both recruitment sources used the same PsyNet
tapping task and audio-marker checks, and their data were combined for the
analysis.

Figures and per-stem metrics regenerate from `data/per_stem_metrics.csv`,
`data/example_stem_panels.npz`, and `data/country_panels.csv`. Stage 4
re-runs from `data/stage_inputs/`. To recompute ISTC or Stages 1–3 from raw
taps, point `PROJECT_ROOT` at a tree that mirrors `raw_tapping_data/` and run
`scripts/compute_istc_per_stem.py` or `./run_pipeline.sh`.

The `n_participants` column is the number of tappers used for each stem's
reference-free measures, not the study's overall participant count. Use the
supplementary participant-flow counts when reporting study participation, and
`n_participants` when interpreting `istc` or `cv_pipeline`.

### License notes

All source corpora retain their original licenses. Outside of
`audio_demos/` (eight short case-study clips, see its `README.md`), this
repository only redistributes derived per-stem **annotation** outputs
(crowd beats, KDE peaks, F-measure tables) and aggregate per-stem metrics.
No other audio is committed.
