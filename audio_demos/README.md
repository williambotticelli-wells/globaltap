# Case-study playhead videos + audio demos (paper §3.6)

Eight pre-rendered Figure-2-layout playhead videos and the matching
click-overlay listening clips. Each excerpt is rendered twice (Crowd
consensus vs Reference annotation) using the same DSP across
conditions (only the beat-time vector differs). All clips and videos
cover the [10–25 s] listener window. Click tracks: 1100 Hz, 12 ms
Hann. The music and click signals are each peak-normalized before
being summed at music 0.35 / click 0.65.

The eight stems span three *structural-divergence* cases (paper §3.6), one
high-consensus contrast case, and four *simple-alignment* cases (KDE peak tips
track the optimizer grid throughout the song):

| Case label                  | Stem                       | Set                  | Phenomenon                                                          |
|-----------------------------|----------------------------|----------------------|---------------------------------------------------------------------|
| `candombe_phase_offset`     | `proyecto.1992_lobo_06`    | structural divergence | Phase offset (~140 ms), consistent with a perceived-onset shift (Morton 1976; Danielsen 2019). |
| `ballroom_halftime`         | `Albums-Chrisanne3-09`     | structural divergence | Tactus mismatch — Viennese Waltz: crowd locks to half-rate felt pulse. |
| `mixed_bimodal`             | `Media-105914`             | structural divergence | Bimodal pooled tapping — secondary KDE peak family at ≥ 40 % strength. |
| `ballroom_baseline`         | `Albums-AnaBelen_Veneo-13` | high-consensus contrast | High-consensus baseline with CV ≈ 0.024 and closely aligned Crowd and Reference beats. |
| `gtzan_disco_simple`        | `gtzan_disco_00000`        | simple alignment      | GTZAN Disco — KDE peaks coincide with the optimizer grid throughout. |
| `tapcorrect_pop_simple`     | `054_youtube_UgL4N4Hf1dk`  | simple alignment      | Pop (TapCorrect) — stable 4/4 with crowd-corrected reference annotation. |
| `ballroom_latin_simple`     | `Albums-Latin_Jam-04`      | simple alignment      | Latin Ballroom — strong dance-floor isochrony.                      |
| `cretan_simple`             | `cretan1`                  | simple alignment      | Cretan — high consensus on a non-Western corpus.                    |

Each case ships:

- `<case-label>__playhead.mp4`  — Figure-2 layout with a moving playhead,
  followed by Crowd and Reference click overlays (~30 s total).
- `<case-label>__crowd_gs.mp3`  — 15-s click overlay using the
  optimized crowd consensus beat grid.
- `<case-label>__gt.mp3`        — 15-s click overlay using the
  reference annotation.

These case studies are discussed in paper §3.6. The included
`data/per_stem_metrics.csv` (CV, ISTC, crowd vs. reference ratings)
identifies which excerpts each phenomenon corresponds to.

## Re-render from raw audio

```
./run_pipeline.sh config/canonical240.json    # stages 1-7
# Stage 7 writes WAVs to outputs/canonical240/clicks/<cond>/<stem>__<cond>.wav

# Convert to MP3 (matching the filenames above):
ffmpeg -y -i outputs/canonical240/clicks/<cond>/<stem>__<cond>.wav \
       -ac 1 -ar 22050 -b:a 96k \
       audio_demos/<case-label>__<cond>.mp3

# Rebuild the playhead videos (figure 2 layout + moving playhead +
# concatenated Crowd→Reference audio):
GLOBALTAP_DATA_ROOT=/path/to/data python figures/render_playhead_videos.py
```

The two **listening conditions** rated in the paper's validation
experiment (§3.2) are `{crowd_gs, gt}` — the crowd-derived pipeline
output and the reference annotation. The `madmom` and
`beat_this` baselines enter the paper only as F-measure references
against the pooled crowd consensus (§3.4). They are not part of the
listener-rating study.
