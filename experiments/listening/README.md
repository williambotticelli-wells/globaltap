# Listener-rating experiment template

Online click-overlay listener-rating experiment built on
[PsyNet](https://www.psynet.dev/). This is the generalized template
used to collect the listener ratings reported in §3.2 of the paper.

## How a session looks

1. Participant is recruited (Prolific / Hot Air / Cint).
2. Audio-volume calibration.
3. For each music excerpt, two click-overlay versions
   (`Crowd` vs `Reference`) are presented as a paired block in a
   randomised order. The participant rates each on a 1–7 scale with
   the prompt "How well do the clicks line up with the beat of the
   music?".
4. Stem blocks are randomised per participant.

Each stimulus is a 10–25 s window with click-track overlays at
1100 Hz, 12 ms Hann clicks, peak-normalized and summed at
music 0.35 / click 0.65.

## Stimuli

Click-overlay WAVs are produced by Stage 7 of the analysis pipeline
(`pipeline/7_render_clicks.py`) — one WAV per
(`stem`, `condition`) pair. Drop them into
`static/audio_stimuli/<condition>/` (not included) and update
the per-stem manifest in `experiment.py` (`STIMULI_MANIFEST` constant)
to point at the filenames.

## Running

See the [PsyNet documentation](https://www.psynet.dev/) for local testing,
build, and deployment instructions.

## What the experiment produces

Every PsyNet deployment writes a per-deployment export folder
containing `Participant.csv`, `Response.csv`, and an
`AbsoluteSliderRating.csv`-style file with listener ratings for each stem and
condition. Use those ratings as the input to the
per-corpus mean-rating tables in §3.2 of the paper.
