# De-identified tap data

Every tap by every participant on the 200 GlobalTap benchmark excerpts, in music
time. These are the onsets the consensus pipeline consumes: REPP-aligned,
expressed in seconds from the first audio sample of the 30 s excerpt, and
restricted to the excerpt window.

```
globalmood200_taps.csv.gz     200 excerpts, 746 participants, 1,201,342 taps
globalmood200_trials.csv.gz   one row per trial (23,493) with the recruitment panel
```

| Column | Description |
|---|---|
| `participant` | Pseudonymous label (`gt-####`), stable across the tap and trial tables |
| `stem_id` | Excerpt identifier, joins to `data/stimuli_index_globalmood200.csv` |
| `tap_time_s` | Tap onset in music time, seconds, rounded to the microsecond |
| `corpus` | `globalmood_200` |
| `country_panel` | ISO-2 recruitment panel |
| `source_batch` | Collection batch |
| `n_taps` | Taps in that trial |

Nothing that could identify a participant is included: no platform or session
identifiers, no timestamps, no demographics, no free text. Demographic
summaries are reported at panel level in
[`supplementary.html`](../../supplementary.html) §3.

Only the benchmark taps are released. For the validation set, the repository
provides the Stage-2 peaks in
[`../stage_inputs/canonical240/`](../stage_inputs/canonical240/) and the
consensus annotations in
[`../annotations/canonical240/`](../annotations/canonical240/), which are the
inputs and beat grids used to render the click-track rating stimuli.

## Which trials are here

The tables contain the usable trials behind the counts reported in the paper: a
trial is kept when the platform did not mark it failed and both the participant
and the excerpt are identified. That is 23,493 trials from 746 participants,
114–120 tappers per excerpt (median 117), with per-panel participant counts in
`data/country_panels.csv`.

The paper's collection total of 2,205,932 parsed taps counts every onset REPP
recovered across both studies, including trials that failed quality checks and
onsets outside the 30 s window. Those are not part of the release.

## Rebuilding the consensus from taps

`scripts/kde_peaks_from_taps.py` recomputes Stage 2 from these tables using the
Stage-2 code path and the fixed settings in
[`METHOD_PARAMETERS.md`](../../supplementary_analyses/METHOD_PARAMETERS.md), so
the benchmark can be rebuilt from raw taps in a fresh clone without audio:

```bash
python scripts/kde_peaks_from_taps.py --verify
```

This recreates the included
`data/stage_inputs/globalmood200/cleaned_kde_peaks.csv` file, matching it
excerpt by excerpt within the 5 ms analysis grid. Stage 4 then turns those peaks
into the released consensus annotations
([`../annotations/`](../annotations/)).

## Provenance

`scripts/build_tap_release.py` builds these tables from the raw session
exports, which are not included here. Taps are parsed with
`supplementary_analyses/pipeline_subset_core.parse_analysis_taps`, the same
routine the published analyses use.

Licensed CC BY 4.0, as with the other data in this repository.
