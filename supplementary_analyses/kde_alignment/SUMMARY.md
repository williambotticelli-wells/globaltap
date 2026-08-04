# Pre-grid KDE peak alignment

Reported in Results §3.1. For each excerpt, `pct_peaks_within_70ms_of_gt` is
the percentage of raw pooled-KDE peaks that fall within 70 ms of a reference
beat, before any grid fitting or post-processing. The paper reports the mean
of that percentage across excerpts, so the values are shares of peaks and not
shares of excerpts.

| Corpus group | n | Mean % of peaks within 70 ms | Mean abs. peak-to-reference offset (ms) |
|---|---:|---:|---:|
| Ballroom | 80 | 88.3 | 30.0 |
| Mixed corpora | 100 | 47.5 | 114.9 |
| MIREX | 20 | 53.2 | 132.5 |
| All three | 200 | 64.4 | 82.7 |

Agreement is lowest on the mixed corpora, driven by the rubato-dominated
(ASAP) and densely syncopated (Candombe) excerpts.
