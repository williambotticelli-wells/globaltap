# Own-country tapping coherence

Does tapping coherence (ISTC) rise when an excerpt's source country matches
the panel that tapped it? The main paper reports the primary test in §3.5.

- Design: the five source countries with 20 benchmark excerpts each
  (US, KR, FR, MX, EG), giving 100 excerpts analyzed across ten tapping
  panels. In 100 of the 1,000 excerpt-panel combinations, the excerpt's source
  country matched the panel's recruitment country.
- To compare panels using the same number of participants, ISTC for each
  excerpt and panel was averaged across 500 random samples of seven tappers.
  Seven was the largest sample size available for every excerpt and panel.
- Primary model: `istc ~ source_match + C(excerpt) + C(panel)`, standard
  errors clustered by excerpt.

## Primary result

- Match effect **+0.066 ISTC units**, 95% CI [-0.030, +0.161],
  two-sided p=.179 (`balanced_five_fixed_effect_model.json`).
- No reliable own-country advantage once excerpt-level and panel-level
  differences in coherence are accounted for.

## Sensitivities

- All 110 match-eligible excerpts: +0.079, 95% CI [-0.018, +0.176], p=.109.
- All 200 excerpts in the model: +0.083, 95% CI [-0.010, +0.176], p=.080.
  The match coefficient is still identified only by the 110 excerpts whose
  source country has a recruitment panel
  (`expanded_fixed_effect_sensitivities.json`).
- A country-wise standardized-effect meta-analysis over the same five
  panels gives d=+0.211, 95% CI [+0.004, +0.418], one-sided p=.023, with a
  positive adjusted effect in 5 of 5 panels (`meta_balanced_five.json`).
  It pools five country-level estimates
  rather than clustering by excerpt, so it does not carry the same weight as
  the primary model, and with five panels it is not a well-powered test.

Read together, the estimates lean positive but do not establish an
own-country effect. The five remaining panels (AU, BR, JP, PL, ZA) each have
only two source-country excerpts in the benchmark, so they appear in
`country_delta_summary.csv` as descriptives and are not treated as
country-level tests.
