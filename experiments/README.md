# GlobalTap experiment templates

PsyNet experiment templates corresponding to the two
deployments that produced the data in the paper. Both run on
[PsyNet](https://www.psynet.dev/) and ship a Dockerfile plus pinned
`requirements.txt` / `constraints.txt`.

| Sub-directory | What it is                                                       |
|---------------|------------------------------------------------------------------|
| `tapping/`    | REPP-instrumented online beat-tapping experiment.                |
| `listening/`  | Click-overlay listener-rating experiment (Crowd vs Reference).   |

See [PsyNet's documentation](https://www.psynet.dev/) for build and
deploy recipes.

## What is and isn't included

Included:

- The full Python experiment definitions (`experiment.py`, helpers
  like `repp_prescreens.py`, `logic.py`, `custom_config.py`).
- Pinned dependency files and the `Dockerfile`.
- Goldsmiths-MSI item text in `gmsi.py` is referenced from the
  experiment but only redistributed in the PsyNet
  installation. Cite Müllensiefen et al. (2014) when reusing.

Not included (intentionally):

- `static/`, `sounds/`, `audio/` — music stimuli. See
  `../data/SOURCES.md` for download recipes.
- `data/`, `exports/` — raw session exports with platform identifiers and
  per-session metadata are not included (see the paper's Ethics Statement).
  The de-identified benchmark taps are in `../data/taps/`, and the Stage-2/3
  outputs behind the released consensus annotations are in
  `../data/stage_inputs/<dataset>/`.
- `.venv/` — Python virtual environment. Recreate from
  `requirements.txt`.

## Data flow back into the analysis pipeline

After a deployment, PsyNet writes a per-deployment `data/` folder.
To feed those into the paper's analysis pipeline:

1. Convert and merge the files into the
   `TapTrialMusic_EXPORT_COMPAT.csv` schema
   (see `../pipeline/1_merge_exports.py`).
2. Place the merged CSV at
   `${PROJECT_ROOT}/outputs/<dataset>/merged/TapTrialMusic_EXPORT_COMPAT.csv`.
3. Run Stages 2–7 as documented in the top-level `README.md`.
