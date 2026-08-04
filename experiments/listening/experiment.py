# pylint: disable=missing-class-docstring,missing-function-docstring
"""
Unified validation 2026-04-26.

Listener study covering all 240 GlobalTap benchmark stems, 4 corpus groups:
    - Ballroom (40 + 40 heldout = 80)         crowd + ref
    - Mixed-corpora (asap, beatles, candombe, carnatic, cretan, groove_midi,
      gtzan, guitarset, hainsworth, harmonix, rwc, tapcorrect, turkish = 100)
                                               crowd + ref
    - MIREX (20)                               crowd + ref (ref = 40 lab annotators
                                               through canonical pipeline +
                                               per-trial MAD, identical algorithm
                                               to Crowd 240)
    - GlobalMood (40)                          crowd ONLY (no expert annotations)

Pipeline behind the stimuli (parity_rerun_2026_04_26 from
canonical240_parity_2026_04_26.json):
    KDE bw=80 ms, peak prom=0.1, dist=0.15 s, IOI=[0.28, 1.25] s,
    lambdas=[2.0, 0.3, 0.3], grid-search optimizer, MAD=3.5 per-trial.

Click rendering (clicks_lufs/):
    1. LUFS-normalize music excerpt to -23 LUFS (EBU R128 / ITU-R BS.1770)
    2. Mix with canonical click DSP: 1100 Hz / 12 ms / amp 0.9 / 35-65 ratio.
    3. Safety peak limit at -0.1 dBFS (never engaged in practice).

Manifest: 440 nodes = 240 crowd + 200 ref.

Per-participant: 50 trials, balanced StaticTrialMaker. Each block = stem;
max 2 trials/block (crowd + ref of same stem heard adjacent). Participants
get ~25-30 stem blocks. Initial recruitment 100 → ~11.4 ratings per (stem,
condition) node when fully filled.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import psynet.translation.utils as _psynet_translation_utils


def _get_pot_from_command_subprocess(cmd, tmp_pot_file, sp):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    if result.stdout and sp:
        lines = result.stdout.strip().split("\n")
        if lines:
            sp.text = lines[-1]
    if result.returncode != 0:
        sys.exit(result.returncode)
    if os.path.exists(tmp_pot_file):
        pot = _psynet_translation_utils.load_po(tmp_pot_file)
        os.remove(tmp_pot_file)
        return list(pot)
    return []


_psynet_translation_utils.get_pot_from_command = _get_pot_from_command_subprocess

import pandas as pd
import psynet.experiment
from markupsafe import Markup
from psynet.asset import asset
from psynet.consent import MainConsent
from psynet.demography.general import BasicDemography, BasicMusic
from psynet.modular_page import AudioPrompt, ModularPage, Prompt, SurveyJSControl
from psynet.page import InfoPage
from psynet.timeline import Event, FailedValidation, Timeline
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker
from psynet.utils import get_logger, get_translator

from .gmsi import GMSI

logger = get_logger()
_ = get_translator()

LOCALE = "en"
MANIFEST_PATH = Path("manifest.csv")

ATTENTION_CHECK_FIELD = "attention_task_understanding"
ATTENTION_CORRECT_VALUE = "clicks_on_piece"

# Per-participant trial budget: 50 rating trials.
# In debug mode (UNIFIED_VAL_DEBUG_BLOCKS=N, N>0) the budget shrinks to 2*N
# so the trial maker terminates after the small subset of nodes is exhausted.
def _trial_budget_per_participant() -> int:
    raw = os.environ.get("UNIFIED_VAL_DEBUG_BLOCKS", "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                # 2 trials per block, except GlobalMood blocks which contribute 1.
                # Cap at 2*N to be safe and let max_trials_per_block limit per-block.
                return max(2, 2 * n)
        except ValueError:
            pass
    return 50


MAX_TRIALS_PER_PARTICIPANT = _trial_budget_per_participant()
EXPECTED_TRIALS_PER_PARTICIPANT = MAX_TRIALS_PER_PARTICIPANT

# Each block = stem; max 2 trials/block (crowd+ref of same stem heard adjacent).
# GlobalMood blocks are 1-trial (crowd only).
MAX_CONDITIONS_PER_STEM = 2


class VersionRatingPage(ModularPage):
    """Play one click track and collect a 1-7 beat-alignment rating."""

    def __init__(
        self,
        stem: str,
        condition: str,
        corpus: str,
        corpus_group: str,
        audio_asset,
        time_estimate: float = 22,
    ):
        self.stem = stem
        self.condition = condition
        self.corpus = corpus
        self.corpus_group = corpus_group

        prompt_text = Markup(
            "<div style='max-width:720px;margin:0 auto;text-align:left'>"
            "<p>"
            + _(
                "Listen to the click track overlaid on the music. "
                "When you are ready, rate how well the clicks line up with "
                "the <strong>beat of the music</strong>."
            )
            + "</p></div>"
        )

        survey_spec = {
            "elements": [
                {
                    "type": "rating",
                    "name": "beat_alignment",
                    "title": _("How well did the clicks match the beat of the music?"),
                    "rateMin": 1,
                    "rateMax": 7,
                    "minRateDescription": _("Poor"),
                    "maxRateDescription": _("Excellent"),
                    "isRequired": True,
                },
            ]
        }

        bot_response = {"beat_alignment": 4}

        super().__init__(
            label=f"beatval_{stem}_{condition}",
            prompt=AudioPrompt(audio_asset, text=prompt_text, controls=False),
            control=SurveyJSControl(survey_spec, bot_response=bot_response),
            events={
                "hideNextButton": Event(
                    is_triggered_by="promptStart",
                    delay=0.0,
                    js="document.getElementById('next-button').style.visibility='hidden'",
                ),
                "showNextButton": Event(
                    is_triggered_by="promptEnd",
                    js="document.getElementById('next-button').style.visibility='visible'",
                ),
            },
            time_estimate=time_estimate,
        )

    def format_answer(self, raw_answer, **kwargs):
        parsed = raw_answer
        if isinstance(raw_answer, str):
            try:
                parsed = json.loads(raw_answer)
            except json.JSONDecodeError:
                return {"_error": True, "raw": raw_answer}
        if not isinstance(parsed, dict):
            return {"_error": True, "raw": raw_answer}
        out = dict(parsed)
        out["_stem"] = self.stem
        out["_condition"] = self.condition
        out["_corpus"] = self.corpus
        out["_corpus_group"] = self.corpus_group
        return out

    def validate(self, response, **kwargs):
        ans = response.answer
        if isinstance(ans, dict) and ans.get("_error"):
            return FailedValidation(_("Please answer the required question."))
        if isinstance(ans, dict) and not ans.get("beat_alignment"):
            return FailedValidation(_("Please rate the beat alignment before continuing."))
        return None


class VersionRatingTrial(StaticTrial):
    time_estimate = 22

    def show_trial(self, experiment, participant):
        return VersionRatingPage(
            stem=self.definition["stem"],
            condition=self.definition["condition"],
            corpus=str(self.definition.get("corpus", "")),
            corpus_group=str(self.definition.get("corpus_group", "")),
            audio_asset=self.assets["audio"],
        )


class AttentionCheckPage(ModularPage):
    def __init__(self):
        intro_html = (
            "<p><strong>"
            + _("Attention check")
            + "</strong></p><p>"
            + _(
                "Almost done. Please answer one question about the task you just did. "
                "If you are unsure, choose the option that best matches what you heard."
            )
            + "</p>"
        )
        ac_spec = {
            "elements": [
                {
                    "type": "radiogroup",
                    "name": ATTENTION_CHECK_FIELD,
                    "title": _(
                        "In the trials you completed, what was added on top of the music "
                        "in each version?"
                    ),
                    "choices": [
                        {
                            "value": ATTENTION_CORRECT_VALUE,
                            "text": _("Short click sounds overlaid on the piece"),
                        },
                        {"value": "drum_loop", "text": _("A continuous drum loop")},
                        {
                            "value": "nothing",
                            "text": _("Nothing — the music was exactly as in the original recording"),
                        },
                        {
                            "value": "voice_count",
                            "text": _('Someone counting out loud (e.g. "1, 2, 3 …")'),
                        },
                    ],
                    "isRequired": True,
                },
            ]
        }
        super().__init__(
            label="attention_check_listening",
            prompt=Prompt(Markup(intro_html)),
            control=SurveyJSControl(
                ac_spec,
                bot_response={ATTENTION_CHECK_FIELD: ATTENTION_CORRECT_VALUE},
            ),
            time_estimate=25,
        )

    def format_answer(self, raw_answer, **kwargs):
        parsed = raw_answer
        if isinstance(raw_answer, str):
            try:
                parsed = json.loads(raw_answer)
            except json.JSONDecodeError:
                return {"_error": True, "raw": raw_answer}
        if not isinstance(parsed, dict):
            return {"_error": True, "raw": raw_answer}
        choice = parsed.get(ATTENTION_CHECK_FIELD)
        if choice is None or (isinstance(choice, str) and not choice.strip()):
            return {"_error": True, "missing": [ATTENTION_CHECK_FIELD], "raw": parsed}
        out = dict(parsed)
        out["_attention_passed"] = choice == ATTENTION_CORRECT_VALUE
        return out

    def validate(self, response, **kwargs):
        ans = response.answer
        if not isinstance(ans, dict):
            return FailedValidation(_("Please select an answer to continue."))
        if ans.get("_error"):
            return FailedValidation(_("Please select an answer to continue."))
        # Read the choice directly so this also works in bot mode, where the
        # SurveyJSControl bypasses format_answer and the augmented
        # `_attention_passed` flag isn't set.
        choice = ans.get(ATTENTION_CHECK_FIELD)
        if choice is None or (isinstance(choice, str) and not choice.strip()):
            return FailedValidation(_("Please select an answer to continue."))
        if choice != ATTENTION_CORRECT_VALUE:
            return FailedValidation(
                _(
                    "That answer does not match the instructions for this study. "
                    "Please read the question and choices again carefully."
                )
            )
        return None


class FinalFeedbackPage(ModularPage):
    def __init__(self):
        intro_html = Markup(
            "<div style='max-width:760px;margin:0 auto;text-align:left'>"
            "<p><strong>" + _("Final feedback") + "</strong></p>"
            "<p>"
            + _(
                "Thank you for completing the listening trials. "
                "Before you finish, we would like to know a bit about "
                "how you approached the task."
            )
            + "</p></div>"
        )
        spec = {
            "elements": [
                {
                    "type": "comment",
                    "name": "strategy_feedback",
                    "title": _(
                        "What strategies or criteria did you use to evaluate "
                        "how well the clicks matched the beat of the music? "
                        "For example, did you focus on timing, rhythm, "
                        "whether clicks felt early or late, density of clicks, "
                        "or anything else?"
                    ),
                    "rows": 4,
                    "isRequired": False,
                },
                {
                    "type": "comment",
                    "name": "general_comments",
                    "title": _("Any other comments about the experiment (optional)."),
                    "rows": 2,
                    "isRequired": False,
                },
            ]
        }
        super().__init__(
            label="final_feedback",
            prompt=Prompt(intro_html),
            control=SurveyJSControl(
                spec,
                bot_response={"strategy_feedback": "", "general_comments": ""},
            ),
            time_estimate=60,
        )

    def format_answer(self, raw_answer, **kwargs):
        parsed = raw_answer
        if isinstance(raw_answer, str):
            try:
                parsed = json.loads(raw_answer)
            except json.JSONDecodeError:
                return {"raw": raw_answer}
        if not isinstance(parsed, dict):
            return {"raw": raw_answer}
        return parsed


class BalancedUnifiedTrialMaker(StaticTrialMaker):
    """Randomize stem block order; balance_across_nodes=True ensures roughly
    equal ratings/node across the 440 nodes."""

    def choose_block_order(self, experiment, participant, blocks):
        block_ids = sorted(blocks) if isinstance(blocks, (dict, set)) else list(blocks)
        return random.sample(block_ids, len(block_ids))


def _load_manifest() -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    return pd.read_csv(root / MANIFEST_PATH)


def _maybe_apply_debug_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    """If env var UNIFIED_VAL_DEBUG_BLOCKS is set to a positive int N, return
    only N stem blocks (a stratified mini-set across all corpus groups for
    realistic QA). Used for `psynet debug local` runs that exercise the full
    consent / instructions / attention check / demographics / GMSI flow on a
    small number of trials."""
    raw = os.environ.get("UNIFIED_VAL_DEBUG_BLOCKS", "").strip()
    if not raw:
        return manifest
    try:
        n_blocks = int(raw)
    except ValueError:
        logger.warning("UNIFIED_VAL_DEBUG_BLOCKS=%r is not an int; ignoring.", raw)
        return manifest
    if n_blocks <= 0 or n_blocks >= manifest["stimulus_id"].nunique():
        return manifest

    groups = ["mirex", "ballroom", "mixed", "globalmood"]
    picks: list[str] = []
    for i, g in enumerate(groups):
        n_pick = (n_blocks // len(groups)) + (1 if i < n_blocks % len(groups) else 0)
        if n_pick == 0:
            continue
        gstems = sorted(manifest.loc[manifest["corpus_group"] == g, "stimulus_id"].unique())
        picks.extend(gstems[:n_pick])
    sub = manifest[manifest["stimulus_id"].isin(picks)].copy()
    logger.info(
        "UNIFIED_VAL_DEBUG_BLOCKS=%d  → %d stems  %d nodes  groups=%s",
        n_blocks, sub["stimulus_id"].nunique(), len(sub),
        sub["corpus_group"].value_counts().to_dict(),
    )
    return sub


def get_validation_nodes():
    """One StaticNode per manifest row (stem x condition).
    240 crowd + 200 ref = 440 nodes (or fewer if UNIFIED_VAL_DEBUG_BLOCKS is set)."""
    root = Path(__file__).resolve().parent
    manifest = _load_manifest()
    manifest = _maybe_apply_debug_subset(manifest)
    audio_dir = root / "data" / "validation_audio"

    required = {"stimulus_id", "condition", "corpus", "corpus_group", "audio_filename"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    nodes: list[StaticNode] = []
    for _, row in manifest.iterrows():
        stem = str(row["stimulus_id"]).strip()
        cond = str(row["condition"]).strip()
        corpus = str(row["corpus"]).strip()
        cgroup = str(row["corpus_group"]).strip()
        wav_filename = str(row["audio_filename"]).strip()
        wav = audio_dir / wav_filename
        if not wav.exists():
            raise FileNotFoundError(
                f"Missing WAV: {wav}\n"
                "Run scripts/build_manifest_and_audio.py to symlink stimulus audio."
            )
        nodes.append(
            StaticNode(
                block=stem,
                definition={
                    "stem": stem,
                    "condition": cond,
                    "corpus": corpus,
                    "corpus_group": cgroup,
                },
                assets={"audio": asset(wav, extension=".wav", cache=True)},
            )
        )

    n_crowd = sum(1 for n in nodes if n.definition["condition"] == "crowd")
    n_ref = sum(1 for n in nodes if n.definition["condition"] == "ref")
    n_blocks = manifest["stimulus_id"].nunique()
    logger.info(
        "Loaded %d nodes (crowd=%d ref=%d) across %d stems",
        len(nodes), n_crowd, n_ref, n_blocks,
    )
    return nodes


def get_timeline():
    return Timeline(
        MainConsent(),
        InfoPage(
            Markup(
                "<h3>" + _("Music and click tracks") + "</h3>"
                "<div style='max-width:760px;margin:0 auto;text-align:left'>"
                "<p>"
                + _(
                "You will hear short music excerpts, each with a click track "
                "overlaid on top. Your task is to rate how well the clicks line up "
                "with the <strong>beat of the music</strong> &mdash; that is, the steady "
                "pulse you would naturally tap your foot or nod your head to."
                )
                + "</p>"
                "<p>"
                + _(
                "For most excerpts you will hear <strong>two</strong> versions of the click "
                "track, one after the other; for a few you will hear just one. "
                "After each version, rate how well the clicks matched the beat."
                )
                + "</p>"
                "<p style='text-align:center'>"
                "<img src='/static/instruction_waveform.gif' "
                "alt='Illustration: waveform with click marks' "
                "style='max-width:100%;border-radius:6px;margin:8px 0' />"
                "</p>"
                "<p>"
                + _(
                    "If you hear two versions of the same excerpt, you can give them the same "
                    "rating or different ratings &mdash; whatever feels right to you. "
                    "There are no correct answers; we are interested in your honest impression "
                    "of how well each click track tracked the beat of the music."
                )
                + "</p></div>"
            ),
            time_estimate=20,
        ),
        BalancedUnifiedTrialMaker(
            id_="unified_beat_validation",
            trial_class=VersionRatingTrial,
            nodes=get_validation_nodes,
            expected_trials_per_participant=EXPECTED_TRIALS_PER_PARTICIPANT,
            max_trials_per_participant=MAX_TRIALS_PER_PARTICIPANT,
            max_trials_per_block=MAX_CONDITIONS_PER_STEM,
            balance_across_nodes=True,
            allow_repeated_nodes=False,
        ),
        AttentionCheckPage(),
        FinalFeedbackPage(),
        BasicDemography(),
        BasicMusic(),
        GMSI(label="gmsi", subscales=["Musical Training"]),
        InfoPage(_("Thank you — your responses were recorded."), time_estimate=5),
    )


def get_prolific_settings():
    # $5 / 30 min  =  $10/hr (slight bump above MIREX's $9/hr for faster collection).
    settings = {
        "recruiter": "prolific",
        "prolific_estimated_completion_minutes": 30,
        "prolific_maximum_allowed_minutes": 50,
        "base_payment": 5,
        "auto_recruit": False,
        "currency": "$",
        "wage_per_hour": 10,
    }
    return settings


def _check_translations_no_extract(path=".", locales=None, **kwargs):
    from psynet.translation.check import check_translations

    return check_translations(path=path, locales=locales, recreate_pot=False)


class Exp(psynet.experiment.Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i, r in enumerate(self.pre_deploy_routines):
            if getattr(r, "label", "") == "check_experiment_translations":
                from psynet.timeline import PreDeployRoutine

                self.pre_deploy_routines[i] = PreDeployRoutine(
                    "check_experiment_translations", _check_translations_no_extract
                )
                break

    config = {
        **get_prolific_settings(),
        "supported_locales": '["en"]',
        "initial_recruitment_size": 130,
        "title": "Listening: rate how well clicks match the beat of short music excerpts",
        "description": (
            "Listen to short music excerpts with click tracks overlaid on top, and rate how "
            "well each click track matches the beat of the music. ~50 trials, ~30 minutes."
        ),
        "contact_email_on_error": "${CONTACT_EMAIL}",
        "organization_name": "${ORGANIZATION}",
        "dashboard_password": "${DASHBOARD_PASSWORD}",
        "dashboard_user": "${DASHBOARD_USER}",
        "dyno_type": "performance-l",
        "language": LOCALE,
        "locale": LOCALE,
        "num_dynos_web": 1,
        "num_dynos_worker": 1,
        "redis_size": "premium-3",
        "host": "0.0.0.0",
        "clock_on": True,
        "heroku_python_version": "3.12.7",
        "database_url": "postgresql://postgres@localhost/dallinger",
        "database_size": "standard-2",
        "show_reward": False,
        "show_progress_bar": True,
    }
    timeline = get_timeline()
