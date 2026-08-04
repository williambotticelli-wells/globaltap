# imports
import json
import tempfile
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

from markupsafe import Markup
from repp.analysis import REPPAnalysis
# from repp.config import sms_tapping
from repp.stimulus import REPPStimulus
from repp.utils import save_json_to_file, save_samples_to_file

from psynet.demography.general import (
    BasicDemography,
    BasicMusic
)
from .gmsi import GMSI

import psynet.experiment
from psynet.asset import CachedFunctionAsset, LocalStorage, S3Storage
from psynet.consent import NoConsent, MainConsent, LucidConsent, AudiovisualConsent
from psynet.modular_page import AudioPrompt, AudioRecordControl, ModularPage, ImagePrompt
from psynet.page import InfoPage, SuccessfulEndPage, UnsuccessfulEndPage
from psynet.timeline import ProgressDisplay, ProgressStage, Timeline, join, conditional
from psynet.trial.audio import AudioRecordTrial
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker
from psynet.recruiters import get_lucid_settings

# Import enhanced analysis functions from separate module
from .repp_beatfinding.beat_detection import (
    do_beat_detection_analysis,
)

from .custom_config import sms_tapping, config_params

# repp
from .repp_prescreens import (
    NumpySerializer,
    REPPMarkersTest,
    REPPTappingCalibration,
    REPPVolumeCalibrationMusic,
)
from .logic import CustomUnsuccessfulEndLogic

########################################################################################################################
# SETUP
########################################################################################################################
DEBUG = False

# recruitment
RECRUITER = "prolific"  # or "lucid" or "hotair" (for local debug)

#LANGUAGE = "ENG"
#COUNTRY = "GB"
#LUCID_CONFIG_PATH = f"qualifications/lucid/lucid-ENG-GB.json"
#WAGE_PER_HOUR = 12.21

INITIAL_RECRUITMENT_SIZE = 10
AUTO_RECRUIT = False
NUM_PARTICIPANTS = 90          # 90 participants × 20 stems/participant ≈ 45 taps/stem (40 stems total)
NUM_TRIALS_PER_PARTICIPANT = 20  # each participant taps 20 of 40 stems
ISO_NUM_TRIALS_PER_PARTICIPANT = 3

def get_prolific_settings():
    qual_path = "qualifications/qualification_prolific_en.json"
    with open(qual_path, "r") as f:
        qualification = json.dumps(json.load(f))
    return {
        "recruiter": RECRUITER,
        "prolific_recruitment_config": qualification,
        "base_payment": 2.50,
        "prolific_estimated_completion_minutes": 30,
        "prolific_is_custom_screening": False,
        "auto_recruit": False,
        "currency": "£",
        "wage_per_hour": 5,
        "title": "Music Tapping Study — Ballroom (Quiet Room, Laptop Speakers Only, Chrome, ~25 min)",
        "description": "Tap along to short music excerpts.",
    }


consent = MainConsent

#if RECRUITER == "lucid":
#    consent = LucidConsent
#    recruiter_settings = get_lucid_settings(
#        lucid_recruitment_config_path=LUCID_CONFIG_PATH,
#        termination_time_in_s=120 * 60,
#        debug_recruiter=False,
#        initial_response_within_s=100,
#        bid_incidence=20,
#        inactivity_timeout_in_s=5 * 60,
#        no_focus_timeout_in_s=120,
#        aggressive_no_focus_timeout_in_s=15,
#    )
#    recruiter_settings = {
#        **recruiter_settings,
#        "title": "Global Music Tapping Study (Quiet Room, Laptop Speakers Only, Chrome, ~30 min)",
#        "description": "Tap to music.",
#        "wage_per_hour": float(WAGE_PER_HOUR),
#        "publish_experiment": True,
#    }

recruiter_settings = get_prolific_settings()


#elif RECRUITER == "hotair":
#    consent = NoConsent
#    recruiter_settings = {
#        "recruiter": "hotair",
#        "title": "Tapping experiment",
#        "wage_per_hour": float(WAGE_PER_HOUR),
#    }

# time estimates (MIREX stimuli are ~30s; +5–10s for silence/transition)
DURATION_ESTIMATED_TRIAL = 50

# failing criteria
MIN_RAW_TAPS = 3
MAX_RAW_TAPS = 300


# Config wrapper to include MIN_RAW_TAPS and MAX_RAW_TAPS for beat detection analysis
class ConfigWithThresholds:
    """Wrapper around sms_tapping config that adds MIN_RAW_TAPS and MAX_RAW_TAPS attributes."""

    def __init__(self, base_config):
        # Copy all attributes from base_config
        for attr in dir(base_config):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(base_config, attr))
                except AttributeError:
                    pass  # Skip attributes that can't be read
        # Add custom thresholds from experiment settings
        self.MIN_RAW_TAPS = MIN_RAW_TAPS
        self.MAX_RAW_TAPS = MAX_RAW_TAPS


static_file = {"tapping_img": "/static/images/tapping_img.png"
               }

########################################################################################################################
# Stimuli
########################################################################################################################
# Isochronus stimuli
tempo_800_ms = [800] * 15  # ISO 800ms
tempo_600_ms = [600] * 12  # ISO 600ms

# iso_stimulus_onsets = [tempo_800_ms, tempo_600_ms]
# iso_stimulus_names = ["iso_800ms", "iso_600ms"]

iso_stimulus_onsets = [
    tempo_800_ms,
    tempo_600_ms,
    tempo_800_ms,
]

iso_stimulus_names = [
    "iso_800ms_Trial1",
    "iso_600ms_Trial2",
    "iso_800ms_Trial3",
]

# 1-minute ISO stimulus
ISO_IOI = 800  # 800 ms between beats
ISO_DURATION_MS = 60_000  # 1 minute
num_beats = ISO_DURATION_MS // ISO_IOI

iso_stimulus_onset_1min = [[ISO_IOI] * num_beats]
iso_stimulus_name_1min = ["iso_800ms_1min"]


@cache
def create_iso_stim_with_repp(stim_name, stim_ioi):
    stimulus = REPPStimulus(stim_name, config=sms_tapping)
    stim_onsets = stimulus.make_onsets_from_ioi(stim_ioi)
    stim_prepared, stim_info, _ = stimulus.prepare_stim_from_onsets(stim_onsets)
    info = json.dumps(stim_info, cls=NumpySerializer)
    return stim_prepared, info


def generate_iso_stimulus_audio(path, stim_name, list_iois):
    stim_prepared, info = create_iso_stim_with_repp(stim_name, tuple(list_iois))
    save_samples_to_file(stim_prepared, path, sms_tapping.FS)


def generate_iso_stimulus_info(path, stim_name, list_iois):
    stim_prepared, info = create_iso_stim_with_repp(stim_name, tuple(list_iois))
    save_json_to_file(info, path)


nodes_iso = [
    StaticNode(
        definition={
            "stim_name": name,
            "list_iois": iois,
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_iso_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_iso_stimulus_info),
        },
    )
    for name, iois in zip(iso_stimulus_names, iso_stimulus_onsets)
]

# # Music stimuli for beat-finding task (no onsets required)
# music_stimulus_name = ["track1", "track2"]
# music_stimulus_audio = ["music/0R8IbpKXavM.wav", "music/ehHKH5PbGYc.wav"]


# 2x data collection: 20 target stems in static/tapping_2x_wav/.
# Sorting keeps stimulus order deterministic across local and deployed runs.
music_folder = "static/ballroom_heldout_wav"
_music_filenames = sorted(
    f for f in os.listdir(music_folder)
    if f.lower().endswith(".wav")
)
music_stimulus_audio = [os.path.join(music_folder, f) for f in _music_filenames]
music_stimulus_name = [os.path.splitext(f)[0] for f in _music_filenames]


def load_audio_only_from_file(fs, audio_filename):
    """
    Load audio file without requiring onsets file.

    Parameters
    ----------
    fs : int
        Target sampling frequency in Hz
    audio_filename : str
        Path to audio file

    Returns
    -------
    np.ndarray
        Loaded and resampled audio data
    """
    stimulus = REPPStimulus("temp", config=config_params)
    return stimulus.load_resample_file(fs, audio_filename)


def filter_and_add_markers_no_onsets(stim, config):
    """
    Apply filtering and add markers without requiring onset information.

    Parameters
    ----------
    stim : np.ndarray
        Raw audio stimulus data
    config : Config
        Configuration parameters

    Returns
    -------
    tuple[np.ndarray, dict]
        - Prepared stimulus array
        - Dictionary containing stimulus information
    """
    stimulus = REPPStimulus("temp", config=config)

    # Apply spectral filtering
    filtered_stim = stimulus.filter_stim(
        config.FS, stim, config.STIM_RANGE, config.STIM_AMPLITUDE
    )

    # Create marker sounds
    markers_sound = stimulus.make_markers_sound(
        config.FS,
        config.MARKERS_DURATION,
        config.MARKERS_ATTACK,
        config.MARKERS_RANGE,
        config.MARKERS_AMPLITUDE
    )

    # Add markers at beginning and end
    markers_onsets, markers_channel = stimulus.add_markers_sound(
        config.FS,
        stim,
        config.MARKERS_IOI,
        config.MARKERS_BEGINNING,
        config.MARKERS_END,
        config.STIM_BEGINNING,
        config.MARKERS_END_SLACK
    )

    # Combine markers with filtered stimulus
    stim_prepared = stimulus.put_clicks_in_audio(markers_channel, config.FS, markers_sound, markers_onsets)
    stim_start_samples = int(round(config.STIM_BEGINNING * config.FS / 1000.0))
    stim_prepared[stim_start_samples:(stim_start_samples + len(filtered_stim))] += filtered_stim

    stim_duration = len(stim_prepared) / config.FS

    # Create minimal stim_info for beat-finding task
    stim_info = {
        'stim_duration': stim_duration,
        'stim_onsets': [],  # Empty for beat-finding
        'stim_shifted_onsets': [],  # Empty for beat-finding
        'onset_is_played': np.array([]),  # Empty for beat-finding
        'markers_onsets': markers_onsets,
        'stim_name': 'beat_finding_music'
    }

    return stim_prepared, stim_info


@cache
def create_music_stim_with_repp_beat_finding(stim_name, audio_filename, fs=44100):
    """
    Create music stimulus for beat-finding task without requiring onsets file.
    """
    # Load audio file
    stim = load_audio_only_from_file(fs, audio_filename)

    # Convert stereo to mono if needed
    if len(stim.shape) == 2:
        stim = stim[:, 0]

    # Apply filtering and add markers
    stim_prepared, stim_info = filter_and_add_markers_no_onsets(stim, config_params)
    stim_info["stim_name"] = stim_name
    stim_info["audio_filename"] = audio_filename  # For export/merge → KDE maps to trainN.txt

    info = json.dumps(stim_info, cls=NumpySerializer)
    return stim_prepared, info


def generate_music_stimulus_audio(path, stim_name, audio_filename):
    stim_prepared, _ = create_music_stim_with_repp_beat_finding(stim_name, audio_filename)
    save_samples_to_file(stim_prepared, path, config_params.FS)


def generate_music_stimulus_info(path, stim_name, audio_filename):
    stim_prepared, info = create_music_stim_with_repp_beat_finding(stim_name, audio_filename)
    save_json_to_file(info, path)


nodes_music = [
    StaticNode(
        definition={
            "stim_name": name,
            "audio_filename": audio,
        },
        assets={
            "stimulus_audio": CachedFunctionAsset(generate_music_stimulus_audio),
            "stimulus_info": CachedFunctionAsset(generate_music_stimulus_info),
        },
    )
    for name, audio in zip(music_stimulus_name, music_stimulus_audio)
]


########################################################################################################################
# Experiment parts  (UPDATED: ISO slack via MAX_RAW_TAPS only)
########################################################################################################################

from copy import deepcopy

# Make an ISO-only REPP config with more slack on raw taps.
# This ONLY changes the "too many detected taps" threshold; nothing else.
sms_tapping_iso = deepcopy(sms_tapping)
sms_tapping_iso.update({
    # Default was 150 (% of stim onsets). Bump to allow extra detections.
    # 200 = up to 2.0x stim count (e.g., 15 stim -> up to 30 raw resp)
    "MAX_RAW_TAPS": 165,
    # (optional) if you also want a bit of forgiveness for too-few:
    # "MIN_RAW_TAPS": 40,
})


class TapTrialAnalysisISO(AudioRecordTrial, StaticTrial):

    def get_info(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assets["stimulus_info"].export(f.name)
            with open(f.name, "r") as reader:
                return json.loads(json.load(reader))  # REPP double-JSON-encodes

    def analyze_recording(self, audio_file: str, output_plot: str):
        info = self.get_info()
        stim_name = info["stim_name"]
        title_in_graph = f"Participant {self.participant_id}"

        # Use ISO-specific config (more raw-tap slack)
        repp = REPPAnalysis(config=sms_tapping_iso)

        output, analysis, is_failed = repp.do_analysis(
            info, audio_file, title_in_graph, output_plot
        )

        is_failed_flag = is_failed.get("failed", True)
        reason = is_failed.get("reason", "Analysis failed")

        extracted_onsets_json = json.dumps(output, cls=NumpySerializer)
        analysis_json = json.dumps(analysis, cls=NumpySerializer)

        # Add a couple debug fields so you can confirm this is what fixed it
        try:
            analysis["max_raw_taps_pct_used"] = sms_tapping_iso.__dict__.get("MAX_RAW_TAPS", None)
        except Exception:
            pass

        return {
            "failed": is_failed_flag,
            "reason": reason,
            "extracted_onsets": extracted_onsets_json,
            "analysis": analysis_json,
            "stim_name": stim_name,
        }


class TapTrialAnalysis(AudioRecordTrial, StaticTrial):
    def get_info(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assets["stimulus_info"].export(f.name)
            with open(f.name, "r") as reader:
                return json.loads(json.load(reader))

    def analyze_recording(self, audio_file: str, output_plot: str):
        info = self.get_info()
        stim_name = info["stim_name"]
        title_in_graph = f"Participant {self.participant_id}"

        config = ConfigWithThresholds(config_params)

        output, analysis, is_failed = do_beat_detection_analysis(
            audio_file,
            title_in_graph,
            output_plot,
            stim_info=info,
            config=config,
            display_zoomed_markers=False,
        )

        is_failed_flag = is_failed.get("failed", True)
        reason = is_failed.get("reason", "Analysis failed")

        extracted_onsets_json = json.dumps(output, cls=NumpySerializer)
        analysis_json = json.dumps(analysis, cls=NumpySerializer)

        return {
            "failed": is_failed_flag,
            "reason": reason,
            "extracted_onsets": extracted_onsets_json,
            "analysis": analysis_json,
            "stim_name": stim_name,
            "audio_filename": info.get("audio_filename", self.definition.get("audio_filename", "")),
        }


class TapTrial(TapTrialAnalysis):
    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        trial_number = self.position + 1
        return ModularPage(
            "trial_main_page",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h4>Keep your taps simple, steady and evenly spaced. Adjust if the tempo changes.</h4>

                    Trial number {trial_number} out of {NUM_TRIALS_PER_PARTICIPANT} trials.
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=False,
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            progress_display=ProgressDisplay(
                show_bar=True,
                stages=[
                    ProgressStage(3.5, "Wait in silence...", "red"),
                    ProgressStage([3.5, (duration_rec - 6)], "START TAPPING!", "green"),
                    ProgressStage(3.5, "Stop tapping and wait in silence...", "red", persistent=False),
                    ProgressStage(0.5, "Press Next when you are ready to continue...", "orange", persistent=True),
                ],
            ),
        )

    def get_bot_response_media(self):
        raise NotImplementedError


class TapTrialISO(TapTrialAnalysisISO):
    time_estimate = 15

    def show_trial(self, experiment, participant):
        info = self.get_info()
        duration_rec = info["stim_duration"]
        trial_number = self.position + 1
        return ModularPage(
            "iso_trial_main_page",
            AudioPrompt(
                self.assets["stimulus_audio"].url,
                Markup(
                    f"""
                    <br><h3>Tap in time to the musical beat.</h3>
                    Trial number {trial_number} out of {ISO_NUM_TRIALS_PER_PARTICIPANT} trials.
                    """
                ),
            ),
            AudioRecordControl(
                duration=duration_rec,
                show_meter=False,
                controls=False,
                auto_advance=False,
                bot_response_media=self.get_bot_response_media(),
            ),
            time_estimate=duration_rec + 5,
            progress_display=ProgressDisplay(
                show_bar=True,
                stages=[
                    ProgressStage(3.5, "Wait in silence...", "red"),
                    ProgressStage([3.5, (duration_rec - 6)], "START TAPPING!", "green"),
                    ProgressStage(3.5, "Stop tapping and wait in silence...", "red", persistent=False),
                    ProgressStage(0.5, "Press Next when you are ready to continue...", "orange", persistent=True),
                ],
            ),
        )

    def get_bot_response_media(self):
        return None


class TapTrialMusic(TapTrial):
    time_estimate = DURATION_ESTIMATED_TRIAL

    def get_bot_response_media(self):
        return None


def welcome():
    return InfoPage(
        Markup(
            """
            <h3>Welcome!</h3>
<hr>

<p><b>Before you continue, please confirm:</b></p>
<ul>
  <li><b>Laptop only</b> (no phone/tablet).</li>
  <li><b>Chrome browser</b>.</li>
  <li><b>Silent environment</b> (no talking, music, TV, fans, street noise).</li>
  <li><b>Use laptop speakers only</b> (no headphones/earbuds, no external speakers, no Bluetooth devices).</li>
</ul>

<p>
In this experiment, you will hear rhythms and music and tap a steady beat using <b>one finger</b>
on the <b>laptop’s surface</b> (next to the trackpad, <b>NOT</b> on the keyboard/trackpad).
</p>

<p>
<b>Important:</b> This experiment uses precise audio recording. If you are not in a quiet room or use the wrong setup,
the experiment will <b>terminate early</b>.
</p>

<hr>
<p>Press <b>Next</b> when you are ready to begin.</p>
            """
        ),
        time_estimate=3
    )


class ISOTappingTrialMaker(StaticTrialMaker):
    performance_check_type = "score"

    def __init__(
            self,
            id_="ISO_tapping",
            nodes=None,
            n_trials=None,
            performance_threshold: int = 1,
            time_estimate_per_trial: float = 5.0,
            target_n_participants=None,
    ):
        self.performance_threshold = performance_threshold
        self.time_estimate_per_trial = time_estimate_per_trial

        super().__init__(
            id_=id_,
            trial_class=TapTrialISO,
            nodes=nodes,
            expected_trials_per_participant=len(nodes) if nodes else n_trials,
            max_trials_per_participant=len(nodes) if nodes else n_trials,
            # target_n_participants=target_n_participants,
            # recruit_mode="n_participants",
            check_performance_at_end=True,
        )

    def performance_check(self, experiment, participant, participant_trials):
        score = 0

        # Count successful trials
        for trial in participant_trials:
            analysis = trial.analysis
            if analysis and not analysis.get("failed", True):
                score += 1

        participant.var.set("iso_score", score)

        # If threshold is None, skip fail check
        if self.performance_threshold is None:
            passed = True
        else:
            passed = score >= self.performance_threshold
            if not passed:
                participant.fail()

        participant.var.set("iso_passed", passed)
        return {"score": score, "passed": passed}


# Tapping tasks
ISO_tapping = join(
    InfoPage(
        Markup(
            """
            <h3>Tapping to Rhythms</h3>
            <hr>
            <p>
        Before the main music trials, you will first practice tapping along to some simple rhythms.  
        </p>
            In each trial, you will hear a rhythm playing at a constant pace.
            <br><br>
            <b><b>Your goal is to tap in time to the beat of the rhythm.</b></b> <br><br>
            Note:
            <ul>
                <li>Start tapping as soon as the rhythm starts and continue tapping until the rhythm ends.</li>
                <li>Please remain silent before the rhythm starts and after the rhythm ends.</li>
                <li>At the beginning and end of each rhythm, you will hear three consecutive beeps.</li>
                <li>Do not tap during these beeps (remain silent), as they signal the beginning and end of each rhythm.</li>
            </ul>
            <br>
            <hr>
            """
        ),
        time_estimate=14,
    ),
    # StaticTrialMaker(
    #     id_="ISO_tapping",
    #     trial_class=TapTrialISO,
    #     nodes=nodes_iso,
    #     expected_trials_per_participant=len(nodes_iso),
    #     target_n_participants=NUM_PARTICIPANTS,
    #     recruit_mode="n_participants",
    #     check_performance_at_end=False,
    # ),
    ISOTappingTrialMaker(
        id_="ISO_tapping",
        nodes=nodes_iso,
        performance_threshold=1,
        # target_n_participants=NUM_PARTICIPANTS,
    ),
)

ISO_tapping_1min = join(
    ISOTappingTrialMaker(
        id_="ISO_tapping_1min",
        nodes=[
            StaticNode(
                definition={
                    "stim_name": iso_stimulus_name_1min[0],
                    "list_iois": iso_stimulus_onset_1min[0],
                },
                assets={
                    "stimulus_audio": CachedFunctionAsset(generate_iso_stimulus_audio),
                    "stimulus_info": CachedFunctionAsset(generate_iso_stimulus_info),
                },
            )
        ],
        performance_threshold=0,
    ),
)

# Tapping tasks
music_tapping = join(
    InfoPage(
        Markup(
            """       
            <h3>Now, Tap to Music!</h3>
            <hr>

            <p>Great job with the practice rhythms!</p>

            <p>Next, you’ll move on to the main part of the experiment. You will tap along with actual music clips.</p>


            <p>➡ Continue to the next page for more detailed instructions before starting the music trials.</p>

            <hr>
            """
        ),
        time_estimate=5,
    ),

    ModularPage(
        "how_to_tap",
        ImagePrompt(
            static_file["tapping_img"],
            Markup(
                """
       <h3>How to Tap</h3>
  <hr>
  <p>
  Please tap with <b>one finger</b> on the laptop’s surface (not the mousepad).  
  </p>

  <p>
  The tempo may change during a clip.  
  Do not stop to listen, instead adjust your tapping continuously to what the music suggests.
  </p>
  <p>
  Finalizing trials can take a few seconds, so please be patient.
  </p>

                """
            ),
            width="65%", height="auto",
        ),
        time_estimate=10,
    ),

    InfoPage(
        Markup(
            """       
            <h3>Tapping to Music</h3>
            <hr>
            <p>You will hear a variety of music excerpts.</p>

            <p>
            Please <b>tap along with the music</b> using <b>steady, evenly spaced beats.</b>
            </p>

            <div style="border: 1px solid #ccc; padding: 12px; border-radius: 6px; background-color: #f9f9f9;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li>You don’t need to copy the rhythm in the music clips exactly or match every sound with a tap. Instead, keep your taps aligned with the general beat and tempo of the music.</li>
                    <li>DO NOT tap too fast or for every sound you hear, keep your tapping simple and relaxed.</li>
                    <li>Tap at a tempo that feels natural to you. Think about times when you or others have marched or perhaps clapped or nodded along to music.</li>

                </ul>
            </div>
            <hr>

            <p>There’s no right or wrong response, just your personal sense of what tempo fits the music.</p>
        """
        ),
        time_estimate=5,
    ),

    StaticTrialMaker(
        id_="music_tapping",
        trial_class=TapTrialMusic,
        nodes=nodes_music,
        expected_trials_per_participant=NUM_TRIALS_PER_PARTICIPANT,
        max_trials_per_participant=NUM_TRIALS_PER_PARTICIPANT,
        # target_n_participants=NUM_PARTICIPANTS,
        # recruit_mode="n_participants",
        check_performance_at_end=False,

    ),
)


# export EXP_MAX_SIZE_MB=800

########################################################################################################################
# Timeline
########################################################################################################################
# Use LocalStorage for local dev when AWS credentials are not configured.
# Set USE_LOCAL_ASSET_STORAGE=1 to force local storage (avoids InvalidAccessKeyId error).
_use_local_storage = os.environ.get("USE_LOCAL_ASSET_STORAGE", "").lower() in ("1", "true", "yes")


class Exp(psynet.experiment.Experiment):
    label = "Tapping Experiment"
    asset_storage = LocalStorage() if _use_local_storage else S3Storage("melody-perception", "ballroom_wav")

    config = {
        "recruiter": RECRUITER,
        **recruiter_settings,
        "initial_recruitment_size": INITIAL_RECRUITMENT_SIZE,
        "auto_recruit": AUTO_RECRUIT,
        "contact_email_on_error": "${CONTACT_EMAIL}",
        "organization_name": "${ORGANIZATION}",
        "show_reward": False,
    }

    if DEBUG:
        timeline = Timeline(
            NoConsent(),
            welcome(),
            #REPPVolumeCalibrationMusic(),
            music_tapping,
            SuccessfulEndPage(),
        )
    else:
        timeline = Timeline(
            consent(),
            AudiovisualConsent(),
            welcome(),
            REPPVolumeCalibrationMusic(),  # calibrate volume with music
            REPPMarkersTest(),  # pre-screening filtering participants based on recording test (markers)
            conditional(
                label="marker_failure_check",
                condition=lambda experiment, participant: participant.failed,
                logic_if_true=CustomUnsuccessfulEndLogic()
            ),
            REPPTappingCalibration(),  # calibrate tapping
            ISO_tapping,
            conditional(
                label="iso_tapping_failure_check",
                condition=lambda experiment, participant: participant.failed,
                logic_if_true=CustomUnsuccessfulEndLogic()
            ),
            music_tapping,
            BasicDemography(),
            BasicMusic(),
            GMSI(label="gmsi_2", subscales=["Musical Training", "Active Engagement", "Perceptual Abilities"]),
            SuccessfulEndPage(),
        )


