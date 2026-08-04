####################################################################################################
# File:     config.py
# Purpose:  Main parameters for rhythm experiments

# Author: GlobalTap authors
####################################################################################################
from copy import deepcopy


class ConfigUpdater:
    """ A class with methods to create the configuration (list of global parameters) to be used with REPP

    Attributes
    -----------
    LABEL : str
        The label of the tapping paradigm.
    FS : int
        Sampling frequency.
    FS0 : int
        Sampling frequency (downsampled).
    STIM_RANGE : list
        Stimulus frequency range (Hz): [min, max].
    STIM_AMPLITUDE : float
        Target amplitude for the audio stimulus to play in the recording phase.
    MARKERS_RANGE : list
        Markers frequency range (Hz): [min, max].
    TEST_RANGE : list
        Test frequency range (Hz): [min, max].
    MARKERS_AMPLITUDE :  float
        Target amplitude of the marker sound.
    MARKERS_DURATION : float
        Duration of the marker sound (ms).
    MARKERS_ATTACK : float
        Attack of the marker sound.
    MARKERS_IOI : list
        List of markers' IOIs.
    MARKERS_BEGINNING : float
        Time of the markers relative to the beginning of the file (ms).
    STIM_BEGINNING : float
        Time of the stimulus relative to the beginning of the file (ms).
    MARKERS_END : float
        Time of the markers relative to the end of the stimulus (ms).
    MARKERS_END_SLACK : float
        Duration of additional slack (including the markers) after the end of the stimulus.
    MARKERS_MAX_ERROR : float
        Maximum time error allowed between the markers' locations and the markers' detected onsets. Used in the
        failing criteria.
    MIN_RAW_TAPS: float
        Percentage of min raw taps allowed (relative to number of stim onsets). Used in the failing criteria.
    MAX_RAW_TAPS: float
        Percentage of max raw taps allowed (relative to number of stim onsets). Used in the failing criteria.
    MIN_NUM_ASYNC : float
        Min number of response-stim asynchronies allowed to calculate mean/ sd of asynchrony. Used in the
        failing criteria.
    MIN_SD_ASYNC : float
        Min threshold for the SD of asynchrony indicating impossible values. Used in the failing criteria.
    CLICK_FILENAME : str
        Name of the audio file to to create the metronome sound.
    USE_CLICK_FILENAME : bool
        Boolean to decide whether to use click_filename or not.
    CLICK_DURATION : float
        Duration of the click sound (ms).
    CLICK_FREQUENCY : int
        Frequency of the click sound (Hz).
    CLICK_ATTACK : float
        Attack of the click sound.
    TAPPING_RANGE : list
        Tapping frequency range (Hz): [min, max].
    EXTRACT_THRESH : list
        Relative threshold for the onset extraction procedure [tapping, markers].
    EXTRACT_FIRST_WINDOW : list
        First time window to separate supra-threshold samples in the onset extraction procedure [tapping, markers] (ms).
    EXTRACT_SECOND_WINDOW : list
        Second time window to separate samples in the onset extraction procedure [tapping, markers] (ms).
    EXTRACT_COMPRESS_FACTOR : float
        Compression exponent to extract the envelope from the recording.
    EXTRACT_FADE_IN : float
        Initial fade in to extract the envelope from the recording (ms).
    CLEAN_BIN_WINDOW : float
        Bin width window to apply to markers' cleaning procedure (ms).
    CLEAN_MAX_RATIO : float
        Maximum clean ratio allowed in the markers' cleaning procedure.
    CLEAN_LOCATION_RATIO : list
        Ratio of the location to normalize the tapping signal in the cleaning procedure [start, end].
    CLEAN_NORMALIZE_FACTOR : float
        Factor to normalize the tapping signal in the location ration.
    ONSET_MATCHING_WINDOW_MS : float
        Matching window to detect tapping onsets corresponding to the stimulus onsets ( in ms). Onset needs to comply
         to both phase and ms constraints.
    ONSET_MATCHING_WINDOW_PHASE : float
        Matching window to detect tapping onsets corresponding to the stimulus onsets (in relative phase units). Onsets
        needs to comply  to both phase and ms constraints.
    MARKERS_MATCHING_WINDOW : float
        Matching window to detect markers onsets corresponding to the known markers onsets.
    DISPLAY_PLOT : bool
        Boolean to decide whether to display plots or not.
    PLOTS_TO_DISPLAY: list
        List of subplots to display in the main plot [rows, columns].
    """
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

    def update(self, *args, **kwargs):
        """ Create a configuration class by specifying a dictionary with global parameters
        """
        return self.__dict__.update(*args, **kwargs)

    def create_config(old_config, *args, **kwargs):
        """ Create a configuration class by updating an old one
        """
        new_config = deepcopy(old_config)
        new_config.update(*args, **kwargs)
        return new_config

sms_tapping = ConfigUpdater({  # Global parameters for sms experiments
    'LABEL': 'sms_tapping',
    'FS': 44100,
    'FS0': 22000,
    # Stimulus preparation step
    'STIM_RANGE': [30, 1000],
    'STIM_AMPLITUDE': 0.12,
    'MARKERS_RANGE': [200, 340],
    'TEST_RANGE': [100, 170],
    'MARKERS_AMPLITUDE': 0.9,
    'MARKERS_ATTACK': 2,
    'MARKERS_DURATION': 15,
    'MARKERS_IOI': [0, 280, 230],
    'MARKERS_BEGINNING': 2000.0,
    'STIM_BEGINNING': 4000.0,
    'MARKERS_END': 2000.0,
    'MARKERS_END_SLACK': 6000.0,
    # failing criteria
    'MIN_RAW_TAPS': 50,
    'MAX_RAW_TAPS': 150,
    'MARKERS_MAX_ERROR': 15,
    'MIN_NUM_ASYNC': 2,
    'MIN_SD_ASYNC': 10,
    # metronome sound
    'CLICK_FILENAME': 'click01.wav',
    'USE_CLICK_FILENAME': False,
    'CLICK_DURATION': 50,
    'CLICK_FREQUENCY': 1000,
    'CLICK_ATTACK': 5,
    # Onset extraction step
    'TAPPING_RANGE': [50, 500],
    'EXTRACT_THRESH': [0.25, 0.12], # [0.19, 0.225]
    'EXTRACT_FIRST_WINDOW': [18, 18],
    'EXTRACT_SECOND_WINDOW': [26, 60], # [26, 120]
    'EXTRACT_COMPRESS_FACTOR': 1.1, #1.3
    'EXTRACT_FADE_IN': 500,
    # Cleaning procedure
    'CLEAN_BIN_WINDOW': 100,
    'CLEAN_MAX_RATIO': 10,
    'CLEAN_LOCATION_RATIO': [0.333, 0.66],
    'CLEAN_NORMALIZE_FACTOR': 0.05,
    # Onset alignment step
    'ONSET_MATCHING_WINDOW_MS': 1999.0,  # if you want to use only phase set it to 1999 (2 sec)
    'ONSET_MATCHING_WINDOW_PHASE': [-0.4, 0.4],  # for relative phase (if you want to use only ms set it to [-1 1])
    'MARKERS_MATCHING_WINDOW': 35.0,
    # Plotting
    'DISPLAY_PLOTS': True,
    'PLOTS_TO_DISPLAY': [3, 4]
    })

config_params = ConfigUpdater.create_config(
    sms_tapping,
    {
        'LABEL': 'mirex tapping',
        'MARKERS_MAX_ERROR': 33,  # 15

    }
)

iterated_tapping = ConfigUpdater.create_config(
    sms_tapping,
    {
        'LABEL': 'iterated tapping',
        'USE_CLICK_FILENAME': True,
        'PLOTS_TO_DISPLAY': [4, 4],
        # Iterated tapping: these variables need better naming and definition on top of the file
        'INTERVAL_RHYTHM': 3,
        'REPEATS': 10,
        'TOTAL_DURATION': 2000,
        'PROB_NO_CHANGE': 1/3,
        'MIN_RATIO': 150.0/1000.0,
        'SLACK_RATIO': 0.95,
        'IS_FIXED_DURATION': True
        })

# Iterated tapping from memory (2-interval)
iterated_tapping_memory = ConfigUpdater.create_config(
    iterated_tapping,
    {
        'LABEL': 'iterated tapping from memory (2-interval)',
        'ONSET_MATCHING_WINDOW_MS': 200.0,
        'ONSET_MATCHING_WINDOW_PHASE': [-1, 1],  # make it inactive
        'CLEAN_NORMALIZE_FACTOR': 0.1,
        'PLOTS_TO_DISPLAY': [2, 2],
        'STIM_RANGE': [30, 800],
        'STIM_AMPLITUDE': 0.15,
        'INTERVAL_RHYTHM': 2,
        'REPEATS': 3,
        'MIN_RATIO': 1.0/10.0,
        'DURATION_RANGE': [500, 2000],
        'TOTAL_DURATION': 1000,
        'MAX_DIFF_RATIO': 0.25,  # set to 10 to make it not active
        'MAX_DIF_IOI': 500,
        'IS_FIXED_DURATION': False,
        'MIN_TAPS_PLAYED': 5,
        'EXTRACT_SECOND_WINDOW': [26, 60],
        'SAMPLE_LOG_SCALE': True
    })

# gsp with rhythms (slider)
gsp_rhythm = ConfigUpdater.create_config(
    iterated_tapping_memory,
    {
        'LABEL': 'GSP rhythm',
        'USE_CLICK_FILENAME': False,
        'INTERVAL_RHYTHM': 2,
        'REPEATS': 3,
        'MIN_RATIO': 1.0/10.0,
        'DURATION_RANGE': [500, 2000],
        'RATIO_RANGE': [0.1, 0.9]
     }
)

