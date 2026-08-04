"""
Shared styling and constants for ISMIR 2026 paper figures.

This module is intentionally small and self-contained: every figure
generated for the paper or supplementary HTML can be regenerated from
the CSVs in ``data/`` plus this stylesheet — no external data
tree or PROJECT_ROOT is required.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Included data ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_STEM_CSV = REPO_ROOT / "data" / "per_stem_metrics.csv"

# ── Method colors (fixed across all figures) ─────────────────────────────────
METHOD_COLORS = {
    "Crowd":   "#2166AC",  # strong blue (the paper's pipeline output)
    "Ref":     "#4DAF4A",  # green
    "GT":      "#4DAF4A",  # alias for Ref (legacy GT label)
    "Opt":     "#2166AC",  # alias for Crowd (legacy Opt label)
}

METHOD_LABELS = {
    "Crowd": "Pipeline (Crowd)",
    "Ref":   "Reference",
}

CORPUS_GROUP_LABELS = {
    "ballroom":     "Ballroom",
    "benchmark100": "Mixed corpora",
    "mirex":        "MIREX",
    "globalmood40": "GlobalMood",
}
CORPUS_GROUP_COLORS = {
    "ballroom":     "#2166AC",
    "benchmark100": "#D6604D",
    "mirex":        "#B2182B",
    "globalmood40": "#4DAF4A",
}
CORPUS_GROUP_MARKERS = {
    "ballroom":     "o",
    "benchmark100": "s",
    "mirex":        "D",
    "globalmood40": "^",
}

MIXED_CORPUS_LABELS = {
    "asap":        "ASAP",
    "beatles":     "Beatles",
    "candombe":    "Candombe",
    "carnatic":    "Carnatic",
    "cretan":      "Cretan",
    "groove_midi": "Groove",
    "gtzan":       "GTZAN",
    "guitarset":   "GuitarSet",
    "hainsworth":  "Hainsworth",
    "harmonix":    "Harmonix",
    "rwc":         "RWC",
    "tapcorrect":  "TapCorrect",
    "turkish":     "Turkish",
}

# Pipeline analysis window for click rendering and ISTC (matches §3.2 / §3.4)
W0, W1 = 10.0, 25.0
KW_MS = 80.0
KDE_DT = 0.005


# ── ISMIR-compatible figure style ────────────────────────────────────────────

def apply_style() -> None:
    """Publication-quality matplotlib defaults for ISMIR two-column figures."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset":  "stix",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "demibold",
        "axes.labelsize":    9,
        "axes.labelweight":  "regular",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth":   1.2,
        "patch.linewidth":   0.6,
    })


# ── Lightweight Gaussian KDE on a regular grid ───────────────────────────────

def gaussian_kde(taps_s, t0: float, t1: float,
                 kw_ms: float = KW_MS, dt: float = KDE_DT):
    """Compute a Gaussian KDE over a regular time grid from raw tap times (s)."""
    kw_s = kw_ms / 1000.0
    x = np.arange(t0, t1 + 1e-12, dt, dtype=float)
    taps = np.asarray(taps_s, dtype=float)
    taps = taps[np.isfinite(taps)]
    taps = taps[(taps >= t0) & (taps <= t1)]
    if taps.size == 0:
        return x, np.zeros_like(x)
    dx = (x[:, None] - taps[None, :]) / kw_s
    y = np.exp(-0.5 * dx * dx) / (kw_s * math.sqrt(2 * math.pi))
    return x, y.sum(axis=1)
