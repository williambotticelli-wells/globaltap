#!/usr/bin/env python3
"""
Figure 2 (in the paper) — four-panel pipeline data view for the example
Ballroom excerpt ``Media-103601``.

This is the self-contained companion-repo version of the figure.  The
script loads everything it needs from a single included NPZ
(``data/example_stem_panels.npz``); no external data tree, audio file, or
``${PROJECT_ROOT}`` is required.  The file was generated with
``paper/figures/build_data_panels_bundle.py`` from the same raw inputs
that produced the figure printed in the paper.

Panels:
    (i)   Audio spectrogram (magma, 0-6 kHz)
    (ii)  Per-participant tap raster
    (iii) Pooled KDE + consensus peaks
    (iv)  Optimized beat grid vs. reference annotation, with audio
          envelope backdrop

Usage:
    python figures/data_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fig_utils import apply_style, gaussian_kde, METHOD_COLORS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "data" / "example_stem_panels.npz"
OUT = Path(__file__).parent / "data_panels.pdf"


def _split_taps(offsets: np.ndarray, taps: np.ndarray) -> list[np.ndarray]:
    return [taps[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def main() -> None:
    if not BUNDLE.exists():
        raise SystemExit(
            f"Missing example data: {BUNDLE}\n"
            "Regenerate it from paper/figures/build_data_panels_bundle.py "
            "or restore the file from the published companion repo."
        )

    apply_style()
    with np.load(BUNDLE, allow_pickle=False) as bundle:
        view_t0 = float(bundle["view_t0"])
        view_t1 = float(bundle["view_t1"])
        n_pids = int(bundle["n_participants"])
        tap_offsets = bundle["tap_offsets"]
        tap_times = bundle["tap_times"]
        kde_peaks = bundle["kde_peaks"]
        crowd_beats = bundle["crowd_beats"]
        ref_beats = bundle["ref_beats"]
        spec = bundle["spec"]
        spec_fmax_khz = float(bundle["spec_fmax_khz"])
        env_t = bundle["env_t"]
        env_y = bundle["env_y"]

    pid_taps = _split_taps(tap_offsets, tap_times)
    all_taps = np.concatenate(pid_taps) if pid_taps else np.array([])
    tg, yk = gaussian_kde(all_taps, 0, 30)
    yk_norm = yk / (np.max(yk) + 1e-12)
    mask_k = (tg >= view_t0) & (tg <= view_t1)

    fig = plt.figure(figsize=(5.4, 3.8))
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.85, 1.6, 0.9, 0.9],
        hspace=0.36,
        left=0.105, right=0.985, top=0.97, bottom=0.105,
    )

    # (i) Audio spectrogram
    ax_spec = fig.add_subplot(gs[0])
    ax_spec.imshow(
        spec, aspect="auto", origin="lower",
        extent=[view_t0, view_t1, 0.0, spec_fmax_khz],
        cmap="magma", interpolation="bilinear",
    )
    ax_spec.set_xlim(view_t0, view_t1)
    ax_spec.set_ylim(0.0, spec_fmax_khz)
    ax_spec.set_yticks([0, 2, 4, 6])
    ax_spec.set_ylabel("Freq.\n(kHz)", fontsize=8)
    ax_spec.set_title("(i)  Audio spectrogram", loc="left",
                      fontweight="bold", fontsize=9.5)
    ax_spec.tick_params(labelbottom=False, labelsize=7.5)

    # (ii) Tap raster
    ax_raster = fig.add_subplot(gs[1])
    for i, taps in enumerate(pid_taps):
        t_v = taps[(taps >= view_t0) & (taps <= view_t1)]
        ax_raster.vlines(t_v, i + 0.05, i + 0.95, linewidth=1.0,
                         color="#1a1a1a", alpha=0.95)
    ax_raster.set_ylabel(f"Participant\n(N={n_pids})", fontsize=8)
    ax_raster.set_yticks([0, n_pids])
    ax_raster.set_xlim(view_t0, view_t1)
    ax_raster.set_ylim(0, n_pids)
    ax_raster.set_title("(ii)  Per-participant tap raster",
                        loc="left", fontweight="bold", fontsize=9.5)
    ax_raster.tick_params(labelbottom=False, labelsize=7.5)

    # (iii) Pooled KDE + consensus peaks
    ax_kde = fig.add_subplot(gs[2])
    ax_kde.fill_between(tg[mask_k], yk_norm[mask_k], alpha=0.22,
                        color=METHOD_COLORS["Opt"], zorder=1)
    ax_kde.plot(tg[mask_k], yk_norm[mask_k],
                color=METHOD_COLORS["Opt"], linewidth=1.5, zorder=2)
    peaks_v = kde_peaks[(kde_peaks >= view_t0) & (kde_peaks <= view_t1)]
    peak_h = np.interp(peaks_v, tg, yk_norm)
    ax_kde.scatter(peaks_v, peak_h, color=METHOD_COLORS["Opt"], s=32,
                   zorder=5, edgecolors="white", linewidths=0.6)
    ax_kde.set_ylabel("KDE\ndensity", fontsize=8)
    ax_kde.set_xlim(view_t0, view_t1)
    ax_kde.set_ylim(0, max(1.0, yk_norm[mask_k].max() * 1.05))
    ax_kde.set_title("(iii)  Pooled KDE + consensus peaks",
                     loc="left", fontweight="bold", fontsize=9.5)
    ax_kde.tick_params(labelbottom=False, labelsize=7.5)

    # (iv) Optimized beat grid + waveform envelope
    ax_beats = fig.add_subplot(gs[3])
    ax_beats.set_xlim(view_t0, view_t1)
    ax_beats.set_ylim(0, 1.0)
    ax_beats.fill_between(env_t, 0, env_y * 0.75,
                          color="#888888", alpha=0.30,
                          linewidth=0, zorder=0)
    opt_v = crowd_beats[(crowd_beats >= view_t0) & (crowd_beats <= view_t1)]
    ref_v = ref_beats[(ref_beats >= view_t0) & (ref_beats <= view_t1)]
    for t in opt_v:
        ax_beats.axvline(t, color=METHOD_COLORS["Opt"], linewidth=2.0,
                         alpha=0.85, ymin=0.0, ymax=0.92, zorder=3)
    for t in ref_v:
        ax_beats.axvline(t, color=METHOD_COLORS["GT"], linewidth=1.4,
                         alpha=0.6, linestyle="--", ymin=0.0,
                         ymax=0.92, zorder=3)

    ax_beats.set_xlabel("Time (s)", fontsize=8.5)
    ax_beats.set_ylabel("Audio\nenvelope", fontsize=8)
    ax_beats.set_yticks([])
    ax_beats.set_title("(iv)  Optimized beat grid vs. reference",
                       loc="left", fontweight="bold", fontsize=9.5)
    ax_beats.tick_params(labelsize=7.5)

    plt.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(OUT.with_suffix(".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.01)
    print(f"Saved -> {OUT}")
    print(f"Saved -> {OUT.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    main()
