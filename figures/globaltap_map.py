#!/usr/bin/env python3
"""
Figure 1B (paper) — GlobalTap benchmark world map.

Visualises the geographic scope of the benchmark:

    * Red dots:  source countries from which the 200 benchmark excerpts
                 were drawn (uniform marker size).
    * Blue dots: country panels of tapping participants
                 (sized by panel N, labeled with ISO + N).

Self-contained: reads ``data/country_panels.csv`` shipped in this
repository.  Country centroids are hand-tuned for label legibility.

Usage:
    python figures/globaltap_map.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent))
from fig_utils import apply_style

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTRY_CSV = REPO_ROOT / "data" / "country_panels.csv"
OUT = Path(__file__).parent / "globaltap_map.pdf"

# ── Country centroids (lon, lat) — hand-picked for label legibility ────────
COUNTRY_LL = {
    "US": (-98.5, 39.8),  "KR": (127.8, 36.5),  "FR": (2.4, 46.6),
    "MX": (-102.5, 23.6), "EG": (30.0, 26.8),   "AU": (133.8, -25.3),
    "BR": (-51.9, -10.5), "JP": (138.3, 36.2),  "PL": (19.1, 52.1),
    "ZA": (24.7, -29.0),
    "AE": (54.4, 24.0),   "PA": (-80.0, 8.5),   "IT": (12.5, 42.8),
    "KE": (37.9, 0.0),    "LU": (6.1, 49.8),    "NG": (8.7, 9.1),
    "NI": (-85.2, 12.9),  "NL": (5.3, 52.1),    "PE": (-75.0, -9.2),
    "IL": (34.9, 31.5),   "PT": (-8.2, 39.4),   "RO": (24.97, 45.94),
    "RS": (21.0, 44.0),   "RU": (95.0, 61.5),   "SE": (16.0, 62.0),
    "TR": (35.2, 39.0),   "TZ": (34.9, -6.4),   "UA": (31.2, 49.0),
    "IS": (-19.0, 64.9),  "IN": (78.9, 22.0),   "IE": (-8.2, 53.4),
    "DK": (10.0, 56.0),   "AT": (14.6, 47.5),   "BE": (4.5, 50.6),
    "BO": (-64.7, -16.3), "CA": (-106.0, 60.0), "CH": (8.2, 46.8),
    "CO": (-74.3, 4.6),   "CR": (-84.1, 9.7),   "DE": (10.4, 51.2),
    "CZ": (15.5, 49.8),   "ZW": (29.8, -19.0),  "HN": (-86.4, 14.7),
    "ID": (113.9, -2.5),  "EC": (-78.2, -1.4),  "EE": (25.0, 58.6),
    "HU": (19.5, 47.2),   "ES": (-3.7, 40.4),   "DO": (-70.6, 18.7),
    "GB": (-2.0, 54.5),   "GT": (-90.2, 15.5),  "UY": (-55.8, -32.5),
    "UG": (32.4, 1.4),    "NZ": (172.5, -41.5), "SA": (45.0, 24.0),
    "CL": (-71.5, -35.7), "PY": (-58.4, -23.4), "AR": (-65.0, -35.5),
    "FI": (26.0, 64.0),
}

LABEL_OFFSETS = {
    "US": (-2,  6,   "center", "bottom"),
    "MX": (-12, -1,  "right",  "center"),
    "BR": (8,   0,   "left",   "center"),
    "FR": (-9,  -2,  "right",  "center"),
    "PL": (5,   3,   "left",   "bottom"),
    "EG": (8,  -1,   "left",   "center"),
    "ZA": (8,  -2,   "left",   "center"),
    "JP": (8,   2,   "left",   "center"),
    "KR": (-7, -3,   "right",  "top"),
    "AU": (1,  -10,  "center", "top"),
}


def load_data():
    df = pd.read_csv(COUNTRY_CSV)
    panels = df[df["role"] == "panel"].set_index("iso2")["participants"].astype(int)
    sources = df[df["role"] == "source"]["iso2"].tolist()
    return panels, sources


def build_figure():
    apply_style()
    panel, src_iso = load_data()
    src_iso = sorted(set(src_iso) | set(panel.index))  # panels are also sources
    src_iso = [iso for iso in src_iso if iso in COUNTRY_LL]

    proj = ccrs.Robinson()
    geo = ccrs.PlateCarree()

    fig = plt.figure(figsize=(7.0, 2.7))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.set_extent([-160, 175, -55, 78], crs=geo)

    ax.add_feature(cfeature.LAND.with_scale("110m"),
                   facecolor="#f3f1ec", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"),
                   facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.35, edgecolor="#9aa0a6", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                   linewidth=0.25, edgecolor="#bdc1c6", zorder=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    panels = list(panel.index)
    pan_lon = [COUNTRY_LL[iso][0] for iso in panels]
    pan_lat = [COUNTRY_LL[iso][1] for iso in panels]
    pan_n = np.array([panel[iso] for iso in panels])
    pan_size = 25 + 0.35 * pan_n
    total_n = int(pan_n.sum())
    ax.scatter(
        pan_lon, pan_lat, s=pan_size,
        c="#1F4E96", edgecolor="white", linewidth=0.6,
        alpha=0.95, transform=geo, zorder=4,
        label=f"Tapping panel ({len(panels)} countries, N={total_n})",
    )

    s_lon = [COUNTRY_LL[iso][0] for iso in src_iso]
    s_lat = [COUNTRY_LL[iso][1] for iso in src_iso]
    ax.scatter(
        s_lon, s_lat, s=14,
        c="#C0392B", edgecolor="white", linewidth=0.3,
        alpha=0.95, transform=geo, zorder=6,
        label=f"Source country ({len(src_iso)})",
    )

    for iso, lon, lat, n in zip(panels, pan_lon, pan_lat, pan_n):
        dlon, dlat, ha, va = LABEL_OFFSETS.get(iso, (0, 0, "center", "center"))
        ax.text(
            lon + dlon, lat + dlat,
            f"{iso}\nN={n}",
            transform=geo, fontsize=6.8, fontweight="bold",
            ha=ha, va=va, color="#0c1e3d", zorder=7,
            linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#1F4E96", linewidth=0.5, alpha=0.92),
        )

    leg = ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.04),
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
        fontsize=7.2, handletextpad=0.4, borderpad=0.4,
        ncol=2, columnspacing=1.6,
    )
    leg.get_frame().set_linewidth(0.4)

    ax.set_title(
        f"GlobalTap benchmark: 200 excerpts from {len(src_iso)} source countries, "
        f"tapped by {total_n} listeners across {len(panels)} country panels",
        fontsize=8.5, pad=4, color="#222222",
    )

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.12)

    out_pdf = OUT
    out_png = OUT.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved \u2192 {out_pdf}")
    print(f"Saved \u2192 {out_png}")


if __name__ == "__main__":
    build_figure()
