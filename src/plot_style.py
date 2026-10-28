#!/usr/bin/env python3
"""
Shared plotting style (non-interactive backend and color palette).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

BLACK = "#111111"
DARKBLUE = "#0B3C6D"
MAINBLUE = "#1F6FB2"
DARKRED = "#8A1C1C"
SOFTBLUE = "#D8E8F6"
SOFTGRAY = "#8B96A3"
GRIDGRAY = "#D5DDE6"

matplotlib.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "font.family": "serif",
        "font.serif": [
            "Palatino Linotype",
            "Palatino",
            "Book Antiqua",
            "URW Palladio L",
            "TeX Gyre Pagella",
            "serif",
        ],
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.linewidth": 0.95,
        "axes.titlesize": 12,
        "axes.titlepad": 8.0,
        "axes.labelsize": 11,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axes(ax: object) -> None:
    ax.set_axisbelow(True)
    # Grid removed per user request (all figures)
    ax.tick_params(direction="out", width=0.9, length=4.5, pad=3)
    for spine in ax.spines.values():
        spine.set_color(BLACK)
        spine.set_linewidth(0.95)
