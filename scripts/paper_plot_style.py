"""Shared typography and export style for manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


BASE_SIZE = 8.5
AXES_LABEL_SIZE = 8.5
TITLE_SIZE = 9.0
TICK_SIZE = 7.5
LEGEND_SIZE = 7.2
ANNOTATION_SIZE = 7.2
FULL_WIDTH_IN = 7.2


def configure_paper_style() -> None:
    """Apply one role-based font scale to every manuscript figure."""
    plt.rcParams.update({
        "font.size": BASE_SIZE,
        "axes.labelsize": AXES_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.35,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
