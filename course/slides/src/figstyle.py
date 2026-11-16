"""Shared matplotlib style for slide figures: Imperial palette, Arial, sizes that read from the back row.

In a NN_slug_figures.py file:

    from figstyle import *
    fig, ax = new_fig()            # half-slide figure; new_fig(wide=True) for a full-width one
    ax.plot(x, y, color=BLUE)
    save(fig, "11_tradeoff")       # writes course/slides/figures/11_tradeoff.png

Every number plotted must be computed in the script (seeded) or copied from a cited source named in the figure.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
BLUE, VIOLET, PINK, RED, ORANGE, GREEN, NAVY, GREY, SMOKE = "#0000CD", "#7B68EE", "#C71585", "#FF0000", "#FF4500", "#006400", "#000080", "#6E6E6E", "#F5F5F5"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 16, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SMOKE, "figure.facecolor": SMOKE,
    "savefig.facecolor": SMOKE, "lines.linewidth": 2.5, "mathtext.fontset": "cm",
})


def new_fig(wide=False, nrows=1, ncols=1, height=5.0, **kw):
    return plt.subplots(nrows, ncols, figsize=(12.4 if wide else 6.0, height), **kw)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
