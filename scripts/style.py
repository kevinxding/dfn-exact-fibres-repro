"""Shared plotting style for manuscript figures (Liberation Sans + muted palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
    "lightgray": "#B3B3B3",
    "fill_n": "#A6CEE3",
    "fill_s": "#F7F0D4",
    "fill_p": "#FDBF6F",
    "electrolyte": "#1B9E77",
    "port": "#D55E00",
    "bg": "#FFFFFF",
    "panel": "#F5F5F5",
}

# Package root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PNG = FIG / "png"
REPORTS = ROOT / "reports"

FS_BODY = 6.0
FS_PANEL = 8.0
FS_SECONDARY = 5.5


def apply_style(fontsize=FS_BODY):
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize - 0.5,
            "ytick.labelsize": fontsize - 0.5,
            "legend.fontsize": fontsize - 0.5,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.35,
            "legend.columnspacing": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.axisbelow": True,
            "lines.linewidth": 0.9,
            "axes.unicode_minus": False,
        }
    )


def save(fig, stem: str):
    FIG.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    pdf = FIG / f"{stem}.pdf"
    png = PNG / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def panel_label(ax, label, x=-0.12, y=1.08, fontsize=FS_PANEL):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
        fontfamily="sans-serif",
        color="#1A1A1A",
    )
