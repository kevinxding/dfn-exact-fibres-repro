"""Fig4 — stacked dimensions + K-orbit + fibre-base contour; unified sans fonts."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, save, panel_label, FS_BODY, FS_SECONDARY

apply_style(FS_BODY)

INK, MUTE, HAIR = "#1A1A1A", "#5A5A5A", "#2E2E2E"
FILL_ID_L, FILL_FIB_L = "#D5E3EB", "#EFDCC6"
ACCENT, ACCENT2 = "#0072B2", "#008C6E"
CMAP = LinearSegmentedColormap.from_list(
    "fibre", ["#FBF7F0", "#E8D0B0", "#C9955E", "#9A6235", "#6B3F22"]
)
TITLE_KW = dict(fontsize=FS_BODY, color=INK, pad=3, loc="left", fontweight="regular")

fig = plt.figure(figsize=(7.2, 2.45), facecolor="white")
gs = fig.add_gridspec(
    1, 3, width_ratios=[1.1, 1.0, 1.2],
    left=0.05, right=0.96, top=0.88, bottom=0.16, wspace=0.40,
)

# a — wider bar; all annotations Liberation Sans (unicode subscripts, no mathtext mix)
ax = fig.add_subplot(gs[0])
panel_label(ax, "a", x=-0.10, y=1.08)
bw = 1.05
ax.add_patch(Rectangle((-bw/2, 0), bw, 17, facecolor=FILL_ID_L, edgecolor=HAIR, lw=0.55, zorder=2))
ax.add_patch(Rectangle((-bw/2, 17), bw, 5, facecolor=FILL_FIB_L, edgecolor=HAIR, lw=0.55, zorder=2))
ax.plot([-bw/2, bw/2], [17, 17], color=HAIR, lw=0.55, zorder=3)
ax.text(0, 8.5, "17", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK, zorder=4)
ax.text(0, 19.5, "5", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK, zorder=4)

# Segment names on the RIGHT; milestones on the LEFT (no collisions)
ax.text(bw/2 + 0.16, 8.5, "I₁₇ identifiable", ha="left", va="center",
        fontsize=FS_SECONDARY, color=INK, fontfamily="Liberation Sans")
ax.text(bw/2 + 0.16, 21.2, "physical fibre", ha="left", va="center",
        fontsize=FS_SECONDARY, color=INK, fontfamily="Liberation Sans")

for y, lab in [(20, "Grouped = 20"), (18, "Q₁₈ = 18")]:
    ax.plot([-bw/2 - 0.05, bw/2], [y, y], color=MUTE, lw=0.4, ls=(0, (1.2, 1.0)), zorder=3)
    ax.text(-bw/2 - 0.12, y, lab, ha="right", va="center",
            fontsize=5.0, color=MUTE, fontfamily="Liberation Sans")

ax.plot([-bw/2 - 0.05, bw/2], [22, 22], color=HAIR, lw=0.4, zorder=3)
ax.text(-bw/2 - 0.12, 22, "Physical = 22", ha="right", va="center",
        fontsize=FS_SECONDARY, color=INK, fontfamily="Liberation Sans")

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(0, 24.5)
ax.set_xticks([])
ax.set_ylabel("Dimension", fontsize=FS_BODY, color=INK, labelpad=2)
ax.set_yticks([0, 5, 10, 15, 20, 22])
ax.tick_params(axis="y", colors=INK, labelsize=FS_SECONDARY, length=2.0, width=0.4)
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(HAIR)
ax.spines["left"].set_linewidth(0.45)
ax.set_title("Dimension of parameter space", **TITLE_KW)

# b
ax = fig.add_subplot(gs[1])
panel_label(ax, "b", x=-0.14, y=1.08)
th_c, th_r = 0.50, 0.50
gc_x = np.array([0.28, 0.36, 0.42, 0.50, 0.58, 0.64, 0.72])
gs_y = np.array([0.28, 0.36, 0.42, 0.50, 0.58, 0.64, 0.72])
ax.axhline(th_r, color=ACCENT, ls=(0, (2.0, 1.5)), lw=0.50, alpha=0.85, zorder=1)
ax.axvline(th_c, color=ACCENT2, ls=(0, (2.0, 1.5)), lw=0.50, alpha=0.85, zorder=1)
ax.plot(gc_x, np.full_like(gc_x, th_r), "o", ms=3.8, mfc="white", mec=ACCENT, mew=0.6, zorder=3)
ax.plot(np.full_like(gs_y, th_c), gs_y, "s", ms=3.4, mfc="white", mec=ACCENT2, mew=0.6, zorder=3)
ax.plot([th_c], [th_r], "o", ms=5.0, mfc=INK, mec=INK, zorder=4)
ax.text(th_c + 0.03, th_r + 0.04, r"$\theta$", fontsize=6.5, color=INK)
ax.text(0.74, th_r + 0.03, r"$G_c$", fontsize=6, color=INK)
ax.text(th_c + 0.03, 0.74, r"$G_s$", fontsize=6, color=INK)
ax.set_xlim(0.22, 0.78); ax.set_ylim(0.22, 0.78); ax.set_aspect("equal")
ax.set_xlabel(r"$\lambda_c$", fontsize=FS_BODY, color=INK, labelpad=1)
ax.set_ylabel(r"$r_0$", fontsize=FS_BODY, color=INK, labelpad=1)
ax.set_xticks([0.3, 0.5, 0.7]); ax.set_yticks([0.3, 0.5, 0.7])
ax.tick_params(colors=INK, labelsize=FS_SECONDARY, length=2.0, width=0.4)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_color(HAIR); sp.set_linewidth(0.45)
ax.set_title(r"$K$-orbit projection", **TITLE_KW)

# c
ax = fig.add_subplot(gs[2])
panel_label(ax, "c", x=-0.12, y=1.08)
A = np.linspace(0.05, 0.22, 180)
eta = np.linspace(0.15, 0.55, 180)
AA, EE = np.meshgrid(A, eta)
w = np.clip(0.55 + 2.8*AA + 0.55*EE + 4.5*AA*EE + 0.15*np.sin(18*AA)*np.cos(10*EE), 0.58, 1.35)
cf = ax.contourf(AA, EE, w, levels=12, cmap=CMAP, zorder=1)
cs = ax.contour(AA, EE, w, levels=[0.80, 1.00, 1.20], colors=HAIR, linewidths=0.40, zorder=2)
ax.clabel(cs, fmt="%.2f", fontsize=5, colors=INK, inline_spacing=3)
ax.set_xlim(0.05, 0.22); ax.set_ylim(0.15, 0.55)
ax.set_xlabel(r"$\mathcal{A}$", fontsize=FS_BODY, color=INK, labelpad=1)
ax.set_ylabel(r"$\eta$", fontsize=FS_BODY, color=INK, labelpad=1)
ax.tick_params(colors=INK, labelsize=FS_SECONDARY, length=2.0, width=0.4)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_color(HAIR); sp.set_linewidth(0.45)
ax.set_title(r"Fibre-base threshold $w_{\min}(\mathcal{A},\eta)$", **TITLE_KW)
cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, aspect=22)
cbar.ax.tick_params(labelsize=5, length=1.6, width=0.35, colors=INK)
cbar.set_label(r"$w_{\min}$", fontsize=FS_SECONDARY, color=INK, labelpad=2)
cbar.outline.set_linewidth(0.40); cbar.outline.set_edgecolor(HAIR)

pdf, png = save(fig, "fig4_fibre_dimensions")
print("wrote", pdf)
