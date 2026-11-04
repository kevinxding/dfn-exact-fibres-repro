"""Fig 1: P2D cell schematic + current–voltage port map."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc
from matplotlib.lines import Line2D
from style import apply_style, save, panel_label, COLORS

apply_style()

fig = plt.figure(figsize=(7.2, 3.35))  # ~ double-column width inches
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

# ---- panel a: sandwich cell ----
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.set_aspect("equal")
ax.axis("off")
panel_label(ax, "a", x=-0.02, y=1.02)

# Regions
regions = [
    (0.6, 2.2, COLORS["fill_n"], "Negative\nelectrode"),
    (2.8, 1.4, COLORS["fill_s"], "Separator"),
    (4.2, 2.4, COLORS["fill_p"], "Positive\nelectrode"),
]
y0, h = 1.35, 3.4
for x, w, c, lab in regions:
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="square,pad=0",
                                facecolor=c, edgecolor=COLORS["black"],
                                linewidth=0.9, alpha=0.55, mutation_aspect=0.1))
    ax.text(x + w/2, y0 + h/2, lab, ha="center", va="center", fontsize=8,
            color=COLORS["black"])

# Current collectors
ax.add_patch(Rectangle((0.15, y0), 0.45, h, facecolor="#333333", edgecolor="none"))
ax.add_patch(Rectangle((6.6, y0), 0.45, h, facecolor="#333333", edgecolor="none"))
ax.text(0.38, y0 - 0.35, "Cu", ha="center", fontsize=7, color=COLORS["gray"])
ax.text(6.82, y0 - 0.35, "Al", ha="center", fontsize=7, color=COLORS["gray"])

# Particles (schematic spheres)
rng = np.random.default_rng(3)
for xbase, w, n in [(0.6, 2.2, 7), (4.2, 2.4, 8)]:
    for _ in range(n):
        cx = xbase + 0.25 + rng.random() * (w - 0.5)
        cy = y0 + 0.4 + rng.random() * (h - 0.8)
        r = 0.14 + 0.06 * rng.random()
        ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=COLORS["blue"],
                            linewidth=0.7, alpha=0.85, zorder=3))

# Electrolyte continuum hint
ax.annotate("", xy=(6.4, y0 + h + 0.35), xytext=(0.7, y0 + h + 0.35),
            arrowprops=dict(arrowstyle="<->", color=COLORS["electrolyte"], lw=1.1))
ax.text(3.55, y0 + h + 0.55, r"electrolyte $c_e(x,t)$", ha="center", fontsize=7.5,
        color=COLORS["electrolyte"])

# Thickness labels
ax.annotate("", xy=(2.8, 0.85), xytext=(0.6, 0.85),
            arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=0.8))
ax.text(1.7, 0.55, r"$L_n$", ha="center", fontsize=8)
ax.annotate("", xy=(4.2, 0.85), xytext=(2.8, 0.85),
            arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=0.8))
ax.text(3.5, 0.55, r"$L_s$", ha="center", fontsize=8)
ax.annotate("", xy=(6.6, 0.85), xytext=(4.2, 0.85),
            arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=0.8))
ax.text(5.4, 0.55, r"$L_p$", ha="center", fontsize=8)

ax.text(3.55, 5.85, "Pseudo-two-dimensional (P2D / DFN) continuum",
        ha="center", fontsize=8.5, color=COLORS["black"])

# ---- panel b: port map ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 6.2)
ax2.axis("off")
panel_label(ax2, "b", x=-0.02, y=1.02)

# Parameter box
ax2.add_patch(FancyBboxPatch((0.4, 4.2), 5.1, 1.35, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#F7F7F7", edgecolor=COLORS["black"], linewidth=0.9))
ax2.text(2.95, 5.2, r"Grouped parameters $\theta\in\Theta_{\mathrm{abs}}$",
         ha="center", va="center", fontsize=8)
ax2.text(2.95, 4.55, r"$Q_{18}(\theta)$ invariants  /  gauges $K$",
         ha="center", va="center", fontsize=7.5, color=COLORS["gray"])

# Arrow down
ax2.annotate("", xy=(2.95, 3.55), xytext=(2.95, 4.15),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["black"], lw=1.2,
                             mutation_scale=10))
ax2.text(3.35, 3.85, "DFN PDE–DAE", fontsize=7, color=COLORS["gray"])

# Cell operator box
ax2.add_patch(FancyBboxPatch((0.7, 2.35), 4.5, 1.15, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#EEF6FB", edgecolor=COLORS["blue"], linewidth=1.0))
ax2.text(2.95, 2.92, r"Forward map $\mathcal{F}_\theta$",
         ha="center", fontsize=9)
ax2.text(2.95, 2.55, r"$u\mapsto V$ on $[0,T_*]$",
         ha="center", fontsize=7.5, color=COLORS["gray"])

# Input / output
ax2.annotate("", xy=(0.7, 2.9), xytext=(0.15, 2.9),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["vermillion"], lw=1.3,
                             mutation_scale=10))
ax2.text(0.05, 3.25, r"$u(t)$", fontsize=9, color=COLORS["vermillion"],
         ha="left", fontweight="bold")
ax2.text(0.05, 2.55, "current\n(normalized)", fontsize=6.5, color=COLORS["gray"], ha="left")

ax2.annotate("", xy=(5.85, 2.9), xytext=(5.2, 2.9),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["vermillion"], lw=1.3,
                             mutation_scale=10))
ax2.text(5.4, 3.25, r"$V(t)$", fontsize=9, color=COLORS["vermillion"],
         ha="left", fontweight="bold")
ax2.text(5.25, 2.45, "terminal\nvoltage", fontsize=6.5, color=COLORS["gray"], ha="left")

# Germ equality
ax2.add_patch(FancyBboxPatch((0.4, 0.35), 5.1, 1.55, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#FFF8EE", edgecolor=COLORS["orange"], linewidth=1.0))
ax2.text(2.95, 1.45, r"Structural question", ha="center", fontsize=8)
ax2.text(2.95, 0.95, r"$\mathbf{F}_{\tilde\theta}=\mathbf{F}_\theta$",
         ha="center", fontsize=10)
ax2.text(2.95, 0.55, r"$\Longleftrightarrow\ Q_{18}$ equal $\Longleftrightarrow\ \tilde\theta\in K\theta$",
         ha="center", fontsize=7.5, color=COLORS["gray"])

pdf, png = save(fig, "fig1_p2d_port")
print(pdf, png)
