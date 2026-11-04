"""Fig 2: Exhaustive proof-branch flowchart."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import apply_style, save, COLORS

apply_style()

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13.2)
ax.axis("off")

def box(x, y, w, h, text, fc, ec=None, fs=7.5, bold=False):
    ec = ec or COLORS["black"]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, linewidth=0.95))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), linespacing=1.3)

def arrow(x1, y1, x2, y2, color=None):
    color = color or COLORS["black"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.05,
                                mutation_scale=9))

box(5.5, 11.6, 9, 1.15, r"Original current–voltage germ  $\mathbf{F}_\theta$",
    "#F7F7F7", bold=True, fs=8.5)
arrow(10, 11.6, 10, 10.95)

box(4.5, 9.8, 11, 1.1,
    "Recover unordered electrical cores\n"
    r"$\{\mathcal{C}_n,\mathcal{C}_p\}$ and $R_{\mathrm{ser}}$ (all ranks)",
    "#EEF6FB", COLORS["blue"], fs=7.8)
arrow(10, 9.8, 10, 9.15)

box(6.5, 8.1, 7, 0.95, "Cores distinct?", "#FFF8EE", COLORS["orange"], fs=8.5, bold=True)
arrow(6.5, 8.55, 3.5, 7.65)
arrow(13.5, 8.55, 16.5, 7.65)
ax.text(4.55, 7.95, "yes", fontsize=7, color=COLORS["green"])
ax.text(14.95, 7.95, "no", fontsize=7, color=COLORS["vermillion"])

box(0.4, 5.95, 6.2, 1.55,
    "Distinct-core inverse\nlabels, particles, sources,\nthen electrolyte",
    "#E8F5E9", COLORS["green"], fs=7.5)

box(13.2, 6.6, 6.4, 0.95, "Prepared OCP slopes",
    "#FFF8EE", COLORS["orange"], fs=8, bold=True)
arrow(16.4, 6.6, 13.5, 5.6)
arrow(16.4, 6.6, 18.5, 5.6)
ax.text(13.6, 6.05, r"$u_n=u_p=0$", fontsize=6.5, color=COLORS["gray"])
ax.text(17.3, 6.05, r"some $u_m\neq 0$", fontsize=6.5, color=COLORS["gray"])

box(10.5, 4.2, 6.0, 1.35,
    "Flat-preparation theorem\ncomplementary / mirror\ncases resolved",
    "#E3F2FD", COLORS["blue"], fs=7.3)
box(16.7, 4.2, 3.0, 1.35, "Nonzero-\nslope split",
    "#FCE4EC", COLORS["purple"], fs=7.3)

arrow(18.2, 4.2, 15.0, 3.4)
arrow(18.2, 4.2, 18.2, 3.4)
box(12.0, 2.2, 5.7, 1.15,
    r"$\Delta_{\mathrm{prep}}\neq 0$ or unequal $\kappa$",
    "#F3E5F5", COLORS["purple"], fs=7)
box(17.9, 2.2, 1.85, 1.15, "tied\ncollision",
    "#FFEBEE", COLORS["vermillion"], fs=7)

arrow(3.5, 5.95, 6.8, 1.6)
arrow(13.5, 4.2, 9.8, 1.6)
arrow(14.9, 2.2, 10.8, 1.6)
arrow(18.2, 2.2, 13.2, 1.6)

box(3.5, 0.25, 13.0, 1.3,
    r"Complete $Q_{18}$  $\Rightarrow$  abstract fibre $K\theta$;"
    "\n"
    r"realizable $\mathcal{I}_{17}$; physical 5-D chart",
    "#FFFDE7", COLORS["orange"], fs=8, bold=True)

ax.text(0.3, 12.85, r"Exhaustive case division for every fixed $C^2$ preparation",
        fontsize=8, color=COLORS["gray"])

pdf, png = save(fig, "fig2_proof_flow")
print(pdf, png)
