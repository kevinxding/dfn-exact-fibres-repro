"""Plotting for Fig. 6 (synthetic-data inversion); called by fig6_inversion.py."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from style import apply_style, save, panel_label, FS_BODY, FS_SECONDARY
from dfn_grouped import NAMES, Q18

INK, MUTE, HAIR = "#1A1A1A", "#5A5A5A", "#2E2E2E"
STEEL, CLAY, SLATE = "#3D5A6C", "#A67C52", "#6B7C8A"
STEEL_L, CLAY_L = "#D5E3EB", "#EFDCC6"

Q_LABELS = [r"$T_n$", r"$T_p$", r"$C_n$", r"$C_p$", r"$\tau_k^n$", r"$\tau_k^p$",
            r"$\tau_n$", r"$\tau_s$", r"$\tau_p$", r"$\kappa^n$", r"$\kappa^p$",
            r"$\sigma^n$", r"$\sigma^p$", r"$\chi$", r"$\zeta_n$", r"$\zeta_s$",
            r"$\zeta_p$", r"$R_{\mathrm{ser}}$"]
G_LABELS = [r"$\gamma$", r"$\nu_e^n$", r"$\nu_e^s$", r"$\nu_e^p$", r"$r_f$", r"$\kappa^s$"]
G_IDX = [NAMES.index(n) for n in ("gamma", "nu_n", "nu_s", "nu_p", "r_f", "kappa_s")]
FLOOR = 1e-8


def style_ax(ax):
    ax.tick_params(labelsize=FS_SECONDARY, length=2.0, width=0.4, colors=INK)
    for s in ax.spines.values():
        s.set_color(HAIR)
        s.set_linewidth(0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def logfmt(v, _):
    return r"$10^{%d}$" % int(round(np.log10(v))) if v > 0 else ""


def plot(d):
    apply_style(FS_BODY)
    theta = d["theta"]
    qt = Q18(theta)
    S = d["starts"]
    fig = plt.figure(figsize=(3.5, 3.25), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.2], left=0.13, right=0.985,
                          top=0.95, bottom=0.10, hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(3)

    # ---- (a) relative error of every coordinate ----
    nq, ng = 18, 6
    xq = np.arange(nq)
    xg = nq + 0.8 + np.arange(ng)
    ax_a.axvspan(nq - 0.4 + 0.0, xg[-1] + 0.6, color="#F4EEE6", lw=0, zorder=0)
    for case, off, filled in (("clean", -0.17, True), ("noisy", 0.17, False)):
        E = d["est_" + case]
        Q = np.array([Q18(e) for e in E])
        rq = np.maximum(np.abs(Q / qt - 1.0), FLOOR)
        rg = np.maximum(np.abs(E[:, G_IDX] / theta[G_IDX] - 1.0), FLOOR)
        for x0, R, col in ((xq, rq, STEEL), (xg, rg, CLAY)):
            for j in range(R.shape[1]):
                jit = rng.uniform(-0.07, 0.07, R.shape[0])
                kw = dict(s=3.2, linewidths=0.45, zorder=3)
                if filled:
                    ax_a.scatter(x0[j] + off + jit, R[:, j], c=col, edgecolors="none",
                                 alpha=0.75, **kw)
                else:
                    ax_a.scatter(x0[j] + off + jit, R[:, j], facecolors="white",
                                 edgecolors=col, **kw)
    ax_a.set_yscale("log")
    ax_a.set_ylim(FLOOR * 0.5, 3e2)
    ax_a.yaxis.set_major_locator(FixedLocator([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]))
    ax_a.yaxis.set_major_formatter(FuncFormatter(logfmt))
    ax_a.yaxis.set_minor_locator(NullLocator())
    ax_a.set_xticks(np.concatenate([xq, xg]))
    ax_a.set_xticklabels(Q_LABELS + G_LABELS, fontsize=4.6)
    ax_a.tick_params(axis="x", length=1.5, pad=1.5)
    ax_a.set_xlim(-0.6, xg[-1] + 0.6)
    ax_a.set_ylabel(r"$|\hat\vartheta/\vartheta^\ast-1|$", color=INK)
    ax_a.text(8.5, 9e1, r"$Q_{18}$ coordinates", ha="center", va="center",
              fontsize=FS_SECONDARY, color=STEEL)
    ax_a.text(xg.mean(), 9e1, "gauge coordinates", ha="center", va="center",
              fontsize=FS_SECONDARY, color=CLAY)
    style_ax(ax_a)
    panel_label(ax_a, "a", x=-0.115, y=1.0)
    hd = [Line2D([0], [0], marker="o", ls="none", mfc=SLATE, mec="none", ms=2.6,
                 label="noise-free"),
          Line2D([0], [0], marker="o", ls="none", mfc="white", mec=SLATE, mew=0.5,
                 ms=2.6, label="0.1 mV noise")]
    ax_a.legend(handles=hd, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                fontsize=FS_SECONDARY, handletextpad=0.2, columnspacing=0.8,
                borderaxespad=0.0)

    # ---- (b) G_c coordinate: estimate tracks the start ----
    ig = NAMES.index("gamma")
    l0 = S[:, ig] / theta[ig]
    ax_b.plot([0.07, 14], [0.07, 14], color=HAIR, lw=0.45, ls=(0, (2.0, 1.4)), zorder=1)
    ax_b.scatter(l0, d["est_clean"][:, ig] / theta[ig], s=6, c=CLAY, edgecolors="none",
                 alpha=0.85, zorder=3)
    ax_b.scatter(l0, d["est_noisy"][:, ig] / theta[ig], s=11, facecolors="none",
                 edgecolors=STEEL, linewidths=0.45, zorder=2)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    for ax_ in (ax_b,):
        ax_.set_xlim(0.07, 14)
        ax_.set_ylim(0.07, 14)
        for axis in (ax_.xaxis, ax_.yaxis):
            axis.set_major_locator(FixedLocator([0.1, 1, 10]))
            axis.set_major_formatter(FuncFormatter(
                lambda v, _: {0.1: "0.1", 1: "1", 10: "10"}.get(round(v, 1), "")))
            axis.set_minor_locator(NullLocator())
    ax_b.set_xlabel(r"start $\lambda_c=\gamma_0/\gamma^\ast$", color=INK, labelpad=1.5)
    ax_b.set_ylabel(r"estimate $\hat\gamma/\gamma^\ast$", color=INK, labelpad=1.5)
    ax_b.set_title(r"$G_c$ orbit", fontsize=FS_BODY, color=INK, pad=3, loc="left")
    style_ax(ax_b)
    panel_label(ax_b, "b", x=-0.36, y=1.0)

    # ---- (c) G_s orbit in the (r_f, 1/kappa^s) plane ----
    ir, ik = NAMES.index("r_f"), NAMES.index("kappa_s")
    R = theta[ir] + 1.0 / theta[ik]
    ax_c.plot([0, R * 1e3], [R * 1e3, 0], color=HAIR, lw=0.55, zorder=1)
    for k in range(S.shape[0]):
        ax_c.plot([S[k, ir] * 1e3, d["est_clean"][k, ir] * 1e3],
                  [1e3 / S[k, ik], 1e3 / d["est_clean"][k, ik]],
                  color="#C9C9C9", lw=0.3, zorder=1)
    ax_c.scatter(S[:, ir] * 1e3, 1e3 / S[:, ik], s=5, facecolors="white",
                 edgecolors="#8C8C8C", linewidths=0.4, zorder=2)
    ax_c.scatter(d["est_clean"][:, ir] * 1e3, 1e3 / d["est_clean"][:, ik], s=6, c=CLAY,
                 edgecolors="none", alpha=0.9, zorder=3)
    ax_c.scatter(theta[ir] * 1e3, 1e3 / theta[ik], s=16, marker="*", c=INK,
                 edgecolors="none", zorder=4)
    lim = max(S[:, ir].max(), (1 / S[:, ik]).max()) * 1e3 * 1.05
    ax_c.set_xlim(0, lim)
    ax_c.set_ylim(0, lim)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xlabel(r"$r_f$ (mV)", color=INK, labelpad=1.5)
    ax_c.set_ylabel(r"$1/\kappa^s$ (mV)", color=INK, labelpad=1.5)
    ax_c.set_title(r"$G_s$ orbit", fontsize=FS_BODY, color=INK, pad=3, loc="left")
    xl, yl = R * 1e3 * 0.62, R * 1e3 * 0.38
    ax_c.annotate(r"$r_f+1/\kappa^s=R_{\mathrm{ser}}^\ast$",
                  xy=(R * 1e3 * 0.47, R * 1e3 * 0.53), xytext=(R * 1e3 * 0.66, R * 1e3 * 0.98),
                  fontsize=4.6, color=MUTE, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.35,
                                  shrinkA=1.0, shrinkB=1.5))
    style_ax(ax_c)
    panel_label(ax_c, "c", x=-0.36, y=1.0)
    hd = [Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#8C8C8C", mew=0.4,
                 ms=2.4, label="start"),
          Line2D([0], [0], marker="o", ls="none", mfc=CLAY, mec="none", ms=2.6,
                 label="estimate"),
          Line2D([0], [0], marker="*", ls="none", mfc=INK, mec="none", ms=4.0,
                 label="truth")]
    ax_c.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=4.6, handletextpad=0.1,
                borderaxespad=0.1, labelspacing=0.25)
    pdf, png = save(fig, "fig6_inversion")
    print("wrote", pdf, png)
