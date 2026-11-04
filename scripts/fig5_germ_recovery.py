"""Fig5 — germ-recovery demo on a reduced port (compact, Fig3 craft)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, save, panel_label, FS_BODY, FS_SECONDARY

apply_style(FS_BODY)
INK, MUTE, HAIR = "#1A1A1A", "#5A5A5A", "#2E2E2E"
BLU, GRN, ORG, GLD = "#0072B2", "#009E73", "#D55E00", "#B8860B"
FILL_T, FILL_R = "#A6CEE3", "#FDBF6F"
VOC = 3.30


class PortParams:
    def __init__(self, Rser=0.045, An=1.2, Ap=0.85, Tn=80.0, Tp=45.0,
                 zeta_s=0.55, gamma=1.0, rf=0.020, kappa_s=None):
        self.Rser = float(Rser)
        self.An, self.Ap = float(An), float(Ap)
        self.Tn, self.Tp = float(Tn), float(Tp)
        self.zeta_s = float(zeta_s)
        self.gamma = float(gamma)
        if kappa_s is None:
            self.rf = float(rf)
            self.kappa_s = 1.0 / (self.Rser - self.rf)
        else:
            self.kappa_s = float(kappa_s)
            self.rf = self.Rser - 1.0 / self.kappa_s
        assert self.rf > 0 and self.kappa_s > 0

    def copy_Gc(self, lam=2.5):
        return PortParams(self.Rser, self.An, self.Ap, self.Tn, self.Tp,
                          self.zeta_s, gamma=self.gamma * lam, rf=self.rf,
                          kappa_s=self.kappa_s)

    def copy_Gs(self, r0=None):
        if r0 is None:
            r0 = 0.40 * self.Rser
        return PortParams(self.Rser, self.An, self.Ap, self.Tn, self.Tp,
                          self.zeta_s, gamma=self.gamma, rf=self.Rser - r0,
                          kappa_s=1.0 / r0)

    def copy_off_R(self, dR=0.12):
        return PortParams(self.Rser + dR, self.An, self.Ap, self.Tn, self.Tp,
                          self.zeta_s, self.gamma, rf=self.rf + dR,
                          kappa_s=self.kappa_s)

    def copy_off_A(self, scale=1.4):
        return PortParams(self.Rser, self.An * scale, self.Ap, self.Tn, self.Tp,
                          self.zeta_s, self.gamma, rf=self.rf,
                          kappa_s=self.kappa_s)


def simulate(p: PortParams, t, u):
    dt = t[1] - t[0]
    sn = np.zeros_like(t); sp = np.zeros_like(t); e = np.zeros_like(t)
    V = np.zeros_like(t)
    tau_e = 12.0 / p.zeta_s
    for i in range(len(t)):
        V[i] = VOC - p.Rser * u[i] - p.An * sn[i] - p.Ap * sp[i] - p.zeta_s * e[i]
        if i + 1 < len(t):
            sn[i + 1] = sn[i] + dt * ((u[i] - sn[i]) / p.Tn)
            sp[i + 1] = sp[i] + dt * ((u[i] - sp[i]) / p.Tp)
            e[i + 1] = e[i] + dt * ((u[i] - e[i]) / tau_e)
    return V


def recover_from_germ(t, u, V):
    iu = int(np.where(np.abs(np.diff(u, prepend=u[0])) > 0.4)[0][0])
    Rser_hat = -(V[iu] - V[iu - 1]) / (u[iu] - u[iu - 1])
    r = -(V - VOC + Rser_hat * u)
    t0 = t[iu]; Uamp = u[iu]
    mask = (t >= t0) & (t <= t0 + 55.0) & (np.abs(u - Uamp) < 1e-12)
    tt = t[mask] - t0; rr = r[mask]
    dt = t[1] - t[0]

    def step_resp(T):
        a = 1.0 - dt / T
        n = np.rint(tt / dt).astype(int)
        return 1.0 - np.power(a, n)

    T_grid = np.unique(np.concatenate([np.linspace(30, 95, 27), [45.0, 80.0]]))
    tau_grid = np.unique(np.concatenate([np.linspace(12, 30, 19), [12.0 / 0.55]]))
    best = None
    for i, Tn in enumerate(T_grid):
        for Tp in T_grid[i + 1:]:
            for tau_e in tau_grid:
                Phi = np.column_stack([
                    Uamp * step_resp(Tn), Uamp * step_resp(Tp), Uamp * step_resp(tau_e)])
                coef, *_ = np.linalg.lstsq(Phi, rr, rcond=None)
                if np.any(coef <= 1e-8):
                    continue
                err = np.linalg.norm(Phi @ coef - rr)
                if best is None or err < best[0]:
                    best = (err, Tn, Tp, float(coef[0]), float(coef[1]),
                            float(coef[2]), float(tau_e))
    err, Tn, Tp, An, Ap, Az, tau_e = best
    cores = sorted([(An, Tn), (Ap, Tp)], key=lambda x: -x[0])
    return {"Rser": float(Rser_hat), "cores": cores, "zeta_s": Az,
            "zeta_from_tau": 12.0 / tau_e, "fit_err": float(err)}


t = np.linspace(0, 200, 4001)
u = np.zeros_like(t)
u[(t >= 10) & (t < 70)] = 1.0
u[(t >= 100) & (t < 140)] = -0.5

nom = PortParams()
cases = {
    "Nominal": nom,
    r"$G_c$ orbit": nom.copy_Gc(2.5),
    r"$G_s$ orbit": nom.copy_Gs(),
    r"Off $\Delta R_{\mathrm{ser}}$": nom.copy_off_R(),
    r"Off $A_n$": nom.copy_off_A(),
}
keys = list(cases.keys())
Vs = {k: simulate(p, t, u) for k, p in cases.items()}
rec = {k: recover_from_germ(t, u, V) for k, V in Vs.items()}

xtick_labs = ["Nom.", r"$G_c$", r"$G_s$", r"Off $\Delta R$", r"Off $A_n$"]

# ---- compact double-column layout (Fig3/Fig4 craft) ----
# Fig4 muted steel/clay fills
STEEL, CLAY = "#D5E3EB", "#EFDCC6"
CLAY_BAR = "#D4B896"

fig = plt.figure(figsize=(7.2, 3.70), facecolor="white")
# Even vertical rhythm: top block / bottom block with clear, uniform gap
outer = fig.add_gridspec(
    2, 1, height_ratios=[1.55, 1.0],
    left=0.078, right=0.988, top=0.97, bottom=0.11,
    hspace=0.42,
)
# a ~2x prior strip; legend strip; voltage
top = outer[0].subgridspec(
    3, 1, height_ratios=[1.20, 0.18, 1.15], hspace=0.10,
)
# bottom: legend strip over d only + c|d
bot = outer[1].subgridspec(
    2, 2, height_ratios=[0.18, 1.0], hspace=0.08, wspace=0.30,
)

ax_a = fig.add_subplot(top[0, 0])
ax_leg = fig.add_subplot(top[1, 0]); ax_leg.axis("off")
ax_b = fig.add_subplot(top[2, 0], sharex=ax_a)
ax_c = fig.add_subplot(bot[1, 0])
ax_d_leg = fig.add_subplot(bot[0, 1]); ax_d_leg.axis("off")
ax_d = fig.add_subplot(bot[1, 1])
# empty cell above c (keeps row alignment)
ax_c_pad = fig.add_subplot(bot[0, 0]); ax_c_pad.axis("off")


def style_ax(ax):
    ax.tick_params(labelsize=FS_SECONDARY, length=2.0, width=0.4, colors=INK)
    for s in ax.spines.values():
        s.set_color(HAIR); s.set_linewidth(0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --- a: current strip (~2x prior height) ---
panel_label(ax_a, "a", x=-0.05, y=1.25)
ax_a.fill_between(t, 0, u, where=u >= 0, color="#D9E6EF", lw=0, zorder=0)
ax_a.fill_between(t, 0, u, where=u < 0, color="#F3E4D4", lw=0, zorder=0)
ax_a.plot(t, u, color=INK, lw=0.85)
ax_a.axhline(0, color=HAIR, lw=0.30, zorder=1)
ax_a.set_xlim(0, 200)
ax_a.set_ylim(-0.75, 1.30)
ax_a.set_ylabel(r"$u$", color=INK, rotation=0, labelpad=8, va="center")
ax_a.set_yticks([0.0, 1.0])
style_ax(ax_a)
plt.setp(ax_a.get_xticklabels(), visible=False)
ax_a.tick_params(axis="x", length=0)

# --- b: voltage; legend strip immediately above ---
panel_label(ax_b, "b", x=-0.05, y=1.06)
styles = {
    "Nominal": (INK, "-", 0.90),
    r"$G_c$ orbit": (BLU, (0, (2.5, 1.2)), 0.75),
    r"$G_s$ orbit": (GRN, (0, (0.8, 1.0)), 0.75),
    r"Off $\Delta R_{\mathrm{ser}}$": (ORG, "-", 0.85),
    r"Off $A_n$": (GLD, "-", 0.85),
}
handles = []
for k, V in Vs.items():
    c, ls, lw = styles[k]
    ln, = ax_b.plot(t, V, color=c, ls=ls, lw=lw)
    handles.append(ln)
# Time label INSIDE axes, bottom-right — avoids collision with panel letter d
ax_b.set_xlabel("")
ax_b.text(0.02, 0.05, "Time (s)", transform=ax_b.transAxes,
          ha="left", va="bottom", fontsize=FS_BODY, color=MUTE)
ax_b.set_ylabel(r"Voltage $V$ (V)", color=INK)
ax_b.set_xlim(0, 200)
ax_b.set_xticks([0, 50, 100, 150, 200])
ymin = min(V.min() for V in Vs.values()) - 0.05
ymax = max(V.max() for V in Vs.values()) + 0.16
ax_b.set_ylim(ymin, ymax)
style_ax(ax_b)
ax_leg.legend(
    handles,
    ["Nominal", r"$G_c$ orbit", r"$G_s$ orbit",
     r"Off $\Delta R_{\mathrm{ser}}$", r"Off $A_n$"],
    loc="center", ncol=5, frameon=False, fontsize=5.0,
    handlelength=1.20, columnspacing=0.90, handletextpad=0.28,
)

# --- c: R_ser true vs recovered (Fig4 steel/clay, no hatch) ---
panel_label(ax_c, "c", x=-0.17, y=1.25)
R_true = [cases[k].Rser for k in keys]
R_hat = [rec[k]["Rser"] for k in keys]
x = np.arange(len(keys)); w = 0.34
ax_c.bar(x - w / 2, R_true, w, color=STEEL, edgecolor=HAIR, lw=0.50,
         label="true", zorder=2)
ax_c.bar(x + w / 2, R_hat, w, color=CLAY, edgecolor=HAIR, lw=0.50,
         label="recovered", zorder=2)
ax_c.set_xticks(x)
ax_c.set_xticklabels(xtick_labs, fontsize=5.0)
ax_c.set_ylabel(r"$R_{\mathrm{ser}}$", color=INK)
ax_c.set_ylim(0, 0.195)
ax_c.set_yticks([0, 0.05, 0.10, 0.15])
style_ax(ax_c)
ax_c.legend(fontsize=5.0, frameon=False, loc="upper left",
            handlelength=1.05, borderaxespad=0.15)

# --- d: distance + fidelity; muted unified bars; legend strip above ---
panel_label(ax_d, "d", x=-0.17, y=1.25)


def qvec(r):
    A = sorted([r["cores"][0][0], r["cores"][1][0]])
    T = sorted([r["cores"][0][1], r["cores"][1][1]])
    return np.array([r["Rser"], A[0], A[1], T[0], T[1], r["zeta_s"]], dtype=float)


q0 = qvec(rec["Nominal"])
dist, fid = [], []
for k in keys:
    p = cases[k]; r = rec[k]
    dist.append(float(np.linalg.norm(qvec(r) - q0) / np.linalg.norm(q0)))
    A_true = np.array(sorted([p.An, p.Ap]))
    A_hat = np.array(sorted([c[0] for c in r["cores"]]))
    T_true = np.array(sorted([p.Tn, p.Tp]))
    T_hat = np.array(sorted([c[1] for c in r["cores"]]))
    e = np.array([
        abs(r["Rser"] - p.Rser) / p.Rser,
        np.linalg.norm(A_hat - A_true) / np.linalg.norm(A_true),
        np.linalg.norm(T_hat - T_true) / np.linalg.norm(T_true),
        abs(r["zeta_s"] - p.zeta_s) / max(p.zeta_s, 1e-30),
    ])
    fid.append(float(np.max(e)))

dist = np.asarray(dist); fid = np.asarray(fid)
FLOOR = 2e-15
fid_plot = np.maximum(fid, FLOOR)

ax_d.axhspan(8e-16, 4e-13, color="#E8EEF2", zorder=0, lw=0)
# Right-aligned band label, clear of left-side gauge markers
# Right-aligned inside band; Nom column reserved for label (no markers there)
ax_d.text(0.18, 1.6e-13, "machine-zero", ha="left", va="center",
          fontsize=3.8, color=MUTE, zorder=6, clip_on=True)

off_mask = dist > 1e-12
ax_d.bar(x[off_mask], dist[off_mask], color=CLAY_BAR, edgecolor=HAIR, lw=0.50,
         width=0.55, zorder=2)
gauge_x = x[~off_mask]
gauge_x = gauge_x[gauge_x > 0]  # skip Nom; label lives there
ax_d.plot(gauge_x, np.full(len(gauge_x), FLOOR), "s", ms=3.2,
          mfc="white", mec="#008C6E", mew=0.70, zorder=5, ls="none")
fid_x = x[x > 0]
ax_d.plot(fid_x, fid_plot[fid_x], "o", ms=4.0, mfc="white", mec=BLU, mew=0.75,
          zorder=4, ls="none")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
leg_handles = [
    Patch(facecolor=CLAY_BAR, edgecolor=HAIR, linewidth=0.50,
          label=r"$\|Q-Q_{\mathrm{nom}}\|/\|Q_{\mathrm{nom}}\|$"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
           markeredgecolor="#008C6E", markeredgewidth=0.70, markersize=3.2,
           label=r"gauge ($=0$)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
           markeredgecolor=BLU, markeredgewidth=0.75, markersize=4.0,
           label="max recovery error"),
]
ax_d_leg.legend(
    handles=leg_handles, loc="center", ncol=3, frameon=False, fontsize=4.5,
    handlelength=1.05, columnspacing=0.85, handletextpad=0.25,
)

ax_d.set_yscale("log")
ax_d.set_ylim(8e-16, 2e-2)
from matplotlib.ticker import FixedLocator, FuncFormatter
ax_d.yaxis.set_major_locator(FixedLocator([1e-15, 1e-12, 1e-9, 1e-6, 1e-3]))
ax_d.yaxis.set_major_formatter(FuncFormatter(
    lambda v, p: r"$10^{%d}$" % int(round(np.log10(v))) if v > 0 else ""))
ax_d.yaxis.set_minor_locator(FixedLocator([]))
ax_d.set_xticks(x)
ax_d.set_xticklabels(xtick_labs, fontsize=5.0)
ax_d.set_ylabel("Relative scale", color=INK)
style_ax(ax_d)

pdf, png = save(fig, "fig5_germ_recovery")

lines = ["# Germ recovery report (reduced port)", ""]
q_ref = None
for k in keys:
    pp = cases[k]; r = rec[k]
    A = sorted([c[0] for c in r["cores"]]); TT = sorted([c[1] for c in r["cores"]])
    q = np.array([r["Rser"], A[0], A[1], TT[0], TT[1], r["zeta_s"]], dtype=float)
    if q_ref is None:
        q_ref = q.copy()
    d = float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))
    lines += [
        f"## {k}",
        f"Rser true={pp.Rser:.8g} hat={r['Rser']:.8g}",
        f"A true={sorted([pp.An,pp.Ap])} hat={A}",
        f"T true={sorted([pp.Tn,pp.Tp])} hat={TT}",
        f"zeta true={pp.zeta_s:.8g} hat={r['zeta_s']:.8g} fit_err={r['fit_err']:.3e}",
        f"dist_to_nominal_Q={d:.6e}",
        "",
    ]
_rep = Path(__file__).resolve().parents[1] / "reports" / "fig5_recovery_report.txt"
_rep.parent.mkdir(parents=True, exist_ok=True)
_rep.write_text("\n".join(lines))
print("wrote", pdf)
print("\n".join(lines))
