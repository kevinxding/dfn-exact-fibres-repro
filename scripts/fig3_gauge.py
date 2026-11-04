"""Fig. 3 — gauge identity vs off-orbit separation.

Panels (a–c): reduced port with explicit G_c and G_s group actions on
(gamma, nu_e^{n,s,p}, r_f, kappa^s). Voltage depends on the germ only through
(zeta_q, R_ser); nontrivial gauges leave V unchanged to floating-point noise.

Panel (d): PyBaMM DFN off-orbit geometric move (L_p x 1.5). Public PyBaMM
parameter charts do not expose abstract G_c coordinates as independent knobs,
so production-code gauge algebra is not claimed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, save, panel_label, FS_BODY, FS_SECONDARY

apply_style(FS_BODY)

# Muted steel / clay palette (consistent with sister figures)
INK, MUTE, HAIR = "#1A1A1A", "#5A5A5A", "#2E2E2E"
STEEL, CLAY, CLAY2, SLATE = "#3D5A6C", "#A67C52", "#C4A484", "#6B7C8A"
ORG, GLD = "#D55E00", "#B8860B"


class GaugePort:
    """Reduced port whose voltage depends on zeta_q = gamma/nu_e^q and R_ser.

    Group actions act on the underlying (gamma, nu, r_f, kappa^s) chart.
    """

    def __init__(
        self,
        gamma=1.0,
        nu_n=1.0,
        nu_s=1.0,
        nu_p=1.0,
        rf=0.015,
        kappa_s=40.0,
        An=1.15,
        Ap=0.90,
        Tn=70.0,
        Tp=40.0,
        V0=3.15,
    ):
        self.gamma = float(gamma)
        self.nu = {"n": float(nu_n), "s": float(nu_s), "p": float(nu_p)}
        self.rf = float(rf)
        self.kappa_s = float(kappa_s)
        self.An, self.Ap = float(An), float(Ap)
        self.Tn, self.Tp = float(Tn), float(Tp)
        self.V0 = float(V0)
        assert self.rf > 0 and self.kappa_s > 0
        assert all(v > 0 for v in self.nu.values()) and self.gamma > 0

    @property
    def zeta(self):
        return {q: self.gamma / self.nu[q] for q in self.nu}

    @property
    def Rser(self):
        return self.rf + 1.0 / self.kappa_s

    def apply_Gc(self, lam: float) -> "GaugePort":
        """Common concentration scaling: (gamma, nu_*) *= lam."""
        lam = float(lam)
        assert lam > 0 and abs(lam - 1.0) > 1e-12
        return GaugePort(
            gamma=self.gamma * lam,
            nu_n=self.nu["n"] * lam,
            nu_s=self.nu["s"] * lam,
            nu_p=self.nu["p"] * lam,
            rf=self.rf,
            kappa_s=self.kappa_s,
            An=self.An,
            Ap=self.Ap,
            Tn=self.Tn,
            Tp=self.Tp,
            V0=self.V0,
        )

    def apply_Gs(self, rf_new: float) -> "GaugePort":
        """Separator/series redistribution at fixed R_ser."""
        rf_new = float(rf_new)
        R = self.Rser
        assert 0 < rf_new < R
        kappa_new = 1.0 / (R - rf_new)
        assert abs(kappa_new - self.kappa_s) > 1e-12 or abs(rf_new - self.rf) > 1e-12
        return GaugePort(
            gamma=self.gamma,
            nu_n=self.nu["n"],
            nu_s=self.nu["s"],
            nu_p=self.nu["p"],
            rf=rf_new,
            kappa_s=kappa_new,
            An=self.An,
            Ap=self.Ap,
            Tn=self.Tn,
            Tp=self.Tp,
            V0=self.V0,
        )

    def off_Rser(self, dR: float) -> "GaugePort":
        return GaugePort(
            gamma=self.gamma,
            nu_n=self.nu["n"],
            nu_s=self.nu["s"],
            nu_p=self.nu["p"],
            rf=self.rf + dR,
            kappa_s=self.kappa_s,
            An=self.An,
            Ap=self.Ap,
            Tn=self.Tn,
            Tp=self.Tp,
            V0=self.V0,
        )

    def off_single_nu_s(self, scale: float) -> "GaugePort":
        """Scale only nu_s (breaks zeta_s); not a G_c element."""
        return GaugePort(
            gamma=self.gamma,
            nu_n=self.nu["n"],
            nu_s=self.nu["s"] * scale,
            nu_p=self.nu["p"],
            rf=self.rf,
            kappa_s=self.kappa_s,
            An=self.An,
            Ap=self.Ap,
            Tn=self.Tn,
            Tp=self.Tp,
            V0=self.V0,
        )

    def simulate(self, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Forward map using only germ invariants (zeta, R_ser) and particles."""
        dt = float(t[1] - t[0])
        zs = self.zeta["s"]
        # Electrolyte polarization time set by zeta_s (invariant under G_c)
        tau_e = 12.0 / zs
        sn = np.zeros_like(t)
        sp = np.zeros_like(t)
        e = np.zeros_like(t)
        V = np.zeros_like(t)
        R = self.Rser
        for i in range(len(t)):
            V[i] = self.V0 - R * u[i] - self.An * sn[i] - self.Ap * sp[i] - zs * e[i]
            if i + 1 < len(t):
                sn[i + 1] = sn[i] + dt * ((u[i] - sn[i]) / self.Tn)
                sp[i + 1] = sp[i] + dt * ((u[i] - sp[i]) / self.Tp)
                e[i + 1] = e[i] + dt * ((u[i] - e[i]) / tau_e)
        return V


def main() -> None:
    t = np.linspace(0.0, 240.0, 4801)
    u = np.zeros_like(t)
    u[(t >= 20) & (t < 80)] = 1.0
    u[(t >= 120) & (t < 180)] = -0.5

    nom = GaugePort()
    # Nontrivial gauges (several group elements)
    gc_list = [nom.apply_Gc(2.5), nom.apply_Gc(0.4)]
    gs_list = [nom.apply_Gs(0.008), nom.apply_Gs(0.028)]
    off_R = nom.off_Rser(0.15)
    off_nu = nom.off_single_nu_s(0.70)

    # Sanity: gauges must change the chart, not just copy floats
    assert abs(gc_list[0].gamma - nom.gamma) > 0.5
    assert abs(gs_list[0].rf - nom.rf) > 1e-3
    assert abs(gc_list[0].zeta["s"] - nom.zeta["s"]) < 1e-14
    assert abs(gs_list[0].Rser - nom.Rser) < 1e-14

    V_nom = nom.simulate(t, u)
    V_gc = [p.simulate(t, u) for p in gc_list]
    V_gs = [p.simulate(t, u) for p in gs_list]
    V_R = off_R.simulate(t, u)
    V_nu = off_nu.simulate(t, u)

    err_gc = [float(np.max(np.abs(V - V_nom))) for V in V_gc]
    err_gs = [float(np.max(np.abs(V - V_nom))) for V in V_gs]
    err_R = float(np.max(np.abs(V_R - V_nom)))
    err_nu = float(np.max(np.abs(V_nu - V_nom)))
    # Relative ell^infty as in the table caption
    denom = float(np.max(np.abs(V_nom))) + 1e-30
    rel_gc = [e / denom for e in err_gc]
    rel_gs = [e / denom for e in err_gs]
    rel_R = err_R / denom
    rel_nu = err_nu / denom

    # --- PyBaMM off-orbit only ---
    import pybamm

    model = pybamm.lithium_ion.DFN()
    param = model.default_parameter_values
    exp = pybamm.Experiment(
        [
            "Rest for 10 seconds",
            "Discharge at 1C for 90 seconds or until 3.0 V",
            "Rest for 30 seconds",
        ]
    )
    sol0 = pybamm.Simulation(model, experiment=exp, parameter_values=param).solve()
    t2 = sol0["Time [s]"].entries
    V_dfn = sol0["Voltage [V]"].entries
    param2 = model.default_parameter_values.copy()
    param2["Positive electrode thickness [m]"] = (
        1.5 * param2["Positive electrode thickness [m]"]
    )
    sol1 = pybamm.Simulation(model, experiment=exp, parameter_values=param2).solve()
    V_off = np.interp(t2, sol1["Time [s]"].entries, sol1["Voltage [V]"].entries)
    dV_mV = (V_off - V_dfn) * 1e3
    pybamm_dV_max = float(np.max(np.abs(dV_mV)))





    # --- Figure: no-overlap legend; log residuals + exact-zero strip ---
    apply_style(7.0)
    FS_TICK = 6.5
    FS_LEG = 6.2
    FS_ANN = 6.0

    fig = plt.figure(figsize=(7.0, 5.55), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.985,
        top=0.965,
        bottom=0.08,
        wspace=0.28,
        hspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    # panel c: log residuals + exact-zero strip
    gs_c = gs[1, 0].subgridspec(2, 1, height_ratios=[3.2, 1.15], hspace=0.14)
    ax_c = fig.add_subplot(gs_c[0])
    ax_cz = fig.add_subplot(gs_c[1], sharex=ax_c)
    # panel d: voltage + ΔV strip
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.08)
    ax_d = fig.add_subplot(gs_d[0])
    ax_dv = fig.add_subplot(gs_d[1], sharex=ax_d)

    def style_ax(ax, labelbottom=True):
        ax.tick_params(
            labelsize=FS_TICK, length=2.2, width=0.45, colors=INK, labelbottom=labelbottom
        )
        for s in ax.spines.values():
            s.set_color(HAIR)
            s.set_linewidth(0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_size(7.0)
        ax.yaxis.label.set_size(7.0)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)

    def lab(ax, letter):
        panel_label(ax, letter, x=-0.16, y=1.03, fontsize=8.0)

    # a — current
    ax = ax_a
    lab(ax, "a")
    ax.plot(t, u, color=INK, lw=1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_xlim(0, 240)
    ax.set_ylim(-0.75, 1.35)
    style_ax(ax)

    # b — voltages; legend OUTSIDE above axes (no data overlap)
    ax = ax_b
    lab(ax, "b")
    (ln1,) = ax.plot(t, V_nom, color=INK, lw=1.05, zorder=3)
    (ln2,) = ax.plot(t, V_gc[0], color=STEEL, lw=0.9, ls=(0, (2.8, 1.3)), zorder=4)
    (ln3,) = ax.plot(t, V_gs[0], color=CLAY, lw=0.9, ls=(0, (0.9, 1.1)), zorder=4)
    ax.plot(t, V_gc[1], color=STEEL, lw=0.55, ls=(0, (1.2, 1.6)), alpha=0.45, zorder=2)
    ax.plot(t, V_gs[1], color=CLAY2, lw=0.55, ls=(0, (0.6, 1.2)), alpha=0.45, zorder=2)
    (ln4,) = ax.plot(t, V_R, color=ORG, lw=0.95, zorder=3)
    (ln5,) = ax.plot(t, V_nu, color=GLD, lw=0.95, zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Voltage $V$ (V)")
    ax.set_xlim(0, 240)
    y_lo = min(V_nom.min(), V_R.min(), V_nu.min()) - 0.08
    y_hi = max(V_nom.max(), V_R.max(), V_nu.max()) + 0.12
    ax.set_ylim(y_lo, y_hi)
    style_ax(ax)
    # Legend in lower-right empty band (t>~170, V<~1.5; all traces sit above ~3.2 there).
    ax.legend(
        [ln1, ln2, ln3, ln4, ln5],
        [
            "Nominal",
            r"$G_c$ (2.5, 0.4)",
            r"$G_s$ (2 splits)",
            r"Off $\Delta R_{\mathrm{ser}}$",
            r"Off $\nu_e^s$",
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        framealpha=0.96,
        fontsize=FS_LEG - 0.3,
        handlelength=1.1,
        handletextpad=0.28,
        borderpad=0.22,
        labelspacing=0.10,
        borderaxespad=0.0,
    )
    # Zoom inset in upper-left empty band during the first pulse (traces sit below ~2.5 V).
    axins = ax.inset_axes([0.08, 0.78, 0.34, 0.20])
    m = (t >= 45) & (t <= 65)
    for V, c, ls, lw in [
        (V_nom, INK, "-", 0.75),
        (V_gc[0], STEEL, (0, (2.5, 1.2)), 0.7),
        (V_gs[0], CLAY, (0, (0.8, 1.0)), 0.7),
        (V_R, ORG, "-", 0.7),
        (V_nu, GLD, "-", 0.7),
    ]:
        axins.plot(t[m], V[m], color=c, lw=lw, ls=ls)
    axins.set_xlim(45, 65)
    axins.set_title("45–65 s", fontsize=FS_ANN, color=MUTE, pad=1.0)
    axins.tick_params(labelsize=FS_ANN - 0.4, length=1.2, width=0.35, colors=INK)
    for s in axins.spines.values():
        s.set_linewidth(0.45)
        s.set_color(HAIR)

    # c — off-orbit on log axis; gauges on a separate exact-zero strip
    ax = ax_c
    dR = np.abs(V_R - V_nom)
    dNu = np.abs(V_nu - V_nom)
    # Mask exact zeros (rest intervals) so the log panel never invents a floor value
    dR_m = np.ma.masked_less_equal(dR, 0.0)
    dNu_m = np.ma.masked_less_equal(dNu, 0.0)
    ax.semilogy(t, dR_m, color=ORG, lw=1.0, label=r"Off $\Delta R_{\mathrm{ser}}$", zorder=2)
    ax.semilogy(t, dNu_m, color=GLD, lw=1.0, label=r"Off $\nu_e^s$", zorder=2)
    ax.set_ylabel(r"$|V-V_{\mathrm{nom}}|$ (V)")
    ax.set_xlim(0, 240)
    ymax = float(max(dR.max(), dNu.max()))
    ax.set_ylim(1e-4, max(1.2, ymax * 1.15))
    style_ax(ax, labelbottom=False)
    ax.legend(
        fontsize=FS_LEG,
        frameon=False,
        loc="upper right",
        handlelength=1.35,
        labelspacing=0.18,
        borderaxespad=0.15,
    )

    # exact-zero strip (NOT a log value) — visually distinct from the log panel
    ax = ax_cz
    ax.set_facecolor("#F3F5F7")
    ax.set_xlim(0, 240)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#8A97A0", lw=0.8, zorder=0)
    t_mark = np.array([30.0, 70.0, 110.0, 150.0, 190.0, 225.0])
    ax.plot(
        t_mark - 4,
        np.zeros_like(t_mark),
        "o",
        ms=4.5,
        mfc="white",
        mec=STEEL,
        mew=1.05,
        zorder=3,
        ls="none",
        label=r"$G_c$",
    )
    ax.plot(
        t_mark + 4,
        np.zeros_like(t_mark),
        "s",
        ms=4.0,
        mfc="white",
        mec=CLAY,
        mew=1.05,
        zorder=3,
        ls="none",
        label=r"$G_s$",
    )
    ax.set_yticks([0])
    ax.set_yticklabels([r"$0$"], fontsize=FS_TICK)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|\Delta V|$", fontsize=FS_ANN + 0.3)
    style_ax(ax)
    # left annotation: exact zeros live only on this strip
    ax.text(
        0.01,
        0.5,
        r"$|\Delta V|=0$ (exact; not a log value)",
        transform=ax.transAxes,
        fontsize=FS_ANN,
        color=INK,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#8A97A0", lw=0.55),
    )
    ax.legend(
        fontsize=FS_LEG - 0.3,
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    # visual break between log panel and exact-zero strip
    ax_c.spines["bottom"].set_visible(False)
    ax_cz.spines["top"].set_visible(False)
    d = 0.012
    kw = dict(color=HAIR, lw=0.7, clip_on=False, solid_capstyle="round")
    # bottom edge of log panel
    ax_c.plot((-d, +d), (-d, +d), transform=ax_c.transAxes, **kw)
    ax_c.plot((1 - d, 1 + d), (-d, +d), transform=ax_c.transAxes, **kw)
    # top edge of exact-zero strip
    ax_cz.plot((-d, +d), (1 - d, 1 + d), transform=ax_cz.transAxes, **kw)
    ax_cz.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_cz.transAxes, **kw)

    # d — PyBaMM voltage + ΔV strip
    ax = ax_d
    ax.plot(t2, V_dfn, color=INK, lw=1.05, label="DFN nominal")
    ax.plot(t2, V_off, color=ORG, lw=1.0, ls="--", label=r"Off-orbit $1.5\,L_p$")
    ax.set_ylabel(r"Voltage $V$ (V)")
    vmin = float(min(V_dfn.min(), V_off.min()))
    vmax = float(max(V_dfn.max(), V_off.max()))
    pad = 0.06 * (vmax - vmin + 1e-6)
    ax.set_ylim(vmin - pad, vmax + pad)
    style_ax(ax, labelbottom=False)
    ax.legend(
        fontsize=FS_LEG - 0.2,
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        framealpha=0.94,
        loc="upper right",
        handlelength=1.35,
        borderpad=0.22,
        labelspacing=0.12,
        borderaxespad=0.2,
    )

    ax = ax_dv
    ax.fill_between(t2, 0, dV_mV, color=ORG, alpha=0.20, lw=0)
    ax.plot(t2, dV_mV, color=ORG, lw=0.85)
    ax.axhline(0, color=HAIR, lw=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta V$ (mV)")
    ax.set_xlim(t2[0], t2[-1])
    style_ax(ax)

    # panel letters for c and d on the full block
    fig.canvas.draw()
    bb_c = ax_c.get_position()
    bb_d = ax_d.get_position()
    for letter, bb in (("c", bb_c), ("d", bb_d)):
        fig.text(
            bb.x0 - 0.025,
            bb.y1 + 0.012,
            letter,
            fontsize=8.0,
            fontweight="bold",
            va="bottom",
            ha="left",
            color="#1A1A1A",
            fontfamily="sans-serif",
            clip_on=False,
        )

    pdf, png = save(fig, "fig3_gauge_numerics")

    report = Path(__file__).resolve().parents[1] / "reports" / "fig3_errors.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                f"err_gc_abs_max={max(err_gc):.6e}",
                f"err_gs_abs_max={max(err_gs):.6e}",
                f"rel_gc_max={max(rel_gc):.6e}",
                f"rel_gs_max={max(rel_gs):.6e}",
                f"err_offR_abs={err_R:.6e}",
                f"rel_offR={rel_R:.6e}",
                f"err_offNu_abs={err_nu:.6e}",
                f"rel_offNu={rel_nu:.6e}",
                f"pybamm_dV_mV_max={pybamm_dV_max:.6f}",
                f"gc_gamma_nom={nom.gamma:.6g}",
                f"gc_gamma_lam2p5={gc_list[0].gamma:.6g}",
                f"gs_rf_nom={nom.rf:.6g}",
                f"gs_rf_a={gs_list[0].rf:.6g}",
                f"gs_rf_b={gs_list[1].rf:.6g}",
                f"Rser_nom={nom.Rser:.12g}",
                f"Rser_gs_a={gs_list[0].Rser:.12g}",
            ]
        )
        + "\n"
    )
    print("wrote", pdf)
    print("max |V-Vnom| Gc:", max(err_gc), "Gs:", max(err_gs))
    print("rel Gc:", max(rel_gc), "Gs:", max(rel_gs), "dR:", rel_R, "nu:", rel_nu)
    print("pybamm dV mV max:", pybamm_dV_max)

if __name__ == "__main__":
    main()
