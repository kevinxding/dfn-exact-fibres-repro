"""Isothermal DFN/P2D in the twenty grouped coordinates of the manuscript.

Finite-volume method of lines for the nondimensional system of Appendix A
(Couto grouped formulation).  The twenty coordinates enter exactly as written
there: (gamma, nu_e^n, nu_e^s, nu_e^p) appear separately in the source terms
and flux-continuity conditions, and (r_f, kappa^s) appear separately in the
separator current law and the output equation.  No invariant is hard-coded.

The semi-discrete DAE is integrated with CasADi/IDAS, which also returns
forward parametric sensitivities for the least-squares inversion.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

# Order of the twenty grouped coordinates (Table I of the manuscript)
NAMES = [
    "T_n", "T_p", "C_n", "C_p",            # solid transport
    "tauk_n", "tauk_p",                     # interfacial kinetics
    "tau_n", "tau_s", "tau_p",              # electrolyte transport
    "kappa_n", "kappa_p", "sigma_n", "sigma_p",  # conduction
    "chi",                                  # thermodynamics
    "gamma", "nu_n", "nu_s", "nu_p",        # concentration scale  (G_c)
    "r_f", "kappa_s",                       # series / film        (G_s)
]
IDX = {n: i for i, n in enumerate(NAMES)}

BETA = 96485.33212 / (2.0 * 8.314462618 * 298.15)  # F/(2 R_g T)


def U_n(x):
    """Graphite OCP (Chen et al. 2020 form), smooth in x."""
    return (1.9793 * ca.exp(-39.3631 * x) + 0.2482
            - 0.0909 * ca.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * ca.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * ca.tanh(30.4444 * (x - 0.6103)))


def U_p(x):
    """NMC811 OCP (Chen et al. 2020 form), smooth in x."""
    return (-0.8090 * x + 4.4875
            - 0.0428 * ca.tanh(18.5138 * (x - 0.5542))
            - 17.7326 * ca.tanh(15.7890 * (x - 0.3117))
            + 17.5842 * ca.tanh(15.9308 * (x - 0.3120)))


PULSES = [(20, 120, 1.5), (170, 60, -2.0), (260, 240, 0.8), (540, 40, 2.5),
          (620, 180, -1.0), (840, 30, -2.5), (900, 300, 1.2)]


def current(t, pulses=None):
    """Continuous input with u(0)=0: smooth sin^2 pulses of both signs."""
    pulses = PULSES if pulses is None else pulses
    def bump(t0, w, a):
        s = (t - t0) / w
        return a * ca.if_else(ca.logic_and(s > 0, s < 1),
                              ca.sin(np.pi * s) ** 2, 0.0)
    return sum(bump(t0, w, a) for (t0, w, a) in pulses)


class GroupedDFN:
    def __init__(self, N=(6, 4, 6), Nr=5, z0=(0.75, 0.45), t_end=1260.0, dt=3.0, tol=1e-10, pulses=None):
        self.N, self.Nr, self.z0, self.tol = N, Nr, z0, tol
        self.pulses = pulses
        self.tgrid = np.arange(0.0, t_end + 1e-9, dt)
        self._build()

    def _build(self):
        tol = self.tol
        Nn, Ns, Np = self.N
        Nr = self.Nr
        lp = ca.MX.sym("logtheta", 20)
        th = ca.exp(lp)
        P = {n: th[i] for n, i in IDX.items()}
        t = ca.MX.sym("t")
        u = current(t, self.pulses)

        regions = [("n", Nn), ("s", Ns), ("p", Np)]
        Ncell = Nn + Ns + Np
        # per-cell properties
        reg_of, h_of = [], []
        for q, Nq in regions:
            for _ in range(Nq):
                reg_of.append(q)
                h_of.append(1.0 / Nq)
        ce = ca.MX.sym("ce", Ncell)
        cs = {m: ca.MX.sym("cs_" + m, (Nn if m == "n" else Np) * Nr) for m in "np"}
        phie = ca.MX.sym("phie", Ncell)
        phis = {m: ca.MX.sym("phis_" + m, Nn if m == "n" else Np) for m in "np"}

        tau = {q: P["tau_" + q] for q in "nsp"}
        nu = {q: P["nu_" + q] for q in "nsp"}
        kap = {"n": P["kappa_n"], "s": P["kappa_s"], "p": P["kappa_p"]}
        gam, chi, rf = P["gamma"], P["chi"], P["r_f"]

        # --- surface concentration, kinetics per electrode cell ---
        dr = 1.0 / Nr
        rf_faces = np.linspace(0, 1, Nr + 1)
        vol = (rf_faces[1:] ** 3 - rf_faces[:-1] ** 3) / 3.0
        j = {}
        dcs = {}
        offset = {"n": 0, "p": Nn + Ns}
        Nm = {"n": Nn, "p": Np}
        for m in "np":
            T, C, tk = P["T_" + m], P["C_" + m], P["tauk_" + m]
            Uf = U_n if m == "n" else U_p
            jm, dlist = [], []
            for k in range(Nm[m]):
                c = cs[m][k * Nr:(k + 1) * Nr]
                cell = offset[m] + k
                # surface value taken at the outermost shell centre
                c_ss = c[Nr - 1]
                eta = phis[m][k] - phie[cell] - Uf(c_ss)
                i0 = 3.0 * (C / tk) * ca.sqrt(ce[cell] * c_ss * (1 - c_ss))
                jj = i0 * 2.0 * ca.sinh(BETA * eta)
                jm.append(jj)
                # FV solid diffusion
                fl = [0.0]
                for f in range(1, Nr):
                    fl.append(rf_faces[f] ** 2 * (c[f] - c[f - 1]) / dr / T)
                fl.append(rf_faces[Nr] ** 2 * (-jj / (3.0 * C)))
                dlist.append(ca.vertcat(*[(fl[f + 1] - fl[f]) / vol[f] for f in range(Nr)]))
            j[m] = jm
            dcs[m] = ca.vertcat(*dlist)

        # --- electrolyte faces (Ncell+1), harmonic transport between regions ---
        # mass flux in "nu" form: F = (nu/tau) d_xi c ; eq: nu dc/dt = dF/dxi + gamma j
        Fm = [0.0]
        Ie = [0.0]
        for f in range(1, Ncell):
            L, R = f - 1, f
            qL, qR = reg_of[L], reg_of[R]
            rmass = 0.5 * h_of[L] * tau[qL] / nu[qL] + 0.5 * h_of[R] * tau[qR] / nu[qR]
            Fm.append((ce[R] - ce[L]) / rmass)
            rch = 0.5 * h_of[L] / kap[qL] + 0.5 * h_of[R] / kap[qR]
            Ie.append(-(phie[R] - phie[L] - (chi / BETA) * (ca.log(ce[R]) - ca.log(ce[L]))) / rch)
        Fm.append(0.0)
        Ie.append(0.0)
        dce, alg = [], []
        for c_ in range(Ncell):
            q = reg_of[c_]
            src = 0.0
            if q == "n":
                src = j["n"][c_]
            elif q == "p":
                src = j["p"][c_ - offset["p"]]
            dce.append(((Fm[c_ + 1] - Fm[c_]) / h_of[c_] + gam * src) / nu[q])
            if c_ < Ncell - 1:  # last electrolyte charge equation is implied
                alg.append((Ie[c_ + 1] - Ie[c_]) / h_of[c_] - src)
        # --- solid potential ---
        for m in "np":
            sig = P["sigma_" + m]
            h = 1.0 / Nm[m]
            Is = [u if m == "n" else 0.0]
            for f in range(1, Nm[m]):
                Is.append(-sig * (phis[m][f] - phis[m][f - 1]) / h)
            Is.append(0.0 if m == "n" else u)
            for k in range(Nm[m]):
                alg.append((Is[k + 1] - Is[k]) / h + j[m][k])
        # gauge phi_s,n(0)=0
        alg.append(phis["n"][0] + 0.5 * (1.0 / Nn) * u / P["sigma_n"])
        V = (phis["p"][Np - 1] - 0.5 * (1.0 / Np) * u / P["sigma_p"]) - rf * u

        x = ca.vertcat(ce, cs["n"], cs["p"])
        z = ca.vertcat(phie, phis["n"], phis["p"])
        ode = ca.vertcat(*dce, dcs["n"], dcs["p"])
        dae = {"t": t, "x": x, "z": z, "p": lp, "ode": ode, "alg": ca.vertcat(*alg)}
        opts = {"abstol": tol, "reltol": tol, "calc_ic": True}
        self.F = ca.integrator("dfn", "idas", dae, 0.0, self.tgrid[1:], opts)
        self.Vfun = ca.Function("V", [t, x, z, lp], [V])
        # initial state
        zn, zp = self.z0
        x0 = np.concatenate([np.ones(Ncell), zn * np.ones(Nn * Nr), zp * np.ones(Np * Nr)])
        Un0 = float(ca.DM(U_n(zn)))
        Up0 = float(ca.DM(U_p(zp)))
        z0 = np.concatenate([-Un0 * np.ones(Ncell), np.zeros(Nn), (Up0 - Un0) * np.ones(Np)])
        self.x0, self.zguess = x0, z0
        self.V0 = Up0 - Un0
        # symbolic map logtheta -> voltage trace (+ Jacobian)
        res = self.F(x0=x0, z0=z0, p=lp)
        Xs, Zs = res["xf"], res["zf"]
        tg = self.tgrid[1:]
        Vmap = self.Vfun.map(len(tg))
        Vt = Vmap(ca.DM(tg).T, Xs, Zs, ca.repmat(lp, 1, len(tg)))
        Vall = ca.horzcat(ca.DM(self.V0), Vt).T
        self.sim = ca.Function("sim", [lp], [Vall])
        self.simjac = ca.Function("simjac", [lp], [Vall, ca.jacobian(Vall, lp)])

    def voltage(self, theta):
        return np.array(self.sim(np.log(theta))).ravel()

    def voltage_jac(self, logtheta):
        V, J = self.simjac(logtheta)
        return np.array(V).ravel(), np.array(J)


def true_theta():
    v = dict(
        T_n=1800.0, T_p=1200.0, C_n=4500.0, C_p=4000.0,
        tauk_n=3000.0, tauk_p=5000.0,
        tau_n=40.0, tau_s=15.0, tau_p=35.0,
        kappa_n=40.0, kappa_p=45.0, sigma_n=120.0, sigma_p=25.0,
        chi=0.6,
        gamma=1.0, nu_n=1 / 0.006, nu_s=1 / 0.012, nu_p=1 / 0.007,
        r_f=0.010, kappa_s=60.0,
    )
    return np.array([v[n] for n in NAMES])


def Q18(theta):
    th = dict(zip(NAMES, theta))
    q = [th[n] for n in NAMES[:14]]
    q += [th["gamma"] / th["nu_n"], th["gamma"] / th["nu_s"], th["gamma"] / th["nu_p"]]
    q += [th["r_f"] + 1.0 / th["kappa_s"]]
    return np.array(q)


Q18_NAMES = NAMES[:14] + ["zeta_n", "zeta_s", "zeta_p", "R_ser"]
