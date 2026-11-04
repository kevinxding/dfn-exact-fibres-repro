"""Exact algebra checks for the P2D grouping map used in the manuscript.

These checks support, but do not replace, the equation-by-equation prose proof.
They intentionally use no numerical substitutions.
"""

from __future__ import annotations

import sympy as sp


def assert_zero(expr: sp.Expr, label: str) -> None:
    simplified = sp.simplify(expr)
    if simplified != 0:
        raise AssertionError(f"{label}: expected 0, got {simplified}")


def main() -> None:
    lam, b = sp.symbols("lambda b", positive=True)
    gamma = sp.symbols("gamma", positive=True)
    nu_n, nu_s, nu_p = sp.symbols("nu_n nu_s nu_p", positive=True)
    tau_n, tau_s, tau_p = sp.symbols("tau_n tau_s tau_p", positive=True)
    dc_n, dc_s, dc_p = sp.symbols("dc_n dc_s dc_p")

    # Abstract action: source ratios are invariant.
    for name, nu in (("n", nu_n), ("s", nu_s), ("p", nu_p)):
        assert_zero((lam * gamma) / (lam * nu) - gamma / nu, f"source ratio {name}")

    # Interface flux residuals are multiplied by a nonzero common factor, so
    # their zero sets are exactly preserved.
    flux_ns = nu_n / tau_n * dc_n - nu_s / tau_s * dc_s
    flux_sp = nu_s / tau_s * dc_s - nu_p / tau_p * dc_p
    flux_ns_lifted = lam * nu_n / tau_n * dc_n - lam * nu_s / tau_s * dc_s
    flux_sp_lifted = lam * nu_s / tau_s * dc_s - lam * nu_p / tau_p * dc_p
    assert_zero(flux_ns_lifted - lam * flux_ns, "negative/separator flux residual")
    assert_zero(flux_sp_lifted - lam * flux_sp, "separator/positive flux residual")

    # Candidate physical lift.
    F, c_e_ref, i_ref = sp.symbols("F c_e_ref i_ref", positive=True)
    A, L = sp.symbols("A L", positive=True)
    eps_e, eps_s = sp.symbols("eps_e eps_s", positive=True)
    D_e, kappa_e = sp.symbols("D_e kappa_e", positive=True)
    sigma_s, c_s_max = sp.symbols("sigma_s c_s_max", positive=True)
    R_s, D_s, k_n, R_f = sp.symbols("R_s D_s k_n R_f", positive=True)
    t_plus, k_f = sp.symbols("t_plus k_f", positive=True)

    tau_ds = R_s**2 / D_s
    tau_cs = F * eps_s * L * A * c_s_max / i_ref
    tau_k = R_s / (k_n * sp.sqrt(c_e_ref))
    tau_de = L**2 * eps_e ** (1 - b) / D_e
    nu_e = F * eps_e * L * c_e_ref / i_ref
    kappa = A * eps_e**b * kappa_e / (L * i_ref)
    sigma = A * eps_s * sigma_s / (L * i_ref)
    r_f = R_f * i_ref / A
    gamma_phys = (1 - t_plus) / A
    chi_f = (1 - t_plus) * k_f

    lift = {
        A: A / lam,
        eps_e: lam * eps_e,
        eps_s: lam * eps_s,
        D_e: lam ** (1 - b) * D_e,
        kappa_e: lam ** (1 - b) * kappa_e,
        R_f: R_f / lam,
    }

    fixed_groups = {
        "tau_ds": tau_ds,
        "tau_cs": tau_cs,
        "tau_k": tau_k,
        "tau_de": tau_de,
        "kappa": kappa,
        "sigma": sigma,
        "r_f": r_f,
        "chi_f": chi_f,
    }
    for name, expr in fixed_groups.items():
        transformed = expr.subs(lift, simultaneous=True)
        assert_zero(transformed - expr, f"physical lift fixes {name}")

    assert_zero(nu_e.subs(lift, simultaneous=True) - lam * nu_e, "physical lift scales nu_e")
    assert_zero(
        gamma_phys.subs(lift, simultaneous=True) - lam * gamma_phys,
        "physical lift scales gamma",
    )

    print("PASS: all exact abstract-action and physical-grouping identities hold")


if __name__ == "__main__":
    main()
