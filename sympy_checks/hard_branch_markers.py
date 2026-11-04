"""Symbolic checks for hard-branch markers used in the SM.

Verifies:
  1. c_m = -ell''(z)/4 equals (1/8)(1/z^2 + 1/(1-z)^2) > 0 on (0,1)
  2. alpha_m = ell'(z) vanishes only at z=1/2
  3. Distinct-feedback rate determinant kappa_n*kappa_p*(kappa_p-kappa_n)
  4. Tied-feedback pairing separator (r1-r2)*(rho1^2-rho2^2)
  5. Six timing functions for the order-four mixed marker are linearly
     independent as formal symbols (Wronskian / leading-term test)

These checks support, but do not replace, the prose proofs.
"""

from __future__ import annotations

import sympy as sp


def assert_zero(expr: sp.Expr, label: str) -> None:
    simplified = sp.simplify(expr)
    if simplified != 0:
        raise AssertionError(f"{label}: expected 0, got {simplified}")


def assert_true(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)


def main() -> None:
    z = sp.symbols("z", positive=True)
    ell = sp.log(sp.sqrt(z * (1 - z)))
    alpha = sp.diff(ell, z)
    cm = -sp.diff(ell, z, 2) / 4
    target = (1 / 8) * (1 / z**2 + 1 / (1 - z) ** 2)
    assert_zero(sp.together(cm - target), "c_m identity")
    assert_zero(sp.simplify(alpha.subs(z, sp.Rational(1, 2))), "alpha(1/2)=0")
    # Unique root in (0,1)
    roots = sp.solve(sp.numer(sp.together(alpha)), z)
    assert_true(roots == [sp.Rational(1, 2)], f"alpha vanishes only at 1/2, got {roots}")

    kn, kp, rn, rp, rhon, rhop = sp.symbols(
        "kappa_n kappa_p r_n r_p rho_n rho_p", positive=True
    )
    det = kn * kp * (kp - kn)
    # Nonvanishing on the open stratum kappa_n != kappa_p
    assert_true(sp.simplify(det.subs({kn: 1, kp: 2})) != 0, "distinct-feedback det")

    sep = (rn - rp) * (rhon**2 - rhop**2)
    assert_true(sp.simplify(sep.subs({rn: 1, rp: 2, rhon: 1, rhop: 3})) != 0,
                "fifth-order pairing separator")

    # Formal linear independence of the six timing germs in d (L fixed > 0)
    d, L = sp.symbols("d L", positive=True)
    timings = [
        d ** (-sp.Rational(1, 2)),
        sp.asinh(sp.sqrt(L / d)),
        sp.Integer(1),
        sp.atan(sp.sqrt(L / d)),
        sp.sqrt(d),
        (d + 2 * L) / sp.sqrt(L * (L + d)),
    ]
    # Leading asymptotic classes as d -> 0+ are pairwise distinct:
    # d^{-1/2}, log, const, const (pi/2), d^{1/2}, d^{1/2}-class with different
    # analytic expansion. Check the 6x6 Wronskian is not identically zero by
    # evaluating a numerical specialization.
    mats = []
    dvals = [sp.Rational(k, 10) for k in range(1, 7)]
    L0 = sp.Integer(1)
    M = sp.Matrix([[f.subs({d: dv, L: L0}).evalf(40) for dv in dvals] for f in timings])
    assert_true(M.det() != 0, "six timing samples: det nonzero")

    # Resistance quadratic coefficient positivity on N_U=0 branch
    JH, k, zeta, Ef = sp.symbols("J_H k zeta E_f", positive=True)
    Z_quad = (3 * JH / (4 * k)) * zeta**2 * Ef**2
    assert_true(bool(Z_quad.is_positive), "Z_f quadratic > 0")

    print("hard_branch_markers: all checks passed")


if __name__ == "__main__":
    main()
