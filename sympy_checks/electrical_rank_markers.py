"""Symbolic checks for electrical-rank and beta-integral markers in the SM."""
from __future__ import annotations
import sympy as sp


def main() -> None:
    # Lorentz--Gram identity on rapidity cross terms
    x, y, z = sp.symbols("x y z")
    lorentz = x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z - 2*x*y*z
    # If x=y=z=t (one-dimensional positive span ansatz), cubic coeff of t^3
    t = sp.symbols("t")
    poly = sp.expand(lorentz.subs({x: t, y: t, z: t}))
    # -2 t^3 + ... ; cubic coefficient must be nonzero
    assert sp.Poly(poly, t).degree() >= 3
    assert sp.Poly(poly, t).coeff_monomial(t**3) != 0

    # Beta integral identity used for Volterra constants
    a, n = sp.symbols("alpha n", positive=True)
    s = sp.symbols("s", positive=True)
    # B(a, n*a) = Gamma(a)Gamma(n a)/Gamma((n+1)a)
    # verify for alpha=1/2, n=1,2,3 numerically via exact Gamma
    alpha = sp.Rational(1, 2)
    for nn in range(1, 6):
        lhs = sp.integrate((1 - s) ** (alpha - 1) * s ** (nn * alpha - 1), (s, 0, 1))
        rhs = sp.gamma(alpha) * sp.gamma(nn * alpha) / sp.gamma((nn + 1) * alpha)
        assert sp.simplify(lhs - rhs) == 0, (nn, lhs, rhs)

    # Distinct-feedback det already in hard_branch; check c_m again quickly
    zz = sp.symbols("z", positive=True)
    ell = sp.log(sp.sqrt(zz * (1 - zz)))
    cm = -sp.diff(ell, zz, 2) / 4
    target = (1 / 8) * (1 / zz**2 + 1 / (1 - zz) ** 2)
    assert sp.simplify(cm - target) == 0

    # Hyperbolic K = d_i d_j (cosh(theta_i-theta_j)-1) >= 0 for real theta
    di, dj, thi, thj = sp.symbols("d_i d_j theta_i theta_j", real=True)
    K = di * dj * (sp.cosh(thi - thj) - 1)
    # cosh(u)-1 = 2 sinh^2(u/2) >= 0
    u = sp.symbols("u", real=True)
    assert sp.simplify(sp.cosh(u) - 1 - 2 * sp.sinh(u / 2) ** 2) == 0

    print("electrical_rank_markers: all checks passed")


if __name__ == "__main__":
    main()
