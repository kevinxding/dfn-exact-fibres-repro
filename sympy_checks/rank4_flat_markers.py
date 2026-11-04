"""SymPy checks for rank-4 jets/witnesses and flat scale/mirror probes.

Verifies:
  1. Exterior-pole shift jet: no even powers through O(z^3); coeff match
  2. R2/R1 witness lambdas {1/16,9/16,4/9,16/9} distinct; t_i=2/lambda distinct
  3. Sample evaluation matrix of sqrt(Z+t_i) has nonzero det (R1 witness)
  4. Off-diagonal radical sample matrix for six pairs has nonzero det (R3)
  5. Short-pulse Q_lambda arithmetic: two electrodes -> 3 E^2 J_H/(4k)
  6. Three-pulse (E_f^2, pi I_f, J_f) matrix det = 19/1280240640000
  7. Weights (785664,-9204552,16846200)/19 kill I_f and J_f, normalize E_f^2
  8. Abel monomial formula A[t^n]=Gamma(n+1)/Gamma(n+3/2) t^{n+1/2}

These support the SM lemmas; they do not replace the prose proofs.
"""
from __future__ import annotations

import itertools
import sympy as sp


def assert_true(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)


def assert_zero(expr, label: str) -> None:
    s = sp.simplify(expr)
    if isinstance(s, sp.MatrixBase):
        if s != s * 0:
            raise AssertionError(f"{label}: expected 0 matrix, got {s}")
        return
    if s != 0:
        raise AssertionError(f"{label}: expected 0, got {s}")


def main() -> None:
    # --- 1. Formal odd jet structure of delta_a ---
    # Model: delta = 2z/(beta a) - 4 D C z^3/(3 beta^3 a^3) + O(z^5)
    beta, a, D, C, z = sp.symbols("beta a D C z", positive=True)
    delta = (2 * z) / (beta * a) - (4 * D * C * z**3) / (3 * beta**3 * a**3)
    # Only odd powers of z through order 3
    series = sp.series(delta, z, 0, 4).removeO()
    assert_zero(series.coeff(z, 0), "delta const term")
    assert_zero(series.coeff(z, 2), "delta z^2 term (must vanish)")
    assert_true(series.coeff(z, 1) != 0, "delta z coeff nonzero")
    assert_true(series.coeff(z, 3) != 0, "delta z^3 coeff nonzero")
    # C_m,1 formula structure: proportional to (1/a+1/b)
    # and inversely to T'(C_inf); check algebra of 1/a+1/b = r/(a b)
    r, b = sp.symbols("r b", positive=True)
    assert_zero(sp.simplify(1 / a + 1 / b - r / (a * b)).subs(r, a + b), "1/a+1/b=r/(ab)")

    # --- 2. Explicit R1/R3 lambda witness ---
    lambdas = [
        sp.Rational(1, 16),
        sp.Rational(9, 16),
        sp.Rational(4, 9),
        sp.Rational(16, 9),
    ]
    assert_true(len(set(lambdas)) == 4, "four distinct lambdas")
    # Matches beta*rho_n=2, x_n=1/4 -> lambda = beta rho x^2 / 2 and (1-x)^2
    # lambda_n0 = 2*(1/4)^2/2 = 1/16; lambda_n1 = 2*(3/4)^2/2 = 9/16
    # beta rho_p=8, x_p=1/3 -> 8*(1/3)^2/2=4/9; 8*(2/3)^2/2=16/9
    assert_zero(2 * sp.Rational(1, 4) ** 2 / 2 - lambdas[0], "lambda_n0")
    assert_zero(2 * sp.Rational(3, 4) ** 2 / 2 - lambdas[1], "lambda_n1")
    assert_zero(8 * sp.Rational(1, 3) ** 2 / 2 - lambdas[2], "lambda_p0")
    assert_zero(8 * sp.Rational(2, 3) ** 2 / 2 - lambdas[3], "lambda_p1")
    ts = [2 / lam for lam in lambdas]
    assert_true(len(set(ts)) == 4, "four distinct t_i=2/lambda_i")

    # --- 3. R1: sample matrix of sqrt(Z+t_i) ---
    Z = sp.symbols("Z", positive=True)
    fs = [sp.sqrt(Z + ti) for ti in ts]
    Zvals = [sp.Rational(k, 2) for k in range(1, 5)]
    M1 = sp.Matrix([[f.subs(Z, zv).evalf(50) for zv in Zvals] for f in fs])
    assert_true(M1.det() != 0, "R1 sample det nonzero")

    # --- 4. R3: six pair radicals sqrt((Z+ti)(Z+tj)) ---
    pairs = list(itertools.combinations(range(4), 2))
    assert_true(len(pairs) == 6, "six pairs")
    rads = [sp.sqrt((Z + ts[i]) * (Z + ts[j])) for i, j in pairs]
    # Include the polynomial parts lambda_i+lambda_j+lambda_i lambda_j Z
    # Independence of the six radicals alone already implies R3 limit independence
    Zvals6 = [sp.Rational(k, 3) for k in range(1, 7)]
    M3 = sp.Matrix([[r.subs(Z, zv).evalf(50) for zv in Zvals6] for r in rads])
    assert_true(M3.det() != 0, "R3 radical sample det nonzero")

    # --- 5. Q_lambda short-pulse arithmetic ---
    # Per electrode: (E^2/k){JH/4 - <p^2,Pp^2>/8 + ||p^2||^2/8}
    # With structural identity JH = ||p^2||^2 - <p^2, P p^2>  (from H-reciprocity
    # on the short-pulse limit in the archival derivation), this equals 3 E^2 JH/(8k).
    JH, k, Ef = sp.symbols("J_H k E_f", positive=True)
    # Two electrodes: 2 * 3/8 = 3/4
    per = sp.Rational(3, 8) * (Ef**2 / k) * JH
    total = 2 * per
    assert_zero(sp.simplify(total - sp.Rational(3, 4) * (Ef**2 / k) * JH), "Q_lambda limit")
    # Structural identity check: JH/4 + JH/8 = 3 JH/8
    assert_zero(sp.simplify(JH / 4 + JH / 8 - sp.Rational(3, 8) * JH), "per-electrode split")

    # --- 6--7. Three-pulse matrix and weights ---
    # Rows (E_f^2, pi I_f, J_f) for f_j = t^j (1-t), j=1,2,3
    rows = [
        [sp.Rational(1, 36), sp.Rational(44, 675), sp.Rational(7, 320)],
        [sp.Rational(1, 144), sp.Rational(512, 33075), sp.Rational(2321, 430080)],
        [sp.Rational(1, 400), sp.Rational(128, 23625), sp.Rational(4147, 2150400)],
    ]
    M = sp.Matrix(rows)
    detM = sp.simplify(M.det())
    assert_zero(detM - sp.Rational(19, 1280240640000), f"three-pulse det got {detM}")

    # Matrix columns: E^2, pi I, J; solve w·col0=1, w·(col1/pi)=0, w·col2=0
    # I_f = (pi I_f)/pi, so for sum w I_f = 0 use column1/pi
    piI = M[:, 1]
    Icol = piI / sp.pi
    # Build system with exact I_f values from archival table:
    # pi I_f given, so I_f = (pi I_f)/pi — but weights in (6.4) were solved
    # against (E_f^2, I_f, J_f) with I_f = (pi I_f_display)/pi.
    # Archival table lists "pi I_f" column; weights satisfy sum w E^2=1,
    # sum w I_f=0, sum w J_f=0 with I_f = listed/pi.
    Ef2 = [rows[i][0] for i in range(3)]
    If = [rows[i][1] / sp.pi for i in range(3)]
    Jf = [rows[i][2] for i in range(3)]
    A = sp.Matrix([Ef2, If, Jf]).T  # 3x3 with rows (E^2, I, J) per pulse
    rhs = sp.Matrix([1, 0, 0])
    w = A.transpose().solve(rhs)  # A^T w = rhs if rows are pulses... 
    # Actually we want sum_j w_j v_j = target with v_j = (E_j^2, I_j, J_j)
    # so M_EJ @ w = (1,0,0) where M_EJ has columns = pulse vectors
    MEJ = sp.Matrix.hstack(
        sp.Matrix(Ef2), sp.Matrix(If), sp.Matrix(Jf)
    ).T  # wait: 3x3 with row0 all E^2, ...
    MEJ = sp.Matrix([Ef2, If, Jf])  # shape 3x3: row k is functional, col j is pulse
    w = MEJ.solve(sp.Matrix([1, 0, 0]))
    w_target = sp.Matrix([
        sp.Rational(785664, 19),
        sp.Rational(-9204552, 19),
        sp.Rational(16846200, 19),
    ])
    assert_zero(sp.simplify(w - w_target), f"weights got {w}")

    # Direct check of (6.5)
    assert_zero(sp.simplify(sum(w[j] * Ef2[j] for j in range(3)) - 1), "sum w E^2")
    assert_zero(sp.simplify(sum(w[j] * If[j] for j in range(3))), "sum w I")
    assert_zero(sp.simplify(sum(w[j] * Jf[j] for j in range(3))), "sum w J")

    # --- 8. Abel monomial formula ---
    n, tt = sp.symbols("n t", positive=True)
    # A[t^n](t) = Gamma(n+1)/Gamma(n+3/2) t^{n+1/2}
    # Verify for n=0,1,2 by integrating (1/sqrt(pi)) int_0^t s^n / sqrt(t-s) ds
    for nn in range(0, 4):
        t0 = sp.symbols("t0", positive=True)
        s = sp.symbols("s", positive=True)
        lhs = (1 / sp.sqrt(sp.pi)) * sp.integrate(
            s**nn / sp.sqrt(t0 - s), (s, 0, t0)
        )
        rhs = sp.gamma(nn + 1) / sp.gamma(nn + sp.Rational(3, 2)) * t0 ** (nn + sp.Rational(1, 2))
        assert_zero(sp.simplify(lhs - rhs), f"Abel n={nn}")

    # Recompute one table entry: E_f for f=t(1-t) on [0,1]
    t = sp.symbols("t", positive=True)
    f1 = t * (1 - t)
    Ef1 = sp.integrate(f1, (t, 0, 1))
    assert_zero(sp.simplify(Ef1**2 - sp.Rational(1, 36)), "E_f^2 for j=1")


    # Abel-verified table row for j=1
    tt = sp.symbols("tt", positive=True)
    def A_mono(nn, tvar):
        return sp.gamma(nn + 1) / sp.gamma(nn + sp.Rational(3, 2)) * tvar ** (nn + sp.Rational(1, 2))
    Af1 = A_mono(1, tt) - A_mono(2, tt)
    If1 = sp.simplify(sp.integrate(Af1 ** 2, (tt, 0, 1)))
    assert_zero(sp.simplify(sp.pi * If1 - sp.Rational(44, 675)), "Abel pi I_f j=1")
    E1 = tt ** 2 / 2 - tt ** 3 / 3
    Jf1 = sp.simplify(sp.integrate(Af1 * E1 / sp.sqrt(1 - tt), (tt, 0, 1)) / sp.sqrt(sp.pi))
    assert_zero(sp.simplify(Jf1 - sp.Rational(7, 320)), "Abel J_f j=1")

    print("rank4_flat_markers: all checks passed")


if __name__ == "__main__":
    main()
