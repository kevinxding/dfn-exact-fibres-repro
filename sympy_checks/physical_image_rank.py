"""Exact log-exponent audit of the selected physical grouping map.

This script proves algebraic rank/relations only.  It does not prove forward
well-posedness or input-output fibre completeness.
"""

import sympy as sp


b = sp.symbols("b", real=True)

variables = (
    "A", "L_n", "L_s", "L_p", "R_n", "R_p", "D_s_n", "D_s_p",
    "D_e", "kappa_e", "sigma_s_n", "sigma_s_p", "k_n", "k_p",
    "R_f", "h", "eps_e_n", "eps_e_s", "eps_e_p", "eps_s_n",
    "eps_s_p", "k_f",
)

grouped = (
    "tau_ds_n", "tau_ds_p", "tau_cs_n", "tau_cs_p", "tau_k_n",
    "tau_k_p", "tau_de_n", "tau_de_s", "tau_de_p", "nu_n", "nu_s",
    "nu_p", "kappa_n", "kappa_s", "kappa_p", "sigma_n", "sigma_p",
    "r_f", "gamma", "chi_f",
)


def exponent_row(**powers):
    return [sp.sympify(powers.get(name, 0)) for name in variables]


M = sp.Matrix(
    [
        exponent_row(R_n=2, D_s_n=-1),
        exponent_row(R_p=2, D_s_p=-1),
        exponent_row(A=1, L_n=1, eps_s_n=1),
        exponent_row(A=1, L_p=1, eps_s_p=1),
        exponent_row(R_n=1, k_n=-1),
        exponent_row(R_p=1, k_p=-1),
        exponent_row(L_n=2, eps_e_n=1 - b, D_e=-1),
        exponent_row(L_s=2, eps_e_s=1 - b, D_e=-1),
        exponent_row(L_p=2, eps_e_p=1 - b, D_e=-1),
        exponent_row(L_n=1, eps_e_n=1),
        exponent_row(L_s=1, eps_e_s=1),
        exponent_row(L_p=1, eps_e_p=1),
        exponent_row(A=1, kappa_e=1, eps_e_n=b, L_n=-1),
        exponent_row(A=1, kappa_e=1, eps_e_s=b, L_s=-1),
        exponent_row(A=1, kappa_e=1, eps_e_p=b, L_p=-1),
        exponent_row(A=1, sigma_s_n=1, eps_s_n=1, L_n=-1),
        exponent_row(A=1, sigma_s_p=1, eps_s_p=1, L_p=-1),
        exponent_row(R_f=1, A=-1),
        exponent_row(h=1, A=-1),
        exponent_row(h=1, k_f=1),
    ]
)


def sparse(vector, labels):
    return {labels[i]: sp.simplify(value) for i, value in enumerate(vector) if value != 0}


# An explicit maximal minor proves rank 18 for every physical b > 0.
row_indices = M.T.rref()[1]
column_indices = M.rref()[1]
minor = sp.factor(M.extract(row_indices, column_indices).det())
assert minor == -(b + 1) ** 2
assert M.rank() == 18

left_relations = M.T.nullspace()
assert len(left_relations) == 2
expected_relations = [
    {
        "tau_de_n": -1,
        "tau_de_s": 1,
        "nu_n": 1,
        "nu_s": -1,
        "kappa_n": -1,
        "kappa_s": 1,
    },
    {
        "tau_de_n": -1,
        "tau_de_p": 1,
        "nu_n": 1,
        "nu_p": -1,
        "kappa_n": -1,
        "kappa_p": 1,
    },
]
assert [sparse(v, grouped) for v in left_relations] == expected_relations

# Replace nu_q by zeta_q=nu_q/gamma and remove gamma.
gamma_row = M[grouped.index("gamma"), :]
reduced_rows = []
reduced_names = []
for name in grouped:
    if name == "gamma":
        continue
    if name.startswith("nu_"):
        reduced_rows.append(M[grouped.index(name), :] - gamma_row)
        reduced_names.append(name.replace("nu_", "zeta_"))
    else:
        reduced_rows.append(M[grouped.index(name), :])
        reduced_names.append(name)
M_reduced = sp.Matrix.vstack(*reduced_rows)
assert M_reduced.shape == (19, 22)
assert M_reduced.rank() == 17
assert len(M_reduced.T.nullspace()) == 2
assert len(M.nullspace()) == 4
assert len(M_reduced.nullspace()) == 5

# Candidate physical lift: all grouped coordinates are fixed except nu_q and
# gamma, which acquire log-increment +1.
lift = sp.Matrix(
    [
        -1, 0, 0, 0, 0, 0, 0, 0, 1 - b, 1 - b, 0, 0, 0, 0, -1,
        0, 1, 1, 1, 1, 1, 0,
    ]
)
lift_image = M * lift
expected_lift_image = sp.Matrix(
    [1 if name in {"nu_n", "nu_s", "nu_p", "gamma"} else 0 for name in grouped]
)
assert sp.simplify(lift_image - expected_lift_image) == sp.zeros(20, 1)

print("PASS: physical grouped image has rank 18 for b > 0")
print("PASS: exactly two log-linear relations; reduced physical image has rank 17")
print("PASS: physical grouping fibres have local dimension 4; reduced fibres 5")
print("maximal minor:", minor)
for relation in expected_relations:
    print("relation:", relation)

if __name__ == "__main__":
    print("physical_image_rank: rank and relations OK")
