# Exact Current–Voltage Fibres of the Doyle–Fuller–Newman Continuum Model: A Structural Classification at Fixed Preparation — Reproducibility Package

This package regenerates the numerical figures and the SymPy algebraic checks
that accompany the manuscript *Exact Current–Voltage Fibres of the Doyle–Fuller–Newman Continuum Model: A Structural Classification at Fixed Preparation*.

Authors: Ding Xu and Changfeng Zhao. License: MIT (see `LICENSE`).

## What is reproduced

| Manuscript item | Script | Output |
| --- | --- | --- |
| Fig. 1 (cell schematic + port) | `scripts/fig1_schematic.py` | `figures/fig1_p2d_port.pdf` |
| Fig. 2 (proof / fibre flow) | `scripts/fig2_flow.py` | `figures/fig2_proof_flow.pdf` |
| Fig. 3 (gauge identity vs off-orbit) | `scripts/fig3_gauge.py` | `figures/fig3_gauge_numerics.pdf`, `reports/fig3_errors.txt` |
| Fig. 4 (fibre geometry) | `scripts/fig4_fibre.py` | `figures/fig4_fibre_dimensions.pdf` |
| Fig. 5 (germ recovery) | `scripts/fig5_germ_recovery.py` | `figures/fig5_germ_recovery.pdf`, `reports/fig5_recovery_report.txt` |
| Fig. 6 (synthetic-data inversion) | `scripts/fig6_inversion.py` (model: `scripts/dfn_grouped.py`, plot: `scripts/fig6_plot.py`) | `figures/fig6_inversion.pdf`, `reports/fig6_inversion_report.txt`, `reports/fig6_inversion_results.npz` |
| SM grouping / gauge algebra | `sympy_checks/symmetry_check.py` | console `PASS` lines |
| SM physical-image rank | `sympy_checks/physical_image_rank.py` | console `PASS` lines + minor |
| SM rank-4 jets / flat three-pulse | `sympy_checks/rank4_flat_markers.py` | console `PASS` |

Table I–V in the main text are LaTeX; numerical residual magnitudes quoted
there for Fig. 3 / Fig. 5 are written to the `reports/` text files.

## Observation setting for Fig. 3

Panels (a)–(c) apply the manuscript gauge group actions \(G_c\) and \(G_s\)
to an explicit reduced-port chart \((\gamma,\nu_e^{n,s,p},r_f,\kappa^s)\).
Voltage depends on that chart only through \((\zeta_q,R_{\mathrm{ser}})\), so
nontrivial group elements leave \(V\) unchanged to floating-point zero.
Panel (d) is a full DFN off-orbit check in PyBaMM (positive-electrode
thickness \(\times 1.5\)). Public PyBaMM parameter charts do not expose the
abstract \(G_c\) coordinates as independent knobs; exact gauge algebra is
therefore shown on the reduced port.

## Setup

Python 3.10+ recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` pins `pybamm==26.10.1.1` (the version used in the paper)
together with NumPy, SciPy, Matplotlib, SymPy, and CasADi.

## One-command run

```bash
python run_all.py
# or
make all
```

Individual targets: `make sympy`, `make figures`.

## Expected Fig. 3 residuals (noise-free)

After `scripts/fig3_gauge.py`, `reports/fig3_errors.txt` should report:

- `rel_gc_max` and `rel_gs_max` equal to `0` (floating-point identity)
- off-orbit relative residuals of order \(10^{-2}\) (\(\Delta R_{\mathrm{ser}}\))
  and \(10^{-1}\) (single \(\nu_e^s\))
- PyBaMM off-orbit \(\lvert\Delta V\rvert_{\max}\approx 20\,\mathrm{mV}\)

Exact floats can vary slightly with BLAS / CasADi builds; gauge identity must
remain at numerical zero and the PyBaMM shift must stay on the order of tens
of millivolts.

## Synthetic-data inversion (Fig. 6)

`scripts/dfn_grouped.py` implements the nondimensional DFN/P2D system of
Appendix A directly in the twenty grouped coordinates, with
\((\gamma,\nu_e^n,\nu_e^s,\nu_e^p)\) and \((r_f,\kappa^s)\) entering
separately as written there (finite volumes: 4/3/4 cells, 4 shells per
particle; CasADi/IDAS with forward sensitivities). `scripts/fig6_inversion.py`
generates synthetic voltage at a known parameter (noise-free and with 0.1 mV
Gaussian noise) and runs a Levenberg–Marquardt fit over all twenty
log-coordinates from 40 random starts (each start: Q18 coordinates within a
factor 1.5 of truth, then a random gauge element). The full run takes about
2 h on 7 worker processes; `python scripts/fig6_inversion.py --plot` redraws
the figure from the saved `reports/fig6_inversion_results.npz`.

Expected (see `reports/fig6_inversion_report.txt`): two singular values of the
sensitivity matrix at the integration tolerance and eighteen above
\(2.9\times10^{-4}\); noise-free fits recover every Q18 coordinate to a
relative error of order \(10^{-6}\) while \(\gamma\) spans a factor of about
100 and \(r_f\) a factor of about 16 across starts. The manuscript numbers
were produced with CasADi 3.8.2.

## Notes

- Paths are relative to the package root; no absolute machine paths are required.
- Figures use Liberation Sans when available (falls back to DejaVu Sans).
- This package is for local reproduction only; it does not publish or push
  anywhere by itself.
