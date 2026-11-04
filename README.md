# Exact Fibres of the Continuum DFN Current–Voltage Germ — Reproducibility

This package regenerates the numerical figures and the SymPy algebraic checks
that accompany the manuscript on structural identifiability of the continuum
Doyle–Fuller–Newman (DFN / P2D) current–voltage germ.

Authors: Ding Xu and Changfeng Zhao. License: MIT (see `LICENSE`).

## What is reproduced

| Manuscript item | Script | Output |
| --- | --- | --- |
| Fig. 1 (cell schematic + port) | `scripts/fig1_schematic.py` | `figures/fig1_p2d_port.pdf` |
| Fig. 2 (proof / fibre flow) | `scripts/fig2_flow.py` | `figures/fig2_proof_flow.pdf` |
| Fig. 3 (gauge identity vs off-orbit) | `scripts/fig3_gauge.py` | `figures/fig3_gauge_numerics.pdf`, `reports/fig3_errors.txt` |
| Fig. 4 (fibre geometry) | `scripts/fig4_fibre.py` | `figures/fig4_fibre_dimensions.pdf` |
| Fig. 5 (germ recovery) | `scripts/fig5_germ_recovery.py` | `figures/fig5_germ_recovery.pdf`, `reports/fig5_recovery_report.txt` |
| SM grouping / gauge algebra | `sympy_checks/symmetry_check.py` | console `PASS` lines |
| SM physical-image rank | `sympy_checks/physical_image_rank.py` | console `PASS` lines + minor |

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

## Notes

- Paths are relative to the package root; no absolute machine paths are required.
- Figures use Liberation Sans when available (falls back to DejaVu Sans).
- This package is for local reproduction only; it does not publish or push
  anywhere by itself.
