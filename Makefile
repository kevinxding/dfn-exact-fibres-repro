PYTHON ?= python3
.PHONY: all figures sympy inversion inversion-plot clean

all: sympy figures

sympy:
	$(PYTHON) sympy_checks/symmetry_check.py
	$(PYTHON) sympy_checks/physical_image_rank.py

figures:
	$(PYTHON) scripts/fig1_schematic.py
	$(PYTHON) scripts/fig2_flow.py
	$(PYTHON) scripts/fig3_gauge.py
	$(PYTHON) scripts/fig4_fibre.py
	$(PYTHON) scripts/fig5_germ_recovery.py

inversion:
	$(PYTHON) scripts/fig6_inversion.py

inversion-plot:
	$(PYTHON) scripts/fig6_inversion.py --plot

clean:
	rm -f figures/*.pdf figures/png/*.png reports/*.txt reports/*.npz
