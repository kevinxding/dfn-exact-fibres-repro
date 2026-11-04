#!/usr/bin/env python3
"""One-command entry: SymPy checks + all manuscript figures/reports."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("sympy: symmetry_check", [sys.executable, str(ROOT / "sympy_checks" / "symmetry_check.py")]),
    ("sympy: physical_image_rank", [sys.executable, str(ROOT / "sympy_checks" / "physical_image_rank.py")]),
    ("sympy: hard_branch_markers", [sys.executable, str(ROOT / "sympy_checks" / "hard_branch_markers.py")]),
    ("sympy: electrical_rank_markers", [sys.executable, str(ROOT / "sympy_checks" / "electrical_rank_markers.py")]),
    ("fig1", [sys.executable, str(ROOT / "scripts" / "fig1_schematic.py")]),
    ("fig2", [sys.executable, str(ROOT / "scripts" / "fig2_flow.py")]),
    ("fig3", [sys.executable, str(ROOT / "scripts" / "fig3_gauge.py")]),
    ("fig4", [sys.executable, str(ROOT / "scripts" / "fig4_fibre.py")]),
    ("fig5", [sys.executable, str(ROOT / "scripts" / "fig5_germ_recovery.py")]),
]


def main() -> int:
    (ROOT / "figures" / "png").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    for label, cmd in STEPS:
        print(f"\n=== {label} ===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"FAILED: {label}", file=sys.stderr)
            return r.returncode
    print("\nAll steps finished. Outputs under figures/ and reports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
