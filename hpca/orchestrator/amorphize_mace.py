#!/usr/bin/env python3
"""
amorphize_mace.py — Melt-quench amorphization using MACE-MPA-0 + ASE MD.

Usage:
  python3 amorphize_mace.py <poscar_path> [T_melt=3000] [T_quench=300]
                            [steps_melt=500] [steps_quench=300]
                            [out=<path>] [device=cpu]

Steps:
  1. Load crystal POSCAR
  2. Run NVT MD at T_melt (FIRE → Langevin to disorder the structure)
  3. Ramp temperature down to T_quench over steps_quench steps
  4. Write amorphous POSCAR to <out> (default: <poscar>.amorphous)

Prints: AMORPHIZED: T_melt={K} steps={n} fmax={f:.4f}
"""
from __future__ import annotations

import sys
from pathlib import Path


def _parse_args() -> dict:
    """Parse command-line arguments into a dict of amorphization parameters."""
    args: dict = {
        "poscar": None,
        "T_melt": 3000,
        "T_quench": 300,
        "steps_melt": 500,
        "steps_quench": 300,
        "out": None,
        "device": "cpu",
    }
    for tok in sys.argv[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k in ("T_melt", "T_quench", "steps_melt", "steps_quench"):
                args[k] = int(v)
            elif k in ("out", "device"):
                args[k] = v
        elif args["poscar"] is None:
            args["poscar"] = tok
    return args


def main() -> None:
    """Entry point — parse args and run the MACE melt-quench amorphization."""
    args = _parse_args()
    poscar_path = Path(args["poscar"])
    out_path = Path(args["out"]) if args["out"] else poscar_path.with_suffix(".amorphous")

    from ase.io import read, write
    from ase.md.langevin import Langevin
    from ase import units

    atoms = read(str(poscar_path), format="vasp")

    try:
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", device=args["device"], default_dtype="float32")
    except Exception as exc:
        print(f"ERROR: Cannot load MACE calculator: {exc}", file=sys.stderr)
        sys.exit(1)

    atoms.calc = calc

    # ── Phase 1: melt at T_melt ───────────────────────────────────────────────
    T_melt = args["T_melt"]
    steps_melt = args["steps_melt"]
    dt = 2.0 * units.fs

    md_melt = Langevin(atoms, dt, temperature_K=T_melt, friction=0.01)
    md_melt.run(steps_melt)

    # ── Phase 2: quench to T_quench ───────────────────────────────────────────
    T_quench = args["T_quench"]
    steps_quench = args["steps_quench"]

    # Linearly ramp temperature in 10 blocks
    n_blocks = 10
    steps_per_block = max(1, steps_quench // n_blocks)
    for block in range(n_blocks):
        T_block = T_melt - (T_melt - T_quench) * (block + 1) / n_blocks
        md_q = Langevin(atoms, dt, temperature_K=T_block, friction=0.05)
        md_q.run(steps_per_block)

    write(str(out_path), atoms, format="vasp", vasp5=True, sort=True)

    # Compute max force for reporting
    try:
        import numpy as np
        forces = atoms.get_forces()
        fmax = float(np.linalg.norm(forces, axis=1).max())
    except Exception:
        fmax = float("nan")

    print(f"AMORPHIZED: T_melt={T_melt} T_quench={T_quench} "
          f"steps_melt={steps_melt} steps_quench={steps_quench} fmax={fmax:.4f}")


if __name__ == "__main__":
    main()
