"""
prerelax_mace.py — Standalone MACE-MPA-0 pre-relaxation script.

Usage:
    python3 prerelax_mace.py <poscar_path> [fmax=0.1] [steps=300] [device=cpu]

Relaxes a POSCAR with MACE-MPA-0 (medium model) using ASE FIRE optimizer.
Backs up original to <poscar_path>.orig before writing relaxed structure.
If relaxation fails, restores original and exits with code 1.

Prints one-line summary:
    RELAXED: E=<eV:.3f> fmax=<fmax:.4f> steps=<n>
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path


def main():
    """Entry point — parse args and run the MACE-MPA-0 ASE FIRE pre-relaxation."""
    if len(sys.argv) < 2:
        print("Usage: python3 prerelax_mace.py <poscar_path> [fmax=0.1] [steps=300] [device=cpu]",
              file=sys.stderr)
        sys.exit(1)

    poscar_path = Path(sys.argv[1])
    if not poscar_path.exists():
        print(f"ERROR: POSCAR not found: {poscar_path}", file=sys.stderr)
        sys.exit(1)

    # Parse optional keyword=value arguments
    kwargs: dict[str, str] = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kwargs[k.strip()] = v.strip()

    fmax   = float(kwargs.get("fmax",   "0.1"))
    steps  = int(  kwargs.get("steps",  "300"))
    device = kwargs.get("device", "cpu")
    model_type = kwargs.get("model_type", "mace_mp")  # "mace_mp" or "mace_off"

    # Back up original
    orig_path = Path(str(poscar_path) + ".orig")
    shutil.copy2(poscar_path, orig_path)

    try:
        from ase.io import read, write
        from ase.optimize import FIRE

        atoms = read(str(poscar_path), format="vasp")

        if model_type == "mace_off":
            try:
                from mace.calculators import mace_off
                calc = mace_off(model="medium", device=device)
            except Exception:
                from mace.calculators import mace_mp
                calc = mace_mp(model="medium", device=device, default_dtype="float32")
        else:
            from mace.calculators import mace_mp
            calc = mace_mp(model="medium", device=device, default_dtype="float32")
        atoms.calc = calc

        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)

        n_steps = opt.get_number_of_steps()
        e = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax_actual = float((forces ** 2).sum(axis=1).max() ** 0.5)

        # Write relaxed structure back (sort=True groups species, vasp5=True writes element names)
        write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)

        print(f"RELAXED: E={e:.3f} fmax={fmax_actual:.4f} steps={n_steps}")
        sys.exit(0)

    except Exception as exc:
        # Restore original on any failure
        shutil.copy2(orig_path, poscar_path)
        print(f"ERROR: MACE relaxation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
