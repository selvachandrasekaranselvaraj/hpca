"""
prerelax_mace_lammps.py — LAMMPS MACE NPT variable-cell pre-relaxation.

Usage:
    python3 prerelax_mace_lammps.py <poscar_path>
        [steps=10000] [temp=300] [ntasks=1]
        [model_type=auto|mace_off|mace_mp]
        [timeout=7200]

Compresses oversized PACKMOL boxes (8× volume) to near-target density via
LAMMPS NPT at T=300 K, P=1 bar with MACE pair potential.

Falls back to ASE UnitCellFilter + FIRE if LAMMPS fails (e.g., no GPU).
On ASE fallback, caps relaxation at 300 steps to stay within timeout.

Backs up original POSCAR to <path>.orig before overwriting.
Exits 0 on success (either path), 1 on total failure.
Prints one-line summary to stdout on success.
"""
from __future__ import annotations

import sys
import os
import shutil
import subprocess
from pathlib import Path

# ── Model and binary paths (all resolved from platform.yaml at runtime) ───────

def _cfg_hpc(key: str, fallback: str = "") -> str:
    """Return the HPC config value for key from platform.yaml, or fallback on any error."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("hpca.core.paths")
        if spec:
            from hpca.core.paths import load_platform_config
            return load_platform_config().get("hpc", {}).get(key, fallback)
    except Exception:
        pass
    return fallback


def _mace_off_model() -> str:
    """Return MACE-OFF23 model path: platform.yaml → mlip_library → ~/.cache/mace."""
    v = _cfg_hpc("mace_off_model", "")
    if v and Path(v).exists():
        return v
    lib = _cfg_hpc("mlip_library", "")
    if lib:
        candidate = str(Path(lib) / "MACE-OFF23_medium.model")
        if Path(candidate).exists():
            return candidate
    import os
    cache = os.path.expanduser("~/.cache/mace")
    for name in ("MACE-OFF23_medium.model", "MACE-OFF23(S).model"):
        p = os.path.join(cache, name)
        if os.path.exists(p):
            return p
    return ""


def _mace_mp_model() -> str:
    """Return MACE-MPA-0 model path: platform.yaml → mlip_library → ~/.cache/mace."""
    v = _cfg_hpc("mace_mp_model", "")
    if v and Path(v).exists():
        return v
    lib = _cfg_hpc("mlip_library", "")
    if lib:
        candidate = str(Path(lib) / "MACE-MPA-0_medium.model")
        if Path(candidate).exists():
            return candidate
    import os
    cache = os.path.expanduser("~/.cache/mace")
    for name in ("macempa0mediummodel", "MACE-MPA-0_medium.model"):
        p = os.path.join(cache, name)
        if os.path.exists(p):
            return p
    return ""


def _mace_create_bin() -> str:
    """Return the path to the mace_create_lammps_model binary."""
    v = _cfg_hpc("mace_create_lammps_bin", "")
    if v:
        return v
    python_deepmd = _cfg_hpc("python_deepmd", "")
    if python_deepmd:
        import os
        candidate = os.path.join(os.path.dirname(python_deepmd), "mace_create_lammps_model")
        if os.path.exists(candidate):
            return candidate
    import shutil
    found = shutil.which("mace_create_lammps_model")
    return found or "mace_create_lammps_model"


def _lmp_bin() -> str:
    """Return the LAMMPS executable path from platform.yaml."""
    return _cfg_hpc("lammps_bin", "lmp")


def _mpirun_bin() -> str:
    """Return the mpirun executable path from platform.yaml or PATH."""
    v = _cfg_hpc("mpirun_bin", "")
    if v:
        return v
    import shutil
    return shutil.which("mpirun") or "mpirun"


def _ensure_lammps_pt(model_path: str) -> str:
    """Convert MACE .model → LAMMPS .pt if not cached. Returns path to .pt."""
    pt_path = model_path + "-lammps.pt"
    if Path(pt_path).exists():
        return pt_path
    result = subprocess.run(
        [_mace_create_bin(), model_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mace_create_lammps_model failed (rc={result.returncode}): "
            f"{result.stderr[:500]}"
        )
    if not Path(pt_path).exists():
        raise RuntimeError(f"mace_create_lammps_model did not produce: {pt_path}")
    return pt_path


def _write_lammps_input(
    work_dir: Path,
    model_pt: str,
    steps: int,
    temp: float,
) -> None:
    """Write LAMMPS NPT input (metal units: ps, bar, Å, eV)."""
    # temp_damp 0.1 ps = 100 fs; press_damp 1.0 ps = 1000 fs
    # pressure 1.0 bar ≈ 1 atm; timestep 0.001 ps = 1 fs
    inp = (
        "units        metal\n"
        "atom_style   atomic\n"
        "boundary     p p p\n"
        "read_data    data.lammps\n"
        "\n"
        f"pair_style   mace no_domain_decomposition {model_pt}\n"
        "pair_coeff   * *\n"
        "\n"
        "timestep     0.001\n"
        "\n"
        "thermo       100\n"
        "thermo_style custom step temp press vol density pe\n"
        "\n"
        "# Remove bad contacts from PACKMOL packing\n"
        "minimize     1.0e-4 1.0e-6 200 2000\n"
        "reset_timestep 0\n"
        "\n"
        "# NPT: isotropic variable-cell at T, P=1 bar ≈ 1 atm\n"
        f"velocity     all create {temp:.1f} 42317 dist gaussian\n"
        f"fix          npt all npt temp {temp:.1f} {temp:.1f} 0.1 iso 1.0 1.0 1.0\n"
        f"run          {steps}\n"
        "unfix        npt\n"
        "\n"
        "write_data   final_npt.lmp\n"
    )
    (work_dir / "in.lammps").write_text(inp)


def _run_lammps(work_dir: Path, ntasks: int, timeout: int) -> bool:
    """Run LAMMPS. Returns True on success, False on failure/timeout."""
    log_path = work_dir / "lammps.log"
    if ntasks > 1 and Path(_mpirun_bin()).exists():
        cmd = [_mpirun_bin(), "-np", str(ntasks), _lmp_bin(), "-in", "in.lammps"]
    else:
        cmd = [_lmp_bin(), "-in", "in.lammps"]
    # Ensure conda env libs are findable (LAMMPS is linked against libpython3.10)
    _conda_lib = str(Path(sys.executable).parents[1] / "lib")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        with open(log_path, "w") as fout:
            proc = subprocess.Popen(
                cmd, cwd=str(work_dir), stdout=fout, stderr=subprocess.STDOUT, env=env,
            )
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"WARN: LAMMPS timed out after {timeout}s", file=sys.stderr)
            return False
        if proc.returncode != 0:
            # Surface last few lines of log for diagnostics
            tail = ""
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                tail = "\n".join(lines[-10:])
            print(f"WARN: LAMMPS exited rc={proc.returncode}\n{tail}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(f"WARN: LAMMPS binary not found: {_lmp_bin()}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"WARN: LAMMPS run error: {exc}", file=sys.stderr)
        return False


def _ase_variable_cell_fallback(
    poscar_path: Path,
    model_type: str,
    fmax: float = 0.5,
    steps: int = 300,
) -> bool:
    """ASE UnitCellFilter + FIRE fallback (CPU, variable cell). Returns True on success."""
    try:
        from ase.io import read, write
        try:
            from ase.filters import UnitCellFilter
        except ImportError:
            from ase.constraints import UnitCellFilter
        from ase.optimize import FIRE

        atoms = read(str(poscar_path), format="vasp")

        if model_type in ("mace_off", "auto") and Path(_mace_off_model()).exists():
            try:
                from mace.calculators import mace_off
                calc = mace_off(model="medium", device="cpu")
            except Exception:
                from mace.calculators import mace_mp
                calc = mace_mp(model="medium", device="cpu", default_dtype="float32")
        else:
            from mace.calculators import mace_mp
            calc = mace_mp(model="medium", device="cpu", default_dtype="float32")

        atoms.calc = calc
        ucf = UnitCellFilter(atoms, scalar_pressure=0.0)
        opt = FIRE(ucf, logfile=None)
        opt.run(fmax=fmax, steps=min(steps, 300))

        n_done = opt.get_number_of_steps()
        vol    = atoms.get_volume()
        cell_L = atoms.cell.lengths()
        write(str(poscar_path), atoms, format="vasp", vasp5=True, direct=True, sort=True)
        print(
            f"RELAXED_ASE: steps={n_done} "
            f"cell=[{cell_L[0]:.1f},{cell_L[1]:.1f},{cell_L[2]:.1f}]Å "
            f"vol={vol:.1f}Å³"
        )
        return True
    except Exception as exc:
        print(f"ERROR: ASE variable-cell fallback failed: {exc}", file=sys.stderr)
        return False


def main():
    """Entry point — parse args and run the LAMMPS MACE NPT pre-relaxation."""
    if len(sys.argv) < 2:
        print(
            "Usage: python3 prerelax_mace_lammps.py <poscar_path> [key=val ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    poscar_path = Path(sys.argv[1])
    if not poscar_path.exists():
        print(f"ERROR: POSCAR not found: {poscar_path}", file=sys.stderr)
        sys.exit(1)

    kwargs: dict[str, str] = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kwargs[k.strip()] = v.strip()

    steps      = int(  kwargs.get("steps",      "10000"))
    temp       = float(kwargs.get("temp",        "300"))
    ntasks     = int(  kwargs.get("ntasks",      "1"))
    model_type = kwargs.get("model_type", "auto")
    timeout    = int(  kwargs.get("timeout",     "7200"))

    # Choose source MACE model
    if model_type == "mace_mp":
        src_model = _mace_mp_model()
    else:  # auto or mace_off → prefer OFF23 for organics
        src_model = (
            _mace_off_model() if Path(_mace_off_model()).exists() else _mace_mp_model()
        )

    # Backup original
    orig_path = Path(str(poscar_path) + ".orig")
    shutil.copy2(poscar_path, orig_path)

    work_dir = poscar_path.parent / f"_lmp_preopt_{poscar_path.stem}"
    work_dir.mkdir(exist_ok=True)

    lammps_success = False
    try:
        from ase.io import read, write

        atoms   = read(str(poscar_path), format="vasp")
        n_atoms = len(atoms)

        # Ensure LAMMPS-compatible .pt model exists (lazy conversion)
        lammps_pt = _ensure_lammps_pt(src_model)

        # Write LAMMPS data file (ASE encodes element→type via Masses section)
        write(
            str(work_dir / "data.lammps"), atoms,
            format="lammps-data", atom_style="atomic",
        )

        _write_lammps_input(work_dir, lammps_pt, steps, temp)

        lammps_success = _run_lammps(work_dir, ntasks, timeout)

        if lammps_success and (work_dir / "final_npt.lmp").exists():
            atoms_out = read(str(work_dir / "final_npt.lmp"), format="lammps-data")
            vol    = atoms_out.get_volume()
            cell_L = atoms_out.cell.lengths()
            write(
                str(poscar_path), atoms_out,
                format="vasp", vasp5=True, direct=True, sort=True,
            )
            shutil.rmtree(work_dir, ignore_errors=True)
            print(
                f"NPT_DONE: atoms={n_atoms} steps={steps} temp={temp:.0f}K "
                f"cell=[{cell_L[0]:.1f},{cell_L[1]:.1f},{cell_L[2]:.1f}]Å "
                f"vol={vol:.1f}Å³"
            )
            sys.exit(0)
        else:
            print("WARN: LAMMPS NPT failed — falling back to ASE variable-cell",
                  file=sys.stderr)

    except Exception as exc:
        print(f"WARN: LAMMPS path raised {exc} — falling back to ASE", file=sys.stderr)

    # ASE fallback: restore original in case LAMMPS partially modified things
    shutil.copy2(orig_path, poscar_path)

    ase_ok = _ase_variable_cell_fallback(poscar_path, model_type, fmax=0.5, steps=300)
    shutil.rmtree(work_dir, ignore_errors=True)
    sys.exit(0 if ase_ok else 1)


if __name__ == "__main__":
    main()
