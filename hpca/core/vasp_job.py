"""
vasp_job.py — Standalone VASP utility functions for all handlers.

Provides INCAR writing, OUTCAR reading, POTCAR parsing, and SLURM script
generation as importable functions (not classmethods), eliminating duplicated
code across DFT, AIMD, and NEB handlers.
"""
from __future__ import annotations

import re
from pathlib import Path

from hpca.core.config import Config
from hpca.core.config import account_fallback as _account_fallback


# ---------------------------------------------------------------------------
# INCAR helpers
# ---------------------------------------------------------------------------

def incar_text(params: dict) -> str:
    """Return an INCAR file as a string built from *params*.

    Rules:
    - Keys are sorted alphabetically.
    - ``None`` values are skipped.
    - Python ``True`` / ``False`` become ``.TRUE.`` / ``.FALSE.``.
    - Everything else is written as-is with ``str()``.
    """
    lines: list[str] = []
    for key in sorted(params):
        val = params[key]
        if val is None:
            continue
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        else:
            val_str = str(val)
        lines.append(f"{key} = {val_str}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_incar(path: Path, params: dict, system_name: str = "") -> None:
    """Write a VASP INCAR file at *path* from *params*.

    Parameters
    ----------
    params:
        Dict of INCAR key → value.  Keys are written in insertion order.
        ``None`` values are skipped.  ``True``/``False`` → ``.TRUE.``/``.FALSE.``.
    system_name:
        Optional value for the ``SYSTEM`` tag written as the first line.
    """
    lines: list[str] = []
    if system_name:
        lines.append(f"SYSTEM = {system_name}")
    for key, val in params.items():
        if val is None:
            continue
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        else:
            val_str = str(val)
        lines.append(f"{key:<12} = {val_str}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_incar(path: Path, patches: dict) -> None:
    """Read an existing INCAR, apply *patches*, and write back.

    - Existing ``KEY = VALUE`` lines are updated in-place (preserving comments
      on other lines).
    - Keys in *patches* that do not yet exist are appended at the end.
    - ``None`` values in *patches* are skipped (not removed).
    """
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = ""

    lines = text.splitlines(keepends=True)
    remaining = dict(patches)  # keys still to be applied

    new_lines: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*)([A-Z_]+)(\s*=\s*)(.*?)(\s*)$", line.rstrip("\n"))
        if m and m.group(2) in remaining:
            key = m.group(2)
            val = remaining.pop(key)
            if val is None:
                new_lines.append(line)  # keep unchanged
                continue
            if isinstance(val, bool):
                val_str = ".TRUE." if val else ".FALSE."
            else:
                val_str = str(val)
            # preserve leading whitespace and spacing around '='
            new_lines.append(f"{m.group(1)}{key}{m.group(3)}{val_str}\n")
        else:
            new_lines.append(line if line.endswith("\n") else line + "\n")

    # Append any keys that were not found in the existing file
    for key in sorted(remaining):
        val = remaining[key]
        if val is None:
            continue
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        else:
            val_str = str(val)
        new_lines.append(f"{key} = {val_str}\n")

    path.write_text("".join(new_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# KPOINTS
# ---------------------------------------------------------------------------

def write_kpoints_from_poscar(
    path: Path,
    poscar: Path,
    gamma: bool = True,
) -> None:
    """Write a Gamma-centred KPOINTS file using the platform.yaml lookup table.

    Reads lattice-vector lengths from *poscar*, looks up the mesh in
    ``kpoints_rules`` in platform.yaml, and writes the result to *path*.

    Parameters
    ----------
    path:
        Destination KPOINTS file.
    poscar:
        POSCAR/CONTCAR from which lattice-vector lengths are read.
        Falls back to ``1 1 1`` if the file does not exist.
    gamma:
        If *True* (default), write a Gamma-centred mesh.  Pass *False*
        for Monkhorst-Pack.
    """
    from hpca.core.kpoints import kpoints_from_poscar, write_kpoints
    mesh = kpoints_from_poscar(poscar) if poscar.exists() else (1, 1, 1)
    write_kpoints(path, mesh, gamma=gamma)


# ---------------------------------------------------------------------------
# POTCAR
# ---------------------------------------------------------------------------

def generate_potcar(poscar: Path, dest: Path) -> None:
    """Build *dest* POTCAR from the elements listed in *poscar*.

    Reads the PAW prefix table from ``hpc.potcar_prefix`` in platform.yaml
    and the base directory from ``hpc.potpaw_dir``.  Raises
    :exc:`FileNotFoundError` if a source POTCAR is missing, :exc:`KeyError`
    if an element is not in the prefix table.
    """
    from hpca.core.potcar import build_potcar
    elements = read_poscar_elements(poscar)
    if not elements:
        raise ValueError(f"Cannot determine elements from {poscar}")
    build_potcar(elements, dest)


# ---------------------------------------------------------------------------
# OUTCAR readers
# ---------------------------------------------------------------------------

def outcar_converged(outcar_path: Path) -> bool:
    """Return True if ``"reached required accuracy"`` appears in the last 4000 bytes."""
    p = Path(outcar_path)
    if not p.exists():
        return False
    try:
        tail = p.read_text(encoding="utf-8", errors="replace")[-4000:]
        return "reached required accuracy" in tail
    except Exception:
        return False


def outcar_last_energy(outcar_path: Path) -> float | None:
    """Return the last ``energy  without entropy`` value from OUTCAR, or None."""
    p = Path(outcar_path)
    if not p.exists():
        return None
    try:
        energy: float | None = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if "energy  without entropy" in line:
                energy = float(line.split()[-1])
        return energy
    except Exception:
        return None


def outcar_ionic_steps(outcar_path: Path) -> int:
    """Return the count of ionic steps (lines matching ``"- Iteration"``)."""
    p = Path(outcar_path)
    if not p.exists():
        return 0
    try:
        count = 0
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if "- Iteration" in line:
                count += 1
        return count
    except Exception:
        return 0


def outcar_summary(outcar_path: Path) -> str:
    """Return a short human-readable summary such as ``"E=-1234.5678 eV, 42 ionic steps"``."""
    p = Path(outcar_path)
    if not p.exists():
        return "no OUTCAR"
    energy = outcar_last_energy(p)
    n_ionic = outcar_ionic_steps(p)
    parts: list[str] = []
    if energy is not None:
        parts.append(f"E={energy:.4f} eV")
    if n_ionic:
        parts.append(f"{n_ionic} ionic steps")
    return ", ".join(parts) if parts else "no energy found"


# ---------------------------------------------------------------------------
# POSCAR readers
# ---------------------------------------------------------------------------

def read_poscar_elements(poscar_path: Path) -> list[str]:
    """Return element symbols from VASP5 POSCAR line 5 (0-indexed).

    Returns an empty list on any error or if the file is not VASP5 format.
    """
    try:
        lines = Path(poscar_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 6:
            return []
        tokens = lines[5].split()
        # VASP5: line 5 starts with element symbols (first character is a letter)
        if tokens and tokens[0][0].isalpha():
            return tokens
        return []
    except Exception:
        return []


def poscar_element_counts(poscar_path: Path) -> dict[str, int]:
    """Return ``{element: count}`` from VASP5 POSCAR lines 5-6 (0-indexed)."""
    try:
        lines = Path(poscar_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 7:
            return {}
        elements = lines[5].split()
        if not elements or not elements[0][0].isalpha():
            return {}
        counts = [int(x) for x in lines[6].split()]
        if len(elements) != len(counts):
            return {}
        return dict(zip(elements, counts))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# POTCAR readers
# ---------------------------------------------------------------------------

def read_potcar_enmax(potcar: Path) -> list[float]:
    """Parse all ENMAX values from a POTCAR file.

    The POTCAR format uses lines like::

        ENMAX  = 400.0; ENMIN = 300.0

    Returns a list of floats, one per species block in the POTCAR.
    """
    enmax_vals: list[float] = []
    try:
        for line in Path(potcar).read_text(encoding="utf-8", errors="replace").splitlines():
            if "ENMAX" in line:
                for part in line.split(";"):
                    if "ENMAX" in part:
                        try:
                            enmax_vals.append(
                                float(part.split("=")[1].strip().split()[0])
                            )
                        except (ValueError, IndexError):
                            pass
    except OSError:
        pass
    return enmax_vals


def compute_encut(potcar: Path, factor: float = 1.3) -> float:
    """Return ``factor * max(ENMAX)`` from the POTCAR, or 520.0 if the POTCAR is missing/empty."""
    vals = read_potcar_enmax(potcar)
    if not vals:
        return 520.0
    return round(factor * max(vals), 1)


# ---------------------------------------------------------------------------
# SLURM script writer
# ---------------------------------------------------------------------------

def write_vasp_slurm(
    path: Path,
    job_name: str,
    wall: str,
    n_nodes: int = 1,
    n_tasks: int = 96,
    cfg: dict | None = None,
) -> None:
    """Write a SLURM submission script that runs ``vasp_std``.

    Parameters
    ----------
    path:
        Destination file (``sub.sh`` or similar).  The parent directory is
        used as the VASP working directory in the script.
    job_name:
        Value for ``#SBATCH --job-name``.
    wall:
        Walltime string, e.g. ``"48:00:00"``.
    n_nodes:
        Number of nodes to allocate (default 1).
    n_tasks:
        Tasks per node (default 96).
    cfg:
        Parsed ``platform.yaml`` as a plain dict.  If *None* the global
        :class:`~hpca.core.config.Config` singleton is used.
    """
    path = Path(path)
    work_dir = str(path.parent)

    if cfg is None:
        _cfg = Config.get()
        vasp_module = _cfg.hpc("vasp_module") or "vasp/6.4.2_openMP"
        account     = _cfg.account("standard")
    else:
        hpc         = cfg.get("hpc", {})
        vasp_module = hpc.get("vasp_module", "vasp/6.4.2_openMP")
        account     = hpc.get("accounts", {}).get("standard") or _account_fallback()

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --nodes={n_nodes}\n"
        f"#SBATCH --ntasks-per-node={n_tasks}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J.stdout\n"
        "ulimit -s unlimited\n"
        "module purge\n"
        f"module load {vasp_module}\n"
        f"cd {work_dir}\n"
        "srun vasp_std &> out\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
