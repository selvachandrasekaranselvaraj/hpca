"""
hpca/io/vasp.py — Consolidated VASP file I/O.

Replaces scattered parsing in:
  tools/vasp.py, h02_aimd.py (_get_poscar_elements, _count_atoms_poscar),
  h07_electronic.py (ACF.dat), h08_echem.py (_read_toten), h13 (XDATCAR)

All functions return plain Python types (no pymatgen/ASE required for basic ops).
Heavier operations (structure manipulation) use pymatgen/ASE with graceful fallback.

Usage:
    from hpca.io.vasp import (
        natoms_from_poscar, elements_from_poscar,
        read_outcar_energy, read_outcar_forces,
        read_xdatcar, read_acf,
        incar_get, incar_set, incar_write,
        encut_from_potcar,
    )
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ── POSCAR / CONTCAR ─────────────────────────────────────────────────────────

def natoms_from_poscar(poscar_path: Path | str) -> int:
    """Return total atom count from POSCAR/CONTCAR (line 7, summed)."""
    path = Path(poscar_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 7:
            return 0
        return sum(int(x) for x in lines[6].split())
    except Exception:
        return 0


def elements_from_poscar(poscar_path: Path | str) -> list[str]:
    """
    Return element symbols from POSCAR/CONTCAR.
    VASP5 format: line 6 (0-indexed) contains element symbols.
    """
    path = Path(poscar_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 6:
            return []
        # VASP5: line index 5 = element symbols, line index 6 = counts
        elems   = lines[5].split()
        counts  = lines[6].split() if len(lines) > 6 else []
        # Verify line 5 looks like element symbols (not counts)
        if elems and not elems[0].isdigit():
            return elems
        # VASP4 fallback: no element line — return empty
        return []
    except Exception:
        return []


def elements_and_counts_from_poscar(poscar_path: Path | str
                                    ) -> tuple[list[str], list[int]]:
    """Return (elements, counts) from POSCAR/CONTCAR."""
    path = Path(poscar_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 7:
            return [], []
        elems  = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        if elems[0].isdigit():
            return [], counts  # VASP4
        return elems, counts
    except Exception:
        return [], []


def poscar_lattice_params(poscar_path: Path | str) -> list[float]:
    """Return [a, b, c] lattice vector lengths from POSCAR (line 3-5)."""
    import math
    path = Path(poscar_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        scale  = float(lines[1].split()[0])
        vecs   = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
        return [math.sqrt(sum(c**2 for c in v)) for v in vecs]
    except Exception:
        return [1.0, 1.0, 1.0]


# ── OUTCAR ────────────────────────────────────────────────────────────────────

def read_outcar_energy(outcar_path: Path | str) -> float | None:
    """Return the final 'free energy TOTEN' from OUTCAR (eV)."""
    path = Path(outcar_path)
    if not path.exists():
        return None
    try:
        # Read last 8 KB — TOTEN appears at every ionic step
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="ignore")
        for line in reversed(tail.splitlines()):
            if "free  energy   TOTEN" in line:
                return float(line.split()[-2])
    except Exception:
        pass
    return None


def read_outcar_forces(outcar_path: Path | str) -> list[list[float]] | None:
    """
    Return forces from the last ionic step in OUTCAR as list of [fx,fy,fz].
    """
    path = Path(outcar_path)
    if not path.exists():
        return None
    try:
        text  = path.read_text(encoding="utf-8", errors="ignore")
        blocks = text.split("TOTAL-FORCE (eV/Angst)")
        if len(blocks) < 2:
            return None
        block  = blocks[-1]
        forces = []
        for line in block.splitlines()[2:]:
            parts = line.split()
            if len(parts) == 6:
                try:
                    forces.append([float(x) for x in parts[3:6]])
                except ValueError:
                    break
        return forces or None
    except Exception:
        return None


def outcar_converged(outcar_path: Path | str) -> bool:
    """Return True if OUTCAR shows ionic convergence (reached NSW or EDIFFG met)."""
    path = Path(outcar_path)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
        return ("reached required accuracy" in tail or
                "General timing and accounting" in tail)
    except Exception:
        return False


# ── XDATCAR ──────────────────────────────────────────────────────────────────

def read_xdatcar_frames(xdatcar_path: Path | str,
                        stride: int = 1,
                        max_frames: int | None = None) -> list[list[list[float]]]:
    """
    Parse XDATCAR and return fractional coordinates for selected frames.

    Returns list of frames; each frame is a list of [x,y,z] fractional coords.
    stride=5 → every 5th frame; max_frames limits total frames returned.
    """
    path = Path(xdatcar_path)
    if not path.exists():
        return []

    frames: list[list[list[float]]] = []
    current: list[list[float]] = []
    frame_idx = 0

    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Direct configuration"):
                if current:
                    if frame_idx % stride == 0:
                        frames.append(current)
                    frame_idx += 1
                    if max_frames and len(frames) >= max_frames:
                        break
                current = []
            else:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        current.append([float(x) for x in parts])
                    except ValueError:
                        pass
        if current and (frame_idx % stride == 0):
            frames.append(current)
    except Exception:
        pass

    return frames


def count_xdatcar_frames(xdatcar_path: Path | str) -> int:
    """Count the number of frames in XDATCAR without loading all coordinates."""
    path = Path(xdatcar_path)
    if not path.exists():
        return 0
    count = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Direct configuration"):
                count += 1
    except Exception:
        pass
    return count


# ── Bader ACF.dat ─────────────────────────────────────────────────────────────

def read_acf(acf_path: Path | str) -> list[dict]:
    """
    Parse Bader ACF.dat.
    Returns list of dicts: {atom_id, x, y, z, charge, min_dist, atomic_vol}
    """
    path = Path(acf_path)
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        in_data = False
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("-"):
                in_data = not in_data if "#" not in stripped else False
                continue
            if in_data:
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        rows.append({
                            "atom_id":   int(parts[0]),
                            "x": float(parts[1]), "y": float(parts[2]),
                            "z": float(parts[3]),
                            "charge":    float(parts[4]),
                            "min_dist":  float(parts[5]),
                            "atomic_vol": float(parts[6]),
                        })
                    except ValueError:
                        pass
    except Exception:
        pass
    return rows


# ── DOSCAR ────────────────────────────────────────────────────────────────────

def read_doscar_total(doscar_path: Path | str) -> dict:
    """
    Parse total DOS from DOSCAR.
    Returns {'energy': [...], 'dos': [...], 'efermi': float}
    """
    path = Path(doscar_path)
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 6:
            return {}
        header = lines[5].split()
        efermi = float(header[3])
        nedos  = int(header[2])
        energy, dos = [], []
        for line in lines[6:6 + nedos]:
            parts = line.split()
            if len(parts) >= 2:
                energy.append(float(parts[0]))
                dos.append(float(parts[1]))
        return {"energy": energy, "dos": dos, "efermi": efermi}
    except Exception:
        return {}


def read_bandgap(vasprun_path: Path | str) -> float | None:
    """Extract band gap from vasprun.xml (fast regex, no XML parsing)."""
    path = Path(vasprun_path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 32768))
            tail = fh.read().decode("utf-8", errors="ignore")
        m = re.search(r'<i name="e_cbm">([\d.\-]+)</i>', tail)
        m2 = re.search(r'<i name="e_vbm">([\d.\-]+)</i>', tail)
        if m and m2:
            return float(m.group(1)) - float(m2.group(1))
    except Exception:
        pass
    return None


# ── INCAR ─────────────────────────────────────────────────────────────────────

def incar_read(path: Path | str) -> dict[str, str]:
    """Read INCAR and return dict of {KEY: value_string}."""
    path = Path(path)
    d: dict[str, str] = {}
    if not path.exists():
        return d
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip().upper()
            v = v.strip()
            if k:
                d[k] = v
    return d


def incar_get(path: Path | str, key: str) -> str | None:
    """Return the value of an INCAR key, or None if absent."""
    return incar_read(path).get(key.upper())


def incar_set(path: Path | str, key: str, value: Any) -> None:
    """Add or update a single key in an INCAR file."""
    d = incar_read(path)
    d[key.upper()] = str(value)
    incar_write(path, d)


def incar_remove(path: Path | str, key: str) -> None:
    """Remove a key from an INCAR file if present."""
    d = incar_read(path)
    d.pop(key.upper(), None)
    incar_write(path, d)


def incar_write(path: Path | str, d: dict) -> None:
    """Write an INCAR dict to file (KEY = value, one per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f" {k} = {v}" for k, v in d.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Aliases for test-suite and external callers
read_incar  = incar_read
write_incar = incar_write


# ── POTCAR ────────────────────────────────────────────────────────────────────

def encut_from_potcar(potcar_path: Path | str, factor: float | None = None) -> float | None:
    """Return ENCUT = factor × max(ENMAX in POTCAR)."""
    from hpca.core.config import Config
    if factor is None:
        factor = Config.get().perf("encut_factor", 1.3)
    path = Path(potcar_path)
    if not path.exists():
        return None
    vals: list[float] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "ENMAX" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    v_str = parts[1].split(";")[0].split()[0]
                    vals.append(float(v_str))
    except Exception:
        return None
    return round(max(vals) * factor, 2) if vals else None


