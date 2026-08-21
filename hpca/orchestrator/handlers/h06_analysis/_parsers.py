"""
_parsers.py — Trajectory parsers for LAMMPS dump and VASP XDATCAR files.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("hpca.orch")


def find_mobile_type_id(system_data: Path, mobile_ion: str) -> int | None:
    """Return LAMMPS atom type ID for mobile_ion by reading Masses section comments."""
    opls_map = {
        "Li": "Li", "Na": "Na", "Na+": "Na",
    }
    if not system_data.exists():
        return None
    in_masses = False
    try:
        for line in system_data.read_text().splitlines():
            s = line.strip()
            if s == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not s:
                    continue
                if not s[0].isdigit():
                    break
                parts = s.split()
                if len(parts) >= 2:
                    type_id = int(parts[0])
                    label = s.split("#", 1)[1].strip() if "#" in s else ""
                    # Direct match or via opls_map
                    if label == mobile_ion or opls_map.get(label) == mobile_ion:
                        return type_id
    except Exception:
        pass
    return None


def parse_dump_lammps(
    dump_file: Path,
    target_element: str | None = None,
    target_type_id: int | None = None,
):
    """Parse dump_unwrapped.lmp. Returns (n_frames, n_target, 3) array."""
    try:
        import numpy as np
    except ImportError:
        return None

    frames = []
    current_atoms: list = []
    n_atoms_total = 0
    element_col: int | None = None
    type_col: int | None = None
    x_col = y_col = z_col = 0
    in_atoms = False
    type_id_str = str(target_type_id) if target_type_id is not None else None

    with open(dump_file, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                next_line = next(fh, "0")
                n_atoms_total = int(next_line.strip())
            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]  # skip "ITEM: ATOMS"
                try:
                    x_col = headers.index("xu")
                    y_col = headers.index("yu")
                    z_col = headers.index("zu")
                except ValueError:
                    try:
                        x_col = headers.index("x")
                        y_col = headers.index("y")
                        z_col = headers.index("z")
                    except ValueError:
                        x_col, y_col, z_col = 2, 3, 4
                element_col = headers.index("element") if "element" in headers else None
                type_col = headers.index("type") if "type" in headers else None
                in_atoms = True
                current_atoms = []
            elif in_atoms and line and not line.startswith("ITEM:"):
                parts = line.split()
                if len(parts) > max(x_col, y_col, z_col):
                    include = False
                    if target_type_id is not None and type_col is not None:
                        include = parts[type_col] == type_id_str
                    elif target_element is not None:
                        el = parts[element_col] if element_col is not None else None
                        include = (el == target_element)
                    else:
                        include = True
                    if include:
                        current_atoms.append([
                            float(parts[x_col]),
                            float(parts[y_col]),
                            float(parts[z_col]),
                        ])
            else:
                if in_atoms and current_atoms:
                    frames.append(current_atoms)
                    current_atoms = []
                in_atoms = False

    if current_atoms:
        frames.append(current_atoms)

    if not frames:
        return None

    # Pad/trim to consistent size
    n_target = len(frames[0]) if frames else 0
    consistent = [f for f in frames if len(f) == n_target]
    if not consistent:
        return None

    import numpy as np
    return np.array(consistent, dtype=np.float64)


def parse_dump_all(dump_file: Path):
    """Parse all atoms from dump. Returns (n_frames, n_atoms, 3) and element list."""
    try:
        import numpy as np
    except ImportError:
        return None, None

    frames: list = []
    elements_frame: list = []
    current_atoms: list = []
    current_elements: list = []
    n_atoms_total = 0
    element_col: int | None = None
    x_col = y_col = z_col = 0
    in_atoms = False

    with open(dump_file, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms_total = int(next(fh, "0").strip())
            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]
                try:
                    x_col = headers.index("xu")
                    y_col = headers.index("yu")
                    z_col = headers.index("zu")
                except ValueError:
                    x_col, y_col, z_col = 2, 3, 4
                element_col = headers.index("element") if "element" in headers else None
                in_atoms = True
                current_atoms = []
                current_elements = []
            elif in_atoms and line and not line.startswith("ITEM:"):
                parts = line.split()
                if len(parts) > max(x_col, y_col, z_col):
                    current_atoms.append([float(parts[x_col]),
                                           float(parts[y_col]),
                                           float(parts[z_col])])
                    if element_col is not None:
                        current_elements.append(parts[element_col])
            else:
                if in_atoms and current_atoms:
                    frames.append(current_atoms)
                    if not elements_frame and current_elements:
                        elements_frame = current_elements
                    current_atoms = []
                in_atoms = False

    if current_atoms:
        frames.append(current_atoms)

    if not frames:
        return None, None

    import numpy as np
    n_at = len(frames[0])
    consistent = [f for f in frames if len(f) == n_at]
    return np.array(consistent, dtype=np.float64), elements_frame


def parse_xdatcar(xdatcar_path: Path, mobile_ion_symbol: str = "Li"):
    """Parse XDATCAR for mobile ion positions only."""
    try:
        import numpy as np
    except ImportError:
        return None

    lines = xdatcar_path.read_text(errors="replace").splitlines()
    if len(lines) < 8:
        return None

    scale = float(lines[1].strip())
    lattice = np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    # Find mobile ion index range
    mi_start = 0
    mi_count = 0
    for sp, cnt in zip(species, counts):
        if sp == mobile_ion_symbol:
            mi_count = cnt
            break
        mi_start += cnt

    if mi_count == 0:
        return None

    frames = []
    idx = 7
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("Direct configuration=") or line.startswith("Direct "):
            idx += 1
            if idx + n_atoms > len(lines):
                break
            frame_frac = np.array([
                [float(v) for v in lines[idx + i].split()[:3]]
                for i in range(n_atoms)
            ])
            mi_frac = frame_frac[mi_start: mi_start + mi_count]
            mi_cart = mi_frac @ lattice
            frames.append(mi_cart)
            idx += n_atoms
        else:
            idx += 1

    if not frames:
        return None

    return np.array(frames, dtype=np.float64)
