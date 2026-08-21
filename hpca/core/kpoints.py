"""kpoints.py — Universal k-points mesh generator.

Reads the kpoints_rules table from platform.yaml (section: kpoints_rules).
Each entry: {max_a: float, mesh: [k1, k2, k3]}

Lattice-parameter-based lookup: for each lattice vector a, b, c, the
appropriate k-point count is chosen **independently per axis** by finding
the first rule whose max_a is >= the length of that axis vector.  This
produces an anisotropic mesh that is properly dense along short axes and
coarser along long axes (e.g. slab or 1-D-like cells).

Usage
-----
    from hpca.core.kpoints import kpoints_for_lattice, kpoints_from_poscar, write_kpoints

    ka, kb, kc = kpoints_for_lattice(a=3.9, b=3.9, c=12.1)
    # → (10, 10, 2)  (from the kpoints_rules table in platform.yaml)

    mesh = kpoints_from_poscar(Path("POSCAR"))
    write_kpoints(Path("KPOINTS"), mesh, gamma=True)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

__all__ = [
    "kpoints_for_lattice",
    "kpoints_from_poscar",
    "write_kpoints",
]

log = logging.getLogger("hpca.core")

# Fallback rules — match platform.yaml kpoints_rules as of project creation.
# These are used only when Config is unavailable; all production runs use
# the table from platform.yaml.
_FALLBACK_RULES: list[dict] = [
    {"max_a":  2,      "mesh": [12, 12, 12]},
    {"max_a":  3,      "mesh": [10, 10, 10]},
    {"max_a":  4,      "mesh": [ 8,  8,  8]},
    {"max_a":  6,      "mesh": [ 6,  6,  6]},
    {"max_a":  8,      "mesh": [ 4,  4,  4]},
    {"max_a": 10,      "mesh": [ 2,  2,  2]},
    {"max_a": 999999,  "mesh": [ 1,  1,  1]},
]


def _get_rules() -> list[dict]:
    """Return kpoints_rules from platform.yaml or the fallback table."""
    try:
        from hpca.core.config import Config
        cfg = Config.get()
        rules = cfg.raw.get("kpoints_rules", [])
        if rules:
            return rules
        log.warning("kpoints: kpoints_rules is empty in platform.yaml; using fallback")
    except Exception as exc:
        log.debug("kpoints: Config unavailable (%s); using fallback rules", exc)
    return _FALLBACK_RULES


def _k_for_axis(length: float, rules: list[dict]) -> int:
    """Return the k-point count for a single axis of *length* Å.

    Iterates through *rules* in order and returns the first mesh[0]
    (scalar) where ``length <= rule['max_a']``.  The mesh list is
    assumed to be isotropic ([k, k, k]); only the first element is used
    here because each axis is handled independently.
    """
    for rule in rules:
        max_a = rule.get("max_a", float("inf"))
        if max_a is None:
            max_a = float("inf")
        if length <= max_a:
            mesh = rule.get("mesh", [1, 1, 1])
            return int(mesh[0])
    # Should not be reached if the table has a catch-all
    return 1


def kpoints_for_lattice(a: float, b: float, c: float) -> tuple[int, int, int]:
    """Return an anisotropic k-point mesh for a cell with lattice lengths a, b, c.

    Each axis is looked up **independently** in the kpoints_rules table
    so that, for example, a slab with c=20 Å gets k_c=1 while a, b each
    get appropriately dense sampling.

    Parameters
    ----------
    a, b, c:
        Lengths of the three lattice vectors in Ångströms.

    Returns
    -------
    tuple[int, int, int]
        (k_a, k_b, k_c) Monkhorst-Pack grid counts.
    """
    rules = _get_rules()
    ka = _k_for_axis(a, rules)
    kb = _k_for_axis(b, rules)
    kc = _k_for_axis(c, rules)
    log.debug(
        "kpoints_for_lattice: (a=%.2f, b=%.2f, c=%.2f) Å → (%d, %d, %d)",
        a, b, c, ka, kb, kc,
    )
    return (ka, kb, kc)


def kpoints_from_poscar(poscar_path: Path) -> tuple[int, int, int]:
    """Read lattice vectors from *poscar_path* and return k-point mesh.

    Parses POSCAR lines 2–4 (1-indexed; the three lattice-vector rows
    after the scale-factor line) to extract the three lattice vectors,
    computes their Euclidean lengths, then calls :func:`kpoints_for_lattice`.

    Parameters
    ----------
    poscar_path:
        Path to a VASP POSCAR or CONTCAR file.

    Returns
    -------
    tuple[int, int, int]
        (k_a, k_b, k_c).  Returns ``(1, 1, 1)`` on any parse error or if
        the file does not exist.
    """
    if not poscar_path.exists():
        log.warning("kpoints_from_poscar: POSCAR not found at %s; returning (1,1,1)", poscar_path)
        return (1, 1, 1)

    try:
        lines = poscar_path.read_text(encoding="utf-8", errors="replace").splitlines()
        # POSCAR format:
        #   line 0 : comment
        #   line 1 : universal scale factor
        #   lines 2-4 : lattice vectors (each has 3 floats)
        if len(lines) < 5:
            log.warning(
                "kpoints_from_poscar: %s has fewer than 5 lines; returning (1,1,1)",
                poscar_path,
            )
            return (1, 1, 1)

        scale = float(lines[1].split()[0])
        if scale < 0:
            # Negative scale means volume in Å³; treat as 1.0 for vector lengths
            log.debug("kpoints_from_poscar: negative scale (volume mode) → using |scale|^(1/3)")
            scale = abs(scale) ** (1.0 / 3.0)

        lengths: list[float] = []
        for row_idx in range(2, 5):
            parts = lines[row_idx].split()
            vec = [float(x) * scale for x in parts[:3]]
            length = math.sqrt(sum(v * v for v in vec))
            lengths.append(length)

        a, b, c = lengths
        log.debug(
            "kpoints_from_poscar: %s → |a|=%.3f |b|=%.3f |c|=%.3f Å",
            poscar_path.name, a, b, c,
        )
        return kpoints_for_lattice(a, b, c)

    except Exception as exc:
        log.warning(
            "kpoints_from_poscar: failed to parse %s (%s); returning (1,1,1)",
            poscar_path, exc,
        )
        return (1, 1, 1)


def write_kpoints(
    path: Path,
    mesh: tuple[int, int, int],
    gamma: bool = True,
) -> None:
    """Write a VASP KPOINTS file for a Gamma-centered Monkhorst-Pack grid.

    Parameters
    ----------
    path:
        Destination path for the KPOINTS file.
    mesh:
        Three-integer tuple ``(k1, k2, k3)`` as returned by
        :func:`kpoints_for_lattice` or :func:`kpoints_from_poscar`.
    gamma:
        If ``True`` (default), write Gamma-centered mesh (``Gamma`` tag).
        If ``False``, write standard Monkhorst-Pack (``Monkhorst-Pack`` tag).

    The file format written is::

        Automatic mesh
         0
        Gamma
         k1  k2  k3
         0   0   0
    """
    scheme = "Gamma" if gamma else "Monkhorst-Pack"
    k1, k2, k3 = int(mesh[0]), int(mesh[1]), int(mesh[2])
    content = (
        "Automatic mesh\n"
        " 0\n"
        f"{scheme}\n"
        f" {k1}  {k2}  {k3}\n"
        " 0   0   0\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.debug("write_kpoints: %s → mesh=(%d,%d,%d) scheme=%s", path, k1, k2, k3, scheme)
