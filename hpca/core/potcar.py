"""potcar.py — POTCAR prefix lookup and concatenation.

Reads the preferred PAW potential subdirectory for each element from
platform.yaml (section: ``hpc.potcar_prefix``, expected as a plain
dict mapping element symbol → subdirectory name, e.g. ``{"Li": "Li",
"Na": "Na_pv", ...}``).

If the key is absent from platform.yaml (or Config is unavailable), the
module falls back to the table that was previously hard-coded in
``h02_aimd_constants.py`` and used by ``h01_dft.py``.

The POTCAR base directory is read from ``hpc.potpaw_dir`` in
platform.yaml (e.g. ``/path/to/apps/apps/vasp_src/potential/potpaw_PBE``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

__all__ = [
    "get_potcar_prefix",
    "build_potcar",
    "read_potcar_enmax",
    "compute_encut",
]

log = logging.getLogger("hpca.core")

# ── Built-in fallback table (matches h02_aimd_constants._PP_PREF) ─────────────
# Follows VASP recommendations: _sv = semi-core s+p, _pv = semi-core p, _d = d-states
_FALLBACK_PP_PREF: dict[str, str] = {
    # Alkali / alkaline-earth
    "Li": "Li",
    "Na": "Na_pv",
    "K":  "K_sv",
    "Mg": "Mg",
    "Ca": "Ca_sv",
    # Common non-metals
    "H":  "H",
    "B":  "B",
    "C":  "C",
    "N":  "N",
    "O":  "O",
    "F":  "F",
    "Si": "Si",
    "P":  "P",
    "S":  "S",
    "Cl": "Cl",
    "Ge": "Ge_d",
    "As": "As",
    "Se": "Se",
    "Br": "Br",
    "Sn": "Sn_d",
    "Sb": "Sb",
    "Te": "Te",
    "I":  "I",
    # Transition metals — cathode / electrode
    "Ti": "Ti_sv",
    "V":  "V_sv",
    "Cr": "Cr_pv",
    "Mn": "Mn_pv",
    "Fe": "Fe_pv",
    "Co": "Co",
    "Ni": "Ni",
    "Cu": "Cu_pv",
    "Zn": "Zn",
    "Nb": "Nb_sv",
    "Mo": "Mo_pv",
    "Ta": "Ta_pv",
    "W":  "W_pv",
    # SSE / halide-SSE framework metals
    "Al": "Al",
    "Ga": "Ga_d",
    "In": "In_d",
    "Sc": "Sc_sv",
    "Y":  "Y_sv",
    "Zr": "Zr_sv",
    "Hf": "Hf_pv",
    "La": "La",
    "Gd": "Gd_3",
    "Nd": "Nd_3",
    "Sm": "Sm_3",
    "Er": "Er_3",
    "Yb": "Yb_2",
    "Lu": "Lu_3",
    # Additional elements from task specification
    "Sr": "Sr_sv",
    "Pb": "Pb_d",
    "Ce": "Ce",
}

_FALLBACK_POTPAW_DIR = ""


def _get_config():
    """Return Config singleton or None if unavailable."""
    try:
        from hpca.core.config import Config
        return Config.get()
    except Exception as exc:
        log.debug("potcar: Config unavailable (%s), using fallback", exc)
        return None


def _yaml_prefix_table() -> dict[str, str] | None:
    """Return hpc.potcar_prefix from platform.yaml, or None if absent."""
    cfg = _get_config()
    if cfg is None:
        return None
    table = cfg.raw.get("hpc", {}).get("potcar_prefix", None)
    if not isinstance(table, dict) or not table:
        return None
    return table


def get_potcar_prefix(element: str) -> str:
    """Return the POTCAR subdirectory name for *element*.

    Lookup order:
    1. ``hpc.potcar_prefix`` dict in platform.yaml
    2. Built-in fallback table (_FALLBACK_PP_PREF)

    Parameters
    ----------
    element:
        Chemical symbol, e.g. ``"Li"``, ``"Fe"``, ``"O"``.

    Raises
    ------
    KeyError
        If the element is not found in either the platform.yaml table
        or the fallback table.
    """
    # Try platform.yaml first
    yaml_table = _yaml_prefix_table()
    if yaml_table is not None:
        if element in yaml_table:
            return yaml_table[element]
        # Fall through to built-in table if not in yaml table

    if element in _FALLBACK_PP_PREF:
        return _FALLBACK_PP_PREF[element]

    raise KeyError(
        f"No POTCAR prefix defined for element '{element}'. "
        "Add it to hpc.potcar_prefix in platform.yaml or the fallback table in potcar.py."
    )


def build_potcar(
    elements: list[str],
    dest: Path,
    potpaw_dir: str = "",
) -> None:
    """Concatenate per-element POTCAR files into *dest*.

    Parameters
    ----------
    elements:
        Ordered list of chemical symbols (must match POSCAR species order).
    dest:
        Destination POTCAR path.  Parent directory must already exist.
    potpaw_dir:
        Base directory containing per-element subdirectories.  When empty
        (the default) the value from ``hpc.potpaw_dir`` in platform.yaml
        is used, falling back to the compile-time constant
        ``_FALLBACK_POTPAW_DIR``.

    Raises
    ------
    FileNotFoundError
        If the source POTCAR for any element is missing.  The error message
        includes the element symbol for easy diagnosis.
    KeyError
        If no prefix is defined for an element (propagated from
        ``get_potcar_prefix``).
    """
    if not potpaw_dir:
        cfg = _get_config()
        if cfg is not None:
            potpaw_dir = cfg.hpc("potpaw_dir", _FALLBACK_POTPAW_DIR)
        else:
            potpaw_dir = _FALLBACK_POTPAW_DIR

    base = Path(potpaw_dir)

    # Validate all sources before writing anything
    sources: list[Path] = []
    for el in elements:
        prefix = get_potcar_prefix(el)
        src = base / prefix / "POTCAR"
        if not src.exists():
            raise FileNotFoundError(
                f"POTCAR for element '{el}' not found at: {src}"
            )
        sources.append(src)
        log.debug("potcar: %s → %s", el, src)

    # Concatenate
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out_fh:
        for src in sources:
            out_fh.write(src.read_bytes())

    log.info("potcar: built POTCAR for %s → %s", elements, dest)


def read_potcar_enmax(potcar: Path) -> list[float]:
    """Read all ENMAX values from a (possibly multi-species) POTCAR file.

    Parameters
    ----------
    potcar:
        Path to an existing POTCAR file.

    Returns
    -------
    list[float]
        One value per element block found.  Empty list if the file cannot
        be read or contains no ENMAX lines.
    """
    values: list[float] = []
    if not potcar.exists():
        log.debug("read_potcar_enmax: file not found (%s)", potcar)
        return values

    pattern = re.compile(r"ENMAX\s*=\s*([\d.]+)", re.IGNORECASE)
    try:
        text = potcar.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("read_potcar_enmax: cannot read %s: %s", potcar, exc)
        return values

    for match in pattern.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass

    log.debug("read_potcar_enmax: %s → ENMAX values %s", potcar.name, values)
    return values


def compute_encut(potcar: Path, factor: float = 1.3) -> float:
    """Return ``factor × max(ENMAX)`` from *potcar*.

    Parameters
    ----------
    potcar:
        Path to the POTCAR file.
    factor:
        Multiplicative scaling factor applied to the maximum ENMAX value.
        Default is 1.3 (30 % above the recommended cutoff).

    Returns
    -------
    float
        Computed ENCUT in eV.  Falls back to 520.0 eV if the POTCAR
        cannot be read or contains no ENMAX values.
    """
    enmax_values = read_potcar_enmax(potcar)
    if not enmax_values:
        log.warning(
            "compute_encut: no ENMAX found in %s; falling back to 520.0 eV", potcar
        )
        return 520.0

    encut = factor * max(enmax_values)
    log.debug(
        "compute_encut: factor=%.2f × max(ENMAX)=%.1f → %.1f eV",
        factor, max(enmax_values), encut,
    )
    return encut
