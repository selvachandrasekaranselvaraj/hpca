"""
h02_aimd_constants.py

Shared simulation constants for h02_aimd.py.

All VASP INCAR parameters live in platform.yaml incar_templates
and are accessed via hpca.registry.incar.build_incar().
"""

from pathlib import Path
from hpca.core.paths import load_platform_config as _lpc


def _p(section: str, key: str, default=None):
    """Read a single key from a platform.yaml section."""
    return _lpc().get(section, {}).get(key, default)


# ---------------------------------------------------------------------------
# General workflow controls
# ---------------------------------------------------------------------------

# Use exact real-space projection (LREAL=F) for small cells only.
# Polymer gel DFT boxes (200-400 atoms) need LREAL=Auto for performance.
# Threshold raised from 100 → 50 so only very small cells (SSE/crystal) use LREAL=F.
SMALL_CELL_THRESHOLD   = _p("aimd_dataset", "small_cell_atoms",         50)

# Maximum number of newly submitted liquid jobs in one orchestrator cycle.
MAX_LIQUID_SUBMIT      = _p("aimd_dataset", "max_liquid_submit",       1500)

# A trajectory with at least this many ionic configurations is usable for
# dataset generation without mandatory restart.
_PARTIAL_THRESHOLD_STEPS = _p("aimd_dataset", "partial_threshold_steps", 3500)


# ---------------------------------------------------------------------------
# VASP potential configuration
# ---------------------------------------------------------------------------

# Cross-ref: hpca/config/platform.yaml hpc.potpaw_dir
POTPAW_DIR = Path(_lpc().get("hpc", {}).get("potpaw_dir", "")  or ".")

# Preferred PAW potential directory for each element.
# Follows VASP recommendations: _sv = semi-core s+p, _pv = semi-core p, _d = d-states
_PP_PREF: dict[str, str] = {
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
}


# ---------------------------------------------------------------------------
# SLURM resource presets
#
# Tuple format: (nodes, MPI tasks per node)
# ---------------------------------------------------------------------------

_VASP_NODES_SMALL  = tuple(_p("vasp_nodes", "small",  [1, 64]))
_VASP_NODES_MEDIUM = tuple(_p("vasp_nodes", "medium", [1, 96]))
_VASP_NODES_LARGE  = tuple(_p("vasp_nodes", "large",  [1, 96]))


# ---------------------------------------------------------------------------
# Solid-state AIMD augmentation settings
# ---------------------------------------------------------------------------

_DEFORM_SCALES = _p("aimd_dataset", "deform_scales",        [0.90, 0.95, 1.00, 1.05, 1.10])
_RANDOM_SCALES = _p("aimd_dataset", "random_scales",        [0.95, 1.00, 1.05])
_DATASET_TEMPS = _p("aimd_dataset", "temps",                [300, 400])
_RATTLE_SIGMA  = _p("aimd_dataset", "rattle_sigma_ang",     0.08)
_MIN_ATOM_DISTANCE = _p("aimd_dataset", "min_atom_distance_ang", 1.2)


# ---------------------------------------------------------------------------
# MACE model selection
# ---------------------------------------------------------------------------

# MACE-OFF23 supports organic chemistry elements. Structures containing other
# elements should use MACE-MPA-0 or another suitable periodic model.
_MACE_OFF_ELEMENTS: frozenset[str] = frozenset(
    {
        "H",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    }
)
