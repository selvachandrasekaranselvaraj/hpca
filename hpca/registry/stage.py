"""Canonical registry of HPCA scientific stages and their dependency DAG.

Defines which handlers must be COMPLETE before others can start.
All handler names use dot notation for subtasks: h01_dft.opt

Category routing uses hpca.core.categories (single source of truth).
To add a new category: register it in categories.py, then add entries to
CATEGORY_DEFAULTS and optionally CATEGORY_DEPENDENCIES below.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hpca.core.categories import is_molecular, is_sse, is_crystalline, needs_cmd

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState


class ExecutionLane(str, Enum):
    """Execution location selected by orchestration policy."""

    DAEMON = "daemon"
    SLURM = "slurm"
    AUTO = "auto"


@dataclass(frozen=True)
class StageDefinition:
    """Declarative scientific stage metadata; contains no execution code."""

    name: str
    handler: str
    lane: ExecutionLane
    description: str


STAGES: dict[str, StageDefinition] = {
    "h00_design": StageDefinition("h00_design", "h00_design", ExecutionLane.DAEMON, "Design and validate structures"),
    "h01_dft": StageDefinition("h01_dft", "h01_dft", ExecutionLane.SLURM, "DFT reference calculations"),
    "h02_aimd": StageDefinition("h02_aimd", "h02_aimd", ExecutionLane.SLURM, "AIMD reference dataset"),
    "h03_neb": StageDefinition("h03_neb", "h03_neb", ExecutionLane.SLURM, "Migration-path calculations"),
    "h04_mlip": StageDefinition("h04_mlip", "h04_mlip", ExecutionLane.SLURM, "MLIP training and validation"),
    "h05_cmd": StageDefinition("h05_cmd", "h05_cmd", ExecutionLane.SLURM, "Classical molecular dynamics"),
    "h05_lammps": StageDefinition("h05_lammps", "h05_lammps", ExecutionLane.SLURM, "MLIP molecular dynamics"),
    "h13_active_learning": StageDefinition("h13_active_learning", "h13_active_learning", ExecutionLane.DAEMON, "Coordinate SLURM active-learning refinement"),
    "h06_analysis": StageDefinition("h06_analysis", "h06_analysis", ExecutionLane.SLURM, "Scientific analysis and uncertainty"),
    "h07_electronic": StageDefinition("h07_electronic", "h07_electronic", ExecutionLane.DAEMON, "Electronic characterization"),
    "h08_echem": StageDefinition("h08_echem", "h08_echem", ExecutionLane.DAEMON, "Electrochemical characterization"),
    "h09_continuum": StageDefinition("h09_continuum", "h09_continuum", ExecutionLane.DAEMON, "Continuum modeling"),
    "h10_plotting": StageDefinition("h10_plotting", "h10_plotting", ExecutionLane.DAEMON, "Validated visualization"),
    "h11_manuscript": StageDefinition("h11_manuscript", "h11_manuscript", ExecutionLane.DAEMON, "Manuscript and FAIR archive"),
    "h12_chaai": StageDefinition("h12_chaai", "h12_chaai", ExecutionLane.DAEMON, "Redacted workflow event dataset"),
}


def get_stage(name: str) -> StageDefinition:
    """Return the definition for a stage or dotted DFT substage."""
    key = name.split(".", 1)[0]
    try:
        return STAGES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown HPCA stage: {name}") from exc

# Maps handler name → list of prerequisite handler names that must be COMPLETE
DEPENDENCIES: dict[str, list[str]] = {
    "h00_design":           [],
    "h01_dft.aimd_relax":   ["h00_design"],
    "h01_dft.vc_relax":     ["h00_design"],
    "h01_dft.opt":          ["h01_dft.vc_relax"],
    "h01_dft.bader":      ["h01_dft.opt"],
    "h01_dft.dos_scf":    ["h01_dft.opt"],
    "h01_dft.dos_nonscf": ["h01_dft.dos_scf"],
    "h01_dft.static":     ["h01_dft.opt"],
    "h01_dft.echem_static":["h01_dft.opt"],
    "h02_aimd":           ["h01_dft.opt"],
    "h03_neb":            ["h01_dft.opt"],
    "h04_mlip":           ["h02_aimd"],
    "h05_cmd":            ["h00_design"],          # classical OPLS-AA MD: runs after design
    "h05_lammps":         ["h04_mlip"],
    "h13_active_learning": ["h05_lammps"],
    # h06 runs as soon as CMD data is available; mlmd_dft and combined variants
    # self-gate inside _collect_sources / is_complete() when MLMD data arrives later.
    "h06_analysis":       ["h05_cmd"],
    "h07_electronic":     ["h01_dft.bader"],   # bader is sufficient trigger
    "h08_echem":          ["h06_analysis"],
    "h09_continuum":      ["h06_analysis"],
    "h10_plotting":       ["h06_analysis"],     # starts as soon as any analysis CSV exists
    "h11_manuscript":     ["h10_plotting"],
    "h12_chaai":          [],                   # independent — always eligible
}

# Per-category dependency overrides — merged at runtime in next_runnable().
# Add a new key here when a new category needs non-default dep wiring.
# For crystalline SSE: after opt, AIMD + NEB + electronic + echem run in parallel.
_SSE_DEP_OVERRIDES: dict[str, list[str]] = {
    # Doped systems: vc_relax waits for aimd_relax pre-equilibration
    "h01_dft.vc_relax":    ["h01_dft.aimd_relax"],
    # ECW runs after opt (not after h06_analysis as in the liquid default)
    "h08_echem":           ["h01_dft.opt"],
    # MLIP → LAMMPS → active learning → analysis/result freeze.
    "h05_lammps":          ["h04_mlip"],
    "h13_active_learning": ["h05_lammps"],
    "h06_analysis":        ["h13_active_learning"],
    "h10_plotting":        ["h06_analysis"],
}

CATEGORY_DEPENDENCIES: dict[str, dict[str, list[str]]] = {
    # All SSE-class categories share the same dep overrides
    "inorganic_sse": _SSE_DEP_OVERRIDES,
    "solid":         _SSE_DEP_OVERRIDES,   # new category, same pipeline
    "inorganic":     _SSE_DEP_OVERRIDES,   # inorganic without NEB/ECW still needs direct deps
}

# Handlers that run directly inside daemon process (no sbatch)
DAEMON_HANDLERS: frozenset[str] = frozenset(
    name for name, definition in STAGES.items()
    if definition.lane is ExecutionLane.DAEMON
)

# Ordered list for iteration (respects dependency order)
HANDLER_ORDER: list[str] = [
    "h00_design",
    "h01_dft.aimd_relax",
    "h01_dft.vc_relax", "h01_dft.opt",
    "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
    "h01_dft.static", "h01_dft.echem_static",
    "h02_aimd", "h03_neb",
    "h04_mlip",
    "h05_cmd",                                     # classical MD (parallel to AIMD/MLIP)
    "h05_lammps",
    "h13_active_learning",
    "h06_analysis",
    "h07_electronic",
    "h08_echem", "h09_continuum",
    "h10_plotting",
    "h11_manuscript",
    "h12_chaai",
]

# inorganic_sse: after opt, AIMD + NEB + electronic + echem all start in parallel
_HANDLER_ORDER_INORGANIC_SSE: list[str] = [
    "h00_design",
    "h01_dft.aimd_relax",
    "h01_dft.vc_relax", "h01_dft.opt",
    "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
    "h01_dft.static", "h01_dft.echem_static",
    # All four start in parallel once opt (and bader for electronic) completes:
    "h02_aimd", "h03_neb", "h07_electronic", "h08_echem",
    "h04_mlip",             # after AIMD
    "h05_lammps",           # after MLIP (both backends)
    "h13_active_learning",  # refine before frozen scientific analysis
    "h06_analysis",
    "h09_continuum",
    "h10_plotting",
    "h11_manuscript",
    "h12_chaai",
]

# ── Shared handler lists — reused by multiple category entries ──────────────
_MOLECULAR_HANDLERS: list[str] = [
    "h00_design",
    "h05_cmd",                              # classical OPLS-AA MD (PACKMOL packing)
    "h01_dft.vc_relax", "h01_dft.opt",
    "h02_aimd", "h04_mlip", "h05_lammps", "h13_active_learning",
    "h06_analysis",
    "h09_continuum", "h10_plotting", "h11_manuscript", "h12_chaai",
]
_MOLECULAR_HANDLERS_WITH_BADER: list[str] = _MOLECULAR_HANDLERS + ["h01_dft.bader"]

_CRYSTALLINE_HANDLERS: list[str] = [
    "h00_design",
    "h01_dft.vc_relax", "h01_dft.opt",
    "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
    "h02_aimd", "h04_mlip", "h05_lammps", "h13_active_learning",
    "h06_analysis",
    "h07_electronic", "h09_continuum", "h10_plotting", "h11_manuscript", "h12_chaai",
]

_SSE_HANDLERS: list[str] = [
    "h00_design",
    "h01_dft.aimd_relax",                  # doped SSE pre-equilibration
    "h01_dft.vc_relax", "h01_dft.opt",
    "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
    "h01_dft.static", "h01_dft.echem_static",
    "h02_aimd", "h03_neb",                 # parallel after opt
    "h04_mlip", "h05_lammps", "h13_active_learning", "h06_analysis",
    "h07_electronic", "h08_echem",         # ECW + electronic
    "h09_continuum", "h10_plotting", "h11_manuscript", "h12_chaai",
]

# Default enabled handlers per category.
# To add a new category: register in categories.py, add entry here.
CATEGORY_DEFAULTS: dict[str, list[str]] = {
    # ── Molecular ───────────────────────────────────────────────────────────
    "solvent":            list(_MOLECULAR_HANDLERS),
    "salt":               list(_MOLECULAR_HANDLERS),
    "liquid_electrolyte": list(_MOLECULAR_HANDLERS_WITH_BADER),  # legacy default
    "polymer":            list(_MOLECULAR_HANDLERS),
    "copolymer":          list(_MOLECULAR_HANDLERS),
    # ── Crystalline ─────────────────────────────────────────────────────────
    "solid":              list(_SSE_HANDLERS),          # new: full SSE pipeline
    "inorganic_sse":      list(_SSE_HANDLERS),          # legacy SSE
    "inorganic":          list(_CRYSTALLINE_HANDLERS),  # no NEB/ECW
    # ── Fallback ────────────────────────────────────────────────────────────
    "custom": [
        "h00_design",
        "h01_dft.vc_relax", "h01_dft.opt",
        "h02_aimd", "h04_mlip",
        "h05_lammps", "h06_analysis",
        "h10_plotting", "h11_manuscript", "h12_chaai",
    ],
}


def can_run(handler_name: str, state: "ProjectState") -> bool:
    """Return True if all dependencies for handler_name are COMPLETE."""
    deps = DEPENDENCIES.get(handler_name, [])
    return all(state.get_stage(dep) == "COMPLETE" for dep in deps)


def next_runnable(state: "ProjectState", enabled: list[str],
                  project_yaml: dict | None = None) -> list[str]:
    """Return all handlers whose deps are met, stage is PENDING, and are enabled.

    Execution order and dependency overrides are driven by category predicates
    from hpca.core.categories — no raw string checks here.
    """
    enabled_set = set(enabled)
    yaml        = project_yaml or {}
    category    = yaml.get("category", "")

    # Category dep overrides: look up by exact category name first,
    # then fall back to the shared SSE overrides for any SSE-class category.
    if category in CATEGORY_DEPENDENCIES:
        cat_dep_overrides = CATEGORY_DEPENDENCIES[category]
    elif is_sse(category):
        cat_dep_overrides = _SSE_DEP_OVERRIDES
    else:
        cat_dep_overrides = {}

    def get_deps(h: str) -> list[str]:
        """Return the dependency list for handler h, applying category overrides."""
        return cat_dep_overrides.get(h, DEPENDENCIES.get(h, []))

    def dep_satisfied(dep: str) -> bool:
        """Return True if dep is not enabled or is already COMPLETE."""
        if dep not in enabled_set:
            return True
        return state.get_stage(dep) == "COMPLETE"

    # Choose handler execution order by material class
    if is_sse(category):
        order = _HANDLER_ORDER_INORGANIC_SSE
    elif is_molecular(category):
        wv = yaml.get("workflow_version", 1)
        system_type = yaml.get("system_type", "")
        if wv >= 2 or is_molecular(category):
            order = _HANDLER_ORDER_LIQUID_V2
        else:
            order = HANDLER_ORDER
    else:
        order = HANDLER_ORDER

    return [
        h for h in order
        if h in enabled_set
        and state.get_stage(h) == "PENDING"
        and all(dep_satisfied(d) for d in get_deps(h))
    ]


# Liquid-first execution order for workflow_version >= 2
_HANDLER_ORDER_LIQUID_V2: list[str] = [
    "h00_design",
    "h05_cmd",                                    # CMD first for liquids
    "h01_dft.vc_relax", "h01_dft.opt",
    "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
    "h01_dft.static", "h01_dft.echem_static",
    "h02_aimd", "h03_neb",                        # AIMD after CMD
    "h04_mlip",
    "h05_lammps",                                 # MLMD after AIMD
    "h13_active_learning",                        # refinement before final analysis
    "h06_analysis",
    "h07_electronic",
    "h08_echem", "h09_continuum",
    "h10_plotting",
    "h11_manuscript",
    "h12_chaai",
]


def get_enabled(project_yaml: dict) -> list[str]:
    """
    Return list of enabled handler names from project.yaml.
    Uses 'stages:' field if present; falls back to category defaults.
    New categories: register in categories.py + add to CATEGORY_DEFAULTS above.
    """
    category = project_yaml.get("category", "inorganic_sse")
    # Fall back to SSE pipeline for any unknown crystalline category,
    # and to liquid pipeline for any unknown molecular category.
    if category not in CATEGORY_DEFAULTS:
        if is_crystalline(category):
            defaults = CATEGORY_DEFAULTS.get("inorganic_sse", [])
        elif is_molecular(category):
            defaults = CATEGORY_DEFAULTS.get("liquid_electrolyte", [])
        else:
            defaults = CATEGORY_DEFAULTS.get("custom", [])
    else:
        defaults = CATEGORY_DEFAULTS[category]

    stages = project_yaml.get("stages")
    if stages is None:
        return defaults

    enabled: list[str] = []

    def _add(key: str, handler_names: list[str], value) -> None:
        """Append handler names to enabled based on whether value is bool, list, or dict."""
        if value is True:
            enabled.extend(handler_names)
        elif isinstance(value, list):
            # List of subtask names: ["vc_relax", "opt"]
            for h_full in handler_names:
                subtask = h_full.split(".")[-1] if "." in h_full else h_full
                if subtask in value:
                    enabled.append(h_full)
        elif isinstance(value, dict):
            # Dict of {subtask_name: bool}: {"vc_relax": true, "opt": true, "bader": false}
            for h_full in handler_names:
                subtask = h_full.split(".")[-1] if "." in h_full else h_full
                if value.get(subtask, False):
                    enabled.append(h_full)

    stage_map = {
        "design":          ["h00_design"],
        "dft":             ["h01_dft.aimd_relax", "h01_dft.vc_relax", "h01_dft.opt",
                            "h01_dft.bader", "h01_dft.dos_scf", "h01_dft.dos_nonscf",
                            "h01_dft.static", "h01_dft.echem_static"],
        "aimd":            ["h02_aimd"],
        "neb":             ["h03_neb"],
        "mlip":            ["h04_mlip"],
        "active_learning": ["h13_active_learning"],
        "cmd":             ["h05_cmd"],
        "classical_md":    ["h05_cmd"],            # alias used in project.yaml
        "lammps":          ["h05_lammps"],
        "analysis":        ["h06_analysis"],
        "electronic":      ["h07_electronic"],
        "echem":           ["h08_echem"],
        "continuum":       ["h09_continuum"],
        "plotting":        ["h10_plotting"],
        "manuscript":      ["h11_manuscript"],
        "chaai":           ["h12_chaai"],
    }

    for key, handlers in stage_map.items():
        val = stages.get(key)
        if val:
            _add(key, handlers, val)

    # h12_chaai always enabled if not explicitly disabled
    if stages.get("chaai", True) and "h12_chaai" not in enabled:
        enabled.append("h12_chaai")

    return enabled if enabled else defaults
