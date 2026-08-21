"""
hpca/core/project_schema.py — project.yaml schema validation and migration.

Validation runs at three points:
  1. Wizard (hpca new) — before writing project.yaml to inbox
  2. Orchestrator — before adding a project to the run queue
  3. CLI (hpca status) — shows warnings for invalid projects

Usage:
    from hpca.core.project_schema import validate, migrate, required_for_category

    errors = validate(data)          # [] = valid
    data   = migrate(data)           # upgrade old schemas in place
"""
from __future__ import annotations

from hpca.core.config import Config


# ── Public API ────────────────────────────────────────────────────────────────

def validate(data: dict) -> list[str]:
    """
    Validate a project.yaml dict.
    Returns a list of human-readable error strings.  Empty = valid.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"project root must be a mapping, got {type(data).__name__}"]
    cfg    = Config.get()
    schema = cfg.section("project_schema", {})

    # Required top-level fields
    for field in schema.get("required_fields", ["name", "category"]):
        if not data.get(field):
            errors.append(f"Missing required field: '{field}'")

    category = str(data.get("category", ""))
    if not category:
        return errors  # already flagged above

    # Category must be known
    known = _known_categories()
    if known and category not in known:
        errors.append(
            f"Unknown category '{category}'. Valid: {sorted(known)}"
        )

    # Per-category required fields
    sim = data.get("simulation", {})
    if not isinstance(sim, dict):
        errors.append("simulation must be a mapping")
        sim = {}
    for field in required_for_category(category):
        # Current wizard stores liquid composition below simulation while older
        # projects used a top-level solvents key.  Both are valid schema forms.
        value = data.get(field)
        if field == "solvents" and not value:
            value = sim.get("solvents") or sim.get("comp_spec", {}).get("solvents")
        if not value:
            errors.append(
                f"Category '{category}' requires field '{field}'"
            )

    # Simulation section sanity checks
    if "aimd_temps" in sim:
        if not isinstance(sim["aimd_temps"], list) or not sim["aimd_temps"]:
            errors.append("simulation.aimd_temps must be a non-empty list")
    if "nvt_temps" in sim:
        if not isinstance(sim["nvt_temps"], list) or not sim["nvt_temps"]:
            errors.append("simulation.nvt_temps must be a non-empty list")

    # mlip_backend must be a known value
    backend = data.get("mlip_backend", "")
    if backend and backend not in ("deepmd", "mace", "both", "uma", ""):
        errors.append(
            f"Unknown mlip_backend '{backend}'. "
            "Valid: deepmd | mace | both | uma"
        )

    # T_ref required field
    if "T_ref" not in data:
        errors.append("T_ref is required")

    # Transport float fields (optional, but must be float if present)
    for field in ("D_aimd", "Ea_aimd", "D_mlmd", "Ea_mlmd"):
        val = data.get(field)
        if val is not None and not isinstance(val, (int, float)):
            errors.append(f"{field} must be a number, got {type(val).__name__}")

    # Mechanical positive-float fields
    for field in ("E_GPa", "nu", "Omega_A3", "MW_mobile", "rho_gcm3"):
        val = data.get(field)
        if val is not None:
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"{field} must be a positive number")

    # composition_variants (optional molarity sweep for MLMD/CMD)
    cvars = data.get("composition_variants")
    if cvars is not None:
        if not isinstance(cvars, list) or not all(
            isinstance(v, dict)
            and v.get("name")
            and isinstance(v.get("salt_molarity"), (int, float))
            and v["salt_molarity"] > 0
            for v in cvars
        ):
            errors.append(
                "composition_variants must be a list of mappings with "
                "'name' and positive 'salt_molarity'"
            )

    # Stages block — warn on unknown keys
    _KNOWN_STAGE_KEYS = frozenset({"design", "dft", "aimd", "mlip", "lammps", "cmd",
                                    "classical_md", "active_learning", "neb", "analysis",
                                    "continuum", "plotting", "manuscript", "electronic",
                                    "echem", "chaai"})
    stages = data.get("stages", {})
    if "stages" in data and not isinstance(stages, dict):
        errors.append("stages must be a mapping")
    if isinstance(stages, dict):
        for k in stages:
            if k not in _KNOWN_STAGE_KEYS:
                errors.append(f"stages.{k} is not a recognised stage key")

    execution = data.get("execution", {})
    if execution and not isinstance(execution, dict):
        errors.append("execution must be a mapping")
    elif isinstance(execution, dict):
        default_lane = execution.get("default_lane", "auto")
        if default_lane not in ("auto", "daemon", "slurm"):
            errors.append("execution.default_lane must be auto, daemon, or slurm")
        lane_overrides = execution.get("stages", {})
        if lane_overrides and not isinstance(lane_overrides, dict):
            errors.append("execution.stages must be a mapping")
        elif isinstance(lane_overrides, dict):
            try:
                from hpca.registry.stage import get_stage
                for stage_name, lane in lane_overrides.items():
                    definition = get_stage(stage_name)
                    if lane not in ("auto", definition.lane.value):
                        errors.append(
                            f"execution.stages.{stage_name}={lane!r} is unsupported; "
                            f"this handler runs on {definition.lane.value}"
                        )
            except KeyError as exc:
                errors.append(str(exc))

    try:
        from hpca.core.autonomy import AutonomyPolicy
        AutonomyPolicy.from_project(data)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    preopt = data.get("preoptimization", {})
    if preopt and not isinstance(preopt, dict):
        errors.append("preoptimization must be a mapping")
    elif isinstance(preopt, dict):
        mode = preopt.get("mode", "auto")
        if mode not in ("auto", "mace", "none"):
            errors.append("preoptimization.mode must be auto, mace, or none")
        for key in ("max_runtime_s", "severe_overlap_A", "repair_min_distance_A",
                    "crystal_min_distance_A", "seconds_per_atom_step"):
            if key in preopt and (not isinstance(preopt[key], (int, float)) or preopt[key] <= 0):
                errors.append(f"preoptimization.{key} must be a positive number")
        for key in ("steps", "ntasks"):
            if key in preopt and (not isinstance(preopt[key], int) or preopt[key] < 1):
                errors.append(f"preoptimization.{key} must be a positive integer")

    # Simulation temps must be positive numbers
    for temp_key in ("aimd_temps", "nvt_temps"):
        temps = sim.get(temp_key, [])
        if temps and not all(isinstance(t, (int, float)) and t > 0 for t in temps):
            errors.append(f"simulation.{temp_key} must be a list of positive numbers")

    return errors


def validate_or_raise(data: dict, context: str = "") -> None:
    """Raise ValueError with all errors if validation fails."""
    errors = validate(data)
    if errors:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Invalid project.yaml:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


def required_for_category(category: str) -> list[str]:
    """Return list of required project.yaml fields for the given category."""
    cfg = Config.get()
    schema = cfg.section("project_schema", {})
    return schema.get("category_required", {}).get(category, [])


# ── Migration ─────────────────────────────────────────────────────────────────

def migrate(data: dict) -> dict:
    """
    Upgrade an older project.yaml dict to the current schema.
    Returns a new dict (does not mutate the argument).

    Migration rules applied in order:
      v0 → current: field renames, structure normalisations
    """
    if not isinstance(data, dict):
        raise TypeError(f"project root must be a mapping, got {type(data).__name__}")
    d = dict(data)
    if isinstance(data.get("simulation"), dict):
        d["simulation"] = dict(data["simulation"])
    if isinstance(data.get("stages"), dict):
        d["stages"] = dict(data["stages"])

    # Field renames
    _rename(d, "system",           "category")
    _rename(d, "mobile_species",   "mobile_ion")
    _rename(d, "system_type",      "category")
    _rename(d, "project_root",     "root")

    # Flatten legacy top-level temperature lists into simulation block
    for old, new in [("aimd_temperatures", "aimd_temps"),
                     ("nvt_temperatures",  "nvt_temps"),
                     ("cmd_temperatures",  "cmd_temps")]:
        if old in d:
            d.setdefault("simulation", {})[new] = d.pop(old)

    # Rename deprecated stage keys
    stages = d.get("stages", {})
    if isinstance(stages, dict):
        _rename(stages, "classical_md", "cmd")

    # Ensure workflow_version present
    d.setdefault("workflow_version", 2)

    return d


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rename(d: dict, old: str, new: str) -> None:
    """Rename key *old* to *new* in dict *d* if *old* is present and *new* is not."""
    if old in d and new not in d:
        d[new] = d.pop(old)


def _known_categories() -> set[str]:
    """Return set of registered category names from the CategoryRegistry."""
    try:
        from hpca.core.categories import registry
        return set(registry._specs.keys())
    except Exception:
        return set()
