"""Side-effect-free integrity checks for canonical registries."""
from __future__ import annotations

from hpca.registry.stage import DEPENDENCIES, STAGES
from hpca.registry.submission import SUBMISSIONS


def validate_registries() -> tuple[str, ...]:
    """Return registry integrity errors; an empty tuple means valid."""
    errors: list[str] = []
    for key, definition in STAGES.items():
        if key != definition.name:
            errors.append(f"stage key {key!r} does not match name {definition.name!r}")
        if not definition.handler:
            errors.append(f"stage {key!r} has no handler")
    known = set(STAGES)
    for stage, dependencies in DEPENDENCIES.items():
        root = stage.split(".", 1)[0]
        if root not in known:
            errors.append(f"dependency owner {stage!r} is not registered")
        for dependency in dependencies:
            if dependency.split(".", 1)[0] not in known:
                errors.append(f"dependency {dependency!r} for {stage!r} is not registered")
    for key, definition in SUBMISSIONS.items():
        overlap = definition.required & definition.optional
        if overlap:
            errors.append(f"submission {key!r} repeats parameters: {sorted(overlap)}")
        if definition.family not in {"slurm", "local"}:
            errors.append(f"submission {key!r} has invalid family {definition.family!r}")
    return tuple(errors)


def require_valid_registries() -> None:
    errors = validate_registries()
    if errors:
        raise RuntimeError("Invalid HPCA registries:\n- " + "\n- ".join(errors))
