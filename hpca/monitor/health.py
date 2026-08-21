"""Read-only project health snapshots for operators and automation."""
from __future__ import annotations

import json
from pathlib import Path

from hpca.daemon.control import get_desired_state


def project_health(project_root: Path) -> dict:
    root = Path(project_root).resolve(strict=True)
    state_path = root / "logs" / "orchestrator_state.json"
    errors: list[str] = []
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid orchestrator state: {exc}")
    handlers = state.get("handlers", {}) if isinstance(state, dict) else {}
    counts = {name: 0 for name in ("PENDING", "RUNNING", "COMPLETE", "FAILED", "SKIPPED")}
    for value in handlers.values() if isinstance(handlers, dict) else ():
        stage = value.get("stage", "PENDING") if isinstance(value, dict) else "PENDING"
        counts[stage] = counts.get(stage, 0) + 1
    try:
        desired = get_desired_state(root)
    except ValueError as exc:
        desired = "INVALID"
        errors.append(str(exc))
    return {"project": root.name, "desired_state": desired,
            "revision": state.get("revision", 0), "stage_counts": counts,
            "healthy": not errors and counts["FAILED"] == 0, "errors": errors}
