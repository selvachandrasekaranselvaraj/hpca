"""Project-local desired-state controls."""
from __future__ import annotations

import json
from pathlib import Path

from hpca.core.atomic import atomic_write_json
from hpca.daemon.schemas import utc_now

CONTROL_STATES = frozenset({"RUNNING", "STOPPED"})


def control_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / ".hpca" / "control.json"


def set_desired_state(project_root: Path, desired_state: str, actor: str) -> Path:
    """Atomically set a project's desired daemon state in its own directory."""
    state = desired_state.upper()
    if state not in CONTROL_STATES:
        raise ValueError(f"desired_state must be one of {sorted(CONTROL_STATES)}")
    path = control_path(project_root)
    atomic_write_json(path, {"schema_version": 1, "desired_state": state,
                             "updated_at": utc_now(), "updated_by": actor})
    return path


def get_desired_state(project_root: Path) -> str:
    """Return RUNNING for a registered project without an explicit control file."""
    path = control_path(project_root)
    if not path.exists():
        return "RUNNING"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError(f"unsupported schema_version {value.get('schema_version')!r}")
        state = str(value["desired_state"]).upper()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid project control file {path}: {exc}") from exc
    if state not in CONTROL_STATES:
        raise ValueError(f"Invalid desired_state in {path}: {state}")
    return state
