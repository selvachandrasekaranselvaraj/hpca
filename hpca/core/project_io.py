"""
hpca/core/project_io.py — Unified project.yaml read / write / validate.

Replaces 4 independent yaml.safe_load() patterns in:
  hpca_orchestrator.py, handlers/base.py, core/project.py, orchestrator/cli.py

Usage:
    from hpca.core.project_io import read_project_yaml, write_project_yaml

    data = read_project_yaml(project_dir)   # raises FileNotFoundError if missing
    errors = validate_project_yaml(data)    # returns [] on success
    write_project_yaml(project_dir, data)   # atomic write via .tmp rename
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_YAML_NAME = "project.yaml"


# ── Read ──────────────────────────────────────────────────────────────────────

def project_yaml_path(project_dir: Path | str) -> Path:
    """Return the expected project.yaml path inside *project_dir*."""
    return Path(project_dir) / _YAML_NAME


def read_project_yaml(project_dir: Path | str) -> dict:
    """
    Load project.yaml from *project_dir*.  Returns a dict.
    Raises FileNotFoundError if the file is absent.
    Raises ValueError if YAML is malformed.
    """
    import yaml
    path = project_yaml_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"project.yaml not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed project.yaml at {path}: {exc}") from exc
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Malformed project.yaml at {path}: root must be a mapping")
    return data


def read_normalized_project_yaml(project_dir: Path | str) -> dict:
    """Read, migrate once, and validate the canonical project configuration."""
    from hpca.core.project_schema import migrate, validate_or_raise
    data = migrate(read_project_yaml(project_dir))
    validate_or_raise(data, context=str(project_yaml_path(project_dir)))
    return data


def read_project_yaml_safe(project_dir: Path | str) -> dict:
    """Like read_project_yaml but returns {} on any error (for orchestrator loops)."""
    try:
        return read_project_yaml(project_dir)
    except Exception:
        return {}


# ── Write ─────────────────────────────────────────────────────────────────────

def write_project_yaml(project_dir: Path | str, data: dict) -> Path:
    """
    Write *data* to project.yaml atomically (write to .tmp, then rename).
    Returns the path written.
    """
    import yaml
    path = project_yaml_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(data, default_flow_style=False, allow_unicode=True,
                     sort_keys=False)
    _atomic_write(path, text)
    return path


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_project_yaml(data: dict) -> list[str]:
    """
    Return a list of error strings.  Empty list = valid.

    Checks:
      - Required top-level fields (name, category)
      - Per-category required fields from project_schema config
      - Category is registered in categories.py
    """
    from hpca.core.project_schema import validate
    return validate(data)


def validate_or_raise(data: dict, context: str = "") -> None:
    """Raise ValueError with all errors if validation fails."""
    errors = validate_project_yaml(data)
    if errors:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}Invalid project.yaml:\n" +
                         "\n".join(f"  - {e}" for e in errors))


# ── Schema migration ──────────────────────────────────────────────────────────

def migrate_project_yaml(data: dict) -> dict:
    """
    Upgrade older project.yaml dicts to the current schema.
    Returns the migrated dict (copy; does not modify in place).
    """
    from hpca.core.project_schema import migrate
    return migrate(data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atomic_write(path: Path, text: str) -> None:
    """Write text to a temp file then rename — prevents partial-write corruption."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".yaml")
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, path)
    except Exception:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
