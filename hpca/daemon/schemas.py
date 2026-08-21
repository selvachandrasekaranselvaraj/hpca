"""Versioned daemon request and runtime schemas."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectRequest:
    """Immutable pointer from the inbox to canonical project configuration."""

    schema_version: int
    request_id: str
    project_id: str
    project_yaml: str
    project_yaml_sha256: str
    requested_action: str
    submitted_at: str
    submitted_by: str

    @classmethod
    def create(cls, project_yaml: Path, project_id: str, submitter: str) -> "ProjectRequest":
        project_yaml = Path(project_yaml).resolve(strict=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return cls(1, f"{project_id}-{stamp}", project_id, str(project_yaml),
                   file_sha256(project_yaml), "run", utc_now(), submitter)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ProjectRequest":
        if not isinstance(value, dict):
            raise TypeError("request must be a mapping")
        expected = set(cls.__dataclass_fields__)
        missing = expected - value.keys()
        unknown = value.keys() - expected
        if missing:
            raise KeyError(f"missing request fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})

    def validate(self, allowed_roots: tuple[Path, ...]) -> Path:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported request schema {self.schema_version}")
        if not _ID.fullmatch(self.project_id):
            raise ValueError("project_id must use lowercase letters, digits, dot, dash, or underscore")
        if self.requested_action != "run":
            raise ValueError("Only the run action is accepted as a project request")
        path = Path(self.project_yaml).resolve(strict=True)
        if path.name != "project.yaml":
            raise ValueError("project_yaml must point to a file named project.yaml")
        if not any(path.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"Project path is outside allowed roots: {path}")
        if file_sha256(path) != self.project_yaml_sha256:
            raise ValueError("project.yaml changed after registration; submit an update")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("project.yaml must contain a mapping")
        from hpca.core.project_schema import migrate, validate
        errors = validate(migrate(data))
        if errors:
            raise ValueError("Invalid project.yaml: " + "; ".join(errors))
        return path.parent

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
