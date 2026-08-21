"""Atomic inbox lifecycle and append-only events."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import yaml

from hpca.core.atomic import atomic_write_json, atomic_write_text
from hpca.daemon.schemas import ProjectRequest, utc_now

STATES = ("incoming", "queued", "active", "paused", "completed", "failed", "rejected")


class Inbox:
    """Manage requests without embedding orchestration or scientific behavior."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def initialize(self) -> None:
        for name in (*STATES, "commands", "events", "logs", "locks", "archive"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def submit(self, request: ProjectRequest) -> Path:
        self.initialize()
        for state in STATES:
            for existing_path, existing in self.requests(state):
                if existing.project_id != request.project_id:
                    continue
                if (state not in ("completed", "failed", "rejected")
                        and existing.project_yaml_sha256 == request.project_yaml_sha256
                        and existing.requested_action == request.requested_action):
                    return existing_path
                if state not in ("completed", "failed", "rejected"):
                    raise ValueError(
                        f"Project {request.project_id!r} is already registered with different "
                        "content; use an explicit update operation"
                    )
        destination = self.root / "incoming" / f"{request.request_id}.yaml"
        if destination.exists():
            raise FileExistsError(f"Request already exists: {request.request_id}")
        atomic_write_text(destination, yaml.safe_dump(request.to_mapping(), sort_keys=False))
        self.event("request.received", request.project_id, request.request_id)
        return destination

    def requests(self, state: str) -> Iterable[tuple[Path, ProjectRequest]]:
        if state not in STATES:
            raise ValueError(f"Unknown inbox state: {state}")
        for path in sorted((self.root / state).glob("*.yaml")):
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                yield path, ProjectRequest.from_mapping(value)
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                continue

    def malformed(self, state: str) -> Iterable[tuple[Path, str]]:
        """Yield request files that cannot be decoded, with an actionable reason."""
        if state not in STATES:
            raise ValueError(f"Unknown inbox state: {state}")
        for path in sorted((self.root / state).glob("*.yaml")):
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                ProjectRequest.from_mapping(value)
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
                yield path, f"Malformed request: {exc}"

    def replace(self, request: ProjectRequest) -> Path:
        """Archive a stopped non-active request and enqueue its validated replacement."""
        self.initialize()
        superseded: list[tuple[str, Path]] = []
        for state in STATES:
            for path, existing in self.requests(state):
                if existing.project_id != request.project_id:
                    continue
                if state == "active":
                    raise ValueError(
                        f"Project {request.project_id!r} is active; stop it and wait for PAUSED"
                    )
                if state not in ("completed", "failed", "rejected"):
                    superseded.append((state, path))
        for state, path in superseded:
            archived = self.root / "archive" / f"{path.stem}.{state}.yaml"
            if archived.exists():
                raise FileExistsError(f"Archived request already exists: {archived}")
            os.replace(path, archived)
            self.event("request.superseded", request.project_id, path.stem,
                       f"replacement={request.request_id}")
        return self.submit(request)

    def transition(self, path: Path, destination: str, detail: str = "") -> Path:
        if destination not in STATES:
            raise ValueError(f"Unknown inbox state: {destination}")
        target = self.root / destination / path.name
        if target.exists():
            raise FileExistsError(f"Transition target already exists: {target}")
        os.replace(path, target)
        self.event(f"request.{destination}", "", path.stem, detail)
        return target

    def runtime_path(self, project_id: str) -> Path:
        return self.root / "active" / f"{project_id}.runtime.json"

    def write_runtime(self, project_id: str, value: dict) -> None:
        atomic_write_json(self.runtime_path(project_id), value)

    def event(self, kind: str, project_id: str, request_id: str, detail: str = "") -> None:
        record = {"at": utc_now(), "kind": kind, "project_id": project_id,
                  "request_id": request_id, "detail": detail}
        day = record["at"][:10]
        path = self.root / "events" / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
