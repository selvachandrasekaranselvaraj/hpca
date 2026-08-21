"""Canonical artifact identity, checksums, and append-only provenance ledger."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    kind: str
    producer: str
    format: str
    size_bytes: int
    sha256: str
    created_at: str
    metadata: dict[str, Any]


def record_artifact(project_root: Path, artifact: Path, *, producer: str,
                    kind: str, metadata: dict[str, Any] | None = None) -> ArtifactRecord:
    """Checksum an in-project artifact and append one durable provenance event."""
    root = Path(project_root).resolve(strict=True)
    path = Path(artifact).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"artifact must be a file inside project root: {path}")
    record = ArtifactRecord(
        path=str(path.relative_to(root)), kind=kind, producer=producer,
        format=path.suffix.lstrip(".").lower() or "text",
        size_bytes=path.stat().st_size, sha256=sha256_file(path),
        created_at=datetime.now(timezone.utc).isoformat(), metadata=dict(metadata or {}),
    )
    ledger = root / ".hpca" / "artifacts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ledger, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, (json.dumps(asdict(record), sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return record


def verify_artifact(project_root: Path, record: ArtifactRecord) -> bool:
    path = Path(project_root).resolve() / record.path
    return path.is_file() and path.stat().st_size == record.size_bytes and sha256_file(path) == record.sha256
