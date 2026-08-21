"""Crash-safe filesystem primitives shared by durable HPCA state."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, value: str, mode: int = 0o640) -> None:
    """Write and replace *path* atomically, including directory synchronization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize JSON deterministically and replace *path* atomically."""
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
