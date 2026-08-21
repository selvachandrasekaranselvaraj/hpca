"""Kernel-backed singleton and per-project leases."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class Lease:
    """An exclusive non-blocking file lock held for this object's lifetime."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o640)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd
        return True

    @property
    def acquired(self) -> bool:
        """Return whether this process currently holds the lease."""
        return self._fd is not None

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "Lease":
        if not self.acquire():
            raise RuntimeError(f"Lease is already held: {self.path}")
        return self

    def __exit__(self, *_args) -> None:
        self.release()
