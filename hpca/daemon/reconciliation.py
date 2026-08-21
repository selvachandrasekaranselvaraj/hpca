"""Process identity checks used before daemon dispatch."""
from __future__ import annotations

import os
from pathlib import Path


def process_matches(pid: int, project_root: Path) -> bool:
    """Return true only for a live HPCA orchestrator managing *project_root*."""
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        return False
    return "hpca.orchestrator.hpca_orchestrator" in cmdline and str(project_root) in cmdline
