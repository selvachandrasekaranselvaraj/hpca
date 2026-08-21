"""
hpca/core/project_discovery.py — Unified project directory scanner.

Three registration methods for daemon_inbox/active/:
  1. Project directory  — place or move the actual project dir here
  2. Symlink            — ln -s /path/to/project inbox/active/ProjectName
  3. YAML pointer file  — drop a .yaml file with ``project_path: /path/to/project``

The daemon picks up all three on every poll.  Completed projects are moved to
inbox/archived/; failed projects go to inbox/failed/.

Usage:
    from hpca.core.project_discovery import (
        discover_projects,
        discover_from_inbox,
        register_project,
        is_project_dir,
        move_to_archived,
        move_to_failed,
    )
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from hpca.core.config import Config

log = logging.getLogger("hpca.discovery")

# A directory is a valid project if it contains at least one of these markers.
PROJECT_MARKERS: frozenset[str] = frozenset({
    "project.yaml",
    "DESIGN_COMPLETE.md",
    "orchestrator_state.json",
})


def is_project_dir(path: Path) -> bool:
    """Return True if *path* looks like an HPCA project directory."""
    if not path.is_dir():
        return False
    return any((path / m).exists() for m in PROJECT_MARKERS)


def discover_projects(root: Path | str, *, max_depth: int = 3) -> list[Path]:
    """
    Recursively scan *root* for project directories up to *max_depth* levels.
    Skips directories listed in orchestrator.skip_dirs.
    Returns sorted list of absolute paths.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    skip = set(Config.get().orch("skip_dirs", []))
    found: list[Path] = []
    _scan(root, skip, max_depth, 0, found)
    return sorted(found)


def _scan(path: Path, skip: set, max_depth: int, depth: int,
          found: list[Path]) -> None:
    """Recursively walk path up to max_depth, appending project directories to found."""
    if depth > max_depth:
        return
    try:
        entries = list(path.iterdir())
    except PermissionError:
        return
    for entry in entries:
        if not entry.is_dir() or entry.name in skip or entry.name.startswith("."):
            continue
        if is_project_dir(entry):
            found.append(entry.resolve())
        else:
            _scan(entry, skip, max_depth, depth + 1, found)


# ── Daemon inbox ──────────────────────────────────────────────────────────────

def inbox_active_dir() -> Path:
    """Return the configured HPCA-local daemon inbox active directory."""
    cfg   = Config.get()
    inbox = Path(cfg.orch("inbox_dir", str(Path(__file__).parents[2] / "daemon_inbox")))
    return inbox / cfg.orch("active_subdir", "active")


def inbox_archived_dir() -> Path:
    """Return the daemon inbox archived subdirectory path."""
    cfg   = Config.get()
    inbox = Path(cfg.orch("inbox_dir", str(Path(__file__).parents[2] / "daemon_inbox")))
    return inbox / cfg.orch("archived_subdir", "archived")


def inbox_failed_dir() -> Path:
    """Return the daemon inbox failed subdirectory path."""
    cfg   = Config.get()
    inbox = Path(cfg.orch("inbox_dir", str(Path(__file__).parents[2] / "daemon_inbox")))
    return inbox / cfg.orch("failed_subdir", "failed")


def _resolve_pointer_yaml(yaml_path: Path) -> Path | None:
    """If yaml_path is a YAML pointer file with a project_path key, return that path."""
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return None
    pp = (data.get("project_path") or data.get("path") or data.get("dir")
          or data.get("project_root") or data.get("root"))
    if pp:
        p = Path(pp)
        if is_project_dir(p):
            return p.resolve()
    return None


def register_project(project_dir: Path | str, *, via_symlink: bool = True) -> Path:
    """Register a project with the daemon inbox.

    Creates a symlink (default) or YAML pointer in inbox/active/ pointing to
    project_dir.  The daemon picks it up on the next poll.

    Args:
        project_dir: path to the HPCA project directory (must contain project.yaml)
        via_symlink: True = create symlink; False = write a YAML pointer file

    Returns:
        Path to the created inbox entry (symlink or .yaml file)
    Raises:
        FileNotFoundError if project_dir is not a valid project directory
        FileExistsError if already registered (call unregister first)
    """
    project_dir = Path(project_dir).resolve()
    if not is_project_dir(project_dir):
        raise FileNotFoundError(
            f"Not a valid HPCA project directory (missing project.yaml): {project_dir}"
        )
    active = inbox_active_dir()
    active.mkdir(parents=True, exist_ok=True)

    name = project_dir.name
    if via_symlink:
        dest = active / name
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"Already registered: {dest}")
        dest.symlink_to(project_dir)
        log.info("Registered %s → %s (symlink)", name, project_dir)
        return dest
    else:
        dest = active / f"{name}.yaml"
        if dest.exists():
            raise FileExistsError(f"Already registered: {dest}")
        import yaml
        dest.write_text(yaml.dump({"project_path": str(project_dir)}, default_flow_style=False))
        log.info("Registered %s → %s (yaml pointer)", name, project_dir)
        return dest


def unregister_project(name_or_path: str | Path) -> bool:
    """Remove a project's inbox registration (symlink or YAML pointer).
    Does NOT delete the actual project directory.
    Returns True if found and removed, False otherwise.
    """
    active = inbox_active_dir()
    # Try exact name match
    for candidate in [
        active / str(name_or_path),
        active / f"{name_or_path}.yaml",
        Path(str(name_or_path)),
    ]:
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or candidate.suffix == ".yaml":
                candidate.unlink(missing_ok=True)
                log.info("Unregistered %s", candidate.name)
                return True
    return False


def discover_from_inbox() -> list[Path]:
    """
    Return all valid project directories registered in the active inbox.

    Supports three entry types in inbox/active/:
      - Subdirectory: direct project directory (or copy)
      - Symlink: symlink pointing to the actual project directory
      - .yaml file: pointer file with ``project_path: /path/to/project``

    Creates the active dir if it doesn't exist yet.
    """
    active = inbox_active_dir()
    active.mkdir(parents=True, exist_ok=True)
    projects: list[Path] = []
    seen: set[Path] = set()
    try:
        for entry in sorted(active.iterdir()):
            # YAML pointer file
            if entry.suffix == ".yaml" and entry.is_file() and not entry.is_symlink():
                p = _resolve_pointer_yaml(entry)
                if p and p not in seen:
                    seen.add(p)
                    projects.append(p)
                continue
            # Directory or symlink to directory
            resolved = entry.resolve() if entry.is_symlink() else entry
            if is_project_dir(entry):
                if resolved not in seen:
                    seen.add(resolved)
                    projects.append(resolved)
    except Exception as exc:
        log.warning("discover_from_inbox: %s", exc)
    return sorted(projects)


def discover_all() -> list[Path]:
    """
    Discover projects from all sources:
      1. Daemon inbox active dir (primary)
      2. Legacy scan roots from platform.yaml (secondary, backwards compat)
    Deduplicates by resolved path.
    """
    seen: set[Path] = set()
    projects: list[Path] = []

    for p in discover_from_inbox():
        if p not in seen:
            seen.add(p)
            projects.append(p)

    cfg = Config.get()
    for root_str in cfg.orch("extra_scan_roots", []):
        for p in discover_projects(Path(root_str)):
            if p not in seen:
                seen.add(p)
                projects.append(p)

    return projects


# ── Project lifecycle moves ───────────────────────────────────────────────────

def move_to_archived(project_dir: Path | str) -> Path | None:
    """Move a completed project to the archived inbox subdir with timestamp suffix."""
    return _move_project(Path(project_dir), inbox_archived_dir())


def move_to_failed(project_dir: Path | str) -> Path | None:
    """Move a failed project to the failed inbox subdir with timestamp suffix."""
    return _move_project(Path(project_dir), inbox_failed_dir())


def _move_project(src: Path, dest_parent: Path) -> Path | None:
    """Move src into dest_parent with a timestamp suffix; return the new path or None on error."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest  = dest_parent / f"{src.name}_{stamp}"
    try:
        shutil.move(str(src), str(dest))
        log.info("Moved %s → %s", src, dest)
        return dest
    except Exception as exc:
        log.error("Could not move %s to %s: %s", src, dest, exc)
        return None


# ── Resolver (CLI helper) ─────────────────────────────────────────────────────

def resolve_project(name_or_path: str, search_roots: list[Path] | None = None
                    ) -> Path | None:
    """
    Find a project by name fragment or absolute path.
    Searches inbox active dir first, then search_roots.
    Returns the first match or None.
    """
    candidate = Path(name_or_path)
    if candidate.is_absolute() and is_project_dir(candidate):
        return candidate

    roots = [inbox_active_dir()] + (search_roots or [])
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if entry.is_dir() and name_or_path in entry.name:
                if is_project_dir(entry):
                    return entry.resolve()
    return None
