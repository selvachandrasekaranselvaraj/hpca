"""
File system tools: read, write, search, list, stat, mkdir, copy.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import Tool, ToolResult

_MAX_READ_BYTES = 65_536   # 64 KB default read cap


class FilesTool(Tool):
    """AI tool for reading, writing, searching, and managing files on the HPC filesystem."""

    name = "files"
    description = (
        "Read, write, append, search, list, stat, mkdir, and copy files "
        "on the HPC filesystem."
    )

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "read", "write", "append", "list_dir",
                        "search", "exists", "stat", "mkdir", "copy",
                    ],
                },
                "path": {"type": "string"},
                "content": {"type": "string"},
                "pattern": {"type": "string"},
                "max_bytes": {"type": "integer"},
                "max_results": {"type": "integer"},
                "recursive": {"type": "boolean"},
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["action"],
        }

    # ── Public methods (direct call interface) ────────────────────────────────

    def read(self, path: str, max_bytes: int = _MAX_READ_BYTES) -> str:
        """Read up to max_bytes from a file; return content as string."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if p.is_dir():
            raise IsADirectoryError(f"{path} is a directory; use list_dir().")
        size = p.stat().st_size
        with open(p, "r", errors="replace") as f:
            content = f.read(max_bytes)
        if size > max_bytes:
            content += f"\n... [truncated: {size - max_bytes} more bytes]"
        return content

    def write(self, path: str, content: str, mode: str = "w") -> ToolResult:
        """Write (or append) content to a file, creating parents as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode) as f:
            f.write(content)
        verb = "Written" if mode == "w" else "Appended"
        return ToolResult(f"{verb} {len(content)} chars to {path}")

    def append(self, path: str, content: str) -> ToolResult:
        """Append content to an existing file, creating it if absent."""
        return self.write(path, content, mode="a")

    def list_dir(
        self,
        path: str,
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[dict]:
        """
        List directory entries.  Returns list of dicts:
        {name, size, mtime, type}  where type is 'file' or 'dir'.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not p.is_dir():
            raise NotADirectoryError(f"{path} is not a directory.")

        import fnmatch, os as _os

        results = []
        if recursive:
            for root, dirs, files in _os.walk(str(p)):
                for name in sorted(dirs) + sorted(files):
                    fp = Path(root) / name
                    if fnmatch.fnmatch(name, pattern):
                        try:
                            s = fp.stat()
                            results.append({
                                "name": str(fp.relative_to(p)),
                                "size": s.st_size,
                                "mtime": s.st_mtime,
                                "type": "dir" if fp.is_dir() else "file",
                            })
                        except OSError:
                            pass
                        if len(results) >= 500:
                            break
                if len(results) >= 500:
                    break
        else:
            for entry in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name)):
                import fnmatch as _fn
                if not _fn.fnmatch(entry.name, pattern):
                    continue
                try:
                    s = entry.stat()
                    results.append({
                        "name": entry.name,
                        "size": s.st_size,
                        "mtime": s.st_mtime,
                        "type": "dir" if entry.is_dir() else "file",
                    })
                except OSError:
                    pass
                if len(results) >= 500:
                    break
        return results

    def search(
        self, path: str, pattern: str, max_results: int = 100
    ) -> list[dict]:
        """
        Search for pattern in files under path.
        Returns list of {file, line_no, content}.
        Uses grep if available, falls back to Python re.
        """
        p = Path(path)
        results = []

        # Try grep first
        grep_cmd = [
            "grep", "-rn", "--include=*.py", "--include=*.sh",
            "--include=*.json", "--include=*.yaml", "--include=*.txt",
            "--include=INCAR", "--include=*.lammps", "--include=*.out",
            "-m", str(max_results),
            pattern, str(p),
        ]
        try:
            proc = subprocess.run(
                grep_cmd, capture_output=True, text=True, timeout=20
            )
            if proc.returncode in (0, 1):  # 0 = matches, 1 = no match
                for line in proc.stdout.splitlines()[:max_results]:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        results.append({
                            "file": parts[0],
                            "line_no": int(parts[1]) if parts[1].isdigit() else 0,
                            "content": parts[2],
                        })
                return results
        except Exception:
            pass

        # Fallback: pure Python
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))

        for fp in p.rglob("*") if p.is_dir() else [p]:
            if fp.is_dir():
                continue
            try:
                for lineno, line in enumerate(
                    fp.read_text(errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        results.append({
                            "file": str(fp),
                            "line_no": lineno,
                            "content": line,
                        })
                        if len(results) >= max_results:
                            return results
            except (OSError, UnicodeDecodeError):
                continue
        return results

    def exists(self, path: str) -> bool:
        """Return True if path exists on the filesystem."""
        return Path(path).exists()

    def stat(self, path: str) -> dict:
        """Return {size, mtime, ctime, is_file, is_dir} for path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {path}")
        s = p.stat()
        return {
            "size":    s.st_size,
            "mtime":   s.st_mtime,
            "ctime":   s.st_ctime,
            "is_file": p.is_file(),
            "is_dir":  p.is_dir(),
        }

    def mkdir(self, path: str) -> ToolResult:
        """Create directory and any missing parents; succeed silently if it already exists."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return ToolResult(f"Directory created: {path}")

    def copy(self, src: str, dst: str) -> ToolResult:
        """Copy a file or directory tree from src to dst."""
        s, d = Path(src), Path(dst)
        if not s.exists():
            return ToolResult(f"Source not found: {src}", success=False)
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(str(s), str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(str(s), str(d))
        return ToolResult(f"Copied {src} -> {dst}")

    # ── execute() for LLM dispatch ────────────────────────────────────────────

    def execute(self, action: str = "stat", **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        try:
            if action == "read":
                content = self.read(kwargs["path"], kwargs.get("max_bytes", _MAX_READ_BYTES))
                return ToolResult(content, metadata={"length": len(content)})
            elif action == "write":
                return self.write(kwargs["path"], kwargs.get("content", ""))
            elif action == "append":
                return self.append(kwargs["path"], kwargs.get("content", ""))
            elif action == "list_dir":
                entries = self.list_dir(
                    kwargs["path"],
                    kwargs.get("pattern", "*"),
                    kwargs.get("recursive", False),
                )
                lines = [
                    f"[{e['type'][0].upper()}] {e['name']:50s}  {e['size']:>12,} B"
                    for e in entries
                ]
                return ToolResult("\n".join(lines) or "(empty)", metadata={"entries": entries})
            elif action == "search":
                hits = self.search(
                    kwargs["path"],
                    kwargs.get("pattern", ""),
                    kwargs.get("max_results", 100),
                )
                lines = [f"{h['file']}:{h['line_no']}: {h['content']}" for h in hits]
                return ToolResult("\n".join(lines) or "No matches.", metadata={"hits": hits})
            elif action == "exists":
                ok = self.exists(kwargs["path"])
                return ToolResult(str(ok), metadata={"exists": ok})
            elif action == "stat":
                d = self.stat(kwargs["path"])
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)
            elif action == "mkdir":
                return self.mkdir(kwargs["path"])
            elif action == "copy":
                return self.copy(kwargs.get("src", ""), kwargs.get("dst", ""))
            else:
                return ToolResult(f"Unknown action: {action}", success=False)
        except Exception as exc:
            return ToolResult(str(exc), success=False)
