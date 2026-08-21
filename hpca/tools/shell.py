"""
Shell command executor with safety guardrails for HPC use.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import Tool, ToolResult

_BLOCK_PATTERNS = [
    "rm -rf /",
    "rm -r /",
    "mkfs",
    "> /dev/sd",
    "dd if=",
    "chmod 777 /",
    "chown -R root",
]

_MAX_OUTPUT = 16_000


class ShellTool(Tool):
    """AI tool for executing shell commands on the HPC node with safety guardrails."""

    name = "shell"
    description = (
        "Execute a shell command on the HPC node. Use for file operations, "
        "log inspection, running short scripts, and environment checks. "
        "Do NOT use for long-running compute — submit via SlurmTool instead."
    )

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: current dir).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 300).",
                    "default": 120,
                },
                "env_extra": {
                    "type": "object",
                    "description": "Extra environment variables to set.",
                },
            },
            "required": ["command"],
        }

    def run(
        self,
        command: str,
        cwd: str = None,
        timeout: int = 120,
        env_extra: dict = None,
    ) -> ToolResult:
        """Execute command, return ToolResult with stdout/stderr combined."""
        # Safety check
        for pat in _BLOCK_PATTERNS:
            if pat in command:
                return ToolResult(
                    f"Blocked: '{pat}' matched a dangerous pattern.",
                    success=False,
                    metadata={"returncode": -1, "stderr": ""},
                )

        timeout = min(int(timeout or 120), 300)
        work_dir = cwd or os.getcwd()
        if not Path(work_dir).exists():
            work_dir = os.getcwd()

        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                env=env,
            )
            combined = result.stdout
            if result.stderr:
                if combined:
                    combined += "\n[stderr]\n" + result.stderr
                else:
                    combined = result.stderr

            if len(combined) > _MAX_OUTPUT:
                half = _MAX_OUTPUT // 2
                combined = (
                    combined[:half]
                    + f"\n... [truncated {len(combined) - _MAX_OUTPUT} chars] ...\n"
                    + combined[-half:]
                )

            return ToolResult(
                combined or "(no output)",
                success=(result.returncode == 0),
                metadata={
                    "returncode": result.returncode,
                    "stderr": result.stderr[:2000],
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Timed out after {timeout}s.",
                success=False,
                metadata={"returncode": -1, "stderr": "timeout"},
            )
        except Exception as exc:
            return ToolResult(
                str(exc),
                success=False,
                metadata={"returncode": -1, "stderr": str(exc)},
            )

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        return self.run(**kwargs)
