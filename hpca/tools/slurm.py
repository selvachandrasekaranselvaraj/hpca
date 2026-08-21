"""
Slurm HPC tools for HPCA: submit, monitor, cancel, inspect jobs and nodes.
Adapted for NREL Kestrel (your_account account, standard + gpu-h100 partitions).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from .base import Tool, ToolResult


def _run(cmd: list[str], timeout: int = 30) -> tuple[str, str, int]:
    """Run a subprocess and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout, result.stderr, result.returncode


def _parse_scontrol_fields(text: str) -> dict:
    """Parse scontrol show job output into a flat dict."""
    d = {}
    for token in text.split():
        if "=" in token:
            k, _, v = token.partition("=")
            d[k] = v
    return d


def _parse_squeue_lines(text: str) -> list[dict]:
    """Parse squeue -o '%.18i %.9P %.30j %.8u %.8T %.12M %Z' output."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split()
    jobs = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        # Build dict from positional fields; pad if short
        entry = {
            "id":      parts[0] if len(parts) > 0 else "",
            "partition": parts[1] if len(parts) > 1 else "",
            "name":    parts[2] if len(parts) > 2 else "",
            "user":    parts[3] if len(parts) > 3 else "",
            "state":   parts[4] if len(parts) > 4 else "",
            "elapsed": parts[5] if len(parts) > 5 else "",
            "workdir": parts[6] if len(parts) > 6 else "",
        }
        jobs.append(entry)
    return jobs


class SlurmTool(Tool):
    """AI tool for interacting with the NREL Kestrel Slurm HPC scheduler."""

    name = "slurm"
    description = (
        "Interact with the NREL Kestrel Slurm scheduler. "
        "Methods: submit, status, cancel, info, history, job_alive, "
        "wait_for_completion, nodes."
    )

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["submit", "status", "cancel", "info",
                             "history", "job_alive", "nodes"],
                },
                "script_path": {"type": "string"},
                "extra_args": {"type": "array", "items": {"type": "string"}},
                "job_id": {"type": "string"},
                "user": {"type": "string"},
                "partition": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["action"],
        }

    # ── Public API (direct method calls) ──────────────────────────────────────

    def submit(self, script_path: str, extra_args: list[str] = None) -> str:
        """
        Submit a Slurm job.  script_path may be:
          - a filesystem path to an existing .sh file
          - a string containing the full script content (starts with #!)

        Returns the job ID string on success; raises RuntimeError on failure.
        """
        extra_args = extra_args or []
        path = Path(script_path)

        if path.exists():
            cmd = ["sbatch"] + extra_args + [str(path)]
            stdout, stderr, rc = _run(cmd)
        else:
            # Treat as script content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, prefix="hpca_"
            ) as f:
                f.write(script_path)
                tmp = f.name
            try:
                cmd = ["sbatch"] + extra_args + [tmp]
                stdout, stderr, rc = _run(cmd)
            finally:
                os.unlink(tmp)

        if rc != 0:
            raise RuntimeError(f"sbatch failed: {stderr.strip()}")

        # "Submitted batch job 12345"
        for token in stdout.split():
            if token.isdigit():
                return token
        raise RuntimeError(f"Could not parse job ID from: {stdout!r}")

    def status(self, job_id: str = None, user: str = None) -> list[dict]:
        """
        Return a list of job dicts with keys:
        id, partition, name, user, state, elapsed, workdir.
        """
        user = user or os.environ.get("USER", "")
        if job_id:
            cmd = ["squeue", "-j", job_id, "-o", "%.18i %.9P %.30j %.8u %.8T %.12M %Z"]
        else:
            cmd = ["squeue", "-u", user, "-o", "%.18i %.9P %.30j %.8u %.8T %.12M %Z"]
        stdout, stderr, rc = _run(cmd)
        if rc != 0:
            return []
        return _parse_squeue_lines(stdout)

    def cancel(self, job_id: str) -> ToolResult:
        """Cancel a job by ID."""
        if not job_id:
            return ToolResult("Provide job_id to cancel.", success=False)
        stdout, stderr, rc = _run(["scancel", str(job_id)])
        if rc == 0:
            return ToolResult(f"Job {job_id} cancelled.")
        return ToolResult(f"scancel failed: {stderr.strip()}", success=False)

    def info(self, job_id: str) -> dict:
        """
        Return all scontrol show job fields as a dict.
        Keys include: JobId, JobName, JobState, WorkDir, TimeLimit, etc.
        """
        if not job_id:
            return {}
        stdout, stderr, rc = _run(["scontrol", "show", "job", str(job_id)])
        if rc != 0:
            return {"error": stderr.strip()}
        return _parse_scontrol_fields(stdout)

    def history(self, days: int = 7) -> list[dict]:
        """
        Return completed jobs from sacct for the last N days.
        Each dict has: JobID, JobName, State, Elapsed, NCPUS, NodeList.
        """
        user = os.environ.get("USER", "")
        cmd = [
            "sacct", "-u", user,
            "--format", "JobID,JobName,Account,Partition,State,ExitCode,Elapsed,NodeList",
            "-X", "--starttime", f"now-{days}days",
        ]
        stdout, stderr, rc = _run(cmd)
        if rc != 0:
            return []
        lines = stdout.strip().splitlines()
        if len(lines) < 3:
            return []
        # Lines 0 = header, 1 = dashes, 2+ = data
        headers = lines[0].split()
        jobs = []
        for line in lines[2:]:
            parts = line.split()
            if not parts:
                continue
            entry = {}
            for i, h in enumerate(headers):
                entry[h] = parts[i] if i < len(parts) else ""
            jobs.append(entry)
        return jobs

    def job_alive(self, job_id: str) -> bool:
        """Return True if the job is in a running or pending state."""
        jobs = self.status(job_id=str(job_id))
        if not jobs:
            return False
        state = jobs[0].get("state", "")
        return state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING",
                         "RESIZING", "SUSPENDED")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int = 86400,
    ) -> str:
        """
        Block until job finishes (or timeout).
        Returns final Slurm state string (COMPLETED, FAILED, TIMEOUT, etc.).
        """
        deadline = time.time() + timeout
        job_id = str(job_id)
        while time.time() < deadline:
            if not self.job_alive(job_id):
                break
            time.sleep(poll_interval)

        # Query sacct for final state
        cmd = [
            "sacct", "-j", job_id,
            "--format", "State", "-X", "--noheader",
        ]
        stdout, _, rc = _run(cmd)
        if rc == 0 and stdout.strip():
            return stdout.strip().split()[0]
        return "UNKNOWN"

    def nodes(self, partition: str = None) -> list[dict]:
        """Return node info dicts with keys: node, partition, state, cpus, memory, gres, features."""
        cmd = ["sinfo", "-N", "-o", "%N %P %t %c %m %G %f"]
        if partition:
            cmd += ["-p", partition]
        stdout, stderr, rc = _run(cmd)
        if rc != 0:
            return []
        lines = stdout.strip().splitlines()
        if len(lines) < 2:
            return []
        result = []
        headers = ["node", "partition", "state", "cpus", "memory", "gres", "features"]
        for line in lines[1:]:
            parts = line.split(None, len(headers) - 1)
            entry = {}
            for i, h in enumerate(headers):
                entry[h] = parts[i] if i < len(parts) else ""
            result.append(entry)
        return result

    # ── execute() dispatch for LLM tool-call interface ─────────────────────

    def execute(self, action: str = "status", **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        try:
            if action == "submit":
                jid = self.submit(
                    kwargs.get("script_path", ""),
                    kwargs.get("extra_args", []),
                )
                return ToolResult(f"Submitted job {jid}", metadata={"job_id": jid})
            elif action == "status":
                jobs = self.status(
                    job_id=kwargs.get("job_id"),
                    user=kwargs.get("user"),
                )
                if not jobs:
                    return ToolResult("No jobs found.")
                lines = [
                    f"{j['id']:>12}  {j['state']:<12}  {j['name']:<30}  "
                    f"{j['elapsed']:>10}  {j.get('workdir', '')}"
                    for j in jobs
                ]
                return ToolResult("\n".join(lines), metadata={"jobs": jobs})
            elif action == "cancel":
                return self.cancel(kwargs.get("job_id", ""))
            elif action == "info":
                d = self.info(kwargs.get("job_id", ""))
                text = "\n".join(f"{k}={v}" for k, v in d.items())
                return ToolResult(text, metadata=d)
            elif action == "history":
                jobs = self.history(kwargs.get("days", 7))
                if not jobs:
                    return ToolResult("No recent job history.")
                lines = [
                    f"{j.get('JobID',''):>12}  {j.get('State',''):<12}  "
                    f"{j.get('JobName',''):<30}  {j.get('Elapsed','')}"
                    for j in jobs
                ]
                return ToolResult("\n".join(lines), metadata={"jobs": jobs})
            elif action == "job_alive":
                alive = self.job_alive(kwargs.get("job_id", ""))
                return ToolResult(str(alive), metadata={"alive": alive})
            elif action == "nodes":
                nodes = self.nodes(kwargs.get("partition"))
                lines = [
                    f"{n['node']:<20}  {n['state']:<10}  "
                    f"cpus={n['cpus']}  gres={n['gres']}"
                    for n in nodes
                ]
                return ToolResult("\n".join(lines) or "No node info.", metadata={"nodes": nodes})
            else:
                return ToolResult(f"Unknown action: {action}", success=False)
        except Exception as exc:
            return ToolResult(str(exc), success=False)
