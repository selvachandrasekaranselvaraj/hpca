"""
ai_advisor.py — Connects HPCA orchestrator to Claude Code for AI reasoning.

Uses `claude -p` (non-interactive print mode) for all inference.
Falls back gracefully when Claude Code is unavailable.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from hpca.core.paths import load_platform_config

log = logging.getLogger("hpca.ai_advisor")


def _claude_bin() -> Path:
    """Return path to the claude binary from platform.yaml hpc.claude_bin."""
    return Path(load_platform_config().get("hpc", {}).get("claude_bin", "claude"))

SYSTEM_PROMPT = (
    "You are an expert computational materials scientist and HPC engineer on NREL Kestrel. "
    "You direct battery materials simulation workflows: VASP DFT → DeepMD MLIP → LAMMPS MD "
    "→ analysis → manuscript. Be concise, technical, and specific. "
    "When writing bash fix scripts, use exact file paths and minimal changes."
)


class AIAdvisor:
    """Routes orchestrator reasoning tasks to Claude Code via `claude -p`."""

    def __init__(self):
        """Initialise with no cached availability state."""
        self._available: Optional[bool] = None
        self._checked_at: float = 0.0

    def is_available(self) -> bool:
        """Return True if the claude binary exists; result is cached for 5 minutes."""
        now = time.time()
        if self._available is not None and (now - self._checked_at) < 300:
            return self._available
        self._available = _claude_bin().exists()
        self._checked_at = now
        return self._available

    def _ask(self, prompt: str, timeout: int = 120) -> Optional[str]:
        """Send prompt to claude -p and return the stripped stdout, or None on failure."""
        if not self.is_available():
            return None
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        t0 = time.time()
        try:
            result = subprocess.run(
                [str(_claude_bin()), "-p", full_prompt, "--dangerously-skip-permissions"],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                log.warning("claude -p exit %d: %s", result.returncode, result.stderr[:200])
                return None
            response = result.stdout.strip()
            log.info("AI: %d chars in %.1fs", len(response), time.time() - t0)
            return response
        except subprocess.TimeoutExpired:
            log.warning("claude -p timed out after %ds", timeout)
            return None
        except Exception as exc:
            log.warning("claude -p error: %s", exc)
            return None

    def fix_failed_handler(self, project_dir: Path, handler_name: str,
                           stderr_path: Path = None) -> Optional[str]:
        """Ask Claude to generate a bash fix script for a failed handler.

        Parameters
        ----------
        stderr_path
            Optional path to a SLURM stderr file; last 150 lines are included in context.

        Returns
        -------
        str or None
            Bash script body extracted from the Claude response, or None if unavailable.
        """
        parts = [f"Project: {project_dir.name}", f"Failed handler: {handler_name}"]
        if stderr_path and stderr_path.exists():
            lines = stderr_path.read_text(errors="replace").splitlines()[-150:]
            parts.append("=== STDERR (last 150 lines) ===")
            parts.extend(lines)
        for candidate in ["INCAR", "in.lammps", "deepmd_input.json"]:
            for root in [project_dir] + list(project_dir.glob("*/")):
                p = root / candidate
                if p.exists():
                    parts.append(f"=== {p.name} ===\n{p.read_text(errors='replace')[:2000]}")
                    break
        prompt = ("\n".join(parts) +
                  f"\n\nWrite a bash script to fix this failure, run from {project_dir}. "
                  "Output ONLY the script between ```bash and ``` markers.")
        response = self._ask(prompt, timeout=90)
        if not response:
            return None
        if "```bash" in response:
            s = response.index("```bash") + 7
            e = response.index("```", s)
            return response[s:e].strip()
        if "```" in response:
            s = response.index("```") + 3
            e = response.index("```", s)
            return response[s:e].strip()
        return response

    def plan_next_stage(self, project_dir: Path, current_state: dict,
                        completed_handler: str) -> dict:
        """Ask Claude whether any subsequent handlers should be skipped or parameter-adjusted.

        Returns
        -------
        dict
            Keys: skip_handlers (list), parameter_overrides (dict), notes (str).
        """
        completed = [k for k, v in current_state.items() if v == "COMPLETE"]
        running   = [k for k, v in current_state.items() if v == "RUNNING"]
        prompt = (
            f"Project: {project_dir.name}\nJust completed: {completed_handler}\n"
            f"Complete: {completed}\nRunning: {running}\n\n"
            "Should any subsequent stages be skipped or have parameters adjusted? "
            'Reply JSON only: {"skip_handlers": [], "parameter_overrides": {}, "notes": "..."}'
        )
        response = self._ask(prompt, timeout=60)
        if not response:
            return {"skip_handlers": [], "parameter_overrides": {}, "notes": ""}
        try:
            s = response.index("{"); e = response.rindex("}") + 1
            return json.loads(response[s:e])
        except Exception:
            return {"skip_handlers": [], "parameter_overrides": {}, "notes": response[:200]}

    def narrate_results(self, project_dir: Path, analysis_results: dict) -> str:
        """Return 2-3 manuscript-quality paragraphs interpreting the analysis results."""
        lines = [f"Project: {project_dir.name}"]
        for k, v in analysis_results.items():
            lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
        prompt = ("\n".join(lines) +
                  "\n\nWrite 2–3 manuscript-quality paragraphs with physical interpretation. "
                  "Use LaTeX notation for units.")
        return self._ask(prompt, timeout=120) or ""

    def diagnose_anomaly(self, project_dir: Path, temps_K: list,
                         D_vals: list, R2: float) -> str:
        """Ask Claude to diagnose a poor Arrhenius R² fit (phase transitions, non-Arrhenius, etc.)."""
        pairs = ", ".join(f"{T}K→{D:.2e}" for T, D in zip(temps_K, D_vals))
        prompt = (f"Project: {project_dir.name}\nArrhenius R²={R2:.3f} (poor)\n"
                  f"T→D: {pairs}\n\n"
                  "Diagnose why the fit is poor. Consider phase transitions, "
                  "non-Arrhenius behaviour, insufficient sampling.")
        return self._ask(prompt, timeout=90) or ""

    def plan_active_learning(self, project_dir: Path, E_rmse: float,
                             F_rmse: float) -> dict:
        """Ask Claude to suggest new AIMD temperatures to close MLIP accuracy gaps.

        Returns
        -------
        dict
            Keys: add_temperatures (list of int), add_configurations (int), reasoning (str).
        """
        existing: list = []
        yp = project_dir / "project.yaml"
        if yp.exists():
            try:
                import yaml
                cfg = yaml.safe_load(yp.read_text()) or {}
                ad = cfg.get("aimd_dirs", {})
                existing = sorted(ad.keys()) if isinstance(ad, dict) else []
            except Exception:
                pass
        prompt = (f"Project: {project_dir.name}\n"
                  f"MLIP errors: E={E_rmse:.2f} meV/atom, F={F_rmse:.1f} meV/Å\n"
                  f"Existing AIMD temps: {existing} K\n\n"
                  "Suggest temperatures to add. "
                  'JSON only: {"add_temperatures": [], "add_configurations": 5000, "reasoning": ""}')
        response = self._ask(prompt, timeout=60)
        default = {"add_temperatures": [], "add_configurations": 5000, "reasoning": ""}
        if not response:
            return default
        try:
            s = response.index("{"); e = response.rindex("}") + 1
            return json.loads(response[s:e])
        except Exception:
            return default

    def summarize_project(self, project_dir: Path) -> str:
        """Return a 3-sentence weekly summary of completed, running, and recommended next steps."""
        state_path = project_dir / "logs" / "orchestrator_state.json"
        info = ""
        if state_path.exists():
            try:
                s = json.loads(state_path.read_text()).get("stages", {})
                done = [k for k, v in s.items() if v.get("status") == "COMPLETE"]
                fail = [k for k, v in s.items() if v.get("status") == "FAILED"]
                info = f"Done: {done}  Failed: {fail}"
            except Exception:
                pass
        prompt = (f"Project: {project_dir.name}\n{info}\n\n"
                  "3-sentence weekly summary: what's done, running, and recommended next step.")
        return self._ask(prompt, timeout=60) or f"Project {project_dir.name}: AI unavailable"

    def generate_project_yaml(self, answers: dict) -> str:
        """Ask Claude to generate a valid project.yaml from user-provided key-value answers."""
        prompt = ("Generate a project.yaml for the HPCA orchestrator:\n"
                  + "\n".join(f"  {k}: {v}" for k, v in answers.items())
                  + "\n\nMust include: name, full_name, mobile_ion, category, T_ref, root, "
                  "aimd_dirs, mlmd_dirs, stages block. Output ONLY valid YAML.")
        return self._ask(prompt, timeout=90) or ""
