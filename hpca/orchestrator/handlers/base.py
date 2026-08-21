"""
base.py — Abstract base class for all HPCA simulation-type handlers.
"""
from __future__ import annotations

import logging
from hpca.core.config import account_fallback as _account_fallback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")


class SimulationHandler(ABC):
    """
    Abstract base for one simulation type (DFT, AIMD, NEB, MLIP, etc.).

    Subclasses set:
      name      -- identifier string, e.g. "h02_aimd"
      is_daemon -- True = runs in-process; False = submits SLURM job
    """

    name: str = ""
    is_daemon: bool = False

    @property
    def stage_definition(self):
        """Return canonical declarative metadata for this execution adapter."""
        from hpca.registry.stage import get_stage
        return get_stage(self.name)

    @property
    def execution_lane(self) -> str:
        """Canonical execution lane; registry metadata is authoritative."""
        return self.stage_definition.lane.value

    @abstractmethod
    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True if prerequisites are met (filesystem checks)."""

    @abstractmethod
    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True if this simulation type is fully done."""

    @abstractmethod
    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """
        For SLURM handlers: submit job and return job_id string.
        For daemon handlers: run the computation directly, return None.
        """

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Update progress metadata in state (step counts, RMSE, etc.); no-op by default."""

    def on_complete(self, project_dir: Path, state: "ProjectState") -> None:
        """Hook called once when is_complete() first returns True; no-op by default."""

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """
        Detect failure and apply an auto-fix.
        Returns True if a fix was applied (caller should resubmit).
        """
        return False

    # ── SLURM helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def sbatch(script: Path, cwd: Path | None = None,
               extra_args: list | None = None) -> str | None:
        """Submit through the scheduler adapter."""
        from hpca.scheduler import get_scheduler
        return get_scheduler().submit(script, cwd=cwd, extra_args=extra_args)

    @staticmethod
    def job_alive(job_id: str | None) -> bool:
        """Return True if job is in RUNNING, PENDING, or COMPLETING state."""
        if not job_id:
            return False
        from hpca.scheduler import get_scheduler
        return get_scheduler().alive(job_id)

    @staticmethod
    def read_project_yaml(project_dir: Path) -> dict:
        """Read project.yaml, return dict. Returns {} if missing or unparseable."""
        p = Path(project_dir) / "project.yaml"
        if not p.exists():
            return {}
        try:
            import yaml
            return yaml.safe_load(p.read_text()) or {}
        except Exception as exc:
            log.warning("Cannot read %s: %s", p, exc)
            return {}

    @staticmethod
    def simulation_approved(project_dir: Path, project_yaml: dict | None = None) -> bool:
        """Return True if design has been approved for SLURM submission.

        The design phase (h00_design) runs on the daemon/login node and writes
        design/DESIGN_COMPLETE.md when finished.  The orchestrator then pauses
        all SLURM-submission handlers until the user creates the approval flag:

            touch <project_dir>/design/simulation_approved.flag

        If the flag file does not exist AND design/ does not exist either
        (e.g. non-polymer projects), this returns True so legacy projects
        are not blocked.
        """
        from hpca.core.autonomy import AutonomyPolicy
        design_dir = Path(project_dir) / "design"
        designed_dir = Path(project_dir) / "designed_structures"
        if not design_dir.exists() and not designed_dir.exists():
            return True   # no design phase → always approved
        return AutonomyPolicy.from_project(project_yaml or {}).design_approved(Path(project_dir))


    @staticmethod
    def _outcar_summary(work_dir: Path) -> str:
        """Parse OUTCAR for final energy and ionic step count. Returns a short string."""
        outcar = work_dir / "OUTCAR"
        if not outcar.exists():
            return "no OUTCAR"
        energy: float | None = None
        n_ionic = 0
        try:
            for line in outcar.read_text(errors="ignore").splitlines():
                if "TOTEN" in line and "eV" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        try:
                            energy = float(parts[-1].split()[0])
                        except ValueError:
                            pass
                elif "- Iteration" in line:
                    try:
                        n_ionic = max(n_ionic, int(line.split("(")[0].split()[-1]))
                    except (ValueError, IndexError):
                        pass
        except Exception:
            return "OUTCAR unreadable"
        parts = []
        if energy is not None:
            parts.append(f"E={energy:.4f} eV")
        if n_ionic:
            parts.append(f"{n_ionic} ionic steps")
        return ", ".join(parts) if parts else "no energy found"

    @staticmethod
    def platform_config() -> dict:
        """Return the parsed platform.yaml as a dict (cached per-process)."""
        import yaml as _yaml
        cfg_path = Path(__file__).parents[2] / "config" / "platform.yaml"
        try:
            return _yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            return {}

    @classmethod
    def hpc_path(cls, key: str, default: str = "") -> str:
        """Look up an hpc.{key} path from platform.yaml."""
        return cls.platform_config().get("hpc", {}).get(key, default)

    @classmethod
    def sim_limit(cls, lane: str, key: str, default=None):
        """Look up a limits.{lane}.{key} value from platform.yaml.

        lane: 'slurm' (only lane supported)
        """
        return cls.platform_config().get("limits", {}).get(lane, {}).get(key, default)

    @classmethod
    def resolve(cls, project_dir: Path, key: str, default=None):
        """Return the simulation parameter for the SLURM lane.

        Reads project.yaml simulation: first, falls back to limits.slurm in platform.yaml.
        """
        yaml = cls.read_project_yaml(project_dir)
        val = yaml.get("simulation", {}).get(key)
        if val is not None:
            return val
        val = cls.sim_limit("slurm", key)
        return val if val is not None else default

    @classmethod
    def plat(cls, section: str, key: str, default=None):
        """Read any top-level section from platform.yaml.

        Example: self.plat('lammps_md', 'timestep_fs_cmd', 2.0)
        """
        return cls.platform_config().get(section, {}).get(key, default)

    @classmethod
    def slurm_time(cls, key: str, default: str = "48:00:00") -> str:
        """Return the SLURM walltime string for a given stage key.

        Example: self.slurm_time('dft_vc') → '48:00:00'
        """
        return cls.platform_config().get("slurm_time", {}).get(key, default)

    @classmethod
    def nvt_temperatures(cls) -> list:
        """Return the canonical NVT temperature list from platform.yaml."""
        return cls.platform_config().get("limits", {}).get(
            "nvt_temperatures", [300, 320, 340, 360, 380, 400, 500, 600]
        )

    @staticmethod
    def _poscar_is_valid(path: Path, min_size_bytes: int = 200) -> bool:
        """Return False if the POSCAR is a placeholder or clearly invalid.

        Checks:
        1. File exists and is large enough to be a real structure.
        2. First line does not contain 'placeholder' (written by grid/PACKMOL fallback).
        3. Minimum interatomic distance > 1.0 Å — only for small structures (<2000 atoms).
           The O(N²) distance check allocates N²×3×8 bytes; for CMD-scale POSCARs
           (50000+ atoms) this would be 60 GB+ per call, causing OOM when many threads
           run the check simultaneously.  Large structures are assumed valid if they
           pass the placeholder check above.
        """
        if not path.exists() or path.stat().st_size < min_size_bytes:
            return False
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            return False
        if not lines or "placeholder" in lines[0].lower():
            return False
        # Atom-count check: line 6 has per-element counts (modern POSCAR format).
        # Fall back to line 5 for old-style POSCARs where line 5 = counts.
        try:
            if len(lines) >= 7:
                try:
                    atom_count = sum(int(x) for x in lines[6].split())
                except ValueError:
                    try:
                        atom_count = sum(int(x) for x in lines[5].split())
                    except ValueError:
                        atom_count = 0
                if atom_count >= 2000:
                    return True   # large structure: skip O(N²) distance check
            from hpca.core.structure_check import min_distance_poscar
            if min_distance_poscar(path) < 1.0:
                return False
        except Exception:
            pass   # if check fails, don't block on it
        return True

    @staticmethod
    def _compute_encut(potcar: Path, factor: float = 1.3) -> float:
        """Return factor × max(ENMAX) read from a POTCAR. Falls back to 520 eV.

        factor=1.3 (VASP standard) is used universally across DFT, AIMD, and NEB.
        Can be overridden per-project via encut: or encut_neb: in project.yaml.
        """
        enmax_vals: list[float] = []
        try:
            for line in potcar.read_text(errors="replace").splitlines():
                if "ENMAX" in line:
                    for part in line.split(";"):
                        if "ENMAX" in part:
                            try:
                                enmax_vals.append(
                                    float(part.split("=")[1].strip().split()[0])
                                )
                            except (ValueError, IndexError):
                                pass
        except OSError:
            pass
        if not enmax_vals:
            return 520.0
        return round(factor * max(enmax_vals), 1)

    @classmethod
    def project_encut(cls, project_dir: Path, yaml_data: dict) -> float:
        """Return the project-wide ENCUT (eV) — one value for DFT, AIMD, and NEB.

        Priority:
          1. explicit `encut:` (or `encut_neb:`) in project.yaml  → honour user override
          2. 1.3 × max(ENMAX) from the first POTCAR found in standard locations
          3. 520 eV hard-coded fallback (only if no POTCAR exists yet)
        """
        explicit = yaml_data.get("encut") or yaml_data.get("encut_neb")
        if explicit:
            return float(explicit)
        potcar_candidates = [
            project_dir / "designed_structures" / "POTCAR",
            project_dir / "dft" / "vc"  / "POTCAR",
            project_dir / "dft" / "opt" / "POTCAR",
            project_dir / "bader"        / "POTCAR",
        ]
        for p in potcar_candidates:
            if p.exists():
                encut = cls._compute_encut(p)
                log.debug("[hpca] project_encut: %.1f eV from %s", encut, p)
                return encut
        log.debug("[hpca] project_encut: no POTCAR found in %s — using 520 eV fallback", project_dir)
        return 520.0

    @classmethod
    def _write_vasp_sub_sh(
        cls,
        path: Path,
        job_name: str,
        nodes: int,
        ntasks: int,
        time: str,
        account: str = "",
    ) -> None:
        """Write a SLURM submission script that runs VASP (vasp_std).

        Shared by DFT and AIMD handlers — all HPC parameters come from platform.yaml.
        No --partition is written; SLURM auto-assigns based on wall time.
        """
        hpc      = cls.platform_config().get("hpc", {})
        acct     = account   or hpc.get("accounts", {}).get("standard") or _account_fallback()
        vasp_mod = hpc.get("vasp_module", "vasp/6.4.2_openMP")
        work_dir = str(path.parent)
        path.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks-per-node={ntasks}\n"
            "#SBATCH --cpus-per-task=1\n"
            "#SBATCH --mem=0\n"
            f"#SBATCH --time={time}\n"
            f"#SBATCH --account={acct}\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --error={work_dir}/%J.stderr\n"
            f"#SBATCH --output={work_dir}/%J.stdout\n"
            "ulimit -s unlimited\n"
            "module purge\n"
            f"module load {vasp_mod}\n"
            f"cd {work_dir}\n"
            "srun vasp_std &> out\n"
        )
        path.chmod(0o755)

    @staticmethod
    def grep_count(path: Path, pattern: str) -> int:
        """Count lines matching pattern in file. Fast using grep."""
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "-c", pattern, str(path)],
                capture_output=True, text=True, timeout=30
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            return 0

    # ── POSCAR utilities (shared across DFT, AIMD, NEB handlers) ─────────────

    @staticmethod
    def _read_poscar_lines(poscar: Path) -> list[str]:
        """Return non-empty stripped lines from a POSCAR file."""
        return [
            line.strip()
            for line in poscar.read_text(errors="replace").splitlines()
            if line.strip()
        ]

    def _get_poscar_elements(self, poscar: Path) -> list[str]:
        """Return the element symbols from POSCAR line 6 (modern format); empty list on failure."""
        import re as _re
        try:
            lines = self._read_poscar_lines(poscar)
            if len(lines) < 7:
                return []
            tokens = lines[5].split()
            if all(_re.fullmatch(r"[A-Z][a-z]?", t) for t in tokens):
                return tokens
            return []
        except OSError:
            return []

    def _count_atoms_poscar(self, poscar: Path, default: int = 200) -> int:
        """Return total atom count from POSCAR; returns default if parsing fails."""
        import re as _re
        try:
            lines = self._read_poscar_lines(poscar)
            if len(lines) < 7:
                return default
            line6 = lines[5].split()
            line7 = lines[6].split()
            if all(_re.fullmatch(r"[A-Z][a-z]?", t) for t in line6):
                return sum(int(x) for x in line7)
            return sum(int(x) for x in line6)
        except (OSError, ValueError, IndexError):
            log.warning("[base] Cannot count atoms in %s", poscar)
            return default
