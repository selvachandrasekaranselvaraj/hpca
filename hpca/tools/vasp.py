"""
VASP tool: write INCAR/KPOINTS/sub.sh, check convergence, parse OSZICAR.
Wraps hpca.sim.dft with a cleaner method-based interface.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult
from hpca.registry.incar import build_incar as _build_incar, get_incar as _get_incar
from hpca.core.config import account_fallback as _account_fallback

VASP_MODULE = "vasp/6.4.2_openMP"
VASP_BIN = "vasp_std"


class VASPTool(Tool):
    """Tool wrapper for VASP: writes input files, checks convergence, parses outputs."""

    name = "vasp"
    description = (
        "Write VASP input files (INCAR, KPOINTS, sub.sh), set up calculation "
        "directories, check convergence, and parse OSZICAR energies."
    )

    def _parameters(self) -> dict:
        """Return the JSON Schema for the LLM tool-call interface."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "write_incar", "write_kpoints_gamma", "write_sub_sh",
                        "setup_calculation", "check_convergence", "parse_oszicar",
                    ],
                },
                "work_dir": {"type": "string"},
                "template": {"type": "string"},
                "overrides": {"type": "object"},
                "temperature": {"type": "number"},
                "job_name": {"type": "string"},
                "nodes": {"type": "integer"},
                "tasks_per_node": {"type": "integer"},
                "walltime": {"type": "string"},
                "account": {"type": "string"},
                "exclusive": {"type": "boolean"},
                "poscar_src": {"type": "string"},
                "calc_type": {"type": "string"},
                "project_name": {"type": "string"},
            },
            "required": ["action", "work_dir"],
        }

    # ── Public methods ────────────────────────────────────────────────────────

    def write_incar(
        self,
        work_dir: str,
        template: str,
        overrides: dict = None,
        temperature: float = None,
    ) -> Path:
        """
        Write INCAR from template + overrides to work_dir/INCAR.
        For 'aimd' template, sets TEBEG/TEEND to temperature if provided.
        Returns the path to the written INCAR.
        """
        tebeg = int(temperature) if temperature is not None else None
        params = _build_incar(template, tebeg=tebeg, teend=tebeg, extra=overrides or None)

        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        incar_path = d / "INCAR"

        lines = []
        system = params.pop("SYSTEM", "")
        if system:
            lines.append(f"SYSTEM = {system}\n")
        for k, v in params.items():
            lines.append(f" {k} = {v}\n")
        incar_path.write_text("".join(lines))
        return incar_path

    def write_kpoints_gamma(
        self, work_dir: str, mesh: tuple = (2, 2, 2)
    ) -> Path:
        """Write a Gamma-centred KPOINTS file."""
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        kp = d / "KPOINTS"
        kp.write_text(
            f"Automatic mesh\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
        )
        return kp

    def write_sub_sh(
        self,
        work_dir: str,
        job_name: str,
        nodes: int = 2,
        tasks_per_node: int = 104,
        walltime: str = "72:00:00",
        account: str = "",
        exclusive: bool = False,
    ) -> Path:
        """Write a Slurm submission script for VASP."""
        account = account or _account_fallback()
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        sub = d / "sub.sh"

        lines = [
            "#!/bin/bash\n",
            f"#SBATCH --nodes={nodes}\n",
            f"#SBATCH --tasks-per-node={tasks_per_node}\n",
            "#SBATCH --cpus-per-task=1\n",
            f"#SBATCH --time={walltime}\n",
            f"#SBATCH --account={account}\n",
            f"#SBATCH --job-name={job_name}\n",
            "#SBATCH --error=%J.stderr\n",
            "#SBATCH --output=%J.stdout\n",
        ]
        if exclusive:
            lines.append("#SBATCH --exclusive\n")
        lines += [
            "ulimit -s unlimited\n",
            f"module load {VASP_MODULE}\n",
            f"srun {VASP_BIN} &> out\n",
        ]
        sub.write_text("".join(lines))
        sub.chmod(0o755)
        return sub

    def setup_calculation(
        self,
        work_dir: str,
        calc_type: str,
        project_name: str,
        poscar_src: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Create work_dir, write INCAR + KPOINTS + sub.sh.
        Optionally copy POSCAR from poscar_src.

        Returns dict with keys: work_dir, incar, kpoints, sub_sh, poscar (if copied).
        """
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Template selection
        tmpl_map = {
            "vc_relax": ("vc_relax", 1, 96, False),
            "vc":       ("vc_relax", 1, 96, False),
            "opt":      ("opt",      1, 96, False),
            "aimd":     ("aimd",     2, 104, False),
            "bader":    ("bader",    1, 96, False),
            "dos_scf":  ("dos_scf",  1, 96, False),
            "dos":      ("dos_scf",  1, 96, False),
            "dos_nonscf": ("dos_nonscf", 1, 96, False),
            "neb":      ("neb",      1, 88, True),
        }
        tmpl, nodes, tasks, excl = tmpl_map.get(
            calc_type, ("opt", 1, 96, False)
        )

        incar = self.write_incar(d, tmpl, temperature=temperature)
        kpoints = self.write_kpoints_gamma(d)
        sub = self.write_sub_sh(
            d,
            job_name=f"{project_name}_{calc_type}",
            nodes=nodes,
            tasks_per_node=tasks,
            exclusive=excl,
        )

        result = {
            "work_dir": str(d),
            "incar":    str(incar),
            "kpoints":  str(kpoints),
            "sub_sh":   str(sub),
        }

        if poscar_src:
            src = Path(poscar_src)
            if src.exists():
                dst = d / "POSCAR"
                shutil.copy2(str(src), str(dst))
                result["poscar"] = str(dst)

        return result

    def check_convergence(self, work_dir: str) -> dict:
        """
        Check VASP convergence from OUTCAR.
        Returns {converged, n_steps, final_energy, reason}.
        """
        d = Path(work_dir)
        outcar = d / "OUTCAR"
        if not outcar.exists():
            return {"converged": False, "n_steps": 0, "final_energy": None,
                    "reason": "OUTCAR not found"}

        text = outcar.read_text(errors="replace")
        lines = text.splitlines()

        converged = False
        reason = "not converged"
        n_steps = 0
        final_energy = None

        # Count ionic steps
        for line in lines:
            if "- Iteration" in line:
                try:
                    parts = line.split("(")
                    n_steps = int(parts[0].split()[-1])
                except Exception:
                    pass

        # Check last 40 lines for convergence indicators
        tail = "\n".join(lines[-40:])
        if "reached required accuracy" in tail:
            converged = True
            reason = "reached required accuracy"
        elif "ZBRENT: fatal error" in tail:
            reason = "ZBRENT: ionic step convergence failure"
        elif "Sub-Space-Matrix is not hermitian" in tail:
            reason = "Sub-Space-Matrix: SCF instability"
        elif "EDDDAV" in tail:
            reason = "EDDDAV: LAPACK eigensolver failure"
        elif "ZPOTRF" in tail:
            reason = "ZPOTRF: Cholesky factorization failed"
        elif "negative volume" in tail:
            reason = "negative volume: cell collapsed"

        # Parse final energy from last TOTEN occurrence
        for line in reversed(lines):
            if "TOTEN" in line and "eV" in line:
                try:
                    final_energy = float(line.split("=")[1].split("eV")[0])
                    break
                except Exception:
                    pass

        return {
            "converged": converged,
            "n_steps": n_steps,
            "final_energy": final_energy,
            "reason": reason,
        }

    def parse_oszicar(self, work_dir: str) -> list[dict]:
        """
        Parse OSZICAR; return list of dicts per ionic step:
        {step, E, dE, dEps, ncg, rms, mag}
        """
        d = Path(work_dir)
        oszicar = d / "OSZICAR"
        if not oszicar.exists():
            return []

        steps = []
        for line in oszicar.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("T=") and not line[0:2] in ("1 ", "2 ", "3 "):
                # AIMD line: "T=  300  E= ..."
                pass
            # Match both ionic-relax steps and AIMD thermo lines
            if "E=" in line:
                entry = {}
                # Try standard OSZICAR format: "   N F= ... E0= ... d E ..."
                parts = line.split()
                try:
                    # Ionic iteration: "  1 F= -xxx E0= -xxx  d E ..."
                    if len(parts) >= 2 and parts[1] == "F=":
                        entry = {
                            "step": int(parts[0]),
                            "E": float(parts[2]),
                            "dE": float(parts[6]) if len(parts) > 6 else None,
                            "dEps": None, "ncg": None, "rms": None, "mag": None,
                        }
                    # AIMD line: "T= 300 E= -xxx F= ..."
                    elif "T=" in parts:
                        t_idx = parts.index("T=")
                        e_idx = parts.index("E=") if "E=" in parts else -1
                        if e_idx >= 0:
                            entry = {
                                "step": None,
                                "T": float(parts[t_idx + 1]),
                                "E": float(parts[e_idx + 1]),
                                "dE": None, "dEps": None,
                                "ncg": None, "rms": None, "mag": None,
                            }
                    if entry:
                        steps.append(entry)
                except Exception:
                    continue
        return steps

    # ── execute() dispatch for LLM tool-call interface ────────────────────────

    def execute(self, action: str, work_dir: str, **kwargs) -> ToolResult:
        """Dispatch an LLM tool-call action and return a ToolResult."""
        try:
            if action == "write_incar":
                p = self.write_incar(
                    work_dir,
                    kwargs.get("template", "opt"),
                    kwargs.get("overrides"),
                    kwargs.get("temperature"),
                )
                return ToolResult(f"INCAR written: {p}")
            elif action == "write_kpoints_gamma":
                p = self.write_kpoints_gamma(work_dir)
                return ToolResult(f"KPOINTS written: {p}")
            elif action == "write_sub_sh":
                p = self.write_sub_sh(
                    work_dir,
                    kwargs.get("job_name", "vasp_job"),
                    nodes=kwargs.get("nodes", 2),
                    tasks_per_node=kwargs.get("tasks_per_node", 104),
                    walltime=kwargs.get("walltime", "72:00:00"),
                    account=kwargs.get("account") or _account_fallback(),
                    exclusive=kwargs.get("exclusive", False),
                )
                return ToolResult(f"sub.sh written: {p}")
            elif action == "setup_calculation":
                result = self.setup_calculation(
                    work_dir,
                    kwargs.get("calc_type", "opt"),
                    kwargs.get("project_name", "PROJECT"),
                    poscar_src=kwargs.get("poscar_src"),
                    temperature=kwargs.get("temperature"),
                )
                text = "\n".join(f"{k}: {v}" for k, v in result.items())
                return ToolResult(text, metadata=result)
            elif action == "check_convergence":
                d = self.check_convergence(work_dir)
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)
            elif action == "parse_oszicar":
                steps = self.parse_oszicar(work_dir)
                if not steps:
                    return ToolResult("No OSZICAR or no ionic steps found.")
                lines = [
                    f"step={s.get('step','')!s:4s}  E={s.get('E', 'N/A')}"
                    for s in steps[-20:]
                ]
                return ToolResult("\n".join(lines), metadata={"steps": steps})
            else:
                return ToolResult(f"Unknown action: {action}", success=False)
        except Exception as exc:
            return ToolResult(str(exc), success=False)
