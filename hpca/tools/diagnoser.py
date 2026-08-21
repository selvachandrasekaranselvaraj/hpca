"""
DiagnoserTool: detect and auto-fix common VASP, LAMMPS, and DeepMD failures.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult

# ── Error catalogues ──────────────────────────────────────────────────────────

VASP_ERRORS = {
    "ZBRENT": {
        "desc": "Ionic step convergence failure (line search failed)",
        "fix":  "reduce_potim",
    },
    "Sub-Space-Matrix": {
        "desc": "SCF instability: sub-space matrix not Hermitian",
        "fix":  "algo_all",
    },
    "EDDDAV": {
        "desc": "LAPACK eigensolver (EDDDAV) failure",
        "fix":  "algo_fast",
    },
    "SICK_JOB": {
        "desc": "Symmetry precision issue causing job abort",
        "fix":  "reduce_symprec",
    },
    "ZPOTRF": {
        "desc": "Cholesky factorization failed (ZPOTRF/ZTRTRS)",
        "fix":  "lreal_false",
    },
    "no POTCAR": {
        "desc": "Missing POTCAR file — cannot start calculation",
        "fix":  "check_potcar",
    },
    "Inconsistent": {
        "desc": "POTCAR species mismatch with POSCAR",
        "fix":  "fix_potcar",
    },
    "negative volume": {
        "desc": "Cell volume collapsed to negative value during vc-relax",
        "fix":  "fix_cell",
    },
    "NELM": {
        "desc": "Electronic SCF did not converge within NELM steps",
        "fix":  "increase_nelm",
    },
}

LAMMPS_ERRORS = {
    "Lost atoms": {
        "desc": "Atoms escaped the simulation box (likely unstable potential or large timestep)",
        "fix":  "reduce_timestep",
    },
    "Neighbor list overflow": {
        "desc": "Neighbour-list overflow: cutoff radius too large for box size",
        "fix":  "reduce_cutoff",
    },
    "ERROR on proc": {
        "desc": "MPI process error — check LAMMPS/GPU compatibility",
        "fix":  "check_lammps_env",
    },
    "Segmentation fault": {
        "desc": "Segmentation fault — often GPU driver / MPI mismatch",
        "fix":  "check_lammps_env",
    },
}

DEEPMD_ERRORS = {
    "nan": {
        "desc": "NaN in force/energy — descriptor or training data issue",
        "fix":  "switch_descriptor",
    },
    "out of memory": {
        "desc": "GPU out-of-memory during training",
        "fix":  "reduce_batch",
    },
    "model.pb not found": {
        "desc": "Frozen model file missing — dp freeze may not have run",
        "fix":  "rerun_freeze",
    },
    "Invalid argument": {
        "desc": "Type map mismatch between model and input",
        "fix":  "fix_type_map",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_tail(path: Path, n: int = 100) -> str:
    """Read last n lines of a file."""
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def read_incar(path: Path) -> dict:
    """Parse VASP INCAR into a dict."""
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip().upper()] = v.strip().split("!")[0].strip()
    return d


def write_incar(path: Path, updates: dict) -> None:
    """Merge updates into existing INCAR, preserving other keys."""
    existing = read_incar(path)
    existing.update({k.upper(): str(v) for k, v in updates.items()})
    lines = []
    for k, v in existing.items():
        lines.append(f"{k:12s} = {v}\n")
    path.write_text("".join(lines))


# ── Main tool class ───────────────────────────────────────────────────────────

class DiagnoserTool(Tool):
    """AI tool that diagnoses and auto-fixes failures in VASP, LAMMPS, and DeepMD calculations."""

    name = "diagnose"
    description = (
        "Diagnose and auto-fix failures in VASP, LAMMPS, and DeepMD calculations. "
        "Methods: diagnose_vasp, diagnose_lammps, diagnose_deepmd, diagnose (auto-detect), "
        "apply_fix."
    )

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "diagnose_vasp", "diagnose_lammps", "diagnose_deepmd",
                        "diagnose", "apply_fix",
                    ],
                },
                "work_dir": {"type": "string"},
                "calc_type": {
                    "type": "string",
                    "enum": ["vasp", "lammps", "deepmd"],
                    "description": "Force calc type for diagnose action.",
                },
                "fix_name": {"type": "string"},
            },
            "required": ["action", "work_dir"],
        }

    # ── Public methods ────────────────────────────────────────────────────────

    def diagnose_vasp(self, work_dir: str) -> dict:
        """
        Scan VASP output files for known error patterns.
        Returns {error_type, description, fix_name, fix_script_bash}.
        """
        d = Path(work_dir)
        # Check OUTCAR, stderr files, and out file
        search_files = []
        for name in ["OUTCAR", "out"]:
            p = d / name
            if p.exists():
                search_files.append(p)
        for p in d.glob("*.stderr"):
            search_files.append(p)
        for p in d.glob("*.err"):
            search_files.append(p)

        combined = "\n".join(_read_tail(p, 200) for p in search_files)

        for pattern, info in VASP_ERRORS.items():
            if pattern in combined:
                fix_script = _vasp_fix_script(work_dir, info["fix"])
                return {
                    "error_type":     pattern,
                    "description":    info["desc"],
                    "fix_name":       info["fix"],
                    "fix_script_bash": fix_script,
                }

        # No known error found — check if actually converged
        outcar = d / "OUTCAR"
        if outcar.exists() and "reached required accuracy" in outcar.read_text(errors="replace"):
            return {
                "error_type":     "none",
                "description":    "Calculation converged successfully.",
                "fix_name":       None,
                "fix_script_bash": "",
            }

        return {
            "error_type":     "unknown",
            "description":    "No recognised error pattern found. Check OUTCAR manually.",
            "fix_name":       None,
            "fix_script_bash": "",
        }

    def diagnose_lammps(self, work_dir: str) -> dict:
        """Scan LAMMPS log/stderr for known error patterns."""
        d = Path(work_dir)
        search_files = []
        for name in ["log.lammps"]:
            p = d / name
            if p.exists():
                search_files.append(p)
        for p in list(d.glob("*.stderr")) + list(d.glob("*.err")):
            search_files.append(p)

        combined = "\n".join(_read_tail(p, 200) for p in search_files)

        for pattern, info in LAMMPS_ERRORS.items():
            if pattern in combined:
                fix_script = _lammps_fix_script(work_dir, info["fix"])
                return {
                    "error_type":     pattern,
                    "description":    info["desc"],
                    "fix_name":       info["fix"],
                    "fix_script_bash": fix_script,
                }

        return {
            "error_type":     "unknown",
            "description":    "No recognised LAMMPS error. Check log.lammps.",
            "fix_name":       None,
            "fix_script_bash": "",
        }

    def diagnose_deepmd(self, work_dir: str) -> dict:
        """Scan DeepMD training logs for known error patterns."""
        d = Path(work_dir)
        search_files = []
        for name in ["train.log", "lcurve.out"]:
            p = d / name
            if p.exists():
                search_files.append(p)
        for p in list(d.glob("*.stderr")) + list(d.glob("*.err")):
            search_files.append(p)

        combined = "\n".join(_read_tail(p, 200) for p in search_files)
        combined_lower = combined.lower()

        for pattern, info in DEEPMD_ERRORS.items():
            if pattern.lower() in combined_lower:
                fix_script = _deepmd_fix_script(work_dir, info["fix"])
                return {
                    "error_type":     pattern,
                    "description":    info["desc"],
                    "fix_name":       info["fix"],
                    "fix_script_bash": fix_script,
                }

        return {
            "error_type":     "unknown",
            "description":    "No recognised DeepMD error. Check train.log.",
            "fix_name":       None,
            "fix_script_bash": "",
        }

    def diagnose(self, work_dir: str, calc_type: Optional[str] = None) -> dict:
        """
        Auto-detect calculation type from directory contents, then diagnose.
        calc_type override: 'vasp', 'lammps', 'deepmd'.
        """
        d = Path(work_dir)

        if calc_type == "vasp" or (calc_type is None and (d / "INCAR").exists()):
            return self.diagnose_vasp(work_dir)
        elif calc_type == "lammps" or (calc_type is None and (d / "in.lammps").exists()):
            return self.diagnose_lammps(work_dir)
        elif calc_type == "deepmd" or (calc_type is None and (d / "deepmd_input.json").exists()):
            return self.diagnose_deepmd(work_dir)

        # Fallback: try all
        for fn in [self.diagnose_vasp, self.diagnose_lammps, self.diagnose_deepmd]:
            result = fn(work_dir)
            if result["error_type"] != "unknown":
                return result

        return {
            "error_type":     "unknown",
            "description":    "Cannot determine calc type. No INCAR/in.lammps/deepmd_input.json found.",
            "fix_name":       None,
            "fix_script_bash": "",
        }

    def apply_fix(self, work_dir: str, fix_name: str) -> ToolResult:
        """
        Apply an auto-fix to INCAR / in.lammps / deepmd_input.json in work_dir.

        Supported fixes:
          VASP:   reduce_potim, algo_all, algo_fast, reduce_symprec, lreal_false, increase_nelm
          LAMMPS: reduce_timestep
          DeepMD: reduce_batch, switch_descriptor
        """
        d = Path(work_dir)

        # ── VASP fixes ────────────────────────────────────────────────────────
        incar = d / "INCAR"
        if fix_name in ("reduce_potim", "algo_all", "algo_fast",
                        "reduce_symprec", "lreal_false", "increase_nelm") and incar.exists():
            params = read_incar(incar)

            if fix_name == "reduce_potim":
                old = float(params.get("POTIM", 0.5))
                new = old * 0.5
                write_incar(incar, {"POTIM": f"{new:.4f}"})
                return ToolResult(f"INCAR: POTIM reduced {old} → {new}")

            elif fix_name == "algo_all":
                write_incar(incar, {"ALGO": "All"})
                return ToolResult("INCAR: ALGO set to All")

            elif fix_name == "algo_fast":
                write_incar(incar, {"ALGO": "Fast", "NELM": 120})
                return ToolResult("INCAR: ALGO=Fast, NELM=120")

            elif fix_name == "reduce_symprec":
                write_incar(incar, {"SYMPREC": "1E-4"})
                return ToolResult("INCAR: SYMPREC set to 1E-4")

            elif fix_name == "lreal_false":
                write_incar(incar, {"LREAL": ".FALSE."})
                return ToolResult("INCAR: LREAL set to .FALSE.")

            elif fix_name == "increase_nelm":
                old = int(params.get("NELM", 60))
                new = old * 2
                write_incar(incar, {"NELM": new})
                return ToolResult(f"INCAR: NELM increased {old} → {new}")

        # ── LAMMPS fixes ──────────────────────────────────────────────────────
        in_lammps = d / "in.lammps"
        if fix_name == "reduce_timestep" and in_lammps.exists():
            text = in_lammps.read_text()
            m = re.search(r"(timestep\s+)([\d.eE+-]+)", text)
            if m:
                old_ts = float(m.group(2))
                new_ts = old_ts * 0.5
                text = text.replace(m.group(0), f"{m.group(1)}{new_ts:.6f}")
                in_lammps.write_text(text)
                return ToolResult(f"in.lammps: timestep reduced {old_ts} → {new_ts}")
            return ToolResult("Could not find 'timestep' in in.lammps.", success=False)

        # ── DeepMD fixes ──────────────────────────────────────────────────────
        deepmd_json = d / "deepmd_input.json"
        if fix_name in ("reduce_batch", "switch_descriptor") and deepmd_json.exists():
            import json as _json
            with open(deepmd_json) as f:
                inp = _json.load(f)

            if fix_name == "reduce_batch":
                bs = inp.get("training", {}).get("training_data", {}).get("batch_size", 32)
                new_bs = max(1, bs // 2)
                inp["training"]["training_data"]["batch_size"] = new_bs
                inp["training"].setdefault("validation_data", {})["batch_size"] = new_bs
                deepmd_json.write_text(_json.dumps(inp, indent=2))
                return ToolResult(f"deepmd_input.json: batch_size reduced {bs} → {new_bs}")

            elif fix_name == "switch_descriptor":
                inp["model"]["descriptor"]["type"] = "se_e2_a"
                deepmd_json.write_text(_json.dumps(inp, indent=2))
                return ToolResult("deepmd_input.json: descriptor switched to se_e2_a")

        # ── Unknown / file missing ────────────────────────────────────────────
        return ToolResult(
            f"Fix '{fix_name}' could not be applied in {work_dir}. "
            f"Check that the relevant input file exists.",
            success=False,
        )

    # ── execute() dispatch ─────────────────────────────────────────────────────

    def execute(self, action: str, work_dir: str, **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        try:
            if action == "diagnose_vasp":
                d = self.diagnose_vasp(work_dir)
                return _diag_result(d)
            elif action == "diagnose_lammps":
                d = self.diagnose_lammps(work_dir)
                return _diag_result(d)
            elif action == "diagnose_deepmd":
                d = self.diagnose_deepmd(work_dir)
                return _diag_result(d)
            elif action == "diagnose":
                d = self.diagnose(work_dir, calc_type=kwargs.get("calc_type"))
                return _diag_result(d)
            elif action == "apply_fix":
                return self.apply_fix(work_dir, kwargs.get("fix_name", ""))
            else:
                return ToolResult(f"Unknown action: {action}", success=False)
        except Exception as exc:
            return ToolResult(str(exc), success=False)


# ── Bash fix-script generators ────────────────────────────────────────────────

def _vasp_fix_script(work_dir: str, fix_name: str) -> str:
    """Generate a bash one-liner to apply the VASP fix."""
    d = work_dir
    scripts = {
        "reduce_potim":   f"cd {d} && python3 -c \"from hpca.tools.diagnoser import DiagnoserTool; DiagnoserTool().apply_fix('{d}', 'reduce_potim')\"",
        "algo_all":       f"cd {d} && sed -i 's/^ALGO.*/ALGO         = All/' INCAR || echo 'ALGO         = All' >> INCAR",
        "algo_fast":      f"cd {d} && sed -i 's/^ALGO.*/ALGO         = Fast/' INCAR",
        "reduce_symprec": f"cd {d} && echo 'SYMPREC      = 1E-4' >> INCAR",
        "lreal_false":    f"cd {d} && sed -i 's/^LREAL.*/LREAL        = .FALSE./' INCAR",
        "check_potcar":   f"ls -la {d}/POTCAR",
        "fix_potcar":     f"echo 'Regenerate POTCAR for species in POSCAR line 6'",
        "fix_cell":       f"echo 'Copy a previously converged CONTCAR as POSCAR and restart with smaller NSW'",
        "increase_nelm":  f"cd {d} && sed -i 's/^NELM.*/NELM         = 200/' INCAR",
    }
    return scripts.get(fix_name, f"# No bash script for fix '{fix_name}'")


def _lammps_fix_script(work_dir: str, fix_name: str) -> str:
    """Generate a bash one-liner to apply the LAMMPS fix."""
    try:
        from hpca.core.paths import load_platform_config as _lpc_d
        _lmp = _lpc_d().get("hpc", {}).get("lammps_bin", "lmp")
    except Exception:
        _lmp = "lmp"
    try:
        from hpca.core.slurm_submit import module_bundle_lines as _mbl
        _mod_line = _mbl("gpu_md").strip() or "true"
    except Exception:
        _mod_line = "true"
    scripts = {
        "reduce_timestep":    f"cd {work_dir} && sed -i 's/^timestep.*/timestep     0.0005/' in.lammps",
        "reduce_cutoff":      f"# Reduce rcut in pair_style or use a larger box",
        "check_lammps_env":   f"{_mod_line} && {_lmp} -h | head",
    }
    return scripts.get(fix_name, f"# No bash script for fix '{fix_name}'")


def _deepmd_fix_script(work_dir: str, fix_name: str) -> str:
    """Generate a bash one-liner to apply the DeepMD fix."""
    try:
        from hpca.core.paths import load_platform_config as _lpc_d
        _hpc_d = _lpc_d().get("hpc", {})
        _denv = _hpc_d.get("deepmd_lammps_gpu_env", "")
    except Exception:
        _denv = ""
    _activate = f"source activate {_denv}" if _denv else "# deepmd_lammps_gpu_env not configured"
    scripts = {
        "reduce_batch":    f"cd {work_dir} && python3 -c \"from hpca.tools.diagnoser import DiagnoserTool; DiagnoserTool().apply_fix('{work_dir}', 'reduce_batch')\"",
        "switch_descriptor": f"cd {work_dir} && python3 -c \"from hpca.tools.diagnoser import DiagnoserTool; DiagnoserTool().apply_fix('{work_dir}', 'switch_descriptor')\"",
        "rerun_freeze":    f"cd {work_dir} && {_activate} && dp freeze -o pot.pb",
        "fix_type_map":    f"# Edit 'type_map' in deepmd_input.json to match POSCAR species order",
    }
    return scripts.get(fix_name, f"# No bash script for fix '{fix_name}'")


def _diag_result(d: dict) -> ToolResult:
    """Format a diagnosis result dict into a ToolResult with human-readable output."""
    lines = [
        f"Error type:  {d.get('error_type', 'unknown')}",
        f"Description: {d.get('description', '')}",
        f"Fix:         {d.get('fix_name', 'N/A')}",
    ]
    if d.get("fix_script_bash"):
        lines.append(f"Bash fix:\n  {d['fix_script_bash']}")
    return ToolResult(
        "\n".join(lines),
        success=(d.get("error_type") in ("none", "unknown")),
        metadata=d,
    )
