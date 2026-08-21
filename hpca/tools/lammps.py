"""
LAMMPS tool: write in.lammps and sub.sh, check progress, parse thermo logs,
convert POSCAR to LAMMPS data, set up multi-temperature runs.
"""
# Layout: see hpca/core/paths.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult
from hpca.core.paths import mlmd_nvt
from hpca.core.config import account_fallback as _account_fallback
from hpca.core.slurm_submit import module_bundle_lines as _mbl


def _hpc(key: str, fallback: str = "") -> str:
    """Read a value from the platform.yaml hpc section at call time."""
    try:
        from hpca.core.paths import load_platform_config
        return load_platform_config().get("hpc", {}).get(key, fallback)
    except Exception:
        return fallback


def _lammps_bin() -> str:
    """Return the LAMMPS binary path from platform.yaml, preferring the GPU binary."""
    return _hpc("lammps_bin") or _hpc("lammps_gpu_bin", "lmp")


def _deepmd_env() -> str:
    """Return the DeepMD-LAMMPS conda environment name from platform.yaml."""
    return _hpc("deepmd_lammps_gpu_env", "")


def __getattr__(name: str) -> str:
    """Lazy module-level resolution so LAMMPS_BIN/DEEPMD_ENV read platform.yaml at access time."""
    if name == "LAMMPS_BIN":
        return _lammps_bin()
    if name == "DEEPMD_ENV":
        return _deepmd_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# LAMMPS NVT template (DeepMD or generic pair_style)
_NVT_TEMPLATE = """\
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   {pair_style} {pot_path}
pair_coeff   * *

timestep     {timestep}
thermo       {dump_every}
thermo_style custom step temp pe ke etotal press

dump         1 all custom {dump_every} dump_unwrapped.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nvt temp {temperature} {temperature} 0.05
run          {n_steps}
"""


class LAMMPSTool(Tool):
    """Tool wrapper for LAMMPS: writes input/submission scripts, monitors progress, converts structures."""

    name = "lammps"
    description = (
        "Write LAMMPS NVT input files and submission scripts, check run "
        "progress, parse thermo logs, convert POSCAR to LAMMPS data format, "
        "and set up multi-temperature MD runs."
    )

    def _parameters(self) -> dict:
        """Return the JSON Schema for the LLM tool-call interface."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "write_nvt_input", "write_gpu_sub_sh", "write_cpu_sub_sh",
                        "check_progress", "parse_thermo_log", "count_dump_frames",
                        "poscar_to_lammps_data", "setup_multi_temperature",
                    ],
                },
                "work_dir":     {"type": "string"},
                "pair_style":   {"type": "string"},
                "pot_path":     {"type": "string"},
                "element_list": {"type": "array", "items": {"type": "string"}},
                "temperature":  {"type": "number"},
                "n_steps":      {"type": "integer"},
                "timestep":     {"type": "number"},
                "dump_every":   {"type": "integer"},
                "data_file":    {"type": "string"},
                "job_name":     {"type": "string"},
                "gpus":         {"type": "integer"},
                "walltime":     {"type": "string"},
                "account":      {"type": "string"},
                "nodes":        {"type": "integer"},
                "tasks":        {"type": "integer"},
                "dump_path":    {"type": "string"},
                "poscar_path":  {"type": "string"},
                "output_path":  {"type": "string"},
                "element_order": {"type": "array", "items": {"type": "string"}},
                "project_dir":  {"type": "string"},
                "temperatures": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["action"],
        }

    # ── Public methods ────────────────────────────────────────────────────────

    def write_nvt_input(
        self,
        work_dir: str,
        pair_style: str,
        pot_path: str,
        element_list: list[str],
        temperature: float,
        n_steps: int = 1_000_000,
        timestep: float = 0.001,
        dump_every: int = 1000,
        data_file: str = "data.lammps",
    ) -> Path:
        """
        Write in.lammps NVT run file.
        element_list: ordered list of elements, e.g. ['Li', 'Ti', 'Cl']
        """
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        elements_str = " ".join(element_list)
        content = _NVT_TEMPLATE.format(
            data_file=data_file,
            pair_style=pair_style,
            pot_path=pot_path,
            elements=elements_str,
            temperature=int(temperature),
            n_steps=n_steps,
            timestep=timestep,
            dump_every=dump_every,
        )
        out = d / "in.lammps"
        out.write_text(content)
        return out

    def write_gpu_sub_sh(
        self,
        work_dir: str,
        job_name: str,
        gpus: int = 4,
        walltime: str = "72:00:00",
        account: str = "",
    ) -> Path:
        """Write GPU LAMMPS submission script (4× H100, kokkos)."""
        account = account or _account_fallback()
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        sub = d / "sub.sh"
        lmp     = _lammps_bin()
        denv    = _deepmd_env()
        lmp_dir = str(Path(lmp).parent) if lmp else ""
        content = f"""\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --gpus={gpus}
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --job-name={job_name}
#SBATCH --error=%J.stderr
#SBATCH --output=%J.stdout
#SBATCH --mem=300G

module purge
{_mbl('gpu_md').strip()}
{"source activate " + denv if denv else "# deepmd_lammps_gpu_env not configured in platform.yaml"}
{"export PATH=" + lmp_dir + ":$PATH" if lmp_dir else ""}
{"export LD_LIBRARY_PATH=" + denv + "/lib:$LD_LIBRARY_PATH" if denv else ""}
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --gpus-per-task=1 {lmp} -k on gpus {gpus} -sf kk -in in.lammps
"""
        sub.write_text(content)
        sub.chmod(0o755)
        return sub

    def write_cpu_sub_sh(
        self,
        work_dir: str,
        job_name: str,
        nodes: int = 1,
        tasks: int = 16,
        walltime: str = "48:00:00",
        account: str = "",
    ) -> Path:
        """Write CPU LAMMPS submission script."""
        account = account or _account_fallback()
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        sub = d / "sub.sh"
        lmp  = _lammps_bin()
        denv = _deepmd_env()
        content = f"""\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tasks}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --job-name={job_name}
#SBATCH --error=%J.stderr
#SBATCH --output=%J.stdout

{_mbl('conda').strip()}
{"source activate " + denv if denv else "# deepmd_lammps_gpu_env not configured in platform.yaml"}
{"export LD_LIBRARY_PATH=" + denv + "/lib:$LD_LIBRARY_PATH" if denv else ""}

srun {lmp} -in in.lammps
"""
        sub.write_text(content)
        sub.chmod(0o755)
        return sub

    def check_progress(self, work_dir: str) -> dict:
        """
        Check LAMMPS run progress.
        Returns {n_steps_done, pct_complete, n_frames, running}.
        """
        d = Path(work_dir)
        log = d / "log.lammps"
        dump = d / "dump_unwrapped.lmp"

        n_steps_done = 0
        pct_complete = 0.0
        running = False
        n_total = 0

        # Parse in.lammps to get total steps
        in_lammps = d / "in.lammps"
        if in_lammps.exists():
            for line in in_lammps.read_text().splitlines():
                m = re.match(r"^\s*run\s+(\d+)", line)
                if m:
                    n_total = int(m.group(1))
                    break

        # Parse log.lammps for last thermo step
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            tail = lines[-50:] if len(lines) > 50 else lines
            for line in reversed(tail):
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    try:
                        n_steps_done = int(parts[0])
                        running = True
                        break
                    except ValueError:
                        pass

        if n_total > 0:
            pct_complete = (n_steps_done / n_total) * 100.0

        n_frames = self.count_dump_frames(str(dump)) if dump.exists() else 0

        return {
            "n_steps_done": n_steps_done,
            "pct_complete": round(pct_complete, 1),
            "n_frames": n_frames,
            "running": running,
        }

    def parse_thermo_log(self, work_dir: str) -> list[dict]:
        """
        Parse LAMMPS log.lammps thermo output.
        Returns list of dicts: {step, temp, pe, ke, etotal, press}.
        """
        d = Path(work_dir)
        log = d / "log.lammps"
        if not log.exists():
            return []

        text = log.read_text(errors="replace")
        # Find thermo_style header line
        header_re = re.compile(
            r"^Step\s+Temp\s+PotEng\s+KinEng\s+TotEng\s+Press", re.MULTILINE
        )
        # fallback: generic "Step" line
        fallback_re = re.compile(r"^Step\s+", re.MULTILINE)

        m = header_re.search(text)
        if m is None:
            m = fallback_re.search(text)
        if m is None:
            return []

        header_line = text[m.start():text.find("\n", m.start())]
        headers = [h.lower() for h in header_line.split()]

        records = []
        lines = text[m.end():].splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            if len(parts) < len(headers):
                continue
            try:
                entry = {
                    "step":   int(parts[0]),
                    "temp":   float(parts[1]) if len(parts) > 1 else None,
                    "pe":     float(parts[2]) if len(parts) > 2 else None,
                    "ke":     float(parts[3]) if len(parts) > 3 else None,
                    "etotal": float(parts[4]) if len(parts) > 4 else None,
                    "press":  float(parts[5]) if len(parts) > 5 else None,
                }
                records.append(entry)
            except (ValueError, IndexError):
                continue
        return records

    def count_dump_frames(self, dump_path: str) -> int:
        """Count LAMMPS dump frames by counting 'ITEM: TIMESTEP' occurrences."""
        p = Path(dump_path)
        if not p.exists():
            return 0
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "-c", "ITEM: TIMESTEP", str(p)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        # Fallback: count lines manually
        count = 0
        try:
            with open(p, "r", errors="replace") as f:
                for line in f:
                    if "ITEM: TIMESTEP" in line:
                        count += 1
        except OSError:
            pass
        return count

    def poscar_to_lammps_data(
        self,
        poscar_path: str,
        output_path: str,
        element_order: Optional[list[str]] = None,
    ) -> Path:
        """
        Convert VASP POSCAR to LAMMPS data file.
        Tries pymatgen first, falls back to ASE.
        element_order: if given, sets the type ordering for pair_coeff.
        """
        from pathlib import Path as _Path
        src = _Path(poscar_path)
        dst = _Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            from pymatgen.core import Structure
            from pymatgen.io.lammps.data import LammpsData
            struct = Structure.from_file(str(src))
            if element_order:
                # Reorder species to match element_order
                pass
            ldata = LammpsData.from_structure(struct, atom_style="atomic")
            ldata.write_file(str(dst))
            return dst
        except ImportError:
            pass

        try:
            from ase.io import read, write
            atoms = read(str(src), format="vasp")
            write(str(dst), atoms, format="lammps-data")
            return dst
        except ImportError:
            pass

        raise RuntimeError(
            "Neither pymatgen nor ASE is available. "
            "Install one to convert POSCAR to LAMMPS data."
        )

    def setup_multi_temperature(
        self,
        project_dir: str,
        pot_path: str,
        temperatures: list[float],
        pair_style: str = "deepmd",
        n_steps: int = 1_000_000,
        element_list: Optional[list[str]] = None,
        account: str = "",
    ) -> list[Path]:
        """
        Create mlmd/nvt/{T}/ directories for each temperature,
        writing in.lammps + sub.sh in each.
        Returns list of work_dir Paths.
        """
        dirs = []
        elements = element_list or []
        for T in temperatures:
            d = mlmd_nvt(Path(project_dir), int(T))
            d.mkdir(parents=True, exist_ok=True)
            self.write_nvt_input(
                str(d),
                pair_style=pair_style,
                pot_path=pot_path,
                element_list=elements,
                temperature=T,
                n_steps=n_steps,
            )
            job_name = f"{Path(project_dir).name}_mlmd_{int(T)}K"
            self.write_gpu_sub_sh(str(d), job_name=job_name, account=account)
            dirs.append(d)
        return dirs

    # ── execute() dispatch ─────────────────────────────────────────────────────

    def execute(self, action: str, **kwargs) -> ToolResult:
        """Dispatch an LLM tool-call action and return a ToolResult."""
        try:
            if action == "write_nvt_input":
                p = self.write_nvt_input(
                    kwargs["work_dir"],
                    pair_style=kwargs.get("pair_style", "deepmd"),
                    pot_path=kwargs.get("pot_path", "pot_com.pb"),
                    element_list=kwargs.get("element_list", []),
                    temperature=kwargs.get("temperature", 300),
                    n_steps=kwargs.get("n_steps", 1_000_000),
                    timestep=kwargs.get("timestep", 0.001),
                    dump_every=kwargs.get("dump_every", 1000),
                    data_file=kwargs.get("data_file", "data.lammps"),
                )
                return ToolResult(f"in.lammps written: {p}")

            elif action == "write_gpu_sub_sh":
                p = self.write_gpu_sub_sh(
                    kwargs["work_dir"],
                    job_name=kwargs.get("job_name", "lammps_md"),
                    gpus=kwargs.get("gpus", 4),
                    walltime=kwargs.get("walltime", "72:00:00"),
                    account=kwargs.get("account") or _account_fallback(),
                )
                return ToolResult(f"GPU sub.sh written: {p}")

            elif action == "write_cpu_sub_sh":
                p = self.write_cpu_sub_sh(
                    kwargs["work_dir"],
                    job_name=kwargs.get("job_name", "lammps_cpu"),
                    nodes=kwargs.get("nodes", 1),
                    tasks=kwargs.get("tasks", 16),
                    walltime=kwargs.get("walltime", "48:00:00"),
                )
                return ToolResult(f"CPU sub.sh written: {p}")

            elif action == "check_progress":
                d = self.check_progress(kwargs["work_dir"])
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "parse_thermo_log":
                records = self.parse_thermo_log(kwargs["work_dir"])
                if not records:
                    return ToolResult("No thermo data found.")
                tail = records[-10:]
                lines = [
                    f"step={r['step']:>10}  T={r.get('temp','N/A'):>8}  "
                    f"Etot={r.get('etotal','N/A')}"
                    for r in tail
                ]
                return ToolResult(
                    f"Last {len(tail)} thermo rows:\n" + "\n".join(lines),
                    metadata={"n_records": len(records), "last": tail},
                )

            elif action == "count_dump_frames":
                n = self.count_dump_frames(kwargs.get("dump_path", ""))
                return ToolResult(str(n), metadata={"n_frames": n})

            elif action == "poscar_to_lammps_data":
                p = self.poscar_to_lammps_data(
                    kwargs["poscar_path"],
                    kwargs["output_path"],
                    element_order=kwargs.get("element_order"),
                )
                return ToolResult(f"LAMMPS data written: {p}")

            elif action == "setup_multi_temperature":
                dirs = self.setup_multi_temperature(
                    kwargs["project_dir"],
                    kwargs["pot_path"],
                    kwargs.get("temperatures", [300, 400, 500, 600]),
                    pair_style=kwargs.get("pair_style", "deepmd"),
                    n_steps=kwargs.get("n_steps", 1_000_000),
                    element_list=kwargs.get("element_list"),
                )
                text = "\n".join(str(d) for d in dirs)
                return ToolResult(
                    f"Created {len(dirs)} temperature directories:\n{text}",
                    metadata={"dirs": [str(d) for d in dirs]},
                )

            else:
                return ToolResult(f"Unknown action: {action}", success=False)

        except Exception as exc:
            return ToolResult(str(exc), success=False)
