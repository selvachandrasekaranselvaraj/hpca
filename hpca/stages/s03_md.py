"""
Stage 03 — LAMMPS MD submission.

Supports: DeepMD NVT, MACE-MP0 NVT, UMA NVT, multi-temperature loops.
Canonical layout:
  mlmd/mlff/   — MLFF potential
  mlmd/npt/    — NPT run (reads poscar_mlmd or data.lammps)
  mlmd/nvt/{T}/ — NVT production runs
  cmd/npt/     — CMD NPT equilibration
  cmd/nvt/{T}/ — CMD NVT production
Generates in.lammps and sub.sh, then optionally submits.
"""
# Layout: see hpca/core/paths.py
from __future__ import annotations
import subprocess
from pathlib import Path

from hpca.core.paths import mlmd_mlff, mlmd_nvt, cmd_nvt, pot_com_pb, load_platform_config

# Workflow helpers from sim/md.py
from hpca.sim.md import (
    setup_mlmd_workflow, setup_cmd_workflow,
    MLMD_TEMPERATURES, DAEMON_MD_LIMITS,
)
from hpca.core.slurm_submit import module_bundle_lines
from hpca.core.config import account_fallback as _account_fallback

# HPC paths read from platform.yaml — cross-ref: hpca/config/platform.yaml
def _hpc(key: str, default: str = "") -> str:
    """Return the HPC config value for key from platform.yaml."""
    return load_platform_config().get("hpc", {}).get(key, default)


# ── LAMMPS input templates ────────────────────────────────────────────────────

LAMMPS_DEEPMD = """\
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   deepmd {pot_file}
pair_coeff   * *

timestep     0.001
thermo       1000
thermo_style custom step temp pe ke etotal press

dump         1 all custom 1000 dump_unwrapped.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nvt temp {T} {T} 0.05
run          {n_steps}
"""

LAMMPS_MACE = """\
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   mace no_domain_decomposition
pair_coeff   * * {model_path} {elements}

timestep     0.001
thermo       1000
thermo_style custom step temp pe ke etotal press

dump         1 all custom 1000 dump_unwrapped.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nvt temp {T} {T} 0.05
run          {n_steps}
"""

LAMMPS_NVE = """\
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   deepmd {pot_file}
pair_coeff   * *

timestep     0.001
thermo       1000
thermo_style custom step temp pe ke etotal press

dump         1 all custom 1000 dump_unwrapped.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nve
run          {n_steps}
"""


def write_gpu_sub(path: Path, job_name: str, conda_env: str = "",
                   time: str = "72:00:00", account: str = "",
                   mem: str = "300G"):
    """Write a GPU SLURM submission script for LAMMPS to path."""
    hpc         = load_platform_config().get("hpc", {})
    lammps_bin  = hpc.get("lammps_bin", "")
    py_deepmd   = hpc.get("python_deepmd", "")
    env         = conda_env or hpc.get("deepmd_lammps_gpu_env", "") or (str(Path(py_deepmd).parent.parent) if py_deepmd else "")
    acct        = account   or hpc.get("accounts", {}).get("gpu_h100") or _account_fallback()
    path.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --account={acct}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --gpus=4\n"
        "#SBATCH --ntasks-per-node=4\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --mem={mem}\n"
        "#SBATCH --output=%j.stdout\n"
        "#SBATCH --error=%j.stderr\n"
        "\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md')}"
        f"source activate {env}\n"
        f"export PATH={Path(lammps_bin).parent}:$PATH\n"
        f"export LD_LIBRARY_PATH={env}/lib:$LD_LIBRARY_PATH\n"
        "export MPICH_GPU_SUPPORT_ENABLED=1\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        f"srun --gpus-per-task=1 {lammps_bin} -k on gpus 4 -sf kk -in in.lammps\n"
    )
    path.chmod(0o755)


def setup_nvt_deepmd(run_dir: Path, data_file: str | Path, pot_file: str | Path,
                      elements: list[str], T_K: int, n_steps: int = 1_000_000) -> Path:
    """Create LAMMPS NVT run using DeepMD potential."""
    run_dir.mkdir(parents=True, exist_ok=True)

    inp = LAMMPS_DEEPMD.format(
        data_file=str(data_file),
        pot_file=str(pot_file),
        elements=" ".join(elements),
        T=T_K,
        n_steps=n_steps,
    )
    (run_dir / "in.lammps").write_text(inp)
    write_gpu_sub(run_dir / "sub.sh", f"md_{T_K}K")

    # Symlink potential if not in run_dir
    pot = Path(pot_file)
    if pot.exists() and not (run_dir / pot.name).exists():
        (run_dir / pot.name).symlink_to(pot)

    return run_dir


def setup_nvt_mace(run_dir: Path, data_file: str | Path,
                    model_path: str, elements: list[str],
                    T_K: int, n_steps: int = 1_000_000) -> Path:
    """Create LAMMPS NVT run using MACE potential."""
    run_dir.mkdir(parents=True, exist_ok=True)

    inp = LAMMPS_MACE.format(
        data_file=str(data_file),
        model_path=model_path,
        elements=" ".join(elements),
        T=T_K,
        n_steps=n_steps,
    )
    (run_dir / "in.lammps").write_text(inp)
    write_gpu_sub(run_dir / "sub.sh", f"mace_md_{T_K}K")
    return run_dir


def setup_multi_temperature(project_dir: Path, data_file: str | Path,
                              pot_file: str | Path, elements: list[str],
                              temperatures: list[int],
                              n_steps: int = 1_000_000,
                              subdir: str = "nvt") -> list[Path]:
    """Create one run directory per temperature."""
    dirs = []
    for T in temperatures:
        run_dir = project_dir / subdir / str(T)
        setup_nvt_deepmd(run_dir, data_file, pot_file, elements, T, n_steps)
        dirs.append(run_dir)
    return dirs


def check_md_progress(run_dir: Path) -> dict:
    """Parse LAMMPS log to get current step and estimated completion."""
    log = run_dir / "log.lammps"
    dump = run_dir / "dump_unwrapped.lmp"

    status = {"status": "not_started"}
    if not log.exists():
        return status

    # Read last few lines for step count
    try:
        lines = log.read_text().splitlines()
        current_step = 0
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                current_step = int(parts[0])
                break

        # Get total steps from in.lammps
        total_steps = 0
        inlmp = run_dir / "in.lammps"
        if inlmp.exists():
            for l in inlmp.read_text().splitlines():
                if l.strip().startswith("run "):
                    total_steps = int(l.split()[-1])
                    break

        pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        status = {
            "status": "complete" if current_step >= total_steps and total_steps > 0 else "running",
            "current_step": current_step,
            "total_steps": total_steps,
            "pct_complete": round(pct, 1),
        }
        if dump.exists():
            status["dump_size_MB"] = round(dump.stat().st_size / 1e6, 1)
    except Exception as e:
        status["error"] = str(e)

    return status


# ── Stage runner ──────────────────────────────────────────────────────────────

def run(project, output_base: Path = None, mlip: str = "deepmd",
        temperatures: list = None, n_steps: int = 1_000_000,
        submit: bool = False, **kwargs) -> dict:
    """
    Stage 03 entry point.

    task options:
      'mlmd_workflow' / 'mlmd' — NPT 300K → NVT multi-T under project/mlmd/
      'cmd_workflow'  / 'cmd'  — NPT 300K → NVT multi-T under project/cmd/
      'all_workflows'          — run both mlmd and cmd workflows

    mode='interactive': small atom counts and step counts for scaffold testing.
    mode='daemon':      production limits.
    """
    proj_dir = Path(project.root)
    mode     = kwargs.get("mode", "daemon")
    task     = kwargs.get("task", "")
    results  = {"mlip": mlip, "status": "prepared", "runs": []}

    elems    = kwargs.get("elements",
                           list(dict.fromkeys([project.mobile_ion] +
                                               kwargs.get("other_species", []))))

    # ── MLMD workflow (canonical mlmd/nvt/ layout) ────────────────────────────
    if task in ("mlmd_workflow", "mlmd", "all_workflows", ""):
        pot_path = pot_com_pb(proj_dir)
        res = setup_mlmd_workflow(
            proj_dir,
            data_file=kwargs.get("data_file", "mlmd/placeholder.data"),
            pot_file=str(pot_path) if pot_path.exists() else "",
            elements=elems,
            temperatures=temperatures or MLMD_TEMPERATURES,
            mode=mode, mlip=mlip,
            model_path=kwargs.get("model_path", ""),
            natoms=kwargs.get("natoms"),
        )
        results["mlmd_npt"]  = str(res["npt"])
        results["mlmd_nvt"]  = [str(d) for d in res["nvt"]]
        results["runs"].extend([str(d) for d in res["nvt"]])
        if submit:
            for d in res["nvt"]:
                r = subprocess.run(["sbatch", "sub.sh"], capture_output=True,
                                    text=True, cwd=str(d))
                results.setdefault("slurm", []).append(r.stdout.strip())

    # ── CMD workflow (canonical cmd/nvt/ layout) ──────────────────────────────
    if task in ("cmd_workflow", "cmd", "all_workflows"):
        res_cmd = setup_cmd_workflow(
            proj_dir,
            data_file=kwargs.get("cmd_data_file", "cmd/placeholder.data"),
            elements=elems,
            temperatures=temperatures or MLMD_TEMPERATURES,
            mode=mode,
            natoms=kwargs.get("cmd_natoms"),
        )
        results["cmd_npt"]  = str(res_cmd["npt"])
        results["cmd_nvt"]  = [str(d) for d in res_cmd["nvt"]]
        results["runs"].extend([str(d) for d in res_cmd["nvt"]])
        if submit:
            for d in res_cmd["nvt"]:
                r = subprocess.run(["sbatch", "sub.sh"], capture_output=True,
                                    text=True, cwd=str(d))
                results.setdefault("slurm", []).append(r.stdout.strip())

    results["status"] = "submitted" if submit else "prepared"

    # Collect progress for existing MLMD NVT runs (canonical mlmd/nvt/ layout)
    progress = {}
    nvt_base = mlmd_nvt(proj_dir, 0).parent  # project_dir/mlmd/nvt/
    if nvt_base.exists():
        for sub in nvt_base.iterdir():
            if sub.is_dir() and sub.name.isdigit():
                progress[int(sub.name)] = check_md_progress(sub)
    results["progress"] = progress

    return results
