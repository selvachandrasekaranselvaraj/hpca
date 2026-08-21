"""
Stage 03 — LAMMPS MD submission.

Supports: DeepMD NVT, MACE-MP0 NVT, UMA NVT, classical force-field MD.
Generates in.lammps and sub.sh with automatically calculated walltimes and
memory-aware GPU/task configurations.

Partition rules
---------------
MLMD (DeepMD / MACE):   gpu-h100,  up to 72h
Classical MD (OPLS-AA): gpu-h100 for large (>10k atoms),
                         standard  for small (<10k atoms, <44h)
"""
# Layout: see hpca/core/paths.py
from __future__ import annotations
import subprocess
from pathlib import Path

from hpca.core.paths import (
    mlmd_base, mlmd_mlff, mlmd_npt, mlmd_nvt,
    cmd_base, cmd_npt, cmd_nvt,
    pot_com_pb, load_platform_config,
)
from hpca.core.slurm_submit import module_bundle_lines
from hpca.core.config import account_fallback as _account_fallback

# HPC paths loaded lazily from platform.yaml — cross-ref: hpca/config/platform.yaml
def _hpc(key: str, default: str = "") -> str:
    """Look up an HPC path or setting from platform.yaml."""
    return load_platform_config().get("hpc", {}).get(key, default)

def _account(key: str = "standard") -> str:
    """Return the Slurm account string for the given account tier from platform.yaml."""
    return load_platform_config().get("hpc", {}).get("accounts", {}).get(key) or _account_fallback()

_GPU_MAX_H      = 72    # GPU job hard wall-time limit
_STANDARD_MAX_H = 44    # CPU job cap: leave 4h buffer below 48h wall

DAEMON_MD_LIMITS = {
    "mlmd_natoms":    6_000,
    "mlmd_npt_steps": 100_000,     # 100 ps
    "mlmd_nvt_steps": 1_000_000,   # 1 ns
    "cmd_natoms":     50_000,
    "cmd_npt_steps":  200_000,     # 200 ps at 1 fs
    "cmd_nvt_steps":  2_000_000,   # 2 ns
}
# Default temperature sweep for MLMD/CMD NVT production runs
MLMD_TEMPERATURES = [300, 320, 340, 360, 380, 400, 500, 600]


# ── Walltime helpers ──────────────────────────────────────────────────────────

def _fmth(hours: float) -> str:
    """Float hours → HH:MM:00 walltime string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}:{m:02d}:00"


def mlmd_gpu_config(natoms: int, mlip: str = "deepmd") -> dict:
    """
    Return optimal Slurm GPU configuration for LAMMPS MLMD on Kestrel H100 nodes.

    Memory layout (H100 nodes: 256 GB CPU RAM, 4 × 80 GB H100 GPU):
    ┌──────────────┬───────┬────────┬──────────────┬──────┐
    │ Atoms        │ GPUs  │ MPI    │ CPUs/task    │ RAM  │
    ├──────────────┼───────┼────────┼──────────────┼──────┤
    │ < 3 000      │  4    │   4    │   2          │ 100G │
    │ 3 000–6 000  │  4    │   4    │   4          │ 150G │
    │ 6 000–10 000 │  4    │   2 *  │   8          │ 200G │
    └──────────────┴───────┴────────┴──────────────┴──────┘
    * Fewer MPI tasks → more CPU RAM per rank → avoids neighbour-list OOM.

    MACE is more memory-intensive than DeepMD: reduce GPU count for medium systems.
    """
    if mlip in ("mace", "mace-mp0", "uma"):
        if natoms < 2_000:
            return {"gpus": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "80G"}
        elif natoms < 5_000:
            return {"gpus": 2, "ntasks": 2, "cpus_per_task": 8, "mem": "150G"}
        else:
            return {"gpus": 4, "ntasks": 4, "cpus_per_task": 8, "mem": "250G"}
    else:   # deepmd
        if natoms < 3_000:
            return {"gpus": 4, "ntasks": 4, "cpus_per_task": 2, "mem": "100G"}
        elif natoms < 6_000:
            return {"gpus": 4, "ntasks": 4, "cpus_per_task": 4, "mem": "150G"}
        else:
            # ≥6 000 atoms: drop to 2 MPI tasks → 128 GB RAM / task
            return {"gpus": 4, "ntasks": 2, "cpus_per_task": 8, "mem": "200G"}


def mlmd_walltime(natoms: int, n_steps: int, n_gpus: int = 4) -> str:
    """
    Estimate LAMMPS+DeepMD/MACE walltime on H100 GPUs.

    Empirical throughput (Kestrel H100, 4 GPUs):
      1 000 atoms  → ~4 000 000 steps / h
      5 000 atoms  → ~1 000 000 steps / h
     10 000 atoms  → ~  400 000 steps / h
    Formula:  steps_per_hour ≈ n_gpus × 1e9 / natoms

    For large systems (≥5 000 atoms) or long runs (≥2 M steps), minimum is
    raised to 24 h so jobs aren't killed during slow startup or I/O.
    Returns HH:MM:00 string, capped at 72h.
    """
    sph       = max(n_gpus, 1) * 1e9 / max(natoms, 100)
    hours     = n_steps / sph * 1.5   # 50% safety buffer
    # Conservative minimums: large systems or long runs need more padding
    if natoms >= 5_000 or n_steps >= 2_000_000:
        hours = max(hours, 24.0)
    else:
        hours = max(hours, 8.0)
    return _fmth(min(hours, _GPU_MAX_H))


def cmd_walltime(natoms: int, n_steps: int, n_cpus: int = 104) -> str:
    """
    Estimate LAMMPS classical MD walltime (OPLS-AA/GAFF on CPU or GPU-accelerated).

    Empirical: LAMMPS LJ+Coul/long + Ewald on 104 CPU cores:
      ~4 ns/day at 50 000 atoms  (conservative, accounts for Ewald O(N^1.5) overhead)
    Scales as:  ns_per_day ≈ (n_cpus/104) × 4 × (50 000 / natoms)
    At 1 fs timestep: 1 ns = 1 000 000 steps.

    Large systems (>10 000 atoms) or long runs are given at least 24 h.
    Returns HH:MM:00 string, capped at 72h.
    """
    ns         = n_steps * 1e-6
    ns_per_day = max(0.5, (n_cpus / 104.0) * 4.0 * (50_000.0 / max(natoms, 1_000)))
    hours      = (ns / ns_per_day) * 24 * 1.4   # 40% buffer
    if natoms > 10_000 or n_steps >= 2_000_000:
        hours = max(hours, 24.0)
    else:
        hours = max(hours, 6.0)
    return _fmth(min(hours, _GPU_MAX_H))


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

# NPT equilibration templates (MLMD — DeepMD / MACE)
LAMMPS_DEEPMD_NPT = """\
# NPT equilibration using DeepMD potential — outputs npt_final.data
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   deepmd {pot_file}
pair_coeff   * *

timestep     0.001
thermo       1000
thermo_style custom step temp pe ke etotal press vol density

dump         1 all custom 1000 dump_npt.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all npt temp {T} {T} 0.05 iso 0.0 0.0 0.5
run          {n_steps}

write_data   npt_final.data
write_restart npt_final.restart
"""

LAMMPS_MACE_NPT = """\
# NPT equilibration using MACE potential — outputs npt_final.data
units        metal
atom_style   atomic
boundary     p p p
read_data    {data_file}

pair_style   mace no_domain_decomposition
pair_coeff   * * {model_path} {elements}

timestep     0.001
thermo       1000
thermo_style custom step temp pe ke etotal press vol density

dump         1 all custom 1000 dump_npt.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all npt temp {T} {T} 0.05 iso 0.0 0.0 0.5
run          {n_steps}

write_data   npt_final.data
write_restart npt_final.restart
"""

LAMMPS_DEEPMD_NVT_FROM_NPT = """\
# NVT production using DeepMD — reads NPT-equilibrated structure from ../npt/
units        metal
atom_style   atomic
boundary     p p p
read_data    ../npt/npt_final.data

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

LAMMPS_MACE_NVT_FROM_NPT = """\
# NVT production using MACE — reads NPT-equilibrated structure from ../npt/
units        metal
atom_style   atomic
boundary     p p p
read_data    ../npt/npt_final.data

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

LAMMPS_CMD_NVT_FROM_NPT = """\
# CMD NVT production — reads NPT-equilibrated data from ../npt/
units        real
atom_style   full
boundary     p p p
read_data    ../npt/equilibrated.data

pair_style   lj/cut/coul/long 12.0
pair_modify  tail yes
kspace_style ewald 1.0e-4
# pair_coeff  * * ...

bond_style   harmonic
angle_style  harmonic
dihedral_style  opls
improper_style  cvff
special_bonds   lj/coul 0.0 0.0 0.5

timestep     1.0
thermo       5000
thermo_style custom step temp press pe ke etotal

dump         1 all custom 5000 dump_nvt.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nvt temp {T} {T} 100.0
run          {n_prod}
"""

# Classical force-field MD (OPLS-AA/GAFF — fill pair_coeff from parameter files)
LAMMPS_CMD_NPT = """\
# NPT equilibration — fill in pair_coeff from FF parameter file
units        real
atom_style   full
boundary     p p p
read_data    {data_file}

pair_style   lj/cut/coul/long 12.0
pair_modify  tail yes
kspace_style ewald 1.0e-4
# pair_coeff   * * ...   ← add from OPLS-AA or GAFF parameter file

bond_style   harmonic
angle_style  harmonic
dihedral_style  opls
improper_style  cvff

special_bonds  lj/coul 0.0 0.0 0.5

timestep     1.0
thermo       5000
thermo_style custom step temp press density pe ke etotal

dump         1 all custom 5000 dump_npt.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all npt temp {T} {T} 100.0 iso 1.0 1.0 1000.0
run          {n_equil}

write_data   equilibrated.data
"""

LAMMPS_CMD_NVT = """\
# NVT production (load equilibrated.data from NPT)
units        real
atom_style   full
boundary     p p p
read_data    equilibrated.data

pair_style   lj/cut/coul/long 12.0
pair_modify  tail yes
kspace_style ewald 1.0e-4
# pair_coeff  * * ...

bond_style   harmonic
angle_style  harmonic
dihedral_style  opls
improper_style  cvff
special_bonds   lj/coul 0.0 0.0 0.5

timestep     1.0
thermo       5000
thermo_style custom step temp press pe ke etotal

dump         1 all custom 5000 dump_nvt.lmp id type xu yu zu element
dump_modify  1 element {elements} sort id

fix          1 all nvt temp {T} {T} 100.0
run          {n_prod}
"""


# ── Submission script writers ─────────────────────────────────────────────────

def write_gpu_sub(path: Path, job_name: str,
                   natoms: int = 1_000, n_steps: int = 1_000_000,
                   mlip: str = "deepmd",
                   conda_env: str = "",
                   account: str = "") -> None:
    """
    Write GPU submission script with memory-aware resource selection.

    GPU count and MPI tasks are chosen by mlmd_gpu_config(natoms) to prevent
    per-rank OOM on the neighbour list for large systems.
    Walltime is calculated from mlmd_walltime(natoms, n_steps).
    HPC paths and account read from platform.yaml (hpca/config/platform.yaml).
    No --partition is written; SLURM auto-assigns based on wall time / GPU request.
    """
    cfg    = mlmd_gpu_config(natoms, mlip)
    wt     = mlmd_walltime(natoms, n_steps, cfg["gpus"])
    n_gpus = cfg["gpus"]
    ntasks = cfg["ntasks"]
    cpt    = cfg["cpus_per_task"]
    mem    = cfg["mem"]

    hpc         = load_platform_config().get("hpc", {})
    lammps_bin  = _hpc("lammps_bin", "")
    py_deepmd  = hpc.get("python_deepmd", "")
    deepmd_env = conda_env or hpc.get("deepmd_lammps_gpu_env", "") or (str(Path(py_deepmd).parent.parent) if py_deepmd else "")
    slurm_account   = account   or hpc.get("accounts", {}).get("gpu_h100") or _account_fallback()

    path.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --account={slurm_account}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --gpus={n_gpus}\n"
        f"#SBATCH --ntasks-per-node={ntasks}\n"
        f"#SBATCH --cpus-per-task={cpt}\n"
        f"#SBATCH --time={wt}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --mem={mem}\n"
        "#SBATCH --output=%j.stdout\n"
        "#SBATCH --error=%j.stderr\n"
        "\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md')}"
        f"source activate {deepmd_env}\n"
        f"export PATH={Path(lammps_bin).parent}:$PATH\n"
        f"export LD_LIBRARY_PATH={deepmd_env}/lib:$LD_LIBRARY_PATH\n"
        "export MPICH_GPU_SUPPORT_ENABLED=1\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        f"srun --gpus-per-task=1 {lammps_bin} -k on gpus {n_gpus} -sf kk -in in.lammps\n"
    )
    path.chmod(0o755)


def write_cmd_sub(path: Path, job_name: str,
                   natoms: int = 10_000,
                   n_equil: int = 2_000_000, n_prod: int = 2_000_000,
                   T_K: int = 300,
                   account: str = "") -> None:
    """
    Write classical MD submission script.

    ≤10 000 atoms  → CPU LAMMPS (≤44h)
    >10 000 atoms  → GPU LAMMPS package (up to 72h)
    No --partition is written; SLURM auto-assigns based on wall time / GPU request.

    2 ns at 1fs timestep = 2 000 000 steps.
    HPC paths and account read from platform.yaml (hpca/config/platform.yaml).
    """
    hpc           = load_platform_config().get("hpc", {})
    lammps_bin    = hpc.get("lammps_bin", "")
    py_deepmd     = hpc.get("python_deepmd", "")
    deepmd_env    = str(Path(py_deepmd).parent.parent) if py_deepmd else ""
    slurm_account = account or hpc.get("accounts", {}).get("standard") or _account_fallback()

    n_prod_steps  = n_prod
    n_total_steps = n_equil + n_prod_steps

    if natoms > 10_000:
        # GPU-accelerated for large systems
        n_gpus    = 2 if natoms <= 30_000 else 4
        ntasks    = n_gpus
        cpt       = 8
        mem       = "200G"
        wt        = _fmth(min(max(float(cmd_walltime(natoms, n_total_steps).split(":")[0]), 4.0), _GPU_MAX_H))
        run_line  = f"srun --gpus-per-task=1 {lammps_bin} -sf gpu -pk gpu {n_gpus} -in in_npt.lammps\n"
        header = (
            "#!/bin/bash\n"
            f"#SBATCH --account={slurm_account}\n"
            "#SBATCH --nodes=1\n"
            f"#SBATCH --gpus={n_gpus}\n"
            f"#SBATCH --ntasks-per-node={ntasks}\n"
            f"#SBATCH --cpus-per-task={cpt}\n"
            f"#SBATCH --time={wt}\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --mem={mem}\n"
            "#SBATCH --output=%j.stdout\n"
            "#SBATCH --error=%j.stderr\n"
            "\n"
            "module purge\n"
            f"{module_bundle_lines('gpu_md')}"
            f"source activate {deepmd_env}\n"
            f"export PATH={Path(lammps_bin).parent}:$PATH\n"
            f"export LD_LIBRARY_PATH={deepmd_env}/lib:$LD_LIBRARY_PATH\n"
            "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        )
    else:
        # CPU for smaller systems
        n_nodes   = 1
        wt_str    = cmd_walltime(natoms, n_total_steps, n_cpus=104)
        wt_h      = int(wt_str.split(":")[0])
        if wt_h > _STANDARD_MAX_H:
            wt_str = _fmth(_STANDARD_MAX_H)  # cap; user may need to restart
        lammps_cpu = hpc.get("lammps_cpu_bin", lammps_bin)
        run_line  = f"srun {lammps_cpu} -in in_npt.lammps\n"
        header = (
            "#!/bin/bash\n"
            f"#SBATCH --account={slurm_account}\n"
            f"#SBATCH --nodes={n_nodes}\n"
            "#SBATCH --ntasks-per-node=104\n"
            "#SBATCH --cpus-per-task=1\n"
            f"#SBATCH --time={wt_str}\n"
            f"#SBATCH --job-name={job_name}\n"
            "#SBATCH --output=%j.stdout\n"
            "#SBATCH --error=%j.stderr\n"
            "\n"
            "module purge\n"
            f"{module_bundle_lines('cpu_md')}"
            f"source activate {deepmd_env}\n"
            f"export PATH={Path(lammps_cpu).parent}:$PATH\n"
        )

    path.write_text(
        header +
        f"# Step 1: NPT equilibration  ({n_equil:,} steps)\n" +
        run_line.replace("in_npt.lammps", "in_npt.lammps") +
        f"# Step 2: NVT production     ({n_prod_steps:,} steps)\n" +
        run_line.replace("in_npt.lammps", "in_nvt.lammps")
    )
    path.chmod(0o755)


# ── Setup functions ───────────────────────────────────────────────────────────

def setup_nvt_deepmd(run_dir: Path, data_file, pot_file,
                      elements: list, T_K: int,
                      n_steps: int = 1_000_000,
                      natoms: int = 1_000) -> Path:
    """Create LAMMPS NVT run using DeepMD potential."""
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = LAMMPS_DEEPMD.format(
        data_file=str(data_file), pot_file=str(pot_file),
        elements=" ".join(elements), T=T_K, n_steps=n_steps,
    )
    (run_dir / "in.lammps").write_text(inp)
    write_gpu_sub(run_dir / "sub.sh", f"md_{T_K}K",
                   natoms=natoms, n_steps=n_steps, mlip="deepmd")
    pot = Path(pot_file)
    if pot.exists() and not (run_dir / pot.name).exists():
        (run_dir / pot.name).symlink_to(pot)
    return run_dir


def setup_nvt_mace(run_dir: Path, data_file,
                    model_path: str, elements: list,
                    T_K: int, n_steps: int = 1_000_000,
                    natoms: int = 1_000) -> Path:
    """Create LAMMPS NVT run using MACE potential."""
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = LAMMPS_MACE.format(
        data_file=str(data_file), model_path=model_path,
        elements=" ".join(elements), T=T_K, n_steps=n_steps,
    )
    (run_dir / "in.lammps").write_text(inp)
    write_gpu_sub(run_dir / "sub.sh", f"mace_md_{T_K}K",
                   natoms=natoms, n_steps=n_steps, mlip="mace")
    return run_dir


def setup_nvt_cmd(run_dir: Path, data_file,
                   elements: list, T_K: int,
                   n_equil: int = 2_000_000,
                   n_prod:  int = 2_000_000,
                   natoms:  int = 10_000) -> Path:
    """
    Create classical force-field MD run directory (NPT equil → NVT production).

    Writes:
      in_npt.lammps  — NPT equilibration (pair_coeff must be filled in)
      in_nvt.lammps  — NVT production (reads equilibrated.data)
      sub.sh         — Slurm script sized for natoms and 2 ns total
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    elems_str = " ".join(elements)
    (run_dir / "in_npt.lammps").write_text(
        LAMMPS_CMD_NPT.format(
            data_file=str(data_file), elements=elems_str,
            T=T_K, n_equil=n_equil,
        )
    )
    (run_dir / "in_nvt.lammps").write_text(
        LAMMPS_CMD_NVT.format(elements=elems_str, T=T_K, n_prod=n_prod)
    )
    write_cmd_sub(run_dir / "sub.sh",
                   job_name=f"cmd_{T_K}K",
                   natoms=natoms, n_equil=n_equil, n_prod=n_prod, T_K=T_K)
    return run_dir


def setup_multi_temperature(project_dir: Path, data_file, pot_file,
                              elements: list, temperatures: list,
                              n_steps: int = 1_000_000,
                              natoms:  int = 1_000,
                              subdir:  str = "nvt") -> list:
    """Create one DeepMD NVT run directory per temperature."""
    dirs = []
    for T in temperatures:
        run_dir = project_dir / subdir / str(T)
        setup_nvt_deepmd(run_dir, data_file, pot_file, elements, T,
                          n_steps=n_steps, natoms=natoms)
        dirs.append(run_dir)
    return dirs


def setup_multi_temperature_cmd(project_dir: Path, data_file,
                                  elements: list, temperatures: list,
                                  n_equil: int = 2_000_000,
                                  n_prod:  int = 2_000_000,
                                  natoms:  int = 10_000,
                                  subdir:  str = "cmd") -> list:
    """Create one classical MD run directory per temperature."""
    dirs = []
    for T in temperatures:
        run_dir = project_dir / subdir / str(T)
        setup_nvt_cmd(run_dir, data_file, elements, T,
                       n_equil=n_equil, n_prod=n_prod, natoms=natoms)
        dirs.append(run_dir)
    return dirs


# ── New NPT→NVT workflow functions ───────────────────────────────────────────

def setup_mlmd_npt(mlmd_dir: Path, data_file,
                    pot_file="", elements: list = None,
                    T_K: int = 300, n_steps: int = 100_000,
                    natoms: int = 1_000,
                    mlip: str = "deepmd",
                    model_path: str = "") -> Path:
    """
    Create NPT equilibration run in mlmd_dir/npt/.

    Outputs npt_final.data which is read by NVT runs.
    mlip='deepmd' uses DeepMD potential; 'mace'/'mace-mp0' uses MACE.
    """
    npt_dir = mlmd_dir / "npt"
    npt_dir.mkdir(parents=True, exist_ok=True)
    elems = " ".join(elements or ["Li", "Cl"])
    if mlip in ("mace", "mace-mp0"):
        inp = LAMMPS_MACE_NPT.format(
            data_file=str(data_file), model_path=model_path or "model.model",
            elements=elems, T=T_K, n_steps=n_steps)
        write_gpu_sub(npt_dir / "sub.sh", f"mlmd_npt_{T_K}K",
                       natoms=natoms, n_steps=n_steps, mlip="mace")
    else:
        pot = str(pot_file) if pot_file else "mlff/pot_com.pb"
        inp = LAMMPS_DEEPMD_NPT.format(
            data_file=str(data_file), pot_file=pot,
            elements=elems, T=T_K, n_steps=n_steps)
        write_gpu_sub(npt_dir / "sub.sh", f"mlmd_npt_{T_K}K",
                       natoms=natoms, n_steps=n_steps, mlip="deepmd")
        pot_path = Path(pot_file) if pot_file else None
        if pot_path and pot_path.exists() and not (npt_dir / pot_path.name).exists():
            (npt_dir / pot_path.name).symlink_to(pot_path)
    (npt_dir / "in.lammps").write_text(inp)
    return npt_dir


def setup_mlmd_nvt(mlmd_dir: Path, pot_file="",
                    elements: list = None,
                    temperatures: list = None,
                    n_steps: int = 1_000_000,
                    natoms: int = 1_000,
                    mlip: str = "deepmd",
                    model_path: str = "") -> list:
    """
    Create NVT production runs in mlmd_dir/nvt/{T}/.

    Reads ../npt/npt_final.data from the sibling NPT run.
    pot_file should be an absolute path (from pot_com_pb(project_dir)).
    """
    temps = temperatures or MLMD_TEMPERATURES
    elems = " ".join(elements or ["Li", "Cl"])
    dirs  = []
    nvt_base = mlmd_dir / "nvt"
    # Resolve absolute pot path: prefer explicit arg, else mlmd/mlff/pot_com.pb
    default_pot = mlmd_mlff(mlmd_dir.parent) / "pot_com.pb"
    for T in temps:
        run_dir = nvt_base / str(T)
        run_dir.mkdir(parents=True, exist_ok=True)
        if mlip in ("mace", "mace-mp0"):
            mp = model_path or "model.model"
            inp = LAMMPS_MACE_NVT_FROM_NPT.format(
                model_path=mp, elements=elems, T=T, n_steps=n_steps)
            write_gpu_sub(run_dir / "sub.sh", f"mlmd_nvt_{T}K",
                           natoms=natoms, n_steps=n_steps, mlip="mace")
        else:
            # Use absolute path so the script works regardless of cwd
            pot_abs = str(pot_file) if pot_file else str(default_pot)
            inp = LAMMPS_DEEPMD_NVT_FROM_NPT.format(
                pot_file=pot_abs, elements=elems, T=T, n_steps=n_steps)
            write_gpu_sub(run_dir / "sub.sh", f"mlmd_nvt_{T}K",
                           natoms=natoms, n_steps=n_steps, mlip="deepmd")
            pot_path = Path(pot_file) if pot_file else None
            if pot_path and pot_path.exists() and not (run_dir / pot_path.name).exists():
                (run_dir / pot_path.name).symlink_to(pot_path)
        (run_dir / "in.lammps").write_text(inp)
        dirs.append(run_dir)
    return dirs


def setup_mlmd_workflow(project_dir: Path, data_file="data.lammps",
                          pot_file="", elements: list = None,
                          temperatures: list = None,
                          mlip: str = "deepmd",
                          model_path: str = "",
                          natoms: int = None) -> dict:
    """
    Full MLMD workflow: mlmd/mlff/ + NPT 300K → NVT at multiple temperatures.

    project_dir/mlmd/mlff/   — MLFF training & potential (created empty for new workflow)
    project_dir/mlmd/npt/    — NPT equilibration at 300 K
    project_dir/mlmd/nvt/{T}/ — NVT production at each temperature
    """
    n    = natoms or DAEMON_MD_LIMITS["mlmd_natoms"]

    # Use canonical paths (see hpca/core/paths.py)
    mlff_dir = mlmd_mlff(project_dir)
    mlff_dir.mkdir(parents=True, exist_ok=True)
    mlmd_dir = mlff_dir.parent  # project_dir/mlmd/

    # Resolve pot_file: prefer explicit, else canonical pot_com_pb path
    resolved_pot = pot_file or str(pot_com_pb(project_dir))

    npt_dir  = setup_mlmd_npt(mlmd_dir, data_file, resolved_pot,
                                elements=elements, T_K=300,
                                n_steps=DAEMON_MD_LIMITS["mlmd_npt_steps"],
                                natoms=n, mlip=mlip, model_path=model_path)
    nvt_dirs = setup_mlmd_nvt(mlmd_dir, resolved_pot, elements,
                                temperatures=temperatures or MLMD_TEMPERATURES,
                                n_steps=DAEMON_MD_LIMITS["mlmd_nvt_steps"],
                                natoms=n, mlip=mlip, model_path=model_path)
    return {"mlmd_dir": mlmd_dir, "npt": npt_dir, "nvt": nvt_dirs,
            "mlff": mlff_dir}


def setup_cmd_workflow(project_dir: Path, data_file="cmd_structure.data",
                         elements: list = None,
                         temperatures: list = None,
                         natoms: int = None) -> dict:
    """
    Full CMD workflow: cmd/npt/ (300 K NPT) → cmd/nvt/{T}/ (NVT production).

    project_dir/cmd/npt/    — NPT equilibration at 300 K
    project_dir/cmd/nvt/{T}/ — NVT production at each temperature
    """
    n     = natoms or DAEMON_MD_LIMITS["cmd_natoms"]
    temps = temperatures or MLMD_TEMPERATURES
    elems_str = " ".join(elements or ["Li", "Cl"])

    # Use canonical paths (see hpca/core/paths.py)
    npt_dir_path = cmd_npt(project_dir)
    npt_dir_path.mkdir(parents=True, exist_ok=True)

    (npt_dir_path / "in.lammps").write_text(
        LAMMPS_CMD_NPT.format(
            data_file=str(data_file), elements=elems_str,
            T=300, n_equil=DAEMON_MD_LIMITS["cmd_npt_steps"]))
    write_cmd_sub(npt_dir_path / "sub.sh", "cmd_npt_300K",
                   natoms=n, n_equil=DAEMON_MD_LIMITS["cmd_npt_steps"], n_prod=0, T_K=300)

    nvt_dirs = []
    for T in temps:
        run_dir = cmd_nvt(project_dir, T)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "in.lammps").write_text(
            LAMMPS_CMD_NVT_FROM_NPT.format(elements=elems_str, T=T,
                                             n_prod=DAEMON_MD_LIMITS["cmd_nvt_steps"]))
        write_cmd_sub(run_dir / "sub.sh", f"cmd_nvt_{T}K",
                       natoms=n, n_equil=0, n_prod=DAEMON_MD_LIMITS["cmd_nvt_steps"], T_K=T)
        nvt_dirs.append(run_dir)

    return {"cmd_dir": cmd_base(project_dir), "npt": npt_dir_path, "nvt": nvt_dirs}


# ── Progress checker ──────────────────────────────────────────────────────────

def check_md_progress(run_dir: Path) -> dict:
    """Parse LAMMPS log to get current step and estimated completion."""
    log  = run_dir / "log.lammps"
    dump = run_dir / "dump_unwrapped.lmp"
    if not log.exists():
        return {"status": "not_started"}
    try:
        lines         = log.read_text().splitlines()
        current_step  = 0
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                current_step = int(parts[0]); break
        total_steps = 0
        inlmp = run_dir / "in.lammps"
        if inlmp.exists():
            for l in inlmp.read_text().splitlines():
                if l.strip().startswith("run "):
                    total_steps = int(l.split()[-1]); break
        pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        status = {
            "status":        "complete" if current_step >= total_steps > 0 else "running",
            "current_step":  current_step,
            "total_steps":   total_steps,
            "pct_complete":  round(pct, 1),
        }
        if dump.exists():
            status["dump_size_MB"] = round(dump.stat().st_size / 1e6, 1)
    except Exception as e:
        status = {"status": "error", "error": str(e)}
    return status


# ── Stage runner ──────────────────────────────────────────────────────────────

def run(project, output_base: Path = None, mlip: str = "deepmd",
        temperatures: list = None, n_steps: int = 1_000_000,
        natoms: int = None, submit: bool = False, **kwargs) -> dict:
    """
    Stage 03 entry point.

    New workflow (when designed_structures/ exists):
      mlmd_workflow — NPT 300K → NVT multi-T under project/mlmd/
      cmd_workflow  — NPT 300K → NVT multi-T under project/cmd/
    Legacy workflow still supported for existing projects.
    mode='interactive' uses small step counts for scaffold testing.
    """
    proj_dir = Path(project.root)
    results  = {"mlip": mlip, "status": "prepared", "runs": []}

    # New workflow shortcut
    task = kwargs.get("task", "")
    if task in ("mlmd_workflow", "mlmd"):
        elems = kwargs.get("elements",
                            [project.mobile_ion] + kwargs.get("other_species", []))
        res = setup_mlmd_workflow(
            proj_dir,
            data_file=kwargs.get("data_file", "mlmd/placeholder.data"),
            pot_file=kwargs.get("pot_file",
                                  str(pot_com_pb(proj_dir))),
            elements=elems,
            temperatures=temperatures or MLMD_TEMPERATURES,
            mlip=mlip,
            model_path=kwargs.get("model_path", ""),
            natoms=natoms,
        )
        results["runs"] = [str(d) for d in res["nvt"]]
        results["mlmd_dir"] = str(res["mlmd_dir"])
        if submit:
            import subprocess
            for d in res["nvt"]:
                subprocess.run(["sbatch", "sub.sh"], cwd=str(d), capture_output=True)
            results["status"] = "submitted"
        return results

    if task in ("cmd_workflow", "cmd"):
        elems = kwargs.get("elements",
                            [project.mobile_ion] + kwargs.get("other_species", []))
        res = setup_cmd_workflow(
            proj_dir,
            data_file=kwargs.get("data_file", "cmd/placeholder.data"),
            elements=elems,
            temperatures=temperatures or MLMD_TEMPERATURES,
            mode=mode,
            natoms=natoms,
        )
        results["runs"] = [str(d) for d in res["nvt"]]
        results["cmd_dir"] = str(res["cmd_dir"])
        if submit:
            import subprocess
            for d in res["nvt"]:
                subprocess.run(["sbatch", "sub.sh"], cwd=str(d), capture_output=True)
            results["status"] = "submitted"
        return results

    temps  = temperatures or list(project.mlmd_temperatures) or [300, 400, 600, 800]
    pot    = proj_dir / project.param("deepmd_pot", "mlff/pot_com.pb")
    data   = kwargs.get("data_file", proj_dir / "Data" / "data.lammps")
    elems  = kwargs.get("elements",
                         list(dict.fromkeys([project.mobile_ion] +
                                             kwargs.get("other_species", []))))
    n      = natoms or kwargs.get("natoms", 1_000)

    if mlip == "deepmd" and pot.exists():
        dirs = setup_multi_temperature(proj_dir, data, pot, elems, temps,
                                        n_steps=n_steps, natoms=n)
        results["runs"] = [str(d) for d in dirs]
        if submit:
            for d in dirs:
                r = subprocess.run(["sbatch", "sub.sh"], capture_output=True,
                                    text=True, cwd=str(d))
                results.setdefault("slurm", []).append(r.stdout.strip())
            results["status"] = "submitted"

    elif mlip in ("mace-mp0", "mace"):
        model = kwargs.get("model_path", "")
        for T in temps:
            d = setup_nvt_mace(proj_dir / "mlmd" / "nvt_mace" / str(T),
                                data, model, elems, T, n_steps, natoms=n)
            results["runs"].append(str(d))
            if submit:
                r = subprocess.run(["sbatch", "sub.sh"], capture_output=True,
                                    text=True, cwd=str(d))
                results.setdefault("slurm", []).append(r.stdout.strip())
        if submit:
            results["status"] = "submitted"

    elif mlip == "classical":
        n_equil = kwargs.get("n_equil", 2_000_000)
        n_prod  = kwargs.get("n_prod",  2_000_000)
        dirs = setup_multi_temperature_cmd(
            proj_dir, data, elems, temps,
            n_equil=n_equil, n_prod=n_prod, natoms=n,
        )
        results["runs"] = [str(d) for d in dirs]
        if submit:
            for d in dirs:
                r = subprocess.run(["sbatch", "sub.sh"], capture_output=True,
                                    text=True, cwd=str(d))
                results.setdefault("slurm", []).append(r.stdout.strip())
            results["status"] = "submitted"

    # Collect progress for existing MLMD NVT runs (canonical mlmd/nvt/ layout)
    progress = {}
    nvt_base = mlmd_base(proj_dir) / "nvt"  # project_dir/mlmd/nvt/
    if nvt_base.exists():
        for run_dir in nvt_base.iterdir():
            if run_dir.is_dir() and run_dir.name.isdigit():
                progress[int(run_dir.name)] = check_md_progress(run_dir)
    results["progress"] = progress
    return results
