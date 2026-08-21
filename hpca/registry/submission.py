"""Canonical SLURM submission-script registry.

All HPC platform values (modules, binary paths, conda envs, accounts, GPU tier
sizing) are read from platform.yaml. No hardcoded paths or module names in Python.

Usage
-----
    from hpca.registry.submission import write_submission

    write_submission(calc_dir / "sub.sh", "vasp", "opt_Si",
                     nodes=1, ntasks=96, time_key="dft_opt")

    write_submission(box_dir / "sub.sh", "vasp_aimd", f"ds_{name}",
                     natoms=64, time_key="aimd_dataset")

    write_submission(mlff_dir / "sub.sh", "deepmd_cpu", f"{proj}_mlip",
                     mlff_dir=mlff_dir)

Key (string)              Description
----------------------------------------------------------------------------------
vasp                      Standard VASP — explicit nodes/ntasks/time
vasp_aimd                 Atom-count scaled VASP (vasp_nodes tiers in platform.yaml)
vasp_batch                Multiple VASP boxes in parallel on one node (background sruns)
vasp_neb_endpoints        Two-node parallel NEB endpoint relaxations
vasp_neb_images           One-node parallel NEB image batch
vasp_ncore_phase1         NCORE benchmark: sequential ntasks sweep (writes best_ntasks.txt)
vasp_ncore_phase2         NCORE benchmark: parallel 6×16 test   (writes par_time.txt)
vasp_ncore_finalize       NCORE finalize: pick winner → ncore_best.txt + ncore_parallel.txt
lammps_gpu                GPU LAMMPS (model-size-based GPU tier, KOKKOS)
lammps_cpu                CPU LAMMPS (DeepMD-CPU venv + mpirun)
deepmd_cpu                DeepMD CPU train → freeze → compress
deepmd_al                 Active-learning DeepMD retrain → test
mace_gpu                  MACE GPU fine-tuning
analysis_cpu              h06_analysis: one variant's transport/structure analysis
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hpca.core.paths import load_platform_config
from hpca.core.slurm_submit import module_bundle_lines
from hpca.core.config import account_fallback as _account_fallback


# ── Platform config helpers ───────────────────────────────────────────────────

def _cfg() -> dict:
    """Return the raw platform.yaml dict."""
    return load_platform_config()

def _hpc(key: str, default: str = "") -> str:
    """Return a value from the ``hpc:`` section of platform.yaml."""
    return _cfg().get("hpc", {}).get(key, default)

def _account(gpu: bool = False) -> str:
    """Return the SLURM account for GPU (H100) or standard CPU jobs."""
    accts = _cfg().get("hpc", {}).get("accounts", {})
    return accts.get("gpu_h100" if gpu else "standard") or _account_fallback()

def _slurm_time(key: str, default: str = "48:00:00") -> str:
    """Return the SLURM walltime string for a named stage key."""
    return _cfg().get("slurm_time", {}).get(key, default)

def _vasp_module() -> str:
    """Return the VASP environment module string from platform.yaml."""
    return _hpc("vasp_module", "vasp/6.4.2_openMP")

def _lammps_bin() -> str:
    """Return the GPU LAMMPS binary path from platform.yaml."""
    return _hpc("lammps_bin", "lmp")

def _gpu_conda_env() -> str:
    """Return the conda environment root for GPU LAMMPS/DeepMD runs."""
    py = _hpc("python_deepmd")
    if py:
        return str(Path(py).parent.parent)
    return _hpc("deepmd_lammps_gpu_env", "")

def _cpu_venv() -> str:
    """Return the DeepMD CPU virtualenv path from platform.yaml."""
    return _hpc("deepmd_cpu_venv", "")

def _lammps_cpu_venv() -> str:
    """Return the CPU LAMMPS/DeepMD 2023 virtualenv path from platform.yaml."""
    return _hpc("deepmd_lammps_venv_2023", "")

def _lammps_cpu_bin() -> str:
    """Return the CPU LAMMPS binary path from platform.yaml."""
    return _hpc("lammps_cpu_bin", "lmp")

def _mpirun() -> str:
    """Return the mpirun binary path from platform.yaml."""
    return _hpc("mpirun_bin", "mpirun")

def _vasp_node_tier(natoms: int) -> tuple[int, int]:
    """Return (nodes, ntasks) for the appropriate VASP node tier given *natoms*."""
    vn = _cfg().get("vasp_nodes", {})
    small_t  = vn.get("small_atoms",  50)
    medium_t = vn.get("medium_atoms", 100)
    if natoms < small_t:
        n, t = vn.get("small",  [1, 64])
    elif natoms <= medium_t:
        n, t = vn.get("medium", [1, 96])
    else:
        n, t = vn.get("large",  [1, 96])
    return int(n), int(t)

def _resolve_time(time_key: str | None, time: str | None,
                  default: str = "48:00:00") -> str:
    """Return the best available walltime: explicit *time* > slurm_time key > *default*."""
    if time:
        return time
    if time_key:
        return _slurm_time(time_key, default)
    return default


# ── Shared VASP script header ─────────────────────────────────────────────────

def _vasp_header(job_name: str, nodes: int, ntasks: int,
                 wall: str, exclusive: bool = False) -> str:
    """Return SLURM header + module-load lines shared by all VASP script builders."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=0",
        f"#SBATCH --time={wall}",
        f"#SBATCH --account={_account()}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --error=%J.stderr",
        "#SBATCH --output=%J.stdout",
    ]
    if exclusive:
        lines.append("#SBATCH --exclusive")
    lines += [
        "ulimit -s unlimited",
        "module purge",
        f"module load {_vasp_module()}",
    ]
    return "\n".join(lines)


# ── Builder functions (return script content as str) ─────────────────────────

def _build_vasp(job_name: str, *, nodes: int = 1, ntasks: int = 96,
                time_key: str | None = None, time: str | None = None,
                exclusive: bool = False, **_) -> str:
    """Build a standard single-calculation VASP submission script."""
    wall = _resolve_time(time_key, time, "48:00:00")
    return _vasp_header(job_name, nodes, ntasks, wall, exclusive) + "\nsrun vasp_std &> out\n"


def _build_vasp_aimd(job_name: str, *, natoms: int,
                     time_key: str | None = None, time: str | None = None,
                     **_) -> str:
    """Build a VASP AIMD submission script with atom-count-scaled node/task selection."""
    nodes, ntasks = _vasp_node_tier(natoms)
    wall = _resolve_time(time_key, time, "48:00:00")
    return _vasp_header(job_name, nodes, ntasks, wall) + "\nsrun vasp_std &> out\n"


def _build_vasp_batch(job_name: str, *, box_dirs: list,
                      ntasks_per_job: int = 16,
                      time_key: str | None = None, time: str | None = None,
                      **_) -> str:
    """Multiple VASP AIMD boxes in parallel on one node."""
    ntotal = len(box_dirs) * ntasks_per_job
    wall   = _resolve_time(time_key, time, "48:00:00")
    lines  = [
        "#!/bin/bash",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={ntotal}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=0",
        f"#SBATCH --time={wall}",
        f"#SBATCH --account={_account()}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --error=%J.stderr",
        "#SBATCH --output=%J.stdout",
        "ulimit -s unlimited",
        "module purge",
        f"module load {_vasp_module()}",
        "",
    ]
    for d in box_dirs:
        lines.append(f"( cd '{d}' && srun --ntasks={ntasks_per_job} vasp_std &> out ) &")
    lines += ["wait", ""]
    return "\n".join(lines)


def _build_vasp_neb_endpoints(job_name: str, *, endpoint_tags: list[str],
                               cores_per_node: int | None = None,
                               time_key: str | None = None, time: str | None = None,
                               **_) -> str:
    """Build a two-node VASP script that relaxes NEB endpoint images in parallel."""
    ntasks  = cores_per_node or int(_cfg().get("vasp_nodes", {}).get("medium", [1, 96])[1])
    wall    = _resolve_time(time_key, time, _slurm_time("neb_endpoint", "48:00:00"))
    t0, t1  = endpoint_tags[0], endpoint_tags[1]
    mod     = _vasp_module()
    return (
        "#!/bin/bash\n"
        "#SBATCH --nodes=2\n"
        f"#SBATCH --ntasks-per-node={ntasks}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --account={_account()}\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --error=job_%j.err\n"
        "#SBATCH --output=job_%j.out\n"
        "\n"
        "unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK "
        "SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU\n"
        "module purge\n"
        f"module load {mod}\n"
        "export OMP_NUM_THREADS=1\n"
        "ulimit -s unlimited\n"
        "\n"
        f"( cd {t0} ; srun --nodes=1 --ntasks-per-node={ntasks} vasp_std > out 2>&1 ) &\n"
        f"( cd {t1} ; srun --nodes=1 --ntasks-per-node={ntasks} vasp_std > out 2>&1 ) &\n"
        "wait\n"
        'echo "Endpoint optimisations complete."\n'
    )


def _build_vasp_neb_images(job_name: str, *, image_tags: list[str],
                            cores_per_image: int = 16,
                            cores_per_node: int | None = None,
                            time_key: str | None = None, time: str | None = None,
                            **_) -> str:
    """Build a one-node VASP script that runs NEB image relaxations in parallel."""
    ntasks   = cores_per_node or int(_cfg().get("vasp_nodes", {}).get("medium", [1, 96])[1])
    wall     = _resolve_time(time_key, time, _slurm_time("neb_image", "150:00:00"))
    tags_str = " ".join(image_tags)
    mod      = _vasp_module()
    return (
        "#!/bin/bash\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node={ntasks}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --account={_account()}\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --error=job_%j.err\n"
        "#SBATCH --output=job_%j.out\n"
        "\n"
        "unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK "
        "SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU\n"
        "module purge\n"
        f"module load {mod}\n"
        "export OMP_NUM_THREADS=1\n"
        "ulimit -s unlimited\n"
        "\n"
        f"for tag in {tags_str}\n"
        "do\n"
        '    if [ -d "$tag" ]; then\n'
        '        echo "Starting image $tag"\n'
        f'        ( cd "$tag" ; srun --ntasks={cores_per_image} --exclusive vasp_std > out 2>&1 ) &\n'
        "    fi\n"
        "done\n"
        "wait\n"
        f'echo "Images {tags_str} complete."\n'
    )


def _build_submit_fanout(job_name: str, *, scripts: list[str], **_) -> str:
    """Build a local launcher that submits registered child scripts.

    This is intentionally run with ``bash``, not submitted as a SLURM job.
    """
    if not scripts:
        raise ValueError("submit_fanout requires at least one child script")
    lines = ["#!/bin/bash", "set -euo pipefail", f"# Submission fan-out: {job_name}"]
    lines.extend(f"sbatch {shlex.quote(str(script))}" for script in scripts)
    lines.append('echo "Submitted all child jobs."')
    return "\n".join(lines) + "\n"


def _build_vasp_ncore_phase1(job_name: str, *, work_dir: str,
                              poscar: str, potcar: str,
                              kpoints_mesh: tuple = (2, 2, 2),
                              candidates: list | None = None, **_) -> str:
    """Sequential ntasks benchmark: finds fastest VASP ntasks configuration."""
    vasp_mod   = _vasp_module()
    account    = _account()
    km         = kpoints_mesh
    cand_str   = " ".join(str(n) for n in (candidates or [96, 80, 72, 64, 48, 32, 16]))
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node=96\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --mem=0\n"
        f"#SBATCH --time=10:00:00\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --job-name={job_name}_p1\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J_p1.stdout\n"
        f"ulimit -s unlimited\n"
        f"module purge\n"
        f"module load {vasp_mod}\n"
        f"\n"
        f"export OMP_NUM_THREADS=1\n"
        f"WORK='{work_dir}'\n"
        f"best_ntasks=96\n"
        f"best_time=999999\n"
        f"\n"
        f"for ntasks in {cand_str}; do\n"
        f"    if [ \"$ntasks\" -ge 40 ]; then ncore=8; else ncore=4; fi\n"
        f"    dir=\"$WORK/ntasks_${{ntasks}}\"\n"
        f"    mkdir -p \"$dir\"\n"
        f"    cp '{poscar}' \"$dir/POSCAR\"\n"
        f"    cp '{potcar}' \"$dir/POTCAR\"\n"
        f"    cat > \"$dir/KPOINTS\" << 'KPTS'\n"
        f"Automatic mesh\n"
        f"0\n"
        f"MP\n"
        f"  {km[0]}  {km[1]}  {km[2]}\n"
        f"  0   0   0\n"
        f"KPTS\n"
        f"    cat > \"$dir/INCAR\" << INCAR_EOF\n"
        f"SYSTEM     = ncore_opt\n"
        f"PREC       = Normal\n"
        f"ENCUT      = 400\n"
        f"EDIFF      = 1E-3\n"
        f"NSW        = 0\n"
        f"IBRION     = -1\n"
        f"ISIF       = 2\n"
        f"ISYM       = 0\n"
        f"LWAVE      = .FALSE.\n"
        f"LCHARG     = .FALSE.\n"
        f"LREAL      = Auto\n"
        f"ISMEAR     = 0\n"
        f"SIGMA      = 0.05\n"
        f"NELM       = 20\n"
        f"NCORE      = ${{ncore}}\n"
        f"INCAR_EOF\n"
        f"    cd \"$dir\"\n"
        f"    t_start=$(date +%s)\n"
        f"    srun -n ${{ntasks}} vasp_std > out 2>&1\n"
        f"    t_end=$(date +%s)\n"
        f"    elapsed=$((t_end - t_start))\n"
        f"    echo \"ntasks=${{ntasks}} NCORE=${{ncore}} TIME=${{elapsed}}s\" | tee -a \"$WORK/timings.txt\"\n"
        f"    if [ \"$elapsed\" -lt \"$best_time\" ]; then\n"
        f"        best_time=$elapsed\n"
        f"        best_ntasks=$ntasks\n"
        f"    fi\n"
        f"    cd \"$WORK\"\n"
        f"done\n"
        f"\n"
        f"echo \"$best_ntasks\" > \"$WORK/best_ntasks.txt\"\n"
        f"echo \"$best_time\"   > \"$WORK/best_time.txt\"\n"
        f"echo \"Phase1 winner: ${{best_ntasks}}t (${{best_time}}s)\" >> \"$WORK/timings.txt\"\n"
    )


def _build_vasp_ncore_phase2(job_name: str, *, work_dir: str,
                              poscar: str, potcar: str,
                              kpoints_mesh: tuple = (2, 2, 2), **_) -> str:
    """Parallel 6×16-core test: checks if batching outperforms sequential."""
    vasp_mod = _vasp_module()
    account  = _account()
    km       = kpoints_mesh
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node=96\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time=10:00:00\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --job-name={job_name}_p2\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J_p2.stdout\n"
        f"ulimit -s unlimited\n"
        f"module purge\n"
        f"module load {vasp_mod}\n"
        f"\n"
        f"WORK='{work_dir}'\n"
        f"par_dir=\"$WORK/parallel_6x16\"\n"
        f"mkdir -p \"$par_dir\"\n"
        f"export OMP_NUM_THREADS=1\n"
        f"\n"
        f"for i in $(seq 1 6); do\n"
        f"    pdir=\"$par_dir/run_${{i}}\"\n"
        f"    mkdir -p \"$pdir\"\n"
        f"    cp '{poscar}' \"$pdir/POSCAR\"\n"
        f"    cp '{potcar}' \"$pdir/POTCAR\"\n"
        f"    cat > \"$pdir/KPOINTS\" << 'KPTS'\n"
        f"Automatic mesh\n"
        f"0\n"
        f"MP\n"
        f"  {km[0]}  {km[1]}  {km[2]}\n"
        f"  0   0   0\n"
        f"KPTS\n"
        f"    cat > \"$pdir/INCAR\" << 'INCAR_PAR'\n"
        f"SYSTEM     = ncore_opt_par\n"
        f"PREC       = Normal\n"
        f"ENCUT      = 400\n"
        f"EDIFF      = 1E-3\n"
        f"NSW        = 0\n"
        f"IBRION     = -1\n"
        f"ISIF       = 2\n"
        f"ISYM       = 0\n"
        f"LWAVE      = .FALSE.\n"
        f"LCHARG     = .FALSE.\n"
        f"LREAL      = Auto\n"
        f"ISMEAR     = 0\n"
        f"SIGMA      = 0.05\n"
        f"NELM       = 20\n"
        f"NCORE      = 4\n"
        f"INCAR_PAR\n"
        f"done\n"
        f"\n"
        f"par_t_start=$(date +%s)\n"
        f"for i in $(seq 1 6); do\n"
        f"    pdir=\"$par_dir/run_${{i}}\"\n"
        f"    (cd \"$pdir\" && srun --ntasks=16 --exclusive vasp_std > out 2>&1) &\n"
        f"done\n"
        f"wait\n"
        f"par_t_end=$(date +%s)\n"
        f"par_elapsed=$((par_t_end - par_t_start))\n"
        f"par_ok=0\n"
        f"for i in $(seq 1 6); do\n"
        f"    [ -s \"$par_dir/run_${{i}}/out\" ] && par_ok=$((par_ok + 1))\n"
        f"done\n"
        f"echo \"$par_elapsed\" > \"$WORK/par_time.txt\"\n"
        f"echo \"$par_ok\"      > \"$WORK/par_ok.txt\"\n"
        f"echo \"parallel_6x16 TIME=${{par_elapsed}}s OK=${{par_ok}}/6\" | tee -a \"$WORK/timings.txt\"\n"
    )


def _build_vasp_ncore_finalize(job_name: str, *, work_dir: str, **_) -> str:
    """Compare phase1+phase2 → write ncore_best.txt and ncore_parallel.txt."""
    account = _account()
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node=1\n"
        f"#SBATCH --time=00:05:00\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --job-name={job_name}_fin\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J_fin.stdout\n"
        f"\n"
        f"WORK='{work_dir}'\n"
        f"\n"
        f"best_ntasks=$(cat \"$WORK/best_ntasks.txt\" 2>/dev/null || echo 96)\n"
        f"best_time=$(cat   \"$WORK/best_time.txt\"   2>/dev/null || echo 999999)\n"
        f"par_elapsed=$(cat \"$WORK/par_time.txt\" 2>/dev/null || echo 999999)\n"
        f"par_ok=$(cat      \"$WORK/par_ok.txt\"   2>/dev/null || echo 0)\n"
        f"\n"
        f"threshold=$((best_time * 6))\n"
        f"if [ \"$par_ok\" -eq 6 ] && [ \"$par_elapsed\" -lt \"$threshold\" ]; then\n"
        f"    echo \"16\" > \"$WORK/ncore_best.txt\"\n"
        f"    echo \"6\"  > \"$WORK/ncore_parallel.txt\"\n"
        f"    echo \"Winner: parallel 6x16 (${{par_elapsed}}s < threshold ${{threshold}}s)\" >> \"$WORK/timings.txt\"\n"
        f"else\n"
        f"    echo \"$best_ntasks\" > \"$WORK/ncore_best.txt\"\n"
        f"    echo \"1\"            > \"$WORK/ncore_parallel.txt\"\n"
        f"    echo \"Winner: sequential ${{best_ntasks}}t (${{best_time}}s)\" >> \"$WORK/timings.txt\"\n"
        f"fi\n"
    )


def _build_lammps_gpu(job_name: str, *, pot_path=None, n_gpus: int | None = None,
                      ntasks: int | None = None, cpus_per_task: int | None = None,
                      mem_gb: int | None = None,
                      time_key: str | None = None, time: str | None = None,
                      **_) -> str:
    """Build a GPU LAMMPS (KOKKOS) submission script with model-size-based GPU tier."""
    from hpca.core.gpu_sizing import get_lammps_gpu_resources
    sizing   = get_lammps_gpu_resources(Path(pot_path) if pot_path else Path("/nonexistent"))
    ngpus    = n_gpus        or sizing["gpus"]
    nt       = ntasks        or sizing["ntasks"]
    cpus     = cpus_per_task or sizing["cpus_per_task"]
    mem      = mem_gb        or sizing["mem_gb"]
    wall     = _resolve_time(time_key, time, "72:00:00")
    lmp_bin  = _lammps_bin()
    lmp_dir  = lmp_bin.rpartition("/")[0]
    lammps_src = f"{lmp_dir}/lammps/src" if lmp_dir else ""
    gpu_env  = _gpu_conda_env()
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account(gpu=True)}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --gpus={ngpus}\n"
        f"#SBATCH --ntasks-per-node={nt}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --mem={mem}G\n"
        "#SBATCH --error=%J.stderr\n"
        "#SBATCH --output=%J.stdout\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md', tolerant=True)}"
        f"source activate {gpu_env}\n"
        f"export PATH={lmp_dir}:$PATH\n"
        f"export LD_LIBRARY_PATH={lammps_src}:$LD_LIBRARY_PATH\n"
        "export MPICH_GPU_SUPPORT_ENABLED=1\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "export OMPI_MCA_mtl=^ofi\n"
        "export OMPI_MCA_btl=^openib\n"
        f"srun --gpus-per-task=1 lmp -k on g {ngpus} -sf kk -in in.lammps\n"
    )


def _build_lammps_cpu(job_name: str, *, ncores: int = 104,
                      time_key: str | None = None, time: str | None = None,
                      exclude: str = "", **_) -> str:
    """Build a CPU LAMMPS submission script using the DeepMD CPU virtualenv."""
    wall    = _resolve_time(time_key, time, "48:00:00")
    venv    = _lammps_cpu_venv()
    lmp_bin = _lammps_cpu_bin()
    mpirun  = _mpirun()
    excl    = f"#SBATCH --exclude={exclude}\n" if exclude else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account()}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node={ncores}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --error=%J.stderr\n"
        "#SBATCH --output=%J.stdout\n"
        f"{excl}"
        "module purge\n"
        f"source {venv}/bin/activate\n"
        "export DP_DISABLE_CUDA=1\n"
        "export OMP_NUM_THREADS=1\n"
        "export DP_INTRA_OP_PARALLELISM_THREADS=1\n"
        "export DP_INTER_OP_PARALLELISM_THREADS=1\n"
        f"{mpirun} -np {ncores} {lmp_bin} -in in.lammps\n"
    )


def _build_deepmd_cpu(job_name: str, *, mlff_dir,
                      time_key: str | None = "mlip_cpu",
                      time: str | None = None, **_) -> str:
    """Build a DeepMD CPU train → freeze → compress submission script."""
    wall     = _resolve_time(time_key, time, "120:00:00")
    cpu_env  = _cpu_venv()
    cpu_dp   = f"{cpu_env}/bin/dp"
    mlff_dir = Path(mlff_dir)
    train_dir = mlff_dir / "01.train"
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account()}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks-per-node=104\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --error={mlff_dir}/%J.stderr\n"
        f"#SBATCH --output={mlff_dir}/%J.stdout\n"
        "module purge\n"
        f"source {cpu_env}/bin/activate\n"
        "export DP_DISABLE_CUDA=1\n"
        "export TF_ENABLE_ONEDNN_OPTS=0\n"
        "export OMP_NUM_THREADS=1\n"
        "export DP_INTRA_OP_PARALLELISM_THREADS=20\n"
        "export DP_INTER_OP_PARALLELISM_THREADS=20\n"
        "export DP_INFER_BATCH_SIZE=16384\n"
        f"cd {train_dir}\n"
        f"{cpu_dp} train deepmd_input.json 2>&1 | tee dp_train.log\n"
        f"{cpu_dp} freeze -o pot 2>&1 | tee freeze.log\n"
        f"{cpu_dp} compress -i pot.pb -o pot_com --training-script deepmd_input.json 2>&1 | tee compress.log\n"
        f"cp pot_com.pb {mlff_dir}/pot_com.pb\n"
    )


def _build_deepmd_al(job_name: str, *, mlff_dir,
                     data_dir: str = "00.data",
                     time_key: str | None = "mlip_cpu",
                     time: str | None = None, **_) -> str:
    """Active-learning DeepMD retrain: train → freeze → compress → test."""
    wall    = _resolve_time(time_key, time, "48:00:00")
    cpu_env = _cpu_venv()
    cpu_dp  = f"{cpu_env}/bin/dp"
    mlff_dir = Path(mlff_dir)
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account()}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks-per-node=104\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --error={mlff_dir}/%J.stderr\n"
        f"#SBATCH --output={mlff_dir}/%J.stdout\n"
        "module purge\n"
        f"source {cpu_env}/bin/activate\n"
        "export DP_DISABLE_CUDA=1\n"
        "export TF_ENABLE_ONEDNN_OPTS=0\n"
        "export OMP_NUM_THREADS=8\n"
        "export DP_INTRA_OP_PARALLELISM_THREADS=8\n"
        "export DP_INTER_OP_PARALLELISM_THREADS=4\n"
        f"cd {mlff_dir}\n"
        "rm -f pot_com.pb pot.pb\n"
        f"{cpu_dp} train deepmd_input.json 2>&1 | tee train_al.log\n"
        f"{cpu_dp} freeze -o pot.pb 2>&1 | tee freeze_al.log\n"
        f"{cpu_dp} compress -i pot.pb -o pot_com.pb 2>&1 | tee compress_al.log\n"
        f"{cpu_dp} test -m pot_com.pb -s {data_dir} -n 1000 -d test_al 2>&1 | tee test_results.txt\n"
    )


def _build_mace_gpu(job_name: str, *, mlff_dir, cfg_path,
                    time_key: str | None = "mlip_gpu",
                    time: str | None = None, **_) -> str:
    """Build a MACE GPU fine-tuning submission script."""
    wall     = _resolve_time(time_key, time, "48:00:00")
    gpu_env  = _gpu_conda_env()
    mlff_dir = Path(mlff_dir)
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account(gpu=True)}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --gpus=2\n"
        "#SBATCH --ntasks-per-node=1\n"
        "#SBATCH --cpus-per-task=16\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --mem=200G\n"
        f"#SBATCH --error={mlff_dir}/%J.stderr\n"
        f"#SBATCH --output={mlff_dir}/%J.stdout\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md', tolerant=True)}"
        f"source activate {gpu_env}\n"
        f"cd {mlff_dir}\n"
        f"mace-train --config {cfg_path} 2>&1 | tee mace_train.log\n"
    )


def _build_analysis_cpu(job_name: str, *, project_dir, variant: str,
                        time_key: str | None = "analysis_cpu",
                        time: str | None = None, **_) -> str:
    """Build a CPU submission script that runs one h06_analysis variant as its own job.

    Moves MSD/RDF/Van Hove/coordination/ion-pair/VACF analysis onto its own
    node instead of the daemon's process — see hpca.orchestrator.handlers.h06_analysis._worker.
    """
    wall    = _resolve_time(time_key, time, "12:00:00")
    ntasks  = int(_hpc("analysis_cpu_ntasks_slurm", 104))
    py_bin  = _hpc("python_cladue", "python3")
    project_dir = Path(project_dir)
    return (
        "#!/bin/bash\n"
        f"#SBATCH --account={_account()}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node={ntasks}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        "#SBATCH --error=%J.stderr\n"
        "#SBATCH --output=%J.stdout\n"
        f"{py_bin} -m hpca.orchestrator.handlers.h06_analysis._worker "
        f"--project-dir {shlex.quote(str(project_dir))} --variant {shlex.quote(variant)}\n"
    )


# ── Dispatch table ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubmissionDefinition:
    """Inspectable contract for one canonical submission template."""

    builder: Callable[..., str]
    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()
    family: str = "slurm"

    def validate(self, key: str, parameters: dict) -> None:
        missing = self.required - parameters.keys()
        unknown = parameters.keys() - self.required - self.optional
        if missing:
            raise TypeError(f"submission {key!r} missing required parameters: {sorted(missing)}")
        if unknown:
            raise TypeError(f"submission {key!r} received unknown parameters: {sorted(unknown)}")


def _definition(builder: Callable[..., str], *, required=(), optional=(),
                family: str = "slurm") -> SubmissionDefinition:
    return SubmissionDefinition(builder, frozenset(required), frozenset(optional), family)


SUBMISSIONS: dict[str, SubmissionDefinition] = {
    "vasp": _definition(_build_vasp, optional=("nodes", "ntasks", "time_key", "time", "exclusive")),
    "vasp_aimd": _definition(_build_vasp_aimd, required=("natoms",), optional=("time_key", "time")),
    "vasp_batch": _definition(_build_vasp_batch, required=("box_dirs",), optional=("ntasks_per_job", "time_key", "time")),
    "vasp_neb_endpoints": _definition(_build_vasp_neb_endpoints, required=("endpoint_tags",), optional=("cores_per_node", "time_key", "time")),
    "vasp_neb_images": _definition(_build_vasp_neb_images, required=("image_tags",), optional=("cores_per_image", "cores_per_node", "time_key", "time")),
    "submit_fanout": _definition(_build_submit_fanout, required=("scripts",), family="local"),
    "vasp_ncore_phase1": _definition(_build_vasp_ncore_phase1, required=("work_dir", "poscar", "potcar"), optional=("kpoints_mesh", "candidates")),
    "vasp_ncore_phase2": _definition(_build_vasp_ncore_phase2, required=("work_dir", "poscar", "potcar"), optional=("kpoints_mesh",)),
    "vasp_ncore_finalize": _definition(_build_vasp_ncore_finalize, required=("work_dir",)),
    "lammps_gpu": _definition(_build_lammps_gpu, optional=("pot_path", "n_gpus", "ntasks", "cpus_per_task", "mem_gb", "time_key", "time")),
    "lammps_cpu": _definition(_build_lammps_cpu, optional=("ncores", "time_key", "time", "exclude")),
    "deepmd_cpu": _definition(_build_deepmd_cpu, required=("mlff_dir",), optional=("time_key", "time")),
    "deepmd_al": _definition(_build_deepmd_al, required=("mlff_dir",), optional=("data_dir", "time_key", "time")),
    "mace_gpu": _definition(_build_mace_gpu, required=("mlff_dir", "cfg_path"), optional=("time_key", "time")),
    "analysis_cpu": _definition(_build_analysis_cpu, required=("project_dir", "variant"), optional=("time_key", "time")),
}


# ── Public API ────────────────────────────────────────────────────────────────

def write_submission(path: Path, key: str, job_name: str, **kwargs) -> Path:
    """Write a SLURM submission script to *path* and return it.

    Parameters
    ----------
    path:
        Destination path (e.g. ``calc_dir / "sub.sh"``).
    key:
        Template key — see module docstring for the full list.
    job_name:
        Value for ``#SBATCH --job-name``.
    **kwargs:
        Template-specific parameters declared by :data:`SUBMISSIONS`. Unknown
        parameters are rejected to prevent misspelled scheduler settings.

    Returns
    -------
    Path
        The written script path (same as *path*).
    """
    definition = SUBMISSIONS.get(key)
    if definition is None:
        raise KeyError(
            f"submission_registry: unknown key {key!r}. "
            f"Valid keys: {sorted(SUBMISSIONS)}"
        )
    definition.validate(key, kwargs)
    content = definition.builder(job_name, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path
