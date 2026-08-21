"""
lammps_job.py — Standalone LAMMPS MD utility functions for all handlers.

Provides NVT/NPT input generation, dump validation, and SLURM script
writing as importable functions, eliminating duplicated code across
h05_cmd.py and h05_lammps.py handlers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from hpca.core.config import Config
from hpca.core.slurm_submit import module_bundle_lines
from hpca.data import load as _load_data
from hpca.core.config import account_fallback as _account_fallback


# ── Atomic masses (lazy-loaded once) ─────────────────────────────────────────

_ATOMIC_MASS: dict | None = None


def _get_atomic_mass() -> dict:
    """Return the cached atomic masses dict, loading it from hpca.data on first call."""
    global _ATOMIC_MASS
    if _ATOMIC_MASS is None:
        _ATOMIC_MASS = _load_data("atomic_masses")  # type: ignore[assignment]
    return _ATOMIC_MASS  # type: ignore[return-value]


# ── Block generators ──────────────────────────────────────────────────────────

def mass_block(elements: list[str]) -> str:
    """Return LAMMPS 'mass N value' lines for each element type (1-indexed).

    Masses are read from hpca.data atomic_masses.json.  Unknown elements
    default to 1.0.
    """
    masses = _get_atomic_mass()
    lines = [f"mass         {i} {masses.get(el, 1.0)}" for i, el in enumerate(elements, 1)]
    return "\n".join(lines)


def pair_style_block(pot_path: str, pot_type: str) -> str:
    """Return LAMMPS pair_style + pair_coeff lines for the given potential type.

    Supported pot_type values: 'deepmd', 'mace', 'uma'.
    Unknown types fall back to deepmd syntax.
    """
    if pot_type == "mace":
        return f"pair_style   mace no_domain_decomposition {pot_path}\npair_coeff   * *\n"
    if pot_type == "uma":
        return f"pair_style   uma {pot_path}\npair_coeff   * *\n"
    # deepmd and default
    return f"pair_style   deepmd {pot_path}\npair_coeff   * *\n"


# ── Input file writers ────────────────────────────────────────────────────────

def write_nvt_input(
    path: Path,
    T: int,
    pot_path: str,
    pot_type: str,
    n_steps: int,
    dump_every: int,
    elements: list[str],
    timestep_ps: float = 0.001,
    nvt_damp: float = 0.05,
    seed: int = 12345,
) -> None:
    """Write NVT LAMMPS input file to *path*.

    Parameters
    ----------
    path:
        Destination file (e.g. nvt_dir / "in.lammps").
    T:
        Target temperature in K.
    pot_path:
        Path string to the potential file.
    pot_type:
        Potential type: 'deepmd', 'mace', or 'uma'.
    n_steps:
        Number of MD steps.
    dump_every:
        Dump/thermo frequency (steps).
    elements:
        Ordered element list matching LAMMPS type indices.
    timestep_ps:
        MD timestep in picoseconds (default 0.001 ps = 1 fs).
    nvt_damp:
        NVT thermostat damping parameter (ps).
    seed:
        Random seed for Maxwell-Boltzmann velocity initialisation.
        Use hash(project_name + str(T)) % 900000 + 100000 for uncorrelated runs.
    """
    el_str = " ".join(elements)
    pair_lines = pair_style_block(pot_path, pot_type)
    script = (
        "units        metal\n"
        "atom_style   atomic\n"
        "boundary     p p p\n"
        "read_data    data.lammps\n"
        f"{mass_block(elements)}\n"
        f"{pair_lines}\n"
        f"timestep     {timestep_ps}\n"
        f"thermo       {dump_every}\n"
        "thermo_style custom step temp pe ke etotal press\n"
        f"velocity     all create {T} {seed} mom yes rot yes dist gaussian\n"
        f"dump         1 all custom {dump_every} dump_unwrapped.lmp id type element xu yu zu\n"
        f"dump_modify  1 element {el_str} sort id\n"
        f"fix          1 all nvt temp {T} {T} {nvt_damp}\n"
        f"run          {n_steps}\n"
    )
    path.write_text(script)


def write_npt_input(
    path: Path,
    T: int,
    pot_path: str,
    pot_type: str,
    n_steps: int,
    dump_every: int,
    elements: list[str],
    timestep_ps: float = 0.001,
    npt_temp_damp: float = 0.1,
    npt_pres_damp: float = 1.0,
) -> None:
    """Write NPT equilibration LAMMPS input file to *path*.

    Saves a restart file and nvt_start.dat for seeding subsequent NVT runs.

    Parameters
    ----------
    path:
        Destination file (e.g. npt_dir / "in.lammps").
    T:
        Target temperature in K.
    pot_path:
        Path string to the potential file.
    pot_type:
        Potential type: 'deepmd', 'mace', or 'uma'.
    n_steps:
        Number of NPT equilibration steps.
    dump_every:
        Dump/thermo frequency (steps).
    elements:
        Ordered element list matching LAMMPS type indices.
    timestep_ps:
        MD timestep in picoseconds (default 0.001 ps = 1 fs).
    npt_temp_damp:
        NPT thermostat damping parameter (ps).
    npt_pres_damp:
        NPT barostat damping parameter (ps).
    """
    el_str = " ".join(elements)
    pair_lines = pair_style_block(pot_path, pot_type)
    script = (
        "units        metal\n"
        "atom_style   atomic\n"
        "boundary     p p p\n"
        "read_data    data.lammps\n"
        f"{mass_block(elements)}\n"
        f"{pair_lines}\n"
        f"timestep     {timestep_ps}\n"
        f"thermo       {dump_every}\n"
        "thermo_style custom step temp pe ke etotal press vol\n"
        f"dump         1 all custom {dump_every} dump_npt.lmp id type xu yu zu\n"
        f"dump_modify  1 element {el_str} sort id\n"
        f"fix          1 all npt temp {T} {T} {npt_temp_damp} iso 0.0 0.0 {npt_pres_damp}\n"
        f"run          {n_steps}\n"
        "write_restart restart.lammps\n"
        "write_data   nvt_start.dat\n"
    )
    path.write_text(script)


# ── Dump validation ───────────────────────────────────────────────────────────

def dump_valid(dump_path: Path, min_bytes: int = 1_000_000) -> bool:
    """Return True if *dump_path* exists and its size is >= *min_bytes*."""
    try:
        return dump_path.exists() and dump_path.stat().st_size >= min_bytes
    except OSError:
        return False


# ── SLURM script writers ──────────────────────────────────────────────────────

def write_gpu_lammps_slurm(
    path: Path,
    job_name: str,
    wall: str,
    pot_path: Optional[Path] = None,
    cfg: Optional[dict] = None,
) -> None:
    """Write GPU SLURM script for MACE/DeepMD-GPU LAMMPS runs.

    HPC settings are read from *cfg["hpc"]* (or platform.yaml if cfg is None).
    GPU resource sizing is determined by the potential file size via
    ``hpca.core.gpu_sizing.get_lammps_gpu_resources``.

    Parameters
    ----------
    path:
        Destination script file (e.g. npt_dir / "sub.sh").  Parent directory
        is used as the SLURM working directory.
    job_name:
        SLURM ``--job-name`` value.
    wall:
        SLURM wall-time string (e.g. ``"12:00:00"``).
    pot_path:
        Path to the potential file; used to size GPU resources.  May be None
        or non-existent — the most-generous tier is used as a fallback.
    cfg:
        Optional pre-loaded config dict.  When None the platform.yaml
        singleton is used instead.
    """
    from hpca.core.gpu_sizing import get_lammps_gpu_resources

    if cfg is not None:
        hpc = cfg.get("hpc", {})
    else:
        hpc = Config.get().raw.get("hpc", {})

    lmp_bin     = hpc.get("lammps_bin",
                           "")
    py_deep     = hpc.get("python_deepmd", "")
    gpu_env_dir = str(Path(py_deep).parent.parent) if py_deep else \
                  ""
    account     = hpc.get("accounts", {}).get("gpu_h100") or _account_fallback()
    lmp_dir     = lmp_bin.rpartition("/")[0]
    work_dir    = str(path.parent)

    sizing   = get_lammps_gpu_resources(pot_path or Path("/nonexistent"))
    ngpus    = sizing["gpus"]
    ntasks   = sizing["ntasks"]
    cpus     = sizing["cpus_per_task"]
    mem_gb   = sizing["mem_gb"]

    # Build LD_LIBRARY_PATH extension from the lammps install tree
    lammps_src = f"{lmp_dir}/lammps/src" if lmp_dir else ""

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --gpus={ngpus}\n"
        f"#SBATCH --ntasks-per-node={ntasks}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --mem={mem_gb}G\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J.stdout\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md', tolerant=True)}"
        f"source activate {gpu_env_dir}\n"
        f"export PATH={lmp_dir}:$PATH\n"
        f"export LD_LIBRARY_PATH={lammps_src}:$LD_LIBRARY_PATH\n"
        "export MPICH_GPU_SUPPORT_ENABLED=1\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "export OMPI_MCA_mtl=^ofi\n"
        "export OMPI_MCA_btl=^openib\n"
        f"cd {work_dir}\n"
        f"srun --gpus-per-task=1 lmp -k on g {ngpus} -sf kk -in in.lammps\n"
    )
    path.write_text(script)
    path.chmod(0o755)


def write_cpu_lammps_slurm(
    path: Path,
    job_name: str,
    wall: str,
    ncores: int = 104,
    cfg: Optional[dict] = None,
) -> None:
    """Write CPU SLURM script for DeepMD-CPU LAMMPS runs.

    HPC settings are read from *cfg["hpc"]* (or platform.yaml if cfg is None).

    Parameters
    ----------
    path:
        Destination script file (e.g. nvt_dir / "sub.sh").  Parent directory
        is used as the SLURM working directory.
    job_name:
        SLURM ``--job-name`` value.
    wall:
        SLURM wall-time string (e.g. ``"48:00:00"``).
    ncores:
        Number of MPI ranks (``--ntasks-per-node``).  Should be the result of
        a benchmark or a reasonable default (104 = full standard node).
    cfg:
        Optional pre-loaded config dict.  When None the platform.yaml
        singleton is used instead.
    """
    if cfg is not None:
        hpc = cfg.get("hpc", {})
    else:
        hpc = Config.get().raw.get("hpc", {})

    venv      = hpc.get("deepmd_lammps_venv_2023",
                         "")
    lmp_bin   = hpc.get("lammps_cpu_bin",
                         "")
    mpirun    = hpc.get("mpirun_bin", "mpirun")
    account   = hpc.get("accounts", {}).get("standard") or _account_fallback()
    work_dir  = str(path.parent)

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks-per-node={ncores}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=0\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --error={work_dir}/%J.stderr\n"
        f"#SBATCH --output={work_dir}/%J.stdout\n"
        "module purge\n"
        f"source {venv}/bin/activate\n"
        "export DP_DISABLE_CUDA=1\n"
        "export OMP_NUM_THREADS=1\n"
        "export DP_INTRA_OP_PARALLELISM_THREADS=1\n"
        "export DP_INTER_OP_PARALLELISM_THREADS=1\n"
        f"cd {work_dir}\n"
        f"{mpirun} -np {ncores} {lmp_bin} -in in.lammps\n"
    )
    path.write_text(script)
    path.chmod(0o755)
