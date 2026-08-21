"""
_benchmark.py — LAMMPS CPU core-count benchmarking for DeepMD potentials.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("hpca.orch")


def benchmark_lammps_ntasks(
    pot_path: Path,
    data_path: Path,
    hpc_config: dict,
) -> int:
    """Run 10-step DeepMD LAMMPS benchmarks (104→32 cores, step -8).

    Result cached in pot_path.parent/_ncores_cached.
    Falls back to 104 if all benchmarks fail.
    """
    cache_file = pot_path.parent / "_ncores_cached"
    if cache_file.exists():
        try:
            cached = int(cache_file.read_text().strip())
            log.info("[h05_lammps] ncores from cache: %d", cached)
            return cached
        except Exception:
            pass

    venv = hpc_config.get("deepmd_lammps_venv_2023", "")
    lmp  = hpc_config.get("lammps_cpu_bin", "")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = Path(tmp)
        shutil.copy(data_path, bench_dir / "data.lammps")
        (bench_dir / "in.lammps").write_text(
            "units        metal\n"
            "atom_style   atomic\n"
            "boundary     p p p\n"
            "read_data    data.lammps\n"
            f"pair_style   deepmd {pot_path}\n"
            "pair_coeff   * *\n"
            "timestep     0.001\n"
            "thermo       10\n"
            "run          10\n"
        )
        env_prefix = (
            f"source {venv}/bin/activate && "
            "export DP_DISABLE_CUDA=1 && "
            "export OMP_NUM_THREADS=1 && "
            "export DP_INTRA_OP_PARALLELISM_THREADS=1 && "
            "export DP_INTER_OP_PARALLELISM_THREADS=1 && "
            "export TF_CPP_MIN_LOG_LEVEL=3 && "
            "export TF_ENABLE_ONEDNN_OPTS=0"
        )
        candidates = [104, 100, 96, 88, 80, 72, 64, 56, 48, 40, 32]
        best_n, best_t = candidates[0], float("inf")
        for n in candidates:
            cmd = f"{env_prefix} && mpirun -np {n} {lmp} -in in.lammps"
            t0 = time.time()
            try:
                r = subprocess.run(
                    ["bash", "-c", cmd],
                    cwd=str(bench_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                )
                elapsed = time.time() - t0
            except subprocess.TimeoutExpired:
                log.warning("[h05_lammps] bench timeout at ncores=%d", n)
                continue
            if r.returncode == 0:
                log.info("[h05_lammps] bench ncores=%3d: %.2f s", n, elapsed)
                if elapsed < best_t:
                    best_t, best_n = elapsed, n
            else:
                log.warning("[h05_lammps] bench failed at ncores=%d (exit %d)", n, r.returncode)

    log.info("[h05_lammps] optimal ncores=%d (%.2f s) — cached", best_n, best_t)
    try:
        cache_file.write_text(str(best_n))
    except OSError as e:
        log.warning("[h05_lammps] could not write ncores cache: %s", e)
    return best_n
