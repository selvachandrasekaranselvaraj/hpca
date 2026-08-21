"""
Stage 01 — DFT input generation and submission (VASP).

Handles: vc-relax, ionic opt, AIMD (NVT/NVE), Bader, DOS/PDOS, NEB.
Generates INCAR/KPOINTS/POSCAR and sub.sh, then optionally submits.
"""
# Layout: see hpca/core/paths.py
from __future__ import annotations
import json
from pathlib import Path

from hpca.core.paths import (
    dft_base as _paths_dft_base,
    dft_vc, dft_opt, dft_aimd,
    designed_structures, preopt, contcar_preopt, poscar_dft,
    load_platform_config,
)
from hpca.registry.incar import build_incar as _build_incar, write_incar as _reg_write_incar
from hpca.core.config import account_fallback as _account_fallback

# HPC paths loaded lazily from platform.yaml — cross-ref: hpca/config/platform.yaml
def _hpc(key: str, default: str = "") -> str:
    """Look up an HPC path or setting from platform.yaml."""
    return load_platform_config().get("hpc", {}).get(key, default)

def _account(key: str = "standard") -> str:
    """Return the Slurm account string for the given account tier from platform.yaml."""
    return load_platform_config().get("hpc", {}).get("accounts", {}).get(key) or _account_fallback()

# Standard partition limit — leave 4h buffer below the 48h hard wall
_STANDARD_MAX_H = 44
# NEB uses long partition (days-long calculation)
_LONG_MAX_H     = 240   # 10 days

# ── Mode-aware simulation limits ────────────────────────────────────────────
# interactive: quick test scaffold (hpca new); daemon: production SLURM runs
_DFT_LIMITS = {
    "natoms":   200,
    "vc_nsw":   300,
    "opt_nsw":  300,
    "aimd_nsw": 2000,
}


# ── Walltime calculation ──────────────────────────────────────────────────────

def _fmth(hours: float) -> str:
    """Float hours → HH:MM:00 walltime string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}:{m:02d}:00"


def vasp_walltime(natoms: int, nsw: int, job_type: str = "aimd",
                   gamma_only: bool = True, n_nodes: int = 2) -> str:
    """
    Calculate VASP walltime, capped at 44h for the standard partition.

    Empirical rates on Kestrel (2 nodes = 208 OpenMP-MPI CPUs):
      AIMD  Gamma-only : 0.06 s / atom / MD step
      AIMD  k-points   : 0.20 s / atom / MD step
      Relax Gamma-only : 0.12 s / atom / ionic step
      Static/Bader/DOS : 0.05 s / atom / SCF

    More nodes → faster; n_nodes scales linearly (roughly).
    """
    kp    = 1.0 if gamma_only else 3.5
    scale = 2.0 / max(n_nodes, 1)   # 2-node baseline

    if job_type == "aimd":
        sps   = 0.06 * natoms * kp * scale
        hours = (nsw * sps) / 3600 * 1.25
        hours = max(hours, 8.0)
    elif job_type in ("relax", "vc_relax"):
        sps   = 0.12 * natoms * kp * scale
        hours = (nsw * sps) / 3600 * 1.3 + 2
        hours = max(hours, 4.0)
    elif job_type == "neb":
        sps   = 0.10 * natoms * kp * scale
        hours = (nsw * sps) / 3600 * 1.5 + 4
        hours = max(hours, 12.0)
    else:   # static, bader, dos
        sps   = 0.05 * natoms * kp * scale
        hours = (nsw * sps) / 3600 * 1.2 + 2
        hours = max(hours, 2.0)

    return _fmth(min(hours, _STANDARD_MAX_H))


def write_incar(incar_dict: dict, path: Path, system_name: str = ""):
    """Write an INCAR file from a dict of tag → value pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if system_name:
        lines.append(f"SYSTEM = {system_name}\n")
    for k, v in incar_dict.items():
        lines.append(f" {k} = {v}\n")
    path.write_text("".join(lines))


def write_kpoints_gamma(path: Path, mesh: tuple = (2, 2, 2)):
    """Write a Gamma-centred KPOINTS file with the given mesh."""
    path.write_text(
        f"Automatic mesh\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    )


def write_sub_sh(path: Path, job_name: str, nodes: int = 2,
                  tasks_per_node: int = 104, time: str = "44:00:00",
                  account: str = "", exclusive: bool = False,
                  mem: str = ""):
    """Write a VASP Slurm submission script, loading the VASP module from platform.yaml."""
    vasp_module   = _hpc("vasp_module", "vasp/6.4.2_openMP")
    slurm_account = account or _account()
    lines = [
        "#!/bin/bash\n",
        f"#SBATCH --nodes={nodes}\n",
        f"#SBATCH --tasks-per-node={tasks_per_node}\n",
        "#SBATCH --cpus-per-task=1\n",
        f"#SBATCH --time={time}\n",
        f"#SBATCH --account={slurm_account}\n",
        f"#SBATCH --job-name={job_name}\n",
        "#SBATCH --error=%J.stderr\n",
        "#SBATCH --output=%J.stdout\n",
    ]
    if exclusive:
        lines.append("#SBATCH --exclusive\n")
    if mem:
        lines.append(f"#SBATCH --mem={mem}\n")
    lines += [
        "ulimit -s unlimited\n",
        f"module load {vasp_module}\n",
        "srun vasp_std &> out\n",
    ]
    path.write_text("".join(lines))
    path.chmod(0o755)


# ── Task builders ─────────────────────────────────────────────────────────────

def _natoms_from_poscar(poscar: Path) -> int:
    """Quick atom count from POSCAR (line 7 = counts per species)."""
    try:
        lines = poscar.read_text().splitlines()
        return sum(int(x) for x in lines[6].split())
    except Exception:
        return 100   # safe fallback


def setup_vc_relax(project_dir: Path, poscar_src: Path = None,
                    system: str = "", natoms: int = None) -> Path:
    """
    Set up variable-cell relaxation (ISIF=3).

    project_dir is the base directory; creates project_dir/vc/.
    For the new workflow pass project_dir=proj_root/'dft' so files go into dft/vc/.
    """
    nsw      = _DFT_LIMITS["vc_nsw"]
    calc_dir = project_dir / "vc"
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("vc_relax", nsw=nsw), calc_dir / "INCAR",
                system or f"{project_dir.name}_vc")
    write_kpoints_gamma(calc_dir / "KPOINTS")
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
        if natoms is None:
            natoms = _natoms_from_poscar(poscar_src)
    n = natoms or _DFT_LIMITS["natoms"]
    wt = vasp_walltime(n, nsw, "vc_relax", n_nodes=1)
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_vc",
                  nodes=1, tasks_per_node=104, time=wt)
    return calc_dir


def setup_opt(project_dir: Path, poscar_src: Path = None,
               system: str = "", natoms: int = None) -> Path:
    """
    Set up ionic relaxation (ISIF=2).

    project_dir/opt/ is created. Pass project_dir=proj_root/'dft' for dft/opt/.
    """
    nsw      = _DFT_LIMITS["opt_nsw"]
    calc_dir = dft_opt(project_dir)
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("opt", nsw=nsw), calc_dir / "INCAR",
                system or f"{project_dir.name}_opt")
    write_kpoints_gamma(calc_dir / "KPOINTS")
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
        if natoms is None:
            natoms = _natoms_from_poscar(poscar_src)
    n = natoms or _DFT_LIMITS["natoms"]
    wt = vasp_walltime(n, nsw, "relax", n_nodes=1)
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_opt",
                  nodes=1, tasks_per_node=104, time=wt)
    return calc_dir


def setup_aimd(project_dir: Path, poscar_src: Path,
                temperatures: list = (300,),
                nsw: int = None, dt_fs: float = 1.0,
                system: str = "", natoms: int = None,
                gamma_only: bool = True, n_nodes: int = 2) -> list:
    """
    Set up multi-temperature NVT AIMD.

    project_dir/aimd/{T}/ directories are created.
    """
    if nsw is None:
        nsw = _DFT_LIMITS["aimd_nsw"]
    if poscar_src and poscar_src.exists() and natoms is None:
        natoms = _natoms_from_poscar(poscar_src)
    n  = natoms or _DFT_LIMITS["natoms"]
    wt = vasp_walltime(n, nsw, "aimd", gamma_only=gamma_only, n_nodes=n_nodes)

    dirs = []
    for T in temperatures:
        calc_dir = project_dir / "aimd" / str(T)
        calc_dir.mkdir(parents=True, exist_ok=True)
        write_incar(_build_incar("aimd", tebeg=T, teend=T, nsw=nsw,
                                 extra={"POTIM": dt_fs}),
                    calc_dir / "INCAR", system or f"{project_dir.name}_AIMD_{T}K")
        write_kpoints_gamma(calc_dir / "KPOINTS", mesh=(1, 1, 1))
        if poscar_src and poscar_src.exists():
            (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
        write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_aimd_{T}K",
                      nodes=n_nodes, tasks_per_node=104, time=wt)
        dirs.append(calc_dir)
    return dirs


def setup_dft_workflow(project_dir: Path, poscar_src: Path,
                        temperatures: list = (300, 400, 600, 800),
                        system: str = "",
                        mode: str = "daemon",
                        gamma_only: bool = True) -> dict:
    """
    Set up the complete DFT workflow under project_dir/dft/:
      dft/vc/     — ISIF=3 variable-cell relaxation
      dft/opt/    — ISIF=2 ionic relaxation (poscar_src used until CONTCAR available)
      aimd/T/     — NVT AIMD at each temperature (at project root, not under dft/)

    mode='interactive' uses small atom counts and step counts for quick testing.
    mode='daemon'      uses production limits for SLURM submission.
    """
    dft_base = project_dir / "dft"
    vc_dir   = setup_vc_relax(dft_base, poscar_src, system, mode=mode)
    opt_dir  = setup_opt(dft_base, poscar_src, system, mode=mode)
    aimd_dirs = setup_aimd(dft_base, poscar_src, temperatures,
                             system=system, mode=mode, gamma_only=gamma_only)
    return {"dft_base": dft_base, "vc": vc_dir, "opt": opt_dir, "aimd": aimd_dirs}


def setup_bader(project_dir: Path, poscar_src: Path = None,
                 system: str = "", natoms: int = None) -> Path:
    """Set up Bader charge analysis static calculation."""
    calc_dir = project_dir / "bader"
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("bader"), calc_dir / "INCAR",
                system or f"{project_dir.name}_bader")
    write_kpoints_gamma(calc_dir / "KPOINTS", mesh=(4, 4, 4))
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
        if natoms is None:
            natoms = _natoms_from_poscar(poscar_src)
    n  = natoms or 100
    wt = vasp_walltime(n, 120, "static", gamma_only=False, n_nodes=1)
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_bader",
                  nodes=1, tasks_per_node=104, time=wt)
    post = calc_dir / "run_bader.sh"
    post.write_text(
        "#!/bin/bash\n"
        "chgsum.pl AECCAR0 AECCAR2\n"
        "bader CHGCAR -ref CHGCAR_sum\n"
    )
    return calc_dir


def setup_dos(project_dir: Path, poscar_src: Path = None,
               E_range: tuple = (-10, 10), nedos: int = 2000,
               system: str = "", natoms: int = None) -> Path:
    """Set up 2-step DOS calculation (SCF → nonSCF)."""
    base       = project_dir / "dos"
    scf_dir    = base / "scf"
    nonscf_dir = base / "nonscf"
    for d in (scf_dir, nonscf_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_incar(_build_incar("dos_scf"), scf_dir / "INCAR",
                system or f"{project_dir.name}_dos_scf")
    write_kpoints_gamma(scf_dir / "KPOINTS", mesh=(4, 4, 4))

    write_incar(_build_incar("dos_nonscf", extra={"EMIN": E_range[0], "EMAX": E_range[1],
                                                   "NEDOS": nedos}),
                nonscf_dir / "INCAR", system or f"{project_dir.name}_dos_nonscf")
    write_kpoints_gamma(nonscf_dir / "KPOINTS", mesh=(8, 8, 8))

    if poscar_src and poscar_src.exists():
        for d in (scf_dir, nonscf_dir):
            (d / "POSCAR").write_bytes(poscar_src.read_bytes())
        if natoms is None:
            natoms = _natoms_from_poscar(poscar_src)

    n  = natoms or 100
    wt = vasp_walltime(n, 200, "static", gamma_only=False, n_nodes=1)
    for d, name in [(scf_dir, "scf"), (nonscf_dir, "nonscf")]:
        write_sub_sh(d / "sub.sh", f"{project_dir.name}_dos_{name}",
                      nodes=1, tasks_per_node=104, time=wt)
    return base


def setup_neb(project_dir: Path, initial_poscar: Path, final_poscar: Path,
               n_images: int = 11, path_label: str = "path_a",
               natoms: int = None) -> Path:
    """
    Set up NEB calculation.  Uses long partition (up to 10 days) with exclusive node.
    Walltime calculated from n_images × natoms × steps.
    """
    calc_dir = project_dir / "neb" / path_label
    calc_dir.mkdir(parents=True, exist_ok=True)

    write_incar(_build_incar("neb", extra={"IMAGES": n_images}), calc_dir / "INCAR",
                f"{project_dir.name}_NEB_{path_label}")
    write_kpoints_gamma(calc_dir / "KPOINTS")

    for i in range(n_images + 2):
        (calc_dir / f"{i:02d}").mkdir(exist_ok=True)

    if initial_poscar.exists():
        (calc_dir / "00" / "POSCAR").write_bytes(initial_poscar.read_bytes())
        if natoms is None:
            natoms = _natoms_from_poscar(initial_poscar)
    if final_poscar.exists():
        (calc_dir / f"{n_images + 1:02d}" / "POSCAR").write_bytes(
            final_poscar.read_bytes())

    try:
        from pymatgen.core import Structure
        s0 = Structure.from_file(str(initial_poscar))
        s1 = Structure.from_file(str(final_poscar))
        images = s0.interpolate(s1, n_images + 1, autosort_tol=0.5)
        for i, img in enumerate(images[1:-1], start=1):
            img.to(fmt="poscar", filename=str(calc_dir / f"{i:02d}" / "POSCAR"))
    except Exception:
        pass

    # NEB walltime: n_images × natoms × nsw — can be very long → long partition
    _neb_nsw = _build_incar("neb").get("NSW", 300)
    n   = (natoms or 100) * n_images
    wt  = vasp_walltime(n, _neb_nsw, "neb", n_nodes=1)
    # If calculated time exceeds standard limit, still use long partition
    write_sub_sh(
        calc_dir / "sub_neb.sh",
        f"{project_dir.name}_neb_{path_label}",
        nodes=1, tasks_per_node=104, time=wt,
        exclusive=True,
    )
    return calc_dir


# ── Stage runner ──────────────────────────────────────────────────────────────

def _best_poscar_for_tier(proj_dir: Path, tier: str = "dft") -> Path:
    """Return the best available POSCAR for the given simulation tier.

    Priority: dft/preopt/CONTCAR for DFT (tier preopt otherwise), then designed structure.
    The returned path may not exist; callers are responsible for checking.
    """
    candidates = [
        contcar_preopt(proj_dir, tier),
        poscar_dft(proj_dir) if tier == "dft" else proj_dir / "designed_structures" / f"poscar_{tier}.vasp",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]  # return even if missing; callers handle non-existence


def run(project, output_base: Path = None,
        task: str = "opt", submit: bool = False, **kwargs) -> dict:
    """
    Stage 01 entry point.

    task options: vc_relax, opt, aimd, bader, dos, neb, dft_workflow

    DFT always goes into dft/ subdirectory (see hpca/core/paths.py).
    Pass mode='interactive' for reduced-size scaffold; 'daemon' for production.
    """
    proj_dir = Path(project.root)
    mode     = kwargs.get("mode", "daemon")
    results  = {"task": task, "status": "prepared"}

    # Always use the canonical dft/ layout
    dft_dir  = _paths_dft_base(proj_dir)

    # Convenience: run vc → opt → aimd in one call
    if task == "dft_workflow":
        poscar = _best_poscar_for_tier(proj_dir, tier="dft")
        temps  = kwargs.get("temperatures", [300, 400, 600, 800])
        res    = setup_dft_workflow(proj_dir, poscar, temps,
                                     system=project.full_name,
                                     mode=mode)
        results.update({k: str(v) if isinstance(v, Path) else
                         [str(d) for d in v] if isinstance(v, list) else v
                         for k, v in res.items()})
        results["status"] = "prepared"
        if submit:
            import subprocess
            for sub in [res["vc"] / "sub.sh", res["opt"] / "sub.sh"]:
                if sub.exists():
                    subprocess.run(["sbatch", str(sub)], cwd=str(sub.parent))
        return results

    calc_dir = None
    if task == "vc_relax":
        poscar = _best_poscar_for_tier(proj_dir, tier="dft")
        calc_dir = setup_vc_relax(dft_dir, poscar, project.full_name, mode=mode)
    elif task == "opt":
        # Prefer CONTCAR from vc; fall back to pre-optimised POSCAR
        poscar = dft_vc(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = _best_poscar_for_tier(proj_dir, tier="dft")
        calc_dir = setup_opt(dft_dir, poscar, project.full_name, mode=mode)
    elif task == "aimd":
        poscar = dft_opt(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_vc(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = _best_poscar_for_tier(proj_dir, tier="dft")
        temps     = kwargs.get("temperatures", [300, 400, 600, 800])
        calc_dirs = setup_aimd(dft_dir, poscar, temps,
                                nsw=kwargs.get("nsw", None),
                                system=project.full_name,
                                mode=mode)
        results["calc_dirs"] = [str(d) for d in calc_dirs]
    elif task == "bader":
        poscar   = dft_opt(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = proj_dir / "POSCAR"
        calc_dir = setup_bader(proj_dir, poscar, project.full_name)
    elif task == "dos":
        poscar   = dft_opt(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = proj_dir / "POSCAR"
        calc_dir = setup_dos(proj_dir, poscar, system=project.full_name)
    elif task == "neb":
        initial = kwargs.get("initial_poscar", proj_dir / "neb" / "initial" / "POSCAR")
        final   = kwargs.get("final_poscar",   proj_dir / "neb" / "final"   / "POSCAR")
        label   = kwargs.get("path_label", "path_a")
        calc_dir = setup_neb(proj_dir, Path(initial), Path(final),
                              n_images=kwargs.get("n_images", 11),
                              path_label=label)

    if calc_dir:
        results["calc_dir"] = str(calc_dir)

    if submit and calc_dir:
        import subprocess
        sub_sh = calc_dir / "sub.sh"
        if not sub_sh.exists():
            sub_sh = calc_dir / "sub_neb.sh"
        if sub_sh.exists():
            r = subprocess.run(["sbatch", str(sub_sh)],
                                capture_output=True, text=True, cwd=str(calc_dir))
            results["slurm_output"] = r.stdout.strip()
            results["status"] = "submitted" if r.returncode == 0 else "submit_failed"
            results["stderr"] = r.stderr.strip()

    return results
