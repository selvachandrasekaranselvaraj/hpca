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
    dft_base, dft_vc, dft_opt, dft_aimd, poscar_dft, contcar_preopt,
    load_platform_config,
)
from hpca.registry.incar import build_incar as _build_incar
from hpca.core.config import account_fallback as _account_fallback

# HPC paths read from platform.yaml — cross-ref: hpca/config/platform.yaml
def _hpc(key: str, default: str = "") -> str:
    """Return the HPC config value for key from platform.yaml."""
    return load_platform_config().get("hpc", {}).get(key, default)

def _account() -> str:
    """Return the standard SLURM account name from platform.yaml."""
    accounts = load_platform_config().get("hpc", {}).get("accounts", {})
    return accounts.get("standard") or _account_fallback()


def write_incar(incar_dict: dict, path: Path, system_name: str = ""):
    """Write an INCAR dict to path, optionally prepending a SYSTEM tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if system_name:
        lines.append(f"SYSTEM = {system_name}\n")
    for k, v in incar_dict.items():
        lines.append(f" {k} = {v}\n")
    path.write_text("".join(lines))


def write_kpoints_gamma(path: Path, mesh: tuple = (2, 2, 2)):
    """Write a Gamma-centred KPOINTS file with the given mesh to path."""
    path.write_text(
        f"Automatic mesh\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    )


def write_sub_sh(path: Path, job_name: str, nodes: int = 2,
                  tasks_per_node: int = 104, time: str = "72:00:00",
                  account: str = "", exclusive: bool = False,
                  mem: str = ""):
    """Write a VASP SLURM submission script to path."""
    vasp_module = _hpc("vasp_module", "vasp/6.4.2_openMP")
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

def setup_vc_relax(project_dir: Path, poscar_src: Path = None,
                    system: str = "") -> Path:
    """Set up variable-cell relaxation (ISIF=3)."""
    calc_dir = project_dir / "vc"
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("vc_relax"), calc_dir / "INCAR", system or f"{project_dir.name}_vc")
    write_kpoints_gamma(calc_dir / "KPOINTS")
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_vc", nodes=1,
                  tasks_per_node=96)
    return calc_dir


def setup_opt(project_dir: Path, poscar_src: Path = None,
               system: str = "") -> Path:
    """Set up ionic relaxation (ISIF=2). project_dir is dft_base(proj), not the project root."""
    calc_dir = project_dir / "opt"
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("opt"), calc_dir / "INCAR", system or f"{project_dir.name}_opt")
    write_kpoints_gamma(calc_dir / "KPOINTS")
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_opt", nodes=1,
                  tasks_per_node=96)
    return calc_dir


def setup_aimd(project_dir: Path, poscar_src: Path,
                temperatures: list[int] = (300,),
                nsw: int = 50000, dt_fs: float = 1.0,
                system: str = "") -> list[Path]:
    """Set up multi-temperature NVT AIMD. project_dir is dft_base(proj), not the project root."""
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
                      nodes=2, tasks_per_node=104, time="72:00:00")
        dirs.append(calc_dir)
    return dirs


def setup_bader(project_dir: Path, poscar_src: Path = None,
                 system: str = "") -> Path:
    """Set up Bader charge analysis static calculation."""
    calc_dir = project_dir / "bader"
    calc_dir.mkdir(parents=True, exist_ok=True)
    write_incar(_build_incar("bader"), calc_dir / "INCAR",
                system or f"{project_dir.name}_bader")
    write_kpoints_gamma(calc_dir / "KPOINTS", mesh=(4, 4, 4))
    if poscar_src and poscar_src.exists():
        (calc_dir / "POSCAR").write_bytes(poscar_src.read_bytes())
    write_sub_sh(calc_dir / "sub.sh", f"{project_dir.name}_bader",
                  nodes=1, tasks_per_node=96)

    # Post-processing script
    post = calc_dir / "run_bader.sh"
    post.write_text(
        "#!/bin/bash\n"
        "# Run after VASP completes\n"
        "chgsum.pl AECCAR0 AECCAR2       # → CHGCAR_sum\n"
        "bader CHGCAR -ref CHGCAR_sum    # → ACF.dat\n"
    )
    return calc_dir


def setup_dos(project_dir: Path, poscar_src: Path = None,
               E_range: tuple = (-10, 10), nedos: int = 2000,
               system: str = "") -> Path:
    """Set up 2-step DOS calculation (SCF → nonSCF)."""
    base = project_dir / "dos"
    scf_dir   = base / "scf"
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

    for d, name in [(scf_dir, "scf"), (nonscf_dir, "nonscf")]:
        write_sub_sh(d / "sub.sh", f"{project_dir.name}_dos_{name}",
                      nodes=1, tasks_per_node=96, time="24:00:00")
    return base


def setup_neb(project_dir: Path, initial_poscar: Path, final_poscar: Path,
               n_images: int = 11, path_label: str = "path_a") -> Path:
    """
    Set up NEB calculation directory with interpolated images.
    Uses pymatgen's NEB path interpolation if available, else writes placeholders.
    """
    calc_dir = project_dir / "neb" / path_label
    calc_dir.mkdir(parents=True, exist_ok=True)

    write_incar(_build_incar("neb", extra={"IMAGES": n_images}), calc_dir / "INCAR",
                f"{project_dir.name}_NEB_{path_label}")
    write_kpoints_gamma(calc_dir / "KPOINTS")

    # Create image directories
    for i in range(n_images + 2):
        img_dir = calc_dir / f"{i:02d}"
        img_dir.mkdir(exist_ok=True)

    # Copy endpoints
    if initial_poscar.exists():
        (calc_dir / "00" / "POSCAR").write_bytes(initial_poscar.read_bytes())
    if final_poscar.exists():
        (calc_dir / f"{n_images + 1:02d}" / "POSCAR").write_bytes(
            final_poscar.read_bytes()
        )

    # Try pymatgen interpolation
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.transition_state import NEBAnalysis
        s0 = Structure.from_file(str(initial_poscar))
        s1 = Structure.from_file(str(final_poscar))
        images = s0.interpolate(s1, n_images + 1, autosort_tol=0.5)
        for i, img in enumerate(images[1:-1], start=1):
            img.to(fmt="poscar", filename=str(calc_dir / f"{i:02d}" / "POSCAR"))
    except Exception:
        pass  # Images must be set up manually

    write_sub_sh(
        calc_dir / "sub_neb.sh",
        f"{project_dir.name}_neb_{path_label}",
        nodes=1, tasks_per_node=88, time="72:00:00",
        exclusive=True,
    )
    return calc_dir


# ── Stage runner ──────────────────────────────────────────────────────────────

def _best_poscar_for_tier(proj_dir: Path, tier: str = "dft") -> Path:
    """Return the best available POSCAR for a given tier (canonical layout only)."""
    candidates = [
        contcar_preopt(proj_dir, tier),
        poscar_dft(proj_dir) if tier == "dft" else proj_dir / "designed_structures" / f"poscar_{tier}.vasp",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def run(project, output_base: Path = None,
        task: str = "opt", submit: bool = False, **kwargs) -> dict:
    """
    Stage 01 entry point.

    task options: vc_relax, opt, aimd, bader, dos, neb, dft_workflow

    DFT always goes into dft/ (see hpca/core/paths.py).
    Pass mode='interactive' for reduced-size test scaffold (50 atoms, 50 steps).
    """
    proj_dir = Path(project.root)
    mode     = kwargs.get("mode", "daemon")
    results  = {"task": task, "status": "prepared"}

    # Always use the canonical dft/ layout
    dft_dir  = dft_base(proj_dir)

    # Convenience: run full vc→opt→aimd workflow
    if task == "dft_workflow":
        poscar    = _best_poscar_for_tier(proj_dir, "dft")
        temps     = kwargs.get("temperatures", [300, 400, 600, 800])
        vc_dir    = setup_vc_relax(dft_dir, poscar, project.full_name)
        opt_dir   = setup_opt(dft_dir, poscar, project.full_name)
        aimd_dirs = setup_aimd(dft_dir, poscar, temps, nsw=kwargs.get("nsw", 50000),
                                 system=project.full_name)
        results["vc"]   = str(vc_dir)
        results["opt"]  = str(opt_dir)
        results["aimd"] = [str(d) for d in aimd_dirs]
        return results

    calc_dir = None
    if task == "vc_relax":
        poscar = _best_poscar_for_tier(proj_dir, "dft")
        calc_dir = setup_vc_relax(dft_dir, poscar, project.full_name)
    elif task == "opt":
        # Prefer CONTCAR from vc; fall back to pre-optimised POSCAR
        poscar = dft_vc(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = _best_poscar_for_tier(proj_dir, "dft")
        calc_dir = setup_opt(dft_dir, poscar, project.full_name)
    elif task == "aimd":
        poscar = dft_opt(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_vc(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = _best_poscar_for_tier(proj_dir, "dft")
        temps     = kwargs.get("temperatures", [300, 600, 700, 800])
        calc_dirs = setup_aimd(dft_dir, poscar, temps,
                                nsw=kwargs.get("nsw", 50000),
                                system=project.full_name)
        results["calc_dirs"] = [str(d) for d in calc_dirs]
    elif task == "bader":
        poscar = dft_opt(proj_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = proj_dir / "POSCAR"
        calc_dir = setup_bader(proj_dir, poscar, project.full_name)
    elif task == "dos":
        poscar = dft_opt(proj_dir) / "CONTCAR"
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
