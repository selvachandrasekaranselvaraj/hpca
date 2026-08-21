"""
Stage 02 — MLIP training and inference.

Supports: DeepMD-kit, MACE (foundation + fine-tune), UMA (zero-shot).
Generates training input files, submits GPU jobs, tracks training.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

# Layout: see hpca/core/paths.py
from hpca.core.paths import mlmd_mlff, dft_opt, load_platform_config
from hpca.core.slurm_submit import module_bundle_lines
from hpca.core.config import account_fallback as _account_fallback

# HPC paths loaded lazily from platform.yaml — cross-ref: hpca/config/platform.yaml
def _deepmd_env() -> str:
    """Return the DeepMD conda env dir from platform.yaml hpc.deepmd_lammps_gpu_env or hpc.python_deepmd."""
    hpc = load_platform_config().get("hpc", {})
    explicit = hpc.get("deepmd_lammps_gpu_env", "")
    if explicit:
        return explicit
    py = hpc.get("python_deepmd", "")
    return str(Path(py).parent.parent) if py else ""

def _account() -> str:
    """Return the GPU H100 SLURM account name from platform.yaml."""
    return load_platform_config().get("hpc", {}).get("accounts", {}).get("gpu_h100") or _account_fallback()


# ── DeepMD-kit ────────────────────────────────────────────────────────────────

DEEPMD_INPUT_TEMPLATE = {
    "model": {
        "type_map": [],
        "descriptor": {
            "type": "se_a",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "sel": [46, 92],
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1,
        },
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-3,
        "stop_lr": 3.51e-8,
        "decay_steps": 2000,
    },
    "loss": {
        "start_pref_e": 0.02, "limit_pref_e": 1,
        "start_pref_f": 1000, "limit_pref_f": 1,
        "start_pref_v": 0, "limit_pref_v": 0,
    },
    "training": {
        "stop_batch": 400000,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "numb_test": 10,
        "save_freq": 1000,
        "save_ckpt": "model.ckpt",
        "training_data": {"systems": ["00.data/set.000"], "batch_size": 32},
        "validation_data": {"systems": ["00.data/validation_data"], "batch_size": 32},
    },
}


def setup_deepmd_training(project_dir: Path, type_map: list[str],
                           n_steps: int = 400000,
                           rcut: float = 6.0) -> Path:
    """
    Set up DeepMD training directory with customized input.json.
    Expects: {project_dir}/mlff/00.data/set.000/ with box.npy etc.
    """
    mlff_dir = mlmd_mlff(project_dir)
    mlff_dir.mkdir(parents=True, exist_ok=True)

    inp = json.loads(json.dumps(DEEPMD_INPUT_TEMPLATE))  # deep copy
    inp["model"]["type_map"] = type_map
    inp["model"]["descriptor"]["rcut"] = rcut
    inp["training"]["stop_batch"] = n_steps

    input_path = mlff_dir / "deepmd_input.json"
    input_path.write_text(json.dumps(inp, indent=2))

    env      = _deepmd_env()
    dp_bin   = f"{env}/bin/dp"
    acct     = _account()
    sub = mlff_dir / "sub_deepmd.sh"
    sub.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --account={acct}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --gpus=4\n"
        f"#SBATCH --ntasks-per-node=4\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --time=48:00:00\n"
        f"#SBATCH --job-name={project_dir.name}_deepmd\n"
        f"#SBATCH --mem=300G\n"
        f"#SBATCH --output={mlff_dir}/train_%j.out\n"
        f"#SBATCH --error={mlff_dir}/train_%j.err\n"
        "\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md')}"
        f"source activate {env}\n"
        f"cd {mlff_dir}\n"
        f"{dp_bin} train deepmd_input.json\n"
        f"{dp_bin} freeze -o pot.pb\n"
        f"{dp_bin} compress -i pot.pb -o pot_com.pb\n"
    )
    sub.chmod(0o755)
    return mlff_dir


def check_deepmd_training(mlff_dir: Path) -> dict:
    """Parse lcurve.out to get training status."""
    lcurve = mlff_dir / "lcurve.out"
    if not lcurve.exists():
        return {"status": "not_started"}

    lines = lcurve.read_text().strip().splitlines()
    if len(lines) < 2:
        return {"status": "running", "steps": 0}

    last = lines[-1].split()
    try:
        step = int(last[0])
        e_rmse = float(last[4])
        f_rmse = float(last[6])
    except (ValueError, IndexError):
        return {"status": "running", "steps": len(lines)}

    pot_com = mlff_dir / "pot_com.pb"
    status = "complete" if pot_com.exists() else "training"
    return {"status": status, "steps": step, "e_rmse_meV": e_rmse * 1000,
            "f_rmse_meV_A": f_rmse * 1000}


# ── MACE foundation model ────────────────────────────────────────────────────

def setup_mace_finetune(project_dir: Path, train_xyz: Path,
                         foundation: str = "mace-mpa-0") -> Path:
    """Set up MACE fine-tune config."""
    mace_dir = mlmd_mlff(project_dir) / "mace_ft"
    mace_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "name": f"{project_dir.name}_mace_ft",
        "train_file": str(train_xyz),
        "valid_fraction": 0.1,
        "foundation_model": foundation,
        "num_interactions": 2,
        "max_ell": 3,
        "r_max": 5.0,
        "max_num_epochs": 100,
        "batch_size": 4,
        "lr": 1e-4,
        "forces_weight": 100.0,
        "save_cpu": True,
    }
    import yaml
    (mace_dir / "mace_config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    env      = _deepmd_env()
    mace_bin = f"{env}/bin/mace-train"
    acct     = _account()
    sub = mace_dir / "sub_mace.sh"
    sub.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --account={acct}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --gpus=4\n"
        "#SBATCH --ntasks-per-node=4\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --time=48:00:00\n"
        f"#SBATCH --job-name={project_dir.name}_mace_ft\n"
        "#SBATCH --mem=300G\n"
        "\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md')}"
        f"source activate {env}\n"
        f"cd {mace_dir}\n"
        f"{mace_bin} --config mace_config.yaml\n"
    )
    sub.chmod(0o755)
    return mace_dir


# ── Stage runner ──────────────────────────────────────────────────────────────

def run(project, output_base: Path = None, mlip: str = "deepmd",
        submit: bool = False, **kwargs) -> dict:
    """Stage 02 entry point. mlip: deepmd | mace | mace-finetune"""
    proj_dir = Path(project.root)
    results  = {"mlip": mlip, "status": "prepared"}

    if mlip == "deepmd":
        type_map = kwargs.get("type_map",
                               list(dict.fromkeys([project.mobile_ion] +
                                                   kwargs.get("other_species", []))))
        mlff_dir = setup_deepmd_training(proj_dir, type_map,
                                          n_steps=kwargs.get("n_steps", 400000))
        results["mlff_dir"] = str(mlff_dir)
        if submit:
            r = subprocess.run(["sbatch", "sub_deepmd.sh"], capture_output=True,
                                text=True, cwd=str(mlff_dir))
            results["slurm"] = r.stdout.strip()
            results["status"] = "submitted" if r.returncode == 0 else "failed"

    elif mlip in ("mace", "mace-finetune"):
        train_xyz = kwargs.get("train_xyz", mlmd_mlff(proj_dir) / "train.xyz")
        mace_dir  = setup_mace_finetune(proj_dir, Path(train_xyz))
        results["mace_dir"] = str(mace_dir)
        if submit:
            r = subprocess.run(["sbatch", "sub_mace.sh"], capture_output=True,
                                text=True, cwd=str(mace_dir))
            results["slurm"] = r.stdout.strip()
            results["status"] = "submitted" if r.returncode == 0 else "failed"

    # Check existing training status
    mlff_dir = mlmd_mlff(proj_dir)
    if mlff_dir.exists():
        results["training_status"] = check_deepmd_training(mlff_dir)

    return results
