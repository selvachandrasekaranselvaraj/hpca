"""
deepmd_job.py — Standalone DeepMD and MACE training utilities.

Provides input file writing, SLURM script generation, and model management
functions as importable standalone utilities, reducing duplication in
h04_mlip.py and h13_active_learning.py handlers.
"""
from __future__ import annotations

import json
from pathlib import Path

from hpca.core.config import Config
from hpca.core.slurm_submit import module_bundle_lines
from hpca.core.config import account_fallback as _account_fallback


def _cfg_mlip(cfg: dict | None, key: str, default):
    """Read a value from cfg['mlip_defaults'][key] or Config.get().mlip(key, default)."""
    if cfg is not None:
        return cfg.get("mlip_defaults", {}).get(key, default)
    return Config.get().mlip(key, default)


def _cfg_hpc(cfg: dict | None, key: str, default: str = "") -> str:
    """Read a value from cfg['hpc'][key] or Config.get().hpc(key, default)."""
    if cfg is not None:
        return cfg.get("hpc", {}).get(key, default)
    return Config.get().hpc(key, default)


def _cfg_hpc_nested(cfg: dict | None, section: str, key: str, default: str = "") -> str:
    """Read cfg['hpc'][section][key] with fallback."""
    if cfg is not None:
        return cfg.get("hpc", {}).get(section, {}).get(key, default)
    return Config.get()._data.get("hpc", {}).get(section, {}).get(key, default)


def _cfg_slurm_time(cfg: dict | None, key: str, default: str = "48:00:00") -> str:
    """Read cfg['slurm_time'][key] or Config.get().slurm_time(key, default)."""
    if cfg is not None:
        return cfg.get("slurm_time", {}).get(key, default)
    return Config.get().slurm_time(key, default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def default_deepmd_input(
    type_map: list[str],
    mlff_dir: Path,
    numb_steps: int,
    cfg: dict | None = None,
) -> dict:
    """Return the deepmd_input.json dict (not written to disk).

    Hyperparameters are read from cfg["mlip_defaults"] when cfg is provided,
    otherwise from Config.get().mlip().
    """
    rcut            = _cfg_mlip(cfg, "rcut",               6.5)
    rcut_smooth     = _cfg_mlip(cfg, "rcut_smooth",         0.5)
    neuron_desc     = _cfg_mlip(cfg, "neuron_descriptor",   [25, 50, 100])
    neuron_fit      = _cfg_mlip(cfg, "neuron_fitting",      [240, 240, 240])
    decay_steps     = _cfg_mlip(cfg, "decay_steps",         5000)
    start_lr        = _cfg_mlip(cfg, "start_lr",            0.001)
    pref_e_start    = _cfg_mlip(cfg, "loss_pref_e_start",   0.02)
    pref_e_limit    = _cfg_mlip(cfg, "loss_pref_e_limit",   1.0)
    pref_f_start    = _cfg_mlip(cfg, "loss_pref_f_start",   1000)
    pref_f_limit    = _cfg_mlip(cfg, "loss_pref_f_limit",   1.0)
    disp_freq       = _cfg_mlip(cfg, "disp_freq",           100)
    save_freq       = _cfg_mlip(cfg, "save_freq",           5000)

    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [25] * len(type_map),
                "rcut_smth": rcut_smooth,
                "rcut": rcut,
                "neuron": neuron_desc,
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": neuron_fit,
                "resnet_dt": True,
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": decay_steps,
            "start_lr": start_lr,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": pref_e_start,
            "limit_pref_e": pref_e_limit,
            "start_pref_f": pref_f_start,
            "limit_pref_f": pref_f_limit,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": ["../00.data/training_data"],
                "batch_size": "auto",
            },
            "validation_data": {
                "systems": ["../00.data/validation_data"],
                "batch_size": 1,
                "numb_btch": 32,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": disp_freq,
            "save_freq": save_freq,
        },
    }


def write_deepmd_input(
    train_dir: Path,
    type_map: list[str],
    mlff_dir: Path,
    numb_steps: int,
    cfg: dict | None = None,
) -> Path:
    """Write train_dir/deepmd_input.json and return the written path."""
    inp = default_deepmd_input(type_map, mlff_dir, numb_steps, cfg=cfg)
    out = train_dir / "deepmd_input.json"
    out.write_text(json.dumps(inp, indent=4))
    return out


def write_deepmd_slurm(
    path: Path,
    project_name: str,
    mlff_dir: Path,
    cfg: dict | None = None,
) -> None:
    """Write SLURM submission script for DeepMD CPU training at *path*."""
    cpu_env   = _cfg_hpc(cfg, "deepmd_cpu_venv", "")
    account   = (_cfg_hpc_nested(cfg, "accounts", "standard", "")
                 or _account_fallback())
    wall      = _cfg_slurm_time(cfg, "mlip_cpu",  "120:00:00")
    cpu_dp    = f"{cpu_env}/bin/dp"
    train_dir = mlff_dir / "01.train"

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --tasks-per-node=104\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={project_name}_mlip\n"
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
    path.write_text(script)
    path.chmod(0o755)


def write_mace_config(
    path: Path,
    project_name: str,
    type_map: list[str],
    mlff_dir: Path,
    cfg: dict | None = None,
) -> None:
    """Write mace_config.yaml at *path*."""
    foundation  = _cfg_mlip(cfg, "mace_foundation",  "mace-mpa-0")
    max_epochs  = _cfg_mlip(cfg, "mace_max_epochs",  200)
    batch       = _cfg_mlip(cfg, "mace_valid_batch",  8)

    lines = [
        f"name: {project_name}",
        f"train_file: {mlff_dir}/00.data",
        f"valid_file: {mlff_dir}/00.data",
        f"foundation_model: {foundation}",
        f"max_num_epochs: {max_epochs}",
        f"batch_size: {batch}",
        f"valid_batch_size: {batch}",
        f"output_dir: {mlff_dir}",
    ]
    if type_map:
        lines.append(f"atomic_numbers: [{', '.join(type_map)}]")

    path.write_text("\n".join(lines) + "\n")


def write_mace_slurm(
    path: Path,
    project_name: str,
    mlff_dir: Path,
    cfg_path: Path,
    cfg: dict | None = None,
) -> None:
    """Write SLURM submission script for MACE GPU training at *path*."""
    py_deep   = _cfg_hpc(cfg, "python_deepmd", "")
    gpu_env   = (
        str(Path(py_deep).parent.parent)
        if py_deep
        else ""
    )
    account   = (_cfg_hpc_nested(cfg, "accounts", "gpu_h100", "")
                 or _account_fallback())
    wall      = _cfg_slurm_time(cfg, "mlip_gpu",  "48:00:00")

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --gpus=2\n"
        "#SBATCH --ntasks-per-node=1\n"
        "#SBATCH --cpus-per-task=16\n"
        f"#SBATCH --time={wall}\n"
        f"#SBATCH --job-name={project_name}_mace\n"
        "#SBATCH --mem=200G\n"
        f"#SBATCH --error={mlff_dir}/%J.stderr\n"
        f"#SBATCH --output={mlff_dir}/%J.stdout\n"
        "module purge\n"
        f"{module_bundle_lines('gpu_md', tolerant=True)}"
        f"source activate {gpu_env}\n"
        f"cd {mlff_dir}\n"
        f"mace-train --config {cfg_path} 2>&1 | tee mace_train.log\n"
    )
    path.write_text(script)
    path.chmod(0o755)


def read_type_map(data_dir: Path) -> list[str]:
    """Read type_map from data_dir/type_map.raw. Returns [] if file not found."""
    tm_file = data_dir / "type_map.raw"
    if not tm_file.exists():
        return []
    return [line.strip() for line in tm_file.read_text().splitlines() if line.strip()]
