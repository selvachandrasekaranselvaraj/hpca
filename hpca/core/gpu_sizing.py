"""gpu_sizing.py — GPU resource sizing for LAMMPS/MACE MLMD jobs.

Reads the gpu_sizing table from platform.yaml (section: gpu_sizing).
Each entry: {max_mb: float, gpus: int, ntasks: int, cpus_per_task: int, mem_gb: int}
The last entry should have max_mb: 999999 or null as the catch-all.

For MACE training jobs, reads defaults from platform.yaml mlip_defaults
(keys: mace_train_gpus, mace_train_ntasks, mace_train_cpus_per_task,
mace_train_mem_gb) with safe fallbacks to the values historically used
in h04_mlip.py (gpus=2, ntasks=1, cpus_per_task=16, mem_gb=200).
"""
from __future__ import annotations

import logging
from pathlib import Path

__all__ = [
    "get_lammps_gpu_resources",
    "get_mace_gpu_resources",
]

log = logging.getLogger("hpca.core")

# ── fallback constants (mirror h04_mlip.py hard-codes) ───────────────────────
_FALLBACK_MACE = {
    "gpus": 2,
    "ntasks": 1,
    "cpus_per_task": 16,
    "mem_gb": 200,
}

# Final-tier fallback when Config is unavailable and pot_path doesn't exist
_FALLBACK_LAMMPS_GENEROUS = {
    "gpus": 4,
    "ntasks": 2,
    "cpus_per_task": 8,
    "mem_gb": 200,
}


def _get_config():
    """Return Config singleton or None if unavailable."""
    try:
        from hpca.core.config import Config
        return Config.get()
    except Exception as exc:
        log.debug("gpu_sizing: Config unavailable (%s), using fallback", exc)
        return None


def get_lammps_gpu_resources(pot_path: Path) -> dict:
    """Return GPU resource dict for a LAMMPS MLMD job based on potential file size.

    Looks up the gpu_sizing table in platform.yaml by comparing the file size
    of *pot_path* (in MB) against each rule's max_mb threshold.  The first
    rule whose max_mb >= file_size_mb is used.

    Parameters
    ----------
    pot_path:
        Path to the potential file (e.g. ``pot_com.pb`` for DeepMD, or a
        MACE model checkpoint).  If the file does not exist the most-generous
        tier (last entry in the table) is returned as a safe default.

    Returns
    -------
    dict
        Keys: ``gpus``, ``ntasks``, ``cpus_per_task``, ``mem_gb``.
    """
    cfg = _get_config()

    # Determine file size in MB
    if cfg is None:
        return dict(_FALLBACK_LAMMPS_GENEROUS)

    rules = cfg.raw.get("gpu_sizing", [])
    if not rules:
        log.warning("gpu_sizing: no gpu_sizing rules in platform.yaml; using fallback")
        return dict(_FALLBACK_LAMMPS_GENEROUS)

    if not pot_path.exists():
        # File absent → use the most-generous tier (last entry in the table)
        log.warning(
            "gpu_sizing: potential file not found (%s); using most-generous tier",
            pot_path,
        )
        last = rules[-1]
        resources = {k: last.get(k, v) for k, v in _FALLBACK_LAMMPS_GENEROUS.items()}
        log.debug("gpu_sizing: most-generous tier → %s", resources)
        return resources

    size_mb = pot_path.stat().st_size / (1024 * 1024)
    log.debug("gpu_sizing: %s → %.1f MB", pot_path.name, size_mb)

    for rule in rules:
        max_mb = rule.get("max_mb", None)
        if max_mb is None:
            max_mb = float("inf")
        if size_mb <= max_mb:
            resources = {k: rule[k] for k in ("gpus", "ntasks", "cpus_per_task", "mem_gb")}
            log.debug(
                "gpu_sizing: size_mb=%.1f matched tier max_mb=%s → %s",
                size_mb, max_mb, resources,
            )
            return resources

    # Should not be reached if the table has a catch-all, but be safe
    last = rules[-1]
    resources = {k: last.get(k, v) for k, v in _FALLBACK_LAMMPS_GENEROUS.items()}
    log.warning("gpu_sizing: fell through all rules, using last entry: %s", resources)
    return resources


def get_mace_gpu_resources() -> dict:
    """Return GPU resources for a MACE training job from platform.yaml.

    Reads the following keys from the ``mlip_defaults`` section of
    platform.yaml (all optional — falls back to the values historically
    used in h04_mlip.py if absent):

    * ``mace_train_gpus``       — number of GPUs        (default 2)
    * ``mace_train_ntasks``     — ``--ntasks-per-node``  (default 1)
    * ``mace_train_cpus_per_task`` — CPUs per task       (default 16)
    * ``mace_train_mem_gb``     — memory in GB           (default 200)

    Returns
    -------
    dict
        Keys: ``gpus``, ``ntasks``, ``cpus_per_task``, ``mem_gb``.
    """
    cfg = _get_config()
    if cfg is None:
        log.debug("get_mace_gpu_resources: Config unavailable, using fallback")
        return dict(_FALLBACK_MACE)

    return {
        "gpus":          cfg.mlip("mace_train_gpus",          _FALLBACK_MACE["gpus"]),
        "ntasks":        cfg.mlip("mace_train_ntasks",         _FALLBACK_MACE["ntasks"]),
        "cpus_per_task": cfg.mlip("mace_train_cpus_per_task",  _FALLBACK_MACE["cpus_per_task"]),
        "mem_gb":        cfg.mlip("mace_train_mem_gb",         _FALLBACK_MACE["mem_gb"]),
    }
