"""Generate the minimal site wrapper for the packaged daemon."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from hpca.core.atomic import atomic_write_text


def write_wrapper(path: Path, *, inbox: Path, account: str, python: str | None = None) -> Path:
    """Write a ten-day wrapper; the service submits its successor with 20 h remaining."""
    executable = python or sys.executable
    text = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --job-name=hpca-daemon
#SBATCH --output={inbox}/logs/daemon_%J.stdout
#SBATCH --error={inbox}/logs/daemon_%J.stderr
#SBATCH --signal=B:USR1@3600
set -euo pipefail
if [[ -n "${{SLURM_JOB_ID:-}}" ]]; then
  echo "$SLURM_JOB_ID" > {inbox}/.daemon_job_id
fi
exec {shlex.quote(executable)} -m hpca.daemon.cli --inbox {shlex.quote(str(inbox))} run --successor-script {shlex.quote(str(path))}
"""
    atomic_write_text(path, text, mode=0o750)
    return path
