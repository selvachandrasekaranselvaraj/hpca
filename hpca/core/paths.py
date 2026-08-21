"""
hpca/core/paths.py — Canonical project directory layout.

hpca canonical project directory layout.

All path construction in hpca imports from here.
To change the layout, update this file FIRST, then update every file listed
in the cross-reference section below.

════════════════════════════════════════════════════════════
Workflow layout (all paths relative to project root):
════════════════════════════════════════════════════════════

  DESIGN
    designed_structures/poscar_dft.vasp      ← input POSCAR for DFT tier
    designed_structures/poscar_mlmd.vasp     ← input POSCAR for MLMD tier
    designed_structures/poscar_cmd.vasp      ← input POSCAR for CMD tier
    dft/preopt/CONTCAR                       ← DFT-tier pre-optimised structure
    preopt/contcar_{mlmd,cmd}_preopt.vasp    ← MLMD/CMD preconditioned structures

  DFT (VASP)
    dft/vc/          ← ISIF=3 variable-cell relax  (starts from dft/preopt/CONTCAR)
    dft/opt/         ← ISIF=2 ionic relax           (starts from dft/vc/CONTCAR)
    aimd/{T}/        ← NVT AIMD at T (K)            (starts from dft/opt/CONTCAR)
                        — multiple temperatures, deformed cells, rattled atoms

  MLMD (ML force field + LAMMPS)
    mlmd/mlff/       ← MACE/DeepMD force field training from DFT data
    mlmd/npt/        ← NPT equilibration at 300 K   (starts from preopt/contcar_mlmd_preopt.vasp)
    mlmd/nvt/{T}/    ← NVT production at T (K)      (starts from mlmd/npt/ final frame)
                        — temperatures: 300,320,340,360,380,400,500,600 K

  CMD (classical force field + LAMMPS)
    cmd/npt/         ← NPT equilibration at 300 K   (starts from preopt/contcar_cmd_preopt.vasp)
    cmd/nvt/{T}/     ← NVT production at T (K)      (starts from cmd/npt/ final frame)
                        — temperatures: 300,320,340,360,380,400,500,600 K

  CONTINUUM
    continuum/       ← PNP / phase-field models fed from mlmd/ and cmd/ analysis

  RESULTS
    results/figures/   ← publication-ready plots
    results/data/      ← processed data (diffusivity, RDF, MSD, …)
    results/reports/   ← HTML reports
    results/manuscript.docx

════════════════════════════════════════════════════════════
Cross-reference — update ALL of these when layout changes:
════════════════════════════════════════════════════════════
  hpca/core/paths.py                          ← THIS FILE
  hpca/core/project.py                        — MaterialProject path properties
  hpca/sim/design.py                          — structure builders
  hpca/sim/dft.py                             — VASP input builders
  hpca/sim/md.py                              — LAMMPS input builders
  hpca/sim/mlip.py                            — MLIP input builders
  hpca/stages/s00_design.py                   — design stage runner
  hpca/stages/s01_dft.py                      — DFT stage runner
  hpca/stages/s02_mlip.py                     — MLIP stage runner
  hpca/stages/s03_md.py                       — MD stage runner
  hpca/tools/lammps.py                        — LAMMPS utility
  hpca/viz/report.py                          — HTML report generator
  hpca/orchestrator/handlers/h00_design.py    — design + preopt handler
  hpca/orchestrator/handlers/h01_dft.py       — DFT vc→opt handler
  hpca/orchestrator/handlers/h02_aimd.py      — AIMD handler
  hpca/orchestrator/handlers/h03_neb.py       — NEB handler
  hpca/orchestrator/handlers/h04_mlip.py      — MLFF training handler
  hpca/orchestrator/handlers/h05_lammps.py    — MLMD NPT→NVT handler
  hpca/orchestrator/handlers/h05_cmd.py       — CMD NPT→NVT handler
  hpca/orchestrator/handlers/h06_analysis.py  — analysis handler
  hpca/orchestrator/handlers/h07_electronic.py — electronic handler
  hpca/orchestrator/handlers/h08_echem.py     — electrochemistry handler
  hpca/orchestrator/handlers/h09_continuum.py — continuum handler
  hpca/orchestrator/handlers/h10_plotting.py  — plotting handler
  hpca/orchestrator/handlers/h11_manuscript.py — manuscript handler
  hpca/orchestrator/handlers/h13_active_learning.py — active learning handler
"""
from __future__ import annotations
from pathlib import Path


def load_platform_config() -> dict:
    """Load platform.yaml — single source of truth for HPC paths and limits.

    Cross-ref: hpca/config/platform.yaml
    Importable by any hpca module via: from hpca.core.paths import load_platform_config
    """
    import yaml
    cfg = Path(__file__).parent.parent / "config" / "platform.yaml"
    if not cfg.exists():
        return {}
    return yaml.safe_load(cfg.read_text()) or {}


# ── Design ────────────────────────────────────────────────────────────────────

def designed_structures(p: Path) -> Path:
    """Return path to the designed_structures/ directory under project root *p*."""
    return p / "designed_structures"

def preopt(p: Path) -> Path:
    """Return path to the preopt/ directory (MACE pre-optimised structures)."""
    return p / "preopt"


# ── DFT (VASP) ────────────────────────────────────────────────────────────────

def dft_base(p: Path) -> Path:
    """Return path to the dft/ base directory under project root *p*."""
    return p / "dft"

def dft_vc(p: Path) -> Path:
    """ISIF=3 variable-cell relaxation. Source: dft/preopt/CONTCAR."""
    return p / "dft" / "vc"

def dft_preopt(p: Path) -> Path:
    """Material-agnostic DFT preoptimization directory."""
    return p / "dft" / "preopt"

def dft_opt(p: Path) -> Path:
    """ISIF=2 ionic relaxation. Source: dft/vc/CONTCAR"""
    return p / "dft" / "opt"

def dft_aimd(p: Path, T: int | str) -> Path:
    """NVT AIMD at temperature T K. Source: dft/opt/CONTCAR"""
    return p / "aimd" / str(T)


# ── MLMD (ML potential + LAMMPS) ──────────────────────────────────────────────

def mlmd_base(p: Path) -> Path:
    """Return path to the mlmd/ base directory under project root *p*."""
    return p / "mlmd"

def mlmd_mlff(p: Path) -> Path:
    """MACE/DeepMD force field training. Source: DFT data (dft/ outputs)"""
    return p / "mlmd" / "mlff"

def mlmd_npt(p: Path, backend: str | None = None) -> Path:
    """NPT equilibration at 300 K. backend prefix used for dual-backend projects."""
    if backend:
        return p / "mlmd" / backend / "npt"
    return p / "mlmd" / "npt"

def mlmd_nvt(p: Path, T: int | str, backend: str | None = None) -> Path:
    """NVT production at T K. backend prefix used for dual-backend projects."""
    if backend:
        return p / "mlmd" / backend / "nvt" / str(T)
    return p / "mlmd" / "nvt" / str(T)


# ── CMD (classical force field + LAMMPS) ──────────────────────────────────────

def cmd_base(p: Path) -> Path:
    """Return path to the cmd/ base directory under project root *p*."""
    return p / "cmd"

def cmd_npt(p: Path) -> Path:
    """NPT equilibration at 300 K. Source: preopt/contcar_cmd_preopt.vasp"""
    return p / "cmd" / "npt"

def cmd_nvt(p: Path, T: int | str) -> Path:
    """NVT production at T K. Source: cmd/npt/ final frame"""
    return p / "cmd" / "nvt" / str(T)


# ── Continuum ─────────────────────────────────────────────────────────────────

def continuum_base(p: Path) -> Path:
    """PNP / electrochemical continuum model outputs. Source: mlmd/ + cmd/ analysis"""
    return p / "continuum"


# ── Results ───────────────────────────────────────────────────────────────────

def results_base(p: Path) -> Path:
    """Return path to the results/ base directory."""
    return p / "results"

def results_figures(p: Path) -> Path:
    """Return path to the results/figures/ directory for publication-ready plots."""
    return p / "results" / "figures"

def results_data(p: Path) -> Path:
    """Return path to the results/data/ directory for processed analysis data."""
    return p / "results" / "data"

def results_reports(p: Path) -> Path:
    """Return path to the results/reports/ directory for HTML reports."""
    return p / "results" / "reports"

def results_manuscript(p: Path) -> Path:
    """Return path to results/manuscript.docx."""
    return p / "results" / "manuscript.docx"


# ── Convenience: POSCAR / CONTCAR paths ───────────────────────────────────────

def poscar_dft(p: Path) -> Path:
    """Return path to designed_structures/poscar_dft.vasp — DFT-tier input structure."""
    return p / "designed_structures" / "poscar_dft.vasp"

def poscar_mlmd(p: Path) -> Path:
    """Return path to designed_structures/poscar_mlmd.vasp — MLMD-tier input structure."""
    return p / "designed_structures" / "poscar_mlmd.vasp"

def poscar_cmd(p: Path) -> Path:
    """Return path to designed_structures/poscar_cmd.vasp — CMD-tier input structure."""
    return p / "designed_structures" / "poscar_cmd.vasp"

def contcar_preopt(p: Path, tier: str) -> Path:
    """tier: 'dft', 'mlmd', or 'cmd'. Written by h00_design (MACE preopt)."""
    if tier == "dft":
        return dft_preopt(p) / "CONTCAR"
    return p / "preopt" / f"contcar_{tier}_preopt.vasp"

def pot_com_pb(p: Path) -> Path:
    """DeepMD potential file produced by mlmd/mlff/ training."""
    return p / "mlmd" / "mlff" / "pot_com.pb"
