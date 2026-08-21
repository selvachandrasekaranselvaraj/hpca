"""Canonical directory and file registry.

Every directory and key file produced or consumed by the pipeline is defined
here as a Path-returning function.  Handlers must import from here rather than
constructing paths inline, so the layout can be changed in one place.

Usage:
    from hpca.registry.folder import (
        # h01
        dft_aimd_relax, dft_bader, dft_dos_scf, dft_dos_nonscf,
        dft_static, dft_echem_static,
        # h02
        aimd_base, aimd_npt_liquid, aimd_npt_sse, aimd_dataset_box,
        # h03
        neb_base, neb_path_dir, neb_preopt_aimd,
        neb_preopt_vcrelax, neb_preopt_atomopt,
        # h04
        mlff_data_dir, mlff_train_dir, mlff_pot, mlff_mace_model,
        # h05_cmd
        cmd_data_file, cmd_npt_nvt_start,
        # h05_lammps
        mlmd_npt_nvt_start,
        # h06
        results_msd, results_rdf, results_transport,
        results_coordination, results_vanhove,
        results_ion_pairs, results_haven_ratio, results_vacf,
        results_sei_kinetics, results_q6_order,
        results_ion_hopping, results_lindemann, results_phase_transitions,
        # h07
        bader_acf_dat,
        # h08
        results_echem,
        # h09
        results_continuum,
        # h10
        figure_msd, figure_arrhenius, figure_rdf, figure_dos,
        figure_neb, figure_van_hove, figure_coordination,
        figure_ion_pairs, figure_bader, figure_ocv,
        # h12
        chaai_data_dir,
        # h13
        al_aimd_dir,
        # completion sentinel helper
        stage_outputs,
    )

All functions follow the convention:
    f(project_dir: Path, *args) -> Path

where project_dir is the sub-project root (e.g. /path/to/workspace/proj/EC_LiPF6/).

Paths already defined in hpca.core.paths are NOT duplicated here.
Import both modules when you need the complete layout:
    from hpca.core import paths
    from hpca.registry import folder as fr
"""
from __future__ import annotations

from pathlib import Path

# ── h00_design ─────────────────────────────────────────────────────────────────
# (designed_structures/, preopt/ already in hpca.core.paths)

def poscar_dft(p: Path) -> Path:
    """POSCAR for DFT box (~200 atoms).  Written by h00_design."""
    return p / "designed_structures" / "poscar_dft.vasp"

def poscar_mlmd(p: Path) -> Path:
    """POSCAR for MLMD box (~6000 atoms).  Written by h00_design."""
    return p / "designed_structures" / "poscar_mlmd.vasp"

def poscar_cmd(p: Path) -> Path:
    """POSCAR for CMD box (~60000 atoms).  Written by h00_design."""
    return p / "designed_structures" / "poscar_cmd.vasp"

def cmd_data_lammps(p: Path, name: str | None = None) -> Path:
    """LAMMPS .data file for CMD (OPLS-AA system). name defaults to project name."""
    filename = f"{name}.data" if name else "system_cmd.data"
    return p / "preopt" / filename

def contcar_dft_preopt(p: Path) -> Path:
    """Material-agnostic DFT preoptimization output."""
    return p / "dft" / "preopt" / "CONTCAR"

def contcar_mlmd_preopt(p: Path) -> Path:
    """MACE pre-optimised MLMD POSCAR.  Written by h00_design preopt step."""
    return p / "preopt" / "contcar_mlmd_preopt.vasp"

def contcar_cmd_preopt(p: Path) -> Path:
    """MACE pre-optimised CMD POSCAR.  Written by h00_design preopt step."""
    return p / "preopt" / "contcar_cmd_preopt.vasp"

def preoptimization_policy(p: Path) -> Path:
    """Auditable per-tier preoptimization decisions."""
    return p / "preopt" / "policy.json"


# ── h01_dft ────────────────────────────────────────────────────────────────────
# (dft_base, dft_vc, dft_opt already in hpca.core.paths)

def dft_aimd_relax(p: Path) -> Path:
    """Short NPT pre-equilibration for doped SSE before vc_relax.  [SSE doped only]"""
    return p / "dft" / "aimd_relax"

def dft_bader(p: Path) -> Path:
    """Bader charge analysis static calculation.  [SSE·INT·MOL(liquid)]"""
    return p / "dft" / "bader"

def dft_dos(p: Path) -> Path:
    """DOS / PDOS base directory (contains scf/ and nonscf/).  [SSE·INT]"""
    return p / "dft" / "dos"

def dft_dos_scf(p: Path) -> Path:
    """SCF calculation for DOS.  [SSE·INT]"""
    return p / "dft" / "dos" / "scf"

def dft_dos_nonscf(p: Path) -> Path:
    """Non-SCF PDOS calculation.  [SSE·INT]"""
    return p / "dft" / "dos" / "nonscf"

def dft_static(p: Path) -> Path:
    """Static VASP calculation (IBRION=-1, NSW=0).  [SSE]"""
    return p / "dft" / "static"

def dft_echem_static(p: Path) -> Path:
    """Electrochemistry static calculation.  [SSE]"""
    return p / "dft" / "echem_static"

# Key files produced by h01_dft
def dft_opt_contcar(p: Path) -> Path:
    """Relaxed geometry — primary input for h02, h03, h07, h08."""
    return p / "dft" / "opt" / "CONTCAR"

def dft_vc_contcar(p: Path) -> Path:
    """Variable-cell relaxed geometry — input for dft/opt."""
    return p / "dft" / "vc" / "CONTCAR"

def bader_acf_dat(p: Path) -> Path:
    """Bader atomic charge file (Henkelman code output).  Written by h07_electronic."""
    return p / "dft" / "bader" / "ACF.dat"


# ── h02_aimd ───────────────────────────────────────────────────────────────────
# (dft_aimd(p, T) already in hpca.core.paths for NVT boxes at aimd/{T}/)

def aimd_base(p: Path) -> Path:
    """Root AIMD directory.  Used as composition_dir for SSE."""
    return p / "aimd"

def aimd_shared_potcar(p: Path) -> Path:
    """Shared POTCAR for all AIMD boxes.  Written once by h02_aimd [SSE]."""
    return p / "aimd" / "POTCAR"

def aimd_npt_liquid(p: Path, T: int | str) -> Path:
    """NPT Step 0 for liquid/molecular box equilibration at T K.  [MOL only]

    Layout: aimd/{T}/NPT/
    """
    return p / "aimd" / str(T) / "NPT"


def aimd_npt_sse(p: Path) -> Path:
    """NPT Step 0 for crystalline/SSE box equilibration at 300 K.  [SSE/INT]

    Layout: aimd/NPT/  (no temperature prefix — SSE NPT is always 300 K).
    """
    return p / "aimd" / "NPT"

def aimd_dataset_box(p: Path, box_name: str) -> Path:
    """One deformed/random dataset box.  [SSE/ALL]
    Layout: aimd/dataset/{box_name}/   e.g. aimd/dataset/d0.90_300K/
    """
    return p / "aimd" / "dataset" / box_name

def aimd_dataset_dir(p: Path) -> Path:
    """Parent of all dataset boxes.  [ALL]"""
    return p / "aimd" / "dataset"

def aimd_nvt_xdatcar(p: Path, T: int | str) -> Path:
    """XDATCAR from NVT run at T K.  Read by h04_mlip for training data."""
    return p / "aimd" / str(T) / "XDATCAR"


# ── h03_neb ────────────────────────────────────────────────────────────────────

def neb_base(p: Path) -> Path:
    """Root NEB directory.  [SSE·INT1·INT2]"""
    return p / "neb"

def neb_path_dir(p: Path, path_label: str = "path_a") -> Path:
    """One migration path directory.  Multiple paths → path_a, path_b, …"""
    return p / "neb" / path_label

def neb_endpoint_dir(p: Path, path_label: str, tag: str) -> Path:
    """Endpoint VASP calculation.  tag: '00' (initial) or '09' (final)."""
    return p / "neb" / path_label / tag

def neb_image_dir(p: Path, path_label: str, image_idx: int) -> Path:
    """One NEB image directory.  image_idx: 1-based index."""
    return p / "neb" / path_label / f"{image_idx:02d}"

def neb_preopt(p: Path) -> Path:
    """Pre-NEB optimisation root directory (for doped SSE structures)."""
    return p / "neb" / "preopt"

def neb_preopt_aimd(p: Path) -> Path:
    """AIMD pre-equilibration before vc-relax (300 K NVT).  [SSE doped]"""
    return p / "neb" / "preopt" / "aimd"

def neb_preopt_vcrelax(p: Path) -> Path:
    """vc-relax (ISIF=3) from AIMD snapshot.  [SSE doped]"""
    return p / "neb" / "preopt" / "vcrelax"

def neb_preopt_atomopt(p: Path) -> Path:
    """Atom-only optimization (ISIF=2) from vc-relax CONTCAR.  [SSE doped]"""
    return p / "neb" / "preopt" / "atomopt"


# ── h04_mlip ───────────────────────────────────────────────────────────────────
# (mlmd_mlff already in hpca.core.paths for mlmd/mlff/)

def mlff_data_dir(p: Path) -> Path:
    """DeepMD dataset directory.  Contains train/ and validation/ sets."""
    return p / "mlmd" / "mlff" / "00.data"

def mlff_train_dir(p: Path) -> Path:
    """DeepMD training directory.  Contains deepmd_input.json and outputs."""
    return p / "mlmd" / "mlff" / "01.train"

def mlff_dataset_data(p: Path) -> Path:
    """Raw VASP outputs collected for training (before train/val split)."""
    return p / "mlmd" / "mlff" / "dataset_data"

def mlff_pot(p: Path) -> Path:
    """DeepMD compressed potential.  Written by h04_mlip, read by h05_lammps."""
    return p / "mlmd" / "mlff" / "pot_com.pb"

def mlff_mace_model(p: Path) -> Path:
    """MACE fine-tuned model.  Written by h04_mlip (MACE path)."""
    return p / "mlmd" / "mlff" / "MACE_model.pt"

def mlff_lcurve(p: Path) -> Path:
    """DeepMD learning curve.  Parsed by h04_mlip for convergence check."""
    return p / "mlmd" / "mlff" / "01.train" / "lcurve.out"

def mlff_test_results(p: Path) -> Path:
    """dp test output.  Parsed by h04_mlip / h13_active_learning."""
    return p / "mlmd" / "mlff" / "test_results.txt"


# ── h05_cmd (OPLS-AA classical MD) ────────────────────────────────────────────
# (cmd_base, cmd_npt, cmd_nvt already in hpca.core.paths)

def cmd_data_file(p: Path, name: str | None = None) -> Path:
    """LAMMPS .data file (topology + charges).  Written by h05_cmd setup."""
    filename = f"{name}.data" if name else "system_cmd.data"
    return p / "cmd" / filename

def cmd_npt_nvt_start(p: Path) -> Path:
    """Equilibrated frame for NVT start.  Written by h05_cmd NPT stage."""
    return p / "cmd" / "npt" / "nvt_start.dat"

def cmd_nvt_dump(p: Path, T: int | str) -> Path:
    """NVT LAMMPS dump file.  Written by h05_cmd NVT stage."""
    return p / "cmd" / "nvt" / str(T) / "dump.lammpstrj"

def cmd_nvt_log(p: Path, T: int | str) -> Path:
    """NVT LAMMPS log file.  Written by h05_cmd NVT stage."""
    return p / "cmd" / "nvt" / str(T) / "log.lammps"


# ── h05_lammps (DeepMD/MACE MLMD) ────────────────────────────────────────────
# (mlmd_npt, mlmd_nvt already in hpca.core.paths)

def mlmd_npt_nvt_start(p: Path) -> Path:
    """Equilibrated frame for MLMD NVT start.  Written by h05_lammps NPT stage."""
    return p / "mlmd" / "npt" / "nvt_start.dat"

def mlmd_nvt_dump(p: Path, T: int | str) -> Path:
    """MLMD NVT dump file.  Written by h05_lammps NVT stage."""
    return p / "mlmd" / "nvt" / str(T) / "dump.lammpstrj"

def mlmd_nvt_log(p: Path, T: int | str) -> Path:
    """MLMD NVT log file.  Written by h05_lammps NVT stage."""
    return p / "mlmd" / "nvt" / str(T) / "log.lammps"


# ── h06_analysis ───────────────────────────────────────────────────────────────
# (results_data already in hpca.core.paths for results/data/)

def results_msd(p: Path) -> Path:
    """MSD + diffusivity results.  Written by h06_analysis."""
    return p / "results" / "data" / "msd.json"

def results_rdf(p: Path) -> Path:
    """Radial distribution functions g(r).  Written by h06_analysis."""
    return p / "results" / "data" / "rdf.json"

def results_transport(p: Path) -> Path:
    """Transport properties (D, Ea, σ, t+).  Written by h06_analysis."""
    return p / "results" / "data" / "transport.json"

def results_coordination(p: Path) -> Path:
    """Coordination numbers vs T.  Written by h06_analysis."""
    return p / "results" / "data" / "coordination.json"

def results_vanhove(p: Path) -> Path:
    """Van Hove self-correlation Gs(r,t).  Written by h06_analysis."""
    return p / "results" / "data" / "vanhove.json"

def results_ion_pairs(p: Path) -> Path:
    """Ion-pair fractions (CIP/SSIP/AGG).  Written by h06_analysis."""
    return p / "results" / "data" / "ion_pairs.json"

def results_haven_ratio(p: Path) -> Path:
    """Haven ratio H_R = D_tracer / D_charge.  Written by h06_analysis."""
    return p / "results" / "data" / "haven_ratio.json"

def results_vacf(p: Path) -> Path:
    """Velocity auto-correlation function + VDOS.  Written by h06_analysis."""
    return p / "results" / "data" / "vacf.json"

def results_sei_kinetics(p: Path) -> Path:
    """SEI growth kinetics (power-law fit N_SEI(t) = A·t^n).  Written by h06_analysis."""
    return p / "results" / "data" / "sei_kinetics.json"

def results_q6_order(p: Path) -> Path:
    """Q6 bond-orientational order parameter and crystallinity phase label.  Written by h06_analysis."""
    return p / "results" / "data" / "q6_order.json"

def results_ion_hopping(p: Path) -> Path:
    """Ion hopping events: counts, rates, and correlation times.  Written by h06_analysis."""
    return p / "results" / "data" / "ion_hopping_stats.json"

def results_lindemann(p: Path) -> Path:
    """Lindemann melting criterion δ = u_rms / d_nn.  Written by h06_analysis."""
    return p / "results" / "data" / "lindemann.json"

def results_phase_transitions(p: Path) -> Path:
    """Structural phase transitions detected via sliding-window Q6 change-point analysis.

    Written last by h06_analysis — used as the h06 stage-completion sentinel.
    """
    return p / "results" / "data" / "phase_transitions.json"


# ── h07_electronic ─────────────────────────────────────────────────────────────

def results_electronic(p: Path) -> Path:
    """Electronic properties (DOS, band gap, Bader).  Written by h07_electronic."""
    return p / "results" / "data" / "electronic.json"


# ── h08_echem ──────────────────────────────────────────────────────────────────

def results_echem(p: Path) -> Path:
    """Electrochemical properties (OCV, ECW, EAH, Ef).  Written by h08_echem."""
    return p / "results" / "data" / "echem.json"


# ── h09_continuum ──────────────────────────────────────────────────────────────
# (continuum_base already in hpca.core.paths)

def results_continuum(p: Path) -> Path:
    """Continuum model outputs (σ(T), VTF, KJMA, …).  Written by h09_continuum."""
    return p / "results" / "data" / "continuum.json"


# ── h10_plotting ───────────────────────────────────────────────────────────────
# (results_figures already in hpca.core.paths)

def figure_msd(p: Path) -> Path:
    """MSD curves (multi-panel, one per temperature).  Written by h10_plotting."""
    return p / "results" / "figures" / "msd.html"

def figure_arrhenius(p: Path) -> Path:
    """Arrhenius D(T) fit with Ea annotation.  Written by h10_plotting (last figure)."""
    return p / "results" / "figures" / "arrhenius.html"

def figure_rdf(p: Path) -> Path:
    """RDF g(r) heatmap for all ion pairs.  Written by h10_plotting."""
    return p / "results" / "figures" / "rdf.html"

def figure_dos(p: Path) -> Path:
    """DOS / PDOS plot.  Written by h10_plotting [SSE·INT]."""
    return p / "results" / "figures" / "dos.html"

def figure_neb(p: Path) -> Path:
    """NEB migration barrier curve.  Written by h10_plotting [SSE·INT]."""
    return p / "results" / "figures" / "neb_barrier.html"

def figure_van_hove(p: Path) -> Path:
    """Van Hove self-correlation G_s(r,t) curves.  Written by h10_plotting."""
    return p / "results" / "figures" / "van_hove.html"

def figure_coordination(p: Path) -> Path:
    """Coordination number vs temperature.  Written by h10_plotting."""
    return p / "results" / "figures" / "coordination.html"

def figure_ion_pairs(p: Path) -> Path:
    """CIP/SSIP/AGG/free ion-pair fraction vs temperature.  Written by h10_plotting."""
    return p / "results" / "figures" / "ion_pairs.html"

def figure_bader(p: Path) -> Path:
    """Bader charge map.  Written by h10_plotting [SSE·INT1·INT2]."""
    return p / "results" / "figures" / "bader.html"

def figure_ocv(p: Path) -> Path:
    """Open-circuit voltage vs Li content.  Written by h10_plotting [SSE]."""
    return p / "results" / "figures" / "ocv.html"


# ── h11_manuscript ─────────────────────────────────────────────────────────────
# (results_manuscript already in hpca.core.paths)

def manuscript_docx(p: Path) -> Path:
    """Auto-generated DOCX report.  Written by h11_manuscript."""
    return p / "results" / "manuscript.docx"


# ── h12_chaai ──────────────────────────────────────────────────────────────────

def chaai_data_dir(p: Path) -> Path:
    """AI training data collection directory.  Written by h12_chaai."""
    return p / "chaai_data"


# ── h13_active_learning ────────────────────────────────────────────────────────
# Reuses dft_aimd(p, f"al_{T}") from hpca.core.paths for the AIMD directories.

def al_aimd_dir(p: Path, T: int | str) -> Path:
    """Active-learning AIMD box at temperature T.
    Layout: aimd/al_{T}/   e.g. aimd/al_300/
    Consistent with dft_aimd(p, f'al_{T}') from hpca.core.paths.
    """
    return p / "aimd" / f"al_{T}"


# ── Stage completion sentinels (what each handler produces last) ───────────────

def stage_outputs(handler: str, p: Path) -> list[Path]:
    """Return the canonical output files that signal a handler stage is done.

    Use these as completion checks:
        if all(f.exists() for f in stage_outputs("h01_dft.opt", project_dir)):
            # opt is done
    """
    _map: dict[str, list[Path]] = {
        # h00
        "h00_design":         [contcar_dft_preopt(p), contcar_mlmd_preopt(p)],
        # h01 subtasks
        "h01_dft.vc_relax":   [dft_vc_contcar(p)],
        "h01_dft.opt":        [dft_opt_contcar(p)],
        "h01_dft.bader":      [bader_acf_dat(p)],
        "h01_dft.dos_nonscf": [dft_dos_nonscf(p) / "OUTCAR"],
        "h01_dft.static":     [dft_static(p) / "OUTCAR"],
        # h02
        "h02_aimd":           [aimd_dataset_dir(p)],
        # h03
        "h03_neb":            [neb_base(p) / "pipeline_state.json"],
        # h04
        "h04_mlip":           [mlff_pot(p)],
        # h05
        "h05_cmd":            [cmd_npt_nvt_start(p)],
        "h05_lammps":         [mlmd_npt_nvt_start(p)],
        # h06 — sentinel is the last file written (phase_transitions); transport is written earlier
        "h06_analysis":       [results_transport(p), results_phase_transitions(p)],
        # h07
        "h07_electronic":     [results_electronic(p)],
        # h08
        "h08_echem":          [results_echem(p)],
        # h09
        "h09_continuum":      [results_continuum(p)],
        # h10
        "h10_plotting":       [figure_arrhenius(p)],
        # h11
        "h11_manuscript":     [manuscript_docx(p)],
        # h13
        "h13_active_learning": [mlff_pot(p)],
    }
    return _map.get(handler, [])
