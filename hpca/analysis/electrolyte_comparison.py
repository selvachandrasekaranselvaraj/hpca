"""
electrolyte_comparison.py — Cross-combination comparative analysis for combinatorial projects.

Computes and compares across electrolyte combinations:
  D_Li      — Li+ diffusion coefficient (MSD from MLMD)
  Ea        — Activation energy (Arrhenius fit over temperatures)
  sigma     — Ionic conductivity (Nernst-Einstein)
  t+        — Li+ transference number (D_Li / (D_Li + D_anion))
  CN(Li-O)  — First solvation shell coordination number
  CN(Li-F)  — Anion contact (FSI/TFSI)
  ESW       — Electrochemical stability window (HOMO/LUMO from DFT)
"""
from __future__ import annotations

import csv
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


def _cfmt(c: float) -> str:
    """Format a float concentration for directory names (e.g. 0.25 → '0p25')."""
    return str(c).replace(".", "p")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComboResult:
    """Per-combination analysis results container for cross-electrolyte comparison."""

    name: str
    label: str
    salt_conc_M: float

    D_Li: Optional[float] = None        # m²/s at T_ref
    D_anion: Optional[float] = None     # m²/s at T_ref
    Ea_eV: Optional[float] = None       # eV from Arrhenius
    sigma_Scm: Optional[float] = None   # S/cm Nernst-Einstein
    t_plus: Optional[float] = None      # Li+ transference number

    CN_Li_O: Optional[float] = None     # coordination: Li-O
    CN_Li_F: Optional[float] = None     # coordination: Li-F
    CN_Li_N: Optional[float] = None     # coordination: Li-N

    HOMO_eV: Optional[float] = None
    LUMO_eV: Optional[float] = None
    ESW_V: Optional[float] = None

    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(proj, output_dir: Optional[Path] = None) -> Dict[str, ComboResult]:
    """
    Analyse all combinations in a combinatorial project and generate comparison figures.

    Usage:
        from hpca.analysis.electrolyte_comparison import run_comparison
        from hpca.core.project import ProjectRegistry
        proj = ProjectRegistry.from_project_yaml("project.yaml")
        results = run_comparison(proj)
    """
    if not proj.is_combinatorial:
        raise ValueError(
            f"{proj.name} has no combinations defined "
            "(set aimd_combinations in project.yaml with 2+ entries)"
        )

    if output_dir is None:
        output_dir = proj.root / "comparison"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = proj.extras.get("simulation", {})
    concs   = sim_cfg.get("salt_concs_M", [1.0])
    T_ref   = proj.T_ref

    results: Dict[str, ComboResult] = {}

    for combo in proj.combinations:
        cname = combo["name"]
        label = combo.get("label", cname)

        for conc in concs:
            key = f"{cname}@{conc}M"
            result = ComboResult(name=cname, label=label, salt_conc_M=conc)

            dump_path = (proj.root / cname / f"dlmd/{_cfmt(conc)}M/{T_ref}K"
                         / "dump_unwrapped.lmp")

            if dump_path.exists():
                _compute_transport(result, dump_path, proj)
                _compute_coordination(result, dump_path, proj)
            else:
                result.errors.append(f"no MLMD dump: {dump_path}")

            _compute_arrhenius(result, proj, cname, conc, sim_cfg)
            _load_esw(result, proj.root / cname)

            results[key] = result
            _print_result(result)

    _save_csv(results, output_dir)
    _plot_dashboard(results, output_dir)
    _save_json(results, output_dir)

    print(f"\n  [comparison] Results in {output_dir}")
    return results


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _compute_transport(result: ComboResult, dump_path: Path, proj) -> None:
    """Compute D_Li, D_anion, t_plus, and Nernst-Einstein sigma from MLMD dump; write into result."""
    try:
        from hpca.analysis.trajectory import parse_trajectory, get_mobile_indices
        from hpca.analysis.msd import compute_msd, fit_diffusivity

        traj = parse_trajectory(dump_path)
        atom_types = np.array(traj["atom_types"])
        positions  = traj["positions"]
        box        = traj["box"][0]

        # Li+ diffusivity
        li_mask = atom_types == proj.mobile_ion
        pos_li  = positions[:, li_mask, :]
        msd, times = compute_msd(pos_li, dt_ps=0.001)
        D_Li, r2 = fit_diffusivity(times, msd)
        if r2 > 0.85:
            result.D_Li = D_Li

        # Anion diffusivity — use heaviest non-solvent element (S for FSI/TFSI, P for PF6)
        anion_candidates = set(atom_types) - {proj.mobile_ion, "O", "C", "H"}
        priority = ["P", "S", "N", "B", "F"]
        anion_elem = next((e for e in priority if e in anion_candidates), None)
        if anion_elem:
            anion_mask = atom_types == anion_elem
            pos_anion  = positions[:, anion_mask, :]
            msd_a, times_a = compute_msd(pos_anion, dt_ps=0.001)
            D_anion, r2_a = fit_diffusivity(times_a, msd_a)
            if r2_a > 0.80:
                result.D_anion = D_anion

        if result.D_Li and result.D_anion:
            result.t_plus = result.D_Li / (result.D_Li + result.D_anion)

        # Nernst-Einstein conductivity
        n_Li = int(np.sum(li_mask))
        V_A3 = np.prod([box[i, 1] - box[i, 0] for i in range(3)])
        V_m3 = V_A3 * 1e-30
        T    = proj.T_ref
        kB   = 1.380649e-23
        e    = 1.602176634e-19
        if result.D_Li and V_m3 > 0 and T > 0:
            D_total = result.D_Li + (result.D_anion or result.D_Li / 0.3)
            sigma_Sm = (n_Li * e**2 / (V_m3 * kB * T)) * D_total
            result.sigma_Scm = sigma_Sm / 100.0

    except Exception as exc:
        result.errors.append(f"transport: {exc}")


# ---------------------------------------------------------------------------
# Coordination (solvation shell)
# ---------------------------------------------------------------------------

def _compute_coordination(result: ComboResult, dump_path: Path, proj) -> None:
    """Compute Li-O, Li-F, and Li-N coordination numbers from the MLMD trajectory."""
    try:
        from hpca.analysis.trajectory import parse_trajectory
        from hpca.analysis.coordination import compute_coordination_number

        traj = parse_trajectory(dump_path)
        atom_types = np.array(traj["atom_types"])
        positions  = traj["positions"]
        box        = traj["box"][0]
        box_len    = np.array([box[i, 1] - box[i, 0] for i in range(3)])

        type_idx = {t: np.where(atom_types == t)[0] for t in set(atom_types)}

        pairs = [("O", "CN_Li_O", 3.0),
                 ("F", "CN_Li_F", 2.5),
                 ("N", "CN_Li_N", 2.8)]
        for elem, attr, rcut in pairs:
            if elem in type_idx:
                CN = compute_coordination_number(
                    positions, proj.mobile_ion, elem,
                    r_cutoff=rcut, box=box_len, type_indices=type_idx
                )
                setattr(result, attr, float(np.mean(CN)))

    except Exception as exc:
        result.errors.append(f"coordination: {exc}")


# ---------------------------------------------------------------------------
# Arrhenius fit across temperatures
# ---------------------------------------------------------------------------

def _compute_arrhenius(result: ComboResult, proj, cname: str,
                       conc: float, sim_cfg: dict) -> None:
    """Fit Arrhenius Ea from multi-temperature MLMD dumps and store in result.Ea_eV."""
    try:
        from hpca.analysis.trajectory import parse_trajectory, get_mobile_indices
        from hpca.analysis.msd import compute_msd, fit_diffusivity, arrhenius_fit

        mlmd_temps = sim_cfg.get("mlmd_temps", [])
        if len(mlmd_temps) < 2:
            return

        temps_K, D_vals = [], []
        for T in mlmd_temps:
            dump = (proj.root / cname / f"dlmd/{_cfmt(conc)}M/{T}K"
                    / "dump_unwrapped.lmp")
            if not dump.exists():
                continue
            try:
                traj = parse_trajectory(dump)
                at   = np.array(traj["atom_types"])
                pos  = traj["positions"][:, at == proj.mobile_ion, :]
                msd, times = compute_msd(pos, dt_ps=0.001)
                D, r2 = fit_diffusivity(times, msd)
                if D and r2 > 0.90:
                    temps_K.append(T); D_vals.append(D)
            except Exception:
                continue

        if len(temps_K) >= 2:
            Ea, _ = arrhenius_fit(temps_K, D_vals)
            result.Ea_eV = Ea

    except Exception as exc:
        result.errors.append(f"arrhenius: {exc}")


# ---------------------------------------------------------------------------
# ESW from DFT
# ---------------------------------------------------------------------------

def _load_esw(result: ComboResult, combo_dir: Path) -> None:
    """Load HOMO/LUMO energies and ESW from homo_lumo.json if present."""
    hl = combo_dir / "electronic" / "homo_lumo.json"
    if hl.exists():
        try:
            data = json.loads(hl.read_text())
            result.HOMO_eV = data.get("HOMO_eV")
            result.LUMO_eV = data.get("LUMO_eV")
            if result.HOMO_eV is not None and result.LUMO_eV is not None:
                result.ESW_V = abs(result.HOMO_eV - result.LUMO_eV)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_result(r: ComboResult) -> None:
    """Print a one-line summary of D, Ea, sigma, t+, and CN(Li-O) for a combination."""
    tag = f"{r.name} @ {r.salt_conc_M} M"
    D   = f"{r.D_Li:.2e}" if r.D_Li else "—"
    Ea  = f"{r.Ea_eV:.3f} eV" if r.Ea_eV else "—"
    sig = f"{r.sigma_Scm:.3e} S/cm" if r.sigma_Scm else "—"
    tp  = f"{r.t_plus:.2f}" if r.t_plus else "—"
    cn  = f"{r.CN_Li_O:.1f}" if r.CN_Li_O else "—"
    errs = f"  WARN: {'; '.join(r.errors)}" if r.errors else ""
    print(f"  [{tag}]  D={D}  Ea={Ea}  σ={sig}  t+={tp}  CN(Li-O)={cn}{errs}")


def _save_csv(results: Dict[str, ComboResult], output_dir: Path) -> None:
    """Write all ComboResult fields to electrolyte_comparison.csv."""
    csv_path = output_dir / "electrolyte_comparison.csv"
    fields = ["key", "name", "label", "conc_M",
              "D_Li_m2s", "D_anion_m2s", "Ea_eV",
              "sigma_Scm", "t_plus",
              "CN_Li_O", "CN_Li_F", "CN_Li_N",
              "HOMO_eV", "LUMO_eV", "ESW_V", "errors"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, r in results.items():
            w.writerow({
                "key": key, "name": r.name, "label": r.label,
                "conc_M": r.salt_conc_M,
                "D_Li_m2s": r.D_Li, "D_anion_m2s": r.D_anion,
                "Ea_eV": r.Ea_eV, "sigma_Scm": r.sigma_Scm, "t_plus": r.t_plus,
                "CN_Li_O": r.CN_Li_O, "CN_Li_F": r.CN_Li_F, "CN_Li_N": r.CN_Li_N,
                "HOMO_eV": r.HOMO_eV, "LUMO_eV": r.LUMO_eV, "ESW_V": r.ESW_V,
                "errors": "; ".join(r.errors),
            })
    print(f"  [comparison] CSV → {csv_path}")


def _save_json(results: Dict[str, ComboResult], output_dir: Path) -> None:
    """Serialise all ComboResult fields to electrolyte_comparison.json."""
    data = {}
    for key, r in results.items():
        data[key] = {k: v for k, v in r.__dict__.items() if k != "errors"}
        data[key]["errors"] = r.errors
    jpath = output_dir / "electrolyte_comparison.json"
    jpath.write_text(json.dumps(data, indent=2, default=str))
    print(f"  [comparison] JSON → {jpath}")


def _plot_dashboard(results: Dict[str, ComboResult], output_dir: Path) -> None:
    """Generate 2×3 bar-chart dashboard of transport and coordination metrics per concentration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    concs = sorted(set(r.salt_conc_M for r in results.values()))

    for conc in concs:
        sub = {k: v for k, v in results.items() if v.salt_conc_M == conc}
        if not sub:
            continue

        labels = [v.label for v in sub.values()]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(f"Electrolyte Comparison — {conc} M", fontsize=13, fontweight="bold")

        panels = [
            (axes[0, 0], [v.D_Li      for v in sub.values()], "D$_{Li}$ (m²/s)",       True),
            (axes[0, 1], [v.Ea_eV     for v in sub.values()], "E$_a$ (eV)",             False),
            (axes[0, 2], [v.sigma_Scm for v in sub.values()], "σ (S/cm)",               True),
            (axes[1, 0], [v.t_plus    for v in sub.values()], "t$^+$ (Li$^+$ transf.)",False),
            (axes[1, 1], [v.CN_Li_O   for v in sub.values()], "CN(Li–O)  solvation",   False),
            (axes[1, 2], [v.ESW_V     for v in sub.values()], "ESW (V)",                False),
        ]

        cmap = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for ax, vals, ylabel, log in panels:
            nums = [v if v is not None else 0.0 for v in vals]
            bars = ax.bar(range(len(labels)), nums, color=cmap)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=9)
            if log and any(n > 0 for n in nums):
                ax.set_yscale("log")
            for i, v in enumerate(vals):
                if v is None:
                    ax.text(i, 0.001 if log else 0, "N/A",
                            ha="center", va="bottom", fontsize=7, color="gray")

        plt.tight_layout()
        tag = _cfmt(conc)
        png = output_dir / f"comparison_{tag}M.png"
        fig.savefig(str(png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [comparison] Plot → {png}")

    # Also save a conductivity vs Ea scatter per concentration
    _plot_scatter(results, output_dir, concs)


def _plot_scatter(results: Dict[str, ComboResult],
                  output_dir: Path, concs: list) -> None:
    """Generate sigma-vs-Ea scatter plots (one per concentration)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for conc in concs:
        sub = {k: v for k, v in results.items() if v.salt_conc_M == conc}
        xs = [v.Ea_eV      for v in sub.values() if v.Ea_eV and v.sigma_Scm]
        ys = [v.sigma_Scm  for v in sub.values() if v.Ea_eV and v.sigma_Scm]
        lbls = [v.label    for v in sub.values() if v.Ea_eV and v.sigma_Scm]
        if len(xs) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, s=80, zorder=3)
        for x, y, lbl in zip(xs, ys, lbls):
            ax.annotate(lbl, (x, y), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("Activation energy Ea (eV)", fontsize=10)
        ax.set_ylabel("Conductivity σ (S/cm)", fontsize=10)
        ax.set_yscale("log")
        ax.set_title(f"σ vs Ea — {conc} M", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        tag = _cfmt(conc)
        png = output_dir / f"scatter_sigma_Ea_{tag}M.png"
        fig.savefig(str(png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [comparison] Scatter → {png}")
