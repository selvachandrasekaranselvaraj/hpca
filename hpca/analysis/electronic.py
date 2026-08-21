"""
Electronic structure analysis: DOS/PDOS from VASP, Bader charges,
differential charge density, NEB barrier extraction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── DOS / PDOS ────────────────────────────────────────────────────────────────

def parse_doscar(doscar_path: str | Path, vasprun_path: str | Path = None) -> dict:
    """
    Parse VASP DOSCAR + optionally vasprun.xml for PDOS.
    Returns:
        energies: np.ndarray (n_E,) in eV relative to E_fermi
        dos_total: np.ndarray (n_E,) states/eV
        dos_integrated: np.ndarray (n_E,) electrons
        pdos: dict {atom_idx: {orbital: np.ndarray}} or None
        E_fermi: float
        n_atoms: int
    """
    doscar_path = Path(doscar_path)
    result = {"pdos": None, "E_fermi": 0.0}

    try:
        with open(doscar_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return result

    n_atoms = int(lines[0].split()[0])
    header = lines[5].split()
    E_max, E_min, n_dos = float(header[0]), float(header[1]), int(header[2])
    E_fermi = float(header[3])
    result["E_fermi"] = E_fermi
    result["n_atoms"] = n_atoms

    # Total DOS block starts at line 6
    dos_block = lines[6 : 6 + n_dos]
    data = np.array([[float(x) for x in l.split()] for l in dos_block])
    energies = data[:, 0] - E_fermi
    dos_total = data[:, 1]
    dos_integrated = data[:, 2]
    result.update({"energies": energies, "dos_total": dos_total,
                   "dos_integrated": dos_integrated})

    # PDOS blocks
    pdos = {}
    offset = 6 + n_dos
    for atom_idx in range(n_atoms):
        start = offset + atom_idx * (n_dos + 1) + 1
        if start + n_dos > len(lines):
            break
        block = lines[start : start + n_dos]
        arr = np.array([[float(x) for x in l.split()] for l in block])
        # Columns after energy: s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2, ...
        orbital_labels = ["s", "py", "pz", "px",
                          "dxy", "dyz", "dz2", "dxz", "dx2y2",
                          "f-3", "f-2", "f-1", "f0", "f1", "f2", "f3"]
        pdos[atom_idx] = {}
        for i, orb in enumerate(orbital_labels):
            col = i + 1
            if col < arr.shape[1]:
                pdos[atom_idx][orb] = arr[:, col]
    if pdos:
        result["pdos"] = pdos

    # Parse E_fermi and PDOS from vasprun.xml for accuracy
    if vasprun_path is not None:
        try:
            from pymatgen.io.vasp.outputs import Vasprun
            vr = Vasprun(str(vasprun_path), parse_dos=True)
            cdos = vr.complete_dos
            result["E_fermi"] = vr.efermi
            result["pymatgen_dos"] = cdos
        except Exception:
            pass

    return result


def project_pdos(pdos_raw: dict, atom_species: list[str],
                 species_list: list[str] = None) -> dict:
    """
    Sum PDOS over atoms of the same species.
    Returns {species: {orbital: np.ndarray}}
    """
    if species_list is None:
        species_list = list(dict.fromkeys(atom_species))
    result = {}
    for sp in species_list:
        indices = [i for i, s in enumerate(atom_species) if s == sp]
        sp_pdos = {}
        for idx in indices:
            if idx not in pdos_raw:
                continue
            for orb, arr in pdos_raw[idx].items():
                sp_pdos[orb] = sp_pdos.get(orb, 0) + arr
        # Aggregate s, p (sum py+pz+px), d, f
        agg = {}
        for orb in ("s",):
            if orb in sp_pdos:
                agg["s"] = sp_pdos[orb]
        p_orbs = [sp_pdos.get(o, 0) for o in ("py", "pz", "px")]
        if any(np.any(a != 0) for a in p_orbs if not isinstance(a, int)):
            agg["p"] = sum(p_orbs)
        d_orbs = [sp_pdos.get(o, 0) for o in ("dxy", "dyz", "dz2", "dxz", "dx2y2")]
        if any(np.any(a != 0) for a in d_orbs if not isinstance(a, int)):
            agg["d"] = sum(d_orbs)
        result[sp] = agg
    return result


def find_band_gap(energies: np.ndarray, dos_total: np.ndarray,
                  threshold: float = 0.01) -> dict:
    """
    Find band gap from total DOS.
    Returns: gap_eV, vbm_eV, cbm_eV, is_metal
    """
    # VBM: highest energy where DOS > threshold and E < 0
    neg_mask = energies < 0
    if not np.any(dos_total[neg_mask] > threshold):
        return {"gap_eV": 0, "vbm_eV": None, "cbm_eV": None, "is_metal": True}

    vbm_idx = np.where(neg_mask & (dos_total > threshold))[0]
    cbm_idx = np.where((energies > 0) & (dos_total > threshold))[0]

    if len(vbm_idx) == 0 or len(cbm_idx) == 0:
        return {"gap_eV": 0, "vbm_eV": 0, "cbm_eV": 0, "is_metal": True}

    vbm_eV = float(energies[vbm_idx[-1]])
    cbm_eV = float(energies[cbm_idx[0]])
    gap_eV = max(0.0, cbm_eV - vbm_eV)

    return {"gap_eV": gap_eV, "vbm_eV": vbm_eV, "cbm_eV": cbm_eV,
            "is_metal": gap_eV < 0.05}


def plot_dos(dos_data: dict, project: str,
             pdos_species: dict = None,
             E_range: tuple = (-8, 8)) -> "go.Figure":
    """
    Interactive DOS/PDOS plot.
    Total DOS as filled area; PDOS species as lines.
    Fermi level at E=0 marked with dashed line.
    """
    if not HAS_PLOTLY:
        return None

    energies = dos_data.get("energies")
    dos_total = dos_data.get("dos_total")
    if energies is None:
        return go.Figure()

    mask = (energies >= E_range[0]) & (energies <= E_range[1])
    E = energies[mask]
    D = dos_total[mask]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=E, y=D, mode="lines", fill="tozeroy",
        name="Total DOS",
        line=dict(color="#0079C2", width=1.5),
        fillcolor="rgba(0,121,194,0.15)",
        hovertemplate="E: %{x:.2f} eV<br>DOS: %{y:.3f}<extra></extra>",
    ))

    PDOS_COLORS = {"s": "#E31C3D", "p": "#5E9732", "d": "#F7A11A", "f": "#7A3988"}
    SPECIES_COLORS = ["#E31C3D", "#5E9732", "#F7A11A", "#7A3988",
                      "#D9531E", "#00846B", "#00A4E4"]

    if pdos_species:
        for si, (sp, orbitals) in enumerate(pdos_species.items()):
            sp_color = SPECIES_COLORS[si % len(SPECIES_COLORS)]
            for orb, arr in orbitals.items():
                d_orb = arr[mask] if len(arr) == len(energies) else arr
                fig.add_trace(go.Scatter(
                    x=E, y=d_orb, mode="lines",
                    name=f"{sp}-{orb}",
                    line=dict(color=PDOS_COLORS.get(orb, sp_color), width=1.2, dash="dot"),
                    hovertemplate=f"{sp}-{orb}<br>E: %{{x:.2f}} eV<br>DOS: %{{y:.3f}}<extra></extra>",
                ))

    # Fermi level
    fig.add_vline(x=0, line=dict(color="black", dash="dash", width=1),
                  annotation_text="E_F", annotation_position="top")

    # Band gap annotation
    bg = find_band_gap(energies, dos_total)
    if not bg["is_metal"] and bg["gap_eV"] > 0.05:
        fig.add_vrect(x0=bg["vbm_eV"], x1=bg["cbm_eV"],
                      fillcolor="rgba(200,200,200,0.3)", line_width=0,
                      annotation_text=f"Gap: {bg['gap_eV']:.2f} eV",
                      annotation_position="top center")

    fig.update_layout(
        title=f"{project} — Density of States",
        xaxis_title="E − E<sub>F</sub> (eV)",
        yaxis_title="DOS (states/eV)",
        xaxis_range=list(E_range),
        template="plotly_white",
        width=900, height=550,
        legend=dict(x=1.01, y=1, bordercolor="lightgray", borderwidth=1),
    )
    return fig


# ── Bader charge analysis ─────────────────────────────────────────────────────

def parse_bader_acf(acf_path: str | Path) -> dict:
    """
    Parse Bader ACF.dat output.
    Returns:
        charges: np.ndarray — Bader charges per atom
        charges_valence: np.ndarray — charge transfer (valence - Bader)
        coords: np.ndarray (n_atoms, 3)
        volumes: np.ndarray
    """
    acf_path = Path(acf_path)
    result = {}
    if not acf_path.exists():
        return result

    charges, coords, volumes = [], [], []
    with open(acf_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                charges.append(float(parts[4]))
                if len(parts) >= 6:
                    volumes.append(float(parts[5]))

    result["charges"] = np.array(charges)
    result["coords"] = np.array(coords)
    if volumes:
        result["volumes"] = np.array(volumes)
    return result


def compute_charge_transfer(bader_charges: np.ndarray,
                             atom_species: list[str],
                             valence_electrons: dict[str, float]) -> dict:
    """
    Charge transfer = valence_electrons[species] - bader_charge.
    Positive = lost electrons (oxidized), negative = gained (reduced).
    """
    transfer = np.zeros(len(bader_charges))
    for i, (q, sp) in enumerate(zip(bader_charges, atom_species)):
        val = valence_electrons.get(sp, 0)
        transfer[i] = val - q

    by_species = {}
    for sp in set(atom_species):
        mask = np.array([s == sp for s in atom_species])
        by_species[sp] = {
            "mean": float(np.mean(transfer[mask])),
            "std": float(np.std(transfer[mask])),
            "values": transfer[mask].tolist(),
        }
    return {"transfer_per_atom": transfer, "by_species": by_species}


def plot_bader_charges(bader_data: dict, species_list: list[str],
                        atom_species: list[str], project: str) -> "go.Figure":
    """Bar chart of mean Bader charge transfer by species."""
    if not HAS_PLOTLY or "charges" not in bader_data:
        return go.Figure()

    ct = compute_charge_transfer(bader_data["charges"], atom_species,
                                  {sp: 0 for sp in species_list})

    species = list(ct["by_species"].keys())
    means = [ct["by_species"][s]["mean"] for s in species]
    stds  = [ct["by_species"][s]["std"]  for s in species]

    fig = go.Figure(go.Bar(
        x=species, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=["#E31C3D" if m > 0 else "#0079C2" for m in means],
        hovertemplate="%{x}: %{y:.3f} ± %{error_y.array:.3f} e<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"{project} — Bader Charge Transfer",
        xaxis_title="Species",
        yaxis_title="Δq (e, positive = oxidized)",
        template="plotly_white",
        width=700, height=450,
    )
    return fig


# ── NEB barrier extraction ────────────────────────────────────────────────────

def parse_neb_energies(neb_dir: str | Path) -> dict:
    """
    Extract NEB barrier from VASP NEB run directory.
    Reads OUTCAR from each image directory (00, 01, ..., NN).
    Returns: images, energies_eV (relative to image 00),
             barrier_fwd_eV, barrier_rev_eV, reaction_energy_eV
    """
    neb_dir = Path(neb_dir)
    image_dirs = sorted([d for d in neb_dir.iterdir()
                          if d.is_dir() and d.name.isdigit()])
    if not image_dirs:
        return {}

    energies = []
    for img_dir in image_dirs:
        outcar = img_dir / "OUTCAR"
        if not outcar.exists():
            energies.append(None)
            continue
        E = None
        with open(outcar) as f:
            for line in f:
                if "energy  without entropy" in line:
                    try:
                        E = float(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
        energies.append(E)

    valid = [(i, e) for i, e in enumerate(energies) if e is not None]
    if len(valid) < 2:
        return {}

    idxs, Es = zip(*valid)
    Es = np.array(Es)
    Es_rel = Es - Es[0]  # relative to first image

    barrier_fwd = float(np.max(Es_rel))
    barrier_rev = float(np.max(Es_rel) - Es_rel[-1])
    rxn_energy  = float(Es_rel[-1])

    return {
        "image_indices": list(idxs),
        "energies_eV": Es_rel.tolist(),
        "barrier_fwd_eV": barrier_fwd,
        "barrier_rev_eV": barrier_rev,
        "reaction_energy_eV": rxn_energy,
        "n_images": len(valid),
    }


def plot_neb_profile(neb_data: dict, project: str,
                     path_label: str = "") -> "go.Figure":
    """Interactive NEB energy profile with barrier annotations."""
    if not HAS_PLOTLY or "energies_eV" not in neb_data:
        return go.Figure()

    images = neb_data["image_indices"]
    energies = neb_data["energies_eV"]
    peak_idx = int(np.argmax(energies))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=images, y=energies, mode="lines+markers",
        line=dict(color="#0079C2", width=2.5),
        marker=dict(size=8, color="#0079C2"),
        hovertemplate="Image %{x}<br>ΔE: %{y:.3f} eV<extra></extra>",
        name="NEB path",
    ))
    # Barrier arrow annotation
    fig.add_annotation(
        x=images[peak_idx], y=energies[peak_idx],
        text=f"E<sub>a</sub> = {neb_data['barrier_fwd_eV']:.3f} eV",
        showarrow=True, arrowhead=2, arrowcolor="#E31C3D",
        font=dict(size=13, color="#E31C3D"),
        ax=40, ay=-30,
    )
    fig.update_layout(
        title=f"{project} NEB — {path_label}" if path_label else f"{project} NEB Profile",
        xaxis_title="Image",
        yaxis_title="Energy (eV)",
        template="plotly_white",
        width=800, height=450,
    )
    return fig


# ── Differential charge density ───────────────────────────────────────────────

def compute_charge_density_diff(chgcar_ab: str | Path,
                                 chgcar_a: str | Path,
                                 chgcar_b: str | Path,
                                 output_path: str | Path = None) -> dict:
    """
    Δρ = ρ(AB) − ρ(A) − ρ(B). Writes CHGCAR_diff if output_path given.
    Returns: total_charge_transfer (integrated |Δρ|), output_path
    """
    try:
        from pymatgen.io.vasp.outputs import Chgcar
        rho_ab = Chgcar.from_file(str(chgcar_ab))
        rho_a  = Chgcar.from_file(str(chgcar_a))
        rho_b  = Chgcar.from_file(str(chgcar_b))
        delta  = rho_ab - rho_a - rho_b

        if output_path:
            delta.write_file(str(output_path))

        # Integrated charge transfer
        vol = rho_ab.structure.volume  # Å³
        ng  = rho_ab.data["total"].size
        dV  = vol / ng  # Å³ per grid point
        total_ct = float(np.sum(np.abs(delta.data["total"])) * dV / vol)

        return {"total_charge_transfer": total_ct,
                "output_path": str(output_path) if output_path else None}
    except Exception as e:
        return {"error": str(e)}


# ── Entry point for stage runner ──────────────────────────────────────────────

def run_electronic_analysis(project_dir: str | Path, output_dir: str | Path,
                              project_name: str = "") -> dict:
    """
    Auto-discover and analyze all electronic structure results in project_dir.
    Checks: dos/, bader/, neb/path_*/
    Returns dict with all results and figure paths.
    """
    project_dir = Path(project_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # DOS
    for dos_dir in [project_dir / "dos", project_dir / "dos" / "nonscf"]:
        doscar = dos_dir / "DOSCAR"
        if doscar.exists():
            vr = dos_dir / "vasprun.xml"
            dos_data = parse_doscar(doscar, vr if vr.exists() else None)
            results["dos"] = dos_data

            if HAS_PLOTLY and "energies" in dos_data:
                fig = plot_dos(dos_data, project_name)
                html_path = output_dir / f"{project_name}_dos.html"
                png_path  = output_dir / f"{project_name}_dos.png"
                fig.write_html(str(html_path))
                try:
                    fig.write_image(str(png_path), width=900, height=550)
                except Exception:
                    pass
                results["dos"]["figures"] = {"html": str(html_path), "png": str(png_path)}
            break

    # Bader
    bader_dir = project_dir / "bader"
    acf = bader_dir / "ACF.dat"
    if acf.exists():
        bader_data = parse_bader_acf(acf)
        results["bader"] = bader_data

    # NEB
    neb_base = project_dir / "neb"
    if neb_base.exists():
        neb_results = {}
        for path_dir in sorted(neb_base.iterdir()):
            if path_dir.is_dir() and "path" in path_dir.name:
                neb_data = parse_neb_energies(path_dir)
                if neb_data:
                    neb_results[path_dir.name] = neb_data
                    if HAS_PLOTLY:
                        fig = plot_neb_profile(neb_data, project_name, path_dir.name)
                        html_out = output_dir / f"{project_name}_neb_{path_dir.name}.html"
                        fig.write_html(str(html_out))
                        neb_data["figure_html"] = str(html_out)
        if neb_results:
            results["neb"] = neb_results

    return results
