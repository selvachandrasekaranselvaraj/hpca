"""
slab_builder.py — Pure-Python crystalline surface slab and interface builder.

Replaces AtomicAI create_bulk_surfaces, build_interface, and build_multilayers
external tools, making HPCA fully standalone (no dependency on AtomicAI Python
package or external scripts).

ASE and pymatgen are used opportunistically when available; all public functions
fall back gracefully to pure stdlib + numpy implementations.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("hpca.slab_builder")


# ---------------------------------------------------------------------------
# Internal POSCAR data structure
# ---------------------------------------------------------------------------

class _PoscarData:
    """Lightweight container for a parsed POSCAR."""

    __slots__ = (
        "comment", "scale", "lattice",
        "species", "counts",
        "coords", "direct",
        "selective_dynamics",
    )

    def __init__(
        self,
        comment: str,
        scale: float,
        lattice: np.ndarray,        # (3, 3) — rows are a, b, c vectors
        species: list[str],
        counts: list[int],
        coords: np.ndarray,         # (N, 3)
        direct: bool,               # True → fractional, False → Cartesian
        selective_dynamics: Optional[list[tuple[str, str, str]]],
    ) -> None:
        """Store all parsed POSCAR fields as instance attributes."""
        self.comment = comment
        self.scale = scale
        self.lattice = lattice
        self.species = species
        self.counts = counts
        self.coords = coords
        self.direct = direct
        self.selective_dynamics = selective_dynamics

    @property
    def n_atoms(self) -> int:
        """Return the total number of atoms as the sum of all species counts."""
        return int(np.sum(self.counts))

    def cart_coords(self) -> np.ndarray:
        """Return Cartesian coordinates (physical, including scale)."""
        phys = self.scale * self.lattice
        if self.direct:
            return self.coords @ phys
        return self.coords * self.scale

    def frac_coords(self) -> np.ndarray:
        """Return fractional coordinates."""
        if self.direct:
            return self.coords.copy()
        phys = self.scale * self.lattice
        inv = np.linalg.inv(phys)
        return (self.coords * self.scale) @ inv


# ---------------------------------------------------------------------------
# Pure-Python POSCAR I/O (no ASE / pymatgen needed)
# ---------------------------------------------------------------------------

def _read_poscar_simple(poscar: Path) -> _PoscarData:
    """Parse a VASP5 POSCAR file.

    Returns a _PoscarData with coordinates in the mode stored in the file
    (direct=True → fractional; direct=False → Cartesian, un-scaled).
    """
    poscar = Path(poscar)
    lines = poscar.read_text().splitlines()

    comment = lines[0]
    scale = float(lines[1].split()[0])

    lattice = np.array(
        [list(map(float, lines[i].split()[:3])) for i in range(2, 5)],
        dtype=float,
    )

    # VASP5 vs VASP4: if line 5 tokens start with a digit it's VASP4
    tok5 = lines[5].split()
    vasp4 = tok5[0][0].isdigit()
    if vasp4:
        species: list[str] = []
        counts = list(map(int, tok5))
        body_start = 6
    else:
        species = tok5
        counts = list(map(int, lines[6].split()))
        body_start = 7

    n_atoms = sum(counts)

    # Optional "Selective dynamics"
    selective: Optional[list[tuple[str, str, str]]] = None
    if lines[body_start].strip().lower().startswith("s"):
        selective = []
        body_start += 1

    # Coordinate mode
    coord_line = lines[body_start].strip().lower()
    is_direct = coord_line.startswith("d")
    body_start += 1

    coords = np.zeros((n_atoms, 3), dtype=float)
    sd_flags: list[tuple[str, str, str]] = []
    for i in range(n_atoms):
        parts = lines[body_start + i].split()
        coords[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if selective is not None and len(parts) >= 6:
            sd_flags.append((parts[3], parts[4], parts[5]))
        elif selective is not None:
            sd_flags.append(("T", "T", "T"))

    if selective is not None:
        selective = sd_flags

    return _PoscarData(
        comment=comment,
        scale=scale,
        lattice=lattice,
        species=species,
        counts=counts,
        coords=coords,
        direct=is_direct,
        selective_dynamics=selective,
    )


def _write_poscar_simple(data: _PoscarData, title: str = "HPCA generated") -> str:
    """Serialise a _PoscarData to a POSCAR string (Direct / fractional)."""
    lines: list[str] = []
    lines.append(title)
    lines.append(f"  {data.scale:.10f}")

    for row in data.lattice:
        lines.append("  " + "  ".join(f"{v:20.16f}" for v in row))

    if data.species:
        lines.append("  " + "  ".join(data.species))

    lines.append("  " + "  ".join(str(c) for c in data.counts))

    has_sd = data.selective_dynamics is not None
    if has_sd:
        lines.append("Selective dynamics")

    lines.append("Direct")

    frac = data.frac_coords()
    for i in range(data.n_atoms):
        coord_str = "  ".join(f"{v:20.16f}" for v in frac[i])
        if has_sd:
            fx, fy, fz = data.selective_dynamics[i]  # type: ignore[index]
            lines.append(f"  {coord_str}  {fx}  {fy}  {fz}")
        else:
            lines.append(f"  {coord_str}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Surface slab builder
# ---------------------------------------------------------------------------

def build_surface_slab(
    bulk_poscar: Path,
    miller: tuple[int, int, int],
    n_layers: int = 4,
    vacuum_A: float = 15.0,
    *,
    min_slab_A: float = 8.0,
    center: bool = True,
) -> str:
    """Build a surface slab POSCAR string from a bulk POSCAR.

    Uses ASE surface module (ase.build.surface) with pymatgen SlabGenerator
    fallback, and finally a pure-Python supercell approach.

    Args:
        bulk_poscar: Path to bulk POSCAR file.
        miller: Miller index (h, k, l).
        n_layers: Number of atomic layers.
        vacuum_A: Vacuum thickness in Angstroms above the slab surface.
        min_slab_A: Minimum slab thickness; adds layers if needed.
        center: Center slab in the unit cell along z.

    Returns:
        POSCAR string (can be written directly to a file).

    Raises:
        RuntimeError: if all methods fail.
    """
    bulk_poscar = Path(bulk_poscar)
    miller_str = "".join(str(abs(m)) for m in miller)
    log.debug(
        "[slab_builder] building (%s) slab, %d layers, vacuum=%.1f Å",
        miller_str, n_layers, vacuum_A,
    )

    # ── Strategy 1: ASE ──────────────────────────────────────────────────────
    try:
        from ase.io import read, write
        from ase.build import surface as ase_surface

        atoms = read(str(bulk_poscar), format="vasp")

        # Ensure enough layers to meet min_slab_A
        layers = n_layers
        while True:
            slab = ase_surface(atoms, miller, layers, vacuum=vacuum_A / 2.0)
            cell = slab.get_cell()
            slab_thickness = float(np.linalg.norm(cell[2])) - vacuum_A
            if slab_thickness >= min_slab_A or layers > 20:
                break
            layers += 1

        if center:
            slab.center(vacuum=vacuum_A / 2.0, axis=2)

        buf = io.StringIO()
        write(buf, slab, format="vasp")
        result = buf.getvalue()
        log.info(
            "[slab_builder] (%s) slab built with ASE: %d atoms, %d layers",
            miller_str, len(slab), layers,
        )
        return result
    except Exception as exc:
        log.debug("[slab_builder] ASE surface failed (%s), trying pymatgen", exc)

    # ── Strategy 2: pymatgen SlabGenerator ──────────────────────────────────
    try:
        from pymatgen.core import Structure
        from pymatgen.core.surface import SlabGenerator

        struct = Structure.from_file(str(bulk_poscar))
        slabgen = SlabGenerator(
            struct,
            miller,
            min_slab_size=max(min_slab_A, n_layers * 2.0),
            min_vacuum_size=vacuum_A,
            center_slab=center,
            lll_reduce=True,
            primitive=False,
        )
        slabs = slabgen.get_slabs()
        if not slabs:
            raise RuntimeError("SlabGenerator returned no slabs")

        slab = slabs[0]
        # Convert to POSCAR string
        poscar_str = slab.to(fmt="poscar")
        log.info(
            "[slab_builder] (%s) slab built with pymatgen: %d atoms",
            miller_str, len(slab),
        )
        return poscar_str
    except Exception as exc:
        log.debug("[slab_builder] pymatgen SlabGenerator failed (%s), using pure-Python", exc)

    # ── Strategy 3: Pure-Python — build slab along z by layer stacking ──────
    try:
        return _build_slab_pure(
            bulk_poscar, miller, n_layers, vacuum_A,
            min_slab_A=min_slab_A, center=center,
        )
    except Exception as exc:
        raise RuntimeError(
            f"build_surface_slab: all strategies failed for {bulk_poscar} "
            f"miller={miller}: {exc}"
        ) from exc


def _build_slab_pure(
    bulk_poscar: Path,
    miller: tuple[int, int, int],
    n_layers: int,
    vacuum_A: float,
    *,
    min_slab_A: float,
    center: bool,
) -> str:
    """Fallback pure-Python slab builder.

    For (001) millers this is trivial (just stack copies of the unit cell
    along z and add vacuum).  For arbitrary millers we still produce a slab
    by repeating the bulk along the c-axis — not crystallographically perfect
    but gives a usable starting structure for relaxation.
    """
    p = _read_poscar_simple(bulk_poscar)

    # Physical lattice (Å)
    phys = p.scale * p.lattice  # (3, 3) rows = a, b, c

    # For an arbitrary miller we orient c along the normal:
    # Here we approximate by repeating along the c-direction.
    # A proper rotation would require finding the rotation matrix to align
    # the miller normal with z — for simplicity we repeat along c and add vacuum.
    h, k, l = miller
    if h == 0 and k == 0 and l != 0:
        # (00l): straightforward stacking along c
        repeat_axis = 2
    elif h == 0 and k != 0 and l == 0:
        repeat_axis = 1
    elif h != 0 and k == 0 and l == 0:
        repeat_axis = 0
    else:
        # General case: use c
        repeat_axis = 2

    # Build supercell with n_layers along repeat_axis
    reps = [1, 1, 1]
    reps[repeat_axis] = n_layers

    frac = p.frac_coords()   # (N, 3)
    new_species = []
    new_counts = []
    all_new_frac = []

    for sp, cnt in zip(p.species, p.counts):
        sp_frac = frac[len(new_species) and sum(new_counts[:p.species.index(sp)]):
                       sum(new_counts) + cnt if new_counts else cnt]
        new_species.append(sp)
        new_counts.append(cnt * n_layers)

    # Rebuild frac coords with full repetition
    orig_frac = p.frac_coords()
    n_orig = p.n_atoms
    new_frac = np.zeros((n_orig * n_layers, 3), dtype=float)
    for layer in range(n_layers):
        start = layer * n_orig
        new_frac[start:start + n_orig] = orig_frac.copy()
        new_frac[start:start + n_orig, repeat_axis] = (
            orig_frac[:, repeat_axis] + layer
        ) / n_layers

    # Expand the lattice
    new_lattice = phys.copy()
    new_lattice[repeat_axis] = phys[repeat_axis] * n_layers

    # Add vacuum along z (always axis 2 in output)
    # If repeat_axis != 2 we just add vacuum on top of c
    slab_thickness = float(np.linalg.norm(new_lattice[2]))
    if slab_thickness < min_slab_A:
        extra = int(np.ceil((min_slab_A - slab_thickness) /
                             float(np.linalg.norm(phys[2])))) + 1
        extra_frac_block = np.zeros((n_orig * extra, 3), dtype=float)
        current_n = new_frac.shape[0]
        for layer in range(extra):
            s = layer * n_orig
            extra_frac_block[s:s + n_orig] = orig_frac.copy()
            extra_frac_block[s:s + n_orig, repeat_axis] = (
                orig_frac[:, repeat_axis] + n_layers + layer
            ) / (n_layers + extra)
        new_frac_rescaled = new_frac.copy()
        new_frac_rescaled[:, repeat_axis] *= n_layers / (n_layers + extra)
        new_frac = np.vstack([new_frac_rescaled, extra_frac_block])
        new_lattice[repeat_axis] *= (n_layers + extra) / n_layers
        for sp_i in range(len(new_counts)):
            new_counts[sp_i] += p.counts[sp_i] * extra

    # Vacuum: extend c by vacuum_A
    c_len = float(np.linalg.norm(new_lattice[2]))
    c_hat = new_lattice[2] / c_len
    new_lattice[2] = c_hat * (c_len + vacuum_A)
    # Rescale z fractional coords to account for extended c
    slab_frac_z_scale = c_len / (c_len + vacuum_A)
    new_frac[:, 2] *= slab_frac_z_scale

    if center:
        # Center slab in z: shift so midpoint is at 0.5
        z_min = new_frac[:, 2].min()
        z_max = new_frac[:, 2].max()
        z_mid = (z_min + z_max) / 2.0
        new_frac[:, 2] += 0.5 - z_mid

    out = _PoscarData(
        comment=f"Surface slab ({''.join(str(m) for m in miller)}) n={n_layers}",
        scale=1.0,
        lattice=new_lattice,
        species=p.species,
        counts=new_counts,
        coords=new_frac,
        direct=True,
        selective_dynamics=None,
    )
    miller_str = "".join(str(abs(m)) for m in miller)
    log.info(
        "[slab_builder] (%s) slab built pure-Python: %d atoms",
        miller_str, out.n_atoms,
    )
    return _write_poscar_simple(out, title=out.comment)


# ---------------------------------------------------------------------------
# Lattice matching helper
# ---------------------------------------------------------------------------

def find_matching_supercell(
    lat_a: np.ndarray,
    lat_b: np.ndarray,
    max_supercell: int = 4,
    max_strain: float = 0.05,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Find (m, n) supercell multiples such that m*lat_a ≈ n*lat_b.

    Only the in-plane (x, y) vectors are considered.

    Args:
        lat_a: (3, 3) lattice matrix of slab A (rows = a, b, c).
        lat_b: (3, 3) lattice matrix of slab B.
        max_supercell: Maximum integer multiple to search.
        max_strain: Maximum allowed linear strain.

    Returns:
        ((ma, mb), (na, nb)) supercell integers, or None if not found.
    """
    # Lengths of in-plane vectors
    a_a = float(np.linalg.norm(lat_a[0]))
    b_a = float(np.linalg.norm(lat_a[1]))
    a_b = float(np.linalg.norm(lat_b[0]))
    b_b = float(np.linalg.norm(lat_b[1]))

    best: Optional[tuple[float, tuple[tuple[int, int], tuple[int, int]]]] = None

    for ma in range(1, max_supercell + 1):
        for mb in range(1, max_supercell + 1):
            for na in range(1, max_supercell + 1):
                for nb in range(1, max_supercell + 1):
                    strain_a = abs(ma * a_a - na * a_b) / (ma * a_a)
                    strain_b = abs(mb * b_a - nb * b_b) / (mb * b_a)
                    total = max(strain_a, strain_b)
                    if total <= max_strain:
                        if best is None or total < best[0]:
                            best = (total, ((ma, mb), (na, nb)))

    if best is None:
        return None
    return best[1]


# ---------------------------------------------------------------------------
# Solid|solid interface builder
# ---------------------------------------------------------------------------

def build_interface(
    slab_a_poscar: Path,
    slab_b_poscar: Path,
    *,
    vacuum_A: float = 0.0,
    gap_A: float = 2.5,
    max_strain: float = 0.05,
    fix_bottom: bool = True,
) -> str:
    """Stack two surface slabs into an interface POSCAR.

    Matches in-plane lattice parameters by straining the film (slab_b) to the
    substrate (slab_a).  Atoms are stacked along z with gap_A between slabs.

    Args:
        slab_a_poscar: Substrate slab — kept fixed; sets the lattice reference.
        slab_b_poscar: Film slab — strained to match substrate in-plane.
        vacuum_A: Vacuum added above the top slab (0 = coherent supercell).
        gap_A: Interlayer gap between the two slabs in Angstroms.
        max_strain: Maximum allowed linear strain; raises RuntimeError if exceeded.
        fix_bottom: Add selective dynamics F F F to bottom slab atoms.

    Returns:
        POSCAR string with combined interface structure.

    Raises:
        RuntimeError: if lattice mismatch exceeds max_strain.
    """
    slab_a_poscar = Path(slab_a_poscar)
    slab_b_poscar = Path(slab_b_poscar)

    pa = _read_poscar_simple(slab_a_poscar)
    pb = _read_poscar_simple(slab_b_poscar)

    phys_a = pa.scale * pa.lattice   # (3, 3)
    phys_b = pb.scale * pb.lattice

    # In-plane lattice vector lengths
    a_A = float(np.linalg.norm(phys_a[0]))
    b_A = float(np.linalg.norm(phys_a[1]))
    a_B = float(np.linalg.norm(phys_b[0]))
    b_B = float(np.linalg.norm(phys_b[1]))

    strain_a = abs(a_A - a_B) / a_A
    strain_b = abs(b_A - b_B) / b_A

    if strain_a > max_strain or strain_b > max_strain:
        raise RuntimeError(
            f"build_interface: lattice mismatch exceeds max_strain={max_strain:.1%}. "
            f"a-strain={strain_a:.3%}, b-strain={strain_b:.3%}. "
            f"Consider using find_matching_supercell() first."
        )

    log.debug(
        "[slab_builder] interface: a-strain=%.3f%%, b-strain=%.3f%%",
        strain_a * 100, strain_b * 100,
    )

    # ── Strain film cell to match substrate ──────────────────────────────────
    new_lat_film = phys_b.copy()
    # Scale in-plane vectors; preserve c direction
    scale_a = a_A / a_B if a_B != 0.0 else 1.0
    scale_b = b_A / b_B if b_B != 0.0 else 1.0
    new_lat_film[0] = phys_b[0] * scale_a
    new_lat_film[1] = phys_b[1] * scale_b
    # c-vector of film is unchanged (z-direction stacking)

    # Convert film fractional coords to Cartesian under old lattice,
    # then re-express under new (strained) in-plane lattice
    frac_b = pb.frac_coords()   # (N, 3)
    cart_b = frac_b @ phys_b   # Cartesian under original film lattice

    # Rescale x, y components; keep z
    # cart_b_new[:, 0] *= scale_a  (same direction, different magnitude)
    # cart_b_new[:, 1] *= scale_b
    cart_b_strained = cart_b.copy()
    cart_b_strained[:, 0] *= scale_a
    cart_b_strained[:, 1] *= scale_b

    # ── Build combined lattice ────────────────────────────────────────────────
    # Substrate height along z (physical)
    cart_a = pa.frac_coords() @ phys_a   # (N, 3)
    z_top_sub = float(cart_a[:, 2].max())
    z_bot_sub = float(cart_a[:, 2].min())
    sub_thickness = z_top_sub - z_bot_sub

    film_z_min = float(cart_b_strained[:, 2].min())
    film_z_max = float(cart_b_strained[:, 2].max())
    film_thickness = film_z_max - film_z_min

    # Shift film so its bottom is gap_A above the top of substrate
    z_shift = z_top_sub + gap_A - film_z_min
    cart_b_strained[:, 2] += z_shift

    total_z = sub_thickness + gap_A + film_thickness + vacuum_A

    # Out-of-plane lattice vector: use substrate's c direction, new length
    c_hat = phys_a[2] / float(np.linalg.norm(phys_a[2]))
    new_c = c_hat * (total_z + z_bot_sub + (z_bot_sub if z_bot_sub < 0 else 0))

    # Safer: just set c length to total required box
    c_len = total_z + z_bot_sub + max(0.0, -z_bot_sub)
    # Simplest robust approach: build new_c so atoms at z_bot_sub..z_top_sub+gap+film fit
    new_c_len = film_z_max + z_shift + vacuum_A - z_bot_sub + abs(z_bot_sub)
    new_c = c_hat * new_c_len

    combined_lattice = np.array([
        new_lat_film[0],   # from (strained) film = substrate in-plane
        new_lat_film[1],
        new_c,
    ])

    # Convert all Cartesian coords to fractional under combined lattice
    inv_combined = np.linalg.inv(combined_lattice)
    frac_a_new = cart_a @ inv_combined          # substrate atoms
    frac_b_new = cart_b_strained @ inv_combined  # film atoms

    # Wrap to [0, 1) in x, y; keep z as-is (let VASP handle periodicity)
    frac_a_new[:, :2] = frac_a_new[:, :2] % 1.0
    frac_b_new[:, :2] = frac_b_new[:, :2] % 1.0

    # ── Combine species and counts ────────────────────────────────────────────
    combined_species = pa.species + pb.species
    combined_counts = pa.counts + pb.counts

    # Merge duplicate species (maintain ordering: substrate first, then film)
    all_coords = np.vstack([frac_a_new, frac_b_new])

    # ── Selective dynamics for bottom slab (substrate) ────────────────────────
    sd: Optional[list[tuple[str, str, str]]] = None
    if fix_bottom:
        sd = (
            [("F", "F", "F")] * pa.n_atoms
            + [("T", "T", "T")] * pb.n_atoms
        )

    out = _PoscarData(
        comment="HPCA interface: substrate|film",
        scale=1.0,
        lattice=combined_lattice,
        species=combined_species,
        counts=combined_counts,
        coords=all_coords,
        direct=True,
        selective_dynamics=sd,
    )

    log.info(
        "[slab_builder] interface built: %d substrate + %d film atoms, "
        "gap=%.1f Å, vacuum=%.1f Å",
        pa.n_atoms, pb.n_atoms, gap_A, vacuum_A,
    )
    return _write_poscar_simple(out, title="HPCA interface: substrate|film")


# ---------------------------------------------------------------------------
# Multilayer stacking
# ---------------------------------------------------------------------------

def build_multilayer(
    base_poscar: Path,
    n_repeats: int = 2,
    *,
    gap_A: float = 2.0,
    vacuum_A: float = 15.0,
) -> str:
    """Stack n_repeats copies of the same slab along z with optional gap.

    Each copy is placed gap_A above the previous one.  The final structure
    has vacuum_A of vacuum on top.

    Args:
        base_poscar: Path to the POSCAR of the slab to replicate.
        n_repeats: Number of copies to stack.
        gap_A: Interlayer gap between repeat units (Angstroms).
        vacuum_A: Vacuum added above the topmost layer.

    Returns:
        POSCAR string with stacked multilayer.
    """
    base_poscar = Path(base_poscar)
    p = _read_poscar_simple(base_poscar)

    phys = p.scale * p.lattice   # (3, 3)
    frac = p.frac_coords()       # (N, 3)
    cart = frac @ phys           # (N, 3) Cartesian

    # Slab extent along z
    z_min = float(cart[:, 2].min())
    z_max = float(cart[:, 2].max())
    slab_thickness = z_max - z_min

    # Period = slab_thickness + gap
    period = slab_thickness + gap_A

    n_atoms_total = p.n_atoms * n_repeats
    all_cart = np.zeros((n_atoms_total, 3), dtype=float)

    for rep in range(n_repeats):
        start = rep * p.n_atoms
        shifted = cart.copy()
        # Shift each repeat so z_min starts at rep * period
        shifted[:, 2] += rep * period - z_min
        all_cart[start:start + p.n_atoms] = shifted

    # New c vector: total height + vacuum
    total_z = n_repeats * period - gap_A + vacuum_A   # no gap after last slab
    c_hat = phys[2] / float(np.linalg.norm(phys[2]))
    new_c = c_hat * total_z

    new_lattice = np.array([phys[0], phys[1], new_c])

    # Convert to fractional
    inv_new = np.linalg.inv(new_lattice)
    all_frac = all_cart @ inv_new
    all_frac[:, :2] = all_frac[:, :2] % 1.0

    combined_counts = [c * n_repeats for c in p.counts]

    out = _PoscarData(
        comment=f"HPCA multilayer x{n_repeats}",
        scale=1.0,
        lattice=new_lattice,
        species=p.species,
        counts=combined_counts,
        coords=all_frac,
        direct=True,
        selective_dynamics=None,
    )

    log.info(
        "[slab_builder] multilayer x%d built: %d atoms, total z=%.1f Å",
        n_repeats, out.n_atoms, total_z,
    )
    return _write_poscar_simple(out, title=out.comment)


# ---------------------------------------------------------------------------
# Convenience re-export of the pure-Python I/O helpers
# ---------------------------------------------------------------------------

__all__ = [
    "build_surface_slab",
    "build_interface",
    "build_multilayer",
    "find_matching_supercell",
    "_read_poscar_simple",
    "_write_poscar_simple",
]
