"""
analysis package — MD trajectory analysis modules for battery materials.

Modules:
    sei              — SEI interface identification, composition, growth kinetics
    rdf              — Radial distribution function and coordination numbers
    phase            — Phase transitions, Lindemann criterion, ion hopping
    trajectory       — LAMMPS dump and VASP XDATCAR parser
    msd              — MSD, diffusivity, Van Hove, Arrhenius transport
    electronic       — Electronic structure analysis
    coordination     — Coordination numbers, bond angles, polyhedral analysis
    vanhove          — Self Van Hove G_s(r,t), non-Gaussian parameter alpha2
    hopping          — Ion hopping site detection, hop rates, Haven ratio
    vacf             — Velocity autocorrelation, phonon DOS, Green-Kubo D
    characterization — Unified characterization runner and result dataclass
"""
from . import sei
from . import rdf
from . import phase
from . import coordination
from . import vanhove
from . import hopping
from . import vacf
from . import characterization

# Coordination
from .coordination import (
    compute_coordination_number,
    bond_angle_distribution,
    polyhedral_analysis,
    coordination_vs_time,
)

# Van Hove
from .vanhove import (
    self_van_hove,
    non_gaussian_parameter,
    displacement_distribution,
)

# Hopping
from .hopping import (
    detect_equilibrium_sites,
    assign_sites,
    extract_hop_events,
    hop_rate_per_atom,
    haven_ratio,
)

# VACF
from .vacf import (
    parse_dump_velocities,
    compute_vacf,
    phonon_dos_from_vacf,
    diffusivity_from_vacf,
)

# Characterization
from .characterization import (
    CharacterizationResult,
    run_full_characterization,
    save_characterization,
    print_summary,
)

__all__ = [
    # submodules
    "sei", "rdf", "phase",
    "coordination", "vanhove", "hopping", "vacf", "characterization",
    # coordination
    "compute_coordination_number",
    "bond_angle_distribution",
    "polyhedral_analysis",
    "coordination_vs_time",
    # vanhove
    "self_van_hove",
    "non_gaussian_parameter",
    "displacement_distribution",
    # hopping
    "detect_equilibrium_sites",
    "assign_sites",
    "extract_hop_events",
    "hop_rate_per_atom",
    "haven_ratio",
    # vacf
    "parse_dump_velocities",
    "compute_vacf",
    "phonon_dos_from_vacf",
    "diffusivity_from_vacf",
    # characterization
    "CharacterizationResult",
    "run_full_characterization",
    "save_characterization",
    "print_summary",
]
