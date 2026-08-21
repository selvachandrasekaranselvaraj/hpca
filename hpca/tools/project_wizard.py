"""
project_wizard.py — End-to-end materials design wizard for HPCA.

Auto-detects structure files, identifies material types, asks domain-specific
questions for every system geometry, and writes a complete project.yaml.
"""
from __future__ import annotations

import json
import sys
from hpca.core.config import account_fallback as _account_fallback
from hpca.core.slurm_submit import module_bundle_lines as _mbl
from datetime import datetime
from pathlib import Path

# ── Colour / print helpers ────────────────────────────────────────────────────
def _bold(s):    """Wrap s in ANSI bold."""; return f"\033[1m{s}\033[0m"
def _green(s):   """Wrap s in ANSI green."""; return f"\033[32m{s}\033[0m"
def _cyan(s):    """Wrap s in ANSI cyan."""; return f"\033[36m{s}\033[0m"
def _yellow(s):  """Wrap s in ANSI yellow."""; return f"\033[33m{s}\033[0m"
def _dim(s):     """Wrap s in ANSI dim."""; return f"\033[2m{s}\033[0m"
def _red(s):     """Wrap s in ANSI red."""; return f"\033[31m{s}\033[0m"
def _hdr(title):
    """Print a styled section header bar to the terminal."""
    bar = "─" * (52 - len(title))
    print(f"\n  {_bold(_cyan(f'── {title} {bar}'))}")

# ── Option tables ─────────────────────────────────────────────────────────────
MOBILE_IONS = [
    ("Li",  "Lithium    — SSEs, Li-metal anodes, Li-ion cathodes"),
    ("Na",  "Sodium     — Na-ion, Na-air, Na-metal"),
    ("K",   "Potassium  — K-ion electrolytes"),
    ("Mg",  "Magnesium  — Mg-ion solid electrolytes"),
    ("Ca",  "Calcium    — Ca-ion batteries"),
    ("F",   "Fluorine   — Fluoride conductors (SrF2, CaF2)"),
    ("Cl",  "Chlorine   — Chloride conductors"),
    ("O",   "Oxygen     — O2- conductors, SOFC"),
    ("custom", "Other   — type the ion name"),
]

BOX_CATEGORIES = [
    ("solvent",    "Solvents     organic liquids (DME, EC, DMC, DMB...)"),
    ("salt",       "Salts        ionic salts (LiFSI, LiPF6, LiTFSI...)"),
    ("polymer",    "Polymers     homopolymers (PEO, PVDF, PTFEP, PMMA...)"),
    ("copolymer",  "Copolymers   mixed chains (PVDF-HFP, PVDF-TrFE...)"),
    ("solid",      "Solids       crystals, hard carbon, MOFs, interfaces..."),
    ("custom",     "custom       Other — type category name"),
]

SYSTEM_TYPES = [
    ("bulk_sse",        "Bulk solid electrolyte    — Li2ZrCl6, LGPS, LLZO, LIPON..."),
    ("bulk_electrode",  "Bulk cathode / anode      — NMC622, LFP, NVO, graphite..."),
    ("bulk_coating",    "Bulk coating / film       — SrF2, AlF3, Al2O3, MgO..."),
    ("liquid_electrolyte","Liquid electrolyte      — solvent(s) + Li/Na salt"),
    ("polymer",         "Polymer electrolyte       — PEO, PVDF, PTFEP + salt"),
    ("gel",             "Gel electrolyte           — polymer + solvent + salt"),
    ("sse_electrode",   "SSE | Electrode interface — Li2ZrCl6|Li, LGPS|NMC622..."),
    ("sse_liquid",      "SSE | Liquid interface    — SSE slab in electrolyte"),
    ("electrode_liquid","Electrode | Liquid        — anode/cathode slab in electrolyte"),
    ("surface",         "Surface / slab            — single material, vacuum above"),
    ("nanoparticle",    "Nanoparticle (free-standing)"),
    ("np_substrate",    "Nanoparticle on substrate — NP + support material"),
]

CATEGORY_MAP = {
    "bulk_sse":          "inorganic_sse",
    "bulk_electrode":    "inorganic",
    "bulk_coating":      "inorganic",
    "liquid_electrolyte":"liquid_electrolyte",
    "polymer":           "polymer",
    "gel":               "polymer",
    "sse_electrode":     "inorganic_sse",
    "sse_liquid":        "inorganic_sse",
    "electrode_liquid":  "inorganic",
    "surface":           "inorganic",
    "nanoparticle":      "inorganic",
    "np_substrate":      "inorganic",
}

SOLVENTS = [
    ("DMB",    "1,2-Dimethoxybutane (ether, high-voltage)"),
    ("DOL",    "1,3-Dioxolane (ether)"),
    ("DME",    "1,2-Dimethoxyethane (ether)"),
    ("EC",     "Ethylene carbonate"),
    ("DMC",    "Dimethyl carbonate"),
    ("EMC",    "Ethyl methyl carbonate"),
    ("DEC",    "Diethyl carbonate"),
    ("PC",     "Propylene carbonate"),
    ("FEC",    "Fluoroethylene carbonate (additive)"),
    ("TEGDME", "Tetraethylene glycol dimethyl ether"),
    ("ACN",    "Acetonitrile"),
    ("custom", "Other — I'll type the name"),
]

SALTS = [
    ("LiFSI",  "LiFSI  — Lithium bis(fluorosulfonyl)imide"),
    ("LiPF6",  "LiPF6  — Lithium hexafluorophosphate (commercial)"),
    ("LiTFSI", "LiTFSI — Lithium bis(trifluoromethanesulfonyl)imide"),
    ("LiClO4", "LiClO4 — Lithium perchlorate"),
    ("NaFSI",  "NaFSI  — Sodium bis(fluorosulfonyl)imide"),
    ("NaTFSI", "NaTFSI — Sodium bis(trifluoromethanesulfonyl)imide"),
    ("NaPF6",  "NaPF6  — Sodium hexafluorophosphate"),
    ("none",   "No salt — pure solvent / custom"),
]

MONOMERS = [
    ("PEO",      "Poly(ethylene oxide)  [-CH2-CH2-O-]n"),
    ("PVDF",     "Poly(vinylidene fluoride)  [-CH2-CF2-]n"),
    ("PTFEP",    "Poly(trifluoroethyl phosphazene)  [-N=P(OCH2CF3)2-]n"),
    ("PVDF-HFP", "PVDF-co-hexafluoropropylene"),
    ("custom",   "Custom monomer — I'll provide SMILES"),
]

COPOLYMERS = [
    ("PVDF-HFP",  "PVDF-co-HFP  (VDF:HFP, common 9:1)"),
    ("PVDF-TrFE", "PVDF-co-TrFE (VDF:TrFE, common 75:25)"),
    ("custom",    "Custom — specify two monomers and ratio"),
]

ELECTRODES = [
    ("Li",       "Li metal anode"),
    ("Na",       "Na metal anode"),
    ("graphite", "Graphite anode"),
    ("NMC622",   "LiNi0.6Mn0.2Co0.2O2 cathode"),
    ("LFP",      "LiFePO4 cathode"),
    ("NVO",      "Na3V2O5 cathode"),
    ("HC",       "Hard carbon anode"),
    ("Sn",       "Sn/SnO2 anode"),
    ("custom",   "Other — I'll type the name"),
]

MILLER = [
    ("001", "(0 0 1) — most common"),
    ("110", "(1 1 0)"),
    ("111", "(1 1 1)"),
    ("010", "(0 1 0)"),
    ("custom", "Other — I'll type"),
]

NP_SHAPES = [
    ("sphere",   "Sphere — carved from bulk"),
    ("wulff",    "Wulff shape — equilibrium morphology from surface energies"),
    ("cube",     "Cube"),
    ("faceted",  "Faceted from (hkl) planes"),
    ("core_shell","Core-shell — two materials"),
]

FORCEFIELDS = [
    ("OPLS-AA",  "OPLS-AA (best for organics, polymers, Li/Na salts)"),
    ("GAFF",     "GAFF / AMBER"),
    ("DREIDING", "DREIDING (general-purpose)"),
    ("custom",   "Custom — I'll provide LAMMPS parameters"),
]

TEMPERATURES_AIMD = [250, 300, 320, 340, 360, 380, 400, 450, 500, 600, 700, 800, 1000, 1200]
TEMPERATURES_MLMD = [250, 300, 320, 340, 360, 380, 400, 450, 500, 600, 700, 800]

_BINARY_RATIO_PRESETS:  list[tuple[int, int]]      = [(2, 1), (1, 1), (1, 2)]
_TERNARY_RATIO_PRESETS: list[tuple[int, int, int]] = [(2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 1, 1)]
_EO_LI_PRESETS: list[int] = [8, 12, 16, 20, 32]

def _lane_defaults() -> dict:
    """Return atom/step defaults from platform.yaml limits.slurm."""
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        _plat = _yaml.safe_load(
            (_Path(__file__).parents[2] / "config" / "platform.yaml").read_text()
        ) or {}
        _lim = _plat.get("limits", {})
    except Exception:
        _plat = {}
        _lim = {}

    _l = _lim.get("slurm", {})
    _ts_cmd = _plat.get("lammps_md", {}).get("timestep_fs_cmd", 1.0)
    return {
        "aimd_atoms":  _l.get("dft_atoms",   300),
        "mlmd_atoms":  _l.get("mlmd_atoms",  5_000),
        "cmd_atoms":   _l.get("cmd_atoms",  50_000),
        "aimd_steps":  _l.get("aimd_steps",  3_000),
        "mlmd_steps":  int(_l.get("mlmd_nvt_ns", 1) * 1e6),
        "npt_steps":   _l.get("npt_steps_aimd", 3_000),
        "nvt_steps":   int(_l.get("cmd_nvt_ns", 2) * 1e6 / _ts_cmd),
    }


def _cfmt(c: float) -> str:
    """Format float concentration for directory names: 0.25 → '0p25', 1.0 → '1p0'."""
    return str(c).replace(".", "p")


# ── Input helpers ─────────────────────────────────────────────────────────────

def _pick(label: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """Prompt the user to select one item from a numbered options list; returns the chosen key."""
    print()
    print(_bold(f"  {label}:"))
    for i, (key, desc) in enumerate(options, 1):
        print(f"  {_cyan(str(i))}) {_bold(key):<16}  {desc}")
    while True:
        try:
            raw = input(f"\n  Choice [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        val = raw or str(default)
        if val.isdigit() and 1 <= int(val) <= len(options):
            chosen = options[int(val) - 1][0]
            print(f"  → {_green(chosen)}")
            return chosen
        for key, _ in options:
            if val.lower() == key.lower():
                print(f"  → {_green(key)}")
                return key
        print(f"  {_yellow('Enter 1–' + str(len(options)) + ' or the key name')}")


def _pick_multi(label: str, options: list[tuple[str, str]], max_picks: int = 3,
                min_picks: int = 1) -> list[str]:
    """Prompt the user to select multiple items from a numbered list; returns list of chosen keys."""
    print()
    hint = f"space-separated, pick {min_picks}–{max_picks}"
    print(_bold(f"  {label}  {_dim('(' + hint + ')')}:"))
    for i, (key, desc) in enumerate(options, 1):
        print(f"  {_cyan(str(i))}) {_bold(key):<16}  {desc}")
    while True:
        try:
            raw = input(f"\n  Choices [e.g. 1 3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not raw:
            continue
        chosen = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(options):
                k = options[int(tok) - 1][0]
                if k not in chosen:
                    chosen.append(k)
        if min_picks <= len(chosen) <= max_picks:
            print(f"  → {_green(', '.join(chosen))}")
            return chosen
        print(f"  {_yellow('Select ' + str(min_picks) + '–' + str(max_picks) + ' items')}")


def _pick_temps(label: str, avail: list[int], existing: list[int] = None) -> list[int]:
    """Prompt for a set of temperatures in K, merging user input with any pre-existing values."""
    existing = existing or []
    print()
    print(_bold(f"  {label}"))
    print(f"  {_dim('Suggestions: ' + '  '.join(str(T) for T in avail))}")
    if existing:
        print(f"  {_dim('Already set:  ' + '  '.join(str(T) for T in existing))}")
    print(f"  {_dim('Type any temperatures in K (space-separated, Enter=skip):')}")
    try:
        raw = input("  Temperatures K: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    if not raw:
        return existing
    chosen = []
    for tok in raw.split():
        if tok.isdigit() and int(tok) > 0:
            chosen.append(int(tok))
    result = sorted(set(existing + chosen))
    if result:
        print(f"  → {result}")
    return result


def _input(label: str, default: str = "") -> str:
    """Prompt for a string value with an optional default; returns the default on empty input."""
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    return val or default


def _float(label: str, default: float) -> float:
    """Prompt for a float value, re-prompting on invalid input."""
    while True:
        raw = _input(label, str(default))
        try:
            return float(raw)
        except ValueError:
            print(f"  {_yellow('Please enter a number')}")


def _int(label: str, default: int) -> int:
    """Prompt for an integer value, re-prompting on invalid input."""
    while True:
        raw = _input(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  {_yellow('Please enter a whole number')}")


def _yes(label: str, default: bool = True) -> bool:
    """Prompt for a yes/no confirmation; returns default on empty input."""
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {label} [{hint}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    return default if not raw else raw in ("y", "yes")


# ── Structure file auto-detection ─────────────────────────────────────────────

def _detect_structures(project_dir: Path) -> list[dict]:
    """Scan for structure files and identify them via pymatgen."""
    found = []
    patterns = ["*.vasp", "*.cif", "*.poscar", "POSCAR", "CONTCAR"]
    seen = set()
    for pat in patterns:
        for f in sorted(project_dir.glob(pat)):
            if f.name in seen:
                continue
            seen.add(f.name)
            info: dict = {"file": f.name, "path": str(f)}
            try:
                from pymatgen.core import Structure
                s = Structure.from_file(str(f))
                info["formula"]   = s.composition.reduced_formula
                info["natoms"]    = len(s)
                info["species"]   = sorted(set(str(e) for e in s.species))
                info["role"]      = _guess_role(s)
                info["structure"] = s
            except Exception:
                info["formula"] = "?"
                info["natoms"]  = 0
                info["species"] = []
                info["role"]    = "unknown"
            found.append(info)
    return found


def _guess_role(s) -> str:
    """Heuristic role assignment from species set."""
    sp = set(str(e) for e in s.species)
    has_li   = "Li" in sp
    has_na   = "Na" in sp
    alkali   = has_li or has_na
    # Y, Zr, La, Sc, In are SSE-formers (Li3YCl6, Li2ZrCl6, LLZO) — exclude from cathode TM set
    has_tm   = bool(sp & {"Ni","Mn","Co","Fe","V","Ti","Nb","Mo","W","Cr"})
    has_hal  = bool(sp & {"F","Cl","Br","I"})
    has_sulf = "S" in sp
    has_N    = "N" in sp
    has_C    = "C" in sp
    has_H    = "H" in sp
    has_O    = "O" in sp
    has_P    = "P" in sp
    has_alloy_host = bool(sp & {"Si", "Sn", "Ge", "Sb", "Bi"})

    # Ionic salts: alkali + sulfonimide/phosphate/borate groups
    # LiFSI: Li S N O F  |  LiTFSI: Li S N O F C  |  LiPF6: Li P F  |  NaPF6: Na P F
    if alkali and has_N and has_sulf and has_hal:    # FSI / TFSI family
        return "salt"
    if alkali and has_P and has_hal and not has_tm:  # LiPF6, NaPF6
        return "salt"
    if alkali and has_hal and not has_tm and not has_C and not has_N and not has_sulf:
        return "halide_sse"
    # Organic molecules without alkali → solvent (DMB, EC, DMC, DOL...)
    if has_C and (has_H or has_O) and not alkali and not has_tm:
        return "solvent"
    # Organic with alkali that also has N or S → salt (e.g. LiTFSI with C)
    if has_C and alkali and (has_N or has_sulf):
        return "salt"
    if alkali and has_sulf and not has_tm and not has_C:
        return "sulfide_sse"
    if alkali and has_tm and has_O:
        return "oxide_electrode"
    if alkali and has_alloy_host and not has_hal and not has_sulf:
        return "alloy_electrode"
    if has_tm and has_O and not alkali:
        return "oxide_electrode"
    if sp <= {"Li"}:
        return "electrode_metal"
    if sp <= {"Na"}:
        return "electrode_metal"
    if sp <= {"C"}:
        return "electrode_carbon"
    if sp <= {"Sn","O"}:
        return "electrode_metal"
    return "unknown"


def _suggest_system(structs: list[dict]) -> str:
    """Guess system type from detected structure roles."""
    roles = [s["role"] for s in structs]
    has_solvent  = "solvent" in roles
    has_salt     = "salt" in roles
    has_sse      = any(r in roles for r in ("halide_sse","sulfide_sse"))
    has_electrode= any("electrode" in r for r in roles)
    if has_solvent and not has_sse:
        return "liquid_electrolyte"
    if has_sse and has_electrode:
        return "sse_electrode"
    if has_sse and has_solvent:
        return "sse_liquid"
    if has_sse:
        return "bulk_sse"
    if has_electrode and has_solvent:
        return "electrode_liquid"
    if has_electrode:
        return "bulk_electrode"
    return "bulk_coating"  # unknown solids require a neutral, non-SSE default


# ── Existing-data scanner ─────────────────────────────────────────────────────

def _detect_sim_data(project_dir: Path) -> dict:
    """Scan project_dir for completed simulation artefacts and return a stage-progress dict."""
    d: dict = {"stages_done": []}
    aimd = []
    for xd in sorted(project_dir.rglob("XDATCAR")):
        rel = str(xd.parent.relative_to(project_dir))
        if rel not in aimd:
            aimd.append(rel)
    if aimd:
        d["aimd_dirs"] = aimd
        d["stages_done"].append("aimd")
    mlmd: dict = {}
    for dump in sorted(project_dir.rglob("dump_unwrapped.lmp")):
        try:
            T = int(dump.parent.name.replace("K", ""))
            mlmd[T] = str(dump.parent.relative_to(project_dir))
        except ValueError:
            pass
    if mlmd:
        d["mlmd_dirs"] = mlmd
        d["stages_done"].append("lammps")
    for pot in project_dir.rglob("pot_com.pb"):
        d["deepmd_pot"] = str(pot.relative_to(project_dir))
        d["stages_done"].append("mlip")
        break
    if any(project_dir.rglob("CONTCAR")):
        d["stages_done"] += ["opt"]
    if any(project_dir.rglob("ACF.dat")):
        d["stages_done"].append("bader")
    if any(project_dir.rglob("neb_barriers.json")):
        d["stages_done"].append("neb")
    if any(project_dir.rglob("homo_lumo.json")):
        d["stages_done"].append("electronic")
    if any(project_dir.rglob("echem_summary.json")):
        d["stages_done"].append("echem")
    d["stages_done"] = list(dict.fromkeys(d["stages_done"]))
    return d


# ── Stages block builder ──────────────────────────────────────────────────────

def _stages_block(system_type: str, done: list[str], run: bool,
                   tiers_selected: list | None = None) -> dict:
    """Build the stages dict (which workflow steps to run) from system type and completed stages."""
    tiers      = set(tiers_selected or [])
    is_sse     = "sse" in system_type
    is_liquid  = "liquid" in system_type
    is_polymer = "polymer" in system_type or system_type == "gel"
    is_interface = "interface" in system_type or system_type in (
        "sse_electrode", "sse_liquid", "electrode_liquid", "np_substrate")
    # MLIP training is needed whenever MLMD is selected (any category)
    needs_mlip   = "MLMD" in tiers or (not is_liquid and not is_polymer)
    needs_lammps = needs_mlip   # LAMMPS MLMD runs follow MLIP training
    needs_neb    = is_sse
    is_electrode = system_type == "bulk_electrode"
    # Continuum PNP model applies to all electrolyte categories
    needs_continuum = is_sse or is_interface or is_liquid or is_polymer

    def _f(key, applicable):
        """Return the run flag if the stage is applicable and not already done."""
        if not applicable:
            return False
        return False if key in done else run

    dft_done = "opt" in done
    return {
        "design":       _f("design",     True),
        "dft": False if dft_done else {
            "vc_relax":  not dft_done,
            "opt":       not dft_done,
            "bader":     "bader" not in done,
            "dos_scf":   is_electrode and "dos_scf" not in done,
            "dos_nonscf":is_electrode and "dos_nonscf" not in done,
            "static":     (is_sse or is_electrode) and "static" not in done,
            "echem_static": (is_sse or is_electrode) and "echem_static" not in done,
        } if run else False,
        "aimd":         _f("aimd",       "AIMD" in tiers or not is_polymer),
        "neb":          _f("neb",        needs_neb),
        "mlip":         _f("mlip",       needs_mlip),
        "lammps":       _f("lammps",     needs_lammps),
        "classical_md": _f("classical_md", "CMD" in tiers or is_polymer or is_liquid),
        "analysis":     True,
        "electronic":   _f("electronic", True),
        "echem":        _f("echem",      is_sse or is_electrode or is_interface),
        "continuum":    _f("continuum",  needs_continuum),
        "plotting":     True,
        "manuscript":   True,
        "chaai":        True,
    }


def _seed_state(project_dir: Path, done: list[str]) -> None:
    """Write an initial orchestrator_state.json marking already-completed handler stages as COMPLETE."""
    HMAP = {
        "opt":        ["h00_design","h01_dft.vc_relax","h01_dft.opt"],
        "bader":      ["h01_dft.bader"],
        "aimd":       ["h02_aimd"],
        "neb":        ["h03_neb"],
        "mlip":       ["h04_mlip","h13_active_learning"],
        "lammps":     ["h05_lammps"],
        "electronic": ["h07_electronic"],
        "echem":      ["h08_echem"],
    }
    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    sp = log_dir / "orchestrator_state.json"
    state = {"stages": {}, "jobs": {}, "meta": {}}
    if sp.exists():
        try:
            state = json.loads(sp.read_text())
        except Exception:
            pass
    now = datetime.now().isoformat()
    for stage in done:
        for h in HMAP.get(stage, []):
            state["stages"].setdefault(h, {})["status"] = "COMPLETE"
            state["stages"][h].setdefault("completed_at", now)
            state["stages"][h]["seeded_by"] = "project_wizard"
    sp.write_text(json.dumps(state, indent=2))


# ── Domain question blocks ────────────────────────────────────────────────────

def _ask_solvents(detected_solvents: list[str]) -> list[dict]:
    """Ask which solvents and their molar ratios."""
    _hdr("Solvent(s)")
    n_sv = int(_pick("Number of solvents", [
        ("1", "Single solvent"),
        ("2", "Binary mixture   e.g. EC/DMC, DOL/DME, DMB/DOL"),
        ("3", "Ternary mixture  e.g. EC/DMC/FEC, EC/EMC/FEC"),
    ], default=1))

    sv_opts = list(SOLVENTS)
    chosen_names: list[str] = []

    # Pre-populate from detected files
    for dsv in detected_solvents:
        if dsv not in [k for k, _ in SOLVENTS]:
            # prepend as detected
            sv_opts.insert(0, (dsv, f"(detected from {dsv}.vasp)"))

    for i in range(n_sv):
        label = f"Solvent {i+1}" if n_sv > 1 else "Solvent"
        sv = _pick(label, sv_opts, default=i+1)
        if sv == "custom":
            sv = _input("  Enter solvent name or abbreviation")
        if sv not in chosen_names:
            chosen_names.append(sv)

    if n_sv == 1:
        return [{"name": chosen_names[0], "ratio": 1}]

    print()
    print(_bold(f"  Molar ratio  {' : '.join(chosen_names)}"))
    print(_dim("  e.g.  1 1  or  3 5 2  (integers, space-separated)"))
    while True:
        try:
            raw = input("  Ratios: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        parts = raw.split()
        if len(parts) == n_sv and all(p.isdigit() for p in parts):
            ratios = [int(p) for p in parts]
            print(f"  → {_green(' : '.join(str(r) for r in ratios))}")
            return [{"name": n, "ratio": r} for n, r in zip(chosen_names, ratios)]
        print(f"  {_yellow('Enter ' + str(n_sv) + ' integers')}")


def _ask_salt(detected_salt: str = "") -> str:
    """Return salt name only (concentrations asked separately)."""
    _hdr("Salt")
    opts = list(SALTS)
    if detected_salt and detected_salt not in [k for k, _ in opts]:
        opts.insert(0, (detected_salt, "(detected from structure file)"))
    salt = _pick("Salt type", opts, default=1)
    return salt


def _ask_concentrations(salt: str) -> list[float]:
    """Ask which salt concentrations to simulate across all stages."""
    if salt == "none":
        return [0.0]
    _hdr("Salt Concentrations")
    std = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5]
    print(_bold("  Available concentrations (pick one or several for a sweep):"))
    for i, c in enumerate(std, 1):
        print(f"  {_cyan(str(i))}) {c} M")
    print(_dim("  Enter numbers or type values directly (space-separated)"))
    print(_dim("  Example: 1 3 5  →  0.25, 1.0, 2.0 M"))
    print(_dim("  Example: 0.5 1.0 2.0  →  type custom values"))
    while True:
        try:
            raw = input("  Concentrations [3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not raw:
            raw = "3"
        concs: list[float] = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(std):
                concs.append(std[int(tok) - 1])
            else:
                try:
                    concs.append(float(tok))
                except ValueError:
                    pass
        concs = sorted(set(concs))
        if concs:
            print(f"  → {_green(', '.join(str(c) + ' M' for c in concs))}")
            return concs
        print(f"  {_yellow('Enter at least one concentration')}")


def _ask_liquid_cell() -> dict:
    """
    Cell sizes for each simulation tier.

    Simulation sequence for liquid electrolytes:
      1. AIMD  (VASP, small cell ~100 atoms)  → ab-initio reference + DeepMD training data
      2. MLMD  (LAMMPS + DeepMD potential, large cell ~1000–5000 atoms)  → transport properties
      3. CMD   (LAMMPS + classical FF, very large cell ~5000–20000 atoms) → equilibration, NPT
    """
    _hdr("Simulation Cell Sizes")
    print()
    print(_bold("  Three simulation tiers for liquid electrolytes:"))
    print(f"  {_cyan('AIMD')}  — VASP ab-initio MD  "
          f"{_dim('(expensive, small cell, generates DeepMD training data)')}")
    print(f"  {_cyan('MLMD')}  — LAMMPS + DeepMD    "
          f"{_dim('(cheap, large cell, needs AIMD data first)')}")
    print(f"  {_cyan('CMD')}   — LAMMPS + classical FF  "
          f"{_dim('(no ML needed, for equilibration / comparison)')}")
    print()

    # AIMD cell
    print(_bold("  ── AIMD cell (VASP):"))
    n_aimd = int(_pick("  Solvent molecules per AIMD cell", [
        ("4",  "4 molecules  (~50–100 atoms)   — fast but small"),
        ("8",  "8 molecules  (~100–200 atoms)  — standard"),
        ("12", "12 molecules (~150–300 atoms)  — larger"),
        ("16", "16 molecules (~200–400 atoms)  — maximum for VASP"),
    ], default=2))

    density = _float("  Target density (g/cm³)", 0.85)

    # MLMD cell
    print()
    print(_bold("  ── MLMD cell (LAMMPS + DeepMD, runs AFTER AIMD + MLIP training):"))
    n_mlmd = int(_pick("  Solvent molecules per MLMD cell", [
        ("50",   "50  molecules  (~600–1000 atoms)   — fast"),
        ("100",  "100 molecules  (~1200–2000 atoms)  — standard"),
        ("200",  "200 molecules  (~2400–4000 atoms)  — accurate"),
        ("500",  "500 molecules  (~6000–10000 atoms) — production"),
    ], default=2))

    mlmd_steps = _int("  MLMD run length per temperature (steps)", _lane_defaults()["mlmd_steps"])
    mlmd_temps = _pick_temps("  MLMD temperatures", TEMPERATURES_MLMD, [])

    # Classical MD (optional)
    print()
    print(_bold("  ── Classical MD cell (LAMMPS + force-field, optional):"))
    run_cmd = _yes("  Run classical force-field MD (OPLS-AA/GAFF)?", default=False)
    n_cmd, cmd_steps_equil, cmd_steps_prod = 0, 0, 0
    if run_cmd:
        n_cmd = int(_pick("  Solvent molecules per classical MD cell", [
            ("100",  "100 molecules  (~1200–2000 atoms)"),
            ("500",  "500 molecules  (~6000–10000 atoms) — standard"),
            ("1000", "1000 molecules (~12000–20000 atoms) — large"),
        ], default=2))
        cmd_steps_equil = _int("  NPT equilibration steps", 2_000_000)
        cmd_steps_prod  = _int("  NVT production steps",    10_000_000)
        cmd_temps = _pick_temps("  CMD temperatures", TEMPERATURES_MLMD, []) if run_cmd else []
    else:
        cmd_temps = []

    return {
        "n_molecules_aimd":   n_aimd,
        "n_molecules_mlmd":   n_mlmd,
        "n_molecules_cmd":    n_cmd,
        "target_density_gcm3":density,
        "classical_md":       run_cmd,
        "mlmd_steps":         mlmd_steps,
        "mlmd_temps":         mlmd_temps,
        "cmd_equil_steps":    cmd_steps_equil,
        "cmd_prod_steps":     cmd_steps_prod,
        "cmd_temps":          cmd_temps,
    }


def _ask_liquid_temps(existing: list[int]) -> tuple[list[int], int]:
    """Prompt for AIMD temperatures and step count for liquid electrolyte projects."""
    _hdr("AIMD Temperatures")
    temps = _pick_temps("AIMD temperatures (one cell built per conc × temp combination)",
                        TEMPERATURES_AIMD, existing)
    steps = _int("AIMD steps per temperature (NSW)", _lane_defaults()["aimd_steps"])
    return temps, steps


def _ask_interface_geometry() -> dict:
    """Questions for any two-material interface."""
    _hdr("Interface Geometry")
    miller = _pick("Surface / Miller index", MILLER, default=1)
    if miller == "custom":
        miller = _input("Miller index (e.g. 011 or 1-10)")

    slab_thick = _float("Slab thickness per material (Å)", 15.0)
    gap        = _float("Initial gap between materials (Å)", 2.5)
    vacuum     = _float("Vacuum buffer above/below bilayer (Å)", 10.0)
    n_rand     = _int("Random shuffle variants per temperature", 5)
    surf_scales = _yes("Generate in-plane strain variants (0.90–1.10 scale)?", default=True)

    return {
        "miller_index":     miller,
        "slab_thickness_A": slab_thick,
        "interface_gap_A":  gap,
        "vacuum_A":         vacuum,
        "n_rand_variants":  n_rand,
        "surf_strain_variants": surf_scales,
    }


def _ask_surface_geometry() -> dict:
    """Questions for a surface slab (single material)."""
    _hdr("Surface / Slab Geometry")
    miller = _pick("Surface orientation", MILLER, default=1)
    if miller == "custom":
        miller = _input("Miller index")

    return {
        "miller_index":     miller,
        "min_slab_A":       _float("Minimum slab thickness (Å)", 10.0),
        "min_vacuum_A":     _float("Vacuum above slab (Å)", 15.0),
        "n_rand_variants":  _int("Random surface variants", 5),
    }


def _ask_nanoparticle_geometry() -> dict:
    """Questions for a nanoparticle system."""
    _hdr("Nanoparticle Geometry")
    shape   = _pick("Nanoparticle shape", NP_SHAPES, default=1)
    radius  = _float("Nanoparticle radius (Å)", 15.0)
    vacuum  = _float("Vacuum padding around particle (Å)", 10.0)

    result: dict = {"np_shape": shape, "np_radius_A": radius, "vacuum_A": vacuum}

    if shape == "core_shell":
        result["core_radius_A"]  = _float("Core radius (Å)", 10.0)
        result["shell_thick_A"]  = _float("Shell thickness (Å)", 3.0)
        result["shell_material"] = _input("Shell / coating material name", "AlF3")

    if shape == "wulff":
        print(_dim("  Wulff shape requires surface energies per facet (will use pymatgen WulffShape)"))

    return result


def _ask_polymer_chain() -> dict:
    """Questions for polymer electrolyte / gel."""
    _hdr("Polymer Chain")
    monomer = _pick("Monomer type", MONOMERS, default=1)
    if monomer == "custom":
        monomer_smiles = _input("Monomer SMILES string")
        monomer_mw    = _float("Monomer MW (g/mol)", 100.0)
    else:
        monomer_smiles = ""
        monomer_mw    = 0.0

    chain_len = int(_pick("Chain length (monomers per chain)", [
        ("5",  "5  monomers  — short"),
        ("10", "10 monomers  — medium"),
        ("20", "20 monomers  — standard"),
        ("40", "40 monomers  — long"),
    ], default=3))

    n_chains = int(_pick("Chains per simulation box", [
        ("4",  "4 chains"),
        ("8",  "8 chains  — standard"),
        ("16", "16 chains — large box"),
    ], default=2))

    use_ff = _yes("Run classical MD with force-field (LAMMPS)?", default=True)
    ff_cfg: dict = {}
    if use_ff:
        ff = _pick("Force field", FORCEFIELDS, default=1)
        ff_cfg = {
            "forcefield":       ff,
            "cmd_equil_steps":  _int("NPT equilibration steps", 2_000_000),
            "cmd_prod_steps":   _int("NVT production steps",    5_000_000),
            "cmd_temp_K":       _int("Classical MD temperature (K)", 303),
        }

    return {
        "monomer":        monomer,
        "monomer_smiles": monomer_smiles,
        "monomer_mw_gmol":monomer_mw,
        "chain_length":   chain_len,
        "n_chains":       n_chains,
        "classical_md":   use_ff,
        **ff_cfg,
    }


def _ask_polymer_compositions(monomer_pool: list[str]) -> list[dict]:
    """
    Ask detailed per-polymer composition: chain length, chain count, copolymer ratios.

    Returns list of dicts, one per monomer, stored in simulation.polymers[].
    """
    _hdr("Polymer Composition")
    print()
    print(_dim("  Literature reference (Zhang et al., Nature Energy 2024):"))
    print(_dim("    PVDF-HFP  VDF:HFP = 9:1,  40 chains × 50 monomers"))
    print(_dim("    PTFEP                       5 chains × 20 monomers"))
    print(_dim("    PEO       EO:Li = 16:1,   8 chains × 20 monomers"))
    print()

    # Copolymers that have a secondary monomer ratio
    COPOLYMERS = {
        "PVDF-HFP": ("VDF", "HFP"),
        "PVDF-TrFE": ("VDF", "TrFE"),
        "P(VDF-co-HFP)": ("VDF", "HFP"),
    }

    compositions: list[dict] = []
    for monomer in monomer_pool:
        print(_bold(f"  ── {monomer} ──────────────────────────────────────────"))

        # Copolymer ratio
        cop_ratio: dict = {}
        if monomer in COPOLYMERS:
            ma, mb = COPOLYMERS[monomer]
            print(f"  {_bold(monomer)} is a copolymer of {ma} and {mb}.")
            print(f"  Enter {ma}:{mb} ratio  {_dim('(e.g.  9 1  →  9:1  or  4 1  →  4:1):')}")
            while True:
                try:
                    raw = input(f"  {ma}:{mb} ratio [9 1]: ").strip() or "9 1"
                except (EOFError, KeyboardInterrupt):
                    print(); sys.exit(0)
                parts = raw.split()
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    ra, rb = int(parts[0]), int(parts[1])
                    print(f"  → {_green(f'{ma}:{mb} = {ra}:{rb}')}")
                    cop_ratio = {ma: ra, mb: rb}
                    break
                print(f"  {_yellow('Enter two integers, e.g.  9 1')}")

        # Chain length
        print()
        print(_bold(f"  Chain length — {monomer}  ") +
              _dim("(monomers per chain, or type a number):"))
        print(f"  {_cyan('1')}) 5   monomers — short / AIMD only")
        print(f"  {_cyan('2')}) 10  monomers — medium")
        print(f"  {_cyan('3')}) 20  monomers — standard")
        print(f"  {_cyan('4')}) 40  monomers — long")
        print(f"  {_cyan('5')}) 50  monomers — literature (PVDF-HFP, Zhang 2024)")
        while True:
            try:
                raw = input("  Choice or number [3]: ").strip() or "3"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            presets = {"1": 5, "2": 10, "3": 20, "4": 40, "5": 50}
            if raw in presets:
                chain_len = presets[raw]
                print(f"  → {_green(str(chain_len))}")
                break
            try:
                chain_len = int(raw)
                print(f"  → {_green(str(chain_len))}")
                break
            except ValueError:
                print(f"  {_yellow('Enter a preset number 1–5 or a custom integer')}")

        # Number of chains
        print()
        print(_bold(f"  Chains per simulation box — {monomer}  ") +
              _dim("(number of chains, or type):"))
        print(f"  {_cyan('1')}) 2   chains — minimal / AIMD")
        print(f"  {_cyan('2')}) 4   chains — small")
        print(f"  {_cyan('3')}) 5   chains — PTFEP reference (Zhang 2024)")
        print(f"  {_cyan('4')}) 8   chains — standard")
        print(f"  {_cyan('5')}) 16  chains — large")
        print(f"  {_cyan('6')}) 40  chains — PVDF-HFP reference (Zhang 2024)")
        while True:
            try:
                raw = input("  Choice or number [4]: ").strip() or "4"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            presets = {"1": 2, "2": 4, "3": 5, "4": 8, "5": 16, "6": 40}
            if raw in presets:
                n_chains = presets[raw]
                print(f"  → {_green(str(n_chains))}")
                break
            try:
                n_chains = int(raw)
                print(f"  → {_green(str(n_chains))}")
                break
            except ValueError:
                print(f"  {_yellow('Enter a preset 1–6 or a custom integer')}")

        entry: dict = {
            "monomer":      monomer,
            "chain_length": chain_len,
            "n_chains":     n_chains,
        }
        if cop_ratio:
            entry["copolymer_ratio"] = cop_ratio
            # Store as hfp_fraction for polymer.py build_sequence
            total = sum(cop_ratio.values())
            minor_key = list(cop_ratio.keys())[1]
            entry["minor_fraction"] = cop_ratio[minor_key] / total
        compositions.append(entry)
        print()

    return compositions


# ── Molecular data (MW g/mol, n_atoms, density g/cm³) ───────────────────────
_MOL_INFO: dict[str, tuple[float, int, float]] = {
    "DME":     (90.12,  16, 0.862),   # C4H10O2
    "DOL":     (74.08,  10, 1.060),   # C3H6O2
    "DMB":     (118.17, 34, 0.789),   # C6H14O2 all-atom OPLS-AA (PubChem CID gives 34 atoms)
    "DMC":     (90.08,  10, 1.069),   # C3H6O3
    "EMC":     (104.10, 13, 1.006),   # C4H8O3
    "DEC":     (118.13, 18, 0.975),   # C5H10O3
    "EC":      (88.06,  10, 1.321),   # C3H4O3
    "PC":      (102.09, 12, 1.200),   # C4H6O3
    "FEC":     (106.05, 11, 1.454),   # C3H3FO3
    "TEGDME":  (222.28, 34, 1.009),   # C10H22O5
    "ACN":     (41.05,   6, 0.786),   # C2H3N
    "TMS":     (88.15,  11, 0.848),   # C4H8OS
    "SN":      (80.09,  10, 1.070),   # C4H4N2
    "LiFSI":   (187.07, 10, 1.55),    # LiF2N2O4S2
    "LiPF6":   (151.90,  8, 1.50),    # LiPF6
    "LiTFSI":  (287.08, 17, 1.33),    # LiC2F6N2O4S2 (incl. C,F)
    "LiClO4":  (106.39,  6, 2.42),    # LiClO4
    "LiDFOB":  (139.77, 10, 1.65),    # approx
    "NaFSI":   (203.07, 10, 1.55),
    "NaTFSI":  (303.08, 17, 1.30),
    "NaPF6":   (167.95,  8, 2.02),
    # polymers — per repeat unit (monomer)
    "PEO":     (44.05,   7, 1.20),    # -CH2CH2O-: C2H4O
    "PVDF":    (64.03,   6, 1.78),    # -CH2CF2-: C2H2F2
    "PVDF-HFP":(72.6,    7, 1.78),    # 9:1 VDF:HFP weighted avg per repeat unit
    "PTFEP":   (243.0,  18, 1.60),    # [-N=P(OCH2CF3)2-]
    "PMMA":    (100.12,  9, 1.19),    # -CH2C(CH3)(COOCH3)-
    "PAN":     (53.06,   6, 1.17),    # -CH2CHCN-
    "PVDF-TrFE":(96.04,  7, 1.88),   # VDF:TrFE 75:25 approx
}
_MOL_INFO_DEFAULT = (100.0, 10, 1.0)  # fallback for unknowns

# atoms per monomer unit for DFT autofill — mirrors h00_design._NATOMS_PER_MOL
_DFT_ATOMS_PER_UNIT: dict[str, int] = {
    "DME": 16, "DMB": 34, "DOL": 9,  "EC": 10, "DMC": 10, "EMC": 12, "DEC": 18,
    "PC":  12, "FEC": 10, "VC":  9,  "ACN": 6, "TMS": 11, "SN":  7,
    "GBL": 9,  "DMSO": 6, "THF": 9,  "DEE": 12, "TEGDME": 26,
    "LiFSI": 10, "LiTFSI": 15, "LiPF6": 8, "LiClO4": 6,
    "NaPF6": 8,  "NaFSI": 10,  "LiBF4": 6, "LiDFOB": 9,
    "NaTFSI": 15, "NaClO4": 6,
    "PEO": 8, "PVDF": 5, "PVDF-HFP": 7, "PVDF-TrFE": 7,
    "PMMA": 13, "PTFEP": 18,
}
_DFT_ATOMS_DEFAULT = 12


def _dft_preview(comp: dict, dft_max: int = 300, target: int = 250) -> dict:
    """Compute DFT box preview using the same autofill logic as the orchestrator.

    Returns {"natoms": int, "box_A": float, "density_gcm3": float, "species": dict}.
    5 units of each SM species + budget-derived polymer monomers, capped at dft_max.
    """
    effective = min(target, dft_max)

    # Collect all species: SM + polymer monomers
    sm_names: list[str] = (
        [sv["name"] for sv in comp.get("solvents", [])] +
        [sl["name"] for sl in comp.get("salts",    [])]
    )
    poly_names: list[str] = (
        [p.get("monomer", p.get("name", "")) for p in comp.get("polymers",   [])] +
        [c.get("monomer", c.get("name", "")) for c in comp.get("copolymers", [])]
    )
    poly_names = [m for m in poly_names if m]
    all_names  = sm_names + poly_names
    if not all_names:
        return {"natoms": 0, "box_A": 15.0, "density_gcm3": 1.0, "species": {}}

    aps = {n: _DFT_ATOMS_PER_UNIT.get(n, _DFT_ATOMS_DEFAULT) for n in all_names}
    per_budget = effective // max(1, len(all_names))
    counts = {n: max(1, per_budget // max(1, aps[n])) for n in all_names}
    total  = sum(counts[n] * aps[n] for n in all_names)

    by_cost = sorted(all_names, key=lambda n: aps[n])
    while total < effective:
        added = False
        for n in by_cost:
            if total + aps[n] <= effective:
                counts[n] += 1
                total += aps[n]
                added = True
                break
        if not added:
            break

    # Rough box size: assume ρ ≈ 1.0 g/cm³, avg MW ~10 g/mol per atom
    import math as _math
    mass_g   = total * 10.0 * 1.66054e-24
    rho      = 1.0
    box_A    = max(12.0, (mass_g / rho * 1e24) ** (1.0 / 3.0) * 2.0)  # 2× for packing
    species  = {n: counts[n] for n in sm_names if counts.get(n, 0) > 0}
    chains   = {n: 1 for n in poly_names if counts.get(n, 0) > 0}

    return {"natoms": total, "box_A": box_A, "density_gcm3": rho,
            "species": species, "chains": chains}


def _compute_box_spec(
    fractions:     dict[str, float],   # {"solv:DME": 45.0, "salt:LiFSI": 15.0, "poly:PVDF-HFP": 40.0, ...}
    target_natoms: int,                # target total atom count
    polymer_comps: list[dict],         # from _ask_polymer_compositions
) -> dict:
    """
    Given volume fractions (%) and a target atom count, return box specification:
    {
      "natoms":  int,
      "box_A":   float,              # cubic box side (Å)
      "species": {"DME": n, ...},    # molecule / monomer count
      "chains":  {"PVDF-HFP": n},   # chain count per polymer
      "density_gcm3": float,
    }
    Volume fractions for polymers and solvents are treated as true volume %.
    For salts they are treated as weight % (converted via density).
    """
    import math

    # Normalise and separate by type
    poly_fracs:  dict[str, float] = {}
    solv_fracs:  dict[str, float] = {}
    salt_wt_pcts: dict[str, float] = {}

    for key, pct in fractions.items():
        if pct <= 0:
            continue
        tag, name = key.split(":", 1)
        if tag == "poly":
            poly_fracs[name] = pct / 100.0
        elif tag == "solv":
            solv_fracs[name] = pct / 100.0
        elif tag == "salt":
            salt_wt_pcts[name] = pct / 100.0   # weight fraction in remaining liquid

    # --- Step 1: compute volume-average density (for volume-fraction species only)
    # vol fraction = polymer + solvent (salt is handled separately by wt%)
    vol_total_frac = sum(poly_fracs.values()) + sum(solv_fracs.values())
    if vol_total_frac <= 0:
        vol_total_frac = 1.0

    # Effective density of the vol-fraction mixture (g/cm³)
    rho_mix = 0.0
    for name, f in {**poly_fracs, **solv_fracs}.items():
        _, _, rho = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        rho_mix += (f / vol_total_frac) * rho

    # --- Step 2: compute V_box from target atom count
    # atoms per cm³ for the mixture
    vol_atoms_per_cm3 = 0.0
    for name, f in {**poly_fracs, **solv_fracs}.items():
        mw, nat, rho = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        vol_atoms_per_cm3 += f * rho * 6.022e23 * nat / (mw * 1e24)  # atoms/Å³

    # Add salt contribution (wt% of liquid phase, approximate)
    # Approximate: salt_vol_frac ≈ salt_wt_frac * rho_liq / rho_salt
    for name, wf in salt_wt_pcts.items():
        mw, nat, rho_s = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        vol_atoms_per_cm3 += wf * rho_mix / rho_s * rho_s * 6.022e23 * nat / (mw * 1e24)

    if vol_atoms_per_cm3 <= 0:
        vol_atoms_per_cm3 = 0.05   # fallback: ~50 atoms/nm³

    # V_box in Å³
    V_box = target_natoms / vol_atoms_per_cm3
    box_A = V_box ** (1.0 / 3.0)

    # --- Step 3: compute molecule counts
    species: dict[str, int] = {}
    chains:  dict[str, int] = {}
    natoms_actual = 0

    for name, f in solv_fracs.items():
        mw, nat, rho = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        n = max(1, round(f * V_box * rho / (mw / 6.022e23 * 1e24)))
        species[name] = n
        natoms_actual += n * nat

    for name, wf in salt_wt_pcts.items():
        mw, nat, rho_s = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        n = max(1, round(wf * rho_mix * V_box / (mw / 6.022e23 * 1e24)))
        species[name] = n
        natoms_actual += n * nat

    for name, f in poly_fracs.items():
        mw, nat, rho = _MOL_INFO.get(name, _MOL_INFO_DEFAULT)
        n_monomers = max(1, round(f * V_box * rho / (mw / 6.022e23 * 1e24)))
        species[name] = n_monomers  # total repeat units
        natoms_actual += n_monomers * nat
        # chain count from polymer_comps or default chain_length=20
        chain_len = next(
            (pc.get("chain_length", 20) for pc in polymer_comps if pc["monomer"] == name),
            20)
        chains[name] = max(1, round(n_monomers / chain_len))

    return {
        "natoms":       natoms_actual,
        "box_A":        box_A,
        "species":      species,
        "chains":       chains,
        "density_gcm3": rho_mix,
    }


def _ask_box_composition_and_tiers(
    sv_pool:       list[str],
    salt_pool:     list[str],
    monomer_pool:  list[str],
    polymer_comps: list[dict],
) -> dict:
    """
    Replace _ask_liquid_cell():
    1. Ask volume % per polymer + solvent, weight % per salt
    2. Compute POSCAR atom counts for each simulation tier
    3. Show preview table, let user adjust tier atom targets
    4. User selects which tiers to build
    5. Returns a sim-config dict compatible with project.yaml
    """
    _hdr("Box Composition")
    print()
    print(_dim("  Enter volume % for each polymer and solvent,"))
    print(_dim("  and weight % for each salt. Must sum to 100%."))
    print()

    fractions: dict[str, float] = {}

    # ── polymers ─────────────────────────────────────────────────────────────
    if monomer_pool:
        print(_bold("  Polymers  ") + _dim("(volume %):"))
        _defaults_poly = {"PVDF-HFP": 30.0, "PTFEP": 10.0, "PEO": 50.0, "PVDF": 20.0, "PMMA": 15.0}
        for m in monomer_pool:
            d = _defaults_poly.get(m, 20.0)
            val = _float(f"    {_bold(m)}  vol%", d)
            fractions[f"poly:{m}"] = val

    # ── solvents ─────────────────────────────────────────────────────────────
    if sv_pool:
        print()
        used_vol = sum(v for k, v in fractions.items() if k.startswith("poly:"))
        remaining_vol = max(0.0, 100.0 - used_vol)
        per_solv = round(remaining_vol / len(sv_pool), 1) if sv_pool else 0.0
        print(_bold("  Solvents  ") + _dim("(volume %):"))
        _defaults_solv = {"DME": 40.0, "DOL": 20.0, "DMB": 35.0, "EC": 30.0, "DMC": 30.0,
                          "EMC": 30.0, "DEC": 30.0, "PC": 30.0, "TEGDME": 25.0, "FEC": 5.0, "ACN": 40.0}
        for sv in sv_pool:
            d = min(_defaults_solv.get(sv, per_solv), remaining_vol)
            val = _float(f"    {_bold(sv)}  vol%", round(d, 1))
            fractions[f"solv:{sv}"] = val

    # ── salts ─────────────────────────────────────────────────────────────────
    if salt_pool:
        print()
        print(_bold("  Salts  ") + _dim("(weight % of total box):"))
        _defaults_salt = {"LiFSI": 15.0, "LiPF6": 12.0, "LiTFSI": 15.0, "LiClO4": 10.0,
                          "NaFSI": 15.0, "NaTFSI": 15.0, "NaPF6": 12.0}
        for salt in salt_pool:
            d = _defaults_salt.get(salt, 12.0)
            val = _float(f"    {_bold(salt)}  wt%", d)
            fractions[f"salt:{salt}"] = val

    # ── validate sum ──────────────────────────────────────────────────────────
    vol_sum  = sum(v for k, v in fractions.items() if not k.startswith("salt:"))
    salt_sum = sum(v for k, v in fractions.items() if k.startswith("salt:"))
    print()
    print(f"  {'Volumes':8} polymer+solvent: {_cyan(f'{vol_sum:.1f}%')}  "
          f"salt weight: {_cyan(f'{salt_sum:.1f}%')}")
    if abs(vol_sum - 100.0) > 1.0:
        print(f"  {_yellow('Warning:')} volume fractions sum to {vol_sum:.1f}% "
              f"(expected ~100%). Normalizing automatically.")

    # ── simulation tiers ──────────────────────────────────────────────────────
    _hdr("Simulation Tiers & Box Preview")
    print()
    print(_bold("  Choose target atom counts for each simulation tier:"))
    print(_dim("  AIMD   : ≤ 300     atoms  (VASP DFT — hard limit; auto-filled to ~250)"))
    print(_dim("  MLMD   : 5k–10k   atoms  (LAMMPS + DeepMD, medium box)"))
    print(_dim("  CMD    : 40k–50k  atoms  (LAMMPS + classical FF, large box)"))
    print()

    _ld0 = _lane_defaults()
    mlmd_target = _ld0["mlmd_atoms"]
    cmd_target  = _ld0["cmd_atoms"]

    # ── compute tiers ─────────────────────────────────────────────────────────
    # DFT: auto-filled by orchestrator — preview uses _dft_preview, not _compute_box_spec
    _poly_for_dft = [{"monomer": m.split(":")[1]} for m in fractions if m.startswith("poly:")]
    tier_specs: dict[str, dict] = {
        "AIMD": _dft_preview(
            {"solvents": [{"name": k.split(":")[1]} for k in fractions if k.startswith("solv:")],
             "salts":    [{"name": k.split(":")[1]} for k in fractions if k.startswith("salt:")],
             "polymers": _poly_for_dft},
            dft_max=_ld0["aimd_atoms"], target=250,
        ),
    }
    for tier, target in [("MLMD", mlmd_target), ("CMD", cmd_target)]:
        spec = _compute_box_spec(fractions, target, polymer_comps)
        tier_specs[tier] = spec

    # ── preview table ─────────────────────────────────────────────────────────
    all_species = sorted(
        {sp for s in tier_specs.values() for sp in s["species"]}
    )
    all_polymers = sorted(
        {ch for s in tier_specs.values() for ch in s["chains"]}
    )

    col_w = max(8, *(len(sp) for sp in all_species + all_polymers), 0) + 2

    print()
    print(_bold("  ── Computed box composition ──────────────────────────────────"))
    header = f"  {'Tier':<6}  {'Atoms':>7}  {'Box(Å)':>7}  {'ρ(g/cm³)':>9}  "
    for sp in all_species:
        header += f"{sp:>{col_w}}"
    for ch in all_polymers:
        header += f"{'chains:'+ch:>{col_w + 7}}"
    print(_bold(header))
    print("  " + "─" * (len(header) - 2))
    for tier, spec in tier_specs.items():
        row = f"  {_cyan(tier):<6}  {spec['natoms']:>7,}  {spec['box_A']:>7.1f}  "
        row += f"{spec['density_gcm3']:>9.3f}  "
        for sp in all_species:
            row += f"{spec['species'].get(sp, 0):>{col_w}}"
        for ch in all_polymers:
            row += f"{spec['chains'].get(ch, 0):>{col_w + 7}}"
        print(row)
    print()

    # ── select tiers ─────────────────────────────────────────────────────────
    print(_bold("  Select tiers to build  ") +
          _dim("(space-separated: 1=AIMD  2=MLMD  3=CMD, e.g.  1 2 3):"))
    print(f"  {_cyan('1')}) AIMD   — small box,  {tier_specs['AIMD']['natoms']:,} atoms,  {tier_specs['AIMD']['box_A']:.0f} Å")
    print(f"  {_cyan('2')}) MLMD   — medium box, {tier_specs['MLMD']['natoms']:,} atoms, {tier_specs['MLMD']['box_A']:.0f} Å")
    print(f"  {_cyan('3')}) CMD    — large box,  {tier_specs['CMD']['natoms']:,} atoms,  {tier_specs['CMD']['box_A']:.0f} Å")
    print(f"  {_cyan('4')}) All three tiers")
    while True:
        try:
            raw = input("  Tiers [1 2 3]: ").strip() or "1 2 3"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw == "4":
            selected = ["AIMD", "MLMD", "CMD"]
        else:
            selected = []
            for tok in raw.split():
                if tok == "1" and "AIMD" not in selected: selected.append("AIMD")
                elif tok == "2" and "MLMD" not in selected: selected.append("MLMD")
                elif tok == "3" and "CMD" not in selected: selected.append("CMD")
        if selected:
            print(f"  → {_green(', '.join(selected))}")
            break
        print(f"  {_yellow('Select at least one tier')}")

    # ── temperatures & run lengths ────────────────────────────────────────────
    run_cmd = "CMD" in selected
    run_mlmd = "MLMD" in selected
    run_aimd = "AIMD" in selected

    mlmd_temps:     list[int] = []
    mlmd_steps:     int = 1_000_000
    cmd_equil_steps: int = 2_000_000
    cmd_prod_steps:  int = 10_000_000
    cmd_temps:       list[int] = []
    aimd_temps_sel:  list[int] = []
    aimd_steps:     int = 3_000

    _ld_t = _lane_defaults()
    if run_aimd:
        aimd_temps_sel = [300, 400, 500]
        aimd_steps     = _ld_t.get("aimd_steps", 3_000)

    if run_mlmd:
        mlmd_temps = _ld_t.get("nvt_temperatures", [300, 320, 340, 360, 380, 400, 500, 600])
        mlmd_steps = _ld_t.get("mlmd_steps", 1_000_000)

    if run_cmd:
        ff = _pick("Force field for CMD", FORCEFIELDS, default=1)
        cmd_equil_steps = _ld_t.get("npt_steps", 2_000_000)
        cmd_prod_steps  = _ld_t.get("nvt_steps", 10_000_000)
        cmd_temps = _ld_t.get("nvt_temperatures", [300, 320, 340, 360, 380, 400, 500, 600])
    else:
        ff = "OPLS-AA"

    # ── build return dict ─────────────────────────────────────────────────────
    result: dict = {
        "box_fractions":         fractions,
        "tiers_selected":        selected,
        "target_density_gcm3":   tier_specs["AIMD"]["density_gcm3"],
        "tier_aimd":             tier_specs.get("AIMD", {}),
        "tier_mlmd":             tier_specs.get("MLMD", {}),
        "tier_cmd":              tier_specs.get("CMD", {}),
        # molecule counts for the AIMD cell (small box reference)
        "n_molecules_aimd":      tier_specs["AIMD"]["natoms"] // max(1, min(
            _MOL_INFO.get(sp, _MOL_INFO_DEFAULT)[1]
            for sp in tier_specs["AIMD"]["species"]
        ) if tier_specs["AIMD"]["species"] else 1),
        "n_molecules_mlmd":      tier_specs["MLMD"]["natoms"] // max(1, min(
            _MOL_INFO.get(sp, _MOL_INFO_DEFAULT)[1]
            for sp in tier_specs["MLMD"]["species"]
        ) if tier_specs["MLMD"]["species"] else 1),
        "n_molecules_cmd":       tier_specs["CMD"]["natoms"] // max(1, min(
            _MOL_INFO.get(sp, _MOL_INFO_DEFAULT)[1]
            for sp in tier_specs["CMD"]["species"]
        ) if tier_specs["CMD"]["species"] else 1),
        "classical_md":          run_cmd,
        "forcefield":            ff,
        "aimd_temps":            aimd_temps_sel,
        "aimd_steps":            aimd_steps,
        "mlmd_temps":            mlmd_temps,
        "mlmd_steps":            mlmd_steps,
        "cmd_equil_steps":       cmd_equil_steps,
        "cmd_prod_steps":        cmd_prod_steps,
        "cmd_temps":             cmd_temps,
        # explicit molecule counts per tier for h00_design
        "molecule_counts_aimd":  tier_specs.get("AIMD", {}).get("species", {}),
        "molecule_counts_mlmd":  tier_specs.get("MLMD", {}).get("species", {}),
        "molecule_counts_cmd":   tier_specs.get("CMD", {}).get("species", {}),
        "chain_counts_aimd":     tier_specs.get("AIMD", {}).get("chains", {}),
        "chain_counts_mlmd":     tier_specs.get("MLMD", {}).get("chains", {}),
        "chain_counts_cmd":      tier_specs.get("CMD", {}).get("chains", {}),
    }
    return result


def _ask_molecule_counts(sv_pool: list[str], salt_pool: list[str]) -> dict | None:
    """
    Ask whether to specify exact molecule counts or compute from concentration/density.

    Returns dict {molecule_name: count} or None (use auto/concentration mode).
    """
    _hdr("Simulation Box Composition")
    print()
    print(_dim("  Literature reference (Zhang et al., Nature Energy 2024):"))
    print(_dim("    DME 930 molecules + LiFSI 1170 molecules  (~4 M LiFSI/DME)"))
    print(_dim("    This gives a high-concentration electrolyte (HCE) inside the gel."))
    print()
    print(_bold("  Molecule count mode:"))
    print(f"  {_cyan('1')}) {_bold('Auto')}   — compute from concentration + box density (standard)")
    print(f"  {_cyan('2')}) {_bold('Manual')} — enter exact molecule counts per species")
    while True:
        try:
            raw = input("  Choice [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in ("1", "2"):
            break
        print(f"  {_yellow('Enter 1 or 2')}")

    if raw == "1":
        return None  # auto

    # Manual counts
    counts: dict = {}
    all_species = list(sv_pool) + list(salt_pool)
    for species in all_species:
        # Suggest a literature default
        defaults = {"DME": 930, "DOL": 800, "DMB": 700, "EC": 600, "DMC": 600,
                    "EMC": 600, "DEC": 600, "PC": 600, "FEC": 400, "TEGDME": 300,
                    "LiFSI": 1170, "LiPF6": 800, "LiTFSI": 600, "LiClO4": 800,
                    "NaFSI": 600, "NaTFSI": 400, "NaPF6": 400}
        default_n = defaults.get(species, 100)
        n = _int(f"  Number of {_bold(species)} molecules", default_n)
        counts[species] = n

    print()
    print(_bold("  Box composition:"))
    for sp, n in counts.items():
        print(f"    {_cyan(sp):<12}  {n} molecules")
    return counts


def _ask_polymer_chain_from_pool(primary_monomer: str = "PEO") -> dict:
    """Ask chain/box/FF settings when monomer is already chosen — also asks FF."""
    chain_len = int(_pick("Chain length (monomers per chain)", [
        ("5",  "5  monomers  — short"),
        ("10", "10 monomers  — medium"),
        ("20", "20 monomers  — standard"),
        ("40", "40 monomers  — long"),
    ], default=3))

    n_chains = int(_pick("Chains per simulation box", [
        ("4",  "4 chains"),
        ("8",  "8 chains  — standard"),
        ("16", "16 chains — large box"),
    ], default=2))

    use_ff = _yes("Run classical MD with force-field (LAMMPS)?", default=True)
    ff_cfg: dict = {}
    if use_ff:
        ff = _pick("Force field", FORCEFIELDS, default=1)
        ff_cfg = {
            "forcefield":       ff,
            "cmd_equil_steps":  _int("NPT equilibration steps", 2_000_000),
            "cmd_prod_steps":   _int("NVT production steps",    5_000_000),
            "cmd_temp_K":       _int("Classical MD temperature (K)", 303),
        }

    return {
        "monomer":      primary_monomer,
        "chain_length": chain_len,
        "n_chains":     n_chains,
        "classical_md": use_ff,
        **ff_cfg,
    }


def _ask_solid_supercell() -> dict:
    """AIMD and MLMD cell sizes for solid-state materials."""
    import re as _re
    _hdr("Supercell Sizes")
    _AIMD_SC_OPTIONS = [
        ("1x1x1", "1×1×1  — unit cell (very small)"),
        ("2x2x2", "2×2×2  — standard AIMD (~50–200 atoms)"),
        ("3x3x3", "3×3×3  — large AIMD (~200–500 atoms)"),
        ("2x2x4", "2×2×4  — elongated slab / migration path"),
        ("custom", "custom — type any supercell (e.g. 1x1x2, 2x3x1)"),
    ]
    aimd_sc_raw = _pick("AIMD supercell (from relaxed unit cell)", _AIMD_SC_OPTIONS, default=2)
    if aimd_sc_raw == "custom":
        while True:
            try:
                raw = input("  Enter custom supercell (e.g. 1x1x2): ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if _re.fullmatch(r"\d+x\d+x\d+", raw):
                aimd_sc = raw
                print(f"  → {_green(aimd_sc)}")
                break
            print(f"  {_yellow('Format must be NxNxN, e.g. 1x1x2')}")
    else:
        aimd_sc = aimd_sc_raw

    mlmd_n = _pick("MLMD supercell target atom count", [
        ("500",   "~500 atoms   — fast"),
        ("1000",  "~1000 atoms  — standard"),
        ("5000",  "~5000 atoms  — high accuracy"),
        ("10000", "~10000 atoms — production"),
        ("22000", "~22000 atoms — NMC622-scale"),
    ], default=2)

    return {"aimd_supercell": aimd_sc, "mlmd_natoms_target": int(mlmd_n)}


def _ask_solid_temps(existing_aimd: list[int], existing_mlmd: list[int]) -> dict:
    """Prompt for AIMD and MLMD temperature sweeps and step counts for solid SSE projects."""
    _hdr("Temperature Sweeps")
    _ld = _lane_defaults()
    aimd_temps = _pick_temps("AIMD temperatures", TEMPERATURES_AIMD, existing_aimd)
    aimd_steps = _int("AIMD steps per temperature (NSW)", _ld["aimd_steps"])
    mlmd_temps = _pick_temps("MLMD temperatures", TEMPERATURES_MLMD, existing_mlmd)
    mlmd_steps = _int("MLMD run length (steps)", _ld["mlmd_steps"])
    return {
        "aimd_temps": aimd_temps, "aimd_steps": aimd_steps,
        "mlmd_temps": mlmd_temps, "mlmd_steps": mlmd_steps,
    }




def _encut_from_elements(elements: list[str]) -> float:
    """Return 1.3 × max(ENMAX) by reading per-element POTCARs from POTPAW_DIR.

    Falls back gracefully if a POTCAR cannot be read.
    """
    try:
        from .h02_aimd_constants import POTPAW_DIR, _PP_PREF  # type: ignore
    except ImportError:
        try:
            from hpca.orchestrator.handlers.h02_aimd_constants import POTPAW_DIR, _PP_PREF
        except ImportError:
            return 0.0

    enmax_vals: list[float] = []
    for el in elements:
        pp_dir = _PP_PREF.get(el, el)
        potcar = POTPAW_DIR / pp_dir / "POTCAR"
        if not potcar.exists():
            print(f"  {_yellow(f'POTCAR not found for {el} ({potcar}) — excluded from ENCUT')}")
            continue
        for line in potcar.read_text(errors="replace").splitlines():
            if "ENMAX" in line:
                for part in line.split(";"):
                    if "ENMAX" in part:
                        try:
                            enmax_vals.append(float(part.split("=")[1].strip().split()[0]))
                        except (ValueError, IndexError):
                            pass
                break
    if not enmax_vals:
        return 0.0
    encut = round(1.3 * max(enmax_vals), 1)
    return encut


def _parse_pcts(raw: str) -> list[float]:
    """Parse space-separated percentage values from user input."""
    pcts = []
    for tok in raw.split():
        try:
            v = float(tok)
            if 0 < v <= 100:
                pcts.append(v)
        except ValueError:
            pass
    return sorted(set(pcts))


def _make_pct_tag(pct: float) -> str:
    """Format a percentage as a compact directory-safe tag (e.g. 5.0 → '05', 2.5 → '2p5')."""
    return f"{int(pct):02d}" if pct == int(pct) else f"{pct:.1f}".replace(".", "p")


def _add_mono_variants(variants: list, dopant_elements: list,
                       project_name: str, host_el: str, n_sites: int,
                       dopant: str, pcts: list[float]) -> None:
    """Append mono-dopant variant entries; update dopant_elements list in place."""
    if dopant not in dopant_elements:
        dopant_elements.append(dopant)
    for pct in pcts:
        n_sub = max(1, round(pct / 100.0 * n_sites)) if n_sites else max(1, round(pct))
        actual_pct = n_sub / n_sites * 100.0 if n_sites else pct
        # Directory names and metadata use the realizable concentration, not
        # the requested value, which may be impossible in a finite cell.
        vname = f"{project_name}_{dopant}{_make_pct_tag(actual_pct)}"
        variants.append({
            "name":            vname,
            "host_element":    host_el,
            "dopant_element":  dopant,
            "n_substitutions": n_sub,
            "requested_pct":   pct,
            "actual_pct":      actual_pct,
            "host_site_count": n_sites,
        })
        note = f" = {n_sub}/{n_sites} sites  actual {actual_pct:.1f}%" if n_sites else ""
        print(f"    {_green('+')} {_cyan(vname)}  {pct}%{note}")


def _ask_doping_variants(project_name: str, struct_path: str = "",
                         aimd_supercell: str = "1x1x1") -> tuple[list, float]:
    """Ask user to define substitution doping variants for SSE projects.

    Per-element Y/N prompts. For each doped site: ask dopant element(s).
    If multiple elements entered → ask mono / di / trinary.
      Mono:    one element, one sub-series per percentage.
      Di:      two elements co-doped simultaneously (1:1 or custom ratio).
      Trinary: three elements co-doped simultaneously.
    ENCUT computed from all base + dopant elements via POTPAW POTCAR files.

    Returns (variants_list, encut).
    """
    _hdr("Doping Variants")
    if not _yes("Create crystal doping variants (substitution doping)?", default=True):
        return [], 0.0

    # ── Load structure ────────────────────────────────────────────────────────
    el_counts: dict[str, int] = {}
    base_elements: list[str] = []
    source_structure = None
    if struct_path:
        try:
            from pymatgen.core import Structure as _Struct
            from collections import Counter
            _s   = _Struct.from_file(struct_path)
            factors = [int(x) for x in aimd_supercell.lower().split("x")]
            if len(factors) == 3 and factors != [1, 1, 1]:
                _s.make_supercell(factors)
            source_structure = _s
            _cnt = Counter(str(e) for e in _s.species)
            el_counts     = dict(_cnt)
            base_elements = list(_cnt.keys())
            total = sum(_cnt.values())
            print(f"\n  Structure: {_s.composition.reduced_formula}  ({total} atoms total)")
            for el, n in _cnt.items():
                print(f"    {_cyan(el):6s} {n:4d} sites")
        except Exception as exc:
            print(f"  {_yellow(f'Could not read structure ({exc}) — enter counts manually')}")

    variants: list[dict]  = []
    dopant_elements: list[str] = []

    # ── Pure baseline ─────────────────────────────────────────────────────────
    if _yes(f"\n  Include undoped baseline ({project_name}_pure)?", default=True):
        variants.append({
            "name": f"{project_name}_pure",
            "host_element": None, "dopant_element": None, "n_substitutions": 0,
        })

    # ── Element list to iterate ───────────────────────────────────────────────
    prompt_elements = base_elements or []
    if not prompt_elements:
        raw = _input("\n  Elements to consider for doping (space-separated, e.g. Cl Y)")
        prompt_elements = raw.split()

    for el in prompt_elements:
        n_sites = el_counts.get(el, 0)
        site_hint = f" ({n_sites} sites)" if n_sites else ""
        print()
        if not _yes(f"  Dope {_cyan(el)}{site_hint}?", default=False):
            continue

        raw_dopant = _input(
            f"    Dopant element(s) replacing {el}  "
            f"{_dim('(one: F   or multiple: In Yb Zr Er Hf)')}"
        ).strip()
        if not raw_dopant:
            print(f"    {_yellow('No dopant — skipping')}")
            continue

        tokens = raw_dopant.split()

        if len(tokens) == 1:
            # ── Mono dopant ───────────────────────────────────────────────────
            dopant = tokens[0]
            hint = f"out of {n_sites} sites  " if n_sites else ""
            raw_pct = _input(
                f"    Doping % of {el} sites  {_dim(f'({hint}e.g. 5 10 20 50)')}"
            ).strip()
            pcts = _parse_pcts(raw_pct)
            if not pcts:
                print(f"    {_yellow('No percentages — skipping')}")
                continue
            _add_mono_variants(variants, dopant_elements,
                               project_name, el, n_sites, dopant, pcts)

        else:
            # ── Multiple elements: ask which doping mode(s) to create ─────────
            print(f"\n    Detected {len(tokens)} elements: {', '.join(tokens)}")
            print(f"    {_cyan('1')}  Separate mono-dopants  "
                  f"{_dim(f'— one sub-series per element ({len(tokens)} series)')}")
            print(f"    {_cyan('2')}  Di-dopant (binary)     "
                  f"— all C({len(tokens)},2)={len(tokens)*(len(tokens)-1)//2} pairs, choose ratio(s)")
            print(f"    {_cyan('3')}  Trinary dopant         "
                  f"— all C({len(tokens)},3)={len(tokens)*(len(tokens)-1)*(len(tokens)-2)//6} triplets, choose ratio")
            print(f"    {_dim('Select multiple e.g. 1 2  to create both mono and di variants')}")
            dtype_raw = _input("    Dopant type(s)", "1").strip()
            selected_types = {tok for tok in dtype_raw.split() if tok in ("1", "2", "3")}
            if not selected_types:
                selected_types = {"1"}

            # Shared total-substitution percentages for all types
            hint = f"total {el} substitution,  " if n_sites else ""
            raw_pct = _input(f"    Doping %  {_dim(f'({hint}e.g. 25 50 100)')}").strip()
            pcts = _parse_pcts(raw_pct)
            if not pcts:
                print(f"    {_yellow('No percentages — skipping')}")
                continue

            # ── Type 1: separate mono-dopants ─────────────────────────────────
            if "1" in selected_types:
                for tok in tokens:
                    _add_mono_variants(variants, dopant_elements,
                                       project_name, el, n_sites, tok, pcts)

            # ── Type 2: di-dopant — all pairs, preset ratios ──────────────────
            if "2" in selected_types:
                from itertools import combinations as _combs
                _DI_RATIOS = [
                    ("1:1", 1, 1, "equal parts"),
                    ("1:2", 1, 2, "El1 minor, El2 major"),
                    ("2:1", 2, 1, "El1 major, El2 minor"),
                    ("1:3", 1, 3, "El1 minor, El2 heavy"),
                    ("3:1", 3, 1, "El1 heavy, El2 minor"),
                ]
                print(f"\n    {_bold('Di co-doping — select ratio(s) between each pair:')}")
                for ri, (rl, r1, r2, rdesc) in enumerate(_DI_RATIOS, 1):
                    print(f"    {_cyan(str(ri))}) {rl:6s}  — {rdesc}")
                print(f"    {_dim('Select one or more, e.g. 1 3  for 1:1 and 2:1')}")
                raw_rsel = _input("    Ratio(s)", "1").strip()
                sel_ratios = [
                    _DI_RATIOS[int(t)-1]
                    for t in raw_rsel.split()
                    if t.isdigit() and 1 <= int(t) <= len(_DI_RATIOS)
                ] or [_DI_RATIOS[0]]

                pairs = list(_combs(tokens, 2))
                for rl, r1, r2, _ in sel_ratios:
                    rsum = r1 + r2
                    print(f"\n    {_dim(f'Ratio {rl} — {len(pairs)} pairs:')}")
                    for el1, el2 in pairs:
                        for pct in pcts:
                            total_n = max(1, round(pct/100.0*n_sites)) if n_sites else max(1, round(pct))
                            n1 = max(0, round(r1/rsum * total_n))
                            n2 = total_n - n1
                            p1 = round(n1/n_sites*100) if n_sites else round(r1/rsum*pct)
                            p2 = round(n2/n_sites*100) if n_sites else round(r2/rsum*pct)
                            vname = f"{project_name}_{el1}{p1:02d}{el2}{p2:02d}"
                            dopant_entries = [
                                {"element": el1, "n_substitutions": n1},
                                {"element": el2, "n_substitutions": n2},
                            ]
                            for e in dopant_entries:
                                if e["element"] not in dopant_elements:
                                    dopant_elements.append(e["element"])
                            variants.append({"name": vname, "host_element": el,
                                             "dopant_elements": dopant_entries})
                            actual = (n1+n2)/n_sites*100 if n_sites else pct
                            note = f" = {n1+n2}/{n_sites} {el} sites  actual {actual:.1f}%" if n_sites else ""
                            print(f"    {_green('+')} {_cyan(vname)}  "
                                  f"({n1} {el1} + {n2} {el2}){note}")

            # ── Type 3: trinary — all triplets, single ratio ──────────────────
            if "3" in selected_types:
                from itertools import combinations as _combs3
                triplets = list(_combs3(tokens, 3))
                default_ratio3 = "1 1 1"
                raw_ratio3 = _input(
                    f"    Trinary ratio  {_dim('(e.g. 1 1 1 = equal, 1 1 2 = last heavier)')}",
                    default=default_ratio3
                ).strip()
                rparts3: list[float] = []
                for tok in raw_ratio3.split():
                    try:
                        rparts3.append(float(tok))
                    except ValueError:
                        pass
                if len(rparts3) != 3:
                    rparts3 = [1.0, 1.0, 1.0]
                rsum3 = sum(rparts3)
                rl3 = ":".join(str(int(r)) if r == int(r) else str(r) for r in rparts3)

                print(f"\n    {_dim(f'Ratio {rl3} — {len(triplets)} triplets:')}")
                for el1, el2, el3 in triplets:
                    for pct in pcts:
                        total_n = max(1, round(pct/100.0*n_sites)) if n_sites else max(1, round(pct))
                        n1 = max(0, round(rparts3[0]/rsum3 * total_n))
                        n2 = max(0, round(rparts3[1]/rsum3 * total_n))
                        n3 = total_n - n1 - n2
                        p1 = round(n1/n_sites*100) if n_sites else round(rparts3[0]/rsum3*pct)
                        p2 = round(n2/n_sites*100) if n_sites else round(rparts3[1]/rsum3*pct)
                        p3 = round(n3/n_sites*100) if n_sites else round(rparts3[2]/rsum3*pct)
                        vname = f"{project_name}_{el1}{p1:02d}{el2}{p2:02d}{el3}{p3:02d}"
                        dopant_entries = [
                            {"element": el1, "n_substitutions": n1},
                            {"element": el2, "n_substitutions": n2},
                            {"element": el3, "n_substitutions": n3},
                        ]
                        for e in dopant_entries:
                            if e["element"] not in dopant_elements:
                                dopant_elements.append(e["element"])
                        variants.append({"name": vname, "host_element": el,
                                         "dopant_elements": dopant_entries})
                        actual = (n1+n2+n3)/n_sites*100 if n_sites else pct
                        note = f" = {n1+n2+n3}/{n_sites} {el} sites  actual {actual:.1f}%" if n_sites else ""
                        print(f"    {_green('+')} {_cyan(vname)}  "
                              f"({n1} {el1} + {n2} {el2} + {n3} {el3}){note}")

    # Expand mono-dopant entries into up to three geometrically distinct site
    # configurations. Explicit indices make generation deterministic and
    # auditable instead of relying on Python's randomized hash seed.
    if source_structure is not None:
        import random as _random
        expanded: list[dict] = []
        for variant in variants:
            host = variant.get("host_element")
            n_sub = int(variant.get("n_substitutions", 0))
            if not host or n_sub <= 0 or variant.get("dopant_elements"):
                expanded.append(variant)
                continue
            host_indices = [i for i, site in enumerate(source_structure)
                            if site.species_string == host]
            rng = _random.Random(f"{project_name}:{host}:{n_sub}")
            candidates: list[tuple[int, ...]] = []
            if n_sub == 1:
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    sym = SpacegroupAnalyzer(source_structure).get_symmetrized_structure()
                    candidates = [tuple([grp[0]]) for grp in sym.equivalent_indices
                                  if grp and grp[0] in host_indices]
                except Exception:
                    candidates = [(i,) for i in host_indices]
            else:
                seen: set[tuple[int, ...]] = set()
                for _ in range(min(300, max(30, len(host_indices) * 3))):
                    pick = tuple(sorted(rng.sample(host_indices, n_sub)))
                    if pick not in seen:
                        seen.add(pick)
                        candidates.append(pick)
            # Deduplicate multi-site configurations by dopant-pair distance signature.
            unique: list[tuple[int, ...]] = []
            signatures: set[tuple[float, ...]] = set()
            for pick in candidates:
                sig = tuple(sorted(round(source_structure.get_distance(i, j), 3)
                                   for x, i in enumerate(pick) for j in pick[x + 1:]))
                if sig not in signatures:
                    signatures.add(sig)
                    unique.append(pick)
                if len(unique) == 3:
                    break
            unique = unique or [tuple(host_indices[:n_sub])]
            for cfg_i, indices in enumerate(unique, 1):
                suffix = f"_cfg{cfg_i:02d}" if len(unique) > 1 else ""
                expanded.append({**variant, "name": variant["name"] + suffix,
                                 "configuration_index": cfg_i,
                                 "substitution_indices": list(indices)})
        variants = expanded

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {_bold(f'{len(variants)} sub-project(s) will be created:')}")
    for v in variants:
        if not v.get("host_element"):
            print(f"    {_cyan(v['name'])}  — pure")
        elif "dopant_elements" in v:
            detail = " + ".join(
                f"{e['n_substitutions']} {e['element']}"
                for e in v["dopant_elements"]
            )
            tot = el_counts.get(v["host_element"], 0)
            total_n = sum(e["n_substitutions"] for e in v["dopant_elements"])
            pct_s = f" ({total_n}/{tot} = {total_n/tot*100:.1f}%)" if tot else ""
            print(f"    {_cyan(v['name'])}  — {v['host_element']}→[{detail}]{pct_s}")
        else:
            n = v["n_substitutions"]
            tot = el_counts.get(v["host_element"] or "", 0)
            pct_s = f" ({n}/{tot} = {n/tot*100:.1f}%)" if tot else f" ({n} sites)"
            print(f"    {_cyan(v['name'])}  — {v['host_element']}→{v['dopant_element']}{pct_s}")

    # ── ENCUT from all elements (base + dopants) ──────────────────────────────
    all_elements = list(dict.fromkeys(base_elements + dopant_elements))
    encut = 0.0
    if all_elements:
        print(f"\n  Computing ENCUT from all elements: {', '.join(all_elements)}")
        encut = _encut_from_elements(all_elements)
        if encut:
            print(f"  {_green('ENCUT')} = 1.3 × max(ENMAX) = {_bold(f'{encut:.1f} eV')}  "
                  f"{_dim('(written to project.yaml as encut:)')}")
        else:
            print(f"  {_yellow('ENCUT could not be determined — add encut: manually')}")

    return variants, encut


def _ask_mechanics(is_crystalline: bool) -> dict:
    """Optionally collect mechanics inputs for crystalline stress/continuum models."""
    if not is_crystalline:
        return {}
    print()
    if not _yes("Add mechanical properties for continuum/stress models?", default=True):
        return {}
    _hdr("Mechanical Properties")
    return {
        "E_GPa":    _float("Young's modulus (GPa)", 30.0),
        "nu":       _float("Poisson's ratio",       0.22),
        "Omega_A3": _float("Partial molar volume of mobile ion (Å³/ion)", 20.0),
        "rho_gcm3": _float("Density (g/cm³)",       2.0),
    }


def _solid_workload_estimate(doc: dict) -> dict[str, int]:
    """Return a conservative pre-submission job/ionic-step estimate for a crystal project."""
    sim = doc.get("simulation", {})
    variants = doc.get("crystal_doping_variants", [])
    n_variants = max(1, len(variants))
    n_temps = len(sim.get("aimd_temps", []))
    dataset_boxes = 16 if doc.get("stages", {}).get("aimd", False) else 0
    aimd_jobs = n_variants * (n_temps + dataset_boxes)
    aimd_steps = int(sim.get("aimd_steps", 0))
    dataset_steps = int(sim.get("aimd_dataset_steps", aimd_steps))
    ionic_steps = n_variants * (n_temps * aimd_steps + dataset_boxes * dataset_steps)
    dft_cfg = doc.get("stages", {}).get("dft", {})
    dft_jobs_per_variant = sum(bool(v) for v in dft_cfg.values()) if isinstance(dft_cfg, dict) else int(bool(dft_cfg))
    doped_variants = sum(bool(v.get("host_element")) for v in variants)
    aimd_relax_jobs = doped_variants if doc.get("category") == "inorganic_sse" else 0
    neb_jobs = n_variants if doc.get("stages", {}).get("neb", False) else 0
    return {
        "variants": n_variants,
        "aimd_jobs": aimd_jobs,
        "dft_jobs": n_variants * dft_jobs_per_variant + aimd_relax_jobs,
        "neb_jobs": neb_jobs,
        "minimum_slurm_jobs": aimd_jobs + n_variants * dft_jobs_per_variant + aimd_relax_jobs + neb_jobs,
        "vasp_ionic_steps": ionic_steps,
    }


def _ask_manuscript(full_name: str) -> dict:
    """Optionally prompt for manuscript title and author list; returns a metadata dict."""
    print()
    if not _yes("Add manuscript metadata (title / authors)?", default=True):
        return {}
    return {
        "manuscript_title": _input("Title", f"{full_name} Computational Study"),
        "authors":          _input("Authors", "Selva Chandrasekaran Selvaraj et al."),
    }


# ── Ingredient-first box composition ─────────────────────────────────────────

BOX_CONTENTS = [
    ("crystal",      "Crystal / inorganic material  — SSE, electrode, coating (CIF/POSCAR/MP)"),
    ("polymer",      "Polymer / monomer chain        — PEO, PVDF, PTFEP, PVDF-HFP"),
    ("solvent",      "Organic solvent(s)             — DME, EC, DMC, DOL, FEC, TEGDME..."),
    ("salt",         "Ionic salt                     — LiFSI, LiPF6, LiTFSI, NaFSI..."),
    ("electrode",    "Electrode material             — Li metal, Na, graphite, NMC622, LFP"),
    ("nanoparticle", "Nanoparticle                   — carved from bulk crystal"),
    ("additive",     "Additive / co-solvent          — VC, PS, DTD, FEC (trace amount)"),
    ("vacuum",       "Vacuum gap                     — surface slab / open interface"),
]


def _ask_box_contents() -> list[str]:
    """Step 1 of wizard: what ingredients go in the box?  Returns list of component keys."""
    _hdr("What's in the simulation box?")
    print()
    print(_dim("  Tell us all the ingredients — we'll configure the workflow from that."))
    print()
    print(_bold("  Box contents  ") + _dim("(space-separated numbers, e.g.  3 4  or  2 3 4):"))
    for i, (k, d) in enumerate(BOX_CONTENTS, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")
    print()
    print(_dim("  Examples:"))
    print(_dim("    3 4         → liquid electrolyte  (solvent + salt)"))
    print(_dim("    2 3 4       → gel electrolyte     (polymer + solvent + salt)"))
    print(_dim("    2 4         → solid polymer electrolyte  (polymer + salt)"))
    print(_dim("    1 5         → SSE | electrode interface"))
    print(_dim("    1           → bulk crystal  (SSE, electrode, or coating)"))
    print(_dim("    1 6         → nanoparticle on substrate"))
    while True:
        try:
            raw = input("  Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(BOX_CONTENTS):
                k = BOX_CONTENTS[int(tok) - 1][0]
                if k not in picks:
                    picks.append(k)
        if picks:
            print(f"  → {_green(', '.join(picks))}")
            return picks
        print(f"  {_yellow('Select at least one component')}")


def _auto_system_type(box: list[str]) -> str:
    """Infer system_type from list of box component keys."""
    has_crystal = "crystal"      in box
    has_polymer = "polymer"      in box
    has_solvent = ("solvent" in box or "additive" in box)
    has_salt    = "salt"         in box
    has_elec    = "electrode"    in box
    has_np      = "nanoparticle" in box
    has_vac     = "vacuum"       in box

    if has_np and has_crystal:              return "np_substrate"
    if has_np:                              return "nanoparticle"
    if has_crystal and has_elec:            return "sse_electrode"
    if has_crystal and has_solvent:         return "sse_liquid"
    if has_crystal and has_vac:             return "surface"
    if has_polymer and has_solvent:         return "gel"
    if has_polymer:                         return "polymer"
    if has_solvent and has_elec:            return "electrode_liquid"
    if has_solvent:                         return "liquid_electrolyte"
    if has_crystal:                         return "bulk_sse"
    return "bulk_sse"


def _ask_monomers_multi(structs: list[dict]) -> list[str]:
    """Multi-select monomer types. Accepts space-separated numbers OR known names."""
    monomer_opts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for s in structs:
        nm = s["file"].replace(".vasp", "").replace(".cif", "")
        if nm not in seen:
            monomer_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms  [detected]"))
            seen.add(nm)
    for k, d in MONOMERS:
        if k not in seen and k != "custom":
            monomer_opts.append((k, d))
            seen.add(k)
    monomer_opts.append(("custom", "Custom monomer — type names or SMILES (space-separated)"))
    known_keys = {k.lower(): k for k, _ in monomer_opts}

    print()
    print(_bold("  Monomer type  ") + _dim("(space-separated numbers or names, e.g.  2 3  or  PVDF-HFP PTFEP):"))
    for i, (k, d) in enumerate(monomer_opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")
    while True:
        try:
            raw = input("  Select [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(monomer_opts):
                k = monomer_opts[int(tok) - 1][0]
                if k == "custom":
                    pass  # handled below
                elif k not in picks:
                    picks.append(k)
            elif tok.lower() in known_keys:
                k = known_keys[tok.lower()]
                if k not in picks and k != "custom":
                    picks.append(k)
            else:
                # treat unknown token as literal custom name
                if tok not in picks:
                    picks.append(tok)
        # if user typed "5" (custom) with no names, ask interactively
        if "custom" in [monomer_opts[int(t)-1][0] for t in raw.split()
                        if t.isdigit() and 1 <= int(t) <= len(monomer_opts)]:
            cust_raw = _input("  Custom monomer name(s) — space-separated")
            for nm in cust_raw.split():
                if nm and nm not in picks:
                    picks.append(nm)
        if picks:
            print(f"  → {_green(', '.join(picks))}")
            return picks
        print(f"  {_yellow('Select at least one monomer')}")


def _ask_salts_multi(detected_salt: str = "") -> list[str]:
    """Multi-select salts (space-separated numbers, e.g. 1 3)."""
    opts = list(SALTS)
    if detected_salt and detected_salt not in [k for k, _ in opts]:
        opts.insert(0, (detected_salt, "(detected from structure file)"))
    print()
    print(_bold("  Salt type  ") + _dim("(space-separated numbers, e.g.  1  or  1 3):"))
    for i, (k, d) in enumerate(opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")
    try:
        raw = input("  Select [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    picks: list[str] = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(opts):
            k = opts[int(tok) - 1][0]
            if k not in picks and k != "none":
                picks.append(k)
    result = picks or ["LiFSI"]
    print(f"  → {_green(', '.join(result))}")
    return result


# ── Category percentage wizard ─────────────────────────────────────────────

def _pick_n(label: str, default: int = 1) -> int:
    """Ask how many species (1, 2, or 3)."""
    print(f"  {label}")
    print(f"  {_cyan('1')}) mono  {_cyan('2')}) di  {_cyan('3')}) tri")
    while True:
        try:
            raw = input(f"  Choice [{default}]: ").strip() or str(default)
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in ("1", "2", "3"):
            return int(raw)
        print(f"  {_yellow('Enter 1, 2, or 3')}")


# _ask_category_percentages defined later (new multi-select version)


def _ask_solvents_in_category(n: int, category_pct: float) -> list[dict]:
    """Ask n solvents with vol% split within category_pct."""
    sv_opts = [(k, d) for k, d in SOLVENTS if k != "custom"]
    sv_opts.append(("custom", "Other — type the name"))
    solvents: list[dict] = []
    remaining = category_pct
    for i in range(n):
        _hdr(f"Solvent {i+1} of {n}")
        for j, (k, d) in enumerate(sv_opts, 1):
            print(f"  {_cyan(str(j))}) {_bold(k):<16}  {d}")
        while True:
            try:
                raw = input("  Select [3]: ").strip() or "3"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw.isdigit() and 1 <= int(raw) <= len(sv_opts):
                k = sv_opts[int(raw)-1][0]
                if k == "custom":
                    k = _input("  Solvent name")
                print(f"  → {_green(k)}")
                break
            print(f"  {_yellow('Enter a number')}")
        if n == 1:
            pct = category_pct
        else:
            default_pct = round(remaining / (n - i), 1)
            pct = _float(f"  {k} vol% (of {category_pct:.0f}% total)", default_pct)
            remaining -= pct
        solvents.append({"name": k, "vol_pct": pct})
    return solvents


def _ask_salts_in_category(n: int, category_pct: float) -> list[dict]:
    """Ask n salts with % split within category_pct."""
    salt_opts = [(k, d) for k, d in SALTS if k != "none"]
    salts: list[dict] = []
    remaining = category_pct
    for i in range(n):
        _hdr(f"Salt {i+1} of {n}")
        for j, (k, d) in enumerate(salt_opts, 1):
            print(f"  {_cyan(str(j))}) {_bold(k):<16}  {d}")
        while True:
            try:
                raw = input("  Select [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw.isdigit() and 1 <= int(raw) <= len(salt_opts):
                k = salt_opts[int(raw)-1][0]
                print(f"  → {_green(k)}")
                break
            print(f"  {_yellow('Enter a number')}")
        if n == 1:
            pct = category_pct
        else:
            default_pct = round(remaining / (n - i), 1)
            pct = _float(f"  {k} vol% (of {category_pct:.0f}% total)", default_pct)
            remaining -= pct
        salts.append({"name": k, "vol_pct": pct})
    return salts


def _ask_polymers_in_category(n: int, category_pct: float) -> list[dict]:
    """Ask n homopolymers with chain/monomer details."""
    mono_opts = [(k, d) for k, d in MONOMERS if k not in ("custom",)]
    polymers: list[dict] = []
    pct_each = round(category_pct / n, 1)
    for i in range(n):
        _hdr(f"Polymer {i+1} of {n}")
        for j, (k, d) in enumerate(mono_opts, 1):
            print(f"  {_cyan(str(j))}) {_bold(k):<16}  {d}")
        print(f"  {_cyan(str(len(mono_opts)+1))}) {'custom':<16}  Custom monomer")
        while True:
            try:
                raw = input("  Select [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            idx = int(raw) if raw.isdigit() else 0
            if 1 <= idx <= len(mono_opts):
                monomer = mono_opts[idx-1][0]
                print(f"  → {_green(monomer)}")
                break
            elif idx == len(mono_opts) + 1:
                monomer = _input("  Monomer name")
                break
            print(f"  {_yellow('Enter a number')}")
        n_chains   = _int(f"  {monomer}: number of chains", 40)
        n_monomers = _int(f"  {monomer}: monomers per chain", 50)
        polymers.append({
            "monomer":    monomer,
            "n_chains":   n_chains,
            "n_monomers": n_monomers,
            "vol_pct":    pct_each,
        })
    return polymers


def _ask_copolymers_in_category(n: int, category_pct: float) -> list[dict]:
    """Ask n copolymers with ratio, chain, and monomer details."""
    _COPOLY_PRESETS = {
        "PVDF-HFP":  (["VDF", "HFP"],  [9, 1],   20, 40),
        "PVDF-TrFE": (["VDF", "TrFE"], [75, 25],  20, 40),
    }
    copolymers: list[dict] = []
    pct_each = round(category_pct / n, 1)
    for i in range(n):
        _hdr(f"Copolymer {i+1} of {n}")
        for j, (k, d) in enumerate(COPOLYMERS, 1):
            print(f"  {_cyan(str(j))}) {_bold(k):<16}  {d}")
        while True:
            try:
                raw = input("  Select [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw.isdigit() and 1 <= int(raw) <= len(COPOLYMERS):
                name = COPOLYMERS[int(raw)-1][0]
                break
            print(f"  {_yellow('Enter a number')}")

        preset = _COPOLY_PRESETS.get(name)
        if name == "custom" or preset is None:
            comp1 = _input("  Component 1 monomer name", "VDF")
            comp2 = _input("  Component 2 monomer name", "HFP")
            components = [comp1, comp2]
            def_ratio = [9, 1]; def_monomers = 20; def_chains = 40
        else:
            components, def_ratio, def_monomers, def_chains = preset
            print(f"  Components: {_green(' + '.join(components))}")

        # Ratio
        print(f"  Ratio  {':'.join(components)}  "
              f"{_dim('(space-separated integers, e.g. 9 1)')}")
        while True:
            try:
                r_raw = input(f"  Ratio [{' '.join(str(r) for r in def_ratio)}]: "
                              ).strip() or " ".join(str(r) for r in def_ratio)
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            parts = r_raw.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                ratio = [int(p) for p in parts]
                break
            print(f"  {_yellow('Enter two integers e.g. 9 1')}")

        total_ratio = sum(ratio)
        n_monomers_total = _int(f"  Total monomers per chain", def_monomers)
        n_chains         = _int(f"  Number of chains", def_chains)
        # Compute per-component monomers
        comp_counts = [round(n_monomers_total * r / total_ratio) for r in ratio]
        minor_frac  = ratio[1] / total_ratio
        print(f"  → {comp_counts[0]} {components[0]} + {comp_counts[1]} {components[1]} "
              f"per chain × {n_chains} chains  "
              f"{_dim(f'({n_chains*n_monomers_total} repeat units total)')}")
        print(f"  → {_green(name)}  minor_fraction={minor_frac:.3f}")

        entry_name = name if name != "custom" else f"{components[0]}-{components[1]}"
        copolymers.append({
            "monomer":            entry_name,
            "components":         components,
            "ratio":              ratio,
            "n_chains":           n_chains,
            "n_monomers_per_chain": n_monomers_total,
            "component_counts":   comp_counts,
            "minor_fraction":     minor_frac,
            "vol_pct":            pct_each,
        })
    return copolymers


# _ask_composition_detail defined later (new multi-select + combo version)


def _compute_box_from_molarity(comp: dict, c: float, target_natoms: int) -> dict:
    """Compute box spec from salt molarity (mol/L) + solvent volume ratios."""
    solvents = comp.get("solvents", [])
    salts    = comp.get("salts",    [])
    if not solvents or not salts or c <= 0:
        return {"natoms": target_natoms, "box_A": 20.0, "density_gcm3": 1.0,
                "species": {}, "chains": {}}

    salt_name                    = salts[0]["name"]
    MW_salt, nat_salt, rho_salt  = _MOL_INFO.get(salt_name, _MOL_INFO_DEFAULT)

    raw      = [(s["name"], float(s.get("ratio", s.get("vol_pct", 1.0)))) for s in solvents]
    mole_w   = [(nm, r / _MOL_INFO.get(nm, _MOL_INFO_DEFAULT)[0]) for nm, r in raw]
    total_mw = sum(w for _, w in mole_w) or 1.0
    solv_mf  = [(nm, w / total_mw) for nm, w in mole_w]

    MW_solv_avg  = sum(f * _MOL_INFO.get(nm, _MOL_INFO_DEFAULT)[0] for nm, f in solv_mf)
    rho_solv_avg = sum(f * _MOL_INFO.get(nm, _MOL_INFO_DEFAULT)[2] for nm, f in solv_mf)
    nat_solv_avg = sum(f * _MOL_INFO.get(nm, _MOL_INFO_DEFAULT)[1] for nm, f in solv_mf)

    denom  = rho_solv_avg * 1000.0 + c * (MW_solv_avg - MW_salt)
    x_salt = max(0.01, min(0.95, c * MW_solv_avg / max(denom, 1e-9)))

    avg_atoms = x_salt * nat_salt + (1.0 - x_salt) * nat_solv_avg
    avg_MW    = x_salt * MW_salt  + (1.0 - x_salt) * MW_solv_avg
    avg_rho   = x_salt * rho_salt + (1.0 - x_salt) * rho_solv_avg

    N_mol_total = target_natoms / max(1.0, avg_atoms)
    V_box_A3    = N_mol_total * avg_MW / (avg_rho * 6.022e23 * 1e-24)
    box_A       = V_box_A3 ** (1.0 / 3.0)

    N_salt = max(1, round(x_salt * N_mol_total))
    N_solv = max(1, round((1.0 - x_salt) * N_mol_total))

    species: dict[str, int] = {salt_name: N_salt}
    for nm, f in solv_mf:
        species[nm] = max(1, round(f * N_solv))

    natoms_actual = sum(_MOL_INFO.get(nm, _MOL_INFO_DEFAULT)[1] * n
                        for nm, n in species.items())
    return {
        "natoms":       natoms_actual,
        "box_A":        box_A,
        "species":      species,
        "chains":       {},
        "density_gcm3": avg_rho,
    }


def _compute_mlmd_box(comp: dict, cat_pcts: dict[str, float]) -> dict:
    """
    Compute MLMD box from polymer chain specs + vol%.
    Polymer n_chains/n_monomers fixes the total polymer volume,
    which (via vol%) fixes V_box. Solvents/salts fill the rest.
    """
    import math

    poly_entries = comp.get("polymers", []) + [
        {**cp, "n_monomers": cp.get("n_monomers_per_chain", 20)}
        for cp in comp.get("copolymers", [])
    ]

    # Individual polymer volume percentages are authoritative.  User-entered
    # chain counts set the approximate total polymer size only; redistribute
    # those chains between polymer species so a request such as 20 vol% PEO +
    # 5 vol% PVDF-HFP does not accidentally become a 1:2 polymer blend.
    if len(poly_entries) > 1:
        requested = [max(0.0, float(p.get("vol_pct", 0.0))) for p in poly_entries]
        if sum(requested) > 0:
            original_volume = 0.0
            for p in poly_entries:
                mw, _, rho = _MOL_INFO.get(p["monomer"], _MOL_INFO_DEFAULT)
                original_volume += (p["n_chains"] * p["n_monomers"] * mw / rho)
            resolved: list[dict] = []
            for p, fraction in zip(poly_entries, requested):
                mw, _, rho = _MOL_INFO.get(p["monomer"], _MOL_INFO_DEFAULT)
                target_volume = original_volume * fraction / sum(requested)
                volume_per_chain = p["n_monomers"] * mw / rho
                resolved.append({
                    **p,
                    "n_chains": max(1, round(target_volume / max(volume_per_chain, 1e-12))),
                })
            poly_entries = resolved

    # Total polymer atoms and their vol fraction
    f_poly = (cat_pcts.get("polymer", 0) + cat_pcts.get("copolymer", 0)) / 100.0

    if poly_entries and f_poly > 0:
        # Compute total polymer atom volume from chain specs
        total_poly_vol_A3 = 0.0
        total_poly_atoms  = 0
        for p in poly_entries:
            mw, nat, rho = _MOL_INFO.get(p["monomer"], _MOL_INFO_DEFAULT)
            n_units = p["n_chains"] * p["n_monomers"]
            total_poly_atoms += n_units * nat
            vol_per_unit = mw / rho / 6.022e23 * 1e24  # Å³ per monomer unit
            total_poly_vol_A3 += n_units * vol_per_unit
        V_box = total_poly_vol_A3 / f_poly
    else:
        # No polymers: use molarity path when specified
        c = float(comp.get("salt_molarity", 0.0))
        if c > 0 and comp.get("solvents") and comp.get("salts"):
            return _compute_box_from_molarity(comp, c, 15_000)

        # vol_pct path — estimate from solvents+salts targeting 15000 atoms
        target_non_poly = 15_000
        f_liq = (cat_pcts.get("solvent", 0) + cat_pcts.get("salt", 0)) / 100.0
        if f_liq <= 0:
            f_liq = 1.0
        atoms_per_A3_liq = 0.0
        for sv in comp.get("solvents", []):
            f = sv["vol_pct"] / 100.0
            mw, nat, rho = _MOL_INFO.get(sv["name"], _MOL_INFO_DEFAULT)
            atoms_per_A3_liq += f * rho * 6.022e23 * nat / (mw * 1e24)
        for sl in comp.get("salts", []):
            f = sl["vol_pct"] / 100.0
            mw, nat, rho = _MOL_INFO.get(sl["name"], _MOL_INFO_DEFAULT)
            atoms_per_A3_liq += f * rho * 6.022e23 * nat / (mw * 1e24)
        if atoms_per_A3_liq <= 0:
            atoms_per_A3_liq = 0.05
        V_box = target_non_poly / atoms_per_A3_liq
        total_poly_atoms = 0

    box_A = V_box ** (1.0/3.0)

    # Count solvents and salts
    species: dict[str, int] = {}
    natoms_actual = total_poly_atoms

    c = float(comp.get("salt_molarity", 0.0))
    if c > 0:
        # Molarity mode: vol_pct is absent from solvents/salts; derive counts
        # from concentration and the liquid fraction of V_box.
        V_liq_A3 = V_box * (1.0 - f_poly)          # Å³ liquid phase
        V_liq_L  = V_liq_A3 * 1e-27                # Å³ → litres

        # Salt: split equally among salt species if multiple
        salt_entries = comp.get("salts", [])
        n_salt_types = max(1, len(salt_entries))
        for sl in salt_entries:
            mw_s, nat_s, _ = _MOL_INFO.get(sl["name"], _MOL_INFO_DEFAULT)
            N_sl = max(1, round(c * V_liq_L * 6.022e23 / n_salt_types))
            species[sl["name"]] = N_sl
            natoms_actual += N_sl * nat_s

        # Solvent: fill the volume left after salt, split by ratio
        salt_vol_A3 = sum(
            species[sl["name"]] * _MOL_INFO.get(sl["name"], _MOL_INFO_DEFAULT)[0]
            / (_MOL_INFO.get(sl["name"], _MOL_INFO_DEFAULT)[2] * 6.022e23 * 1e-24)
            for sl in salt_entries
        )
        V_solv_A3 = max(1.0, V_liq_A3 - salt_vol_A3)
        sv_entries = comp.get("solvents", [])
        total_ratio = sum(sv.get("ratio", 1.0) for sv in sv_entries) or 1.0
        for sv in sv_entries:
            mw_v, nat_v, rho_v = _MOL_INFO.get(sv["name"], _MOL_INFO_DEFAULT)
            frac = sv.get("ratio", 1.0) / total_ratio
            vol_per_mol = mw_v / rho_v / 6.022e23 * 1e24   # Å³/molecule
            N_sv = max(1, round(frac * V_solv_A3 / vol_per_mol))
            species[sv["name"]] = N_sv
            natoms_actual += N_sv * nat_v
    else:
        # vol_pct mode
        for sv in comp.get("solvents", []):
            f = sv["vol_pct"] / 100.0
            mw, nat, rho = _MOL_INFO.get(sv["name"], _MOL_INFO_DEFAULT)
            n = max(1, round(f * V_box * rho / (mw / 6.022e23 * 1e24)))
            species[sv["name"]] = n
            natoms_actual += n * nat
        for sl in comp.get("salts", []):
            f = sl["vol_pct"] / 100.0
            mw, nat, rho = _MOL_INFO.get(sl["name"], _MOL_INFO_DEFAULT)
            n = max(1, round(f * V_box * rho / (mw / 6.022e23 * 1e24)))
            species[sl["name"]] = n
            natoms_actual += n * nat

    chains: dict[str, int] = {}
    for p in poly_entries:
        species[p["monomer"]] = p["n_chains"] * p["n_monomers"]
        chains[p["monomer"]]  = p["n_chains"]

    # Overall density
    avg_rho = sum(
        _MOL_INFO.get(k, _MOL_INFO_DEFAULT)[2] * v
        for k, v in species.items()
    ) / max(1, sum(species.values()))

    return {
        "natoms":       natoms_actual,
        "box_A":        box_A,
        "species":      species,
        "chains":       chains,
        "density_gcm3": avg_rho,
    }


def _scale_box(mlmd_spec: dict, target_natoms: int,
               comp: dict, is_aimd: bool = False) -> dict:
    """Scale MLMD box to a different atom count."""
    scale = target_natoms / max(1, mlmd_spec["natoms"])
    species: dict[str, int] = {}
    chains:  dict[str, int] = {}

    poly_names = {p["monomer"] for p in comp.get("polymers", [])} | \
                 {cp["monomer"] for cp in comp.get("copolymers", [])}

    poly_chain_map = {p["monomer"]: p["n_monomers"]
                      for p in comp.get("polymers", [])}
    poly_chain_map.update({cp["monomer"]: cp.get("n_monomers_per_chain", 20)
                            for cp in comp.get("copolymers", [])})

    for name, count in mlmd_spec["species"].items():
        if name in poly_names:
            n_chains_orig  = mlmd_spec["chains"].get(name, 1)
            n_mono_per_chain = poly_chain_map.get(name, 20)
            if is_aimd:
                # AIMD: keep 1 chain, cap monomers at 10
                n_chains_new  = 1
                n_mono_new    = min(n_mono_per_chain, 10)
            else:
                n_chains_new  = max(1, round(n_chains_orig * scale))
                n_mono_new    = n_mono_per_chain
            chains[name]  = n_chains_new
            species[name] = n_chains_new * n_mono_new
        else:
            species[name] = max(1, round(count * scale))

    natoms_actual = sum(
        species[k] * _MOL_INFO.get(k, _MOL_INFO_DEFAULT)[1]
        for k in species
    )
    if natoms_actual > 0:
        V_box = mlmd_spec["box_A"]**3 * natoms_actual / max(1, mlmd_spec["natoms"])
        box_A = V_box ** (1.0/3.0)
    else:
        box_A = mlmd_spec["box_A"] * (scale ** (1.0/3.0))

    return {
        "natoms":       natoms_actual,
        "box_A":        box_A,
        "species":      species,
        "chains":       chains,
        "density_gcm3": mlmd_spec["density_gcm3"],
    }


def _aimd_box_auto(comp: dict, cat_pcts: dict, target_atoms: int) -> dict:
    """
    AIMD box: 1 chain per polymer, monomer count auto-scaled to fit within
    target_atoms while preserving chain-length ratios between polymer types.

    total_atoms ∝ factor (V_box ∝ polymer volume ∝ monomer count), so we
    solve analytically and do one rounding-correction pass.
    """
    def _scaled_comp(factor: float) -> dict:
        """Return a copy of comp with polymer chain counts set to 1 and monomer counts scaled by factor."""
        return {
            **comp,
            "polymers": [
                {**p, "n_chains": 1,
                 "n_monomers": max(1, round(p.get("n_monomers", 20) * factor))}
                for p in comp.get("polymers", [])
            ],
            "copolymers": [
                {**cp, "n_chains": 1,
                 "n_monomers_per_chain": max(1, round(cp.get("n_monomers_per_chain", 20) * factor))}
                for cp in comp.get("copolymers", [])
            ],
        }

    # Try original monomer count, 1 chain each
    spec = _compute_mlmd_box(_scaled_comp(1.0), cat_pcts)
    if spec["natoms"] <= target_atoms:
        return spec

    # Atoms scale linearly with factor — solve for target
    factor = target_atoms / max(1, spec["natoms"])
    spec = _compute_mlmd_box(_scaled_comp(factor), cat_pcts)

    # One correction pass to absorb integer-rounding overshoot
    if spec["natoms"] > target_atoms:
        factor *= target_atoms / max(1, spec["natoms"])
        spec = _compute_mlmd_box(_scaled_comp(factor), cat_pcts)

    return spec


def _ask_composition_tiers(comp: dict, cat_pcts: dict[str, float]) -> dict:
    """
    Compute AIMD/MLMD/CMD boxes from polymer specs + vol%.
    Show preview table. Let user select tiers and adjust targets.
    Returns full result dict for sim block.
    """
    _hdr("Simulation Tiers — Box Preview")
    print()
    print(_dim("  Chain ratios from your polymer specs are preserved;"))
    print(_dim("  chain counts are scaled to hit each tier's atom target."))
    print()

    # Reference: MLMD from polymer chain specs (chain ratios preserved)
    chain_ref_spec = _compute_mlmd_box(comp, cat_pcts)

    # Let user set all three tier targets; scale chain_ref_spec to each
    _ld1 = _lane_defaults()
    print(_bold("  Choose target atom counts for each simulation tier:"))
    print(_dim("  DFT    : ≤ 300 atoms  (auto-filled by orchestrator — no input needed)"))
    print(_dim("  MLMD   : 5k–10k   atoms  (LAMMPS + DeepMD, medium box)"))
    print(_dim("  CMD    : 40k–50k  atoms  (LAMMPS + classical FF, large box)"))
    print()
    mlmd_target = _ld1["mlmd_atoms"]
    cmd_target  = _ld1["cmd_atoms"]

    aimd_spec = _dft_preview(comp, dft_max=_ld1["aimd_atoms"], target=250)
    mlmd_spec = _scale_box(chain_ref_spec, mlmd_target, comp, is_aimd=False)
    cmd_spec  = _scale_box(chain_ref_spec, cmd_target,  comp, is_aimd=False)

    tier_specs = {"AIMD": aimd_spec, "MLMD": mlmd_spec, "CMD": cmd_spec}

    # Preview table
    all_species = sorted({sp for s in tier_specs.values() for sp in s["species"]})
    all_chains  = sorted({ch for s in tier_specs.values() for ch in s["chains"]})
    col_w = max(8, *(len(sp) for sp in all_species + all_chains), 0) + 2

    print()
    print(_bold("  ── Computed box composition ──────────────────────────────────"))
    header = f"  {'Tier':<6}  {'Atoms':>7}  {'Box(Å)':>7}  {'ρ(g/cm³)':>9}  "
    for sp in all_species:
        header += f"{sp:>{col_w}}"
    for ch in all_chains:
        header += f"{'chains:'+ch:>{col_w+7}}"
    print(_bold(header))
    print("  " + "─" * (len(header) - 2))
    for tier, spec in tier_specs.items():
        row = f"  {_cyan(tier):<6}  {spec['natoms']:>7,}  {spec['box_A']:>7.1f}  "
        row += f"{spec['density_gcm3']:>9.3f}  "
        for sp in all_species:
            row += f"{spec['species'].get(sp, 0):>{col_w}}"
        for ch in all_chains:
            row += f"{spec['chains'].get(ch, 0):>{col_w+7}}"
        print(row)
    print()

    # Select tiers
    print(_bold("  Select tiers to build  ") +
          _dim("(space-separated: 1=AIMD  2=MLMD  3=CMD  4=all, default=1 2 3):"))
    print(f"  {_cyan('1')}) AIMD  — {aimd_spec['natoms']:,} atoms, {aimd_spec['box_A']:.0f} Å  (DFT data generation)")
    print(f"  {_cyan('2')}) MLMD  — {mlmd_spec['natoms']:,} atoms, {mlmd_spec['box_A']:.0f} Å  (LAMMPS + DeepMD)")
    print(f"  {_cyan('3')}) CMD   — {cmd_spec['natoms']:,} atoms, {cmd_spec['box_A']:.0f} Å  (classical FF)")
    print(f"  {_cyan('4')}) All three")
    while True:
        try:
            raw = input("  Tiers [1 2 3]: ").strip() or "1 2 3"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw == "4":
            selected = ["AIMD", "MLMD", "CMD"]
        else:
            selected = []
            for tok in raw.split():
                if tok == "1" and "AIMD" not in selected: selected.append("AIMD")
                elif tok == "2" and "MLMD" not in selected: selected.append("MLMD")
                elif tok == "3" and "CMD" not in selected: selected.append("CMD")
        if selected:
            print(f"  → {_green(', '.join(selected))}")
            break
        print(f"  {_yellow('Select at least one tier')}")

    # Build result
    result: dict = {
        "box_fractions":         {f"solv:{sv['name']}": sv.get("vol_pct", 0)
                                   for sv in comp.get("solvents", [])} |
                                 {f"salt:{sl['name']}": sl.get("vol_pct", 0)
                                   for sl in comp.get("salts", [])} |
                                 {f"poly:{p['monomer']}": p["vol_pct"]
                                   for p in comp.get("polymers", []) + [
                                       {**cp, "monomer": cp["monomer"]}
                                       for cp in comp.get("copolymers", [])
                                   ]},
        **({"salt_molarity": comp["salt_molarity"]} if comp.get("salt_molarity", 0) > 0 else {}),
        "tiers_selected":        selected,
        "tier_aimd":             aimd_spec,
        "tier_mlmd":             mlmd_spec,
        "tier_cmd":              cmd_spec,
        "target_density_gcm3":   mlmd_spec["density_gcm3"],
        "molecule_counts_aimd":  aimd_spec["species"],
        "molecule_counts_mlmd":  mlmd_spec["species"],
        "molecule_counts_cmd":   cmd_spec["species"],
        "chain_counts_aimd":     aimd_spec["chains"],
        "chain_counts_mlmd":     mlmd_spec["chains"],
        "chain_counts_cmd":      cmd_spec["chains"],
        "classical_md":          "CMD" in selected,
        # composition lists for YAML
        "solvents":              [{"name": sv["name"], "ratio": 1}
                                   for sv in comp.get("solvents", [])],
        "salt":                  comp.get("salts", [{}])[0].get("name", "LiFSI")
                                  if comp.get("salts") else "",
    }
    return result


def _ask_temperatures_hierarchy(selected_tiers: list[str]) -> dict:
    """Return temperature/step defaults silently (no prompts)."""
    _ld = _lane_defaults()
    result: dict = {}
    default_aimd = [300, 400, 500]
    default_md   = _ld.get("nvt_temperatures", [300, 320, 340, 360, 380, 400, 500, 600])

    if "AIMD" in selected_tiers:
        result["aimd_temps"] = default_aimd
        result["aimd_steps"] = _ld.get("aimd_steps", 3_000)

    if "MLMD" in selected_tiers:
        result["mlmd_temps"]     = default_md
        result["mlmd_npt_steps"] = _ld.get("npt_steps", 1_000)
        result["mlmd_nvt_steps"] = _ld.get("mlmd_steps", 1_000_000)
        result["mlmd_steps"]     = result["mlmd_nvt_steps"]

    if "CMD" in selected_tiers:
        result["cmd_temps"]      = default_md
        result["forcefield"]     = "OPLS-AA"
        result["cmd_equil_steps"] = _ld.get("npt_steps", 1_000)
        result["cmd_prod_steps"]  = _ld.get("nvt_steps", 2_000_000)

    return result


def _ask_composition_variants(comp: dict, cat_pcts: dict[str, float],
                               tier_specs: dict) -> list[dict]:
    """
    Ask whether to define additional composition variants for MLMD/CMD sweep.
    Returns list of variant dicts; each has cat_pcts + comp + tier_specs.
    """
    _hdr("Combinatorial Compositions")
    print()
    print(_dim("  The base composition will be used for AIMD, MLMD, and CMD."))
    print(_dim("  You can define additional composition variants to sweep for MLMD/CMD."))
    print(_dim("  All variants share the same simulation box structure (POSCAR template)."))
    print()
    if not _yes("Define additional composition variants for MLMD/CMD sweep?", default=False):
        return []

    variants: list[dict] = [{"label": "base", "cat_pcts": cat_pcts, "comp": comp}]
    while True:
        print()
        _hdr(f"Composition Variant {len(variants)}")
        v_sel_cats = _ask_box_categories()
        v_cat_pcts = _ask_category_percentages(v_sel_cats)
        v_comp     = _ask_composition_detail(v_cat_pcts, [], v_sel_cats)
        v_spec     = _compute_mlmd_box(v_comp, v_cat_pcts)
        label_parts = []
        for sl in v_comp.get("salts", []):
            label_parts.append(sl["name"])
        for sv in v_comp.get("solvents", []):
            label_parts.append(sv["name"])
        label = "_".join(label_parts) or f"variant_{len(variants)}"
        print(f"\n  Variant: {_green(label)}  →  "
              f"{v_spec['natoms']:,} atoms, {v_spec['box_A']:.0f} Å")
        variants.append({"label": label, "cat_pcts": v_cat_pcts, "comp": v_comp})
        if not _yes("Add another variant?", default=False):
            break

    print(f"\n  {_green(str(len(variants)))} composition(s) selected  "
          f"(1 base + {len(variants)-1} variant(s))")
    return variants[1:]  # return only the extra variants; base is already in comp


# ── Combinatorial design helpers ─────────────────────────────────────────────

def _ask_study_design() -> str:
    """Prompt the user to choose between a single-system and a combinatorial study mode."""
    _hdr("Study Design")
    print()
    print(_dim("  Single   — one solvent/salt system, one workflow"))
    print(_dim("  Combinatorial — multiple systems run in parallel, results compared"))
    print()
    return _pick("Study mode", [
        ("single",        "Single system  — standard workflow"),
        ("combinatorial", "Combinatorial  — compare multiple solvent/salt systems"),
    ], default=1)


def _build_component_pool(structs: list[dict]) -> tuple[list, list]:
    """Multi-select solvents + salts from detected structure files or built-in list."""
    _hdr("Component Pool  (all ingredients for this study)")

    # ── Solvents ─────────────────────────────────────────────────────────────
    sv_opts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for s in structs:
        if s["role"] == "solvent":
            nm = s["file"].replace(".vasp", "").replace(".cif", "")
            if nm not in seen:
                sv_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms   [detected]"))
                seen.add(nm)
    for k, d in SOLVENTS:
        if k not in seen and k != "custom":
            sv_opts.append((k, d))
            seen.add(k)
    sv_opts.append(("custom", "Other — type the name"))

    print()
    print(_bold("  Solvents  ") + _dim("(space-separated numbers, e.g.  1 2  or  3 4):"))
    for i, (k, d) in enumerate(sv_opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")

    sv_pool: list[str] = []
    while True:
        try:
            raw = input("  Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not raw:
            print(f"  {_yellow('Select at least one solvent')}")
            continue
        picks: list[str] = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(sv_opts):
                k = sv_opts[int(tok) - 1][0]
                if k == "custom":
                    cust = _input("  Solvent name")
                    if cust and cust not in picks:
                        picks.append(cust)
                elif k not in picks:
                    picks.append(k)
        if picks:
            sv_pool = picks
            print(f"  → {_green(', '.join(sv_pool))}")
            break
        print(f"  {_yellow('Select at least one solvent')}")

    # ── Salts ─────────────────────────────────────────────────────────────────
    salt_opts: list[tuple[str, str]] = []
    seen2: set[str] = set()
    for s in structs:
        if s["role"] == "salt":
            nm = s["file"].replace(".vasp", "").replace(".cif", "")
            if nm not in seen2:
                salt_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms   [detected]"))
                seen2.add(nm)
    for k, d in SALTS:
        if k not in seen2:
            salt_opts.append((k, d))

    print()
    print(_bold("  Salts  ") + _dim("(space-separated numbers, Enter for none):"))
    for i, (k, d) in enumerate(salt_opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")

    try:
        raw = input("  Select [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)

    salt_pool: list[str] = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(salt_opts):
            k = salt_opts[int(tok) - 1][0]
            if k not in salt_pool and k != "none":
                salt_pool.append(k)
    print(f"  → {_green(', '.join(salt_pool) if salt_pool else 'none (pure solvent)')}")

    return sv_pool, salt_pool


def _ask_combinations_matrix(sv_pool: list, salt_pool: list) -> list:
    """Build and confirm the list of electrolyte combinations to compare."""
    _hdr("Combinations to Compare")

    # Auto-generate: every single-solvent × every salt
    combos: list = []
    for sv in sv_pool:
        for salt in salt_pool:
            combos.append({
                "name":     f"{sv}_{salt}",
                "label":    f"{sv} + {salt}",
                "solvents": [{"name": sv, "ratio": 1}],
                "salt":     salt,
                "enabled":  True,
            })

    print()
    print(_bold("  Auto-suggested combinations (single-solvent × each salt):"))
    for i, c in enumerate(combos, 1):
        print(f"  {_green('✓')} {_cyan(str(i))}) {c['label']}")

    # Binary solvent mixtures with preset ratios (multi-select → one combo per ratio)
    if len(sv_pool) >= 2:
        print()
        if _yes("Add binary solvent mixtures?", default=True):
            for i in range(len(sv_pool)):
                for j in range(i + 1, len(sv_pool)):
                    sv1, sv2 = sv_pool[i], sv_pool[j]
                    print()
                    print(_bold(f"  Binary mixture: {sv1} + {sv2}"))
                    print(_dim("  Multiple selections generate separate combinations:"))
                    for k, (r1, r2) in enumerate(_BINARY_RATIO_PRESETS, 1):
                        tag = (f"({sv1}-rich)" if r1 > r2
                               else "(equal)" if r1 == r2
                               else f"({sv2}-rich)")
                        print(f"  {_cyan(str(k))}) {r1}:{r2}  {tag}")
                    n_pre = len(_BINARY_RATIO_PRESETS)
                    print(f"  {_cyan(str(n_pre + 1))}) User-defined ratio")
                    try:
                        raw = input("  Select [1 2 3]: ").strip() or "1 2 3"
                    except (EOFError, KeyboardInterrupt):
                        print(); sys.exit(0)
                    ratio_list: list[tuple[int, int]] = []
                    for tok in raw.split():
                        if tok.isdigit():
                            idx = int(tok)
                            if 1 <= idx <= n_pre:
                                ratio_list.append(_BINARY_RATIO_PRESETS[idx - 1])
                            elif idx == n_pre + 1:
                                try:
                                    r_raw = input(
                                        f"  Custom ratio {sv1}:{sv2} (e.g. 3 1): "
                                    ).strip()
                                    parts = r_raw.split()
                                    r1c = int(parts[0])
                                    r2c = int(parts[1]) if len(parts) > 1 else 1
                                    ratio_list.append((r1c, r2c))
                                except (ValueError, IndexError):
                                    pass
                    if not ratio_list:
                        ratio_list = [(1, 1)]
                    for r1, r2 in ratio_list:
                        for salt in (salt_pool if salt_pool else ["none"]):
                            cname = f"{sv1}_{sv2}_{r1}-{r2}_{salt}"
                            combos.append({
                                "name":     cname,
                                "label":    f"{sv1}:{sv2} {r1}:{r2} + {salt}",
                                "solvents": [{"name": sv1, "ratio": r1},
                                             {"name": sv2, "ratio": r2}],
                                "salt":     salt,
                                "enabled":  True,
                            })
                            print(f"  {_green('+')} Added: {cname}")

    # Ternary mixtures with preset ratios
    if len(sv_pool) >= 3:
        print()
        if _yes("Add ternary mixtures?", default=False):
            sv1, sv2, sv3 = sv_pool[0], sv_pool[1], sv_pool[2]
            print(_bold(f"  Ternary: {sv1} + {sv2} + {sv3}"))
            print(_dim("  Multiple selections generate separate combinations:"))
            for k, (r1, r2, r3) in enumerate(_TERNARY_RATIO_PRESETS, 1):
                print(f"  {_cyan(str(k))}) {r1}:{r2}:{r3}")
            n_pre3 = len(_TERNARY_RATIO_PRESETS)
            print(f"  {_cyan(str(n_pre3 + 1))}) User-defined ratio")
            try:
                raw = input("  Select [1 2 3 4]: ").strip() or "1 2 3 4"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            ratio_list3: list[tuple[int, int, int]] = []
            for tok in raw.split():
                if tok.isdigit():
                    idx = int(tok)
                    if 1 <= idx <= n_pre3:
                        ratio_list3.append(_TERNARY_RATIO_PRESETS[idx - 1])
                    elif idx == n_pre3 + 1:
                        try:
                            r_raw = input(
                                f"  Custom {sv1}:{sv2}:{sv3} (e.g. 2 1 1): "
                            ).strip()
                            parts = r_raw.split()
                            r1c = int(parts[0]); r2c = int(parts[1])
                            r3c = int(parts[2]) if len(parts) > 2 else 1
                            ratio_list3.append((r1c, r2c, r3c))
                        except (ValueError, IndexError):
                            pass
            if not ratio_list3:
                ratio_list3 = [(1, 1, 1)]
            for r1, r2, r3 in ratio_list3:
                for salt in (salt_pool if salt_pool else ["none"]):
                    cname = f"{sv1}_{sv2}_{sv3}_{r1}-{r2}-{r3}_{salt}"
                    combos.append({
                        "name":     cname,
                        "label":    f"{sv1}:{sv2}:{sv3} {r1}:{r2}:{r3} + {salt}",
                        "solvents": [{"name": sv1, "ratio": r1},
                                     {"name": sv2, "ratio": r2},
                                     {"name": sv3, "ratio": r3}],
                        "salt":     salt,
                        "enabled":  True,
                    })

    # Show full list and let user toggle off unwanted
    print()
    print(_bold(f"  All {len(combos)} combinations:"))
    for i, c in enumerate(combos, 1):
        mark = _green("✓") if c["enabled"] else _dim("✗")
        print(f"  {mark} {_cyan(str(i))}) {c['label']}")
    print()
    print(_dim("  Enter numbers to disable (space-separated), or Enter to keep all:"))
    try:
        raw = input("  Disable: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(combos):
            combos[int(tok) - 1]["enabled"] = False

    final = [c for c in combos if c["enabled"]]
    for c in final:
        del c["enabled"]

    print()
    print(f"  → {_green(str(len(final)) + ' combinations')} selected:")
    for c in final:
        print(f"     {_dim('•')} {c['label']}")
    return final


def _build_polymer_pool(structs: list[dict]) -> tuple[list, list]:
    """Multi-select monomers + salts for polymer combinatorial study."""
    _hdr("Polymer Component Pool")

    monomer_opts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for s in structs:
        nm = s["file"].replace(".vasp", "").replace(".cif", "")
        if nm not in seen:
            monomer_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms   [detected]"))
            seen.add(nm)
    for k, d in MONOMERS:
        if k not in seen and k != "custom":
            monomer_opts.append((k, d))
            seen.add(k)
    monomer_opts.append(("custom", "Custom monomer — SMILES or name"))

    print()
    print(_bold("  Monomers  ") + _dim("(space-separated numbers):"))
    for i, (k, d) in enumerate(monomer_opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")

    monomer_pool: list[str] = []
    while True:
        try:
            raw = input("  Select [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(monomer_opts):
                k = monomer_opts[int(tok) - 1][0]
                if k == "custom":
                    cust = _input("  Monomer SMILES or abbreviation")
                    if cust and cust not in picks:
                        picks.append(cust)
                elif k not in picks:
                    picks.append(k)
        if picks:
            monomer_pool = picks
            print(f"  → {_green(', '.join(monomer_pool))}")
            break
        print(f"  {_yellow('Select at least one monomer')}")

    salt_opts: list[tuple[str, str]] = []
    seen2: set[str] = set()
    for s in structs:
        if s["role"] == "salt":
            nm = s["file"].replace(".vasp", "").replace(".cif", "")
            if nm not in seen2:
                salt_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms   [detected]"))
                seen2.add(nm)
    for k, d in SALTS:
        if k not in seen2:
            salt_opts.append((k, d))

    print()
    print(_bold("  Salts  ") + _dim("(space-separated numbers, Enter for none):"))
    for i, (k, d) in enumerate(salt_opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(k):<16}  {d}")
    try:
        raw = input("  Select [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    poly_salt_pool: list[str] = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(salt_opts):
            k = salt_opts[int(tok) - 1][0]
            if k not in poly_salt_pool and k != "none":
                poly_salt_pool.append(k)
    print(f"  → {_green(', '.join(poly_salt_pool) if poly_salt_pool else 'none (neat polymer)')}")

    return monomer_pool, poly_salt_pool


def _ask_polymer_combinations_matrix(monomer_pool: list, salt_pool: list) -> list:
    """Polymer combinatorial: monomer × salt × EO:Li ratio."""
    _hdr("Polymer Combinations")
    print()
    print(_bold("  EO:Li ratios  ") + _dim("(EO repeat units per Li⁺ ion):"))
    for i, r in enumerate(_EO_LI_PRESETS, 1):
        print(f"  {_cyan(str(i))}) {r}:1")
    n_pre = len(_EO_LI_PRESETS)
    print(f"  {_cyan(str(n_pre + 1))}) User-defined")
    try:
        raw = input("  Select [1 2 3]: ").strip() or "1 2 3"
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    ratio_list: list[int] = []
    for tok in raw.split():
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= n_pre:
                ratio_list.append(_EO_LI_PRESETS[idx - 1])
            elif idx == n_pre + 1:
                ratio_list.append(_int("  EO:Li ratio (integer)", 12))
    if not ratio_list:
        ratio_list = [8, 12, 16]
    print(f"  → EO:Li = {_green(', '.join(str(r) + ':1' for r in ratio_list))}")

    combos: list = []
    for monomer in monomer_pool:
        for salt in (salt_pool if salt_pool else ["none"]):
            for eo_li in ratio_list:
                salt_tag = salt if salt != "none" else "neat"
                cname = f"{monomer}_{salt_tag}_EO{eo_li}Li1"
                combos.append({
                    "name":    cname,
                    "label":   f"{monomer} + {salt_tag}  EO:Li={eo_li}:1",
                    "monomer": monomer,
                    "salt":    salt,
                    "eo_li":   eo_li,
                    "enabled": True,
                })

    print()
    print(_bold(f"  All {len(combos)} polymer combinations:"))
    for i, c in enumerate(combos, 1):
        print(f"  {_green('✓')} {_cyan(str(i))}) {c['label']}")
    print()
    print(_dim("  Enter numbers to disable (space-separated), or Enter to keep all:"))
    try:
        raw = input("  Disable: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(combos):
            combos[int(tok) - 1]["enabled"] = False
    final = [c for c in combos if c["enabled"]]
    for c in final:
        del c["enabled"]
    print()
    print(f"  → {_green(str(len(final)) + ' combinations')} selected:")
    for c in final:
        print(f"     {_dim('•')} {c['label']}")
    return final


# ── Multi-select mobile ions ──────────────────────────────────────────────────

def _pick_mobile_ions_multi() -> list[str]:
    """Multi-select mobile ions. Returns list of ion strings (primary first)."""
    _hdr("Mobile Ion(s)")
    print()
    print(_bold("  Mobile ion  ") + _dim("(space-separated numbers, e.g.  1 3  for Li + K):"))
    for i, (key, desc) in enumerate(MOBILE_IONS, 1):
        print(f"  {_cyan(str(i))}) {_bold(key):<12}  {desc}")
    while True:
        try:
            raw = input("  Choice (space-separated) [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        has_custom = False
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(MOBILE_IONS):
                key = MOBILE_IONS[int(tok) - 1][0]
                if key == "custom":
                    has_custom = True
                elif key not in picks:
                    picks.append(key)
            else:
                # treat as literal ion name
                if tok not in picks:
                    picks.append(tok)
        if has_custom:
            cust = _input("  Custom ion name (e.g. Zn, H)")
            if cust and cust not in picks:
                picks.append(cust)
        if picks:
            print(f"  → {_green(', '.join(picks))}")
            return picks
        print(f"  {_yellow('Select at least one ion')}")


# ── Box category selection ────────────────────────────────────────────────────

def _ask_box_categories() -> list[str]:
    """Ask what goes in the box. Returns list of category keys."""
    _hdr("What's in the simulation box?")
    print()
    print(_dim("  Select all categories present in the simulation box."))
    print()
    print(_bold("  Categories  ") + _dim("(space-separated, e.g.  1 2 3 4):"))
    for i, (key, desc) in enumerate(BOX_CATEGORIES, 1):
        print(f"  {_cyan(str(i))}) {_bold(key):<14}  {desc}")
    print()
    print(_dim("  Examples:"))
    print(_dim("    1 2      → liquid electrolyte  (solvent + salt)"))
    print(_dim("    1 2 3    → gel (polymer + solvent + salt)"))
    print(_dim("    3 2      → solid polymer electrolyte"))
    print(_dim("    5        → bulk crystal / SSE / electrode"))
    print()
    default_hint = "1 2"
    while True:
        try:
            raw = input(f"  Select [{default_hint}]: ").strip() or default_hint
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        has_custom = False
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(BOX_CATEGORIES):
                key = BOX_CATEGORIES[int(tok) - 1][0]
                if key == "custom":
                    has_custom = True
                elif key not in picks:
                    picks.append(key)
            elif tok.lower() in {k.lower() for k, _ in BOX_CATEGORIES if k != "custom"}:
                # match by name
                for k, _ in BOX_CATEGORIES:
                    if tok.lower() == k.lower() and k not in picks and k != "custom":
                        picks.append(k)
        if has_custom:
            cust = _input("  Custom category name")
            if cust and cust not in picks:
                picks.append(cust)
        if picks:
            print(f"  → {_green(', '.join(picks))}")
            return picks
        print(f"  {_yellow('Select at least one category')}")


# ── Updated category percentages — accepts selected_categories parameter ──────

def _ask_category_percentages(selected_categories: list[str] | None = None,
                               normalize: bool = True) -> dict[str, float]:
    """Ask total vol% for each selected category.

    normalize=True  (default): must sum to ~100%; normalizes automatically.
    normalize=False: accepts partial sums; caller fills the remainder with
                     other categories (e.g. solvent+salt in a gel system).
    """
    _hdr("Box Composition — Category Volumes")
    print()
    if normalize:
        print(_dim("  Enter volume % for each category. Total must sum to 100%."))
    else:
        print(_dim("  Enter volume % of the TOTAL box for the polymer component(s)."))
        print(_dim("  Solvent and salt will fill the remaining volume automatically."))
    print(_dim("  Enter 0 to skip a category."))
    print()

    # Build list of (key, label) to ask
    if selected_categories:
        cats_to_ask = [(cat, cat.capitalize() + "  ") for cat in selected_categories
                       if cat not in ("solid", "custom")]
    else:
        cats_to_ask = [
            ("solvent",   "Solvents   "),
            ("salt",      "Salts      "),
            ("polymer",   "Polymers   "),
            ("copolymer", "Copolymers "),
        ]

    if not cats_to_ask:
        # Only solid/crystal selected — no volume fractions needed
        return {}

    while True:
        result: dict[str, float] = {}
        for key, label in cats_to_ask:
            v = _float(f"  {label} vol%", 0.0)
            if v > 0.0:
                result[key] = v
        s = sum(result.values())
        if not result:
            print(f"  {_yellow('Enter at least one category > 0%')}")
            continue
        if not normalize:
            if s > 100.0:
                print(f"  {_yellow(f'Total {s:.1f}% exceeds 100% — please re-enter.')}")
                continue
            print(f"\n  Polymer volume: {_cyan(f'{s:.1f}%')}  "
                  f"Remaining for solvent+salt: {_cyan(f'{100.0 - s:.1f}%')}  {_green('OK')}")
            break
        print(f"\n  Total: {_cyan(f'{s:.1f}%')}", end="")
        if abs(s - 100.0) < 2.0:
            print(f"  {_green(' OK')}")
            break
        print(f"  {_yellow(f' — expected 100%. Normalizing.')}")
        for k in result:
            result[k] = round(result[k] * 100.0 / s, 2)
        break
    return result


# ── Multi-select species for a category ──────────────────────────────────────

def _ask_species_multi(opts: list[tuple], category_name: str,
                       default_idx: int = 1) -> list[str]:
    """Multi-select species from opts list. Returns list of names."""
    print()
    print(_bold(f"  Select {category_name} types  ") +
          _dim("(space-separated, e.g.  1 3  or Enter for default):"))
    for i, (key, desc) in enumerate(opts, 1):
        print(f"  {_cyan(str(i))}) {_bold(key):<16}  {desc}")
    while True:
        try:
            raw = input(f"  Select [{default_idx}]: ").strip() or str(default_idx)
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[str] = []
        has_custom = False
        for tok in raw.split():
            if tok.isdigit() and 1 <= int(tok) <= len(opts):
                key = opts[int(tok) - 1][0]
                if key == "custom":
                    has_custom = True
                elif key not in picks:
                    picks.append(key)
        if has_custom:
            cust = _input(f"  Custom {category_name} name(s) — space-separated")
            for nm in cust.split():
                if nm and nm not in picks:
                    picks.append(nm)
        if picks:
            print(f"  → {_green(', '.join(picks))}")
            return picks
        print(f"  {_yellow('Select at least one item')}")


# ── Mixing level selection ────────────────────────────────────────────────────

def _ask_mixing_levels(n_species: int, category_name: str) -> list[int]:
    """Ask which mixing levels to generate combos for. Returns list of ints (1=mono, etc.)."""
    _hdr(f"Mixing levels for {category_name}")
    opts = [
        (1, "mono",    "single-component"),
        (2, "di",      "two-component"),
        (3, "tri",     "three-component"),
        (4, "quarter", "four-component"),
    ]
    available = [(lv, nm, desc) for lv, nm, desc in opts if lv <= n_species]
    print()
    print(_bold(f"  Mixing levels for {category_name}  ") +
          _dim("(space-separated, e.g.  1 2  for mono + binary):"))
    for lv, nm, desc in available:
        combos_hint = ""
        if lv == 1:
            combos_hint = f"  ({n_species} combos)"
        elif lv == 2:
            import math
            combos_hint = f"  (C({n_species},2)×3 = {math.comb(n_species,2)*3} combos)"
        elif lv == 3:
            import math
            combos_hint = f"  (C({n_species},3)×4 = {math.comb(n_species,3)*4} combos)"
        elif lv == 4:
            import math
            combos_hint = f"  (C({n_species},4)×5 = {math.comb(n_species,4)*5} combos)"
        print(f"  {_cyan(str(lv))}) {_bold(nm):<10}  {desc}{_dim(combos_hint)}")
    default = "1 2" if n_species >= 2 else "1"
    while True:
        try:
            raw = input(f"  Select [{default}]: ").strip() or default
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        picks: list[int] = []
        for tok in raw.split():
            if tok.isdigit():
                lv = int(tok)
                if any(lv == a[0] for a in available) and lv not in picks:
                    picks.append(lv)
        if picks:
            level_names = [nm for lv, nm, _ in available if lv in picks]
            print(f"  → {_green(', '.join(level_names))}")
            return sorted(picks)
        print(f"  {_yellow('Select at least one mixing level')}")


# ── Combination generator ─────────────────────────────────────────────────────

def _generate_combos(species_list: list[str], levels: list[int]) -> list[dict]:
    """
    Generate combination dicts for given mixing levels.
    Returns list of {"name", "label", "level", "components": [{"name", "ratio", "vol_frac"}]}
    """
    import itertools
    import math

    _DI_RATIOS    = [(2, 1), (1, 1), (1, 2)]
    _TRI_RATIOS   = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]
    _QUAD_RATIOS  = [(1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)]

    combos: list[dict] = []

    for level in sorted(levels):
        if level == 1:
            for sp in species_list:
                combos.append({
                    "name":  sp,
                    "label": sp,
                    "level": 1,
                    "components": [{"name": sp, "ratio": 1, "vol_frac": 1.0}],
                })
        elif level == 2:
            for sp1, sp2 in itertools.combinations(species_list, 2):
                for r1, r2 in _DI_RATIOS:
                    tot = r1 + r2
                    name = f"{sp1}_{sp2}_{r1}-{r2}"
                    combos.append({
                        "name":  name,
                        "label": f"{sp1}+{sp2} {r1}:{r2}",
                        "level": 2,
                        "components": [
                            {"name": sp1, "ratio": r1, "vol_frac": r1/tot},
                            {"name": sp2, "ratio": r2, "vol_frac": r2/tot},
                        ],
                    })
        elif level == 3:
            for sp1, sp2, sp3 in itertools.combinations(species_list, 3):
                for r1, r2, r3 in _TRI_RATIOS:
                    tot = r1 + r2 + r3
                    name = f"{sp1}_{sp2}_{sp3}_{r1}-{r2}-{r3}"
                    combos.append({
                        "name":  name,
                        "label": f"{sp1}+{sp2}+{sp3} {r1}:{r2}:{r3}",
                        "level": 3,
                        "components": [
                            {"name": sp1, "ratio": r1, "vol_frac": r1/tot},
                            {"name": sp2, "ratio": r2, "vol_frac": r2/tot},
                            {"name": sp3, "ratio": r3, "vol_frac": r3/tot},
                        ],
                    })
        elif level == 4:
            for sps in itertools.combinations(species_list, 4):
                for ratios in _QUAD_RATIOS:
                    tot = sum(ratios)
                    name = "_".join(sps) + "_" + "-".join(str(r) for r in ratios)
                    combos.append({
                        "name":  name,
                        "label": "+".join(sps) + " " + ":".join(str(r) for r in ratios),
                        "level": 4,
                        "components": [
                            {"name": sp, "ratio": r, "vol_frac": r/tot}
                            for sp, r in zip(sps, ratios)
                        ],
                    })
    return combos


def _show_and_prune_combos(combos: list[dict], category: str) -> list[dict]:
    """Show generated combos, let user disable any. Returns pruned list."""
    if not combos:
        return combos
    _hdr(f"Generated {category} combinations")
    # Group by level
    by_level: dict[int, list] = {}
    for c in combos:
        by_level.setdefault(c["level"], []).append(c)
    level_names = {1: "Mono", 2: "Di", 3: "Tri", 4: "Quarter"}
    idx = 1
    idx_map: dict[int, dict] = {}  # display idx → combo dict
    for lv in sorted(by_level):
        items = by_level[lv]
        print(f"\n  {_bold(level_names.get(lv, str(lv)))}  ({len(items)}):  " +
              "  |  ".join(_dim(c["label"]) for c in items[:6]) +
              (_dim(f"  ...+{len(items)-6}") if len(items) > 6 else ""))
        for c in items:
            print(f"  {_cyan(str(idx))}) {c['label']}")
            idx_map[idx] = c
            idx += 1
    print(f"\n  {_bold('Total:')} {len(combos)} combinations")
    print()
    print(_dim("  Enter numbers to disable (space-separated), or Enter to keep all:"))
    try:
        raw = input("  Disable: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    disabled: set[int] = set()
    for tok in raw.split():
        if tok.isdigit() and int(tok) in idx_map:
            disabled.add(int(tok))
    result = [idx_map[i] for i in sorted(idx_map) if i not in disabled]
    if disabled:
        print(f"  → {_green(str(len(result)))} combinations kept  "
              f"({len(disabled)} disabled)")
    return result


# ── Polymer chain specs per selected polymer ──────────────────────────────────

def _ask_polymer_chain_specs(polymer_names: list[str]) -> dict[str, dict]:
    """For each polymer name, ask n_chains and n_monomers."""
    _hdr("Polymer Chain Specifications")
    specs: dict[str, dict] = {}
    for name in polymer_names:
        print()
        print(_bold(f"  ── {name} ──────────────────────────────────────────────"))
        n_chains   = _int(f"  {name}: number of chains", 40)
        n_monomers = _int(f"  {name}: monomers per chain", 20)
        specs[name] = {"n_chains": n_chains, "n_monomers": n_monomers}
    return specs


def _ask_copolymer_specs(copoly_names: list[str]) -> dict[str, dict]:
    """For each copolymer name, ask ratio + chain details."""
    _hdr("Copolymer Specifications")
    _COPOLY_PRESETS = {
        "PVDF-HFP":  (["VDF", "HFP"],   [9, 1],   20, 40),
        "PVDF-TrFE": (["VDF", "TrFE"],  [75, 25], 20, 40),
    }
    specs: dict[str, dict] = {}
    for name in copoly_names:
        print()
        print(_bold(f"  ── {name} ──────────────────────────────────────────────"))
        preset = _COPOLY_PRESETS.get(name)
        if preset:
            components, def_ratio, def_monomers, def_chains = preset
            print(f"  Components: {_green(' + '.join(components))}")
        else:
            comp1 = _input("  Component 1 monomer name", "VDF")
            comp2 = _input("  Component 2 monomer name", "HFP")
            components = [comp1, comp2]
            def_ratio = [9, 1]; def_monomers = 20; def_chains = 40

        print(f"  Ratio  {':'.join(components)}  {_dim('(space-separated integers, e.g. 9 1)')}")
        while True:
            try:
                r_raw = input(f"  Ratio [{' '.join(str(r) for r in def_ratio)}]: "
                              ).strip() or " ".join(str(r) for r in def_ratio)
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            parts = r_raw.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                ratio = [int(p) for p in parts]
                break
            print(f"  {_yellow('Enter two integers e.g. 9 1')}")

        total_ratio = sum(ratio)
        minor_frac  = ratio[1] / total_ratio
        n_monomers  = _int(f"  Total monomers per chain", def_monomers)
        n_chains    = _int(f"  Number of chains", def_chains)
        comp_counts = [round(n_monomers * r / total_ratio) for r in ratio]
        print(f"  → {comp_counts[0]} {components[0]} + {comp_counts[1]} {components[1]} "
              f"per chain × {n_chains} chains")
        specs[name] = {
            "components":            components,
            "ratio":                 ratio,
            "n_chains":              n_chains,
            "n_monomers_per_chain":  n_monomers,
            "component_counts":      comp_counts,
            "minor_fraction":        minor_frac,
        }
    return specs


# ── Grand combination builder ─────────────────────────────────────────────────

def _build_grand_combinations(category_combos: dict[str, list[dict]]) -> list[dict]:
    """Cartesian product of all category combinations. Returns flat list of grand combos."""
    import itertools

    # Filter out empty categories
    active = {cat: combos for cat, combos in category_combos.items() if combos}
    if not active:
        return []

    cats  = list(active.keys())
    pools = [active[c] for c in cats]

    grand: list[dict] = []
    for product in itertools.product(*pools):
        name_parts  = []
        label_parts = []
        components: dict[str, dict] = {}
        for cat, combo in zip(cats, product):
            name_parts.append(combo["name"])
            label_parts.append(f"{cat}:{combo['label']}")
            components[cat] = combo
        grand.append({
            "name":       "_".join(name_parts),
            "label":      "  |  ".join(label_parts),
            "components": components,
        })
    return grand


def _ask_aimd_subset(grand_combos: list[dict]) -> list[dict]:
    """If many grand combos, ask user to select AIMD subset. Returns subset list."""
    n = len(grand_combos)
    _hdr("Grand Combinations")

    # Show summary per category
    if grand_combos:
        cats = list(grand_combos[0]["components"].keys())
        for cat in cats:
            cat_combos = list({gc["components"][cat]["name"]: gc["components"][cat]
                               for gc in grand_combos}.values())
            print(f"  {_bold(cat.capitalize() + ' combos'):<20}: {len(cat_combos)}")
    print(f"  {'─'*50}")
    cat_counts = []
    if grand_combos:
        cats = list(grand_combos[0]["components"].keys())
        for cat in cats:
            n_cat = len({gc["components"][cat]["name"] for gc in grand_combos})
            cat_counts.append(str(n_cat))
    formula = " × ".join(cat_counts) + f" = {n:,}" if cat_counts else str(n)
    print(f"  {_bold('Total AIMD cells')}: {formula}")

    if n > 500:
        print(f"\n  {_yellow('Warning:')} {n:,} grand combinations is very large.")
        print(_dim("  Recommend selecting an AIMD subset for DFT training data."))

    print()
    print(_bold("  Select AIMD subset:"))
    print(f"  {_cyan('1')}) All mono combinations only")
    print(f"  {_cyan('2')}) First N combinations")
    print(f"  {_cyan('3')}) All  (use full set for MLMD/CMD)")
    print(f"  {_cyan('4')}) Custom — enter numbers to include")

    while True:
        try:
            raw = input("  AIMD selection [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw == "1":
            subset = [gc for gc in grand_combos
                      if all(gc["components"][cat]["level"] == 1
                             for cat in gc["components"])]
            if not subset:
                subset = grand_combos[:min(10, n)]
            print(f"  → {_green(str(len(subset)))} mono-only combinations selected")
            return subset
        elif raw == "2":
            nn = _int("  Number of combinations (N)", min(20, n))
            subset = grand_combos[:nn]
            print(f"  → {_green(str(len(subset)))} combinations selected")
            return subset
        elif raw == "3":
            print(f"  → {_green(str(n))} combinations (all)")
            return grand_combos
        elif raw == "4":
            print(_dim(f"  Enter numbers 1–{n} (space-separated):"))
            for i, gc in enumerate(grand_combos[:min(30, n)], 1):
                print(f"  {_cyan(str(i))}) {gc['label'][:70]}")
            if n > 30:
                print(_dim(f"  ... and {n-30} more"))
            try:
                sel_raw = input("  Include: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            idxs = []
            for tok in sel_raw.split():
                if tok.isdigit() and 1 <= int(tok) <= n:
                    idxs.append(int(tok) - 1)
            subset = [grand_combos[i] for i in sorted(set(idxs))]
            if subset:
                print(f"  → {_green(str(len(subset)))} combinations selected")
                return subset
            print(f"  {_yellow('Select at least one')}")
        else:
            print(f"  {_yellow('Enter 1, 2, 3, or 4')}")


def _ask_salt_molarity() -> list[float]:
    """Ask for one or more salt molarities; returns [] for vol% mode.

    Multiple space-separated values create a combinatorial molarity series,
    e.g. '1.0 1.5 2.0' yields three sub-projects at each concentration.
    """
    _hdr("Salt Concentration")
    print("  Specify salt concentration as molarity (mol/L).")
    print("  This replaces the volume % questions for solvent and salt.")
    print("  Enter multiple space-separated values for a combinatorial series")
    print("  (e.g.  1.0 1.5 2.0  → one sub-project per concentration).")
    print("  Press Enter to use volume % instead.")
    try:
        raw = input("  Salt molarity [mol/L, or Enter for vol%]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    if not raw:
        return []
    try:
        return [v for v in (max(0.0, float(x)) for x in raw.split()) if v > 0]
    except ValueError:
        return []


# ── New _ask_composition_detail — multi-select + combo generation ─────────────

def _ask_composition_detail(cat_pcts: dict[str, float],
                             structs: list[dict],
                             selected_categories: list[str] | None = None,
                             molarity_mode: bool = False) -> dict:
    """
    Collect species details for each non-zero category using the new multi-select
    + mixing level + combo generation flow.

    molarity_mode=True: solvent and salt species are asked regardless of their
    vol% in cat_pcts (which is 0 or absent in gel+molarity mode because actual
    ratios are determined by salt_molarity, not vol%).

    Returns comp dict with keys:
      solvent_combos, salt_combos, polymer_combos, copolymer_combos,
      polymer_chain_specs, copolymer_specs,
      solvents (legacy list), salts (legacy list),
      polymers (legacy list), copolymers (legacy list)
    """
    comp: dict = {}
    active_cats = selected_categories or list(cat_pcts.keys())

    # ── Solvents ──────────────────────────────────────────────────────────────
    if "solvent" in active_cats and (molarity_mode or cat_pcts.get("solvent", 0) > 0):
        _hdr("Solvents")
        sv_opts: list[tuple[str, str]] = []
        seen: set[str] = set()
        for s in structs:
            if s.get("role") == "solvent":
                nm = s["file"].replace(".vasp","").replace(".cif","")
                if nm not in seen:
                    sv_opts.append((nm, f"{s['formula']:12}  {s['natoms']} atoms  [detected]"))
                    seen.add(nm)
        for k, d in SOLVENTS:
            if k not in seen and k != "custom":
                sv_opts.append((k, d))
                seen.add(k)
        sv_opts.append(("custom", "Other — type the name"))

        sv_names = _ask_species_multi(sv_opts, "solvent", default_idx=3)
        sv_levels = _ask_mixing_levels(len(sv_names), "solvents")
        sv_combos = _generate_combos(sv_names, sv_levels)
        sv_combos = _show_and_prune_combos(sv_combos, "solvent")
        comp["solvent_combos"] = sv_combos
        # Legacy: flat list of unique solvents used
        comp["solvents"] = [{"name": nm, "vol_pct": cat_pcts.get("solvent", 0.0)}
                            for nm in sv_names]

    # ── Salts ─────────────────────────────────────────────────────────────────
    if "salt" in active_cats and (molarity_mode or cat_pcts.get("salt", 0) > 0):
        _hdr("Salts")
        salt_opts = [(k, d) for k, d in SALTS if k != "none"]
        salt_opts.append(("custom", "Other — type the name"))
        salt_names = _ask_species_multi(salt_opts, "salt", default_idx=1)
        salt_levels = _ask_mixing_levels(len(salt_names), "salts")
        salt_combos = _generate_combos(salt_names, salt_levels)
        salt_combos = _show_and_prune_combos(salt_combos, "salt")
        comp["salt_combos"] = salt_combos
        # Legacy
        comp["salts"] = [{"name": nm, "vol_pct": cat_pcts.get("salt", 0.0)}
                         for nm in salt_names]

    # ── Homopolymers ──────────────────────────────────────────────────────────
    if "polymer" in active_cats and cat_pcts.get("polymer", 0) > 0:
        _hdr("Polymers (homopolymers)")
        mono_opts = [(k, d) for k, d in MONOMERS if k != "custom"]
        mono_opts.append(("custom", "Custom monomer — type name or SMILES"))
        poly_names = _ask_species_multi(mono_opts, "polymer", default_idx=1)
        poly_levels = _ask_mixing_levels(len(poly_names), "polymers")
        poly_combos = _generate_combos(poly_names, poly_levels)
        poly_combos = _show_and_prune_combos(poly_combos, "polymer")
        comp["polymer_combos"] = poly_combos
        # Ask chain specs for each polymer
        chain_specs = _ask_polymer_chain_specs(poly_names)
        comp["polymer_chain_specs"] = chain_specs
        # Legacy: list of polymer dicts
        pct_each = round(cat_pcts["polymer"] / max(1, len(poly_names)), 1)
        comp["polymers"] = [
            {
                "monomer":    nm,
                "n_chains":   chain_specs[nm]["n_chains"],
                "n_monomers": chain_specs[nm]["n_monomers"],
                "vol_pct":    pct_each,
            }
            for nm in poly_names
        ]

    # ── Copolymers ────────────────────────────────────────────────────────────
    if "copolymer" in active_cats and cat_pcts.get("copolymer", 0) > 0:
        _hdr("Copolymers")
        copoly_opts = [(k, d) for k, d in COPOLYMERS]
        copoly_names = _ask_species_multi(copoly_opts, "copolymer", default_idx=1)
        copoly_levels = _ask_mixing_levels(len(copoly_names), "copolymers")
        copoly_combos = _generate_combos(copoly_names, copoly_levels)
        copoly_combos = _show_and_prune_combos(copoly_combos, "copolymer")
        comp["copolymer_combos"] = copoly_combos
        # Ask ratio + chain specs
        copoly_specs = _ask_copolymer_specs(copoly_names)
        comp["copolymer_specs"] = copoly_specs
        # Legacy: list of copolymer dicts
        pct_each = round(cat_pcts["copolymer"] / max(1, len(copoly_names)), 1)
        comp["copolymers"] = [
            {
                "monomer":              nm,
                "components":           copoly_specs[nm]["components"],
                "ratio":                copoly_specs[nm]["ratio"],
                "n_chains":             copoly_specs[nm]["n_chains"],
                "n_monomers_per_chain": copoly_specs[nm]["n_monomers_per_chain"],
                "component_counts":     copoly_specs[nm]["component_counts"],
                "minor_fraction":       copoly_specs[nm]["minor_fraction"],
                "vol_pct":              pct_each,
            }
            for nm in copoly_names
        ]

    return comp




# ── Main wizard ───────────────────────────────────────────────────────────────

def _submit_to_inbox(project_dir: Path, name: str, doc: dict) -> "Path | None":
    """Validate, set local RUNNING control, and register canonical project.yaml."""
    try:
        from hpca.core.project_schema import validate
        from hpca.daemon.config import DaemonConfig
        from hpca.daemon.service import start_project
    except ImportError as exc:
        print(f"  {_yellow(f'Cannot import hpca.core modules: {exc}')}")
        return None

    # Validate schema — warn but allow user to override
    errors = validate(doc)
    if errors:
        print(f"\n  {_yellow('Validation warnings:')}")
        for err in errors:
            print(f"    {_yellow('·')} {err}")
        return None
    else:
        print(f"  {_green('✓')} project.yaml passes schema validation")

    try:
        return start_project(DaemonConfig(), project_dir / "project.yaml", name.lower())
    except Exception as exc:
        print(f"  {_yellow(f'Error registering project: {exc}')}")
        return None


def run_wizard(project_dir: Path) -> Path:
    """Run the interactive HPCA materials design wizard and write project.yaml to project_dir.

    Guides the user through project identity, box composition, system type, temperature
    sweeps, and workflow selection, then seeds the orchestrator state file for
    already-completed stages.

    Returns
    -------
    Path
        Path to the written project.yaml file.
    """
    if project_dir is None:
        project_dir = Path.cwd()

    print()
    print(_bold("=" * 58))
    print(_bold("   HPCA — Materials Design Wizard"))
    print(_bold("=" * 58))
    print(f"  Directory: {_cyan(str(project_dir))}")

    project_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = project_dir / "project.yaml"

    if yaml_path.exists():
        print(f"\n  {_yellow('project.yaml already exists.')}")
        if not _yes("Overwrite?", default=False):
            print("  Keeping existing file.")
            return yaml_path

    # ── Phase 0: Scan for structure files & existing data ────────────────────
    print(f"\n  Scanning {project_dir.name}...")

    structs = _detect_structures(project_dir)
    det     = _detect_sim_data(project_dir)
    done    = det.get("stages_done", [])

    if structs:
        print()
        print(_bold("  Structure files found:"))
        for s in structs:
            role_tag = _dim(f"[{s['role']}]")
            print(f"    {_cyan(s['file']):<22}  {s['formula']:<18}  "
                  f"{s['natoms']} atoms   {role_tag}")

    if done:
        print(f"  Completed stages: {_green(', '.join(done))}")

    # ── Phase 0b: Execution lane (always SLURM) ──────────────────────────────
    _sim_mode = "slurm"
    _ld = _lane_defaults()  # convenience alias for this session

    # ── Phase 1: Project identity ────────────────────────────────────────────
    _hdr("Project Identity")
    name       = _input("Project name", default=project_dir.name)
    full_name  = _input("Full material name", default=name.replace("_", " "))
    mobile_ions = _pick_mobile_ions_multi()
    mobile_ion  = mobile_ions[0]  # primary ion for backward compat

    # ── Phase 2: Box contents and composition ────────────────────────────────
    selected_categories = _ask_box_categories()

    # Offer molarity mode before vol% — only when both solvent and salt present
    has_liquid   = "solvent" in selected_categories and "salt" in selected_categories
    has_polymer  = bool({"polymer", "copolymer"} & set(selected_categories))
    salt_molarities: list[float] = []
    if has_liquid:
        salt_molarities = _ask_salt_molarity()
    salt_molarity = salt_molarities[0] if salt_molarities else 0.0

    if salt_molarity > 0 and not has_polymer:
        # Pure liquid: molarity defines all ratios — skip vol% questions
        cat_pcts = {"solvent": 75.0, "salt": 25.0}
    elif salt_molarity > 0 and has_polymer:
        # Gel/polymer-in-electrolyte: ask only polymer vol% of the total box.
        # normalize=False keeps the raw sum (e.g. 20%) so cat_pcts reflects
        # the true polymer fraction. Solvent/salt ratios come from salt_molarity
        # — no dummy vol% entries needed; molarity_mode=True bypasses the vol%>0 gate.
        cat_pcts = _ask_category_percentages(
            [c for c in selected_categories if c in ("polymer", "copolymer")],
            normalize=False)
    else:
        cat_pcts = _ask_category_percentages(selected_categories)

    comp = _ask_composition_detail(cat_pcts, structs, selected_categories,
                                   molarity_mode=(salt_molarity > 0))
    if salt_molarity > 0:
        # Molarity mode: strip vol_pct from solvents/salts — ratio is the only
        # relevant field. Keeps comp_spec clean with one concentration spec.
        comp["solvents"]      = [{"name": s["name"], "ratio": 1.0} for s in comp.get("solvents", [])]
        comp["salts"]         = [{"name": s["name"]}               for s in comp.get("salts",    [])]
        comp["salt_molarity"] = salt_molarity

    # ── Phase 2b: Grand combinations ─────────────────────────────────────────
    grand_combos = _build_grand_combinations({
        "solvent":   comp.get("solvent_combos", []),
        "salt":      comp.get("salt_combos", []),
        "polymer":   comp.get("polymer_combos", []),
        "copolymer": comp.get("copolymer_combos", []),
    })
    if grand_combos and len(grand_combos) > 1:
        aimd_combos = _ask_aimd_subset(grand_combos)
    elif grand_combos:
        aimd_combos = grand_combos
    else:
        aimd_combos = []

    # ── Expand by multiple molarities to create concentration series ─────────
    if len(salt_molarities) > 1:
        def _expand_molarity(combos: list[dict]) -> list[dict]:
            """Cross each grand combo with every requested salt molarity, appending a concentration tag."""
            base = combos
            if not base:
                # Single-species system: synthesize a base combo from comp
                sv_names = [s["name"] for s in comp.get("solvents", [])]
                sl_names = [s["name"] for s in comp.get("salts",    [])]
                components: dict = {}
                if sv_names:
                    components["solvent"] = {"name": "_".join(sv_names),
                                             "label": "+".join(sv_names), "level": 1,
                                             "components": [{"name": n, "ratio": 1} for n in sv_names]}
                if sl_names:
                    components["salt"]    = {"name": "_".join(sl_names),
                                             "label": "+".join(sl_names), "level": 1,
                                             "components": [{"name": n, "ratio": 1} for n in sl_names]}
                nm  = "_".join(c["name"]  for c in components.values())
                lbl = " + ".join(c["label"] for c in components.values())
                base = [{"name": nm, "label": lbl, "components": components}]
            expanded = []
            for gc in base:
                for m in salt_molarities:
                    m_tag = f"{m:.1f}M".replace(".", "p")
                    expanded.append({**gc,
                        "name":          f"{gc['name']}_{m_tag}",
                        "label":         f"{gc['label']} {m:.1f} M",
                        "salt_molarity": m})
            return expanded
        grand_combos = _expand_molarity(grand_combos)
        # AIMD is a chemistry-coverage dataset, not a bulk-concentration
        # simulation.  Do not duplicate identical reference cells for every
        # requested MLMD/CMD molarity.

    # Convert grand_combos to legacy _combinations format
    _combinations = [
        {
            "name":          gc["name"],
            "label":         gc["label"],
            "solvents": [
                {"name": c["name"], "ratio": c["ratio"]}
                for c in gc["components"].get("solvent", {}).get("components", [])
            ] if "solvent" in gc["components"] else [],
            "salt": (gc["components"].get("salt", {}).get("components", [{}])[0].get("name", "")
                     if "salt" in gc["components"] else ""),
            **({"salt_molarity": gc["salt_molarity"]} if "salt_molarity" in gc else {}),
        }
        for gc in aimd_combos
    ]

    # ── Phase 3: System type (auto-detected from composition) ─────────────
    _hdr("System Type")
    # Auto-detect from what's in the box
    has_poly    = bool(comp.get("polymers") or comp.get("copolymers"))
    has_solvent = bool(comp.get("solvents"))
    has_salt    = bool(comp.get("salts"))
    if has_poly and has_solvent:
        suggested = "gel"
    elif has_poly:
        suggested = "polymer"
    elif has_solvent:
        suggested = "liquid_electrolyte"
    else:
        suggested = "bulk_sse"
    if structs and not has_poly and not has_solvent:
        suggested = _suggest_system(structs)
    suggested_desc = next((d for k, d in SYSTEM_TYPES if k == suggested), suggested)
    print(f"\n  Based on your box contents: {_green(suggested_desc)}")
    if _yes("Is that correct?", default=True):
        system_type = suggested
    else:
        system_type = _pick("System type", SYSTEM_TYPES, default=1)

    category   = CATEGORY_MAP[system_type]
    is_liquid  = "liquid" in system_type
    is_polymer = "polymer" in system_type or system_type == "gel"
    is_sse     = "sse" in system_type
    is_surface = system_type in ("surface",)
    is_np      = system_type in ("nanoparticle", "np_substrate")
    is_iface   = system_type in ("sse_electrode","sse_liquid","electrode_liquid","np_substrate")
    is_solid   = not is_liquid and not is_polymer

    # ── Input structure file (inorganic_sse / solid systems) ─────────────────
    _input_structure_path: str = ""
    if is_sse or (is_solid and not is_iface and not is_surface and not is_np):
        _hdr("Input Structure File")
        # Check if a crystal structure was already detected in the project directory
        _crystal_structs = [s for s in structs if s["role"] in
                            ("halide_sse", "sulfide_sse", "oxide_electrode", "unknown")
                            and s["natoms"] > 0]
        if _crystal_structs:
            s0 = _crystal_structs[0]
            print(f"\n  Auto-detected: {_cyan(s0['file'])}  ({s0['formula']}, {s0['natoms']} atoms)")
            if _yes("Use this as the input structure?", default=True):
                _input_structure_path = str(project_dir / s0["file"])
        if not _input_structure_path:
            print(f"\n  Enter the absolute path to your input VASP/CIF structure file.")
            from hpca.core.config import Config as _ExCfg
            _ex_base = _ExCfg.get().hpc("project_base", "/path/to/projects")
            print(f"  {_dim(f'Example: {_ex_base}/MyMaterial/inputs/MyMaterial.vasp')}")
            while True:
                raw = _input("Structure file path")
                if not raw:
                    print(f"  {_yellow('No structure file specified — you can add cif: manually in project.yaml')}")
                    break
                p = Path(raw.strip())
                if p.exists():
                    try:
                        from pymatgen.core import Structure as _Struct
                        _s = _Struct.from_file(str(p))
                        print(f"  {_green('✓')} {_s.composition.reduced_formula}  "
                              f"{len(_s)} atoms  species: {sorted(set(str(e) for e in _s.species))}")
                    except Exception:
                        print(f"  {_yellow('⚠ Could not parse structure — will use path as-is')}")
                    _input_structure_path = str(p)
                    break
                else:
                    print(f"  {_yellow('File not found:')} {p}  — try again (or press Enter to skip)")

    # HPCA reports and continuum models use a fixed reference temperature.
    # Keeping this out of the wizard avoids presenting a parameter that is not
    # intended to vary between projects.
    T_ref = 300

    # ── Phase 4: Simulation tiers, temperatures, and variants ─────────────
    sim: dict = {}
    _combinations  = []
    # Deduplicated pools from all combos
    _sv_pool = list(dict.fromkeys(
        c["name"]
        for combo in comp.get("solvent_combos", [])
        for c in combo.get("components", [])
    ))
    if not _sv_pool:
        _sv_pool = [sv["name"] for sv in comp.get("solvents", [])]

    _salt_pool = list(dict.fromkeys(
        c["name"]
        for combo in comp.get("salt_combos", [])
        for c in combo.get("components", [])
    ))
    if not _salt_pool:
        _salt_pool = [sl["name"] for sl in comp.get("salts", [])]

    _monomer_pool  = ([p["monomer"] for p in comp.get("polymers", [])] +
                      [cp["monomer"] for cp in comp.get("copolymers", [])])
    _poly_salt_pool = []
    _polymer_comps  = []  # for YAML polymers block

    # Build polymers block for YAML
    for p in comp.get("polymers", []):
        _polymer_comps.append({
            "monomer":    p["monomer"],
            "chain_length": p["n_monomers"],
            "n_chains":   p["n_chains"],
        })
    for cp in comp.get("copolymers", []):
        _polymer_comps.append({
            "monomer":       cp["monomer"],
            "chain_length":  cp["n_monomers_per_chain"],
            "n_chains":      cp["n_chains"],
            "copolymer_ratio": dict(zip(cp["components"], cp["ratio"])),
            "minor_fraction":  cp["minor_fraction"],
        })

    if is_liquid or is_polymer or system_type == "gel":
        tier_result = _ask_composition_tiers(comp, cat_pcts)
        temp_cfg    = _ask_temperatures_hierarchy(tier_result["tiers_selected"])
        variants    = _ask_composition_variants(comp, cat_pcts, tier_result)
        sim.update(tier_result)
        sim.update(temp_cfg)
        sim["cat_pcts"]  = cat_pcts
        _cs: dict = {
            "solvents":   comp.get("solvents",   []),
            "salts":      comp.get("salts",       []),
            "polymers":   comp.get("polymers",    []),
            "copolymers": comp.get("copolymers",  []),
        }
        if comp.get("salt_molarity", 0) > 0:
            _cs["salt_molarity"] = comp["salt_molarity"]
        sim["comp_spec"] = _cs
        if variants:
            sim["composition_variants"] = variants

    if is_solid or is_iface:
        sc_cfg = _ask_solid_supercell()
        sim.update(sc_cfg)

    if is_iface:
        _hdr("Interface Materials")
        if is_sse:
            mat1_name = _input("SSE material name / formula", "Li2ZrCl6")
        else:
            mat1_name = _input("Material 1 name", "electrode")
        sim["material_1"] = mat1_name

        if system_type == "sse_electrode":
            electrode = _pick("Electrode material", ELECTRODES, default=1)
            if electrode == "custom":
                electrode = _input("Electrode name")
            sim["material_2"] = electrode
        elif system_type == "sse_liquid":
            sim["material_2"] = "electrolyte"
        elif system_type == "np_substrate":
            sim["material_2"] = _input("Substrate material", "graphite")

        iface_geo = _ask_interface_geometry()
        sim.update(iface_geo)

    if is_surface:
        surf_geo = _ask_surface_geometry()
        sim.update(surf_geo)

    if is_np:
        np_geo = _ask_nanoparticle_geometry()
        sim.update(np_geo)

    # ── Polymer / gel configuration ───────────────────────────────────────────
    # For gel: the liquid branch above already called _ask_study_design() and
    # _ask_box_composition_and_tiers() (which sets forcefield, aimd_temps, etc.).
    # Skip those questions here to avoid double-asking.
    _tier_system_ran = bool(sim.get("box_fractions"))
    if is_polymer or system_type == "gel":
        if _monomer_pool and len(_monomer_pool) > 1:
            # Multiple monomers → automatically multi-combination
            if not _monomer_pool:
                _monomer_pool, _poly_salt_pool = _build_polymer_pool(structs)
            else:
                _poly_salt_pool = _salt_pool if _salt_pool else []
                if not _poly_salt_pool:
                    _, _poly_salt_pool = _build_polymer_pool(structs)
            if not _tier_system_ran:
                _combinations = _ask_polymer_combinations_matrix(_monomer_pool, _poly_salt_pool)

        primary_monomer = _monomer_pool[0] if _monomer_pool else "PEO"
        # Skip chain/FF questions if tier system already captured them
        if not _tier_system_ran:
            poly_cfg = _ask_polymer_chain_from_pool(primary_monomer)
            sim.update(poly_cfg)

        if system_type == "gel":
            if "solvents" not in sim:
                if _sv_pool:
                    sim["solvents"] = [{"name": sv, "ratio": 1} for sv in _sv_pool]
                else:
                    sim["solvents"] = _ask_solvents([])
            if "salt" not in sim:
                if _salt_pool:
                    sim["salt"] = _salt_pool[0]
                else:
                    sim["salt"] = _ask_salt("")

    # Solid temperature sweeps
    if is_solid or is_iface:
        existing_aimd = [int(Path(p).name.replace("K",""))
                         for p in det.get("aimd_dirs",[])
                         if Path(p).name.replace("K","").isdigit()]
        existing_mlmd = list(det.get("mlmd_dirs",{}).keys())
        temp_cfg = _ask_solid_temps(existing_aimd, existing_mlmd)
        sim.update(temp_cfg)

    # SSE doping variants
    _doping_variants: list = []
    _doping_encut: float = 0.0
    if is_solid:
        _doping_variants, _doping_encut = _ask_doping_variants(
            name, struct_path=_input_structure_path,
            aimd_supercell=sim.get("aimd_supercell", "1x1x1"),
        )

    # ── Phase 5: DFT / VASP settings ────────────────────────────────────────
    _hdr("VASP / DFT Settings")
    kspacing_dft  = 100.0 if (is_liquid or is_polymer) else 0.2
    kspacing_aimd = 100.0 if (is_liquid or is_polymer) else 0.4
    sim["dft"] = {
        "kspacing_dft":  kspacing_dft,
        "kspacing_aimd": kspacing_aimd,
        "aimd_potim_fs": 1.0,
    }

    # ── Phase 6: Mechanical (SSE only) ───────────────────────────────────────
    extras = _ask_mechanics(is_solid or is_iface)

    # ── Phase 7: Manuscript ──────────────────────────────────────────────────
    ms_meta = _ask_manuscript(full_name)

    # ── Phase 8: Enable stages ───────────────────────────────────────────────
    print()
    run_rem = _yes("Enable remaining stages for autonomous execution?", default=True)

    # ── Build directory lists ─────────────────────────────────────────────────
    aimd_temps_final: list = sim.get("aimd_temps", [])
    mlmd_temps_final: list = sim.get("mlmd_temps", [])
    # concs kept for backward-compat summary prints; new wizard uses single vol% box
    concs:            list = [1.0]
    aimd_concs_final: list = [1.0]

    aimd_dirs: list = []
    mlmd_dirs_final: dict = {}
    cmd_dirs: list = []

    aimd_dirs = det.get("aimd_dirs", [])
    if not aimd_dirs and aimd_temps_final:
        aimd_dirs = [f"aimd/{T}K" for T in aimd_temps_final]

    mlmd_dirs_final = det.get("mlmd_dirs", {})
    if not mlmd_dirs_final and mlmd_temps_final:
        mlmd_dirs_final = {f"{T}K": f"dlmd/{T}K" for T in mlmd_temps_final}

    if sim.get("classical_md"):
        if grand_combos:
            cmd_dirs = [f"cmd/{gc['name']}" for gc in grand_combos]
        else:
            cmd_dirs = ["cmd"]

    # ── Assemble and write YAML ───────────────────────────────────────────────
    import yaml

    doc: dict = {
        "name":         name,
        "full_name":        full_name,
        "mobile_ion":       mobile_ion,
        "mobile_ions":      mobile_ions,
        "category":         category,
        "system_type":      system_type,
        "execution_mode":   "slurm",
        "workflow_version": 2,
        "T_ref":            T_ref,
        "root":             str(project_dir),
        "project_root":     str(project_dir),
    }

    if _combinations:
        if _monomer_pool:
            doc["components"] = {
                "monomers": [{"name": m} for m in _monomer_pool],
                "salts":    [{"name": s} for s in _poly_salt_pool],
            }
        else:
            doc["components"] = {
                "solvents": [{"name": sv} for sv in _sv_pool],
                "salts":    [{"name": s}  for s in _salt_pool],
            }
        doc["combinations"] = _combinations

    # Store all grand combinations and AIMD subset
    if grand_combos:
        def _gc_entry(gc: dict) -> dict:
            """Extract minimal identifying fields (name, label, optional molarity) from a grand-combo dict."""
            entry: dict = {"name": gc["name"], "label": gc["label"]}
            if "salt_molarity" in gc:
                entry["salt_molarity"] = gc["salt_molarity"]
            return entry

        doc["grand_combinations_total"] = len(grand_combos)
        doc["aimd_combinations"] = [_gc_entry(gc) for gc in aimd_combos]
        doc["mlmd_combinations"] = [_gc_entry(gc) for gc in grand_combos]
        # cmd_combinations: full component detail for per-combination box building
        if sim.get("classical_md"):
            doc["cmd_combinations"] = [
                {
                    **_gc_entry(gc),
                    "components": {
                        cat: {
                            "label":      combo["label"],
                            "level":      combo["level"],
                            "components": combo["components"],
                        }
                        for cat, combo in gc["components"].items()
                    },
                }
                for gc in grand_combos
            ]

    if structs:
        doc["structure_files"] = [
            {"file": s["file"], "formula": s["formula"], "role": s["role"]}
            for s in structs
        ]

    if _input_structure_path:
        doc["cif"] = _input_structure_path

    if _doping_variants:
        doc["crystal_doping_variants"] = _doping_variants
    if _doping_encut:
        doc["encut"] = _doping_encut

    if det.get("deepmd_pot"):
        doc["deepmd_pot"] = det["deepmd_pot"]
    if aimd_dirs:
        doc["aimd_dirs"] = aimd_dirs
    if mlmd_dirs_final:
        doc["mlmd_dirs"] = mlmd_dirs_final
    else:
        doc["mlmd_dirs"] = {}
    if cmd_dirs:
        doc["cmd_dirs"] = cmd_dirs

    doc.update(extras)
    doc.update(ms_meta)

    # Polymer per-chain composition detail (copolymer ratios, chain counts)
    if _polymer_comps:
        doc["polymers"] = _polymer_comps

    # Inject canonical SLURM production parameters from platform.yaml
    # so users can review and edit them before the production run.
    # resolve() in base.py reads these keys; platform.yaml limits.slurm is the fallback.
    try:
        _plat_sim = (yaml.safe_load(
            (Path(__file__).parents[2] / "config" / "platform.yaml").read_text()
        ) or {}).get("limits", {}).get("slurm", {})
        _slurm_canon = [
            "aimd_steps", "aimd_dataset_steps", "npt_steps_aimd",
            "dft_nsw_vcrelax", "dft_nsw_opt",
            "cmd_npt_ps", "cmd_nvt_ns",
            "mlmd_npt_ps", "mlmd_nvt_ns",
            "mlip_numb_steps",
        ]
        for _k in _slurm_canon:
            if _k in _plat_sim and _k not in sim:
                sim[_k] = _plat_sim[_k]
    except Exception:
        pass

    doc["simulation"] = sim
    doc["stages"]     = _stages_block(system_type, done, run_rem,
                                       tiers_selected=sim.get("tiers_selected", []))
    doc["autonomy"]   = {
        "mode": "unattended" if run_rem else "attended",
        "auto_approve_validated_design": bool(run_rem),
    }

    if is_solid:
        estimate = _solid_workload_estimate(doc)
        _hdr("Pre-submission Workload Estimate")
        print(f"  Crystal variants       : {estimate['variants']}")
        print(f"  DFT jobs               : ~{estimate['dft_jobs']}")
        print(f"  AIMD/dataset jobs      : ~{estimate['aimd_jobs']}")
        print(f"  NEB pipelines          : ~{estimate['neb_jobs']}")
        print(f"  Minimum SLURM jobs     : ~{estimate['minimum_slurm_jobs']}")
        print(f"  VASP ionic steps       : ~{estimate['vasp_ionic_steps']:,}")
        print(_dim("  This excludes retries, active-learning labels, and scheduler restarts."))
        if not _yes("Write project.yaml with this workload?", default=False):
            print(f"  {_yellow('Cancelled before writing or daemon registration.')}")
            return yaml_path

    with open(yaml_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if done:
        _seed_state(project_dir, done)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(_green(f"  ✓ project.yaml written → {yaml_path}"))
    if done:
        print(_green(f"  ✓ Pre-seeded {len(done)} completed stages"))
    print()
    print(_bold("  Summary"))
    print(f"    Material   : {full_name}")
    print(f"    System     : {system_type}")
    _ion_str = ", ".join(mobile_ions) if len(mobile_ions) > 1 else mobile_ion
    print(f"    Mobile ion : {_ion_str}")
    if salt_molarity > 0:
        _mol_str = " / ".join(f"{m:.2g} M" for m in salt_molarities)
        print(f"    Molarity   : {_mol_str}")
    else:
        _box_cats = [f"{cat}({pct:.0f}%)" for cat, pct in cat_pcts.items()] if cat_pcts else []
        print(f"    Box        : {', '.join(_box_cats) if _box_cats else 'crystal/solid'}")

    if _combinations and _monomer_pool:
        print(f"    Mode       : {_green('multi-combination')}  ({len(_combinations)} polymer systems)")
        print(f"    Monomers   : {', '.join(_monomer_pool)}")
        print(f"    Salts      : {', '.join(_poly_salt_pool) if _poly_salt_pool else 'none'}")
        print(f"    Systems    :")
        for c in _combinations:
            print(f"                 {_dim('•')} {c['label']}")
    elif aimd_combos:
        # Combinatorial mode (species combos, concentration series, or both)
        print(f"    Mode       : {_green('combinatorial')}  ({len(aimd_combos)} sub-projects)")
        for gc in aimd_combos[:8]:
            m_tag = (f"  {gc['salt_molarity']:.2g} M"
                     if "salt_molarity" in gc else "")
            sv = gc["components"].get("solvent", {}).get("label", "")
            sl = gc["components"].get("salt",    {}).get("label", "")
            desc = " + ".join(x for x in [sv, sl] if x) or gc["label"]
            print(f"                 {_dim('•')} {desc}{_dim(m_tag)}")
        if len(aimd_combos) > 8:
            print(f"                 {_dim(f'... and {len(aimd_combos)-8} more')}")
        n_temps = len(aimd_temps_final) if aimd_temps_final else 0
        if n_temps:
            print(f"    AIMD       : {n_temps} temps × {len(aimd_combos)} sub-projects"
                  f" = {n_temps * len(aimd_combos)} cells")
        if mlmd_dirs_final:
            print(f"    MLMD       : {len(mlmd_dirs_final)} temps × {len(grand_combos)} variants"
                  f" = {len(mlmd_dirs_final) * len(grand_combos)} cells")
        if cmd_dirs:
            print(f"    CMD        : {len(cmd_dirs)} variants × {len(sim.get('cmd_temps', []))} temps"
                  f" = {len(cmd_dirs) * len(sim.get('cmd_temps', []))} cells")
    elif "solvents" in sim:
        sv_str = " + ".join(
            (f"{s['name']}({s['ratio']})" if len(sim['solvents']) > 1 else s['name'])
            for s in sim["solvents"]
        )
        print(f"    Solvents   : {sv_str}")
        if sim.get("salt", "none") != "none":
            print(f"    Salt       : {sim['salt']}")
        if aimd_dirs:
            print(f"    AIMD cells : {len(aimd_dirs)}  ({len(aimd_temps_final)} temps)")
            if len(aimd_dirs) <= 8:
                for d in aimd_dirs:
                    print(f"                 {_dim(d)}")
        if mlmd_dirs_final:
            print(f"    MLMD cells : {len(mlmd_dirs_final)}")
        if cmd_dirs:
            print(f"    CMD cells  : {len(cmd_dirs)}")

    # ── Structure check and auto-fetch ───────────────────────────────────────
    import os
    struct_list = doc.get("structure_files", [])
    if struct_list:
        from hpca.sim.structure_fetch import check_missing, fetch_structure
        missing = check_missing(project_dir, struct_list)
        if missing:
            _hdr("Missing Structure Files")
            for s in missing:
                print(f"  {_yellow('✗')} {s['file']}  [{s['role']}]")
            if _yes("Auto-download missing structures (PubChem / Materials Project)?",
                    default=True):
                mp_key = os.environ.get("MP_API_KEY", "")
                if not mp_key and any(s["role"] != "solvent" for s in missing):
                    mp_key = _input(
                        "  MP_API_KEY for inorganic structures (Enter to skip)", ""
                    )
                for s in missing:
                    sname = s["file"].replace(".vasp", "")
                    fetch_structure(sname, s["role"], project_dir,
                                    api_key=mp_key or None)
        else:
            print(f"\n  {_green('✓')} All structure files present in {project_dir.name}/")

    # ── Schema validation + inbox submission ─────────────────────────────────
    _inbox_path: "Path | None" = None
    _hdr("Daemon Inbox")
    print(f"  {_dim('The HPCA daemon polls daemon_inbox/active/ every 60 s.')}")
    print(f"  {_dim('Submitting here registers this project for automatic processing.')}")
    print()
    if _yes("Submit project to daemon inbox?", default=True):
        _inbox_path = _submit_to_inbox(project_dir, name, doc)
        if _inbox_path:
            print(f"\n  {_green('✓')} Registered: {_dim(str(_inbox_path))}")
            print(f"  {_dim('The daemon will pick it up on its next poll cycle.')}")
        else:
            print(f"\n  {_yellow('⚠')} Could not register — fall back to per-project script below")

    # ── Resolve paths from platform.yaml ────────────────────────────────────
    from hpca.core.paths import load_platform_config as _lpc_wiz
    _plat_wiz   = _lpc_wiz()
    _hpc_wiz    = _plat_wiz.get("hpc", {})
    _orch_py    = str(Path(__file__).resolve().parents[1] / "orchestrator" / "hpca_orchestrator.py")
    _python_bin = _hpc_wiz.get("python_cladue", sys.executable)
    _conda_env  = str(Path(_python_bin).parents[1])   # strip /bin/python3
    _pkg_parent = str(Path(__file__).resolve().parents[2])  # parent of hpca/ package
    _slurm_acct = _plat_wiz.get("hpc", {}).get("accounts", {}).get("standard") or _account_fallback()

    # ── Write project-local orchestrator submission script ───────────────────
    local_orch_sh = project_dir / "sub_orchestrator.sh"
    _orch_script = f"""\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --account={_slurm_acct}
#SBATCH --job-name=hpca-orc
#SBATCH --error={project_dir}/logs/%J.stderr
#SBATCH --output={project_dir}/logs/%J.stdout
#SBATCH --chdir={project_dir}
#SBATCH --signal=B:USR1@3600

set -euo pipefail
SELF="$(realpath "${{BASH_SOURCE[0]}}")"
PROJECT_DIR="{project_dir}"

_resubmit() {{
    echo "[$(date)] Wall-time approaching — resubmitting"
    NEW_JOB=$(sbatch --parsable "$SELF")
    echo "[$(date)] Resubmitted as job $NEW_JOB"
    kill -TERM "$ORCH_PID" 2>/dev/null || true
    wait "$ORCH_PID" 2>/dev/null || true
    exit 0
}}
trap '_resubmit' USR1

{_mbl('conda').strip()}
source activate {_conda_env}

export PACKMOL_BIN="${{PACKMOL_BIN:-{_hpc_wiz.get('packmol_bin', 'packmol')}}}"
export MP_API_KEY="${{MP_API_KEY:-}}"
export PYTHONPATH="{_pkg_parent}:${{PYTHONPATH:-}}"

LOG_DIR="{project_dir}/logs"
mkdir -p "${{LOG_DIR}}"
echo "Job ${{SLURM_JOB_ID}} started on $(hostname) at $(date)" >> "${{LOG_DIR}}/job_history.txt"

ORCH_PY="{_orch_py}"
{_python_bin} "${{ORCH_PY}}" --resume \\
    --log-dir="{project_dir}/logs" \\
    --root="{project_dir}" &
ORCH_PID=$!

echo "[$(date)] Orchestrator PID: $ORCH_PID  JobID: ${{SLURM_JOB_ID}}"
wait "$ORCH_PID"
EXIT_CODE=$?
echo "[$(date)] Orchestrator exited with code $EXIT_CODE"
echo "Job ${{SLURM_JOB_ID}} finished (exit=$EXIT_CODE) at $(date)" >> "${{LOG_DIR}}/job_history.txt"
exit $EXIT_CODE
"""
    local_orch_sh.write_text(_orch_script)
    local_orch_sh.chmod(0o755)
    print(f"\n  {_green('✓')} Submission script written → {_dim('sub_orchestrator.sh')}")

    # ── Launch orchestrator ───────────────────────────────────────────────────
    _hdr("Launch Orchestrator")

    import subprocess as _sp

    _orch_env = {
        **__import__("os").environ,
        "PACKMOL_BIN": _hpc_wiz.get("packmol_bin", "packmol"),
        "PYTHONPATH": f"{_pkg_parent}:{__import__('os').environ.get('PYTHONPATH', '')}",
    }

    # ── SLURM lane: inbox mode or per-project ─────────────────────────────────
    if _inbox_path is not None:
        # Project was submitted to inbox — guide user to start/check the daemon
        print(f"  Project in inbox: {_dim(str(_inbox_path))}")
        print(f"  Script           : {_dim(str(local_orch_sh))}")
        print()
        print(_bold("  Daemon options:"))
        print(f"  {_cyan('1')}) {_bold('Start daemon')}  — launch inbox daemon on this node (background)")
        print(f"  {_cyan('2')}) {_bold('SLURM daemon')} — submit inbox daemon as SLURM job  "
              f"{_dim('(10 days, auto-resubmit)')}")
        print(f"  {_cyan('3')}) {_bold('Skip')}          — daemon already running / I'll start it manually")
        while True:
            try:
                raw = input("  Choice [3]: ").strip() or "3"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw in ("1", "2", "3"):
                break
            print(f"  {_yellow('Enter 1, 2, or 3')}")

        if raw == "1":
            _inbox_log = _inbox_path.parent.parent / "logs"
            _inbox_log.mkdir(parents=True, exist_ok=True)
            proc = _sp.Popen(
                [_python_bin, _orch_py, "--inbox", "--resume",
                 f"--log-dir={_inbox_log}"],
                stdout=open(str(_inbox_log / "daemon.log"), "a"),
                stderr=_sp.STDOUT,
                env=_orch_env,
                start_new_session=True,
            )
            print(f"  {_green('✓')} Inbox daemon PID {_bold(str(proc.pid))}  "
                  f"(background, this node)")
            print(f"  {_dim('  Log:  tail -f ' + str(_inbox_log / 'daemon.log'))}")
        elif raw == "2":
            r = _sp.run(["sbatch", str(local_orch_sh)], capture_output=True, text=True)
            if r.returncode == 0:
                job_id = r.stdout.strip().split()[-1]
                print(f"  {_green('✓')} Submitted as SLURM job {_bold(job_id)}  "
                      f"(10 days, auto-resubmit)")
            else:
                print(f"  {_red('✗')} sbatch failed: {r.stderr.strip()}")
                print(f"  {_dim('  Retry:  sbatch sub_orchestrator.sh')}")
        else:
            print(f"  {_dim('  Start daemon manually when ready:')}")
            print(f"  {_dim('    ' + _python_bin + ' ' + _orch_py + ' --inbox --resume')}")
            print(f"  {_dim('    (or: sbatch sub_orchestrator.sh)')}")
    else:
        # Inbox was declined or failed — per-project SLURM script
        print(f"  Project dir  : {_dim(str(project_dir))}")
        print(f"  Script       : {_dim(str(local_orch_sh))}")
        print()
        print(_bold("  How should the orchestrator run?"))
        print(f"  {_cyan('1')}) {_bold('This node')}   — background process on this login/compute node (no queue)")
        print(f"  {_cyan('2')}) {_bold('SLURM queue')} — submit as a SLURM daemon job  "
              f"{_dim('(10 days, auto-resubmit)')}")
        print(f"  {_cyan('3')}) {_bold('Both')}         — run here now + submit a SLURM backup")
        print(f"  {_cyan('4')}) {_bold('Skip')}         — I'll launch manually later")
        while True:
            try:
                raw = input("  Choice [2]: ").strip() or "2"
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw in ("1", "2", "3", "4"):
                break
            print(f"  {_yellow('Enter 1, 2, 3, or 4')}")

        def _run_on_this_node() -> None:
            """Launch the HPCA orchestrator as a background subprocess on the current login node."""
            logs_dir = project_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / "orch_daemon.log"
            proc = _sp.Popen(
                [_python_bin, _orch_py,
                 "--resume",
                 f"--log-dir={logs_dir}",
                 f"--root={project_dir}"],
                stdout=open(str(log_file), "w"),
                stderr=_sp.STDOUT,
                cwd=str(project_dir),
                env=_orch_env,
                start_new_session=True,
            )
            print(f"  {_green('✓')} Orchestrator PID {_bold(str(proc.pid))}  "
                  f"(background, this node)")
            print(f"  {_dim('  Log:  tail -f ' + str(log_file))}")

        def _submit_slurm() -> None:
            """Submit the orchestrator as a 10-day auto-resubmitting Slurm job."""
            r = _sp.run(["sbatch", str(local_orch_sh)], capture_output=True, text=True)
            if r.returncode == 0:
                job_id = r.stdout.strip().split()[-1]
                print(f"  {_green('✓')} Submitted as SLURM job {_bold(job_id)}  "
                      f"(10 days, auto-resubmit)")
            else:
                print(f"  {_red('✗')} sbatch failed: {r.stderr.strip()}")
                print(f"  {_dim('  Retry:  sbatch sub_orchestrator.sh')}")

        if raw == "1":
            _run_on_this_node()
        elif raw == "2":
            _submit_slurm()
        elif raw == "3":
            _run_on_this_node()
            _submit_slurm()
        else:
            print(f"  {_dim('  Run when ready:')}")
            print(f"  {_dim('    This node : ' + _python_bin + ' ' + _orch_py + ' --resume --root=' + str(project_dir))}")
            print(f"  {_dim('    SLURM     : sbatch sub_orchestrator.sh')}")

    return yaml_path


# ── Interactive scaffold generation ──────────────────────────────────────────

def _write_placeholder_poscar(path: Path, formula: str = "Li8Cl8",
                                natoms: int = 16) -> None:
    """Write a minimal VASP-format placeholder POSCAR for scaffold testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = max(5.0, (natoms ** (1/3)) * 2.5)
    sp_parts = []
    for el in ["Li", "Na", "K", "Cl", "F", "O", "S", "C", "H", "N", "P",
               "Zr", "Y", "Al", "Mg", "Ca", "Ni", "Mn", "Co", "Fe", "V"]:
        if el in formula:
            sp_parts.append(el)
    if not sp_parts:
        sp_parts = ["X", "Y"]
    n_each = max(1, natoms // len(sp_parts))
    total  = n_each * len(sp_parts)
    coords = "\n".join(
        f"  {((i * 1.0/total) % 1):.6f}  {((i * 0.37/total) % 1):.6f}  "
        f"{((i * 0.71/total) % 1):.6f}"
        for i in range(total)
    )
    path.write_text(
        f"{formula} placeholder — replace with real POSCAR\n"
        f"1.0\n"
        f"  {a:.6f}  0.000000  0.000000\n"
        f"  0.000000  {a:.6f}  0.000000\n"
        f"  0.000000  0.000000  {a:.6f}\n"
        f"  {' '.join(sp_parts)}\n"
        f"  {' '.join(str(n_each) for _ in sp_parts)}\n"
        f"Direct\n"
        f"{coords}\n"
    )


# ── Standalone ────────────────────────────────────────────────────────────────

def main():
    """CLI entry point: parse args and launch the wizard for the given (or current) project directory."""
    import argparse
    p = argparse.ArgumentParser(description="HPCA materials design wizard")
    p.add_argument("project_dir", nargs="?")
    args = p.parse_args()
    import os
    def _shell_cwd():
        """Return the shell's $PWD (respects symlinks) rather than os.getcwd()."""
        return Path(os.environ.get("PWD", os.getcwd()))
    if args.project_dir:
        d = Path(args.project_dir)
        project_dir = (_shell_cwd() / d) if not d.is_absolute() else d
    else:
        project_dir = _shell_cwd()
    run_wizard(project_dir)


if __name__ == "__main__":
    main()
