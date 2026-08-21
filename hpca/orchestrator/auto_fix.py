"""
auto_fix.py — Auto-detect and fix common VASP/LAMMPS/DeepMD failures.

All fix functions return True if a fix was applied (caller should then resubmit).
Uses only stdlib + pathlib — no heavy imports.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("hpca.orch")


# ── VASP OUTCAR error detection ────────────────────────────────────────────────

def detect_vasp_error(work_dir: Path) -> str | None:
    """
    Scan OUTCAR and OSZICAR in work_dir for known error signatures.
    Returns an error key string or None if no error found.
    """
    outcar = work_dir / "OUTCAR"
    oszicar = work_dir / "OSZICAR"

    if outcar.exists():
        # Read last 300 lines for efficiency
        try:
            lines = outcar.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-300:])
        except Exception:
            tail = ""

        if "triple product of the basis vectors is negative" in tail:
            return "NEGATIVE_VOLUME"
        if "I REFUSE TO CONTINUE WITH THIS SICK JOB" in tail:
            # Check for Bravais mismatch before generic SICK_JOB
            if "Inconsistent Bravais lattice" in tail or "monoclinic" in tail or "triclinic" in tail:
                return "SICK_JOB_SYMPREC"
            return "SICK_JOB"
        if "number of potentials on File POTCAR incompatible" in tail:
            return "POTCAR_MISMATCH"
        if "LAPACK: Routine ZPOTRF failed" in tail or "Cholesky" in tail:
            return "ZPOTRF"
        if "ZBRENT: fatal error" in tail or "ZBRENT: FATAL ERROR" in tail:
            return "ZBRENT"
        if "Sub-Space-Matrix is not hermitian" in tail:
            return "SUB_SPACE"
        if "reached the maximum number of electronic SC-loops" in tail:
            return "NELM"
        if "Error EDDDAV" in tail:
            return "EDDDAV"
        if "WARNING in PSSYEVX" in tail or "Error in PSSYEVX" in tail:
            return "PSSYEVX"
        if "Error EDDRMM" in tail:
            return "EDDRMM"
        # TOO_FEW_BANDS
        if re.search(r"TOO FEW BANDS", tail, re.IGNORECASE):
            return "TOO_FEW_BANDS"
        # REAL_OPT
        if re.search(r"WARNING: Sub-Space-Matrix is not hermitian|REAL_OPT", tail):
            return "REAL_OPT"
        # PRICEL symmetry error
        if re.search(r"PRICEL|internal error in subroutine PRICEL", tail, re.IGNORECASE):
            return "PRICEL"
        # Very large forces
        if re.search(r"VERY LARGE FORCES", tail, re.IGNORECASE):
            return "VERY_LARGE_FORCES"
        # Check if converged (not an error)
        if "reached required accuracy" in tail or "Voluntary context" in tail:
            return None  # actually done

    if oszicar.exists():
        try:
            osc_tail = oszicar.read_text(errors="replace").splitlines()[-20:]
            if any("NaN" in line for line in osc_tail):
                return "NAN_TEMP"
        except Exception:
            pass

    return None


def detect_walltime(work_dir: Path, job_id: str | None = None) -> bool:
    """
    Return True if job appears to have hit wall time:
    OUTCAR exists but is incomplete (no "Voluntary" at end) and job not alive.
    """
    outcar = work_dir / "OUTCAR"
    if not outcar.exists():
        return False
    try:
        tail = outcar.read_text(errors="replace").splitlines()[-20:]
        finished = any("Voluntary" in l or "reached required" in l for l in tail)
        return not finished
    except Exception:
        return False


# ── SYMPREC escalation helpers ─────────────────────────────────────────────────

def _current_symprec(incar_path: Path) -> float | None:
    """Return the float value of SYMPREC from INCAR, or None if the tag is absent."""
    if not incar_path.exists():
        return None
    for line in incar_path.read_text().splitlines():
        m = re.match(r"^\s*SYMPREC\s*=\s*([0-9eE+\-\.]+)", line, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def fix_sick_job_symprec(incar_path: Path, is_neb: bool = False) -> bool:
    """
    Fix SICK_JOB / SICK_JOB_SYMPREC "Inconsistent Bravais lattice" error.

    For NEB (is_neb=True): tighten SYMPREC to 1E-8 so VASP treats images
    consistently with the endpoint reference geometry.

    For regular VASP (is_neb=False): loosen SYMPREC progressively so VASP
    uses the same (more permissive) Bravais class for direct and reciprocal
    lattice — default(1E-5) → 1E-3 → 1E-2 → GIVE UP.

    Returns True if INCAR was modified (caller should resubmit), False to skip.
    """
    if is_neb:
        incar_set(incar_path, "SYMPREC", "1E-8")
        log.warning("fix_sick_job_symprec (NEB): set SYMPREC=1E-8 in %s", incar_path)
        return True

    try:
        from hpca.core.config import Config
        escalation = Config.get().auto_fix("symprec_escalation", [1e-5, 1e-3, 1e-2])
    except Exception:
        escalation = [1e-5, 1e-3, 1e-2]

    cur = _current_symprec(incar_path) or escalation[0]
    # Find next threshold strictly larger than current
    next_val = next((v for v in escalation if v > cur + 1e-15), None)
    if next_val is None:
        log.warning("fix_sick_job_symprec: SYMPREC already at max (%.0e) in %s — giving up", cur, incar_path)
        return False

    next_str = f"{next_val:.0E}"
    incar_set(incar_path, "SYMPREC", next_str)
    log.warning("fix_sick_job_symprec: escalated SYMPREC %.0e → %s in %s", cur, next_str, incar_path)
    return True


# ── ZPOTRF (Cholesky) fix ──────────────────────────────────────────────────────

def fix_wavecar_zero_byte(work_dir: Path) -> bool:
    """Delete a 0-byte WAVECAR that causes Cholesky decomposition failure on restart."""
    wavecar = work_dir / "WAVECAR"
    if wavecar.exists() and wavecar.stat().st_size == 0:
        wavecar.unlink()
        log.warning("fix_wavecar_zero_byte: deleted 0-byte WAVECAR in %s", work_dir)
        return True
    return False


def fix_too_few_bands(incar_path: Path, outcar_path: Path) -> bool:
    """Read current NBANDS from OUTCAR and write NBANDS*1.25 to INCAR."""
    import math, re as _re
    try:
        text = outcar_path.read_text(errors="replace")
        m = _re.search(r"NBANDS\s*=\s*(\d+)", text)
        if not m:
            return False
        n_current = int(m.group(1))
        n_new = math.ceil(n_current * 1.25)
        # Read/update INCAR
        lines = incar_path.read_text().splitlines()
        new_lines = []
        found = False
        for line in lines:
            if re.match(r"\s*NBANDS\s*=", line, re.IGNORECASE):
                new_lines.append(f"NBANDS = {n_new}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"NBANDS = {n_new}")
        incar_path.write_text("\n".join(new_lines) + "\n")
        log.warning("[auto_fix] TOO_FEW_BANDS: NBANDS %d → %d", n_current, n_new)
        return True
    except Exception as exc:
        log.debug("[auto_fix] fix_too_few_bands failed: %s", exc)
        return False


# ── Main INCAR patcher ─────────────────────────────────────────────────────────

def fix_incar(incar_path: Path, error_key: str, is_neb: bool = False) -> bool:
    """
    Apply fix to INCAR for a known error_key.
    Returns True if fix was applied.

    Pass is_neb=True when fixing a NEB image INCAR so SICK_JOB_SYMPREC uses
    the tighter 1E-8 fix instead of the lenient escalation path.
    """
    # SICK_JOB variants use escalation logic, not a fixed patch dict
    if error_key in ("SICK_JOB", "SICK_JOB_SYMPREC"):
        return fix_sick_job_symprec(incar_path, is_neb=is_neb)

    # TOO_FEW_BANDS requires reading OUTCAR for current NBANDS
    if error_key == "TOO_FEW_BANDS":
        outcar_path = incar_path.parent / "OUTCAR"
        if outcar_path.exists():
            return fix_too_few_bands(incar_path, outcar_path)
        return False

    # REAL_OPT: disable real-space projection
    if error_key == "REAL_OPT":
        try:
            incar_set(incar_path, "LREAL", "False")
            log.warning("Fixed INCAR %s: set LREAL=False for REAL_OPT", incar_path)
            return True
        except Exception as exc:
            log.error("Failed to fix INCAR %s for REAL_OPT: %s", incar_path, exc)
            return False

    # PRICEL: disable symmetry
    if error_key == "PRICEL":
        try:
            incar_set(incar_path, "ISYM", "0")
            log.warning("Fixed INCAR %s: set ISYM=0 for PRICEL", incar_path)
            return True
        except Exception as exc:
            log.error("Failed to fix INCAR %s for PRICEL: %s", incar_path, exc)
            return False

    # VERY_LARGE_FORCES: switch to RMM-DIIS
    if error_key == "VERY_LARGE_FORCES":
        try:
            incar_set(incar_path, "IBRION", "1")
            log.warning("Fixed INCAR %s: set IBRION=1 for VERY_LARGE_FORCES", incar_path)
            return True
        except Exception as exc:
            log.error("Failed to fix INCAR %s for VERY_LARGE_FORCES: %s", incar_path, exc)
            return False

    # Hardcoded fallbacks for errors not covered by platform.yaml
    _HARDCODED: dict[str, dict] = {
        "ZPOTRF":    {"LREAL": "F"},
        "ZBRENT":    {"IBRION": "1", "POTIM": "0.10"},
        "SUB_SPACE": {"ALGO": "All", "AMIX": "0.2"},
        "NELM":      {"NELM": "200", "AMIX": "0.1"},
        "WALLTIME":  {"ISTART": "1", "ICHARG": "1"},
        "NAN_TEMP":  {"ISTART": "1", "ICHARG": "1"},
    }
    # detect_vasp_error() returns NELM/SUB_SPACE; platform.yaml uses NELM_MAX/Sub-Space
    _CONFIG_KEY_ALIAS = {"NELM": "NELM_MAX", "SUB_SPACE": "Sub-Space"}

    try:
        from hpca.core.config import Config
        cfg_fixes = Config.get().auto_fix("incar_fixes", {})
    except Exception:
        cfg_fixes = {}

    config_key = _CONFIG_KEY_ALIAS.get(error_key, error_key)
    if config_key in cfg_fixes:
        patch = {str(k): str(v) for k, v in cfg_fixes[config_key].items()}
    elif error_key in _HARDCODED:
        patch = _HARDCODED[error_key]
    else:
        return False
    try:
        if error_key == "NAN_TEMP":
            # Primary fix: remove NELMDL (its negative default causes T=NaN on restart)
            incar_remove(incar_path, "NELMDL")

        for key, value in patch.items():
            incar_set(incar_path, key, value)

        log.warning("Fixed INCAR %s: applied %s patches for %s", incar_path, list(patch), error_key)
        return True
    except Exception as exc:
        log.error("Failed to fix INCAR %s: %s", incar_path, exc)
        return False


# ── DeepMD error detection ─────────────────────────────────────────────────────

def detect_deepmd_error(log_path: Path, step_history: list[int] | None = None) -> str | None:
    """Detect DeepMD training failures from log file."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None

    if "CUDA out of memory" in text or "RuntimeError: CUDA" in text:
        return "OOM"
    if "version" in text.lower() and "mismatch" in text.lower():
        return "TORCH_MISMATCH"

    # Stalled: same step appears in last 6 recorded steps
    if step_history and len(step_history) >= 6:
        if len(set(step_history[-6:])) == 1:
            return "STALLED"

    return None


def fix_deepmd_input(json_path: Path, error_key: str) -> bool:
    """Apply fix to deepmd_input.json. Returns True if fix applied."""
    if error_key != "OOM" or not json_path.exists():
        return False
    try:
        try:
            from hpca.core.config import Config
            factor = float(Config.get().auto_fix("deepmd_batch_reduction_factor", 0.5))
        except Exception:
            factor = 0.5

        data = json.loads(json_path.read_text())
        batch = data.get("training", {}).get("batch_size", 32)
        new_batch = max(1, int(batch * factor))
        data["training"]["batch_size"] = new_batch
        json_path.write_text(json.dumps(data, indent=2))
        log.warning("Fixed DeepMD batch_size %d → %d for OOM (factor=%.2f)", batch, new_batch, factor)
        return True
    except Exception as exc:
        log.error("Failed to fix deepmd_input.json: %s", exc)
        return False


# ── POSCAR negative volume fix ─────────────────────────────────────────────────

def fix_negative_volume(poscar_path: Path) -> bool:
    """
    If POSCAR has a left-handed (negative volume) cell, swap b and c vectors.
    Returns True if POSCAR was modified.
    """
    if not poscar_path.exists():
        return False
    try:
        lines = poscar_path.read_text().splitlines()
        if len(lines) < 7:
            return False
        scale = float(lines[1].strip())
        a = [float(x) for x in lines[2].split()]
        b = [float(x) for x in lines[3].split()]
        c = [float(x) for x in lines[4].split()]
        # Triple product = a · (b × c)
        bxc = [
            b[1]*c[2] - b[2]*c[1],
            b[2]*c[0] - b[0]*c[2],
            b[0]*c[1] - b[1]*c[0],
        ]
        triple = a[0]*bxc[0] + a[1]*bxc[1] + a[2]*bxc[2]
        if triple >= 0:
            return False
        # Swap b and c to make triple product positive
        new_lines = list(lines)
        new_lines[3] = lines[4]
        new_lines[4] = lines[3]
        poscar_path.write_text("\n".join(new_lines) + "\n")
        log.warning("fix_negative_volume: swapped b↔c vectors in %s (triple was %.4f)", poscar_path, triple)
        return True
    except Exception as exc:
        log.error("fix_negative_volume failed for %s: %s", poscar_path, exc)
        return False


# ── POTCAR species fix ─────────────────────────────────────────────────────────

def fix_potcar_species(poscar_path: Path, potcar_path: Path) -> bool:
    """
    Ensure POTCAR species count matches POSCAR species list.
    Scans upstream directories for a POTCAR with the correct number of sections.
    Returns True if POTCAR was fixed/replaced.
    """
    if not poscar_path.exists() or not potcar_path.exists():
        return False

    # Read species from POSCAR line 6 (0-indexed line 5)
    try:
        lines = poscar_path.read_text().splitlines()
        species_line = lines[5].split()
        n_species_poscar = len(species_line)
    except Exception as exc:
        log.warning("fix_potcar_species: cannot read POSCAR %s: %s", poscar_path, exc)
        return False

    # Count species in current POTCAR
    try:
        potcar_text = potcar_path.read_text(errors="replace")
        n_sections = potcar_text.count("End of Dataset")
    except Exception:
        return False

    if n_sections == n_species_poscar:
        return False  # already correct

    log.warning("fix_potcar_species: POSCAR has %d species, POTCAR has %d sections — searching fix",
                n_species_poscar, n_sections)

    # Search parent directories for a POTCAR with the correct section count
    search_dirs = list(poscar_path.parents)[:4]
    for parent in search_dirs:
        candidate = parent / "POTCAR"
        if candidate == potcar_path or not candidate.exists():
            continue
        try:
            n = candidate.read_text(errors="replace").count("End of Dataset")
            if n == n_species_poscar:
                import shutil
                shutil.copy(candidate, potcar_path)
                log.warning("fix_potcar_species: replaced %s with %s (%d sections)",
                            potcar_path, candidate, n)
                return True
        except Exception:
            continue

    log.error("fix_potcar_species: could not find POTCAR with %d sections near %s",
              n_species_poscar, poscar_path)
    return False


# ── Auto-fix attempt budget ────────────────────────────────────────────────────

def within_fix_budget(state: "object", handler_key: str) -> bool:
    """Return True if the handler has not yet exhausted its auto-fix attempt budget."""
    try:
        from hpca.core.config import Config
        max_attempts = int(Config.get().auto_fix("max_auto_fix_attempts", 3))
    except Exception:
        max_attempts = 3
    count = state.get_handler(handler_key).get("fix_count", 0)
    return count < max_attempts


def increment_fix_count(state: "object", handler_key: str) -> int:
    """Increment and persist the fix attempt counter for handler_key. Returns new count."""
    handler_data = dict(state.get_handler(handler_key))
    count = handler_data.get("fix_count", 0) + 1
    handler_data["fix_count"] = count
    state.set_handler(handler_key, handler_data)
    return count


# ── INCAR key helpers ──────────────────────────────────────────────────────────

def incar_get(incar_path: Path, key: str) -> str | None:
    """Read one INCAR key value. Returns None if not found."""
    if not incar_path.exists():
        return None
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*(.+?)(\s*!.*)?$", re.IGNORECASE)
    for line in incar_path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


def incar_set(incar_path: Path, key: str, value: str) -> None:
    """Set or add a KEY = value line in INCAR, replacing any existing occurrence."""
    lines = incar_path.read_text().splitlines() if incar_path.exists() else []
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=", re.IGNORECASE)
    new_line = f"{key:<12} = {value}"
    replaced = False
    new_lines = []
    for line in lines:
        if pattern.match(line):
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(new_line)
    incar_path.write_text("\n".join(new_lines) + "\n")


def incar_remove(incar_path: Path, key: str) -> None:
    """Remove all lines matching KEY = ... from INCAR (case-insensitive)."""
    if not incar_path.exists():
        return
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=", re.IGNORECASE)
    lines = [l for l in incar_path.read_text().splitlines() if not pattern.match(l)]
    incar_path.write_text("\n".join(lines) + "\n")
