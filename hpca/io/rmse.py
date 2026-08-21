"""
hpca/io/rmse.py — DeePMD-kit / MACE training convergence parsing.

Consolidates duplicate logic from:
  tools/deepmd.py (check_training, parse_lcurve, test_model),
  orchestrator/handlers/h04_mlip.py (poll convergence),
  orchestrator/handlers/h13_active_learning.py (_parse_test_results),
  sim/mlip.py (parse_lcurve_status)

lcurve.out column layout (DeePMD-kit):
  step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  lr
  [0]   [1]       [2]       [3]         [4]          [5]         [6]         [7]

Usage:
    from hpca.io.rmse import parse_deepmd_lcurve, rmse_summary, converged
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ── lcurve.out parsing ────────────────────────────────────────────────────────

def parse_deepmd_lcurve(lcurve_path: Path | str) -> dict[str, Any]:
    """
    Parse last non-comment line of lcurve.out.
    Returns {step, e_rmse_eV, f_rmse_eV_A, lr} or {} on failure.
    """
    path = Path(lcurve_path)
    if not path.exists():
        return {}
    last: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                last = stripped.split()
    except Exception:
        return {}
    return _lcurve_row(last)


def parse_deepmd_lcurve_history(lcurve_path: Path | str) -> list[dict[str, Any]]:
    """
    Parse all non-comment lines of lcurve.out.
    Returns list of {step, e_rmse_eV, f_rmse_eV_A, lr}.
    """
    path = Path(lcurve_path)
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                r = _lcurve_row(stripped.split())
                if r:
                    rows.append(r)
    except Exception:
        pass
    return rows


def _lcurve_row(parts: list[str]) -> dict[str, Any]:
    """Convert a split lcurve.out row into a {step, e_rmse_eV, f_rmse_eV_A, lr} dict."""
    if len(parts) < 6:
        return {}
    try:
        return {
            "step":        int(parts[0]),
            "e_rmse_eV":   float(parts[3]),
            "f_rmse_eV_A": float(parts[5]),
            "lr":          float(parts[7]) if len(parts) > 7 else None,
        }
    except (ValueError, IndexError):
        return {}


# ── dp test output parsing ────────────────────────────────────────────────────

_RE_E_RMSE    = re.compile(r"Energy\s+RMSE\s*:\s*([\d.eE+\-]+)\s*eV/atom", re.I)
_RE_F_RMSE    = re.compile(r"Force\s+RMSE\s*:\s*([\d.eE+\-]+)\s*eV", re.I)
_RE_N_FRAMES  = re.compile(r"Total\s+(\d+)\s+frames", re.I)


def parse_test_results(txt_path: Path | str) -> dict[str, Any]:
    """
    Parse dp test output (test_results.txt or similar).
    Returns {e_rmse_eV, f_rmse_eV_A, n_frames} or partial dict on failure.
    """
    path = Path(txt_path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    result: dict[str, Any] = {}
    m = _RE_E_RMSE.search(text)
    if m:
        result["e_rmse_eV"] = float(m.group(1))
    m = _RE_F_RMSE.search(text)
    if m:
        result["f_rmse_eV_A"] = float(m.group(1))
    m = _RE_N_FRAMES.search(text)
    if m:
        result["n_frames"] = int(m.group(1))
    return result


# ── convergence gate ──────────────────────────────────────────────────────────

def converged(rmse: dict[str, Any], cfg=None) -> bool:
    """
    Return True if e_rmse_eV < threshold_e AND f_rmse_eV_A < threshold_f.
    Thresholds read from cfg.mlip() (keys: rmse_e_threshold, rmse_f_threshold).
    Defaults: 5e-3 eV/atom, 0.1 eV/Å.
    """
    if not rmse:
        return False

    thr_e = 5e-3
    thr_f = 0.1
    if cfg is None:
        try:
            from hpca.core.config import Config
            cfg = Config.get()
        except Exception:
            pass
    if cfg is not None:
        try:
            ml = cfg.mlip()
            thr_e = ml.get("rmse_e_threshold", thr_e)
            thr_f = ml.get("rmse_f_threshold", thr_f)
        except Exception:
            pass

    e = rmse.get("e_rmse_eV")
    f = rmse.get("f_rmse_eV_A")
    if e is None or f is None:
        return False
    return e < thr_e and f < thr_f


# ── aggregate summary ─────────────────────────────────────────────────────────

def rmse_summary(mlff_dir: Path | str, cfg=None) -> dict[str, Any]:
    """
    Return RMSE summary for a completed MLIP training directory.

    Search order:
      1. mlff_dir/test_results.txt
      2. mlff_dir/01.train/lcurve.out
      3. mlff_dir/lcurve.out

    Returns {e_rmse_eV, f_rmse_eV_A, n_frames?, converged, source}.
    """
    d = Path(mlff_dir)

    # 1. dp test output
    test_file = d / "test_results.txt"
    r = parse_test_results(test_file)
    if r.get("e_rmse_eV") is not None:
        r["converged"] = converged(r, cfg)
        r["source"]    = str(test_file)
        return r

    # 2. lcurve in DeePMD 01.train subdir
    for lc_path in (d / "01.train" / "lcurve.out", d / "lcurve.out"):
        r = parse_deepmd_lcurve(lc_path)
        if r:
            r["converged"] = converged(r, cfg)
            r["source"]    = str(lc_path)
            return r

    return {"converged": False, "source": None}
