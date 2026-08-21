"""Tests for hpca.io.rmse DeepMD lcurve parsing."""
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
LCURVE = FIXTURES / "lcurve.out"


def test_parse_deepmd_lcurve_returns_dict():
    from hpca.io.rmse import parse_deepmd_lcurve
    result = parse_deepmd_lcurve(LCURVE)
    assert result is not None
    assert "step" in result
    assert "e_rmse_eV" in result
    assert "f_rmse_eV_A" in result


def test_parse_deepmd_lcurve_last_step():
    from hpca.io.rmse import parse_deepmd_lcurve
    result = parse_deepmd_lcurve(LCURVE)
    assert result["step"] == 10000


def test_parse_deepmd_lcurve_values():
    from hpca.io.rmse import parse_deepmd_lcurve
    result = parse_deepmd_lcurve(LCURVE)
    assert result["e_rmse_eV"] < 0.01
    assert result["f_rmse_eV_A"] < 0.1


def test_parse_deepmd_lcurve_history():
    from hpca.io.rmse import parse_deepmd_lcurve_history
    history = parse_deepmd_lcurve_history(LCURVE)
    assert isinstance(history, list)
    assert len(history) >= 5
    assert history[0]["step"] == 0
    assert history[-1]["step"] == 10000


def test_converged_true():
    from hpca.io.rmse import converged
    assert converged({"e_rmse_eV": 0.002, "f_rmse_eV_A": 0.015}) == True


def test_converged_false():
    from hpca.io.rmse import converged
    assert converged({"e_rmse_eV": 0.1, "f_rmse_eV_A": 0.5}) == False
