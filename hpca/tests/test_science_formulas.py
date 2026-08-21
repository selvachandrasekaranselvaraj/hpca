import pytest

from hpca.science.formulas import (
    arrhenius_diffusivity, einstein_diffusivity, nernst_einstein_conductivity,
)


def test_einstein_conversion_and_dimensions():
    assert einstein_diffusivity(6.0, 3) == pytest.approx(1.0e-8)
    with pytest.raises(ValueError, match="dimensions"):
        einstein_diffusivity(1.0, 4)


def test_reference_arrhenius_is_identity_at_300_k():
    assert arrhenius_diffusivity(2e-10, 0.3, 300.0) == pytest.approx(2e-10)
    assert arrhenius_diffusivity(2e-10, 0.3, 600.0) > 2e-10


def test_nernst_einstein_reference_value_and_domains():
    sigma = nernst_einstein_conductivity(1e-10, 1e27, 300)
    assert sigma == pytest.approx(0.6196, rel=1e-3)
    with pytest.raises(ValueError, match="positive Kelvin"):
        nernst_einstein_conductivity(1e-10, 1e27, 0)
