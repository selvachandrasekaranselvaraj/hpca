"""Scientific reference checks for RDF normalization."""
from __future__ import annotations

import numpy as np

from hpca.analysis.rdf import compute_rdf


def test_uniform_periodic_fluid_rdf_approaches_one():
    rng = np.random.default_rng(42)
    n_frames, n_atoms, box_length = 80, 300, 20.0
    positions = rng.uniform(0, box_length, size=(n_frames, n_atoms, 3))
    box = np.tile(np.array([[0.0, box_length]] * 3), (n_frames, 1, 1))
    result = compute_rdf({"positions": positions, "species": ["Li"] * n_atoms,
                          "box": box}, "Li", "Li", r_max=7.0, n_bins=70,
                         skip_frac=0.0)
    r = result["r_centers"]
    bulk = result["g_r"][(r > 1.0) & (r < 6.0)]
    np.testing.assert_allclose(np.mean(bulk), 1.0, atol=0.04)
