import numpy as np
import pytest
import pcc
from astropy.table import Table


def test_apply_gains_preserves_pedestal_and_scales_signal():
    ped = 1000.0
    img = np.full((4, 4, 3), ped, dtype=float)
    img[0, 0] = [ped + 100, ped + 100, ped + 100]
    out = pcc.apply_gains(img, (2.0, 1.0, 0.5), ped)
    # background (at pedestal) is unchanged
    assert np.allclose(out[1, 1], ped)
    # signal scales around the pedestal, per channel
    assert np.allclose(out[0, 0], [ped + 200, ped + 100, ped + 50])


def test_apply_gains_identity():
    rng = np.random.RandomState(0)
    img = rng.uniform(0, 1000, (3, 3, 3))
    out = pcc.apply_gains(img, (1.0, 1.0, 1.0), 100.0)
    assert np.allclose(out, img)


def test_solve_gains_recovers_known_color_law():
    rng = np.random.RandomState(1)
    bp_rp = rng.uniform(0.2, 2.0, 300)
    g = rng.uniform(100, 1000, 300)
    cr = 0.30 * bp_rp + 0.70          # instrumental r/g
    cb = -0.20 * bp_rp + 1.30         # instrumental b/g
    matched = Table({"r": cr * g, "g": g, "b": cb * g, "bp_rp": bp_rp})

    gains, diag = pcc.solve_gains(matched, ref_bp_rp=0.82)

    assert np.isclose(gains[1], 1.0)
    assert np.isclose(gains[0], 1.0 / (0.30 * 0.82 + 0.70), rtol=1e-3)
    assert np.isclose(gains[2], 1.0 / (-0.20 * 0.82 + 1.30), rtol=1e-3)
    assert diag["n"] > 250
