import numpy as np
import pytest
import pcc


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
