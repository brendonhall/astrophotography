import numpy as np
import pytest
import pcc
from astropy.table import Table
from astropy.wcs import WCS


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


def _toy_wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-0.001, 0.001]   # deg/pixel
    w.wcs.crval = [211.0, 54.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_cross_match_pairs_within_tolerance():
    w = _toy_wcs()
    xs, ys = np.array([50.0, 60.0, 30.0]), np.array([50.0, 40.0, 70.0])
    sky = w.pixel_to_world(xs, ys)
    stars = Table({"x": xs, "y": ys,
                   "r": [10.0, 20.0, 30.0], "g": [10.0, 20.0, 30.0], "b": [10.0, 20.0, 30.0]})
    # Gaia: first two coincide with stars; third is far away (no match expected)
    gaia = Table({
        "ra": [sky[0].ra.deg, sky[1].ra.deg, 100.0],
        "dec": [sky[0].dec.deg, sky[1].dec.deg, 10.0],
        "bp_rp": [0.5, 1.5, 2.5],
    })
    matched = pcc.cross_match(stars, gaia, w, tol_arcsec=5.0)
    assert len(matched) == 2
    assert set(np.round(matched["bp_rp"], 1)) == {0.5, 1.5}
    assert np.all(matched["sep_arcsec"] < 5.0)
