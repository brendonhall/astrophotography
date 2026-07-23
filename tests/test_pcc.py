import numpy as np
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


def test_cross_match_dedup_keeps_closest_star():
    w = _toy_wcs()
    # Two stars near the same Gaia source: star0 is an exact sky match (sep=0),
    # star1 is offset by 1 pixel (cdelt=0.001 deg/px -> ~3.6"), still within tol_arcsec.
    xs, ys = np.array([50.0, 51.0]), np.array([50.0, 50.0])
    sky = w.pixel_to_world(xs, ys)
    stars = Table({"x": xs, "y": ys,
                   "r": [10.0, 40.0], "g": [20.0, 50.0], "b": [30.0, 60.0]})
    gaia = Table({"ra": [sky[0].ra.deg], "dec": [sky[0].dec.deg], "bp_rp": [0.75]})

    matched = pcc.cross_match(stars, gaia, w, tol_arcsec=5.0)

    # Only one Gaia source exists, so only one row can survive dedup -- and it
    # must be the closer star (star0), not the farther one (star1).
    assert len(matched) == 1
    row = matched[0]
    assert np.isclose(row["x"], 50.0)
    assert np.isclose(row["y"], 50.0)
    assert np.isclose(row["r"], 10.0)
    assert np.isclose(row["g"], 20.0)
    assert np.isclose(row["b"], 30.0)
    assert row["sep_arcsec"] < 1.0  # much closer than the discarded ~3.6" star


def test_cross_match_columns_aligned_per_row():
    w = _toy_wcs()
    xs, ys = np.array([50.0, 60.0, 30.0]), np.array([50.0, 40.0, 70.0])
    sky = w.pixel_to_world(xs, ys)
    stars = Table({"x": xs, "y": ys,
                   "r": [11.0, 22.0, 33.0], "g": [111.0, 222.0, 333.0], "b": [1.1, 2.2, 3.3]})
    gaia = Table({
        "ra": [sky[0].ra.deg, sky[1].ra.deg],
        "dec": [sky[0].dec.deg, sky[1].dec.deg],
        "bp_rp": [0.5, 1.5],
    })

    matched = pcc.cross_match(stars, gaia, w, tol_arcsec=5.0)

    # Dedup sorts by sep_arcsec, so row order need not match input order --
    # look rows up by x to confirm r/g/b/bp_rp stay aligned to the right star.
    assert len(matched) == 2
    by_x = {round(float(row["x"]), 3): row for row in matched}
    row0 = by_x[50.0]
    assert (float(row0["r"]), float(row0["g"]), float(row0["b"])) == (11.0, 111.0, 1.1)
    assert np.isclose(row0["bp_rp"], 0.5)
    row1 = by_x[60.0]
    assert (float(row1["r"]), float(row1["g"]), float(row1["b"])) == (22.0, 222.0, 2.2)
    assert np.isclose(row1["bp_rp"], 1.5)


def _inject_star(img, x, y, amp, fwhm=3.0):
    H, W, _ = img.shape
    sig = fwhm / 2.355
    yy, xx = np.mgrid[0:H, 0:W]
    g2d = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
    for c in range(3):
        img[..., c] += amp[c] * g2d


def test_detect_stars_finds_injected_and_measures_color():
    rng = np.random.RandomState(2)
    img = rng.normal(1000.0, 5.0, (120, 120, 3))       # flat noisy background
    # three stars with distinct colors (r:g:b amplitudes)
    _inject_star(img, 30, 40, (2000, 1000, 500))
    _inject_star(img, 80, 70, (800, 1000, 1200))
    _inject_star(img, 60, 20, (1500, 1500, 1500))

    stars = pcc.detect_stars(img, fwhm=3.0, threshold_sigma=5.0)
    assert len(stars) >= 3
    # locate the star nearest (30,40) and check its r/g > 1 (red star)
    d = (stars["x"] - 30) ** 2 + (stars["y"] - 40) ** 2
    red = stars[int(np.argmin(d))]
    assert red["r"] / red["g"] > 1.3
    assert red["b"] / red["g"] < 0.9


def test_detect_stars_rejects_single_channel_saturation():
    rng = np.random.RandomState(3)
    img = rng.normal(1000.0, 5.0, (120, 120, 3))
    # Saturated in R ONLY: R pixel value (~66000) sits well above `sat`
    # (60000), while G/B stay low. Luminance (mean-of-RGB) peak is only
    # ~(66000+1500+1500)/3 ~= 23000, i.e. it would NOT be caught by a
    # luminance-based saturation check -- this is the bug under test.
    _inject_star(img, 40, 40, (65000, 500, 500))
    # A normal, unsaturated star elsewhere that must still be detected.
    _inject_star(img, 80, 80, (1500, 1500, 1500))

    stars = pcc.detect_stars(img, fwhm=3.0, threshold_sigma=5.0)

    d_sat = (stars["x"] - 40) ** 2 + (stars["y"] - 40) ** 2
    assert np.all(d_sat > 25)  # no surviving detection near the saturated star

    d_ok = (stars["x"] - 80) ** 2 + (stars["y"] - 80) ** 2
    assert np.min(d_ok) < 4  # the normal star still survives


def test_detect_stars_rejects_near_edge():
    rng = np.random.RandomState(4)
    img = rng.normal(1000.0, 5.0, (120, 120, 3))
    # Within `edge` (default 20) pixels of the left border -- must be rejected.
    _inject_star(img, 10, 60, (2000, 1500, 1000))
    # A normal star far from any edge that must still be detected.
    _inject_star(img, 60, 60, (2000, 1500, 1000))

    stars = pcc.detect_stars(img, fwhm=3.0, threshold_sigma=5.0)

    d_edge = (stars["x"] - 10) ** 2 + (stars["y"] - 60) ** 2
    assert np.all(d_edge > 25)  # no surviving detection near the edge star

    d_ok = (stars["x"] - 60) ** 2 + (stars["y"] - 60) ** 2
    assert np.min(d_ok) < 4  # the normal star still survives
