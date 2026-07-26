import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stages.image import Image, Space
from stages.io import save_fits, load_fits, crop_header


def _wcs_header():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 100.0, 120.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_save_load_roundtrips_wcs_and_space(tmp_path):
    img = Image(np.random.rand(64, 48, 3).astype(np.float32) * 1000,
                Space.LINEAR_ADU, _wcs_header())
    p = str(tmp_path / "x.fit")
    save_fits(p, img)
    back = load_fits(p)
    assert back.space is Space.LINEAR_ADU          # from PIPESPCE
    assert back.pixels.shape == (64, 48, 3)
    assert back.wcs is not None and back.wcs.has_celestial


def test_nonlinear_not_rescaled(tmp_path):
    img = Image(np.full((8, 8, 3), 0.5, np.float32), Space.NONLINEAR)
    p = str(tmp_path / "n.fit")
    save_fits(p, img)
    back = load_fits(p)
    assert back.space is Space.NONLINEAR
    assert np.allclose(back.pixels, 0.5)           # NOT scaled by 65535


def test_crop_header_shifts_crpix_consistently():
    h = _wcs_header()
    m = 40
    h2 = crop_header(h, m)
    w1, w2 = WCS(h, naxis=2).celestial, WCS(h2, naxis=2).celestial
    # a sky point at old pixel (100,120) sits at new pixel (100-m,120-m)
    sky = w1.pixel_to_world(99, 119)               # 0-based -> CRPIX ref
    x2, y2 = w2.world_to_pixel(sky)
    assert abs(x2 - (99 - m)) < 1e-6 and abs(y2 - (119 - m)) < 1e-6
