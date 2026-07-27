import numpy as np
import pytest
from astropy.io import fits
import stages
from stages.base import StageError
from stages.image import Image, Space
from stages.geometry import CropStage


def _hdr():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 100.0, 120.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_crop_trims_and_shifts_wcs():
    img = Image(np.arange(20 * 24 * 3, dtype=np.float32).reshape(20, 24, 3),
                Space.LINEAR_ADU, _hdr())
    out = CropStage().run({"image": img}, {"margin": 5})["image"]
    assert out.pixels.shape == (10, 14, 3)
    assert out.header["CRPIX1"] == 95.0 and out.header["CRPIX2"] == 115.0


def test_crop_registered():
    assert stages.get("crop") is CropStage
    assert any(s["id"] == "crop" for s in stages.list_stages())


def test_crop_margin_too_large_raises():
    img = Image(np.arange(20 * 24 * 3, dtype=np.float32).reshape(20, 24, 3),
                Space.LINEAR_ADU, _hdr())
    with pytest.raises(StageError):
        CropStage().run({"image": img}, {"margin": 15})
