import importlib.util
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stages.io import load_fits, save_fits
from stages.image import Image, Space

SC = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(SC, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hdr():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 200.0, 200.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_crop_shim_preserves_wcs(tmp_path):
    src = str(tmp_path / "in.fit")
    save_fits(src, Image((1000 + np.random.rand(600, 600, 3) * 100).astype(np.float32),
                         Space.LINEAR_ADU, _hdr()))
    out = str(tmp_path / "crop.fit")
    _load("01_crop.py").main(src, out)
    back = load_fits(out)
    assert back.pixels.shape[0] < 600 and back.wcs is not None and back.wcs.has_celestial


def test_background_then_stretch_shims(tmp_path):
    src = str(tmp_path / "in.fit")
    save_fits(src, Image((1000 + np.random.rand(200, 200, 3) * 100).astype(np.float32),
                         Space.LINEAR_ADU, _hdr()))
    bg = str(tmp_path / "bg.fit")
    _load("02_background.py").main(src, bg)
    assert load_fits(bg).space is Space.LINEAR_ADU
    st = str(tmp_path / "st.fit")
    _load("04_stretch.py").main(bg, st)
    out = load_fits(st)
    assert out.space is Space.NONLINEAR and out.pixels.max() <= 1.0 + 1e-6


def test_finish_shim_writes_outputs(tmp_path):
    st = str(tmp_path / "st.fit")
    save_fits(st, Image(np.clip(np.random.rand(64, 64, 3), 0, 1).astype(np.float32),
                        Space.NONLINEAR, _hdr()))
    base = str(tmp_path / "final")
    _load("05_finish.py").main(st, base)
    for s in (".tif", ".png", "_preview.png"):
        assert os.path.exists(base + s)
