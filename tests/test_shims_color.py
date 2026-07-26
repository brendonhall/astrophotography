import importlib.util
import os
import numpy as np
import pcc
from astropy.io import fits
from stages.io import load_fits, save_fits
from stages.image import Image, Space

SC = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location("color_shim", os.path.join(SC, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hdr(crval1=210.0):
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 50.0, 50.0
    h["CRVAL1"], h["CRVAL2"] = crval1, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_color_shim_reads_wcs_from_input_and_writes(tmp_path, monkeypatch):
    h = _hdr()
    src = str(tmp_path / "bg.fit")
    save_fits(src, Image((5000 + np.random.rand(60, 60, 3) * 300).astype(np.float32),
                         Space.LINEAR_ADU, h))
    seen = {}
    def fake_pcc(px, hdr, **k):
        seen["has_wcs"] = ("CRPIX1" in hdr)
        return ((1.05, 1.0, 0.95), {"n_matched": 40})
    monkeypatch.setattr(pcc, "photometric_calibration", fake_pcc)
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, p: None)
    out = str(tmp_path / "color.fit")
    _load("03_color.py").main(src, out)
    assert seen["has_wcs"] is True                 # WCS came from the input, no --original
    assert load_fits(out).space is Space.LINEAR_ADU


def test_color_shim_original_routes_to_reference_port(tmp_path, monkeypatch):
    # Input and reference are two DIFFERENT frames, distinguished by CRVAL1.
    src = str(tmp_path / "bg.fit")
    save_fits(src, Image((5000 + np.random.rand(60, 60, 3) * 300).astype(np.float32),
                         Space.LINEAR_ADU, _hdr(crval1=210.0)))
    ref = str(tmp_path / "original.fit")
    save_fits(ref, Image((5000 + np.random.rand(60, 60, 3) * 300).astype(np.float32),
                         Space.LINEAR_ADU, _hdr(crval1=999.0)))

    seen = {}
    def fake_pcc(px, hdr, **k):
        seen["crval1"] = float(hdr["CRVAL1"])
        return ((1.05, 1.0, 0.95), {"n_matched": 40})
    monkeypatch.setattr(pcc, "photometric_calibration", fake_pcc)
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, p: None)

    out = str(tmp_path / "color.fit")
    _load("03_color.py").main(src, out, original=ref)

    # PCC must have measured on the REFERENCE frame's header, not the input's.
    assert seen["crval1"] == 999.0
