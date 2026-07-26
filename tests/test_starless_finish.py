import importlib.util
import os
import numpy as np
import pytest
from astropy.io import fits
import starnet
import astrolib as al
from stages.io import save_fits
from stages.image import Image, Space

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "scripts", "05b_starless_finish.py")


def _load():
    spec = importlib.util.spec_from_file_location("starless_finish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_recombines_and_writes_five_files(tmp_path, monkeypatch):
    base = np.clip(np.random.RandomState(0).rand(512, 512, 3), 0, 1).astype(np.float32)
    infile = str(tmp_path / "in.fit")
    save_fits(infile, Image(base, Space.NONLINEAR, fits.Header()))
    starless = base * 0.5
    stars = np.zeros_like(base); stars[100, 100] = [1.0, 1.0, 1.0]
    monkeypatch.setattr(starnet, "remove_stars", lambda px, **k: (starless.copy(), stars.copy()))

    mod = _load()
    out_base = str(tmp_path / "out")
    result = mod.main(infile, out_base)

    # expected: masked_denoise -> saturation-only finish, then screen stars back in
    proc = al.finish(al.masked_denoise(starless, bg_luma=mod.BG_LUMA_DENOISE,
                                       bg_chroma=mod.BG_CHROMA_DENOISE, gal_luma=mod.GAL_LUMA_DENOISE,
                                       gal_chroma=mod.GAL_CHROMA_DENOISE, feather=mod.MASK_FEATHER),
                     saturation=mod.SATURATION, luma_denoise=0, chroma_denoise=0, scnr=False)
    expected = al.screen(proc, stars)
    assert np.allclose(result, expected, atol=1e-6)
    for suffix in (".tif", ".png", "_preview.png", "_starless.png", "_starlayer.png"):
        assert os.path.exists(out_base + suffix), suffix
