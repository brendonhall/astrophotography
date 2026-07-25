import importlib.util
import os
import numpy as np
import pytest
from astropy.io import fits
import starnet
import astrolib as al

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "scripts", "05b_starless_finish.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("starless_finish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_recombines_and_writes_outputs(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    base = rng.uniform(0, 1, (512, 512, 3)).astype(np.float32)  # [0,1] stretched
    infile = tmp_path / "in.fit"
    fits.PrimaryHDU(data=np.moveaxis(base, -1, 0)).writeto(infile, overwrite=True)

    starless = base * 0.5
    stars = np.zeros_like(base)
    stars[100, 100] = [1.0, 1.0, 1.0]
    monkeypatch.setattr(starnet, "remove_stars",
                        lambda img01, **kw: (starless.copy(), stars.copy()))

    mod = _load_module()
    out_base = str(tmp_path / "out")
    result = mod.main(str(infile), out_base)

    expected = al.screen(mod.process_starless(starless), stars)
    assert np.allclose(result, expected, atol=1e-6)
    for suffix in (".tif", ".png", "_preview.png", "_starless.png", "_starlayer.png"):
        assert os.path.exists(out_base + suffix), suffix


def test_no_sharpen_skips_unsharp(tmp_path):
    mod = _load_module()
    starless = np.full((32, 32, 3), 0.3)
    # with no sharpen, process_starless == finish(scnr off) of the input
    got = mod.process_starless(starless, sharpen=False)
    exp = al.finish(starless, saturation=mod.SATURATION,
                    luma_denoise=mod.LUMA_DENOISE,
                    chroma_denoise=mod.CHROMA_DENOISE, scnr=False)
    assert np.allclose(got, exp)
