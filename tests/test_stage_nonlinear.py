import numpy as np
import pytest
import astrolib as al
from stages.image import Image, Space
from stages.base import StageError
from stages.finish import FinishStage, SaturateStage
from stages.denoise import MaskedDenoiseStage, UnsharpLumaStage


def _nl(seed=0):
    rng = np.random.RandomState(seed)
    return Image(np.clip(rng.rand(40, 40, 3), 0, 1).astype(np.float64), Space.NONLINEAR)


def test_finish_stage_matches_astrolib():
    img = _nl()
    out = FinishStage().run({"image": img},
        {"saturation": 1.20, "luma_denoise": 0.012, "chroma_denoise": 4.0, "scnr": True})["image"]
    exp = al.finish(img.pixels, saturation=1.20, luma_denoise=0.012, chroma_denoise=4.0, scnr=True)
    assert np.allclose(out.pixels, exp)


def test_saturate_is_saturation_only():
    img = _nl(2)
    out = SaturateStage().run({"image": img}, {"saturation": 1.3})["image"]
    exp = al.finish(img.pixels, saturation=1.3, luma_denoise=0, chroma_denoise=0, scnr=False)
    assert np.allclose(out.pixels, exp)


def test_masked_denoise_matches_and_requires_nonlinear():
    img = _nl(3)
    out = MaskedDenoiseStage().run({"image": img},
        {"bg_luma": 0.06, "bg_chroma": 10.0, "gal_luma": 0.010, "gal_chroma": 3.0, "feather": 25.0})["image"]
    exp = al.masked_denoise(img.pixels, bg_luma=0.06, bg_chroma=10.0, gal_luma=0.010,
                            gal_chroma=3.0, feather=25.0)
    assert np.allclose(out.pixels, exp)
    linear = Image(img.pixels, Space.LINEAR_ADU)
    with pytest.raises(StageError):
        MaskedDenoiseStage().run({"image": linear}, {})


def test_unsharp_matches_astrolib():
    img = _nl(4)
    out = UnsharpLumaStage().run({"image": img}, {"amount": 0.5, "radius": 2.0})["image"]
    exp = al.unsharp_luma(img.pixels, amount=0.5, radius=2.0)
    assert np.allclose(out.pixels, exp)
