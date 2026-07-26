import numpy as np
import astrolib as al
from stages.image import Image, Space
from stages.background import BackgroundExtractStage
from stages.stretch import StretchStage


def test_background_stage_matches_astrolib_per_channel():
    rng = np.random.RandomState(0)
    px = (1000 + 200 * rng.rand(40, 50, 3)).astype(np.float32)
    img = Image(px, Space.LINEAR_ADU)
    out = BackgroundExtractStage().run({"image": img},
        {"degree": 3, "sample": 4, "pedestal": 0.10})["image"]
    ped = 0.10 * 65535.0
    exp = np.empty_like(px)
    for c in range(3):
        model, _ = al.background_model(px[..., c], 3, 4)
        exp[..., c] = px[..., c] - model + ped
    assert np.allclose(out.pixels, exp)
    assert out.space is Space.LINEAR_ADU


def test_stretch_stage_matches_astrolib_and_flips_space():
    rng = np.random.RandomState(1)
    px = (3000 + 500 * rng.rand(32, 32, 3)).astype(np.float32)
    img = Image(px, Space.LINEAR_ADU)
    out = StretchStage().run({"image": img}, {"target_bg": 0.18, "shadows_clip": -1.8})["image"]
    exp = al.linked_stretch(np.clip(px / 65535.0, 0, 1), 0.18, -1.8)
    assert np.allclose(out.pixels, exp)
    assert out.space is Space.NONLINEAR
