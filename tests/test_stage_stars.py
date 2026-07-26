import numpy as np
import pytest
import astrolib as al
import starnet
from stages.image import Image, Space
from stages.base import StageError
from stages.stars import RemoveStarsStage, ScreenRecombineStage


def test_remove_stars_maps_ports(monkeypatch):
    img = Image(np.clip(np.random.rand(512, 512, 3), 0, 1).astype(np.float32), Space.NONLINEAR)
    sl = img.pixels * 0.5
    st = np.zeros_like(img.pixels)
    monkeypatch.setattr(starnet, "remove_stars", lambda px, **k: (sl, st))
    out = RemoveStarsStage().run({"image": img}, {"stride": 128})
    assert np.allclose(out["starless"].pixels, sl)
    assert np.allclose(out["stars"].pixels, st)
    assert out["starless"].space is Space.NONLINEAR


def test_remove_stars_min_size_precondition():
    small = Image(np.zeros((100, 100, 3), np.float32), Space.NONLINEAR)
    with pytest.raises(StageError):
        RemoveStarsStage().run({"image": small}, {})


def test_screen_recombine_matches_astrolib():
    rng = np.random.RandomState(0)
    base = Image(np.clip(rng.rand(8, 8, 3), 0, 1), Space.NONLINEAR)
    overlay = Image(np.clip(rng.rand(8, 8, 3), 0, 1), Space.NONLINEAR)
    out = ScreenRecombineStage().run({"base": base, "overlay": overlay})["image"]
    assert np.allclose(out.pixels, al.screen(base.pixels, overlay.pixels))
