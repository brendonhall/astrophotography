import numpy as np
from astropy.io import fits
from flow.graph import Node
from flow import cache as c
from stages.image import Image, Space


def test_recipe_hash_stable_and_sensitive():
    n = Node("x", "crop", {"margin": 40})
    h1 = c.recipe_hash(n, {"image": "up1"})
    assert h1 == c.recipe_hash(Node("x", "crop", {"margin": 40}), {"image": "up1"})  # stable
    assert h1 != c.recipe_hash(Node("x", "crop", {"margin": 41}), {"image": "up1"})  # param change
    assert h1 != c.recipe_hash(n, {"image": "up2"})                                  # upstream change


def test_store_load_roundtrip(tmp_path):
    img = Image(np.random.rand(6, 6, 3).astype(np.float32), Space.NONLINEAR, fits.Header())
    p = c.cache_path(str(tmp_path), "abc123", "image")
    c.store_cached(p, img)
    back = c.load_cached(p)
    assert back.space is Space.NONLINEAR and back.pixels.shape == (6, 6, 3)
    assert np.allclose(back.pixels, img.pixels)
