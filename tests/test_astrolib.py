import numpy as np
import astrolib as al


def test_screen_identity_with_black():
    a = np.linspace(0, 1, 25).reshape(5, 5)
    black = np.zeros_like(a)
    # screening with black is a no-op
    assert np.allclose(al.screen(a, black), a)


def test_screen_commutative_and_bounded():
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 1, (8, 8, 3))
    b = rng.uniform(0, 1, (8, 8, 3))
    s = al.screen(a, b)
    assert np.allclose(s, al.screen(b, a))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_screen_reconstructs_unscreen_split():
    # If stars is the unscreen inverse of starless, screen() rebuilds the source.
    rng = np.random.RandomState(1)
    source = rng.uniform(0, 1, (6, 6, 3))
    starless = source * 0.5
    # unscreen star layer: stars = 1 - (1-source)/(1-starless)
    stars = 1.0 - (1.0 - source) / (1.0 - starless)
    assert np.allclose(al.screen(starless, stars), source, atol=1e-6)


def test_unsharp_luma_amount_zero_is_copy():
    rng = np.random.RandomState(2)
    img = rng.uniform(0, 1, (10, 10, 3))
    out = al.unsharp_luma(img, amount=0.0)
    assert np.allclose(out, np.clip(img, 0, 1))
    assert out is not img


def test_unsharp_luma_flat_field_unchanged():
    img = np.full((16, 16, 3), 0.4)
    out = al.unsharp_luma(img, amount=1.0, radius=2.0)
    assert np.allclose(out, img, atol=1e-6)


def test_unsharp_luma_increases_edge_contrast_preserving_gray():
    img = np.zeros((16, 16, 3))
    img[:, 8:] = 0.6  # a gray step edge
    out = al.unsharp_luma(img, amount=1.0, radius=1.5)
    # gray stays gray (no hue shift): channels equal everywhere
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])
    # overshoot at the edge => new max exceeds the original plateau
    assert out[..., 0].max() > 0.6 + 1e-3
