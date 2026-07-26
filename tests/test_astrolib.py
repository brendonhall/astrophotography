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


def test_masked_denoise_preserves_shape_and_range():
    rng = np.random.RandomState(3)
    img = rng.uniform(0, 1, (64, 64, 3))
    out = al.masked_denoise(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_masked_denoise_smooths_background_more_than_galaxy():
    from scipy import ndimage
    rng = np.random.RandomState(4)
    size = 200
    img = 0.1 + rng.normal(0, 0.03, (size, size, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    blob = 0.6 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 30.0 ** 2)))
    img = img + blob[..., None]
    img = np.clip(img, 0.0, 1.0)

    strong = al.finish(img, saturation=1.0, luma_denoise=0.06, chroma_denoise=10.0, scnr=False)
    masked = al.masked_denoise(img)

    def hf(a):
        return np.std(a - ndimage.gaussian_filter(a, 3))

    def bg_std(a):
        corner = a[:40, :40].mean(axis=2)
        return np.std(corner)

    # galaxy region crop around the blob center
    gal_slice = (slice(cy - 40, cy + 40), slice(cx - 40, cx + 40))
    masked_gal = masked[gal_slice].mean(axis=2)
    strong_gal = strong[gal_slice].mean(axis=2)

    assert hf(masked_gal) > hf(strong_gal)
    assert bg_std(masked) <= bg_std(strong) * 1.5
    assert bg_std(masked) < 0.6 * bg_std(img)
