import numpy as np
import astrolib as al


def test_background_model_recovers_smooth_gradient():
    h, w = 80, 100
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx / (w - 1)) * 2 - 1
    yn = (yy / (h - 1)) * 2 - 1
    truth = 500 + 120 * xn + 80 * yn + 30 * xn * yn   # smooth low-order background
    model, frac = al.background_model(truth.astype(np.float32), degree=3, sample=4)
    assert np.allclose(model, truth, atol=1.0)        # fit recovers the gradient
    assert 0.0 <= frac <= 1.0


def test_linked_stretch_matches_reference_and_lifts_median():
    rng = np.random.RandomState(0)
    img01 = np.clip(0.05 + 0.02 * rng.rand(40, 40, 3), 0, 1).astype(np.float64)
    # reference formula (verbatim from the old step 04)
    med = np.median(img01); mad = np.median(np.abs(img01 - med)) * 1.4826
    black = np.clip(med + (-1.8) * mad, 0.0, 1.0)
    scaled = np.clip((img01 - black) / (1.0 - black), 0.0, 1.0)
    m_shift = (med - black) / (1.0 - black)
    ref = al._mtf(scaled, al._midtone(m_shift, 0.18))
    got = al.linked_stretch(img01, target_bg=0.18, shadows_clip=-1.8)
    assert np.allclose(got, ref)
    assert got.min() >= 0.0 and got.max() <= 1.0
