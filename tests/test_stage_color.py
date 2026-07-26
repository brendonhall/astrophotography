import numpy as np
import pcc
from astropy.io import fits
from stages.image import Image, Space
from stages.color import ColorCalibrateStage, _gentle_white_balance

_PED = 0.10 * 65535.0


def _linear(seed=0):
    rng = np.random.RandomState(seed)
    return Image((5000 + 400 * rng.rand(30, 30, 3)).astype(np.float64),
                 Space.LINEAR_ADU, fits.Header())


def test_no_pcc_uses_gentle_white_balance():
    img = _linear()
    out = ColorCalibrateStage().run({"image": img}, {"no_pcc": True})["image"]
    # neutralize then gentle-WB, computed directly
    from stages.color import _neutralize
    exp = _gentle_white_balance(_neutralize(img.pixels, _PED), _PED)
    assert np.allclose(out.pixels, exp)


def test_pcc_success_applies_gains(monkeypatch):
    img = _linear(1)
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.1, 1.0, 0.9), {"n_matched": 42}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda report, path: None)
    out = ColorCalibrateStage().run({"image": img}, {})["image"]
    from stages.color import _neutralize
    exp = pcc.apply_gains(_neutralize(img.pixels, _PED), (1.1, 1.0, 0.9), _PED)
    assert np.allclose(out.pixels, exp)


def test_pcc_error_falls_back(monkeypatch):
    img = _linear(2)
    def boom(px, hdr, **k):
        raise pcc.PCCError("no stars")
    monkeypatch.setattr(pcc, "photometric_calibration", boom)
    out = ColorCalibrateStage().run({"image": img}, {})["image"]
    from stages.color import _neutralize
    exp = _gentle_white_balance(_neutralize(img.pixels, _PED), _PED)
    assert np.allclose(out.pixels, exp)
