"""Color-calibration stage: background neutralize + Gaia PCC (gentle-WB fallback)."""
from __future__ import annotations
import numpy as np
import astrolib as al
import pcc
from .base import Stage, Param, Port
from .image import Space
from .registry import register

_PEDESTAL = 0.10 * 65535.0


def _bg_level(chan):
    bg = chan[~al.source_mask(chan)]
    med = np.median(bg)
    for _ in range(3):
        s = bg.std()
        bg = bg[np.abs(bg - med) < 3 * s]
        med = np.median(bg)
    return med


def _neutralize(pixels, pedestal):
    out = np.empty_like(pixels)
    for c in range(3):
        out[..., c] = pixels[..., c] - _bg_level(pixels[..., c]) + pedestal
    return out


def _gentle_white_balance(img, pedestal, clamp=(0.85, 1.15)):
    lum = img.mean(axis=2)
    lo, hi = np.percentile(lum, 60), np.percentile(lum, 99)
    band = (lum > lo) & (lum < hi)
    means = np.array([img[..., c][band].mean() - pedestal for c in range(3)])
    gains = np.clip(means[1] / means, *clamp)
    out = img.copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    return out


@register
class ColorCalibrateStage(Stage):
    id = "color_calibrate"
    label = "Color calibration (PCC)"
    description = ("Neutralize background, then Gaia photometric color calibration; "
                   "falls back to gentle white balance.")
    INPUTS = [Port("image", space=Space.LINEAR_ADU),
              Port("reference", space=Space.LINEAR_ADU, required=False,
                   help="optional WCS-bearing frame to measure PCC on (default: 'image')")]
    OUTPUTS = [Port("image", space=Space.LINEAR_ADU)]
    PARAMS = [
        Param("ref_bp_rp", "float", 0.82, "White point (BP-RP)", min=-1, max=3, step=0.01),
        Param("min_stars", "int", 30, "Min matched stars", min=3, max=1000, step=1),
        Param("no_pcc", "bool", False, "Skip PCC (gentle WB only)"),
        Param("diagnostic_path", "str", "", "PCC diagnostic PNG path"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        neutral = _neutralize(img.pixels, _PEDESTAL)
        ref = inputs.get("reference") or img
        if params["no_pcc"] or ref.header is None:
            return {"image": img.replace(pixels=_gentle_white_balance(neutral, _PEDESTAL))}
        try:
            gains, report = pcc.photometric_calibration(
                ref.pixels, ref.header, ref_bp_rp=params["ref_bp_rp"],
                min_stars=params["min_stars"])
            if params["diagnostic_path"]:
                pcc.save_diagnostic(report, params["diagnostic_path"])
            out = pcc.apply_gains(neutral, gains, _PEDESTAL)
        except Exception:
            out = _gentle_white_balance(neutral, _PEDESTAL)
        return {"image": img.replace(pixels=np.asarray(out, dtype=img.pixels.dtype))}
