#!/usr/bin/env python3
"""Empirical Gaia DR3 photometric color calibration for OSC astro images.

Measures star colors on the original (WCS-bearing) stacked FITS, regresses them
against Gaia colors, and solves for R/B channel gains relative to G that put a
chosen white point (default solar) at neutral. See
docs/superpowers/specs/2026-07-23-photometric-color-calibration-design.md.
"""
import numpy as np


class PCCError(Exception):
    """Raised for any condition that should trigger the gentle-WB fallback."""


def apply_gains(img_adu, gains, pedestal):
    """Scale each channel around the neutral pedestal: out = (img-ped)*gain + ped."""
    out = np.asarray(img_adu, dtype=np.float64).copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    return out


def _robust_linfit(x, y, sigma=3.0, iters=3):
    """Sigma-clipped least-squares line y ~ slope*x + intercept."""
    m = np.isfinite(x) & np.isfinite(y)
    xf, yf = np.asarray(x)[m], np.asarray(y)[m]
    slope = intercept = 0.0
    for _ in range(iters):
        if len(xf) < 3:
            break
        A = np.vstack([xf, np.ones_like(xf)]).T
        slope, intercept = np.linalg.lstsq(A, yf, rcond=None)[0]
        resid = yf - (slope * xf + intercept)
        s = resid.std()
        if s == 0:
            break
        keep = np.abs(resid) < sigma * s
        xf, yf = xf[keep], yf[keep]
    return slope, intercept, len(xf)


def solve_gains(matched, ref_bp_rp=0.82, sigma=3.0):
    """Regress instrumental color ratios vs Gaia bp_rp; solve gains at the white point."""
    r = np.asarray(matched["r"], float)
    g = np.asarray(matched["g"], float)
    b = np.asarray(matched["b"], float)
    bp_rp = np.asarray(matched["bp_rp"], float)
    good = (g > 0) & (r > 0) & (b > 0)
    r, g, b, bp_rp = r[good], g[good], b[good], bp_rp[good]
    cr, cb = r / g, b / g

    slope_r, int_r, n = _robust_linfit(bp_rp, cr, sigma)
    slope_b, int_b, _ = _robust_linfit(bp_rp, cb, sigma)
    cr0 = slope_r * ref_bp_rp + int_r
    cb0 = slope_b * ref_bp_rp + int_b
    if not (cr0 > 0 and cb0 > 0):
        raise PCCError("degenerate color fit (non-positive ratio at white point)")

    gains = (1.0 / cr0, 1.0, 1.0 / cb0)
    diag = dict(n=n, slope_r=slope_r, int_r=int_r, slope_b=slope_b, int_b=int_b,
                cr=cr, cb=cb, bp_rp=bp_rp, ref_bp_rp=ref_bp_rp)
    return gains, diag
