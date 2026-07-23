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


def cross_match(stars, gaia, wcs, tol_arcsec=5.0):
    """Match detected stars to Gaia by sky position; dedupe (closest wins)."""
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    star_sky = wcs.pixel_to_world(np.asarray(stars["x"]), np.asarray(stars["y"]))
    gaia_sky = SkyCoord(np.asarray(gaia["ra"]) * u.deg, np.asarray(gaia["dec"]) * u.deg)

    idx, sep2d, _ = star_sky.match_to_catalog_sky(gaia_sky)
    sep = sep2d.arcsec
    keep = sep < tol_arcsec

    rows = Table({
        "x": np.asarray(stars["x"])[keep],
        "y": np.asarray(stars["y"])[keep],
        "r": np.asarray(stars["r"])[keep],
        "g": np.asarray(stars["g"])[keep],
        "b": np.asarray(stars["b"])[keep],
        "bp_rp": np.asarray(gaia["bp_rp"])[idx[keep]],
        "gaia_idx": idx[keep],
        "sep_arcsec": sep[keep],
    })
    # dedupe: keep the closest star per Gaia source
    rows.sort("sep_arcsec")
    _, first = np.unique(rows["gaia_idx"], return_index=True)
    rows = rows[np.sort(first)]
    rows.remove_column("gaia_idx")
    return rows


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
