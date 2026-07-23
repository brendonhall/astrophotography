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


def detect_stars(img_adu, fwhm=3.0, threshold_sigma=5.0, aperture_r=4.0,
                 annulus=(6.0, 9.0), sat=60000.0, edge=20, min_flux=1.0):
    """Detect stars on luminance and measure per-channel aperture flux."""
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from photutils.detection import DAOStarFinder
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry)

    img = np.asarray(img_adu, dtype=np.float64)
    lum = img.mean(axis=2)
    _, med, std = sigma_clipped_stats(lum, sigma=3.0)

    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(lum - med)
    if sources is None or len(sources) == 0:
        raise PCCError("no stars detected")

    H, W = lum.shape
    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])

    # Reject on per-channel raw ADU, not the luminance (mean-of-RGB) peak: a
    # star saturated in a single channel (e.g. R=70000, G=B=500) has a
    # luminance peak that looks unsaturated, but its clipped channel flux
    # would still corrupt the color regression.
    chan_max = img.max(axis=2)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    win = 1  # sample a small window around the centroid, not just one pixel
    sat_peak = np.array([
        chan_max[max(0, yc - win):yc + win + 1, max(0, xc - win):xc + win + 1].max()
        for xc, yc in zip(xi, yi)
    ])

    keep = ((x > edge) & (x < W - edge) & (y > edge) & (y < H - edge)
            & (sat_peak < sat))
    x, y = x[keep], y[keep]
    if len(x) == 0:
        raise PCCError("no stars survive edge/saturation cuts")

    positions = np.column_stack([x, y])
    ap = CircularAperture(positions, r=aperture_r)
    ann = CircularAnnulus(positions, r_in=annulus[0], r_out=annulus[1])
    flux = {}
    for c, name in enumerate("rgb"):
        chan = img[..., c]
        src = aperture_photometry(chan, ap)["aperture_sum"]
        bkg = aperture_photometry(chan, ann)["aperture_sum"] / ann.area
        flux[name] = np.asarray(src) - np.asarray(bkg) * ap.area

    tab = Table({"x": x, "y": y, "r": flux["r"], "g": flux["g"], "b": flux["b"]})
    good = (tab["r"] > min_flux) & (tab["g"] > min_flux) & (tab["b"] > min_flux)
    tab = tab[good]
    if len(tab) == 0:
        raise PCCError("no stars with valid positive photometry")
    return tab


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


def query_gaia(hdr, radius_deg=2.3, mag_limit=16.0, row_limit=50000):
    """Cone-search Gaia DR3 around the header RA/Dec. Raises PCCError on failure."""
    if "RA" not in hdr or "DEC" not in hdr:
        raise PCCError("header missing RA/DEC for Gaia query")
    ra, dec = float(hdr["RA"]), float(hdr["DEC"])
    try:
        from astroquery.gaia import Gaia
    except Exception as e:
        raise PCCError(f"astroquery unavailable: {e}")

    adql = (
        f"SELECT TOP {row_limit} ra, dec, phot_g_mean_mag, bp_rp "
        f"FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
        f"AND phot_g_mean_mag < {mag_limit} "
        f"AND bp_rp IS NOT NULL "
        f"AND astrometric_params_solved > 3"
    )
    try:
        job = Gaia.launch_job_async(adql)
        tab = job.get_results()
    except Exception as e:
        raise PCCError(f"Gaia query failed: {e}")
    if len(tab) == 0:
        raise PCCError("Gaia query returned no rows")
    return tab["ra", "dec", "phot_g_mean_mag", "bp_rp"]
