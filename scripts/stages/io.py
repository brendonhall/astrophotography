"""FITS IO for stage payloads: preserves the FULL header (WCS) + a space tag."""
from __future__ import annotations
import numpy as np
from astropy.io import fits
from .image import Image, Space


def save_fits(path, img: Image):
    arr = np.moveaxis(np.asarray(img.pixels, dtype=np.float32), -1, 0)  # (H,W,3)->(3,H,W)
    hdr = img.header.copy() if img.header is not None else fits.Header()
    hdr["PIPESPCE"] = (str(img.space), "pipeline color space")
    fits.PrimaryHDU(data=arr, header=hdr).writeto(path, overwrite=True)


def load_fits(path, space: Space | None = None) -> Image:
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W)->(H,W,3)
    sp = space or Space(hdr.get("PIPESPCE", Space.LINEAR_ADU.value))
    if sp is Space.LINEAR_ADU and data.size and float(np.nanmax(data)) <= 1.5:
        data = data * 65535.0            # Siril [0,1] master -> ADU (matches astrolib.load)
    return Image(data, sp, hdr)


def crop_header(hdr, margin):
    """Shift CRPIX for a margin trimmed off top/left. SIP coeffs are CRPIX-relative
    so they follow; for higher-order distortion see astropy.wcs.WCS.slice."""
    if hdr is None:
        return None
    h = hdr.copy()
    for k in ("CRPIX1", "CRPIX2"):
        if k in h:
            h[k] = float(h[k]) - margin
    return h
