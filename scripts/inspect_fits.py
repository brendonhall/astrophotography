#!/usr/bin/env python3
"""Inspect a FITS file: print key header cards, data geometry, and per-channel stats.

Usage: python inspect_fits.py <path-to.fit>
"""
import sys
import numpy as np
from astropy.io import fits


def main(path):
    with fits.open(path) as hdul:
        hdul.info()
        hdu = hdul[0]
        hdr = hdu.header
        data = hdu.data

    print("\n=== Selected header cards ===")
    keys = [
        "OBJECT", "DATE-OBS", "EXPTIME", "EXPOSURE", "STACKCNT", "LIVETIME",
        "FILTER", "GAIN", "CCD-TEMP", "BAYERPAT", "XBAYROFF", "YBAYROFF",
        "INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ", "YPIXSZ",
        "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BZERO", "BSCALE",
        "RA", "DEC", "CTYPE1", "CTYPE2",
    ]
    for k in keys:
        if k in hdr:
            print(f"  {k:9s} = {hdr[k]}")

    print("\n=== Data ===")
    print(f"  shape={data.shape} dtype={data.dtype}")
    arr = data.astype(np.float64)

    # Normalize to a list of (name, 2D-plane) for stats
    planes = []
    if arr.ndim == 2:
        planes = [("mono", arr)]
    elif arr.ndim == 3:
        # FITS color is typically (3, H, W)
        if arr.shape[0] == 3:
            for i, name in enumerate(["R", "G", "B"]):
                planes.append((name, arr[i]))
        elif arr.shape[-1] == 3:
            for i, name in enumerate(["R", "G", "B"]):
                planes.append((name, arr[..., i]))
        else:
            planes = [(f"plane{i}", arr[i]) for i in range(arr.shape[0])]

    print(f"  {'chan':6s} {'min':>12s} {'median':>12s} {'mean':>12s} {'max':>12s} {'p99.5':>12s}")
    for name, p in planes:
        print(f"  {name:6s} {p.min():12.2f} {np.median(p):12.2f} {p.mean():12.2f} "
              f"{p.max():12.2f} {np.percentile(p, 99.5):12.2f}")

    # Clipping / saturation check on the raw range
    theo_max = 65535.0 if arr.max() > 1.5 else 1.0
    for name, p in planes:
        clipped = np.mean(p >= theo_max * 0.999) * 100
        print(f"  {name}: {clipped:.3f}% pixels near saturation ({theo_max:g})")


if __name__ == "__main__":
    main(sys.argv[1])
