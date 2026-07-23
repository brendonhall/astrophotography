#!/usr/bin/env python3
"""Render a FITS image to PNG for viewing, with optional STF autostretch.

STF autostretch = the PixInsight/Siril "ScreenTransferFunction" midtones stretch:
a robust, non-destructive preview stretch driven by the image median and MAD.

Usage:
  python render.py <in.fit> <out.png> [--linear] [--width N]
"""
import sys
import argparse
import numpy as np
from astropy.io import fits


def mtf(x, m):
    """Midtones Transfer Function (PixInsight form). m in (0,1) is the midtones balance."""
    x = np.clip(x, 0.0, 1.0)
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    out = np.divide(num, den, out=np.zeros_like(x), where=den != 0)
    return out


def stf_params(img, target_bg=0.25, shadows_clip=-2.8):
    """Compute (black_point, midtone) from robust stats on the whole image (linked channels)."""
    med = np.median(img)
    mad = np.median(np.abs(img - med)) * 1.4826  # sigma-equivalent
    black = np.clip(med + shadows_clip * mad, 0.0, 1.0)
    # median position after black-point subtraction
    m_shifted = (med - black) / (1.0 - black) if (1.0 - black) > 0 else med
    midtone = mtf_midtones(m_shifted, target_bg)
    return black, midtone


def mtf_midtones(x, target):
    """Find midtone m such that MTF(x, m) == target (closed form)."""
    if x <= 0:
        return 0.5
    if x >= 1:
        return 0.5
    return ((target - 1.0) * x) / ((2.0 * target - 1.0) * x - target)


def autostretch(img):
    black, midtone = stf_params(img)
    scaled = np.clip((img - black) / (1.0 - black), 0.0, 1.0)
    return mtf(scaled, midtone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--linear", action="store_true", help="no autostretch (raw linear, just normalized)")
    ap.add_argument("--width", type=int, default=1400, help="output width in px (downsample for viewing)")
    args = ap.parse_args()

    data = fits.getdata(args.infile).astype(np.float64)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W) -> (H,W,3)
    img = data / 65535.0

    if not args.linear:
        img = autostretch(img)

    # downsample for viewing
    from PIL import Image
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(out)
    if args.width and im.width > args.width:
        h = round(im.height * args.width / im.width)
        im = im.resize((args.width, h), Image.LANCZOS)
    im.save(args.outfile)
    print(f"wrote {args.outfile} ({im.width}x{im.height})")


if __name__ == "__main__":
    main()
