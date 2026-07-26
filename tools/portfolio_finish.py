#!/usr/bin/env python3
"""Portfolio finish for deep-sky images — the "M101 v1" recipe, repeatable.

Takes a COLOR-CALIBRATED, STRETCHED image (e.g. a PCC'd 16-bit TIFF exported
from Siril) and produces a finished portfolio image. By default it reproduces
the look we settled on for M101:

    Cosmic Clarity full denoise  ->  deepen+neutralize sky  ->  global
    saturation  ->  mild mid-contrast

Optional star reduction (StarNet2) and a centered crop are OFF by default.

Run with the Cosmic Clarity venv (it has tifffile, cv2, numpy, astropy, and can
import scripts/starnet):

    ~/CosmicClarity/.venv/bin/python tools/portfolio_finish.py <in> <out> [opts]

Examples:
    # v1 recipe (full denoise + polish), full frame:
    portfolio_finish.py in_pcc.tif  out.tif
    # skip denoise (input already denoised):
    portfolio_finish.py in.tif out.tif --denoise none
    # star-reduced + centered 1500px portrait crop:
    portfolio_finish.py in_pcc.tif out.tif --star-reduce 0.5 --crop 1500

Input must be nonlinear (already stretched). Output is never overwritten.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
CCDENOISE = os.path.join(REPO, "tools", "ccdenoise.sh")
LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load01(path):
    """Load an image as (H,W,3) float32 in [0,1], from TIFF/PNG or FITS."""
    import tifffile
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fit", ".fits"):
        from astropy.io import fits
        d = fits.getdata(path).astype(np.float64)
        if d.ndim == 3 and d.shape[0] == 3:
            d = np.moveaxis(d, 0, -1)
        if d.max() > 1.5:                       # not already normalized
            d = d / np.percentile(d, 99.99)
    else:
        raw = tifffile.imread(path)
        if raw.dtype == np.uint16:
            d = raw.astype(np.float64) / 65535.0
        elif raw.dtype == np.uint8:
            d = raw.astype(np.float64) / 255.0
        else:
            d = raw.astype(np.float64)
            if d.max() > 1.5:
                d = d / np.percentile(d, 99.99)
    if d.ndim == 2:
        d = np.stack([d] * 3, axis=-1)
    return np.clip(np.nan_to_num(d), 0.0, 1.0).astype(np.float32)


def screen(a, b):
    return np.clip(1.0 - (1.0 - a) * (1.0 - b), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Portfolio finish (M101 v1 recipe).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--denoise", choices=["full", "luminance", "none"], default="full",
                    help="Cosmic Clarity denoise mode to run first (default full)")
    ap.add_argument("--denoise-strength", type=float, default=0.5)
    ap.add_argument("--saturation", type=float, default=1.6, help="color saturation (default 1.6)")
    ap.add_argument("--black-offset", type=float, default=0.02,
                    help="deepen sky: black point = channel median - this (default 0.02)")
    ap.add_argument("--star-reduce", type=float, default=0.0,
                    help="StarNet2 star reduction: kept star brightness 0..1 (0=off, e.g. 0.5)")
    ap.add_argument("--crop", type=int, default=0, help="centered NxN crop in px (0=off)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output):
        sys.exit(f"error: output '{args.output}' exists — choose another name")

    # --- Stage 1: Cosmic Clarity denoise (optional) ---
    src = args.input
    tmpdir = None
    if args.denoise != "none":
        tmpdir = tempfile.mkdtemp(prefix="pfinish_")
        dn = os.path.join(tmpdir, "denoised.tif")
        print(f">> Cosmic Clarity denoise ({args.denoise}, {args.denoise_strength})")
        subprocess.run([CCDENOISE, src, dn, str(args.denoise_strength), args.denoise],
                       check=True)
        src = dn

    img = load01(src)

    # --- Stage 2: star reduction (optional) ---
    star_reduced = args.star_reduce > 0.0
    if star_reduced:
        import starnet
        print(f">> StarNet2 star reduction (keep {args.star_reduce:.0%})")
        starless, stars = starnet.remove_stars(img)
        img = screen(starless, stars * args.star_reduce)

    # --- Stage 3: deepen + neutralize sky (per-channel black point) ---
    med = np.median(img.reshape(-1, 3), axis=0)
    bp = np.clip(med - args.black_offset, 0, None)
    img = np.clip((img - bp) / (1.0 - bp), 0.0, 1.0)

    # --- Stage 4: saturation ---
    if star_reduced:
        # star-reduced frames need care: smooth chroma + saturate only where there
        # is signal, so the sky doesn't turn blotchy.
        import cv2
        luma = (img * LUMA_W).sum(axis=2, keepdims=True)
        chroma = img - luma
        for c in range(3):
            chroma[..., c] = cv2.GaussianBlur(chroma[..., c], (0, 0), 3.0)
        img = np.clip(luma + chroma, 0.0, 1.0)
        lum_s = (img * LUMA_W).sum(axis=2, keepdims=True)
        weight = np.clip((lum_s - 0.06) / (0.30 - 0.06), 0.0, 1.0)
        sat_map = 1.0 + (args.saturation - 1.0) * weight
        img = np.clip(lum_s + (img - lum_s) * sat_map, 0.0, 1.0)
    else:
        # v1 path: simple global saturation (boosts star colors across the field).
        lum_s = (img * LUMA_W).sum(axis=2, keepdims=True)
        img = np.clip(lum_s + (img - lum_s) * args.saturation, 0.0, 1.0)

    # --- Stage 5: mild mid-contrast (protects black/white ends) ---
    lum = img.mean(axis=2, keepdims=True)
    new_lum = np.clip(lum + 0.12 * (lum - 0.5) * (1 - np.abs(2 * lum - 1)), 0, 1)
    ratio = np.divide(new_lum, lum, out=np.ones_like(lum), where=lum > 1e-4)
    img = np.clip(img * ratio, 0.0, 1.0)

    # --- optional centered crop ---
    if args.crop > 0:
        H, W = img.shape[:2]
        s = args.crop
        r, c = max(0, H // 2 - s // 2), max(0, W // 2 - s // 2)
        img = img[r:r + s, c:c + s]

    # --- write outputs ---
    import tifffile
    tifffile.imwrite(args.output, (img * 65535).astype(np.uint16))
    print(f">> wrote {args.output} {img.shape}")
    if not args.no_preview:
        from PIL import Image
        base = args.output.rsplit(".", 1)[0]
        step = max(1, img.shape[1] // 1400)
        Image.fromarray((img[::step, ::step] * 255).astype(np.uint8)).save(base + "_preview.png")
        print(f">> wrote {base}_preview.png")

    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
