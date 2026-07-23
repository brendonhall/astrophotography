#!/usr/bin/env python3
"""Generate before/after denoise comparison crops from a stretched FITS.

Each comparison isolates ONE finishing operation (varies only that step; all
others held at the pipeline's v2 settings) so its effect is unambiguous. Uses the
same al.finish() as the real pipeline, so comparisons can't drift from output.

Usage:
  python compare.py <stretched.fit> [--out DIR] [--which luminance chroma combined]
                    [--crop CX CY HALF]

Writes output/M101_compare_<which>-denoise.png for each requested comparison.
"""
import os
import sys
import argparse
import numpy as np
import astrolib as al
from PIL import Image, ImageDraw

# Pipeline v2 finishing settings (mirror scripts/05_finish.py).
SATURATION, LUMA, CHROMA = 1.20, 0.012, 4.0

# (name, before-kwargs, after-kwargs, before-label, after-label)
COMPARISONS = {
    "luminance": (
        dict(luma_denoise=0, chroma_denoise=0),
        dict(luma_denoise=LUMA, chroma_denoise=0),
        "BEFORE  -  no luminance denoise",
        f"AFTER  -  luminance TV denoise (w={LUMA})",
    ),
    "chroma": (
        dict(luma_denoise=0, chroma_denoise=0),
        dict(luma_denoise=0, chroma_denoise=CHROMA),
        "BEFORE  -  no chroma denoise",
        f"AFTER  -  chroma denoise (sigma {CHROMA})",
    ),
    "combined": (
        dict(luma_denoise=0, chroma_denoise=0),
        dict(luma_denoise=LUMA, chroma_denoise=CHROMA),
        "BEFORE  -  no denoise (SCNR + saturation)",
        "AFTER  -  full v2 (luminance + chroma)",
    ),
}


def noise_stats(a, box):
    y, x, s = box
    p = a[y:y+s, x:x+s]
    chroma = (p - p.mean(2, keepdims=True)).std() * 1000
    luma = p.mean(2).std() * 1000
    return chroma, luma


def crop_img(a, cx, cy, half, out=860):
    c = np.clip(a[cy-half:cy+half, cx-half:cx+half], 0, 1)
    return Image.fromarray((c*255+0.5).astype(np.uint8)).resize((out, out), Image.LANCZOS)


def stitch(before, after, ltext, rtext, cx, cy, half, path):
    b, a = crop_img(before, cx, cy, half), crop_img(after, cx, cy, half)
    W, H = b.size
    div, top = 6, 36
    canvas = Image.new("RGB", (W*2+div, H+top), (18, 18, 18))
    canvas.paste(b, (0, top)); canvas.paste(a, (W+div, top))
    d = ImageDraw.Draw(canvas)
    d.text((10, 12), ltext, fill=(255, 255, 255))
    d.text((W+div+10, 12), rtext, fill=(255, 255, 255))
    canvas.save(path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(here), "output")

    ap = argparse.ArgumentParser()
    ap.add_argument("stretched", help="nonlinear stretched FITS (e.g. work/04_stretch.fit)")
    ap.add_argument("--out", default=default_out, help="output directory")
    ap.add_argument("--which", nargs="+", default=list(COMPARISONS),
                    choices=list(COMPARISONS), help="which comparisons to make")
    ap.add_argument("--crop", nargs=3, type=int, metavar=("CX", "CY", "HALF"),
                    help="crop center + half-size (default: image center, H/6)")
    args = ap.parse_args()

    base, _ = al.load(args.stretched)
    base = np.clip(base / 65535.0, 0, 1)
    H, W, _ = base.shape

    if args.crop:
        cx, cy, half = args.crop
    else:
        cx, cy, half = W // 2, H // 2, min(H, W) // 6
    # a background patch for noise stats: offset toward a corner, clamped in-frame
    box = (min(int(0.2*H), H-180), min(int(0.15*W), W-180), 180)

    os.makedirs(args.out, exist_ok=True)
    for name in args.which:
        before_kw, after_kw, lt, rt = COMPARISONS[name]
        before = al.finish(base, saturation=SATURATION, **before_kw)
        after = al.finish(base, saturation=SATURATION, **after_kw)
        path = os.path.join(args.out, f"M101_compare_{name}-denoise.png")
        stitch(before, after, lt, rt, cx, cy, half, path)
        cb, lb = noise_stats(before, box)
        ca, la = noise_stats(after, box)
        print(f"{name:10s} chroma {cb:5.1f}->{ca:5.1f}  luma {lb:5.1f}->{la:5.1f} "
              f"(x1e-3)  ->  {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
