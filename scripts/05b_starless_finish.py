#!/usr/bin/env python3
"""Step 5b - Starless galaxy finish (alternative to 05_finish).

Recreates the SetiAstroSuitePro galaxy workflow in Python: remove stars with
the StarNet2 CLI, denoise the starless galaxy, then screen the star layer
back in. Operates on the stretched (nonlinear) image from step 04. See
docs/superpowers/specs/2026-07-25-starless-galaxy-workflow-design.md.

Denoise is background-aware (mask-based): with no stars left to protect, the
empty sky is denoised hard while a feathered luminance mask keeps the galaxy
itself gentle/sharp. See astrolib.masked_denoise. Sharpening defaults OFF
(SHARPEN_AMOUNT=0.0) since the experiment showed sharpening the sky just adds
grain; set SHARPEN_AMOUNT>0 to re-enable it (it applies globally, sky and
galaxy alike).
"""
import sys
import numpy as np
import astrolib as al
import starnet

SHARPEN_AMOUNT = 0.0    # unsharp amount on starless luma (0 = off, default)
SHARPEN_RADIUS = 2.0    # unsharp gaussian radius (px)
BG_LUMA_DENOISE = 0.06  # TV-denoise weight on background luminance (hard)
BG_CHROMA_DENOISE = 10.0  # gaussian sigma on background chroma (px, hard)
GAL_LUMA_DENOISE = 0.010  # TV-denoise weight on galaxy luminance (gentle)
GAL_CHROMA_DENOISE = 3.0  # gaussian sigma on galaxy chroma (px, gentle)
MASK_FEATHER = 25.0     # gaussian feather (px) on the galaxy/background mask
SATURATION = 1.20       # HSV saturation multiplier
STRIDE = 256            # StarNet2 tile stride


def process_starless(starless01, sharpen=False):
    """Sharpen (luma, optional) then background-aware denoise + saturate a
    [0,1] starless RGB image.

    SCNR is off here: the green cast is handled upstream in step 03, and the
    star layer (added back later) is where residual green usually lives.
    """
    img = starless01
    if sharpen and SHARPEN_AMOUNT > 0:
        img = al.unsharp_luma(img, amount=SHARPEN_AMOUNT, radius=SHARPEN_RADIUS)
    img = al.masked_denoise(img, bg_luma=BG_LUMA_DENOISE, bg_chroma=BG_CHROMA_DENOISE,
                            gal_luma=GAL_LUMA_DENOISE, gal_chroma=GAL_CHROMA_DENOISE,
                            feather=MASK_FEATHER)
    # saturation only (denoise already done above)
    return al.finish(img, saturation=SATURATION, luma_denoise=0, chroma_denoise=0,
                     scnr=False)


def main(infile, outfile_base, stride=STRIDE, no_sharpen=False):
    img, _hdr = al.load(infile)
    img01 = np.clip(img / 65535.0, 0.0, 1.0)

    print(">> starnet: removing stars")
    starless, stars = starnet.remove_stars(img01, stride=stride)

    print(">> processing starless (sharpen -> denoise -> saturation)")
    starless_proc = process_starless(starless, sharpen=not no_sharpen)

    result = al.screen(starless_proc, stars)

    from PIL import Image
    u16 = (result * 65535.0 + 0.5).astype(np.uint16)
    try:
        from skimage.io import imsave
        imsave(outfile_base + ".tif", u16, check_contrast=False)
        print(f"wrote {outfile_base}.tif (16-bit)")
    except Exception as e:
        print("TIFF export skipped:", e)
    Image.fromarray((result * 255 + 0.5).astype(np.uint8)).save(outfile_base + ".png")
    print(f"wrote {outfile_base}.png")
    al.save_preview(outfile_base + "_preview.png", img01=result, stretch=False)
    # diagnostics: inspect for star halos / over-sharpening
    al.save_preview(outfile_base + "_starless.png", img01=starless_proc, stretch=False)
    al.save_preview(outfile_base + "_starlayer.png", img01=stars, stretch=False)
    print(f"wrote diagnostics {outfile_base}_starless.png / _starlayer.png")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--no-sharpen", action="store_true")
    a = ap.parse_args()
    try:
        main(a.infile, a.outfile, stride=a.stride, no_sharpen=a.no_sharpen)
    except starnet.StarNetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
