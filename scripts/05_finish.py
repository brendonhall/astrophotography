#!/usr/bin/env python3
"""Step 5 - Finish and export (operates on the nonlinear image).

  1. SCNR green - remove residual green pixels (g clamped to the R/B average),
     the standard fix for the magenta-free 'astro green' cast on stars/sky.
  2. Luminance denoise - light edge-preserving (TV) denoise to knock down grain
     without softening stars or galaxy structure.
  3. Chroma denoise - heavier blur of the color channels. Real color is
     large-scale; color noise is fine-scale, so strong chroma blur is safe.
  4. Saturation boost - applied AFTER denoise (so it doesn't amplify noise),
     nudged so HII regions and star colors show without going garish.
Exports a full-resolution 16-bit TIFF and an 8-bit PNG.
"""
import sys
import numpy as np
import astrolib as al

SATURATION = 1.20     # was 1.35 - lower so we don't re-amplify color noise
LUMA_DENOISE = 0.012  # TV-denoise weight on luminance (0 = off)
CHROMA_DENOISE = 4.0  # gaussian sigma on chroma (px) - was 1.2


def main(infile, outfile_base):
    img, hdr = al.load(infile)
    img = np.clip(img / 65535.0, 0, 1)

    # SCNR green -> luminance denoise -> chroma denoise -> saturation.
    # The shared al.finish() is the single source of truth for this so the
    # comparison tooling (scripts/compare.py) can't drift from the pipeline.
    img = al.finish(img, saturation=SATURATION, luma_denoise=LUMA_DENOISE,
                    chroma_denoise=CHROMA_DENOISE)

    # export
    from PIL import Image
    u16 = (img * 65535.0 + 0.5).astype(np.uint16)
    try:
        from skimage.io import imsave
        imsave(outfile_base + ".tif", u16, check_contrast=False)
        print(f"wrote {outfile_base}.tif (16-bit)")
    except Exception as e:
        print("TIFF export skipped:", e)
    u8 = (img * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(outfile_base + ".png")
    print(f"wrote {outfile_base}.png ({u8.shape[1]}x{u8.shape[0]})")
    # viewing-size preview
    al.save_preview(outfile_base + "_preview.png", img01=img, stretch=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
