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


def rgb_to_hsv(rgb):
    from matplotlib.colors import rgb_to_hsv as f
    return f(np.clip(rgb, 0, 1))


def hsv_to_rgb(hsv):
    from matplotlib.colors import hsv_to_rgb as f
    return f(hsv)


def main(infile, outfile_base):
    from scipy import ndimage
    img, hdr = al.load(infile)
    img = np.clip(img / 65535.0, 0, 1)

    # 1. SCNR green (average neutral protection)
    img[..., 1] = np.minimum(img[..., 1], 0.5 * (img[..., 0] + img[..., 2]))

    # 2. luminance denoise (edge-preserving), then 3. chroma denoise (heavy),
    #    working in a luminance/chroma split so we can treat them differently.
    lum = img.mean(axis=2, keepdims=True)
    chroma = img - lum

    if LUMA_DENOISE > 0:
        from skimage.restoration import denoise_tv_chambolle
        lum[..., 0] = denoise_tv_chambolle(lum[..., 0], weight=LUMA_DENOISE)

    for c in range(3):
        chroma[..., c] = ndimage.gaussian_filter(chroma[..., c], CHROMA_DENOISE)
    img = np.clip(lum + chroma, 0, 1)

    # 4. saturation boost (after denoise, so it doesn't re-amplify color noise)
    hsv = rgb_to_hsv(img)
    hsv[..., 1] = np.clip(hsv[..., 1] * SATURATION, 0, 1)
    img = hsv_to_rgb(hsv)

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
