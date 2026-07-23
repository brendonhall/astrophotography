#!/usr/bin/env python3
"""Step 4 - Stretch: linear -> nonlinear (the step that makes the image).

Applies a LINKED midtones-transfer stretch: robust stats are computed on the
whole image and the SAME transfer curve is applied to all three channels, which
preserves color ratios while lifting the faint galaxy signal off the floor.

target_bg controls how bright the sky sits (higher = brighter/flatter-looking);
shadows_clip sets how many MADs below the median get crushed to black.
Output is written in a [0,1] range (stored *65535 so the FITS stays 16-bit-scale).
"""
import sys
import numpy as np
import astrolib as al

TARGET_BG = 0.18     # sky brightness after stretch
SHADOWS_CLIP = -1.8  # keep faint outer arms; less aggressive black clip


def main(infile, outfile):
    img, hdr = al.load(infile)
    img01 = np.clip(img / 65535.0, 0, 1)

    # robust global stats -> one linked MTF for all channels
    med = np.median(img01)
    mad = np.median(np.abs(img01 - med)) * 1.4826
    black = np.clip(med + SHADOWS_CLIP * mad, 0.0, 1.0)
    scaled = np.clip((img01 - black) / (1.0 - black), 0.0, 1.0)
    m_shift = (med - black) / (1.0 - black)
    m = al._midtone(m_shift, TARGET_BG)
    stretched = al._mtf(scaled, m)
    print(f"  black={black:.4f}  midtone={m:.4f}  sky->{TARGET_BG}")

    al.save(outfile, stretched * 65535.0, hdr)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img01=stretched, stretch=False)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
