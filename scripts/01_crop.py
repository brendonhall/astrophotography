#!/usr/bin/env python3
"""Step 1 - Crop a thin border off the stack.

The SeeStar already crops its stacks, so we only trim a small margin to drop
any edge softness / partial-coverage pixels from dithering. Keeps full field.
"""
import sys
import astrolib as al

MARGIN = 40  # px trimmed from every side

def main(infile, outfile):
    img, hdr = al.load(infile)
    h, w, _ = img.shape
    cropped = img[MARGIN:h - MARGIN, MARGIN:w - MARGIN, :]
    al.save(outfile, cropped, hdr)
    print(f"cropped {img.shape} -> {cropped.shape}, wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
