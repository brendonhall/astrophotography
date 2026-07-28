#!/usr/bin/env python3
"""Step 1 - Crop a thin border off the stack (shim over CropStage)."""
import sys
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.geometry import CropStage

MARGIN = 40

def main(infile, outfile, margin=MARGIN):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = CropStage().run({"image": img}, {"margin": margin})["image"]
    save_fits(outfile, out)
    print(f"cropped -> {out.pixels.shape}, wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
