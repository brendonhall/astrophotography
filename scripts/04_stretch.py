#!/usr/bin/env python3
"""Step 4 - Linked stretch, linear -> nonlinear (shim over StretchStage)."""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.stretch import StretchStage

TARGET_BG, SHADOWS_CLIP = 0.18, -1.8

def main(infile, outfile):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = StretchStage().run(
        {"image": img}, {"target_bg": TARGET_BG, "shadows_clip": SHADOWS_CLIP})["image"]
    save_fits(outfile, out)  # nonlinear, stored as [0,1] + PIPESPCE
    al.save_preview(outfile.replace(".fit", "_preview.png"), img01=out.pixels, stretch=False)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
