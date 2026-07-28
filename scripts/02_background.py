#!/usr/bin/env python3
"""Step 2 - Background/gradient removal (shim over BackgroundExtractStage)."""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.background import BackgroundExtractStage

DEGREE, SAMPLE, PEDESTAL = 3, 12, 0.10

def main(infile, outfile):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = BackgroundExtractStage().run(
        {"image": img}, {"degree": DEGREE, "sample": SAMPLE, "pedestal": PEDESTAL})["image"]
    save_fits(outfile, out)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out.pixels)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
