#!/usr/bin/env python3
"""Step 5 - Finish and export (shim over FinishStage + ExportImageStage)."""
import sys
from stages.image import Space
from stages.io import load_fits
from stages.finish import FinishStage
from stages.export import ExportImageStage

SATURATION, LUMA_DENOISE, CHROMA_DENOISE = 1.20, 0.012, 4.0

def main(infile, outfile_base):
    img = load_fits(infile, Space.NONLINEAR)
    finished = FinishStage().run({"image": img}, {
        "saturation": SATURATION, "luma_denoise": LUMA_DENOISE,
        "chroma_denoise": CHROMA_DENOISE, "scnr": True})["image"]
    ExportImageStage().run({"image": finished}, {"out_base": outfile_base})

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
