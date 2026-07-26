#!/usr/bin/env python3
"""Step 3 - Color calibration (shim over ColorCalibrateStage).

WCS now travels in the payload, so PCC measures on this step's own input; the
optional --original arg becomes a 'reference' frame override.
"""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.color import ColorCalibrateStage

REF_BP_RP, MIN_STARS = 0.82, 30

def main(infile, outfile, original=None, no_pcc=False, diagnostic=None):
    img = load_fits(infile, Space.LINEAR_ADU)
    inputs = {"image": img}
    if original:
        inputs["reference"] = load_fits(original, Space.LINEAR_ADU)
    out = ColorCalibrateStage().run(inputs, {
        "ref_bp_rp": REF_BP_RP, "min_stars": MIN_STARS,
        "no_pcc": no_pcc, "diagnostic_path": diagnostic or ""})["image"]
    save_fits(outfile, out)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out.pixels)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--original", help="optional WCS-bearing reference frame for PCC")
    ap.add_argument("--no-pcc", action="store_true")
    ap.add_argument("--diagnostic")
    a = ap.parse_args()
    main(a.infile, a.outfile, original=a.original, no_pcc=a.no_pcc, diagnostic=a.diagnostic)
