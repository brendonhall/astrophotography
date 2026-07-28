#!/usr/bin/env python3
"""Step 5b - Starless galaxy finish (shim orchestrating decomposed stages).

remove_stars -> [unsharp] -> masked_denoise -> saturate -> screen(stars) -> export,
with preview taps on the starless and star layers. See the stages package.
"""
import sys
from stages.image import Space
from stages.io import load_fits
from stages.stars import RemoveStarsStage, ScreenRecombineStage
from stages.denoise import UnsharpLumaStage, MaskedDenoiseStage
from stages.finish import SaturateStage
from stages.export import ExportImageStage, PreviewSink
import starnet

SHARPEN_AMOUNT, SHARPEN_RADIUS = 0.0, 2.0
BG_LUMA_DENOISE, BG_CHROMA_DENOISE = 0.06, 10.0
GAL_LUMA_DENOISE, GAL_CHROMA_DENOISE = 0.010, 3.0
MASK_FEATHER, SATURATION, STRIDE = 25.0, 1.20, 256

def main(infile, outfile_base, stride=STRIDE, no_sharpen=False):
    img = load_fits(infile, Space.NONLINEAR)
    split = RemoveStarsStage().run({"image": img}, {"stride": stride})
    starless, stars = split["starless"], split["stars"]

    proc = starless
    if not no_sharpen and SHARPEN_AMOUNT > 0:
        proc = UnsharpLumaStage().run({"image": proc},
            {"amount": SHARPEN_AMOUNT, "radius": SHARPEN_RADIUS})["image"]
    proc = MaskedDenoiseStage().run({"image": proc}, {
        "bg_luma": BG_LUMA_DENOISE, "bg_chroma": BG_CHROMA_DENOISE,
        "gal_luma": GAL_LUMA_DENOISE, "gal_chroma": GAL_CHROMA_DENOISE,
        "feather": MASK_FEATHER})["image"]
    proc = SaturateStage().run({"image": proc}, {"saturation": SATURATION})["image"]

    result = ScreenRecombineStage().run({"base": proc, "overlay": stars})["image"]
    ExportImageStage().run({"image": result}, {"out_base": outfile_base})
    PreviewSink().run({"image": proc}, {"out_path": outfile_base + "_starless.png", "stretch": False})
    PreviewSink().run({"image": stars}, {"out_path": outfile_base + "_starlayer.png", "stretch": False})
    return result.pixels

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--no-sharpen", action="store_true")
    a = ap.parse_args()
    try:
        main(a.infile, a.outfile, stride=a.stride, no_sharpen=a.no_sharpen)
    except starnet.StarNetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
