"""Star-layer stages (StarNet2 removal, screen recombine)."""
from __future__ import annotations
import astrolib as al
import starnet
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class RemoveStarsStage(Stage):
    id = "remove_stars"
    label = "Remove stars (StarNet2)"
    description = "Split into starless + star layers; screen(starless, stars) reconstructs input."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("starless", space=Space.NONLINEAR), Port("stars", space=Space.NONLINEAR)]
    PARAMS = [Param("stride", "int", 256, "Tile stride", min=64, max=1024, step=64, unit="px")]

    def check(self, inputs, params):
        errs = super().check(inputs, params)
        img = inputs.get("image")
        if img is not None and min(img.pixels.shape[0], img.pixels.shape[1]) < starnet.MIN_SIZE:
            errs.append(f"image {img.pixels.shape[1]}x{img.pixels.shape[0]} "
                        f"below StarNet2 {starnet.MIN_SIZE} minimum")
        return errs

    def apply(self, inputs, params):
        img = inputs["image"]
        starless, stars = starnet.remove_stars(img.pixels, stride=params["stride"])
        return {"starless": img.replace(pixels=starless), "stars": img.replace(pixels=stars)}


@register
class ScreenRecombineStage(Stage):
    id = "screen_recombine"
    label = "Screen recombine"
    description = "Screen an overlay (e.g. stars) back over a base image."
    INPUTS = [Port("base", space=Space.NONLINEAR), Port("overlay", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = []

    def apply(self, inputs, params):
        base = inputs["base"]
        out = al.screen(base.pixels, inputs["overlay"].pixels)
        return {"image": base.replace(pixels=out)}
