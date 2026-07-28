"""Finishing stages (full finish, saturation-only)."""
from __future__ import annotations
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class FinishStage(Stage):
    id = "finish"
    label = "Finish"
    description = "SCNR green, luma + chroma denoise, saturation on a nonlinear image."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("saturation", "float", 1.20, "Saturation", min=0, max=3, step=0.05),
        Param("luma_denoise", "float", 0.012, "Luma denoise", min=0, max=1, step=0.001),
        Param("chroma_denoise", "float", 4.0, "Chroma sigma", min=0, max=50, step=0.5, unit="px"),
        Param("scnr", "bool", True, "SCNR green"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.finish(img.pixels, saturation=params["saturation"],
                        luma_denoise=params["luma_denoise"],
                        chroma_denoise=params["chroma_denoise"], scnr=params["scnr"])
        return {"image": img.replace(pixels=out)}


@register
class SaturateStage(Stage):
    id = "saturate"
    label = "Saturation"
    description = "HSV saturation multiplier only (no denoise/SCNR)."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [Param("saturation", "float", 1.20, "Saturation", min=0, max=3, step=0.05)]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.finish(img.pixels, saturation=params["saturation"],
                        luma_denoise=0, chroma_denoise=0, scnr=False)
        return {"image": img.replace(pixels=out)}
