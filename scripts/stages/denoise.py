"""Denoise / sharpen stages."""
from __future__ import annotations
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class MaskedDenoiseStage(Stage):
    id = "masked_denoise"
    label = "Background-aware denoise"
    description = "Denoise sky hard, galaxy gently, blended by a feathered source mask."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("bg_luma", "float", 0.06, "BG luma", min=0, max=1, step=0.005),
        Param("bg_chroma", "float", 10.0, "BG chroma", min=0, max=50, step=0.5, unit="px"),
        Param("gal_luma", "float", 0.010, "Galaxy luma", min=0, max=1, step=0.005),
        Param("gal_chroma", "float", 3.0, "Galaxy chroma", min=0, max=50, step=0.5, unit="px"),
        Param("feather", "float", 25.0, "Mask feather", min=0, max=200, step=1, unit="px"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.masked_denoise(img.pixels, bg_luma=params["bg_luma"], bg_chroma=params["bg_chroma"],
                                gal_luma=params["gal_luma"], gal_chroma=params["gal_chroma"],
                                feather=params["feather"])
        return {"image": img.replace(pixels=out)}


@register
class UnsharpLumaStage(Stage):
    id = "unsharp_luma"
    label = "Unsharp (luminance)"
    description = "Unsharp-mask the luminance only; hue preserved."
    INPUTS = [Port("image")]
    OUTPUTS = [Port("image")]
    PARAMS = [
        Param("amount", "float", 0.0, "Amount", min=0, max=5, step=0.1),
        Param("radius", "float", 2.0, "Radius", min=0.1, max=20, step=0.1, unit="px"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.unsharp_luma(img.pixels, amount=params["amount"], radius=params["radius"])
        return {"image": img.replace(pixels=out)}
