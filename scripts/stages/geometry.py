"""Geometry stages (crop)."""
from __future__ import annotations
from .base import Stage, Param, Port
from .io import crop_header
from .registry import register


@register
class CropStage(Stage):
    id = "crop"
    label = "Crop border"
    description = "Trim a uniform margin off every side; shifts WCS CRPIX to match."
    INPUTS = [Port("image")]
    OUTPUTS = [Port("image")]
    PARAMS = [Param("margin", "int", 40, "Margin", "Pixels trimmed per side",
                    min=0, max=2000, step=1, unit="px")]

    def apply(self, inputs, params):
        img = inputs["image"]
        m = params["margin"]
        h, w = img.pixels.shape[:2]
        px = img.pixels[m:h - m, m:w - m, :]
        return {"image": img.replace(pixels=px, header=crop_header(img.header, m))}
