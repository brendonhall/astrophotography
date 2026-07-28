"""Background/gradient extraction stage."""
from __future__ import annotations
import numpy as np
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class BackgroundExtractStage(Stage):
    id = "background_extract"
    label = "Background extraction"
    description = "Per-channel low-order polynomial gradient removal; re-adds a neutral pedestal."
    INPUTS = [Port("image", space=Space.LINEAR_ADU)]
    OUTPUTS = [Port("image", space=Space.LINEAR_ADU)]
    PARAMS = [
        Param("degree", "int", 3, "Polynomial degree", min=1, max=6, step=1),
        Param("sample", "int", 12, "Pixel subsampling", min=1, max=64, step=1),
        Param("pedestal", "float", 0.10, "Pedestal", "Re-added level (fraction of 65535)",
              min=0, max=1, step=0.01),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        ped = params["pedestal"] * 65535.0
        out = np.empty_like(img.pixels)
        for c in range(3):
            model, _ = al.background_model(img.pixels[..., c], params["degree"], params["sample"])
            out[..., c] = img.pixels[..., c] - model + ped
        return {"image": img.replace(pixels=out)}
