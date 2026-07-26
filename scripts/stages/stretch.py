"""Linear -> nonlinear stretch stage."""
from __future__ import annotations
import numpy as np
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class StretchStage(Stage):
    id = "stretch"
    label = "Stretch (linear -> nonlinear)"
    description = "Linked midtones-transfer stretch; converts linear ADU to nonlinear [0,1]."
    INPUTS = [Port("image", space=Space.LINEAR_ADU)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("target_bg", "float", 0.18, "Target sky", min=0.01, max=0.9, step=0.01),
        Param("shadows_clip", "float", -1.8, "Shadows clip (MAD)", min=-5, max=0, step=0.1),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        img01 = np.clip(img.pixels / 65535.0, 0, 1)
        stretched = al.linked_stretch(img01, target_bg=params["target_bg"],
                                      shadows_clip=params["shadows_clip"])
        return {"image": img.replace(pixels=stretched, space=Space.NONLINEAR)}
