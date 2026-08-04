"""Source stage: load a FITS file into the pipeline."""
from __future__ import annotations
from .base import Stage, Param, Port
from .image import Space
from .io import load_fits
from .registry import register


@register
class LoadStage(Stage):
    id = "load"
    label = "Load FITS"
    description = "Read a FITS file into the pipeline as the source image."
    INPUTS = []
    OUTPUTS = [Port("image")]
    PARAMS = [
        Param("path", "str", "", "FITS path"),
        Param("space", "enum", "linear-adu", "Space",
              choices=("linear-adu", "nonlinear")),
    ]

    def apply(self, inputs, params):
        return {"image": load_fits(params["path"], Space(params["space"]))}
