"""Sink stages: export final image + preview taps."""
from __future__ import annotations
import numpy as np
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class ExportImageStage(Stage):
    id = "export_image"
    label = "Export (TIFF/PNG)"
    description = "Write a 16-bit TIFF, an 8-bit PNG, and a viewing-size preview PNG."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = []
    PARAMS = [Param("out_base", "str", "", "Output base path (no extension)")]

    def apply(self, inputs, params):
        base = params["out_base"]
        result = np.clip(inputs["image"].pixels, 0, 1)
        from PIL import Image as PILImage
        u16 = (result * 65535.0 + 0.5).astype(np.uint16)
        try:
            from skimage.io import imsave
            imsave(base + ".tif", u16, check_contrast=False)
        except Exception as e:
            print("TIFF export skipped:", e)
        PILImage.fromarray((result * 255 + 0.5).astype(np.uint8)).save(base + ".png")
        al.save_preview(base + "_preview.png", img01=result, stretch=False)
        return {}


@register
class PreviewSink(Stage):
    id = "preview_sink"
    label = "Preview PNG"
    description = "Write a single preview PNG (diagnostic tap)."
    INPUTS = [Port("image")]
    OUTPUTS = []
    PARAMS = [Param("out_path", "str", "", "Output PNG path"),
              Param("stretch", "bool", False, "Apply autostretch")]

    def apply(self, inputs, params):
        al.save_preview(params["out_path"], img01=np.clip(inputs["image"].pixels, 0, 1),
                        stretch=params["stretch"])
        return {}
