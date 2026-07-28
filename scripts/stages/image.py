"""Image payload flowing between stages: pixels + FITS header/WCS + space tag."""
from __future__ import annotations
from dataclasses import dataclass, replace as _replace
from enum import Enum
import numpy as np
from astropy.io import fits


class Space(str, Enum):
    LINEAR_ADU = "linear-adu"   # crop..color: float ADU ~0..65535
    NONLINEAR = "nonlinear"     # after stretch: [0,1]

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Image:
    pixels: np.ndarray                     # (H,W,3) float32
    space: Space
    header: fits.Header | None = None

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def wcs(self):
        from astropy.wcs import WCS
        if self.header is None:
            return None
        try:
            w = WCS(self.header, naxis=2).celestial
            return w if w.has_celestial else None
        except Exception:
            return None

    def replace(self, *, pixels=None, space=None, header=None) -> "Image":
        return _replace(
            self,
            pixels=self.pixels if pixels is None else pixels,
            space=self.space if space is None else space,
            header=self.header if header is None else header,
        )
