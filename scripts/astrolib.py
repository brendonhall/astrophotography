"""Shared helpers for the M101 processing pipeline.

Convention: images are carried as float32 arrays shaped (H, W, 3) in ADU
(same scale as the original 16-bit data, ~0..65535). FITS on disk is (3, H, W).
"""
import numpy as np
from astropy.io import fits


# ---------- I/O ----------

def load(path):
    """Return (img (H,W,3) float32 ADU, header)."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W) -> (H,W,3)
    return data, hdr


def save(path, img, header=None):
    """Write (H,W,3) ADU float image as a 3-plane float32 FITS."""
    arr = np.moveaxis(img.astype(np.float32), -1, 0)  # (3,H,W)
    hdu = fits.PrimaryHDU(data=arr)
    if header is not None:
        for k in ("OBJECT", "DATE-OBS", "EXPTIME", "STACKCNT", "FILTER",
                  "GAIN", "TELESCOP", "FOCALLEN"):
            if k in header:
                hdu.header[k] = header[k]
    hdu.writeto(path, overwrite=True)


# ---------- STF autostretch (for previewing LINEAR stages) ----------

def _mtf(x, m):
    x = np.clip(x, 0.0, 1.0)
    den = (2.0 * m - 1.0) * x - m
    return np.divide((m - 1.0) * x, den, out=np.zeros_like(x), where=den != 0)


def _midtone(x, target):
    if x <= 0 or x >= 1:
        return 0.5
    return ((target - 1.0) * x) / ((2.0 * target - 1.0) * x - target)


def autostretch(img01, target_bg=0.25, shadows_clip=-2.8):
    """STF preview stretch on a [0,1] image using global robust stats."""
    med = np.median(img01)
    mad = np.median(np.abs(img01 - med)) * 1.4826
    black = np.clip(med + shadows_clip * mad, 0.0, 1.0)
    scaled = np.clip((img01 - black) / (1.0 - black), 0.0, 1.0)
    m_shift = (med - black) / (1.0 - black) if (1.0 - black) > 0 else med
    return _mtf(scaled, _midtone(m_shift, target_bg))


def save_preview(path, img_adu=None, img01=None, stretch=True, width=1400):
    """Save a PNG. Pass ADU image (auto /65535) or an already-[0,1] image.
    stretch=True applies STF autostretch (use for linear stages)."""
    from PIL import Image
    if img01 is None:
        img01 = np.clip(img_adu / 65535.0, 0, 1)
    disp = autostretch(img01) if stretch else np.clip(img01, 0, 1)
    out = (disp * 255).astype(np.uint8)
    im = Image.fromarray(out)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(path)
    return im.size


# ---------- robust background source mask ----------

def source_mask(chan, k=3.0, dilate=6):
    """Boolean mask of likely source (star/galaxy) pixels to EXCLUDE from bg fit."""
    from scipy import ndimage
    med = np.median(chan)
    mad = np.median(np.abs(chan - med)) * 1.4826
    m = chan > med + k * mad
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m
