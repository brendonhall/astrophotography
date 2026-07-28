"""Shared helpers for the M101 processing pipeline.

Convention: images are carried as float32 arrays shaped (H, W, 3) in ADU
(same scale as the original 16-bit data, ~0..65535). FITS on disk is (3, H, W).
"""
import numpy as np
from astropy.io import fits


# ---------- I/O ----------

def load(path):
    """Return (img (H,W,3) float32 ADU, header).

    Siril writes 32-bit stacked masters normalized to [0,1], but this pipeline
    works in ADU (~0..65535). A normalized-float input (max <= 1.5) is scaled up
    so Siril re-stacks feed the pipeline directly with no manual conversion.
    Integer/ADU stacks (max in the thousands) are left untouched.
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W) -> (H,W,3)
    if data.size and float(np.nanmax(data)) <= 1.5:
        data = data * 65535.0
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


# ---------- nonlinear finishing ----------

def finish(img01, saturation=1.20, luma_denoise=0.012, chroma_denoise=4.0, scnr=True):
    """Finish a nonlinear [0,1] RGB image and return [0,1] RGB.

    Steps (in order): SCNR green -> luminance denoise -> chroma denoise ->
    saturation. Passing 0 (or False) for a step disables it, which is what the
    comparison tooling uses to isolate one operation at a time.

      saturation     : HSV saturation multiplier (1.0 = no change)
      luma_denoise   : TV-denoise weight on luminance (0 = off)
      chroma_denoise : gaussian sigma (px) on the color channels (0 = off)
      scnr           : remove residual green (g clamped to the R/B average)
    """
    from scipy import ndimage
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    img = np.clip(img01, 0.0, 1.0).astype(np.float64).copy()

    if scnr:
        img[..., 1] = np.minimum(img[..., 1], 0.5 * (img[..., 0] + img[..., 2]))

    lum = img.mean(axis=2, keepdims=True)
    chroma = img - lum
    if luma_denoise:
        from skimage.restoration import denoise_tv_chambolle
        lum[..., 0] = denoise_tv_chambolle(lum[..., 0], weight=luma_denoise)
    if chroma_denoise:
        for c in range(3):
            chroma[..., c] = ndimage.gaussian_filter(chroma[..., c], chroma_denoise)
    img = np.clip(lum + chroma, 0.0, 1.0)

    if saturation and saturation != 1.0:
        hsv = rgb_to_hsv(img)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        img = hsv_to_rgb(hsv)
    return img


# ---------- layer blending / sharpening (starless workflow) ----------

def screen(a, b):
    """Screen blend of two [0,1] arrays: 1 - (1-a)(1-b), clipped to [0,1].

    Screen is the recombination used to add a star layer back over a
    processed starless image. It is the exact inverse of StarNet2's
    --unscreen output, so screen(starless, stars) reconstructs the source.
    """
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return np.clip(1.0 - (1.0 - a) * (1.0 - b), 0.0, 1.0)


def masked_denoise(img01, bg_luma=0.06, bg_chroma=10.0, gal_luma=0.010,
                   gal_chroma=3.0, feather=25.0, mask_k=2.0, mask_dilate=8):
    """Denoise the background hard and the galaxy gently, blended by a feathered
    source mask. Because a starless image has no stars to protect, the empty sky
    can be denoised aggressively while a luminance mask keeps the galaxy sharp.
    Returns a [0,1] RGB image (no saturation change - caller applies that)."""
    from scipy import ndimage
    img = np.clip(img01, 0.0, 1.0)
    gentle = finish(img, saturation=1.0, luma_denoise=gal_luma,
                    chroma_denoise=gal_chroma, scnr=False)
    heavy = finish(img, saturation=1.0, luma_denoise=bg_luma,
                   chroma_denoise=bg_chroma, scnr=False)
    m = source_mask(img.mean(axis=2), k=mask_k, dilate=mask_dilate).astype(float)
    m = np.clip(ndimage.gaussian_filter(m, feather), 0.0, 1.0)
    blended = m[..., None] * gentle + (1.0 - m[..., None]) * heavy
    return np.clip(blended, 0.0, 1.0)


def unsharp_luma(img01, amount=0.5, radius=2.0):
    """Unsharp-mask the luminance of a [0,1] RGB image; hue preserved.

    Sharpens detail without shifting color: sharpen the luma channel, then
    rescale RGB by the per-pixel luma ratio so chroma rides along. amount<=0
    returns a clipped copy (an explicit no-op for the --no-sharpen path).
    """
    from scipy import ndimage
    img = np.clip(img01, 0.0, 1.0).astype(np.float64)
    if amount <= 0:
        return img.copy()
    lum = img.mean(axis=2)
    blur = ndimage.gaussian_filter(lum, radius)
    sharp = np.clip(lum + amount * (lum - blur), 0.0, 1.0)
    ratio = np.divide(sharp, lum, out=np.ones_like(lum), where=lum > 1e-6)
    return np.clip(img * ratio[..., None], 0.0, 1.0)


# ---------- promoted step logic (background extraction, linked stretch) ----------

def _poly_design(x, y, degree):
    """Columns for all terms x^i y^j with i+j <= degree."""
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=-1)


def background_model(chan, degree=3, sample=12):
    """Fit a low-order 2D polynomial to non-source pixels of one channel.

    Returns (model (H,W), source_fraction). Ported verbatim from step 02.
    """
    h, w = chan.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx / (w - 1)) * 2 - 1
    yn = (yy / (h - 1)) * 2 - 1
    mask = source_mask(chan)
    bg = ~mask
    A = _poly_design(xn[bg][::sample], yn[bg][::sample], degree)
    coef, *_ = np.linalg.lstsq(A, chan[bg][::sample], rcond=None)
    full = _poly_design(xn.ravel(), yn.ravel(), degree) @ coef
    return full.reshape(h, w), float(mask.mean())


def linked_stretch(img01, target_bg=0.18, shadows_clip=-1.8):
    """Linked midtones-transfer stretch using global robust stats. Returns [0,1].

    Ported verbatim from step 04.
    """
    med = np.median(img01)
    mad = np.median(np.abs(img01 - med)) * 1.4826
    black = np.clip(med + shadows_clip * mad, 0.0, 1.0)
    scaled = np.clip((img01 - black) / (1.0 - black), 0.0, 1.0)
    m_shift = (med - black) / (1.0 - black) if (1.0 - black) > 0 else med
    return _mtf(scaled, _midtone(m_shift, target_bg))
