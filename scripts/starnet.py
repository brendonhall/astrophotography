"""Wrapper around the StarNet2 CLI for scripted star removal.

Isolates the one external dependency (the StarNet2 binary) behind a small,
mockable interface -- mirrors how pcc.py isolates Gaia/astroquery. See
docs/superpowers/specs/2026-07-25-starless-galaxy-workflow-design.md.
"""
import os
import subprocess
import tempfile
import numpy as np
from astropy.io import fits

DEFAULT_BINARY = os.path.expanduser("~/StarNet2/starnet2")
MIN_SIZE = 512


class StarNetError(Exception):
    pass


def find_binary(binary=None):
    """Return a runnable StarNet2 binary path, or raise StarNetError.

    Resolution order: explicit `binary` arg, then $STARNET2_CLI, then the
    default ~/StarNet2/starnet2.
    """
    cand = binary or os.environ.get("STARNET2_CLI") or DEFAULT_BINARY
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    raise StarNetError(
        f"StarNet2 CLI not found or not executable at '{cand}'.\n"
        "Apple Silicon install: download\n"
        "  https://download.starnetastro.com/"
        "starnet2_macos-arm64_2.5.4-0214_COREML_arm64_cli.zip\n"
        "unzip to ~/StarNet2/ (keep 'starnet2' beside 'StarNet2_weights.mlpackage'),\n"
        "then 'xattr -dr com.apple.quarantine ~/StarNet2'. Or set $STARNET2_CLI."
    )


def _write_fits(path, img01):
    arr = np.moveaxis(np.clip(img01, 0.0, 1.0).astype(np.float32), -1, 0)  # (3,H,W)
    fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)


def _read_fits(path):
    data = fits.getdata(path).astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W) -> (H,W,3)
    elif data.ndim == 2:
        # StarNet2 collapses to mono FITS when input R==G==B; broadcast back
        # to (H,W,3) so callers always get the documented shape.
        data = np.stack([data] * 3, axis=-1)
    return np.clip(data, 0.0, 1.0)


def remove_stars(img01, binary=None, stride=256, tmpdir=None):
    """Split a [0,1] RGB image into (starless, stars) via the StarNet2 CLI.

    `stars` is StarNet2's --unscreen layer (the screen-blend inverse of
    starless), so astrolib.screen(starless, stars) reconstructs the input.
    Raises StarNetError on any failure.
    """
    img = np.asarray(img01)
    if img.ndim != 3 or img.shape[2] != 3:
        raise StarNetError(f"expected (H,W,3) RGB image, got shape {img.shape}")
    if min(img.shape[0], img.shape[1]) < MIN_SIZE:
        raise StarNetError(
            f"image {img.shape[1]}x{img.shape[0]} is smaller than StarNet2's "
            f"{MIN_SIZE}x{MIN_SIZE} minimum")
    exe = find_binary(binary)
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        inp = os.path.join(td, "in.fits")
        starless_p = os.path.join(td, "starless.fits")
        stars_p = os.path.join(td, "stars.fits")
        _write_fits(inp, img)
        argv = [exe, "-i", inp, "-o", starless_p, "-n", stars_p,
                "-s", str(int(stride)), "-q"]
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise StarNetError(
                f"StarNet2 failed (exit {proc.returncode}): "
                f"{(proc.stderr or '').strip()[-500:]}")
        if not (os.path.exists(starless_p) and os.path.exists(stars_p)):
            raise StarNetError("StarNet2 produced no output files")
        return _read_fits(starless_p), _read_fits(stars_p)
