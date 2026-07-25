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
