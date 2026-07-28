"""Modular pipeline stages. Import a submodule (or the package) to register stages."""
from __future__ import annotations
from .image import Image, Space                      # noqa: F401
from .base import Param, Port, Stage, StageError     # noqa: F401
from .registry import register, get, list_stages     # noqa: F401


def _autoload():
    import importlib
    import pkgutil
    skip = {"base", "image", "io", "registry"}
    for m in pkgutil.iter_modules(__path__):
        if m.name not in skip:
            importlib.import_module(f"{__name__}.{m.name}")


_autoload()
