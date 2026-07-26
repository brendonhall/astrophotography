"""Stage registry: discover stage types and their JSON schemas (GUI palette)."""
from __future__ import annotations

_REGISTRY: dict = {}


def register(cls):
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate stage id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def get(stage_id):
    return _REGISTRY[stage_id]


def list_stages():
    return [c.schema() for c in _REGISTRY.values()]
