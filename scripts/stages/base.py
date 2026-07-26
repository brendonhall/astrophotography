"""Stage abstraction: typed params, named ports, run/apply contract."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
from .image import Space


class StageError(Exception):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    type: str                       # "float"|"int"|"bool"|"enum"|"str"
    default: Any
    label: str = ""
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple | None = None
    unit: str | None = None

    def coerce(self, value):
        if value is None:
            return self.default
        if self.type == "int":
            v = int(value)
        elif self.type == "float":
            v = float(value)
        elif self.type == "bool":
            v = bool(value)
        else:
            v = value
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.name}={v} below min {self.min}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.name}={v} above max {self.max}")
        if self.choices is not None and v not in self.choices:
            raise ValueError(f"{self.name}={v} not in {self.choices}")
        return v


@dataclass(frozen=True)
class Port:
    name: str
    space: Space | None = None
    required: bool = True
    help: str = ""


class Stage:
    id: str = ""
    label: str = ""
    description: str = ""
    INPUTS: list = []
    OUTPUTS: list = []
    PARAMS: list = []

    @classmethod
    def schema(cls) -> dict:
        def port_d(p):
            d = asdict(p)
            if d.get("space") is not None:
                d["space"] = str(d["space"])
            return d
        return {
            "id": cls.id, "label": cls.label, "description": cls.description,
            "inputs": [port_d(p) for p in cls.INPUTS],
            "outputs": [port_d(p) for p in cls.OUTPUTS],
            "params": [asdict(p) for p in cls.PARAMS],
        }

    @classmethod
    def coerce_params(cls, params) -> dict:
        params = params or {}
        return {p.name: p.coerce(params.get(p.name)) for p in cls.PARAMS}

    def check(self, inputs, params) -> list:
        errs = []
        for port in self.INPUTS:
            img = inputs.get(port.name)
            if img is None:
                if port.required:
                    errs.append(f"missing required input '{port.name}'")
                continue
            if port.space is not None and img.space is not port.space:
                errs.append(f"input '{port.name}' requires {port.space}, got {img.space}")
        return errs

    def run(self, inputs, params=None) -> dict:
        p = self.coerce_params(params)
        errs = self.check(inputs, p)
        if errs:
            raise StageError(f"{self.id or type(self).__name__}: " + "; ".join(errs))
        return self.apply(inputs, p)

    def apply(self, inputs, params) -> dict:
        raise NotImplementedError
