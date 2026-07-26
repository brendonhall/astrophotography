import json
import numpy as np
import pytest
import stages
from stages.image import Image, Space
from stages.base import Param, Port, Stage, StageError


def test_param_coerce_defaults_types_bounds():
    p = Param("x", "float", 1.0, min=0.0, max=2.0)
    assert p.coerce(None) == 1.0          # default
    assert p.coerce("1.5") == 1.5         # type coercion
    with pytest.raises(ValueError):
        p.coerce(3.0)                     # above max
    enum = Param("m", "enum", "a", choices=("a", "b"))
    with pytest.raises(ValueError):
        enum.coerce("z")


def test_image_replace_is_immutable_copy():
    img = Image(np.zeros((4, 4, 3), np.float32), Space.LINEAR_ADU)
    out = img.replace(space=Space.NONLINEAR)
    assert out.space is Space.NONLINEAR and img.space is Space.LINEAR_ADU
    assert out.wcs is None                # no header -> no wcs


class _Doubler(Stage):
    id = "doubler"; label = "Doubler"; description = "x2"
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [Param("k", "float", 2.0, min=0, max=10)]

    def apply(self, inputs, params):
        img = inputs["image"]
        return {"image": img.replace(pixels=img.pixels * params["k"])}


def test_stage_schema_is_json_serializable():
    s = json.dumps(_Doubler.schema())
    d = json.loads(s)
    assert d["id"] == "doubler"
    assert d["inputs"][0]["space"] == "nonlinear"
    assert d["params"][0]["name"] == "k"


def test_stage_run_coerces_checks_applies():
    img = Image(np.ones((2, 2, 3), np.float32), Space.NONLINEAR)
    out = _Doubler().run({"image": img}, {"k": 3})
    assert np.allclose(out["image"].pixels, 3.0)


def test_stage_run_rejects_missing_and_wrong_space():
    with pytest.raises(StageError):
        _Doubler().run({}, {})                       # missing required input
    linear = Image(np.ones((2, 2, 3), np.float32), Space.LINEAR_ADU)
    with pytest.raises(StageError):
        _Doubler().run({"image": linear}, {})        # wrong space


def test_registry_roundtrip():
    stages.register(_Doubler) if "doubler" not in [s["id"] for s in stages.list_stages()] else None
    assert stages.get("doubler") is _Doubler
