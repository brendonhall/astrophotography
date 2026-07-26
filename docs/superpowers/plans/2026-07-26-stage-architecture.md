# Stage Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each pipeline step into a self-describing, connectable "stage" with a typed, introspectable parameter schema and named input/output ports, over the existing numeric core — the foundation for a future node-editor GUI.

**Architecture:** New `scripts/stages/` package: an `Image` payload (pixels + FITS header/WCS + `space` tag), a `Stage` base with `Param`/`Port` descriptors and a `run()`→`apply()` contract, and a registry. Each stage is a thin declarative wrapper over `astrolib`/`pcc`/`starnet`. Numbered scripts become thin shims so the CLI keeps working. WCS is carried through the payload (crop shifts CRPIX), removing the color step's original-stack side input.

**Tech Stack:** Python 3.9 (`.venv`), numpy, astropy, scipy, scikit-image, PIL; stdlib `dataclasses`/`enum` only (no new deps). pytest.

## Global Constraints

- **Python 3.9 compatible.** The venv interpreter is 3.9.6. Every new module starts with `from __future__ import annotations` so `X | None` / `list[Port]` type hints are legal. Never evaluate PEP-604 unions at runtime.
- **No new dependencies.** stdlib `dataclasses`/`enum` only; no pydantic.
- **Data conventions:** payload pixels are `(H,W,3)` float32. `Space.LINEAR_ADU` = ~0..65535 ADU; `Space.NONLINEAR` = `[0,1]`. On disk FITS is `(3,H,W)`.
- **Preserve WCS:** stage IO writes the FULL header (+ a `PIPESPCE` space card), not a whitelist. Crop shifts `CRPIX1`/`CRPIX2`.
- **Stages are thin wrappers:** call the existing pure functions in `astrolib`/`pcc`/`starnet`; do not reimplement numeric logic.
- **Never overwrite outputs:** the versioned `output/<name>_<label>` convention (run_pipeline.sh) stays.
- **Tests stay offline** and dependency-free (mock `pcc`/`starnet`); the one guarded real-binary StarNet2 test stays `skipif`.
- Run everything with `.venv/bin/python`; `tests/conftest.py` already puts `scripts/` on `sys.path`, so `import stages`, `import astrolib` resolve.
- Existing numeric functions to reuse: `astrolib.finish/screen/masked_denoise/unsharp_luma/source_mask/save_preview/_mtf/_midtone`, `pcc.photometric_calibration/apply_gains/save_diagnostic/PCCError`, `starnet.remove_stars/MIN_SIZE/StarNetError`.

---

### Task 1: Core payload, Stage base, registry

**Files:**
- Create: `scripts/stages/image.py`, `scripts/stages/base.py`, `scripts/stages/registry.py`, `scripts/stages/__init__.py`
- Test: `tests/test_stages_core.py`

**Interfaces:**
- Produces: `stages.image.Space` (enum: `LINEAR_ADU="linear-adu"`, `NONLINEAR="nonlinear"`); `stages.image.Image(pixels, space, header=None)` with `.shape`, `.wcs`, `.replace(pixels=,space=,header=)`; `stages.base.Param(name,type,default,label,help,min,max,step,choices,unit)` with `.coerce(value)`; `stages.base.Port(name,space=None,required=True,help="")`; `stages.base.Stage` with classmethods `schema()`, `coerce_params(params)`, instance `check(inputs,params)`, `run(inputs,params=None)`, `apply(inputs,params)`; `stages.base.StageError`; `stages.registry.register/get/list_stages`; package re-exports these and auto-imports stage submodules.

- [ ] **Step 1: Write the failing test** — `tests/test_stages_core.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stages_core.py -v` → FAIL (`No module named 'stages'`).

- [ ] **Step 3: Implement** — create the four files:

`scripts/stages/image.py`:
```python
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
```

`scripts/stages/base.py`:
```python
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
```

`scripts/stages/registry.py`:
```python
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
```

`scripts/stages/__init__.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stages_core.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/image.py scripts/stages/base.py scripts/stages/registry.py scripts/stages/__init__.py tests/test_stages_core.py
git commit -m "Add stages core: Image payload, Stage/Param/Port, registry"
```

---

### Task 2: Stage FITS IO with WCS preservation

**Files:**
- Create: `scripts/stages/io.py`
- Test: `tests/test_stages_io.py`

**Interfaces:**
- Consumes: `Image`, `Space` (Task 1).
- Produces: `stages.io.save_fits(path, img)`, `stages.io.load_fits(path, space=None) -> Image`, `stages.io.crop_header(hdr, margin) -> Header|None`.

- [ ] **Step 1: Write the failing test** — `tests/test_stages_io.py`:

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stages.image import Image, Space
from stages.io import save_fits, load_fits, crop_header


def _wcs_header():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 100.0, 120.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_save_load_roundtrips_wcs_and_space(tmp_path):
    img = Image(np.random.rand(64, 48, 3).astype(np.float32) * 1000,
                Space.LINEAR_ADU, _wcs_header())
    p = str(tmp_path / "x.fit")
    save_fits(p, img)
    back = load_fits(p)
    assert back.space is Space.LINEAR_ADU          # from PIPESPCE
    assert back.pixels.shape == (64, 48, 3)
    assert back.wcs is not None and back.wcs.has_celestial


def test_nonlinear_not_rescaled(tmp_path):
    img = Image(np.full((8, 8, 3), 0.5, np.float32), Space.NONLINEAR)
    p = str(tmp_path / "n.fit")
    save_fits(p, img)
    back = load_fits(p)
    assert back.space is Space.NONLINEAR
    assert np.allclose(back.pixels, 0.5)           # NOT scaled by 65535


def test_crop_header_shifts_crpix_consistently():
    h = _wcs_header()
    m = 40
    h2 = crop_header(h, m)
    w1, w2 = WCS(h, naxis=2).celestial, WCS(h2, naxis=2).celestial
    # a sky point at old pixel (100,120) sits at new pixel (100-m,120-m)
    sky = w1.pixel_to_world(99, 119)               # 0-based -> CRPIX ref
    x2, y2 = w2.world_to_pixel(sky)
    assert abs(x2 - (99 - m)) < 1e-6 and abs(y2 - (119 - m)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stages_io.py -v` → FAIL (`No module named 'stages.io'`).

- [ ] **Step 3: Implement** — `scripts/stages/io.py`:
```python
"""FITS IO for stage payloads: preserves the FULL header (WCS) + a space tag."""
from __future__ import annotations
import numpy as np
from astropy.io import fits
from .image import Image, Space


def save_fits(path, img: Image):
    arr = np.moveaxis(np.asarray(img.pixels, dtype=np.float32), -1, 0)  # (H,W,3)->(3,H,W)
    hdr = img.header.copy() if img.header is not None else fits.Header()
    hdr["PIPESPCE"] = (str(img.space), "pipeline color space")
    fits.PrimaryHDU(data=arr, header=hdr).writeto(path, overwrite=True)


def load_fits(path, space: Space | None = None) -> Image:
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W)->(H,W,3)
    sp = space or Space(hdr.get("PIPESPCE", Space.LINEAR_ADU.value))
    if sp is Space.LINEAR_ADU and data.size and float(np.nanmax(data)) <= 1.5:
        data = data * 65535.0            # Siril [0,1] master -> ADU (matches astrolib.load)
    return Image(data, sp, hdr)


def crop_header(hdr, margin):
    """Shift CRPIX for a margin trimmed off top/left. SIP coeffs are CRPIX-relative
    so they follow; for higher-order distortion see astropy.wcs.WCS.slice."""
    if hdr is None:
        return None
    h = hdr.copy()
    for k in ("CRPIX1", "CRPIX2"):
        if k in h:
            h[k] = float(h[k]) - margin
    return h
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stages_io.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/io.py tests/test_stages_io.py
git commit -m "Add stage FITS IO preserving full WCS header + space tag"
```

---

### Task 3: Promote background + stretch math into astrolib

**Files:**
- Modify: `scripts/astrolib.py` (append functions)
- Test: `tests/test_astrolib_promoted.py`

**Interfaces:**
- Produces: `astrolib.background_model(chan, degree=3, sample=12) -> (model (H,W), src_frac float)`; `astrolib._poly_design(x, y, degree)`; `astrolib.linked_stretch(img01, target_bg=0.18, shadows_clip=-1.8) -> stretched [0,1]`.

- [ ] **Step 1: Write the failing test** — `tests/test_astrolib_promoted.py`:

```python
import numpy as np
import astrolib as al


def test_background_model_recovers_smooth_gradient():
    h, w = 80, 100
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx / (w - 1)) * 2 - 1
    yn = (yy / (h - 1)) * 2 - 1
    truth = 500 + 120 * xn + 80 * yn + 30 * xn * yn   # smooth low-order background
    model, frac = al.background_model(truth.astype(np.float32), degree=3, sample=4)
    assert np.allclose(model, truth, atol=1.0)        # fit recovers the gradient
    assert 0.0 <= frac <= 1.0


def test_linked_stretch_matches_reference_and_lifts_median():
    rng = np.random.RandomState(0)
    img01 = np.clip(0.05 + 0.02 * rng.rand(40, 40, 3), 0, 1).astype(np.float64)
    # reference formula (verbatim from the old step 04)
    med = np.median(img01); mad = np.median(np.abs(img01 - med)) * 1.4826
    black = np.clip(med + (-1.8) * mad, 0.0, 1.0)
    scaled = np.clip((img01 - black) / (1.0 - black), 0.0, 1.0)
    m_shift = (med - black) / (1.0 - black)
    ref = al._mtf(scaled, al._midtone(m_shift, 0.18))
    got = al.linked_stretch(img01, target_bg=0.18, shadows_clip=-1.8)
    assert np.allclose(got, ref)
    assert got.min() >= 0.0 and got.max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_astrolib_promoted.py -v` → FAIL (`no attribute 'background_model'`).

- [ ] **Step 3: Implement** — append to `scripts/astrolib.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_astrolib_promoted.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/astrolib.py tests/test_astrolib_promoted.py
git commit -m "Promote background_model + linked_stretch into astrolib"
```

---

### Task 4: Crop stage (first real stage)

**Files:**
- Create: `scripts/stages/geometry.py`
- Test: `tests/test_stage_geometry.py`

**Interfaces:**
- Consumes: `Stage/Param/Port` (Task 1), `crop_header` (Task 2), `register` (Task 1).
- Produces: `stages.geometry.CropStage` (id `crop`; `image`→`image`; param `margin`).

- [ ] **Step 1: Write the failing test** — `tests/test_stage_geometry.py`:
```python
import numpy as np
from astropy.io import fits
import stages
from stages.image import Image, Space
from stages.geometry import CropStage


def _hdr():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 100.0, 120.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_crop_trims_and_shifts_wcs():
    img = Image(np.arange(20 * 24 * 3, dtype=np.float32).reshape(20, 24, 3),
                Space.LINEAR_ADU, _hdr())
    out = CropStage().run({"image": img}, {"margin": 5})["image"]
    assert out.pixels.shape == (10, 14, 3)
    assert out.header["CRPIX1"] == 95.0 and out.header["CRPIX2"] == 115.0


def test_crop_registered():
    assert stages.get("crop") is CropStage
    assert any(s["id"] == "crop" for s in stages.list_stages())
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_geometry.py -v` → FAIL (`No module named 'stages.geometry'`).

- [ ] **Step 3: Implement** — `scripts/stages/geometry.py`:
```python
"""Geometry stages (crop)."""
from __future__ import annotations
from .base import Stage, Param, Port
from .io import crop_header
from .registry import register


@register
class CropStage(Stage):
    id = "crop"
    label = "Crop border"
    description = "Trim a uniform margin off every side; shifts WCS CRPIX to match."
    INPUTS = [Port("image")]
    OUTPUTS = [Port("image")]
    PARAMS = [Param("margin", "int", 40, "Margin", "Pixels trimmed per side",
                    min=0, max=2000, step=1, unit="px")]

    def apply(self, inputs, params):
        img = inputs["image"]
        m = params["margin"]
        h, w = img.pixels.shape[:2]
        px = img.pixels[m:h - m, m:w - m, :]
        return {"image": img.replace(pixels=px, header=crop_header(img.header, m))}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_geometry.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/geometry.py tests/test_stage_geometry.py
git commit -m "Add CropStage (WCS-aware) + registry integration"
```

---

### Task 5: Background + stretch stages (linear domain)

**Files:**
- Create: `scripts/stages/background.py`, `scripts/stages/stretch.py`
- Test: `tests/test_stage_linear.py`

**Interfaces:**
- Consumes: `astrolib.background_model/linked_stretch` (Task 3), Stage/Param/Port.
- Produces: `stages.background.BackgroundExtractStage` (id `background_extract`; linear→linear; params degree/sample/pedestal); `stages.stretch.StretchStage` (id `stretch`; linear→**nonlinear**; params target_bg/shadows_clip).

- [ ] **Step 1: Write the failing test** — `tests/test_stage_linear.py`:
```python
import numpy as np
import astrolib as al
from stages.image import Image, Space
from stages.background import BackgroundExtractStage
from stages.stretch import StretchStage


def test_background_stage_matches_astrolib_per_channel():
    rng = np.random.RandomState(0)
    px = (1000 + 200 * rng.rand(40, 50, 3)).astype(np.float32)
    img = Image(px, Space.LINEAR_ADU)
    out = BackgroundExtractStage().run({"image": img},
        {"degree": 3, "sample": 4, "pedestal": 0.10})["image"]
    ped = 0.10 * 65535.0
    exp = np.empty_like(px)
    for c in range(3):
        model, _ = al.background_model(px[..., c], 3, 4)
        exp[..., c] = px[..., c] - model + ped
    assert np.allclose(out.pixels, exp)
    assert out.space is Space.LINEAR_ADU


def test_stretch_stage_matches_astrolib_and_flips_space():
    rng = np.random.RandomState(1)
    px = (3000 + 500 * rng.rand(32, 32, 3)).astype(np.float32)
    img = Image(px, Space.LINEAR_ADU)
    out = StretchStage().run({"image": img}, {"target_bg": 0.18, "shadows_clip": -1.8})["image"]
    exp = al.linked_stretch(np.clip(px / 65535.0, 0, 1), 0.18, -1.8)
    assert np.allclose(out.pixels, exp)
    assert out.space is Space.NONLINEAR
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_linear.py -v` → FAIL (missing modules).

- [ ] **Step 3: Implement** — `scripts/stages/background.py`:
```python
"""Background/gradient extraction stage."""
from __future__ import annotations
import numpy as np
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class BackgroundExtractStage(Stage):
    id = "background_extract"
    label = "Background extraction"
    description = "Per-channel low-order polynomial gradient removal; re-adds a neutral pedestal."
    INPUTS = [Port("image", space=Space.LINEAR_ADU)]
    OUTPUTS = [Port("image", space=Space.LINEAR_ADU)]
    PARAMS = [
        Param("degree", "int", 3, "Polynomial degree", min=1, max=6, step=1),
        Param("sample", "int", 12, "Pixel subsampling", min=1, max=64, step=1),
        Param("pedestal", "float", 0.10, "Pedestal", "Re-added level (fraction of 65535)",
              min=0, max=1, step=0.01),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        ped = params["pedestal"] * 65535.0
        out = np.empty_like(img.pixels)
        for c in range(3):
            model, _ = al.background_model(img.pixels[..., c], params["degree"], params["sample"])
            out[..., c] = img.pixels[..., c] - model + ped
        return {"image": img.replace(pixels=out)}
```

`scripts/stages/stretch.py`:
```python
"""Linear -> nonlinear stretch stage."""
from __future__ import annotations
import numpy as np
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class StretchStage(Stage):
    id = "stretch"
    label = "Stretch (linear -> nonlinear)"
    description = "Linked midtones-transfer stretch; converts linear ADU to nonlinear [0,1]."
    INPUTS = [Port("image", space=Space.LINEAR_ADU)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("target_bg", "float", 0.18, "Target sky", min=0.01, max=0.9, step=0.01),
        Param("shadows_clip", "float", -1.8, "Shadows clip (MAD)", min=-5, max=0, step=0.1),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        img01 = np.clip(img.pixels / 65535.0, 0, 1)
        stretched = al.linked_stretch(img01, target_bg=params["target_bg"],
                                      shadows_clip=params["shadows_clip"])
        return {"image": img.replace(pixels=stretched, space=Space.NONLINEAR)}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_linear.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/background.py scripts/stages/stretch.py tests/test_stage_linear.py
git commit -m "Add BackgroundExtract + Stretch stages"
```

---

### Task 6: Finish + denoise stages (nonlinear pixel ops)

**Files:**
- Create: `scripts/stages/finish.py`, `scripts/stages/denoise.py`
- Test: `tests/test_stage_nonlinear.py`

**Interfaces:**
- Consumes: `astrolib.finish/masked_denoise/unsharp_luma`.
- Produces: `stages.finish.FinishStage` (id `finish`), `stages.finish.SaturateStage` (id `saturate`), `stages.denoise.MaskedDenoiseStage` (id `masked_denoise`), `stages.denoise.UnsharpLumaStage` (id `unsharp_luma`).

- [ ] **Step 1: Write the failing test** — `tests/test_stage_nonlinear.py`:
```python
import numpy as np
import pytest
import astrolib as al
from stages.image import Image, Space
from stages.base import StageError
from stages.finish import FinishStage, SaturateStage
from stages.denoise import MaskedDenoiseStage, UnsharpLumaStage


def _nl(seed=0):
    rng = np.random.RandomState(seed)
    return Image(np.clip(rng.rand(40, 40, 3), 0, 1).astype(np.float64), Space.NONLINEAR)


def test_finish_stage_matches_astrolib():
    img = _nl()
    out = FinishStage().run({"image": img},
        {"saturation": 1.20, "luma_denoise": 0.012, "chroma_denoise": 4.0, "scnr": True})["image"]
    exp = al.finish(img.pixels, saturation=1.20, luma_denoise=0.012, chroma_denoise=4.0, scnr=True)
    assert np.allclose(out.pixels, exp)


def test_saturate_is_saturation_only():
    img = _nl(2)
    out = SaturateStage().run({"image": img}, {"saturation": 1.3})["image"]
    exp = al.finish(img.pixels, saturation=1.3, luma_denoise=0, chroma_denoise=0, scnr=False)
    assert np.allclose(out.pixels, exp)


def test_masked_denoise_matches_and_requires_nonlinear():
    img = _nl(3)
    out = MaskedDenoiseStage().run({"image": img},
        {"bg_luma": 0.06, "bg_chroma": 10.0, "gal_luma": 0.010, "gal_chroma": 3.0, "feather": 25.0})["image"]
    exp = al.masked_denoise(img.pixels, bg_luma=0.06, bg_chroma=10.0, gal_luma=0.010,
                            gal_chroma=3.0, feather=25.0)
    assert np.allclose(out.pixels, exp)
    linear = Image(img.pixels, Space.LINEAR_ADU)
    with pytest.raises(StageError):
        MaskedDenoiseStage().run({"image": linear}, {})


def test_unsharp_matches_astrolib():
    img = _nl(4)
    out = UnsharpLumaStage().run({"image": img}, {"amount": 0.5, "radius": 2.0})["image"]
    exp = al.unsharp_luma(img.pixels, amount=0.5, radius=2.0)
    assert np.allclose(out.pixels, exp)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_nonlinear.py -v` → FAIL (missing modules).

- [ ] **Step 3: Implement** — `scripts/stages/finish.py`:
```python
"""Finishing stages (full finish, saturation-only)."""
from __future__ import annotations
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class FinishStage(Stage):
    id = "finish"
    label = "Finish"
    description = "SCNR green, luma + chroma denoise, saturation on a nonlinear image."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("saturation", "float", 1.20, "Saturation", min=0, max=3, step=0.05),
        Param("luma_denoise", "float", 0.012, "Luma denoise", min=0, max=1, step=0.001),
        Param("chroma_denoise", "float", 4.0, "Chroma sigma", min=0, max=50, step=0.5, unit="px"),
        Param("scnr", "bool", True, "SCNR green"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.finish(img.pixels, saturation=params["saturation"],
                        luma_denoise=params["luma_denoise"],
                        chroma_denoise=params["chroma_denoise"], scnr=params["scnr"])
        return {"image": img.replace(pixels=out)}


@register
class SaturateStage(Stage):
    id = "saturate"
    label = "Saturation"
    description = "HSV saturation multiplier only (no denoise/SCNR)."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [Param("saturation", "float", 1.20, "Saturation", min=0, max=3, step=0.05)]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.finish(img.pixels, saturation=params["saturation"],
                        luma_denoise=0, chroma_denoise=0, scnr=False)
        return {"image": img.replace(pixels=out)}
```

`scripts/stages/denoise.py`:
```python
"""Denoise / sharpen stages."""
from __future__ import annotations
import astrolib as al
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class MaskedDenoiseStage(Stage):
    id = "masked_denoise"
    label = "Background-aware denoise"
    description = "Denoise sky hard, galaxy gently, blended by a feathered source mask."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = [
        Param("bg_luma", "float", 0.06, "BG luma", min=0, max=1, step=0.005),
        Param("bg_chroma", "float", 10.0, "BG chroma", min=0, max=50, step=0.5, unit="px"),
        Param("gal_luma", "float", 0.010, "Galaxy luma", min=0, max=1, step=0.005),
        Param("gal_chroma", "float", 3.0, "Galaxy chroma", min=0, max=50, step=0.5, unit="px"),
        Param("feather", "float", 25.0, "Mask feather", min=0, max=200, step=1, unit="px"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.masked_denoise(img.pixels, bg_luma=params["bg_luma"], bg_chroma=params["bg_chroma"],
                                gal_luma=params["gal_luma"], gal_chroma=params["gal_chroma"],
                                feather=params["feather"])
        return {"image": img.replace(pixels=out)}


@register
class UnsharpLumaStage(Stage):
    id = "unsharp_luma"
    label = "Unsharp (luminance)"
    description = "Unsharp-mask the luminance only; hue preserved."
    INPUTS = [Port("image")]
    OUTPUTS = [Port("image")]
    PARAMS = [
        Param("amount", "float", 0.0, "Amount", min=0, max=5, step=0.1),
        Param("radius", "float", 2.0, "Radius", min=0.1, max=20, step=0.1, unit="px"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        out = al.unsharp_luma(img.pixels, amount=params["amount"], radius=params["radius"])
        return {"image": img.replace(pixels=out)}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_nonlinear.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/finish.py scripts/stages/denoise.py tests/test_stage_nonlinear.py
git commit -m "Add Finish, Saturate, MaskedDenoise, UnsharpLuma stages"
```

---

### Task 7: Star stages (remove + recombine)

**Files:**
- Create: `scripts/stages/stars.py`
- Test: `tests/test_stage_stars.py`

**Interfaces:**
- Consumes: `starnet.remove_stars/MIN_SIZE`, `astrolib.screen`.
- Produces: `stages.stars.RemoveStarsStage` (id `remove_stars`; `image`→`starless`,`stars`; min-size precondition); `stages.stars.ScreenRecombineStage` (id `screen_recombine`; `base`,`overlay`→`image`).

- [ ] **Step 1: Write the failing test** — `tests/test_stage_stars.py`:
```python
import numpy as np
import pytest
import astrolib as al
import starnet
from stages.image import Image, Space
from stages.base import StageError
from stages.stars import RemoveStarsStage, ScreenRecombineStage


def test_remove_stars_maps_ports(monkeypatch):
    img = Image(np.clip(np.random.rand(512, 512, 3), 0, 1).astype(np.float32), Space.NONLINEAR)
    sl = img.pixels * 0.5
    st = np.zeros_like(img.pixels)
    monkeypatch.setattr(starnet, "remove_stars", lambda px, **k: (sl, st))
    out = RemoveStarsStage().run({"image": img}, {"stride": 128})
    assert np.allclose(out["starless"].pixels, sl)
    assert np.allclose(out["stars"].pixels, st)
    assert out["starless"].space is Space.NONLINEAR


def test_remove_stars_min_size_precondition():
    small = Image(np.zeros((100, 100, 3), np.float32), Space.NONLINEAR)
    with pytest.raises(StageError):
        RemoveStarsStage().run({"image": small}, {})


def test_screen_recombine_matches_astrolib():
    rng = np.random.RandomState(0)
    base = Image(np.clip(rng.rand(8, 8, 3), 0, 1), Space.NONLINEAR)
    overlay = Image(np.clip(rng.rand(8, 8, 3), 0, 1), Space.NONLINEAR)
    out = ScreenRecombineStage().run({"base": base, "overlay": overlay})["image"]
    assert np.allclose(out.pixels, al.screen(base.pixels, overlay.pixels))
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_stars.py -v` → FAIL (`No module named 'stages.stars'`).

- [ ] **Step 3: Implement** — `scripts/stages/stars.py`:
```python
"""Star-layer stages (StarNet2 removal, screen recombine)."""
from __future__ import annotations
import astrolib as al
import starnet
from .base import Stage, Param, Port
from .image import Space
from .registry import register


@register
class RemoveStarsStage(Stage):
    id = "remove_stars"
    label = "Remove stars (StarNet2)"
    description = "Split into starless + star layers; screen(starless, stars) reconstructs input."
    INPUTS = [Port("image", space=Space.NONLINEAR)]
    OUTPUTS = [Port("starless", space=Space.NONLINEAR), Port("stars", space=Space.NONLINEAR)]
    PARAMS = [Param("stride", "int", 256, "Tile stride", min=64, max=1024, step=64, unit="px")]

    def check(self, inputs, params):
        errs = super().check(inputs, params)
        img = inputs.get("image")
        if img is not None and min(img.pixels.shape[0], img.pixels.shape[1]) < starnet.MIN_SIZE:
            errs.append(f"image {img.pixels.shape[1]}x{img.pixels.shape[0]} "
                        f"below StarNet2 {starnet.MIN_SIZE} minimum")
        return errs

    def apply(self, inputs, params):
        img = inputs["image"]
        starless, stars = starnet.remove_stars(img.pixels, stride=params["stride"])
        return {"starless": img.replace(pixels=starless), "stars": img.replace(pixels=stars)}


@register
class ScreenRecombineStage(Stage):
    id = "screen_recombine"
    label = "Screen recombine"
    description = "Screen an overlay (e.g. stars) back over a base image."
    INPUTS = [Port("base", space=Space.NONLINEAR), Port("overlay", space=Space.NONLINEAR)]
    OUTPUTS = [Port("image", space=Space.NONLINEAR)]
    PARAMS = []

    def apply(self, inputs, params):
        base = inputs["base"]
        out = al.screen(base.pixels, inputs["overlay"].pixels)
        return {"image": base.replace(pixels=out)}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_stars.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/stars.py tests/test_stage_stars.py
git commit -m "Add RemoveStars + ScreenRecombine stages"
```

---

### Task 8: Color calibration stage

**Files:**
- Create: `scripts/stages/color.py`
- Test: `tests/test_stage_color.py`

**Interfaces:**
- Consumes: `pcc.photometric_calibration/apply_gains/save_diagnostic/PCCError`, `astrolib.source_mask`.
- Produces: `stages.color.ColorCalibrateStage` (id `color_calibrate`; `image`(+optional `reference`)→`image`; params ref_bp_rp/min_stars/no_pcc/diagnostic_path); module helpers `_bg_level`, `_neutralize`, `_gentle_white_balance` (ported from step 03).

- [ ] **Step 1: Write the failing test** — `tests/test_stage_color.py`:
```python
import numpy as np
import pcc
from astropy.io import fits
from stages.image import Image, Space
from stages.color import ColorCalibrateStage, _gentle_white_balance

_PED = 0.10 * 65535.0


def _linear(seed=0):
    rng = np.random.RandomState(seed)
    return Image((5000 + 400 * rng.rand(30, 30, 3)).astype(np.float64),
                 Space.LINEAR_ADU, fits.Header())


def test_no_pcc_uses_gentle_white_balance():
    img = _linear()
    out = ColorCalibrateStage().run({"image": img}, {"no_pcc": True})["image"]
    # neutralize then gentle-WB, computed directly
    from stages.color import _neutralize
    exp = _gentle_white_balance(_neutralize(img.pixels, _PED), _PED)
    assert np.allclose(out.pixels, exp)


def test_pcc_success_applies_gains(monkeypatch):
    img = _linear(1)
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.1, 1.0, 0.9), {"n_matched": 42}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda report, path: None)
    out = ColorCalibrateStage().run({"image": img}, {})["image"]
    from stages.color import _neutralize
    exp = pcc.apply_gains(_neutralize(img.pixels, _PED), (1.1, 1.0, 0.9), _PED)
    assert np.allclose(out.pixels, exp)


def test_pcc_error_falls_back(monkeypatch):
    img = _linear(2)
    def boom(px, hdr, **k):
        raise pcc.PCCError("no stars")
    monkeypatch.setattr(pcc, "photometric_calibration", boom)
    out = ColorCalibrateStage().run({"image": img}, {})["image"]
    from stages.color import _neutralize
    exp = _gentle_white_balance(_neutralize(img.pixels, _PED), _PED)
    assert np.allclose(out.pixels, exp)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_color.py -v` → FAIL (`No module named 'stages.color'`).

- [ ] **Step 3: Implement** — `scripts/stages/color.py`:
```python
"""Color-calibration stage: background neutralize + Gaia PCC (gentle-WB fallback)."""
from __future__ import annotations
import numpy as np
import astrolib as al
import pcc
from .base import Stage, Param, Port
from .image import Space
from .registry import register

_PEDESTAL = 0.10 * 65535.0


def _bg_level(chan):
    bg = chan[~al.source_mask(chan)]
    med = np.median(bg)
    for _ in range(3):
        s = bg.std()
        bg = bg[np.abs(bg - med) < 3 * s]
        med = np.median(bg)
    return med


def _neutralize(pixels, pedestal):
    out = np.empty_like(pixels)
    for c in range(3):
        out[..., c] = pixels[..., c] - _bg_level(pixels[..., c]) + pedestal
    return out


def _gentle_white_balance(img, pedestal, clamp=(0.85, 1.15)):
    lum = img.mean(axis=2)
    lo, hi = np.percentile(lum, 60), np.percentile(lum, 99)
    band = (lum > lo) & (lum < hi)
    means = np.array([img[..., c][band].mean() - pedestal for c in range(3)])
    gains = np.clip(means[1] / means, *clamp)
    out = img.copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    return out


@register
class ColorCalibrateStage(Stage):
    id = "color_calibrate"
    label = "Color calibration (PCC)"
    description = ("Neutralize background, then Gaia photometric color calibration; "
                   "falls back to gentle white balance.")
    INPUTS = [Port("image", space=Space.LINEAR_ADU),
              Port("reference", space=Space.LINEAR_ADU, required=False,
                   help="optional WCS-bearing frame to measure PCC on (default: 'image')")]
    OUTPUTS = [Port("image", space=Space.LINEAR_ADU)]
    PARAMS = [
        Param("ref_bp_rp", "float", 0.82, "White point (BP-RP)", min=-1, max=3, step=0.01),
        Param("min_stars", "int", 30, "Min matched stars", min=3, max=1000, step=1),
        Param("no_pcc", "bool", False, "Skip PCC (gentle WB only)"),
        Param("diagnostic_path", "str", "", "PCC diagnostic PNG path"),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        neutral = _neutralize(img.pixels, _PEDESTAL)
        ref = inputs.get("reference") or img
        if params["no_pcc"] or ref.header is None:
            return {"image": img.replace(pixels=_gentle_white_balance(neutral, _PEDESTAL))}
        try:
            gains, report = pcc.photometric_calibration(
                ref.pixels, ref.header, ref_bp_rp=params["ref_bp_rp"],
                min_stars=params["min_stars"])
            if params["diagnostic_path"]:
                pcc.save_diagnostic(report, params["diagnostic_path"])
            out = pcc.apply_gains(neutral, gains, _PEDESTAL)
        except pcc.PCCError:
            out = _gentle_white_balance(neutral, _PEDESTAL)
        except Exception:
            out = _gentle_white_balance(neutral, _PEDESTAL)
        return {"image": img.replace(pixels=np.asarray(out, dtype=img.pixels.dtype))}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_color.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/color.py tests/test_stage_color.py
git commit -m "Add ColorCalibrate stage (WCS from input, gentle-WB fallback)"
```

---

### Task 9: Export/preview sinks + registry completeness

**Files:**
- Create: `scripts/stages/export.py`
- Test: `tests/test_stage_export.py`, `tests/test_stages_registry.py`

**Interfaces:**
- Consumes: `astrolib.save_preview`.
- Produces: `stages.export.ExportImageStage` (id `export_image`; `image`→sink; param `out_base`); `stages.export.PreviewSink` (id `preview_sink`; `image`→sink; params out_path/stretch).

- [ ] **Step 1: Write the failing tests** — `tests/test_stage_export.py`:
```python
import os
import numpy as np
from stages.image import Image, Space
from stages.export import ExportImageStage, PreviewSink


def test_export_writes_three_files(tmp_path):
    img = Image(np.clip(np.random.rand(20, 20, 3), 0, 1).astype(np.float64), Space.NONLINEAR)
    base = str(tmp_path / "out")
    ExportImageStage().run({"image": img}, {"out_base": base})
    for suffix in (".tif", ".png", "_preview.png"):
        assert os.path.exists(base + suffix), suffix


def test_preview_sink_writes_png(tmp_path):
    img = Image(np.clip(np.random.rand(20, 20, 3), 0, 1).astype(np.float64), Space.NONLINEAR)
    p = str(tmp_path / "prev.png")
    PreviewSink().run({"image": img}, {"out_path": p, "stretch": False})
    assert os.path.exists(p)
```

`tests/test_stages_registry.py`:
```python
import json
import stages


def test_all_expected_stages_registered_and_serializable():
    ids = {s["id"] for s in stages.list_stages()}
    expected = {"crop", "background_extract", "color_calibrate", "stretch", "finish",
                "saturate", "masked_denoise", "unsharp_luma", "remove_stars",
                "screen_recombine", "export_image", "preview_sink"}
    assert expected <= ids
    json.dumps(stages.list_stages())     # every schema is JSON-serializable
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_stage_export.py tests/test_stages_registry.py -v` → FAIL.

- [ ] **Step 3: Implement** — `scripts/stages/export.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_export.py tests/test_stages_registry.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/export.py tests/test_stage_export.py tests/test_stages_registry.py
git commit -m "Add Export/Preview sink stages + registry completeness test"
```

---

### Task 10: Shims for crop/background/stretch/finish steps

**Files:**
- Modify: `scripts/01_crop.py`, `scripts/02_background.py`, `scripts/04_stretch.py`, `scripts/05_finish.py`
- Test: `tests/test_shims_linear.py`

**Interfaces:**
- Consumes: all stages + `stages.io.load_fits/save_fits`.
- Produces: numbered `main()` shims with unchanged CLI behavior; intermediates now carry WCS.

- [ ] **Step 1: Write the failing test** — `tests/test_shims_linear.py`:
```python
import importlib.util
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stages.io import load_fits, save_fits
from stages.image import Image, Space

SC = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(SC, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hdr():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 200.0, 200.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    return h


def test_crop_shim_preserves_wcs(tmp_path):
    src = str(tmp_path / "in.fit")
    save_fits(src, Image((1000 + np.random.rand(600, 600, 3) * 100).astype(np.float32),
                         Space.LINEAR_ADU, _hdr()))
    out = str(tmp_path / "crop.fit")
    _load("01_crop.py").main(src, out)
    back = load_fits(out)
    assert back.pixels.shape[0] < 600 and back.wcs is not None and back.wcs.has_celestial


def test_background_then_stretch_shims(tmp_path):
    src = str(tmp_path / "in.fit")
    save_fits(src, Image((1000 + np.random.rand(200, 200, 3) * 100).astype(np.float32),
                         Space.LINEAR_ADU, _hdr()))
    bg = str(tmp_path / "bg.fit")
    _load("02_background.py").main(src, bg)
    assert load_fits(bg).space is Space.LINEAR_ADU
    st = str(tmp_path / "st.fit")
    _load("04_stretch.py").main(bg, st)
    out = load_fits(st)
    assert out.space is Space.NONLINEAR and out.pixels.max() <= 1.0 + 1e-6


def test_finish_shim_writes_outputs(tmp_path):
    st = str(tmp_path / "st.fit")
    save_fits(st, Image(np.clip(np.random.rand(64, 64, 3), 0, 1).astype(np.float32),
                        Space.NONLINEAR, _hdr()))
    base = str(tmp_path / "final")
    _load("05_finish.py").main(st, base)
    for s in (".tif", ".png", "_preview.png"):
        assert os.path.exists(base + s)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_shims_linear.py -v` → FAIL (old scripts still use `al.load`/`al.save`; the WCS-preservation assertion fails).

- [ ] **Step 3: Implement the shims** —

`scripts/01_crop.py`:
```python
#!/usr/bin/env python3
"""Step 1 - Crop a thin border off the stack (shim over CropStage)."""
import sys
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.geometry import CropStage

MARGIN = 40

def main(infile, outfile, margin=MARGIN):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = CropStage().run({"image": img}, {"margin": margin})["image"]
    save_fits(outfile, out)
    print(f"cropped -> {out.pixels.shape}, wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

`scripts/02_background.py`:
```python
#!/usr/bin/env python3
"""Step 2 - Background/gradient removal (shim over BackgroundExtractStage)."""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.background import BackgroundExtractStage

DEGREE, SAMPLE, PEDESTAL = 3, 12, 0.10

def main(infile, outfile):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = BackgroundExtractStage().run(
        {"image": img}, {"degree": DEGREE, "sample": SAMPLE, "pedestal": PEDESTAL})["image"]
    save_fits(outfile, out)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out.pixels)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

`scripts/04_stretch.py`:
```python
#!/usr/bin/env python3
"""Step 4 - Linked stretch, linear -> nonlinear (shim over StretchStage)."""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.stretch import StretchStage

TARGET_BG, SHADOWS_CLIP = 0.18, -1.8

def main(infile, outfile):
    img = load_fits(infile, Space.LINEAR_ADU)
    out = StretchStage().run(
        {"image": img}, {"target_bg": TARGET_BG, "shadows_clip": SHADOWS_CLIP})["image"]
    save_fits(outfile, out)  # nonlinear, stored as [0,1] + PIPESPCE
    al.save_preview(outfile.replace(".fit", "_preview.png"), img01=out.pixels, stretch=False)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

`scripts/05_finish.py`:
```python
#!/usr/bin/env python3
"""Step 5 - Finish and export (shim over FinishStage + ExportImageStage)."""
import sys
from stages.image import Space
from stages.io import load_fits
from stages.finish import FinishStage
from stages.export import ExportImageStage

SATURATION, LUMA_DENOISE, CHROMA_DENOISE = 1.20, 0.012, 4.0

def main(infile, outfile_base):
    img = load_fits(infile, Space.NONLINEAR)
    finished = FinishStage().run({"image": img}, {
        "saturation": SATURATION, "luma_denoise": LUMA_DENOISE,
        "chroma_denoise": CHROMA_DENOISE, "scnr": True})["image"]
    ExportImageStage().run({"image": finished}, {"out_base": outfile_base})

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_shims_linear.py -v` → PASS. Then full suite: `.venv/bin/python -m pytest -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add scripts/01_crop.py scripts/02_background.py scripts/04_stretch.py scripts/05_finish.py tests/test_shims_linear.py
git commit -m "Refactor crop/background/stretch/finish steps into stage shims"
```

---

### Task 11: Color shim + run_pipeline update

**Files:**
- Modify: `scripts/03_color.py`, `scripts/run_pipeline.sh`
- Test: `tests/test_shims_color.py`

**Interfaces:**
- Consumes: `ColorCalibrateStage`, `stages.io`.
- Produces: `03_color.py` shim reading WCS from its input (`--original` becomes an optional `reference` override); run_pipeline no longer passes `--original`.

- [ ] **Step 1: Write the failing test** — `tests/test_shims_color.py`:
```python
import importlib.util
import os
import numpy as np
import pcc
from astropy.io import fits
from stages.io import load_fits, save_fits
from stages.image import Image, Space

SC = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location("color_shim", os.path.join(SC, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_color_shim_reads_wcs_from_input_and_writes(tmp_path, monkeypatch):
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 50.0, 50.0
    h["CRVAL1"], h["CRVAL2"] = 210.0, 54.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -1e-4, 0.0, 0.0, 1e-4
    src = str(tmp_path / "bg.fit")
    save_fits(src, Image((5000 + np.random.rand(60, 60, 3) * 300).astype(np.float32),
                         Space.LINEAR_ADU, h))
    seen = {}
    def fake_pcc(px, hdr, **k):
        seen["has_wcs"] = ("CRPIX1" in hdr)
        return ((1.05, 1.0, 0.95), {"n_matched": 40})
    monkeypatch.setattr(pcc, "photometric_calibration", fake_pcc)
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, p: None)
    out = str(tmp_path / "color.fit")
    _load("03_color.py").main(src, out)
    assert seen["has_wcs"] is True                 # WCS came from the input, no --original
    assert load_fits(out).space is Space.LINEAR_ADU
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_shims_color.py -v` → FAIL (old 03 requires `--original` for WCS).

- [ ] **Step 3: Implement** — `scripts/03_color.py`:
```python
#!/usr/bin/env python3
"""Step 3 - Color calibration (shim over ColorCalibrateStage).

WCS now travels in the payload, so PCC measures on this step's own input; the
optional --original arg becomes a 'reference' frame override.
"""
import sys
import astrolib as al
from stages.image import Space
from stages.io import load_fits, save_fits
from stages.color import ColorCalibrateStage

REF_BP_RP, MIN_STARS = 0.82, 30

def main(infile, outfile, original=None, no_pcc=False, diagnostic=None):
    img = load_fits(infile, Space.LINEAR_ADU)
    inputs = {"image": img}
    if original:
        inputs["reference"] = load_fits(original, Space.LINEAR_ADU)
    out = ColorCalibrateStage().run(inputs, {
        "ref_bp_rp": REF_BP_RP, "min_stars": MIN_STARS,
        "no_pcc": no_pcc, "diagnostic_path": diagnostic or ""})["image"]
    save_fits(outfile, out)
    al.save_preview(outfile.replace(".fit", "_preview.png"), img_adu=out.pixels)
    print(f"wrote {outfile}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--original", help="optional WCS-bearing reference frame for PCC")
    ap.add_argument("--no-pcc", action="store_true")
    ap.add_argument("--diagnostic")
    a = ap.parse_args()
    main(a.infile, a.outfile, original=a.original, no_pcc=a.no_pcc, diagnostic=a.diagnostic)
```

Then edit `scripts/run_pipeline.sh` step 03 line to drop `--original` (WCS now travels in the payload). Change:
```bash
echo ">> 03 color";      "$PY" 03_color.py      "$WORK/02_bg.fit"   "$WORK/03_color.fit" --original "$IN" \
                            --diagnostic "$ROOT/output/${NAME}_${LABEL}_pcc_diagnostic.png"
```
to:
```bash
echo ">> 03 color";      "$PY" 03_color.py      "$WORK/02_bg.fit"   "$WORK/03_color.fit" \
                            --diagnostic "$ROOT/output/${NAME}_${LABEL}_pcc_diagnostic.png"
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_shims_color.py -v` → PASS; `bash -n scripts/run_pipeline.sh` → exit 0; full suite green.

- [ ] **Step 5: Commit**
```bash
git add scripts/03_color.py scripts/run_pipeline.sh tests/test_shims_color.py
git commit -m "Refactor color step into stage shim; drop --original (WCS in payload)"
```

---

### Task 12: Starless finish shim (decomposed stages)

**Files:**
- Modify: `scripts/05b_starless_finish.py`, `tests/test_starless_finish.py`
- Test: `tests/test_starless_finish.py` (rewritten)

**Interfaces:**
- Consumes: `RemoveStarsStage`, `UnsharpLumaStage`, `MaskedDenoiseStage`, `SaturateStage`, `ScreenRecombineStage`, `ExportImageStage`, `PreviewSink`, `stages.io.load_fits`, `starnet.StarNetError`.
- Produces: `05b` shim orchestrating the decomposed stages; still writes 5 files; still returns the result pixels; keeps `--stride`/`--no-sharpen`.

- [ ] **Step 1: Rewrite the test** — replace `tests/test_starless_finish.py` with:
```python
import importlib.util
import os
import numpy as np
import pytest
from astropy.io import fits
import starnet
import astrolib as al
from stages.io import save_fits
from stages.image import Image, Space

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "scripts", "05b_starless_finish.py")


def _load():
    spec = importlib.util.spec_from_file_location("starless_finish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_recombines_and_writes_five_files(tmp_path, monkeypatch):
    base = np.clip(np.random.RandomState(0).rand(512, 512, 3), 0, 1).astype(np.float32)
    infile = str(tmp_path / "in.fit")
    save_fits(infile, Image(base, Space.NONLINEAR, fits.Header()))
    starless = base * 0.5
    stars = np.zeros_like(base); stars[100, 100] = [1.0, 1.0, 1.0]
    monkeypatch.setattr(starnet, "remove_stars", lambda px, **k: (starless.copy(), stars.copy()))

    mod = _load()
    out_base = str(tmp_path / "out")
    result = mod.main(infile, out_base)

    # expected: masked_denoise -> saturation-only finish, then screen stars back in
    proc = al.finish(al.masked_denoise(starless, bg_luma=mod.BG_LUMA_DENOISE,
                                       bg_chroma=mod.BG_CHROMA_DENOISE, gal_luma=mod.GAL_LUMA_DENOISE,
                                       gal_chroma=mod.GAL_CHROMA_DENOISE, feather=mod.MASK_FEATHER),
                     saturation=mod.SATURATION, luma_denoise=0, chroma_denoise=0, scnr=False)
    expected = al.screen(proc, stars)
    assert np.allclose(result, expected, atol=1e-6)
    for suffix in (".tif", ".png", "_preview.png", "_starless.png", "_starlayer.png"):
        assert os.path.exists(out_base + suffix), suffix
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_starless_finish.py -v` → FAIL (old 05b defines `process_starless`/uses `al.load`; new expected pipeline differs).

- [ ] **Step 3: Implement** — `scripts/05b_starless_finish.py`:
```python
#!/usr/bin/env python3
"""Step 5b - Starless galaxy finish (shim orchestrating decomposed stages).

remove_stars -> [unsharp] -> masked_denoise -> saturate -> screen(stars) -> export,
with preview taps on the starless and star layers. See the stages package.
"""
import sys
from stages.image import Space
from stages.io import load_fits
from stages.stars import RemoveStarsStage, ScreenRecombineStage
from stages.denoise import UnsharpLumaStage, MaskedDenoiseStage
from stages.finish import SaturateStage
from stages.export import ExportImageStage, PreviewSink
import starnet

SHARPEN_AMOUNT, SHARPEN_RADIUS = 0.0, 2.0
BG_LUMA_DENOISE, BG_CHROMA_DENOISE = 0.06, 10.0
GAL_LUMA_DENOISE, GAL_CHROMA_DENOISE = 0.010, 3.0
MASK_FEATHER, SATURATION, STRIDE = 25.0, 1.20, 256

def main(infile, outfile_base, stride=STRIDE, no_sharpen=False):
    img = load_fits(infile, Space.NONLINEAR)
    split = RemoveStarsStage().run({"image": img}, {"stride": stride})
    starless, stars = split["starless"], split["stars"]

    proc = starless
    if not no_sharpen and SHARPEN_AMOUNT > 0:
        proc = UnsharpLumaStage().run({"image": proc},
            {"amount": SHARPEN_AMOUNT, "radius": SHARPEN_RADIUS})["image"]
    proc = MaskedDenoiseStage().run({"image": proc}, {
        "bg_luma": BG_LUMA_DENOISE, "bg_chroma": BG_CHROMA_DENOISE,
        "gal_luma": GAL_LUMA_DENOISE, "gal_chroma": GAL_CHROMA_DENOISE,
        "feather": MASK_FEATHER})["image"]
    proc = SaturateStage().run({"image": proc}, {"saturation": SATURATION})["image"]

    result = ScreenRecombineStage().run({"base": proc, "overlay": stars})["image"]
    ExportImageStage().run({"image": result}, {"out_base": outfile_base})
    PreviewSink().run({"image": proc}, {"out_path": outfile_base + "_starless.png", "stretch": False})
    PreviewSink().run({"image": stars}, {"out_path": outfile_base + "_starlayer.png", "stretch": False})
    return result.pixels

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--no-sharpen", action="store_true")
    a = ap.parse_args()
    try:
        main(a.infile, a.outfile, stride=a.stride, no_sharpen=a.no_sharpen)
    except starnet.StarNetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_starless_finish.py -v` → PASS; full suite `.venv/bin/python -m pytest -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add scripts/05b_starless_finish.py tests/test_starless_finish.py
git commit -m "Refactor starless finish into decomposed stage orchestration"
```

---

### Task 13: README + end-to-end verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the whole stages package + shims.

- [ ] **Step 1: Document the stages package** — add a README section describing `scripts/stages/` (Stage/Param/Port, `Space`, the registry, `stages.list_stages()` as the GUI palette contract) and noting the numbered scripts are now thin shims; WCS travels in the payload so step 03 no longer needs `--original`.

- [ ] **Step 2: Full suite** — `.venv/bin/python -m pytest -q` → all pass (existing + new stage/shim tests).

- [ ] **Step 3: Registry palette check** — `.venv/bin/python -c "import json,stages; print(json.dumps(stages.list_stages(), indent=2))"` → every stage lists ports + typed params; output is valid JSON.

- [ ] **Step 4: End-to-end parity (standard finish)** —
```bash
make run FITS="data/M101_restack_solved.fit" V=stages-check
```
Expected: steps 01–05 run; `output/M101_restack_solved_stages-check.{tif,png}` produced. Compare visually against the pre-refactor `output/M101_restack_solved_standard-v1.png` — equivalent. (PCC gains may differ slightly since PCC now measures on the background-subtracted input rather than the raw stack; star-match count and result should be comparable.)

- [ ] **Step 5: End-to-end parity (starless finish) + WCS check** —
```bash
make run-starless FITS="data/M101_restack_solved.fit" V=stages-check
.venv/bin/python -c "from astropy.io import fits; from astropy.wcs import WCS; h=fits.open('work/03_color.fit')[0].header; print('WCS ok:', WCS(h,naxis=2).celestial.has_celestial)"
```
Expected: `>> 05b starless finish` runs with StarNet2, produces `output/M101_restack_solved_stages-check.{tif,png}` + `_starless.png` + `_starlayer.png`; the WCS check prints `WCS ok: True` (proving intermediates now carry WCS and PCC ran without `--original`). Compare against `output/M101_restack_solved_starless-v2_*`.

- [ ] **Step 6: Commit**
```bash
git add README.md
git commit -m "Document stages package; verify end-to-end parity"
```

---

## Self-Review

**Spec coverage:**
- Stage/Param/Port abstraction + registry → Tasks 1, 9. ✓
- Image payload with WCS + space → Task 1; IO preserving WCS + crop CRPIX → Task 2. ✓
- astrolib promotions (background_model, linked_stretch) → Task 3. ✓
- Every step as a stage → crop (4), background/stretch (5), finish/saturate + denoise/unsharp (6), stars remove/recombine (7), color (8), export/preview sinks (9). ✓
- Starless monolith decomposed into connectable nodes → Tasks 7 + 12. ✓
- JSON-serializable param schema (GUI palette) → Tasks 1, 9 (`list_stages()` + `json.dumps`). ✓
- WCS preserved, color side-input removed → Tasks 2, 8, 11. ✓
- Backward-compat shims + run_pipeline working → Tasks 10, 11, 12. ✓
- Stdlib dataclasses, no pydantic, 3.9-compatible → Global Constraints + Task 1. ✓
- Tests offline (mock pcc/starnet) → Tasks 7, 8, 12. ✓
- Verification (make test, parity runs, registry JSON, WCS) → Task 13. ✓
- Deferred (graph runner, GUI, finer finish nodes) → not in plan (correct). ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `Image(pixels, space, header)`, `.replace(...)`, `Stage.run(inputs, params)->dict[port]`, `Param(name,type,default,...).coerce`, `Port(name,space,required,help)`, `load_fits/save_fits/crop_header`, `astrolib.background_model/linked_stretch`, stage ids used across shim tasks all match their defining tasks. ✓
