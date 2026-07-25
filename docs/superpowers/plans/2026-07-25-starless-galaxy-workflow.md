# Starless Galaxy Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in "starless galaxy" finishing path that removes stars with the StarNet2 CLI, sharpens/denoises the starless galaxy, and screens the stars back in.

**Architecture:** A thin wrapper module (`scripts/starnet.py`) isolates the StarNet2 CLI dependency (mirroring how `pcc.py` isolates Gaia/astroquery). A thin orchestrator (`scripts/05b_starless_finish.py`) drives the layer workflow using shared `astrolib` helpers. The existing `05_finish.py` is untouched; the new path is selected via a `--starless` flag on `run_pipeline.sh`.

**Tech Stack:** Python 3 (project `.venv`), numpy, astropy.io.fits, scipy.ndimage, scikit-image, PIL; StarNet2 CLI v2.5.4 (installed at `~/StarNet2/`); pytest.

## Global Constraints

- Images are carried as float32 `(H, W, 3)` in ADU (~0..65535) in-pipeline; FITS on disk is `(3, H, W)`. The starless workflow operates in nonlinear `[0,1]` space.
- Never overwrite processed images — outputs use the existing versioned `<name>_<label>` convention via `run_pipeline.sh`.
- `make test` must stay offline and dependency-free: unit tests mock the CLI; any real-binary test is guarded with `pytest.mark.skipif` on binary presence.
- Tests import modules from `scripts/` (enabled by `tests/conftest.py`). Use `pcc.py`/`test_pcc.py` as the style reference.
- StarNet2 invocation: `starnet2 -i in.fits -o starless.fits -n stars.fits -s <stride> -q`; native float32 FITS I/O; `--unscreen` (`-n`) star layer is the exact screen-blend inverse of the starless image; input must be ≥ 512×512.
- The binary is resolved via `$STARNET2_CLI`, else `~/StarNet2/starnet2`.

---

### Task 1: `astrolib` blend + sharpen helpers

**Files:**
- Modify: `scripts/astrolib.py` (append two functions)
- Test: `tests/test_astrolib.py` (create)

**Interfaces:**
- Consumes: numpy; `scipy.ndimage` (already used elsewhere in the file).
- Produces:
  - `screen(a, b) -> np.ndarray` — `1-(1-a)(1-b)`, inputs/output clipped to `[0,1]`.
  - `unsharp_luma(img01, amount=0.5, radius=2.0) -> np.ndarray` — unsharp mask on luminance only, hue preserved; `amount<=0` returns a copy of the clipped input.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_astrolib.py`:

```python
import numpy as np
import astrolib as al


def test_screen_identity_with_black():
    a = np.linspace(0, 1, 25).reshape(5, 5)
    black = np.zeros_like(a)
    # screening with black is a no-op
    assert np.allclose(al.screen(a, black), a)


def test_screen_commutative_and_bounded():
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 1, (8, 8, 3))
    b = rng.uniform(0, 1, (8, 8, 3))
    s = al.screen(a, b)
    assert np.allclose(s, al.screen(b, a))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_screen_reconstructs_unscreen_split():
    # If stars is the unscreen inverse of starless, screen() rebuilds the source.
    rng = np.random.RandomState(1)
    source = rng.uniform(0, 1, (6, 6, 3))
    starless = source * 0.5
    # unscreen star layer: stars = 1 - (1-source)/(1-starless)
    stars = 1.0 - (1.0 - source) / (1.0 - starless)
    assert np.allclose(al.screen(starless, stars), source, atol=1e-6)


def test_unsharp_luma_amount_zero_is_copy():
    rng = np.random.RandomState(2)
    img = rng.uniform(0, 1, (10, 10, 3))
    out = al.unsharp_luma(img, amount=0.0)
    assert np.allclose(out, np.clip(img, 0, 1))
    assert out is not img


def test_unsharp_luma_flat_field_unchanged():
    img = np.full((16, 16, 3), 0.4)
    out = al.unsharp_luma(img, amount=1.0, radius=2.0)
    assert np.allclose(out, img, atol=1e-6)


def test_unsharp_luma_increases_edge_contrast_preserving_gray():
    img = np.zeros((16, 16, 3))
    img[:, 8:] = 0.6  # a gray step edge
    out = al.unsharp_luma(img, amount=1.0, radius=1.5)
    # gray stays gray (no hue shift): channels equal everywhere
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])
    # overshoot at the edge => new max exceeds the original plateau
    assert out[..., 0].max() > 0.6 + 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_astrolib.py -v`
Expected: FAIL with `AttributeError: module 'astrolib' has no attribute 'screen'`.

- [ ] **Step 3: Implement the helpers**

Append to `scripts/astrolib.py` (after `finish`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_astrolib.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/astrolib.py tests/test_astrolib.py
git commit -m "Add screen blend + luma unsharp helpers to astrolib"
```

---

### Task 2: StarNet2 binary resolution

**Files:**
- Create: `scripts/starnet.py`
- Test: `tests/test_starnet.py` (create)

**Interfaces:**
- Consumes: `os`, `subprocess`, `tempfile`, numpy, `astropy.io.fits`.
- Produces:
  - `class StarNetError(Exception)`.
  - `DEFAULT_BINARY` (str), `MIN_SIZE = 512`.
  - `find_binary(binary=None) -> str` — resolves explicit arg → `$STARNET2_CLI` → `DEFAULT_BINARY`; raises `StarNetError` (with install hint) if not a runnable file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_starnet.py`:

```python
import os
import numpy as np
import pytest
import starnet


def _fake_exe(tmp_path, name="starnet2"):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return str(p)


def test_find_binary_prefers_explicit_arg(tmp_path):
    exe = _fake_exe(tmp_path)
    assert starnet.find_binary(exe) == exe


def test_find_binary_uses_env(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    monkeypatch.setenv("STARNET2_CLI", exe)
    assert starnet.find_binary() == exe


def test_find_binary_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("STARNET2_CLI", raising=False)
    missing = str(tmp_path / "nope")
    with pytest.raises(starnet.StarNetError):
        starnet.find_binary(missing)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_starnet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'starnet'`.

- [ ] **Step 3: Implement module skeleton + `find_binary`**

Create `scripts/starnet.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_starnet.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/starnet.py tests/test_starnet.py
git commit -m "Add starnet.py with StarNet2 binary resolution"
```

---

### Task 3: `remove_stars` CLI round-trip

**Files:**
- Modify: `scripts/starnet.py` (add FITS I/O helpers + `remove_stars`)
- Test: `tests/test_starnet.py` (add tests)

**Interfaces:**
- Consumes: `find_binary`, `StarNetError`, `MIN_SIZE` from Task 2.
- Produces:
  - `remove_stars(img01, binary=None, stride=256, tmpdir=None) -> (starless01, stars01)` — both `(H,W,3)` float32 in `[0,1]`; `stars` is the `--unscreen` layer. Raises `StarNetError` on bad shape, sub-`MIN_SIZE` input, nonzero CLI exit, or missing outputs.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_starnet.py`:

```python
from astropy.io import fits


def _write_cube(path, img):  # (H,W,3) -> (3,H,W) float32 FITS
    fits.PrimaryHDU(data=np.moveaxis(img.astype(np.float32), -1, 0)).writeto(
        path, overwrite=True)


def test_remove_stars_rejects_small_image(tmp_path):
    small = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(starnet.StarNetError):
        starnet.remove_stars(small, binary=_fake_exe(tmp_path))


def test_remove_stars_roundtrip(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    img = np.random.RandomState(0).uniform(0, 1, (512, 512, 3)).astype(np.float32)
    calls = {}

    def fake_run(argv, capture_output, text):
        calls["argv"] = argv
        out = argv[argv.index("-o") + 1]
        stars = argv[argv.index("-n") + 1]
        _write_cube(out, img * 0.5)
        _write_cube(stars, np.zeros_like(img))

        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        return R()

    monkeypatch.setattr(starnet.subprocess, "run", fake_run)
    starless, stars = starnet.remove_stars(img, binary=exe, stride=128)
    assert starless.shape == (512, 512, 3) and stars.shape == (512, 512, 3)
    assert starless.min() >= 0 and starless.max() <= 1
    argv = calls["argv"]
    assert argv[0] == exe
    for flag in ("-i", "-o", "-n"):
        assert flag in argv
    assert "128" in argv  # stride forwarded


def test_remove_stars_nonzero_exit_raises(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    img = np.zeros((512, 512, 3), dtype=np.float32)

    def fake_run(argv, capture_output, text):
        class R:
            returncode = 3
            stderr = "boom"
            stdout = ""
        return R()

    monkeypatch.setattr(starnet.subprocess, "run", fake_run)
    with pytest.raises(starnet.StarNetError) as e:
        starnet.remove_stars(img, binary=exe)
    assert "boom" in str(e.value)


@pytest.mark.skipif(
    not os.access(os.environ.get("STARNET2_CLI", starnet.DEFAULT_BINARY), os.X_OK),
    reason="StarNet2 binary not installed")
def test_remove_stars_real_binary_removes_stars():
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:512, 0:512]
    field = 0.15 + 0.05 * np.exp(-(((xx - 256) ** 2 + (yy - 256) ** 2) / (2 * 120.0 ** 2)))
    for _ in range(30):
        cy, cx = rng.integers(20, 492), rng.integers(20, 492)
        field += 0.8 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2)))
    img = np.clip(np.stack([field] * 3, -1), 0, 1).astype(np.float32)
    starless, stars = starnet.remove_stars(img)
    assert starless.max() < img.max() - 0.2          # stars removed
    import astrolib as al
    assert np.allclose(al.screen(starless, stars), img, atol=0.05)  # split recombines
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_starnet.py -v`
Expected: the four new tests FAIL with `AttributeError: module 'starnet' has no attribute 'remove_stars'` (the real-binary test runs since the CLI is installed).

- [ ] **Step 3: Implement FITS helpers + `remove_stars`**

Append to `scripts/starnet.py`:

```python
def _write_fits(path, img01):
    arr = np.moveaxis(np.clip(img01, 0.0, 1.0).astype(np.float32), -1, 0)  # (3,H,W)
    fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)


def _read_fits(path):
    data = fits.getdata(path).astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)  # (3,H,W) -> (H,W,3)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_starnet.py -v`
Expected: PASS (7 tests total, including the real-binary end-to-end).

- [ ] **Step 5: Commit**

```bash
git add scripts/starnet.py tests/test_starnet.py
git commit -m "Implement starnet.remove_stars CLI round-trip"
```

---

### Task 4: Starless finish orchestrator

**Files:**
- Create: `scripts/05b_starless_finish.py`
- Test: `tests/test_starless_finish.py` (create)

**Interfaces:**
- Consumes: `astrolib` (`load`, `screen`, `unsharp_luma`, `finish`, `save_preview`); `starnet.remove_stars`, `starnet.StarNetError`.
- Produces (importable via file-path, since the module name starts with a digit):
  - `process_starless(starless01, sharpen=True) -> np.ndarray` — unsharp luma (if `sharpen`) then `al.finish(..., scnr=False)`.
  - `main(infile, outfile_base, stride=STRIDE, no_sharpen=False) -> np.ndarray` — full workflow; writes `.tif`/`.png`/`_preview.png`/`_starless.png`/`_starlayer.png`; returns the final `[0,1]` result.

- [ ] **Step 1: Write the failing test**

Create `tests/test_starless_finish.py`:

```python
import importlib.util
import os
import numpy as np
import pytest
from astropy.io import fits
import starnet
import astrolib as al

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "scripts", "05b_starless_finish.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("starless_finish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_recombines_and_writes_outputs(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    base = rng.uniform(0, 1, (512, 512, 3)).astype(np.float32)  # [0,1] stretched
    infile = tmp_path / "in.fit"
    fits.PrimaryHDU(data=np.moveaxis(base, -1, 0)).writeto(infile, overwrite=True)

    starless = base * 0.5
    stars = np.zeros_like(base)
    stars[100, 100] = [1.0, 1.0, 1.0]
    monkeypatch.setattr(starnet, "remove_stars",
                        lambda img01, **kw: (starless.copy(), stars.copy()))

    mod = _load_module()
    out_base = str(tmp_path / "out")
    result = mod.main(str(infile), out_base)

    expected = al.screen(mod.process_starless(starless), stars)
    assert np.allclose(result, expected, atol=1e-6)
    for suffix in (".tif", ".png", "_preview.png", "_starless.png", "_starlayer.png"):
        assert os.path.exists(out_base + suffix), suffix


def test_no_sharpen_skips_unsharp(tmp_path):
    mod = _load_module()
    starless = np.full((32, 32, 3), 0.3)
    # with no sharpen, process_starless == finish(scnr off) of the input
    got = mod.process_starless(starless, sharpen=False)
    exp = al.finish(starless, saturation=mod.SATURATION,
                    luma_denoise=mod.LUMA_DENOISE,
                    chroma_denoise=mod.CHROMA_DENOISE, scnr=False)
    assert np.allclose(got, exp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_starless_finish.py -v`
Expected: FAIL — script file does not exist yet (`spec.loader.exec_module` raises `FileNotFoundError`).

- [ ] **Step 3: Implement the orchestrator**

Create `scripts/05b_starless_finish.py`:

```python
#!/usr/bin/env python3
"""Step 5b - Starless galaxy finish (alternative to 05_finish).

Recreates the SetiAstroSuitePro galaxy workflow in Python: remove stars with
the StarNet2 CLI, sharpen + denoise the starless galaxy, then screen the star
layer back in. Operates on the stretched (nonlinear) image from step 04. See
docs/superpowers/specs/2026-07-25-starless-galaxy-workflow-design.md.
"""
import sys
import numpy as np
import astrolib as al
import starnet

SHARPEN_AMOUNT = 0.5   # unsharp amount on starless luma (0 = off)
SHARPEN_RADIUS = 2.0   # unsharp gaussian radius (px)
LUMA_DENOISE = 0.012   # TV-denoise weight on luminance
CHROMA_DENOISE = 4.0   # gaussian sigma on chroma (px)
SATURATION = 1.20      # HSV saturation multiplier
STRIDE = 256           # StarNet2 tile stride


def process_starless(starless01, sharpen=True):
    """Sharpen (luma) then denoise + saturate a [0,1] starless RGB image.

    SCNR is off here: the green cast is handled upstream in step 03, and the
    star layer (added back later) is where residual green usually lives.
    """
    img = starless01
    if sharpen and SHARPEN_AMOUNT > 0:
        img = al.unsharp_luma(img, amount=SHARPEN_AMOUNT, radius=SHARPEN_RADIUS)
    return al.finish(img, saturation=SATURATION, luma_denoise=LUMA_DENOISE,
                     chroma_denoise=CHROMA_DENOISE, scnr=False)


def main(infile, outfile_base, stride=STRIDE, no_sharpen=False):
    img, _hdr = al.load(infile)
    img01 = np.clip(img / 65535.0, 0.0, 1.0)

    print(">> starnet: removing stars")
    starless, stars = starnet.remove_stars(img01, stride=stride)

    print(">> processing starless (sharpen -> denoise -> saturation)")
    starless_proc = process_starless(starless, sharpen=not no_sharpen)

    result = al.screen(starless_proc, stars)

    from PIL import Image
    u16 = (result * 65535.0 + 0.5).astype(np.uint16)
    try:
        from skimage.io import imsave
        imsave(outfile_base + ".tif", u16, check_contrast=False)
        print(f"wrote {outfile_base}.tif (16-bit)")
    except Exception as e:
        print("TIFF export skipped:", e)
    Image.fromarray((result * 255 + 0.5).astype(np.uint8)).save(outfile_base + ".png")
    print(f"wrote {outfile_base}.png")
    al.save_preview(outfile_base + "_preview.png", img01=result, stretch=False)
    # diagnostics: inspect for star halos / over-sharpening
    al.save_preview(outfile_base + "_starless.png", img01=starless_proc, stretch=False)
    al.save_preview(outfile_base + "_starlayer.png", img01=stars, stretch=False)
    print(f"wrote diagnostics {outfile_base}_starless.png / _starlayer.png")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--no-sharpen", action="store_true")
    a = ap.parse_args()
    try:
        main(a.infile, a.outfile, stride=a.stride, no_sharpen=a.no_sharpen)
    except starnet.StarNetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_starless_finish.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (existing PCC + astrolib + starnet + starless_finish).

- [ ] **Step 6: Commit**

```bash
git add scripts/05b_starless_finish.py tests/test_starless_finish.py
git commit -m "Add 05b starless galaxy finish orchestrator"
```

---

### Task 5: Pipeline wiring, Makefile target, docs, and real run

**Files:**
- Modify: `scripts/run_pipeline.sh` (add `--starless` flag)
- Modify: `Makefile` (add `run-starless` target + help line)
- Modify: `README.md` (document the alternate path + StarNet2 prerequisite)

**Interfaces:**
- Consumes: `05b_starless_finish.py` from Task 4; existing steps 01–04.
- Produces: `run_pipeline.sh <in> [label] --starless` runs 01–04 then 05b instead of 05; `make run-starless FITS=... [V=...]` wraps it.

- [ ] **Step 1: Add the `--starless` flag to `run_pipeline.sh`**

In `scripts/run_pipeline.sh`, after `set -euo pipefail` and the `HERE/ROOT/PY` block, add flag parsing that strips `--starless` from anywhere in the args:

```bash
STARLESS=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --starless) STARLESS=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]}"
```

Then replace the step-05 line:

```bash
echo ">> 05 finish";     "$PY" 05_finish.py     "$WORK/04_stretch.fit" "$OUTBASE"
```

with:

```bash
if [[ "$STARLESS" == "1" ]]; then
  echo ">> 05b starless finish"
  "$PY" 05b_starless_finish.py "$WORK/04_stretch.fit" "$OUTBASE"
else
  echo ">> 05 finish"
  "$PY" 05_finish.py "$WORK/04_stretch.fit" "$OUTBASE"
fi
```

- [ ] **Step 2: Syntax-check the script**

Run: `bash -n scripts/run_pipeline.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Add the Makefile target**

In `Makefile`, add `run-starless` to `.PHONY`, add a help line, and add the target:

```makefile
run-starless:
	scripts/run_pipeline.sh "$(FITS)" $(V) --starless
```

Help line (in the `help` target):

```makefile
	@echo "make run-starless FITS=path [V=label] - full pipeline w/ StarNet2 starless finish"
```

- [ ] **Step 4: Verify the Makefile parses**

Run: `make -n run-starless FITS=x`
Expected: prints `scripts/run_pipeline.sh "x"  --starless` (no execution).

- [ ] **Step 5: Document in README.md**

Add a subsection under the processing-pipeline docs describing the starless path. Include verbatim:

```markdown
### Starless galaxy finish (optional)

An alternate finishing path that recreates the SetiAstroSuitePro galaxy
workflow: remove stars, sharpen/denoise the starless galaxy, then screen the
stars back in. Runs steps 01-04 unchanged, then `05b_starless_finish.py`
instead of `05_finish.py`.

    make run-starless FITS="data/<master>.fit" V=starless

Prerequisite: the **StarNet2 CLI** (Apple Silicon build) installed at
`~/StarNet2/` (or point `$STARNET2_CLI` at the binary). It removes stars with
a CoreML model; `05b` uses its `--unscreen` output so the star layer recombines
exactly. Outputs the finished image plus `_starless.png` and `_starlayer.png`
diagnostics. Sharpening is gentle by default and can be disabled with
`--no-sharpen` (pass through `scripts/05b_starless_finish.py` directly).
```

- [ ] **Step 6: Commit the wiring + docs**

```bash
git add scripts/run_pipeline.sh Makefile README.md
git commit -m "Wire --starless into pipeline (flag, make target, README)"
```

- [ ] **Step 7: Real end-to-end run on the M101 master**

Run (substitute the current re-stacked, plate-solved master under `data/`):

```bash
make run-starless FITS="data/<restacked-master>.fit" V=starless-v1
```

Expected: steps 01–04 print, then `>> 05b starless finish` and `>> starnet: removing stars` with StarNet2 progress, then output files under `output/<name>_starless-v1.{tif,png}` plus `_starless.png` and `_starlayer.png`. Inspect the star-layer diagnostic for halos and the final image for over-sharpening.

- [ ] **Step 8: Side-by-side comparison against the standard finish**

Produce a standard finish for the same master and compare:

```bash
make run FITS="data/<restacked-master>.fit" V=standard-v1
.venv/bin/python scripts/compare.py --pair \
  output/<name>_standard-v1.png output/<name>_starless-v1.png
```

Expected: a side-by-side comparison image in `output/` so the user can judge whether the starless route wins on this data. (No commit — outputs are git-ignored.)

---

## Self-Review

**Spec coverage:**
- StarNet2 CLI wrapper isolated like `pcc.py` → Tasks 2–3 (`starnet.py`). ✓
- `--unscreen` exact star split → Task 3 (`-n` flag) + Task 1 `screen`. ✓
- float32 FITS round-trip → Task 3 `_write_fits`/`_read_fits`. ✓
- Process starless: unsharp → denoise → saturation → Tasks 1 + 4. ✓
- Screen recombination → Task 4 `main`. ✓
- Diagnostics (star layer + starless) → Task 4. ✓
- `--starless` flag, `make run-starless`, README → Task 5. ✓
- Error handling (missing binary, nonzero exit, sub-512) → Tasks 2–3; CLI-exit path in `main` → Task 4. ✓
- Tests offline + guarded real test → Tasks 1–4; `make test` clean. ✓
- Existing `05_finish.py` untouched → confirmed (Task 5 branches, does not modify). ✓
- Out of scope (AI denoise, star reduction, mono/upsample) → not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `screen`, `unsharp_luma`, `remove_stars(img01, binary, stride, tmpdir) -> (starless, stars)`, `process_starless(starless01, sharpen)`, `main(infile, outfile_base, stride, no_sharpen) -> result` are named/typed identically across defining and consuming tasks. ✓
