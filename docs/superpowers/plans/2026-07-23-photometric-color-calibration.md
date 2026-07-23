# Photometric Color Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add empirical Gaia DR3 photometric color calibration that derives R/B channel gains (relative to G) from catalog star colors and applies them in pipeline step 03, with graceful fallback to the existing gentle white-balance.

**Architecture:** A new `scripts/pcc.py` module with small, independently testable units (detect → query → cross-match → solve → apply) plus a diagnostic plot. PCC *measures* on the original stacked FITS (retains SIP WCS) and returns three global gains; step 03 *applies* them to its neutralized working image, falling back to gentle WB on any `PCCError`.

**Tech Stack:** Python 3.9+, numpy, astropy (io.fits, wcs, table, stats, coordinates), photutils (detection + aperture), astroquery (Gaia), matplotlib, pytest.

## Global Constraints

- numpy pinned `>=1.26,<2.0` (cross-platform wheel compatibility) — do not un-pin.
- New deps: `astroquery>=0.4.7`, `pytest>=8.0` — add to `requirements.txt`.
- Run Python via the project venv: `.venv/bin/python`.
- Images carried as `float` arrays shaped `(H, W, 3)` in ADU (~0..65535); FITS on disk is `(3, H, W)`. Use `astrolib.load` / `astrolib.save`.
- `PEDESTAL = 0.10 * 65535` is the neutral background level used by steps 02/03.
- White point default: `REF_BP_RP = 0.82` (G2V). Match tolerance `5.0`", `MIN_STARS = 30`, gain sanity band `(0.5, 2.0)`.
- Output convention: never overwrite processed images; every variant gets a versioned/descriptive name. `data/`, `output/`, `work/` are git-ignored.
- Git: commit each task; **do not push or open a PR without explicit confirmation**. Work stays on branch `feature/compare-tool`.

---

### Task 1: Dependencies, test scaffold, and `apply_gains`

**Files:**
- Modify: `requirements.txt`
- Modify: `Makefile`
- Create: `scripts/pcc.py`
- Create: `tests/conftest.py`
- Create: `tests/test_pcc.py`

**Interfaces:**
- Produces: `class PCCError(Exception)`; `apply_gains(img_adu, gains, pedestal) -> np.ndarray` where `gains` is a 3-tuple `(gr, gg, gb)`.

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:
```
astroquery>=0.4.7
pytest>=8.0
```

- [ ] **Step 2: Install them**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: astroquery and pytest install successfully.

- [ ] **Step 3: Add a `test` target to the Makefile**

Add under the `.PHONY` line (append target to `Makefile`):
```makefile
test:
	$(PY) -m pytest -q
```
Also add `test` to the `.PHONY:` list.

- [ ] **Step 4: Create the test path shim**

Create `tests/conftest.py`:
```python
import os
import sys

# Make scripts/ importable (astrolib, pcc) from tests.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_pcc.py`:
```python
import numpy as np
import pytest
import pcc


def test_apply_gains_preserves_pedestal_and_scales_signal():
    ped = 1000.0
    img = np.full((4, 4, 3), ped, dtype=float)
    img[0, 0] = [ped + 100, ped + 100, ped + 100]
    out = pcc.apply_gains(img, (2.0, 1.0, 0.5), ped)
    # background (at pedestal) is unchanged
    assert np.allclose(out[1, 1], ped)
    # signal scales around the pedestal, per channel
    assert np.allclose(out[0, 0], [ped + 200, ped + 100, ped + 50])


def test_apply_gains_identity():
    rng = np.random.RandomState(0)
    img = rng.uniform(0, 1000, (3, 3, 3))
    out = pcc.apply_gains(img, (1.0, 1.0, 1.0), 100.0)
    assert np.allclose(out, img)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pcc.py -q`
Expected: FAIL — `AttributeError: module 'pcc' has no attribute 'apply_gains'` (or ImportError if pcc.py absent).

- [ ] **Step 7: Create `scripts/pcc.py` with minimal implementation**

```python
#!/usr/bin/env python3
"""Empirical Gaia DR3 photometric color calibration for OSC astro images.

Measures star colors on the original (WCS-bearing) stacked FITS, regresses them
against Gaia colors, and solves for R/B channel gains relative to G that put a
chosen white point (default solar) at neutral. See
docs/superpowers/specs/2026-07-23-photometric-color-calibration-design.md.
"""
import numpy as np


class PCCError(Exception):
    """Raised for any condition that should trigger the gentle-WB fallback."""


def apply_gains(img_adu, gains, pedestal):
    """Scale each channel around the neutral pedestal: out = (img-ped)*gain + ped."""
    out = np.asarray(img_adu, dtype=np.float64).copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    return out
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pcc.py -q`
Expected: PASS (2 passed).

- [ ] **Step 9: Commit**

```bash
git add requirements.txt Makefile scripts/pcc.py tests/conftest.py tests/test_pcc.py
git commit -m "PCC: deps, pytest scaffold, apply_gains"
```

---

### Task 2: `solve_gains`

**Files:**
- Modify: `scripts/pcc.py`
- Modify: `tests/test_pcc.py`

**Interfaces:**
- Consumes: an `astropy.table.Table` (or dict of arrays) `matched` with columns `r, g, b, bp_rp`.
- Produces: `solve_gains(matched, ref_bp_rp=0.82, sigma=3.0) -> (gains, diag)` where `gains = (gr, 1.0, gb)` and `diag` is a dict with keys `n, slope_r, int_r, slope_b, int_b, cr, cb, bp_rp, ref_bp_rp`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pcc.py`:
```python
from astropy.table import Table


def test_solve_gains_recovers_known_color_law():
    rng = np.random.RandomState(1)
    bp_rp = rng.uniform(0.2, 2.0, 300)
    g = rng.uniform(100, 1000, 300)
    cr = 0.30 * bp_rp + 0.70          # instrumental r/g
    cb = -0.20 * bp_rp + 1.30         # instrumental b/g
    matched = Table({"r": cr * g, "g": g, "b": cb * g, "bp_rp": bp_rp})

    gains, diag = pcc.solve_gains(matched, ref_bp_rp=0.82)

    assert np.isclose(gains[1], 1.0)
    assert np.isclose(gains[0], 1.0 / (0.30 * 0.82 + 0.70), rtol=1e-3)
    assert np.isclose(gains[2], 1.0 / (-0.20 * 0.82 + 1.30), rtol=1e-3)
    assert diag["n"] > 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pcc.py::test_solve_gains_recovers_known_color_law -q`
Expected: FAIL — `AttributeError: module 'pcc' has no attribute 'solve_gains'`.

- [ ] **Step 3: Implement `solve_gains`**

Add to `scripts/pcc.py`:
```python
def _robust_linfit(x, y, sigma=3.0, iters=3):
    """Sigma-clipped least-squares line y ~ slope*x + intercept."""
    m = np.isfinite(x) & np.isfinite(y)
    xf, yf = np.asarray(x)[m], np.asarray(y)[m]
    slope = intercept = 0.0
    for _ in range(iters):
        if len(xf) < 3:
            break
        A = np.vstack([xf, np.ones_like(xf)]).T
        slope, intercept = np.linalg.lstsq(A, yf, rcond=None)[0]
        resid = yf - (slope * xf + intercept)
        s = resid.std()
        if s == 0:
            break
        keep = np.abs(resid) < sigma * s
        xf, yf = xf[keep], yf[keep]
    return slope, intercept, len(xf)


def solve_gains(matched, ref_bp_rp=0.82, sigma=3.0):
    """Regress instrumental color ratios vs Gaia bp_rp; solve gains at the white point."""
    r = np.asarray(matched["r"], float)
    g = np.asarray(matched["g"], float)
    b = np.asarray(matched["b"], float)
    bp_rp = np.asarray(matched["bp_rp"], float)
    good = (g > 0) & (r > 0) & (b > 0)
    r, g, b, bp_rp = r[good], g[good], b[good], bp_rp[good]
    cr, cb = r / g, b / g

    slope_r, int_r, n = _robust_linfit(bp_rp, cr, sigma)
    slope_b, int_b, _ = _robust_linfit(bp_rp, cb, sigma)
    cr0 = slope_r * ref_bp_rp + int_r
    cb0 = slope_b * ref_bp_rp + int_b
    if not (cr0 > 0 and cb0 > 0):
        raise PCCError("degenerate color fit (non-positive ratio at white point)")

    gains = (1.0 / cr0, 1.0, 1.0 / cb0)
    diag = dict(n=n, slope_r=slope_r, int_r=int_r, slope_b=slope_b, int_b=int_b,
                cr=cr, cb=cb, bp_rp=bp_rp, ref_bp_rp=ref_bp_rp)
    return gains, diag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pcc.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pcc.py tests/test_pcc.py
git commit -m "PCC: solve_gains with sigma-clipped color regression"
```

---

### Task 3: `cross_match`

**Files:**
- Modify: `scripts/pcc.py`
- Modify: `tests/test_pcc.py`

**Interfaces:**
- Consumes: `stars` Table with `x, y, r, g, b`; `gaia` Table with `ra, dec, bp_rp`; an `astropy.wcs.WCS` (celestial, 2-D).
- Produces: `cross_match(stars, gaia, wcs, tol_arcsec=5.0) -> Table` with columns `x, y, r, g, b, bp_rp, sep_arcsec`, deduplicated so each Gaia source matches at most one star (closest wins).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pcc.py`:
```python
from astropy.wcs import WCS


def _toy_wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-0.001, 0.001]   # deg/pixel
    w.wcs.crval = [211.0, 54.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_cross_match_pairs_within_tolerance():
    w = _toy_wcs()
    xs, ys = np.array([50.0, 60.0, 30.0]), np.array([50.0, 40.0, 70.0])
    sky = w.pixel_to_world(xs, ys)
    stars = Table({"x": xs, "y": ys,
                   "r": [10.0, 20.0, 30.0], "g": [10.0, 20.0, 30.0], "b": [10.0, 20.0, 30.0]})
    # Gaia: first two coincide with stars; third is far away (no match expected)
    gaia = Table({
        "ra": [sky[0].ra.deg, sky[1].ra.deg, 100.0],
        "dec": [sky[0].dec.deg, sky[1].dec.deg, 10.0],
        "bp_rp": [0.5, 1.5, 2.5],
    })
    matched = pcc.cross_match(stars, gaia, w, tol_arcsec=5.0)
    assert len(matched) == 2
    assert set(np.round(matched["bp_rp"], 1)) == {0.5, 1.5}
    assert np.all(matched["sep_arcsec"] < 5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pcc.py::test_cross_match_pairs_within_tolerance -q`
Expected: FAIL — `AttributeError: module 'pcc' has no attribute 'cross_match'`.

- [ ] **Step 3: Implement `cross_match`**

Add to `scripts/pcc.py`:
```python
def cross_match(stars, gaia, wcs, tol_arcsec=5.0):
    """Match detected stars to Gaia by sky position; dedupe (closest wins)."""
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    star_sky = wcs.pixel_to_world(np.asarray(stars["x"]), np.asarray(stars["y"]))
    gaia_sky = SkyCoord(np.asarray(gaia["ra"]) * u.deg, np.asarray(gaia["dec"]) * u.deg)

    idx, sep2d, _ = star_sky.match_to_catalog_sky(gaia_sky)
    sep = sep2d.arcsec
    keep = sep < tol_arcsec

    rows = Table({
        "x": np.asarray(stars["x"])[keep],
        "y": np.asarray(stars["y"])[keep],
        "r": np.asarray(stars["r"])[keep],
        "g": np.asarray(stars["g"])[keep],
        "b": np.asarray(stars["b"])[keep],
        "bp_rp": np.asarray(gaia["bp_rp"])[idx[keep]],
        "gaia_idx": idx[keep],
        "sep_arcsec": sep[keep],
    })
    # dedupe: keep the closest star per Gaia source
    rows.sort("sep_arcsec")
    _, first = np.unique(rows["gaia_idx"], return_index=True)
    rows = rows[np.sort(first)]
    rows.remove_column("gaia_idx")
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pcc.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pcc.py tests/test_pcc.py
git commit -m "PCC: cross_match stars to Gaia via WCS"
```

---

### Task 4: `detect_stars`

**Files:**
- Modify: `scripts/pcc.py`
- Modify: `tests/test_pcc.py`

**Interfaces:**
- Consumes: `img_adu` shaped `(H, W, 3)`.
- Produces: `detect_stars(img_adu, fwhm=3.0, threshold_sigma=5.0, aperture_r=4.0, annulus=(6.0, 9.0), sat=60000.0, edge=20, min_flux=1.0) -> Table` with columns `x, y, r, g, b` (background-subtracted instrumental flux per channel). Raises `PCCError` if nothing usable is found.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pcc.py`:
```python
def _inject_star(img, x, y, amp, fwhm=3.0):
    H, W, _ = img.shape
    sig = fwhm / 2.355
    yy, xx = np.mgrid[0:H, 0:W]
    g2d = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
    for c in range(3):
        img[..., c] += amp[c] * g2d


def test_detect_stars_finds_injected_and_measures_color():
    rng = np.random.RandomState(2)
    img = rng.normal(1000.0, 5.0, (120, 120, 3))       # flat noisy background
    # three stars with distinct colors (r:g:b amplitudes)
    _inject_star(img, 30, 40, (2000, 1000, 500))
    _inject_star(img, 80, 70, (800, 1000, 1200))
    _inject_star(img, 60, 20, (1500, 1500, 1500))

    stars = pcc.detect_stars(img, fwhm=3.0, threshold_sigma=5.0)
    assert len(stars) >= 3
    # locate the star nearest (30,40) and check its r/g > 1 (red star)
    d = (stars["x"] - 30) ** 2 + (stars["y"] - 40) ** 2
    red = stars[int(np.argmin(d))]
    assert red["r"] / red["g"] > 1.3
    assert red["b"] / red["g"] < 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pcc.py::test_detect_stars_finds_injected_and_measures_color -q`
Expected: FAIL — `AttributeError: module 'pcc' has no attribute 'detect_stars'`.

- [ ] **Step 3: Implement `detect_stars`**

Add to `scripts/pcc.py`:
```python
def detect_stars(img_adu, fwhm=3.0, threshold_sigma=5.0, aperture_r=4.0,
                 annulus=(6.0, 9.0), sat=60000.0, edge=20, min_flux=1.0):
    """Detect stars on luminance and measure per-channel aperture flux."""
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from photutils.detection import DAOStarFinder
    from photutils.aperture import (CircularAperture, CircularAnnulus,
                                    aperture_photometry)

    img = np.asarray(img_adu, dtype=np.float64)
    lum = img.mean(axis=2)
    _, med, std = sigma_clipped_stats(lum, sigma=3.0)

    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(lum - med)
    if sources is None or len(sources) == 0:
        raise PCCError("no stars detected")

    H, W = lum.shape
    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])
    peak = np.asarray(sources["peak"])
    keep = ((x > edge) & (x < W - edge) & (y > edge) & (y < H - edge)
            & (peak + med < sat))
    x, y = x[keep], y[keep]
    if len(x) == 0:
        raise PCCError("no stars survive edge/saturation cuts")

    positions = np.column_stack([x, y])
    ap = CircularAperture(positions, r=aperture_r)
    ann = CircularAnnulus(positions, r_in=annulus[0], r_out=annulus[1])
    flux = {}
    for c, name in enumerate("rgb"):
        chan = img[..., c]
        src = aperture_photometry(chan, ap)["aperture_sum"]
        bkg = aperture_photometry(chan, ann)["aperture_sum"] / ann.area
        flux[name] = np.asarray(src) - np.asarray(bkg) * ap.area

    tab = Table({"x": x, "y": y, "r": flux["r"], "g": flux["g"], "b": flux["b"]})
    good = (tab["r"] > min_flux) & (tab["g"] > min_flux) & (tab["b"] > min_flux)
    tab = tab[good]
    if len(tab) == 0:
        raise PCCError("no stars with valid positive photometry")
    return tab
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pcc.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pcc.py tests/test_pcc.py
git commit -m "PCC: detect_stars with per-channel aperture photometry"
```

---

### Task 5: `query_gaia`

**Files:**
- Modify: `scripts/pcc.py`

**Interfaces:**
- Consumes: FITS `hdr` with `RA`, `DEC` (degrees).
- Produces: `query_gaia(hdr, radius_deg=2.3, mag_limit=16.0, row_limit=50000) -> Table` with columns `ra, dec, phot_g_mean_mag, bp_rp`. Raises `PCCError` on missing coords, unavailable astroquery, query failure, or empty result. No unit test (network); manual verification below.

- [ ] **Step 1: Implement `query_gaia`**

Add to `scripts/pcc.py`:
```python
def query_gaia(hdr, radius_deg=2.3, mag_limit=16.0, row_limit=50000):
    """Cone-search Gaia DR3 around the header RA/Dec. Raises PCCError on failure."""
    if "RA" not in hdr or "DEC" not in hdr:
        raise PCCError("header missing RA/DEC for Gaia query")
    ra, dec = float(hdr["RA"]), float(hdr["DEC"])
    try:
        from astroquery.gaia import Gaia
    except Exception as e:
        raise PCCError(f"astroquery unavailable: {e}")

    adql = (
        f"SELECT TOP {row_limit} ra, dec, phot_g_mean_mag, bp_rp "
        f"FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
        f"AND phot_g_mean_mag < {mag_limit} "
        f"AND bp_rp IS NOT NULL "
        f"AND astrometric_params_solved > 3"
    )
    try:
        job = Gaia.launch_job_async(adql)
        tab = job.get_results()
    except Exception as e:
        raise PCCError(f"Gaia query failed: {e}")
    if len(tab) == 0:
        raise PCCError("Gaia query returned no rows")
    return tab["ra", "dec", "phot_g_mean_mag", "bp_rp"]
```

- [ ] **Step 2: Manual verification against the real header**

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
import pcc, astrolib as al
_, hdr = al.load('data/Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit')
t = pcc.query_gaia(hdr)
print('gaia rows:', len(t)); print(t[:3])
"
```
Expected: prints several thousand rows and a small preview table with `ra, dec, phot_g_mean_mag, bp_rp`. (Requires internet. If offline, note it and defer to Task 6/9.)

- [ ] **Step 3: Commit**

```bash
git add scripts/pcc.py
git commit -m "PCC: query_gaia cone search (DR3)"
```

---

### Task 6: `photometric_calibration` orchestration + diagnostic plot

**Files:**
- Modify: `scripts/pcc.py`

**Interfaces:**
- Consumes: original `img_adu (H,W,3)` and its `hdr` (with WCS + RA/DEC); `detect_stars`, `query_gaia`, `cross_match`, `solve_gains`.
- Produces:
  - `photometric_calibration(img_adu, hdr, ref_bp_rp=0.82, min_stars=30, gain_band=(0.5, 2.0), tol_arcsec=5.0) -> (gains, report)` where `report` is a dict with `gains, n_matched, ref_bp_rp` and the `solve_gains` diag keys plus `matched` (the matched Table). Raises `PCCError` on no-WCS / too-few-stars / out-of-band gains.
  - `save_diagnostic(report, path)` writes a color-color PNG.

- [ ] **Step 1: Implement `photometric_calibration`**

Add to `scripts/pcc.py`:
```python
def photometric_calibration(img_adu, hdr, ref_bp_rp=0.82, min_stars=30,
                            gain_band=(0.5, 2.0), tol_arcsec=5.0):
    """End-to-end: detect -> Gaia -> match -> solve gains. Raises PCCError to fall back."""
    from astropy.wcs import WCS
    try:
        wcs = WCS(hdr).celestial
        if not wcs.has_celestial:
            raise PCCError("no celestial WCS in header")
    except PCCError:
        raise
    except Exception as e:
        raise PCCError(f"WCS parse failed: {e}")

    stars = detect_stars(img_adu)
    gaia = query_gaia(hdr)
    matched = cross_match(stars, gaia, wcs, tol_arcsec=tol_arcsec)
    if len(matched) < min_stars:
        raise PCCError(f"only {len(matched)} matched stars (< {min_stars})")

    gains, diag = solve_gains(matched, ref_bp_rp=ref_bp_rp)
    for name, gv in zip("rgb", gains):
        if not (gain_band[0] <= gv <= gain_band[1]):
            raise PCCError(f"gain {name}={gv:.3f} outside sanity band {gain_band}")

    report = dict(gains=gains, n_matched=len(matched), matched=matched, **diag)
    return gains, report
```

- [ ] **Step 2: Implement `save_diagnostic`**

Add to `scripts/pcc.py`:
```python
def save_diagnostic(report, path):
    """Color-color scatter (r/g and b/g vs Gaia bp_rp) with fits + white point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bp_rp = report["bp_rp"]; cr = report["cr"]; cb = report["cb"]
    xr = np.linspace(float(np.min(bp_rp)), float(np.max(bp_rp)), 50)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bp_rp, cr, s=8, c="tab:red", alpha=0.4, label="r/g")
    ax.scatter(bp_rp, cb, s=8, c="tab:blue", alpha=0.4, label="b/g")
    ax.plot(xr, report["slope_r"] * xr + report["int_r"], c="darkred")
    ax.plot(xr, report["slope_b"] * xr + report["int_b"], c="darkblue")
    ax.axvline(report["ref_bp_rp"], ls="--", c="k", lw=1,
               label=f"white point ({report['ref_bp_rp']})")
    gr, _, gb = report["gains"]
    ax.set_title(f"PCC — {report['n_matched']} stars — gains R={gr:.3f} B={gb:.3f}")
    ax.set_xlabel("Gaia BP-RP"); ax.set_ylabel("instrumental ratio")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
```

- [ ] **Step 3: Add a standalone CLI for manual runs**

Append to `scripts/pcc.py`:
```python
def _main(argv):
    import argparse
    import astrolib as al
    ap = argparse.ArgumentParser(description="Run PCC on a stacked FITS (measurement only).")
    ap.add_argument("fits")
    ap.add_argument("--ref-bp-rp", type=float, default=0.82)
    ap.add_argument("--diagnostic", help="path to write the color-color PNG")
    args = ap.parse_args(argv)
    img, hdr = al.load(args.fits)
    gains, report = photometric_calibration(img, hdr, ref_bp_rp=args.ref_bp_rp)
    print(f"matched={report['n_matched']}  gains R={gains[0]:.4f} G=1 B={gains[2]:.4f}")
    if args.diagnostic:
        save_diagnostic(report, args.diagnostic)
        print("wrote", args.diagnostic)


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
```

- [ ] **Step 4: Manual verification on the M101 stack**

Run:
```bash
.venv/bin/python scripts/pcc.py "data/Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit" \
  --diagnostic output/M101_pcc_diagnostic.png
```
Expected: prints `matched=<N≥30>  gains R=... G=1 B=...` with gains inside (0.5, 2.0); writes the diagnostic PNG. Open it: the r/g and b/g point clouds should show a clean trend with the fit lines through them. (Requires internet.)

- [ ] **Step 5: Commit**

```bash
git add scripts/pcc.py
git commit -m "PCC: photometric_calibration orchestration + diagnostic plot + CLI"
```

---

### Task 7: Integrate PCC into step 03 + pipeline wiring

**Files:**
- Modify: `scripts/03_color.py`
- Modify: `scripts/run_pipeline.sh`

**Interfaces:**
- Consumes: `pcc.photometric_calibration`, `pcc.apply_gains`, `pcc.PCCError`.
- Produces: `03_color.py` CLI `03_color.py <in.fit> <out.fit> [--original PATH] [--no-pcc]`. When `--original` is given and PCC succeeds, applies PCC gains; otherwise falls back to `gentle_white_balance(out, PEDESTAL)`.

- [ ] **Step 1: Extract the existing gentle WB into a function**

In `scripts/03_color.py`, refactor the current step-2 white-balance block in `main` into a module-level function (keep the exact math), then call it. Add near the top (after the existing `bg_level`):
```python
def gentle_white_balance(img, pedestal, clamp=(0.85, 1.15)):
    """Match mid-signal (p60..p99 luminance) across channels to green; clamped."""
    lum = img.mean(axis=2)
    lo, hi = np.percentile(lum, 60), np.percentile(lum, 99)
    band = (lum > lo) & (lum < hi)
    means = np.array([img[..., c][band].mean() - pedestal for c in range(3)])
    gains = np.clip(means[1] / means, *clamp)
    out = img.copy()
    for c in range(3):
        out[..., c] = (out[..., c] - pedestal) * gains[c] + pedestal
    print(f"  gentle-WB gains (R,G,B) = {gains.round(3)}")
    return out
```

- [ ] **Step 2: Rewrite `main` to try PCC then fall back**

Replace the white-balance section of `main` (everything after background neutralization, before `al.save`) with:
```python
    # 2. color balance: PCC from the original (WCS-bearing) stack, else gentle WB
    if original and not no_pcc:
        try:
            import pcc
            oimg, ohdr = al.load(original)
            gains, report = pcc.photometric_calibration(
                oimg, ohdr, ref_bp_rp=REF_BP_RP, min_stars=MIN_STARS)
            out = pcc.apply_gains(out, gains, PEDESTAL)
            print(f"  PCC gains (R,G,B) = ({gains[0]:.3f}, 1.000, {gains[2]:.3f}) "
                  f"from {report['n_matched']} stars")
            pcc.save_diagnostic(report, outfile.replace(".fit", "_pcc_diagnostic.png"))
        except pcc.PCCError as e:
            print(f"  WARNING: PCC unavailable ({e}); using gentle white balance")
            out = gentle_white_balance(out, PEDESTAL)
    else:
        out = gentle_white_balance(out, PEDESTAL)
```

- [ ] **Step 3: Add constants and update the CLI signature**

Near the top of `scripts/03_color.py` add:
```python
REF_BP_RP = 0.82
MIN_STARS = 30
```
Change `def main(infile, outfile):` to `def main(infile, outfile, original=None, no_pcc=False):` and update the `__main__` block:
```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--original", help="original stacked FITS (with WCS) for PCC")
    ap.add_argument("--no-pcc", action="store_true")
    a = ap.parse_args()
    main(a.infile, a.outfile, original=a.original, no_pcc=a.no_pcc)
```

- [ ] **Step 4: Pass the original path from the pipeline runner**

In `scripts/run_pipeline.sh`, change the step 03 line to forward the original input:
```bash
echo ">> 03 color";      "$PY" 03_color.py      "$WORK/02_bg.fit"   "$WORK/03_color.fit" --original "$IN"
```

- [ ] **Step 5: Verify fallback path works offline-safe (`--no-pcc`)**

Run:
```bash
.venv/bin/python scripts/03_color.py work/02_bg.fit /tmp/03_nopcc.fit --no-pcc
```
Expected: prints `gentle-WB gains ...`, writes the file, no PCC attempted. (Assumes `work/02_bg.fit` exists from a prior run; if not, run `make run FITS=... V=tmp` first.)

- [ ] **Step 6: Verify PCC path end to end**

Run:
```bash
.venv/bin/python scripts/03_color.py work/02_bg.fit /tmp/03_pcc.fit \
  --original "data/Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit"
```
Expected: prints `PCC gains (R,G,B) = (..., 1.000, ...) from <N> stars` and writes `/tmp/03_pcc_pcc_diagnostic.png`. (Requires internet; on failure it prints the WARNING and falls back — that is also acceptable behavior to observe.)

- [ ] **Step 7: Commit**

```bash
git add scripts/03_color.py scripts/run_pipeline.sh
git commit -m "PCC: integrate into step 03 with gentle-WB fallback; wire pipeline"
```

---

### Task 8: `compare.py` PCC before/after (generic pair mode)

**Files:**
- Modify: `scripts/compare.py`

**Interfaces:**
- Produces: a `--pair BEFORE.fit AFTER.fit --name NAME --label-before TEXT --label-after TEXT` mode that crops+stitches two arbitrary [0,1]-scale FITS into `output/M101_compare_<NAME>.png`, reusing the existing `crop_img` / `stitch` helpers.

- [ ] **Step 1: Add a pair-comparison path to `compare.py`**

In `scripts/compare.py`, add these args in `main()`'s argparse block:
```python
    ap.add_argument("--pair", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="two stretched FITS to compare directly (bypasses finish variants)")
    ap.add_argument("--name", default="pair", help="output name suffix for --pair mode")
    ap.add_argument("--label-before", default="BEFORE")
    ap.add_argument("--label-after", default="AFTER")
```
Then, immediately after `args = ap.parse_args()`, handle the pair mode and return early:
```python
    if args.pair:
        before, _ = al.load(args.pair[0]); before = np.clip(before / 65535.0, 0, 1)
        after, _ = al.load(args.pair[1]); after = np.clip(after / 65535.0, 0, 1)
        H, W, _ = before.shape
        if args.crop:
            cx, cy, half = args.crop
        else:
            cx, cy, half = W // 2, H // 2, min(H, W) // 6
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"M101_compare_{args.name}.png")
        stitch(before, after, args.label_before, args.label_after, cx, cy, half, path)
        print(f"wrote {os.path.relpath(path)}")
        return
```
Note: `args.stretched` stays a positional but is unused in `--pair` mode; pass any existing FITS path to satisfy it, or make it optional — set `nargs="?"` on the `stretched` positional so `--pair` can run without it:
```python
    ap.add_argument("stretched", nargs="?", help="...")   # was required
```

- [ ] **Step 2: Produce the two stretched inputs and the comparison**

Run:
```bash
IN="data/Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit"
make run FITS="$IN" V=pcc          # PCC path (03 gets --original via run_pipeline)
# gentle-WB variant: rebuild 03 with --no-pcc, then stretch
.venv/bin/python scripts/03_color.py work/02_bg.fit work/03_nopcc.fit --no-pcc
.venv/bin/python scripts/04_stretch.py work/03_nopcc.fit work/04_nopcc.fit
.venv/bin/python scripts/compare.py --pair work/04_nopcc.fit work/04_stretch.fit \
  --name pcc --label-before "gentle WB" --label-after "PCC" --crop 1040 1936 300
```
Expected: writes `output/M101_compare_pcc.png`. Open it: star colors should be better differentiated (blue/white/orange) under PCC vs the flatter gentle-WB rendering.

- [ ] **Step 3: Run the test suite (guard against regressions)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (5 passed) — compare.py has no unit tests but ensure nothing imports-broke.

- [ ] **Step 4: Commit**

```bash
git add scripts/compare.py
git commit -m "PCC: add generic pair-compare mode; PCC before/after"
```

---

### Task 9: Docs + full end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:** none (documentation + verification).

- [ ] **Step 1: Document PCC in README**

In `README.md`, add a row to the pipeline table for the color step noting PCC, and a short "Photometric color calibration" paragraph under Notes: PCC queries Gaia DR3 online, needs internet, falls back to gentle WB, writes a `_pcc_diagnostic.png`. Mention `make test`.

- [ ] **Step 2: Document in CLAUDE.md**

In `CLAUDE.md`, under Commands add `make test`; under the pipeline description note that step 03 does Gaia PCC (measured on the original stack, applied to the working image) with gentle-WB fallback.

- [ ] **Step 3: Full pipeline run with PCC**

Run:
```bash
make test
make run FITS="data/Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit" V=pcc-final
```
Expected: tests pass; pipeline prints PCC gains + star count at step 03; writes `output/..._pcc-final.{tif,png}` and `output/..._pcc-final*_pcc_diagnostic.png`.

- [ ] **Step 4: Visual confirmation**

Open `output/M101_compare_pcc.png` and the final image. Confirm star colors look physically plausible (hot stars blue-white, cool stars orange) and the background stays neutral. If PCC fell back to gentle WB (e.g. offline), note that in the run summary rather than claiming PCC ran.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "PCC: document Gaia calibration and make test target"
```

---

## Notes for the executor

- **Internet dependency:** Tasks 5, 6, 8, 9 hit Gaia. If offline, the code path still works via fallback; verify the fallback branch and clearly report that PCC did not actually run, rather than asserting success.
- **`work/` intermediates:** several verification steps assume `work/02_bg.fit` exists. If missing, run `make run FITS="<the M101 stack>" V=tmp` once to populate `work/`.
- **Do not push / open a PR** until the human confirms. All commits stay on `feature/compare-tool`.
