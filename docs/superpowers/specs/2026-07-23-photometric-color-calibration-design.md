# Photometric Color Calibration (PCC) — Design

**Date:** 2026-07-23
**Branch:** `feature/compare-tool`
**Status:** Approved design, pending spec review

## Goal

Derive per-channel color-correction gains (R and B relative to G) from the Gaia
DR3 catalog so that star colors in a processed SeeStar S30 Pro image are
physically faithful, replacing the ad-hoc gentle white-balance currently in
pipeline step 03 (`03_color.py`). The result is repeatable across targets and
anchored to a fixed physical reference rather than the star population of a
single frame.

## Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Catalog source | **Gaia DR3, queried online** via `astroquery` | Gold standard; deepest/most complete at this focal length. We have a full SIP WCS + RA/Dec center to query against. |
| Calibration model | **Empirical** (regress measured star color vs Gaia color) | The SeeStar applies its own internal color processing, so the data is not raw linear sensor response. Full spectral SPCC (QE × filter × stellar spectra) assumptions do not hold. |
| White point | **Parameterized, default G2V/solar** (`ref_bp_rp = 0.82`) | Solar white point is physically faithful and repeatable across targets; exposing it as a parameter lets us experiment at ~no cost. |
| Failure behavior | **Fall back to gentle WB** with a loud warning | Scripted pipeline must always produce an image; a flaky network or sparse field should degrade gracefully, not break the run. |
| Tests | **In** — bootstrap `tests/` with pytest | First tests in the repo; cover the pure-math units. |
| Before/after | **Reuse `compare.py`** | Consistent comparison tooling; PCC before/after as a new comparison. |

## Non-goals

- Full spectrophotometric calibration (sensor QE / filter transmission curves).
- Committed/offline Gaia cache (we chose fresh online query each run).
- Recalibrating anything but color (no flux/magnitude zero-point science output).

## Architecture

### New module: `scripts/pcc.py`

Importable library + standalone CLI. Small, independently testable units:

1. **`detect_stars(img_adu) -> Table`**
   - Build luminance from the (H,W,3) linear image.
   - `photutils.detection.DAOStarFinder` on background-subtracted luminance;
     threshold from sigma-clipped image stats, FWHM estimated from the data.
   - Per-channel aperture photometry (`photutils.aperture`) at each centroid,
     with a local background annulus, producing instrumental `r, g, b` flux.
   - Reject: saturated (near 65535), low-SNR/faint, and near-edge stars.

2. **`query_gaia(hdr, radius_deg, mag_limit=16, row_limit=50000) -> Table`**
   - Cone search `gaiadr3.gaia_source` via `astroquery.gaia` around header
     RA/Dec. Field of view ≈ 2.2° × 4.0° (FL 160 mm, 2.9 µm px, 2160×3840),
     so default radius ≈ 2.3°.
   - Keep rows with valid `bp_rp`, within `mag_limit`, good astrometry.
   - Returns `ra, dec, phot_g_mean_mag, bp_rp`.

3. **`cross_match(stars, gaia, wcs, tol_arcsec=5) -> Table`**
   - Detected pixel coords → sky via the SIP WCS (`astropy.wcs.WCS`).
   - Nearest-neighbor match within `tol_arcsec`; keep unique best pairs.

4. **`solve_gains(matched, ref_bp_rp=0.82) -> (gains, diagnostics)`**
   - Instrumental colors `cr = r/g`, `cb = b/g`.
   - Sigma-clipped linear regression of `cr`, `cb` vs Gaia `bp_rp`.
   - Evaluate fits at `ref_bp_rp` → expected instrumental ratios `cr0, cb0`
     for a solar star. Gains: `gain_r = 1/cr0`, `gain_b = 1/cb0`, `gain_g = 1`.
   - Returns gains + diagnostics (n matched, fit slopes/intercepts, residual
     scatter).

5. **`photometric_calibration(img_adu, hdr, ref_bp_rp=0.82, min_stars=30) -> (gains, report)`**
   - Orchestrates 1–4. Raises `PCCError` on: no WCS in header, `astroquery`
     unavailable, network/query failure, `< min_stars` matches, or gains
     outside a plausible sanity band (e.g. [0.5, 2.0]).

6. **`apply_gains(img_adu, gains, pedestal) -> img_adu`**
   - `out_c = (img_c - pedestal) * gain_c + pedestal` so the neutral background
     pedestal is preserved and only signal is rescaled.

7. **Diagnostic plot** — `save_diagnostic(matched, fits, gains, path)` writes a
   color-color scatter (`cr`, `cb` vs Gaia `bp_rp`) with the fit lines and the
   solar white point marked → `output/<name>_pcc_diagnostic.png`.

`PCCError` is a module-level exception used for all graceful-fallback cases.

### Integration: `scripts/03_color.py`

- Keep background neutralization (still required — removes the background color
  cast before gain application).
- Replace the gentle mid-signal white-balance block with:
  ```
  try:
      gains, report = pcc.photometric_calibration(img, hdr, ref_bp_rp=REF_BP_RP,
                                                   min_stars=MIN_STARS)
      img = pcc.apply_gains(img, gains, PEDESTAL)
  except pcc.PCCError as e:
      warn(f"PCC unavailable ({e}); falling back to gentle white balance")
      img = gentle_white_balance(img)   # existing logic, extracted to a function
  ```
- New constants: `REF_BP_RP = 0.82`, `MIN_STARS = 30`.
- CLI flag `--no-pcc` to force the gentle-WB path.
- Extract the current gentle-WB code into a `gentle_white_balance()` function so
  it is reusable as the fallback.

### Data flow position

PCC runs inside step 03, on **linear, background-extracted, background-neutralized**
data (input `work/02_bg.fit`). Photometry must be linear, so this placement is
correct; it happens before the nonlinear stretch (step 04).

## Dependencies

- Add `astroquery>=0.4.7` to `requirements.txt`.
- Add `pytest>=8.0` to `requirements.txt` (dev/test).

## Testing

New `tests/test_pcc.py` (pytest):

- **`solve_gains`**: synthetic matched table with a known color law
  (`cr = a·bp_rp + b`) → assert recovered gains match the closed-form
  expectation within tolerance; assert `gain_g == 1`.
- **`apply_gains`**: assert background pedestal is preserved and channels scale
  as specified; assert idempotence at `gains = (1,1,1)`.
- **`cross_match`**: synthetic pixel/sky coords via a trivial WCS → assert
  correct pairs within tolerance and correct rejection beyond it.
- Network-dependent `query_gaia` is kept thin and is not unit-tested (covered by
  the integration run).

Run: `.venv/bin/pytest` (add a `test` target to the Makefile).

## Verification plan

1. Run the full pipeline (`make run`) on the M101 stack with PCC active.
2. Confirm: matched-star count ≥ `MIN_STARS`, gains within the sanity band,
   diagnostic plot looks like a clean color-color relation.
3. Use `compare.py` to produce a PCC before/after (gentle WB vs PCC) color crop
   → `output/M101_compare_pcc.png`.
4. Sanity-check that star colors span a believable blue→white→orange range and
   the galaxy core/arms look physically plausible.

## Deliverables

- `scripts/pcc.py` (new)
- `scripts/03_color.py` (PCC integration + extracted gentle-WB fallback)
- `scripts/compare.py` (add a PCC before/after comparison)
- `tests/test_pcc.py` (new)
- `requirements.txt` (+ astroquery, pytest)
- `Makefile` (+ `test` target)
- Run artifacts (git-ignored): `output/*_pcc_diagnostic.png`, `output/M101_compare_pcc.png`

## Edge cases / failure modes

- No WCS / missing RA-Dec → `PCCError` → gentle-WB fallback.
- `astroquery` import error → `PCCError` → fallback (dependency should prevent this).
- Network error / Gaia timeout → `PCCError` → fallback.
- `< min_stars` matched → `PCCError` → fallback.
- Gains outside sanity band → `PCCError` → fallback (guards against a bad fit).
- All fallbacks emit a clear warning so a downgraded run is never silent.
