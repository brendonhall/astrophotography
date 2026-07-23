# Astrophotography

Data and code for processing astrophotography captures into finished images.
Captures come from a **SeeStar S30 Pro** smart telescope, which stacks, registers,
and plate-solves on-device and exports stacked **FITS** frames.

This repo holds **only scripts, code, and documentation**. Raw data and rendered
images are kept local (and in Dropbox) — never committed to git.

## Setup

Requires Python 3.9+ and standard build tools.

```sh
make setup          # creates .venv and installs requirements.txt
```

The virtualenv (`.venv/`) is git-ignored and tagged to skip Dropbox sync, because
its binaries aren't portable between macOS and Linux. On each machine, run
`make setup` to build a local environment from `requirements.txt`.

## Usage

```sh
# Inspect a FITS file (header cards, geometry, per-channel stats)
make inspect FITS="data/Stacked_283_M 101_....fit"

# Run the full pipeline (writes a versioned image to output/)
make run FITS="data/Stacked_283_M 101_....fit" V=v3-tweaks
# -> output/Stacked_283_M_101_..._v3-tweaks.{tif,png}
```

If you omit `V=`, the run is labeled with a timestamp. **Every run produces a
uniquely named output — nothing is ever overwritten**, so processing variants can
be compared side by side.

## Pipeline

The processing chain (in `scripts/`) turns a linear stacked FITS into a finished
image. Each numbered step reads a FITS and writes a FITS plus a preview PNG.

| Step | Script | Purpose |
|------|--------|---------|
| — | `inspect_fits.py` | Report header, geometry, per-channel statistics |
| — | `render.py`       | Render any FITS to PNG (with optional STF autostretch) |
| 1 | `01_crop.py`       | Trim the dither margin |
| 2 | `02_background.py` | Remove the light-pollution gradient (masked polynomial fit) |
| 3 | `03_color.py`      | Neutralize background + gentle white balance |
| 4 | `04_stretch.py`    | Linked midtones stretch: linear → nonlinear |
| 5 | `05_finish.py`     | SCNR green, denoise, saturation → 16-bit TIFF + PNG |

`scripts/astrolib.py` holds shared helpers (FITS I/O, STF autostretch, source
masking). Tuning parameters live as constants at the top of each step script.

## Layout

```
scripts/        pipeline code (tracked)
data/           stacked FITS exports (local only, git-ignored)
output/         processed TIFF/PNG (local only, git-ignored)
work/           intermediate stage files (local only, git-ignored)
CONCEPT.md      project intent / goals
CLAUDE.md       guidance for Claude Code
```

## Notes & next steps

- The SeeStar exports only the finished stack, so satellite trails that survive
  its rejection can't be cleanly removed. Re-stacking the individual subframes
  (if saved) with sigma-clip rejection is the real fix.
- The stacks carry a full WCS, so **photometric color calibration** against a star
  catalog (Gaia) is a natural upgrade over the current white-balance approximation.
- Tools also installed locally for GUI/alternative workflows: **Siril**
  (`siril-cli`) and **SetiAstroSuitePro**.
