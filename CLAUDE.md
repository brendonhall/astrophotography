# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal astronomy / astrophotography project, just getting started. Not a software
package — it's a working space for turning raw telescope captures into finished images,
plus any Python code written along the way to process them. See `CONCEPT.md` for the
owner's intent.

The user is a computational scientist, fluent in Python and scientific software. They
want to learn the hobby, are open to advanced techniques, and are doing this
recreationally. Coach and explain trade-offs; don't just hand over black-box recipes.

## Hardware & data

- Capture device: **SeeStar S30 Pro** smart telescope (ZWO). It does its own on-device
  alignment and live stacking, then exports stacked frames.
- Raw exports land in `data/`. Files are **FITS** (`.fit`), the standard astronomy image
  format — a header of key/value metadata followed by a floating-point pixel array.
- Filename convention from the SeeStar, e.g.
  `Stacked_283_M 101_30.0s_IRCUT_20260723-020000.fit`:
  `Stacked_<frame count>_<target, e.g. Messier object>_<sub-exposure>_<filter>_<timestamp>`.
  So that example is 283 stacked 30s subs of M101 with the IR-cut filter.

## Installed tools

Two GUI apps are installed; both also have command-line / scripting entry points:

- **Siril** — `/Applications/Siril.app/Contents/MacOS/siril-cli` (headless, scriptable via
  `.ssf` scripts). Core stacking, registration, background extraction, color calibration,
  stretching.
- **SetiAstroSuitePro** — `/Applications/SetiAstroSuitePro.app`. GUI suite for gradient
  removal, star removal, stretching, and AI-based tools.

## Python environment

- The project uses a **local `.venv`** (git-ignored, and tagged to skip Dropbox sync
  because venv binaries aren't portable across macOS/Linux). Build it with `make setup`,
  which installs `requirements.txt`. Use `.venv/bin/python` — the system Python 3.9.6 on
  `PATH` has no astronomy packages.
- Siril ships its own bundled Python 3.12
  (`/Applications/Siril.app/Contents/Frameworks/.../python3`) used for its internal
  scripting; don't rely on it as the project interpreter.

## Commands

- `make setup` — create `.venv`, install requirements
- `make inspect FITS="<path>"` — header + per-channel stats
- `make run FITS="<path>" [V=label]` — run the full pipeline; output is always written
  under a **unique versioned name** (`output/<name>_<label>.{tif,png}`), never overwriting
  a prior run. Omitting `V=` uses a timestamp.
- `make clean` — remove `work/` intermediates

## Processing pipeline

`scripts/` holds a numbered chain that turns a linear stacked FITS into a finished image:
crop → background/gradient removal → color calibration → nonlinear stretch → finish/export.
Each step reads a FITS and writes a FITS + preview PNG; intermediates go in `work/`.
`scripts/astrolib.py` has shared helpers (FITS I/O, STF autostretch, source masking).
Tuning parameters are constants at the top of each step script. See README.md for the
step-by-step table.

**Output convention:** never overwrite a processed image — every processing variant gets
its own versioned filename so results can be compared side by side. `run_pipeline.sh`
enforces this automatically.

## Working notes

- FITS files are large (tens of MB each) and binary — read them with `astropy.io.fits`,
  not text tools. Inspect the header before processing; exposure, filter, gain, and
  target all live there and drive the right processing choices.
- Data and rendered images (`data/`, `output/`, `work/`) are git-ignored — only scripts,
  code, and docs are committed.
