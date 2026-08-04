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
- `make run FITS="<path>" [V=label]` — run the full standard pipeline; output is always
  written under a **unique versioned name** (`output/<name>_<label>.{tif,png}`), never
  overwriting a prior run. Omitting `V=` uses a timestamp.
- `make run-starless FITS="<path>" [V=label]` — same, but the starless finish (remove
  stars → sharpen/denoise the starless galaxy → screen the stars back in).
- `make clean` — remove `work/` intermediates (including the flow cache)
- `make test` — run the pytest suite (offline-safe; no live Gaia queries)
- `python -m flow run (FLOW.json | --builtin linear|starless) --input X.fit --label v1`
  — run a flow graph directly (`--no-cache` to force recompute); also
  `python -m flow validate …` and `python -m flow schema` (the node palette).
  Run from the repo root with `scripts/` on `PYTHONPATH` (as `run_pipeline.sh` does).

## Processing pipeline

The pipeline is layered — **read `docs/ARCHITECTURE.md` first** for the full map. In short:

- **Numeric core** (`scripts/astrolib.py`, `pcc.py`, `starnet.py`) — pure functions
  (FITS I/O, background fit, linked stretch, finish, screen, masked denoise, PCC, StarNet2).
- **Stages** (`scripts/stages/`) — one small self-describing class per operation, with a
  typed parameter schema + named input/output ports, wrapping the core. A registry
  (`stages.list_stages()`) exposes them; images flow as an `Image` payload carrying
  pixels, a `space` tag (`linear-adu`/`nonlinear`), **and the full FITS header (so WCS
  travels with the data)**.
- **Flow** (`scripts/flow/`) — connects stages into a DAG, (de)serializes it to JSON,
  validates it, and executes it with content-addressed caching in `work/cache/`. The
  built-in `linear`/`starless` flows reproduce the standard pipelines; `python -m flow`
  is the CLI.

The numbered scripts (`scripts/01_crop.py` … `05b_starless_finish.py`) are now **thin
shims over the stages**, kept as standalone CLIs. `run_pipeline.sh` (and `make run` /
`make run-starless`) is unified over the flow executor.

**Color calibration** (`color_calibrate` stage / `03_color.py`) does Gaia DR3 photometric
color calibration via `scripts/pcc.py`. Because WCS now travels in the payload, it
measures on its own input (no `--original` needed; an optional `reference` port can
override the frame it measures on). PCC needs internet (`astroquery`) and writes a
`..._pcc_diagnostic.png`; on query failure or too few matched stars it falls back to a
gentle white-balance. To add a new step, drop a `@register`ed `Stage` into
`scripts/stages/` — it auto-registers and becomes usable in flows and `flow schema`.

**Output convention:** never overwrite a processed image — every variant gets its own
versioned filename (`output/<name>_<label>`) so results compare side by side; the flow
executor enforces this. External tools: StarNet2 (`~/StarNet2`), Cosmic Clarity
(`~/CosmicClarity`, via `tools/ccdenoise.sh` / `tools/portfolio_finish.py`).

## Working notes

- FITS files are large (tens of MB each) and binary — read them with `astropy.io.fits`,
  not text tools. Inspect the header before processing; exposure, filter, gain, and
  target all live there and drive the right processing choices.
- Data and rendered images (`data/`, `output/`, `work/`) are git-ignored — only scripts,
  code, and docs are committed.
