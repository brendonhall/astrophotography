# Starless Galaxy Workflow — Design

**Date:** 2026-07-25
**Status:** Approved (design shape); pending spec review
**Branch:** `feature/starless-workflow`

## Goal

Recreate, as scriptable Python, the SetiAstroSuitePro "galaxy" finishing workflow:

> stretched image → remove stars → sharpen the starless galaxy → denoise → recombine stars

This is an **alternate finishing path** for the existing pipeline. It operates on the
**stretched, nonlinear** image (the output of `04_stretch.py`) and produces a finished,
versioned image the same way `05_finish.py` does. The existing `05_finish.py` stays
untouched as the simple default; this workflow is opt-in.

## Why this shape

Star removal and AI denoise are the two genuinely-AI steps in the SASP chain. We resolved
each:

- **Star removal → StarNet2 CLI** (installed at `~/StarNet2/`, v2.5.4, Apple Silicon
  CoreML). It is the one external dependency; we isolate it behind a wrapper module
  exactly the way `pcc.py` isolates the Gaia/astroquery dependency for step 03.
- **Denoise → classical Python** (the TV-luma + chroma denoise already in
  `astrolib.finish`). No Cosmic Clarity CLI exists on the machine; a real, tunable
  classical denoise keeps the whole chain in one command.

The remaining steps — layer separation, sharpening the starless layer, and screen
recombination — are deterministic NumPy.

### Key discovery that shaped the design

StarNet2's `--unscreen` output writes the **star layer directly** (the exact screen-blend
inverse of the starless image). Verified empirically: `screen(starless, stars)`
reconstructs the input to mean error 6e-5. This means:

- We do **not** derive `stars = original − starless` ourselves (the failure-prone step,
  where bright-star halos leave residuals).
- Recombination is exact: process the starless layer, then screen it back with the
  untouched star layer.

We use **float32 FITS** for the StarNet2 round-trip (native support), avoiding 16-bit TIFF
quantization.

## Architecture

Approach A (chosen): a thin CLI wrapper module + one orchestrating finish script.

```
scripts/starnet.py            # isolates the StarNet2 CLI dependency (like pcc.py)
scripts/05b_starless_finish.py# orchestrates the layer workflow (thin, like other steps)
```

### Data flow

Input: stretched nonlinear RGB, [0,1] (from `work/04_stretch.fit`).

```
stretched RGB [0,1]
   │
   │  starnet.remove_stars()  →  writes temp float32 FITS, runs starnet2 -n, reads back
   ├──────────────► starless RGB [0,1]
   │                star layer RGB [0,1]      (from --unscreen; NOT derived by subtraction)
   │
   ├─ process starless:
   │     gentle luma unsharp (radius, amount)      ← default gentle; tunable; disable-able
   │     → TV-luma + chroma denoise (astrolib.finish pieces)
   │     → saturation
   │
   └─ recombine:  result = screen(starless_processed, stars)
                         = 1 − (1 − starless_processed)(1 − stars)
   │
   └─ export versioned .tif (16-bit) + .png + preview   (same convention as 05_finish)
```

Diagnostic PNGs saved alongside output (versioned): the **star layer** and the
**starless-before/after-processing**, so residual halos or over-sharpening are visible.

## Components

### `scripts/starnet.py`

Single responsibility: turn an in-memory [0,1] RGB image into (starless, stars) using the
StarNet2 CLI. Depends only on: the CLI binary, astropy (FITS temp I/O), numpy.

```
class StarNetError(Exception): ...

def find_binary() -> str
    # $STARNET2_CLI env var, else ~/StarNet2/starnet2 default.
    # Raise StarNetError with install instructions if missing/not executable.

def remove_stars(img01, binary=None, stride=256, tmpdir=None) -> (starless01, stars01)
    # 1. write img01 as float32 FITS (3,H,W) to a temp dir
    # 2. run: <binary> -i in.fits -o starless.fits -n stars.fits -s <stride> --machine-progress -q
    # 3. read both back as (H,W,3) float32 in [0,1]
    # 4. clean up temp files
    # Raise StarNetError on nonzero exit or missing outputs (message includes stderr tail).
```

Notes:
- Input to StarNet2 must be ≥ 512×512 (our crops satisfy this; wrapper asserts and raises
  a clear error otherwise).
- StarNet2 clamps/expects [0,1]; wrapper clips before writing.
- Progress: parse `--machine-progress` JSON-lines from stderr into `print()` lines so the
  pipeline shows "starnet: 50%" like other steps. (Best-effort; failure to parse never
  fails the run.)

### `scripts/05b_starless_finish.py`

Thin orchestrator. Tuning constants at the top (matches the other step scripts):

```
SHARPEN_AMOUNT = 0.5     # unsharp-mask amount on starless luma (0 = off)
SHARPEN_RADIUS = 2.0     # gaussian radius (px) for the unsharp mask
LUMA_DENOISE   = 0.012   # reuse astrolib.finish defaults
CHROMA_DENOISE = 4.0
SATURATION     = 1.20
STRIDE         = 256
```

`main(infile, outfile_base, stride=..., no_sharpen=False)`:
1. `img, hdr = al.load(infile)`; to [0,1].
2. `starless, stars = starnet.remove_stars(img01, stride=STRIDE)`.
3. Process starless: unsharp luma → `al.finish(..., scnr=False)` for denoise+saturation.
   (SCNR off here — green cast was already handled upstream; revisit if needed.)
4. `result = screen(starless_proc, stars)`.
5. Export versioned `.tif`/`.png`/`_preview.png` + `_starlayer.png` + `_starless.png`
   diagnostics.

Reused helpers live in `astrolib` (screen blend + unsharp added there so both the script
and tests share one source of truth).

### `astrolib.py` additions

```
def screen(a, b):                    # 1 - (1-a)(1-b), clipped to [0,1]
def unsharp_luma(img01, amount, radius):  # unsharp mask applied to luminance only
```

### Wiring

- `run_pipeline.sh`: add `--starless` flag. When set, replace the `05 finish` step with
  `05b_starless_finish.py` (steps 01–04 unchanged). Default behavior unchanged.
- `Makefile`: add `make run-starless FITS=... [V=label]`.
- `README.md`: document the alternate path, the StarNet2 prerequisite, and the
  `$STARNET2_CLI` override.

## Error handling

- **Missing binary:** `starnet.find_binary()` raises `StarNetError` with the exact install
  hint (download URL + `~/StarNet2/` layout + `$STARNET2_CLI`). The finish script prints it
  and exits non-zero (this path is opt-in, so failing loudly is correct — unlike PCC, there
  is no meaningful classical fallback the user asked for).
- **CLI nonzero exit / missing output:** `StarNetError` with stderr tail.
- **Image too small:** clear error before invoking the CLI.
- **Sharpen amplifying noise:** default gentle + `--no-sharpen` escape hatch + denoise runs
  after sharpen; diagnostics make over-sharpening visible.

## Testing

Pure-NumPy unit tests (no binary needed):
- `screen()` identity: `screen(starless, stars)` reconstructs a known composite; screen is
  commutative and stays in [0,1].
- `unsharp_luma()`: increases high-frequency contrast on luma, leaves a flat field
  unchanged, does not touch hue.

Wrapper tests with the CLI **mocked** (monkeypatch `subprocess.run` + write fake FITS):
- `remove_stars()` writes a valid float32 FITS input, invokes the expected argv, reads
  outputs back into [0,1] (H,W,3).
- `find_binary()` honors `$STARNET2_CLI`, falls back to the default, raises `StarNetError`
  when absent.
- Nonzero CLI exit → `StarNetError` with stderr included.
- Sub-512 input → `StarNetError`.

Optional (skipped if binary absent, marked): one real end-to-end run on a small synthetic
field asserting starless max < input max and the screen identity holds — guarded by
`pytest.mark.skipif(not binary)` so `make test` stays offline/dependency-free.

## Out of scope (YAGNI)

- AI denoise (Cosmic Clarity) — using classical denoise per decision.
- Separate star-layer *processing* (e.g., star reduction/shrink) — recombine stars as-is
  for v1; star reduction is a natural follow-up once the base workflow is validated.
- Mono/`--upsample` paths — OSC RGB only for now.

## Success criteria

1. `make run-starless FITS=<stretched-master>` produces a versioned finished image plus
   star-layer and starless diagnostics, in one command.
2. Result is comparable side-by-side against `05_finish` output (via existing `compare.py`
   `--pair`) so the user can judge whether the starless route wins on this data.
3. `make test` passes with the new tests and no new external dependency at test time.
