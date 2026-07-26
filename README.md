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
| 3 | `03_color.py`      | Neutralize background + photometric color calibration (Gaia DR3), falling back to gentle white balance |
| 4 | `04_stretch.py`    | Linked midtones stretch: linear → nonlinear |
| 5 | `05_finish.py`     | SCNR green, denoise, saturation → 16-bit TIFF + PNG |

`scripts/astrolib.py` holds shared helpers (FITS I/O, STF autostretch, source
masking). Tuning parameters live as constants at the top of each step script.

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
diagnostics.

The starless layer is denoised with a background-aware (masked) denoise: with
no stars left to protect, the empty sky is smoothed aggressively while a
feathered luminance mask keeps the galaxy itself gentle and sharp (see
`astrolib.masked_denoise`). Sharpening now defaults **off** — the experiment
showed sharpening the sky just adds grain — set `SHARPEN_AMOUNT > 0` in
`scripts/05b_starless_finish.py` to re-enable it (it applies globally, sky
and galaxy alike).

## Re-stacking subframes

The SeeStar exports only its on-device stack, whose built-in rejection can leave
satellite/plane trails and hot pixels behind. Re-stacking the **individual
subframes** with proper sigma-clip rejection is the real fix — and it produces a
cleaner linear FITS to feed the pipeline above.

**1. Capture the subs.** In the SeeStar app, enable **Settings → Advanced
Settings → "Save each frame in enhancing"** *before* imaging (it's forward-only —
past sessions have no subs). Each target then gets a second `<target>-sub` folder
of per-exposure FITS alongside the stack.

**2. Copy them over.** Transfer via Wi-Fi Station Mode (app → Wi-Fi → Station Mode
→ note the SeeStar's IP → Finder → *Connect to Server*), and drop the
`<target>-sub` folder into `data/`.

**3. Re-stack with Siril.** The `siril/` scripts register and sigma-clip-stack the
subs into a single linear FITS in `data/` (no darks/flats/bias — SeeStar subs are
dark-subtracted on-device):

```sh
# Check whether the subs are color (NAXIS=3) or raw Bayer (NAXIS=2 + BAYERPAT):
make inspect FITS="data/M 101-sub/<one-sub>.fit"

# Color subs (usual case):
./siril/restack.sh "data/M 101-sub" M101_restacked
# Raw Bayer subs: add --cfa to debayer during conversion:
./siril/restack.sh --cfa "data/M 101-sub" M101_restacked

# Then run the pipeline on the result:
make run FITS="data/M101_restacked.fit" V=restack
```

The wrapper writes a **uniquely named** FITS and refuses to overwrite an existing
one. `restack_seestar.ssf` handles color subs; `restack_seestar_cfa.ssf` debayers
raw Bayer subs first. Rejection defaults to winsorized sigma-clip (`rej 3 3`).

### Siril GUI gotchas (color calibration)

Two things that make Siril's Photometric Color Calibration (PCC) look broken when
it isn't:

- **Linked vs. unlinked autostretch.** Siril's *display* Autostretch has a
  linked/unlinked toggle. **Unlinked** (per-channel) normalizes each of R/G/B to
  its own range, which mathematically **cancels any constant per-channel scaling**
  — so a color-calibration change is *invisible* in the preview, and PCC looks
  like it did nothing. Switch the display to **linked** to judge color. (This is
  a display setting only; it never changes the pixels.)
- **PCC does two separate jobs.** (1) **White-balance factors** (Siril's `K0/K1/K2`)
  correct *star colors* so a sun-like star reads white; (2) **background
  neutralization** (`B0/B1/B2`) flattens the *background* color. Don't judge PCC
  by "did the background change" — a raw SeeStar background can be blue- or
  green-dominant, and the factors alone can shift it either way. Judge success by
  **star colors after stretching**: a natural spread of white / blue-white /
  yellow-orange stars, not a uniform tint. The Siril console also prints how many
  stars matched and the factors applied — that's the objective confirmation PCC ran.

Note this is Siril-GUI behavior; the pipeline's own PCC (`03_color.py`, below)
is a separate implementation.

## Layout

```
scripts/        pipeline code (tracked)
siril/          Siril .ssf scripts + wrapper for re-stacking subframes (tracked)
data/           stacked FITS exports (local only, git-ignored)
output/         processed TIFF/PNG (local only, git-ignored)
work/           intermediate stage files (local only, git-ignored)
CONCEPT.md      project intent / goals
CLAUDE.md       guidance for Claude Code
```

## Notes & next steps

- The SeeStar exports only the finished stack, so satellite trails that survive
  its rejection can't be cleanly removed. Re-stacking the individual subframes
  with sigma-clip rejection is the real fix — see **Re-stacking subframes** above.
- The stacks carry a full WCS, which step 3 now uses for **photometric color
  calibration** against the Gaia DR3 catalog — see below.
- Tools also installed locally for GUI/alternative workflows: **Siril**
  (`siril-cli`) and **SetiAstroSuitePro**.

### Photometric color calibration

`03_color.py` measures per-channel color gains by matching detected stars in
the **original** stacked FITS (which carries the WCS) against **Gaia DR3**
colors (`bp_rp`) queried live via `astroquery`, then applies those gains to the
working image. This needs **internet access** — the query goes out to the
Gaia archive over the network — and takes a couple of minutes on a typical
stack. It writes a `..._pcc_diagnostic.png` alongside the FITS output, plotting
matched-star color vs. catalog color so the fit can be sanity-checked.

If the query fails, too few stars match (fewer than `MIN_STARS`), or the
network is unavailable, the step **falls back automatically** to the simpler
gentle white-balance approach (equalizing mid-signal levels across channels)
that was the original default — the pipeline never hard-fails for lack of
internet.

Run `make test` to exercise the PCC and fallback code paths (offline-safe unit
tests, no network calls) before trusting the output of a live run.
