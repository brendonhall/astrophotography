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

The numbered scripts above are thin CLI shims: each just loads a FITS, calls
one `stages.Stage` with a params dict, and saves the result. See **The
`stages/` package** below for what actually does the work.

### The `stages/` package

The actual processing logic lives in `scripts/stages/`, not in the numbered
scripts. Each pipeline step is a small, self-contained **Stage** — a class
with typed inputs, typed outputs, and typed parameters — and the numbered
scripts (and `05b_starless_finish.py`) are thin shims that wire FITS-in /
FITS-out CLIs around them. This exists so the same processing steps can
eventually be driven by something other than a CLI (e.g. a node-graph GUI)
without rewriting the image math.

- **`Image`** (`stages/image.py`) is the payload that flows between stages:
  a `pixels` array, a `header` (which carries the WCS, when present), and a
  `space` tag — `Space.LINEAR_ADU` (crop through color calibration, float ADU)
  or `Space.NONLINEAR` (after the stretch step, `[0,1]`). Because the header
  travels with the image end to end, any stage that needs the WCS (PCC) can
  read it straight off its own input — nothing has to be threaded in
  separately from the original stacked file. `Image.replace(...)` returns a
  modified copy (pixels/space/header), keeping stages side-effect-free.
- **`Stage`** (`stages/base.py`) is the base class every step subclasses:
  `INPUTS`/`OUTPUTS` are lists of named **`Port`**s (each optionally
  constrained to a `Space`, and markable `required=False`), and `PARAMS` is a
  list of typed **`Param`**s (`float`/`int`/`bool`/`enum`/`str`, with
  `default`, `min`/`max`/`step`, `choices`, `label`, `help`). Calling
  `stage.run(inputs, params)` coerces and range-checks the params, validates
  the inputs against the declared ports (presence + space), then dispatches
  to `apply(inputs, params) -> dict` (the actual per-stage logic, keyed by
  output port name). `Param`/`Port` are plain frozen dataclasses (stdlib only,
  no pydantic), and `Stage.schema()` serializes a stage's id/label/description
  plus its ports and params to a JSON-safe dict.
- **Registry** (`stages/registry.py`): every stage class is decorated with
  `@register`, which files it by its `id` in a module-level dict.
  `stages.list_stages()` returns `[cls.schema() for cls in registry]` — the
  full palette of available building blocks (ids, ports with their spaces,
  and typed parameter schemas) as plain JSON-serializable data. This is the
  contract a future GUI would read to populate a node palette and generate
  parameter forms; see `import json, stages; json.dumps(stages.list_stages())`.
  There are 12 registered stages today: `crop`, `background_extract`,
  `color_calibrate`, `stretch`, `finish`, `saturate`, `masked_denoise`,
  `unsharp_luma`, `remove_stars`, `screen_recombine`, `export_image`,
  `preview_sink` — one module apiece under `stages/` (`geometry.py`,
  `background.py`, `color.py`, `stretch.py`, `finish.py`, `denoise.py`,
  `stars.py`, `export.py`), auto-imported by `stages/__init__.py` so
  importing `stages` is enough to populate the registry.
- **`stages/io.py`** has the FITS load/save helpers used by the shims
  (`load_fits`/`save_fits`), plus `crop_header`, which keeps `CRPIX1/2` (and
  therefore the WCS) correct when a stage trims pixels off the array.

Practically: `03_color.py`'s `ColorCalibrateStage` declares an `image` input
port and an optional `reference` port (defaulting to `image` itself if not
given) — so PCC runs on whatever WCS-bearing header is attached to its input,
which is why the pipeline no longer needs a separate `--original` pass-through
of the raw stacked file (see **Photometric color calibration** below).

### The `flow/` package (graph executor)

`scripts/flow/` is a headless DAG executor that runs `stages/` building blocks
wired into an arbitrary graph, instead of a fixed numbered chain. It's the
engine behind `run_pipeline.sh`, and the foundation a future node-graph GUI
would sit on top of.

- **Graph model** (`flow/graph.py`): `Graph(nodes, edges, name)`, where each
  `Node(id, type, params)` names a `stages` registry id plus its param
  values, and each `Edge(id, Endpoint(node, port), Endpoint(node, port))`
  wires one node's output port to another's input port. `Graph.to_json()` /
  `Graph.from_json()` round-trip a graph to plain JSON — a UI-authored graph
  and a hand-written one are the same shape. Unknown top-level keys (e.g. a
  future GUI's node positions under `"ui"`) are preserved through a
  round-trip rather than dropped.
- **Validation** (`flow/validate.py`): `validate(graph) -> [Issue]` checks
  structural rules — unknown stage/port ids, duplicate node ids, space
  mismatches across an edge, missing required inputs, cycles, etc. — as
  errors, plus a dead-branch warning for each output port that has no
  consumer (rather than tracing whether it eventually reaches a sink).
  `flow validate` (see below) exits non-zero if any `Issue` is an error;
  warnings surface too — `flow run` prints them before its `ran/cached`
  summary.
- **Executor** (`flow/executor.py`): `run(graph, input_path, label, work_dir="work", out_dir="output", cache=True) -> RunReport`
  validates the graph, resolves `{input}`/`{out}`/`{work}` tokens in node
  params, topologically sorts the nodes, and runs each one's `stages.Stage`
  in order, threading `Image` payloads along the edges. Nodes with output
  ports are cache-checked/stored (see below); sink nodes (`export_image`,
  `preview_sink` — anything with no declared `OUTPUTS`) always run, since
  they write files rather than pass along a payload. If validation fails, a
  `FlowError` is raised before any node runs.
- **Caching** (`flow/cache.py`): each node's outputs are cached under
  `work/cache/<recipe_hash>__<port>.fits`, content-addressed by
  `recipe_hash(node, input_hashes)` — a hash of the node's type, its params,
  and the hashes of *its own inputs* (so changing an upstream node
  invalidates everything downstream of it, transitively). The `load` source
  node instead hashes the input file's mtime+size (`file_sig`), so editing
  the input FITS also busts the cache. Re-running the same graph on the same
  input reuses every node whose recipe hash is unchanged — only the nodes
  downstream of an actual change re-run. Pass `--no-cache` to force a full
  re-run, or `make clean` to blow away `work/` (and the cache with it).
- **Built-in graphs** (`flow/builtins.py`): `linear_flow()` and
  `starless_flow()` are Python-built `Graph`s reproducing the standard
  `01`-`05` chain and the `05b` starless finish, respectively — used by
  `run_pipeline.sh` via `--builtin`, and as fixtures for the flow test suite.
- **CLI** (`python -m flow ...`, run with `scripts/` on `PYTHONPATH`):
  - `flow run (FLOW.json | --builtin linear|starless) --input PATH --label L [--no-cache]`
    — runs a graph end to end; prints `ran N, cached M` when done.
  - `flow validate (FLOW.json | --builtin linear|starless)` — runs `validate()`
    and prints each `Issue`; exits 1 if any is an error, 0 otherwise (also
    accepts a graph saved from a future GUI).
  - `flow schema` — prints `stages.list_stages()` as JSON: the full stage
    palette (ids, ports with their spaces, typed param schemas). This *is*
    the GUI node-palette contract — anything a node-graph editor needs to
    populate a palette and generate parameter forms comes from this one call.

**`run_pipeline.sh` is now a thin shim** over this executor: it does the venv
check and `--starless`/label-arg parsing it always did, then picks
`--builtin linear` or `--builtin starless` and delegates to
`python -m flow run`. `make run` / `make run-starless` are unchanged from the
user's point of view — same args, same versioned `output/` naming — but the
numbered scripts (`01`-`05b`) are no longer invoked directly; they remain in
the repo as thin single-stage CLI shims (useful for running one step by hand)
but the default path through `make run` goes through the flow graph.

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

### Cosmic Clarity denoise (optional)

AI denoise via SetiAstro's **Cosmic Clarity**, run as a native CLI. The SASpro
GUI's bundled PyTorch fails to import on this Mac (frozen-app packaging bug), so
Cosmic Clarity is installed standalone and driven by a wrapper:

    tools/ccdenoise.sh <input> <output> [strength] [mode]
    #  strength  0..1                        (default 0.5)
    #  mode      luminance | full | separate (default luminance)
    #  'full' also denoises chroma (color) noise — use it when the background
    #  has colored speckle (SeeStar OSC data usually does).

Feed a **stretched** (nonlinear) image — tif/fits/png; the CLI has no linear
flag. Apply it **late**, after color calibration and stretch (e.g. on a PCC'd
TIFF exported from Siril). Output is never overwritten. Runs on the Apple GPU
(MPS), ~1 min for a full-frame S30 image.

**One-time install:** clone `setiastro/cosmicclarity` to `~/CosmicClarity`, make
a native arm64 venv (`python3.13 -m venv .venv`), then
`pip install torch torchvision numpy tifffile astropy Pillow PyQt6 opencv-python-headless xisf lz4 zstandard`
(the repo's `requirements.txt` is incomplete). Model weights are **not** in the
repo — fetch from the GitHub releases, e.g.
`gh release download Linux --repo setiastro/cosmicclarity --pattern deep_denoise_cnn_AI3_6.pth --dir ~/CosmicClarity`.
Point `$COSMIC_CLARITY_DIR` at the install if it lives elsewhere.

### Portfolio finish

`tools/portfolio_finish.py` turns a **color-calibrated, stretched** image (e.g. a
PCC'd 16-bit TIFF exported from Siril) into a finished portfolio image in one
command. By default it reproduces the **M101 "v1"** look:

    Cosmic Clarity full denoise  ->  deepen + neutralize sky
      ->  global saturation (1.6)  ->  mild mid-contrast

Run it with the Cosmic Clarity venv (it needs `tifffile`/`cv2`/`starnet`):

    ~/CosmicClarity/.venv/bin/python tools/portfolio_finish.py <in> <out> [opts]

    # v1 recipe (full denoise + polish), full frame — the default:
    portfolio_finish.py pcc.tif  M101_finish.tif
    # input already denoised -> skip the denoise stage:
    portfolio_finish.py denoised.tif out.tif --denoise none
    # star-reduced + centered 1500 px portrait crop:
    portfolio_finish.py pcc.tif out.tif --star-reduce 0.5 --crop 1500

Options: `--denoise full|luminance|none` (default full), `--denoise-strength`
(0.5), `--saturation` (1.6), `--black-offset` (0.02, how far below the sky median
to set the black point), `--star-reduce F` (0=off; keeps star brightness F, and
switches saturation to a signal-masked version with chroma smoothing so the sky
stays clean), `--crop N` (centered N×N). Output is never overwritten; a
downsampled `_preview.png` is written alongside.

**Full end-to-end sequence** (what produced the M101 portfolio image):

1. **Capture** — SeeStar S30 Pro, enable *Save each frame in enhancing*; here
   367 × 30 s subs (IRCUT, gain 200) ≈ 3 hr.
2. **Re-stack** — `./siril/restack.sh "data/<target>-sub" <name>` (or the
   SeeStar preprocessing script): register + winsorized sigma-clip + plate-solve.
3. **Calibrate & stretch (Siril GUI)** — Background Extraction → Photometric
   Color Calibration → nonlinear stretch → export a **16-bit TIFF**.
4. **Finish** — `portfolio_finish.py <that TIFF> <out.tif>` (denoise + polish).

Steps 1–3 are per-target manual/GUI work; step 4 is the repeatable one-command
finish and is verified to reproduce the M101 v1 master pixel-for-pixel.

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
tools/          external-tool wrappers + finishing (ccdenoise, portfolio_finish) (tracked)
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

`03_color.py` measures per-channel color gains by matching detected stars
against **Gaia DR3** colors (`bp_rp`) queried live via `astroquery`, then
applies those gains to the working image. WCS now travels with the image
through the pipeline (see **The `stages/` package** below), so PCC measures
directly on the step's own input FITS — the old `--original` requirement is
gone. `--original` is still accepted as an *optional* argument, but now means
"measure PCC on this other WCS-bearing frame instead of the input" (e.g. to
point PCC at an earlier, less-processed frame); it's a reference override, not
a requirement. This needs **internet access** — the query goes out to the
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
