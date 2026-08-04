# Architecture

Orientation for anyone (human or a fresh Claude session) picking up this project.
It explains how the pieces fit together and where to look. For step-by-step
*usage*, see `README.md`; for the *why* behind each feature, see the design specs
under `docs/superpowers/specs/`.

## The big picture: three layers

The image pipeline is built in three layers, each thin and testable on its own.
Higher layers wrap lower ones — they never reimplement the math.

```
┌─ flow/     graph model · validation · cache · executor · CLI   (orchestration)
│            builds DAGs of stages, runs them headlessly, caches intermediates
├─ stages/   self-describing steps: typed params + named ports    (composition)
│            one thin wrapper per operation, over the numeric core
└─ astrolib.py + pcc.py + starnet.py   pure functions             (numeric core)
             FITS I/O, background fit, stretch, denoise, screen, PCC, StarNet2
```

- **Numeric core** — `scripts/astrolib.py` (load/save, `background_model`,
  `linked_stretch`, `finish`, `screen`, `masked_denoise`, `unsharp_luma`,
  `source_mask`, autostretch), `scripts/pcc.py` (Gaia DR3 photometric color
  calibration), `scripts/starnet.py` (StarNet2 CLI wrapper). Plain functions on
  numpy arrays; no notion of "a pipeline step."
- **Stages** — `scripts/stages/`. Each stage is a small class declaring a typed
  parameter schema and named input/output ports, whose `apply()` calls the core.
  A registry makes them discoverable. This is what a GUI would render as nodes.
- **Flow** — `scripts/flow/`. Connects stages into a DAG (`Graph` of `Node`/`Edge`),
  serializes to/from JSON, validates, and executes with content-addressed caching.
  `python -m flow` is the CLI; the built-in `linear`/`starless` flows reproduce
  the standard pipelines.

**The numbered scripts (`scripts/01_crop.py` … `05b_starless_finish.py`) are now
thin shims** over the stages — kept as standalone CLIs, but no longer where the
logic lives. `run_pipeline.sh` (and `make run`/`make run-starless`) is unified
over the flow executor.

## Data contracts

Everything flows as a `stages.image.Image` payload:

- `pixels` — `(H, W, 3)` float32.
- `space` — `Space.LINEAR_ADU` (~0..65535 ADU, steps crop→color) or
  `Space.NONLINEAR` ([0,1], after stretch). Stages declare the space each port
  requires; validation and runtime checks enforce it.
- `header` — the **full FITS header**, so **WCS travels in the payload**. (The old
  per-step `al.save` dropped WCS via a whitelist; `stages.io.save_fits` writes the
  whole header + a `PIPESPCE` card. Crop shifts `CRPIX`.) This is why color
  calibration no longer needs the `--original` stack — it reads WCS from its own
  input. On disk, FITS is `(3, H, W)`.

## The stages (13)

`import stages; stages.list_stages()` (or `python -m flow schema`) returns the
JSON schema of every stage — the node palette a GUI consumes.

| id | ports (in → out) | wraps |
|----|------------------|-------|
| `load` | – → image | `stages.io.load_fits` (source node) |
| `crop` | image → image | slice + `crop_header` (CRPIX shift) |
| `background_extract` | image → image (linear) | `astrolib.background_model` |
| `color_calibrate` | image (+ optional `reference`) → image | `pcc.photometric_calibration` / gentle-WB fallback |
| `stretch` | image (linear) → image (nonlinear) | `astrolib.linked_stretch` |
| `finish` | image → image | `astrolib.finish` (SCNR + denoise + saturation) |
| `saturate` | image → image | `astrolib.finish` (saturation only) |
| `masked_denoise` | image → image | `astrolib.masked_denoise` (bg-aware) |
| `unsharp_luma` | image → image | `astrolib.unsharp_luma` |
| `remove_stars` | image → `starless`, `stars` | `starnet.remove_stars` (StarNet2) |
| `screen_recombine` | `base`, `overlay` → image | `astrolib.screen` |
| `export_image` | image → (sink) | 16-bit TIFF + 8-bit PNG + preview |
| `preview_sink` | image → (sink) | `astrolib.save_preview` (diagnostic tap) |

Adding a stage: drop a `@register`ed `Stage` subclass into `scripts/stages/` — the
package auto-imports submodules, so it appears in the registry and `flow schema`
with no other wiring.

## Running it

All of these produce **versioned, never-overwritten** outputs under
`output/<name>_<label>.{tif,png}`:

- `make run FITS="data/<master>.fit" [V=label]` — standard finish (flow `linear`).
- `make run-starless FITS="data/<master>.fit" [V=label]` — starless finish (flow `starless`).
- `python -m flow run (FLOW.json | --builtin linear|starless) --input X.fit --label v1 [--no-cache]`
- `python -m flow validate (FLOW.json | --builtin …)` — structural errors (GUI would badge these).
- `python -m flow schema` — the node palette (all stages, ports, typed params) as JSON.
- Numbered shims still work standalone, e.g. `python scripts/01_crop.py in.fit out.fit`.

Two finishing tools live in `tools/` (see README for details):

- `tools/portfolio_finish.py` — the repeatable "M101 v1" portfolio recipe
  (Cosmic Clarity denoise → sky deepen/neutralize → saturation → mid-contrast;
  optional StarNet2 star reduction + crop). Run with the Cosmic Clarity venv.
- `tools/ccdenoise.sh` — wraps the native Cosmic Clarity denoise CLI for
  arbitrary paths.

## Flow format, tokens, and caching

A flow is JSON: `nodes` (`id`, `type`, `params`, optional `ui` for GUI
positions) and `edges` (`from {node,port}` → `to {node,port}`). String params may
contain run-scoped tokens the executor resolves: `{input}`, `{out}`
(= `output/<name>_<label>`), `{work}`.

The executor validates → topologically sorts → runs each node, **caching each
producing node's output** in `work/cache/<hash>__<port>.fits`. The hash folds the
stage type, its params, and its upstream nodes' hashes (and the source file's
signature) — so re-running after changing one downstream param reuses everything
upstream, including the ~25 s Gaia PCC query and StarNet2. `--no-cache` forces a
clean run; `make clean` clears the cache. Measured: a warm re-run dropped from
~25 s to ~3 s, and the starless flow reuses the linear flow's cached head.

## External tools

| tool | where | used by |
|------|-------|---------|
| StarNet2 CLI | `~/StarNet2/starnet2` (`$STARNET2_CLI` overrides) | `starnet.py` / `remove_stars` |
| Cosmic Clarity | `~/CosmicClarity` (`$COSMIC_CLARITY_DIR` overrides) | `tools/ccdenoise.sh`, `portfolio_finish.py` |
| Siril | `/Applications/Siril.app/.../siril-cli` | re-stacking (`siril/*.ssf`, `siril/restack.sh`) |
| SetiAstroSuitePro | `/Applications/SetiAstroSuitePro.app` | GUI star removal / gradient / stretch |

## Repository map

```
scripts/
  astrolib.py            numeric core (pure functions)
  pcc.py                 Gaia DR3 photometric color calibration
  starnet.py             StarNet2 CLI wrapper
  compare.py             before/after comparison tool
  inspect_fits.py        `make inspect` — header + stats
  01_crop … 05b_*.py     thin shims over the stages
  run_pipeline.sh        shim over `python -m flow run --builtin`
  stages/                image.py, base.py (Param/Port/Stage), registry.py,
                         io.py, source.py, geometry.py, background.py, color.py,
                         stretch.py, finish.py, denoise.py, stars.py, export.py
  flow/                  graph.py, validate.py, cache.py, executor.py,
                         builtins.py, __main__.py (CLI)
tools/                   portfolio_finish.py, ccdenoise.sh
siril/                   .ssf re-stack scripts + restack.sh wrapper
tests/                   pytest suite (offline; StarNet2/Gaia mocked)
docs/                    this file; superpowers/specs (designs) + plans
data/ output/ work/      git-ignored (captures, results, intermediates + cache)
```

## Testing

`make test` runs pytest, offline and dependency-free: `pcc`/`starnet` are mocked,
Gaia is never queried, and the one real-StarNet2 test is `skipif`-guarded on the
binary. Stage tests assert parity against the core functions they wrap; flow tests
cover JSON round-trip, every validation rule, cache reuse/invalidation, and the
DAG branch/merge.

## Status & roadmap (as of 2026-08-04)

The architecture was built in three merged/pending increments (each with a design
spec + plan under `docs/superpowers/` and its own PR):

1. **Stages** — self-describing steps + registry. *(PR #2, merged to `main`.)*
2. **Flow** — graph/JSON/validate/cache/executor + unified runner. *(PR #3, open.)*
3. **React Flow GUI** — *not built.* The natural next layer, and now genuinely just
   a front-end: `flow schema` is the node palette, `flow validate` badges the
   canvas, the JSON format is save/load, and `flow run` executes. All logic already
   lives behind that JSON contract.

Separately, `tools/portfolio_finish.py` gained a `--protect-nebula` mode and a
FITS-input fix on in-progress branches (`feature/portfolio-protect-nebula`,
`fix/portfolio-finish-fits-input`) not yet reflected above.

## Leveraging this in a new session

- Read this file + `README.md` for the mental model and usage.
- To process an image: `make run` / `make run-starless`, or compose a flow JSON
  and `python -m flow run`.
- To extend the pipeline: add a `Stage` (see `scripts/stages/`), and it's
  immediately usable in flows and visible to `flow schema`.
- Design rationale for any feature is in `docs/superpowers/specs/`.
- Env note: the environment lives OUTSIDE Dropbox at `~/.venvs/astrophotography`
  (managed by uv), so nothing venv-related syncs between macOS and Linux; run
  `make setup` (`uv sync`) per machine.
