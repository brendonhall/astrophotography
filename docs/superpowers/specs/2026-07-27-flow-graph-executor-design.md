# Flow Graph + JSON Executor — Design

**Date:** 2026-07-27
**Status:** Approved (design shape); pending spec review
**Depends on:** the `scripts/stages/` package (PR #2, `feature/stage-architecture`).

## Context

The pipeline's steps are now self-describing **stages** (`scripts/stages/`): each has a typed parameter schema and named input/output ports, over the existing `astrolib`/`pcc`/`starnet` numeric core. What's missing is a way to *connect* stages into a flow, serialize that flow, validate it, and execute it headlessly. That layer is what a future React Flow node-editor GUI drives — the GUI is a thin front-end over a JSON graph the executor already understands.

This spec designs that graph + execution layer. The **GUI is out of scope**; the deliverable is the Python graph model, a JSON flow format, a validating headless executor with intermediate caching, and the unification of the existing pipeline runner on top of it.

**Scope decisions (confirmed with the user):**
- **Content-addressed caching** of intermediates, so re-running skips unchanged upstream stages (the GUI's core iterate-on-one-node use case).
- **Unify the runner**: `run_pipeline.sh` / `make run` / `make run-starless` become thin shims over the executor, with the current linear and starless pipelines expressed as built-in flow graphs.
- **Token substitution** (`{input}`/`{out}`) binds run-scoped data in/out, keeping graphs portable.
- **Keep the numbered shims** (01–05b) as standalone CLIs; they're just no longer the pipeline's execution path.

## Package layout

New `scripts/flow/` package (importable as `flow`; `tests/conftest.py` already puts `scripts/` on `sys.path`):

```
scripts/flow/
  __init__.py     # re-exports Graph, validate, run, builtins
  graph.py        # Node, Edge, Graph + to_json/from_json
  validate.py     # Issue, validate(graph) -> list[Issue]
  cache.py        # recipe_hash(), cache load/store via stages.io
  executor.py     # run(graph, input, label, cache=True) -> RunReport
  builtins.py     # linear_flow(), starless_flow() -> Graph
  __main__.py     # CLI: run | validate | schema
```

Plus one addition to the **stages** package: `scripts/stages/source.py` with `LoadStage` (id `load`, 0 inputs → `image`, params `path`(str)/`space`(enum linear-adu|nonlinear)) so a graph's data source is a normal node.

## Data model (`graph.py`)

```python
@dataclass(frozen=True)
class Endpoint:      # one side of an edge
    node: str
    port: str

@dataclass(frozen=True)
class Edge:
    id: str
    src: Endpoint    # an OUTPUT port
    dst: Endpoint    # an INPUT port

@dataclass(frozen=True)
class Node:
    id: str
    type: str                 # a registered stage id
    params: dict              # may contain {input}/{out}/... string tokens
    ui: dict = field(default_factory=dict)   # GUI-only (x/y); runner ignores, round-trips

@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    name: str = ""
    version: int = 1

    def to_json(self) -> dict         # -> the JSON flow format below
    @classmethod
    def from_json(cls, data) -> "Graph"
    def node(self, node_id) -> Node
    def in_edges(self, node_id) -> list[Edge]
    def out_edges(self, node_id) -> list[Edge]
```

Unknown keys in the JSON are tolerated and preserved on round-trip (forward-compat with GUI additions).

## JSON flow format (GUI save/load contract)

```json
{
  "version": 1,
  "name": "linear",
  "nodes": [
    {"id": "src",  "type": "load",           "params": {"path": "{input}", "space": "linear-adu"}, "ui": {"x": 0, "y": 0}},
    {"id": "crop", "type": "crop",            "params": {"margin": 40}},
    {"id": "bg",   "type": "background_extract","params": {"degree": 3, "sample": 12, "pedestal": 0.10}},
    {"id": "col",  "type": "color_calibrate", "params": {"ref_bp_rp": 0.82, "min_stars": 30, "diagnostic_path": "{out}_pcc_diagnostic.png"}},
    {"id": "str",  "type": "stretch",         "params": {"target_bg": 0.18, "shadows_clip": -1.8}},
    {"id": "fin",  "type": "finish",          "params": {"saturation": 1.20, "luma_denoise": 0.012, "chroma_denoise": 4.0, "scnr": true}},
    {"id": "exp",  "type": "export_image",    "params": {"out_base": "{out}"}}
  ],
  "edges": [
    {"id": "e1", "from": {"node": "src",  "port": "image"}, "to": {"node": "crop", "port": "image"}},
    {"id": "e2", "from": {"node": "crop", "port": "image"}, "to": {"node": "bg",   "port": "image"}},
    {"id": "e3", "from": {"node": "bg",   "port": "image"}, "to": {"node": "col",  "port": "image"}},
    {"id": "e4", "from": {"node": "col",  "port": "image"}, "to": {"node": "str",  "port": "image"}},
    {"id": "e5", "from": {"node": "str",  "port": "image"}, "to": {"node": "fin",  "port": "image"}},
    {"id": "e6", "from": {"node": "fin",  "port": "image"}, "to": {"node": "exp",  "port": "image"}}
  ]
}
```

The **starless** flow is the same head (`src→crop→bg→col→str`) then a DAG tail:
`str→remove_stars`; `remove_stars.starless→masked_denoise→saturate→screen_recombine.base`; `remove_stars.stars→screen_recombine.overlay`; `screen_recombine→export_image`; plus `preview_sink` taps on the processed-starless and star layers (`out_path` = `{out}_starless.png` / `{out}_starlayer.png`).

## Token substitution

The executor resolves run-scoped tokens in **string** param values before running a node:
- `{input}` → the `--input` FITS path
- `{out}` → `output/<name>_<label>` (name = input basename sans ext, spaces→`_`; label = `--label` or a timestamp passed in — never `Date.now()` inside library code)
- `{work}` → the `work/` dir

Substitution is a plain `str.replace` over each string param; non-string params pass through. Tokens the GUI doesn't fill stay literal in the JSON (portable).

## Executor (`executor.py`)

```python
@dataclass
class RunReport:
    outputs: dict            # node_id -> {port: cache_path or None}
    warnings: list           # e.g. "col: PCC fell back to gentle WB"
    cached: list             # node ids served from cache
    ran: list                # node ids executed

def run(graph, input_path, label, work_dir="work", cache=True) -> RunReport
```

Algorithm:
1. `errs = validate(graph)`; if any error-level issues, raise `FlowError` with them (never partially execute).
2. Resolve tokens using `{input, out, work}`.
3. Kahn topological sort over edges (acyclicity already validated).
4. For each node in order: gather inputs from upstream edges (`{port: Image}`); compute `recipe_hash`; if `cache` and all output cache files exist, **load** them (served-from-cache) else call `stages.get(node.type)().run(inputs, params)` and **store** each output payload to cache. Sinks (no outputs) always execute (they write files, not payloads) — they are cheap and their effect is the output, so they are not cached.
5. Collect PCC-style fallbacks into `warnings` (the color stage already handles fallback internally; the executor surfaces it if the stage reports one — v1 may leave warnings empty and add later).

Errors: a missing StarNet2 binary raises `starnet.StarNetError` out of `remove_stars`, which propagates as a `FlowError(node="…")` **before** any sink runs (sinks are downstream), so no partial output is written.

## Cache (`cache.py`)

```python
def recipe_hash(node, input_hashes: dict[port, str]) -> str
    # sha1 of json(type, sorted params) + sorted upstream input hashes -> node identity
def cache_path(work_dir, node_hash, port) -> str        # work/cache/<hash>__<port>.fits
def load_cached(path) -> Image                           # via stages.io.load_fits
def store_cached(path, img) -> None                      # via stages.io.save_fits
```

- A node's hash folds in its upstream nodes' hashes, so any change anywhere upstream (params or structure) invalidates it and everything downstream. Same graph + same params + same input → 100% hit.
- Cache keyed on the **recipe**, not pixels, so it's cheap to compute without running anything.
- Non-determinism (Gaia catalog drift, StarNet2) is acceptable for reuse: identical recipe reuses the prior output, which is exactly the iterate-fast intent; `--no-cache` forces recompute, `make clean` clears `work/` (cache included).
- The source `load` node's hash includes the input file's path + mtime + size (so a changed input invalidates the chain).

## Validation (`validate.py`)

```python
@dataclass(frozen=True)
class Issue:
    level: str        # "error" | "warning"
    where: str        # node id or edge id
    message: str

def validate(graph) -> list[Issue]
```

Error-level rules: unknown stage `type`; param fails the stage's `Param.coerce` (type/range/choice); edge endpoint node missing; edge port not on the stage (wrong direction = output port used as input or vice-versa); **space mismatch** across an edge (source output space vs. target input space, when both declared); a required input port with no inbound edge; two edges into one input port; a cycle. Warning-level: an output port with no consumer (dead branch). The GUI calls `validate` on every edit to badge the canvas.

## Runner unification

- `flow.builtins.linear_flow()` and `starless_flow()` return `Graph` objects encoding the pipelines above.
- `scripts/flow/__main__.py`:
  - `python -m flow run (FLOW.json | --builtin linear|starless) --input X.fit --label v1 [--no-cache]`
  - `python -m flow validate (FLOW.json | --builtin …)` → prints issues, exit non-zero on errors
  - `python -m flow schema` → `json.dumps(stages.list_stages())` (the GUI node palette)
- `scripts/run_pipeline.sh` becomes a thin shim: parse `--starless`/label as today, then `exec python -m flow run --builtin <linear|starless> --input "$IN" --label "$LABEL"`. `make run` / `make run-starless` unchanged.
- The numbered shims (01–05b) remain as standalone CLIs; the pipeline just no longer calls them.

## Testing (TDD, offline)

- **graph.py:** `to_json`/`from_json` round-trip preserves nodes/edges/params and the `ui` block and unknown keys.
- **validate.py:** one test per rule (unknown type, bad param, dangling edge endpoint, wrong-direction port, space mismatch, missing required input, double-edge-into-input, cycle, dead-branch warning).
- **cache.py:** `recipe_hash` stable for identical recipe, changes when a param or an upstream hash changes; `store_cached`/`load_cached` round-trip an Image (space + WCS) via `stages.io`.
- **executor.py:** build a small graph in memory (`load→crop→…→export`) over a synthetic FITS; assert the export files appear and topo order is respected. A DAG test with `starnet.remove_stars` monkeypatched exercises the 1→2 / 2→1 branch/merge. A cache test wraps a stage with an invocation counter (monkeypatch) and asserts the second run serves it from cache (counter unchanged) while a param change forces recompute. A missing-required-input graph raises `FlowError` before any sink runs.
- **builtins.py:** `validate(linear_flow())` and `validate(starless_flow())` return no errors; a run of `linear_flow()` on a synthetic FITS (with `pcc`/`starnet` mocked) produces the same finished array as calling the `05_finish` shim pipeline (parity).
- **CLI:** `run --builtin` on a tmp FITS writes outputs; `validate` exits non-zero on a broken graph; `schema` emits valid JSON with all stage ids.
- The existing real-data end-to-end runs (`make run` / `make run-starless`) remain the whole-system parity proof once run_pipeline is unified.

## Out of scope (deferred / future)

- The React Flow GUI and any web/API layer (this JSON contract is the boundary).
- Parallel node execution (topo order is sequential; the cap on wall-clock is the slow stages, which cache handles).
- A cache-eviction policy (rely on `make clean`).
- Sub-graphs / reusable node groups; live progress streaming to a GUI (add when the GUI exists).

## Success criteria

1. `python -m flow run --builtin linear --input <master> --label v1` reproduces the standard finish; `--builtin starless` reproduces the starless finish (same outputs as the shim pipeline).
2. Re-running after changing only a downstream param reuses the cached upstream (crop/background/color/stretch/remove_stars not recomputed); `--no-cache` recomputes all.
3. `python -m flow schema` + a saved/loaded flow JSON round-trip give a GUI everything it needs (node palette, ports, params, positions).
4. `run_pipeline.sh` / `make run` / `make run-starless` behave exactly as before (verified by the existing real-data runs).
5. `make test` stays green and offline.
