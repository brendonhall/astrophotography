# Architectural style: chainable modules + JSON flow graph + GUI-ready

**What this is.** A self-contained description of the module/flow architecture used in
this astrophotography pipeline, written so another project can adopt the same style.
The goal shape: **small self-describing modules ("stages"), each declaring its
parameters as typed JSON, chained into a graph that a GUI can render and edit, executed
headlessly from a JSON flow definition.**

If you have this repo checked out, the canonical source files are listed at the bottom;
this doc reproduces the load-bearing interfaces so you can replicate the pattern without
reading everything.

---

## 1. The core idea: three layers

Keep three layers strictly separated. Each depends only on the one below it.

1. **Numeric/logic core** — pure functions. No knowledge of parameters-as-config,
   graphs, or I/O. In this repo that's `astrolib` (numpy math). *Your domain's actual
   work lives here.*
2. **Modules ("stages")** — one small class per operation. Each wraps a core function
   and declares: a stable `id`, a typed **parameter schema**, and named **input/output
   ports**. A registry makes them discoverable. **This is the layer a GUI renders as
   nodes.**
3. **Flow** — a graph (DAG) of module instances connected port-to-port, (de)serialized
   to **JSON**, validated, and executed with caching.

Why this holds up: a module can be understood, tested, and swapped in isolation; the
flow layer never needs to know what any module does internally; and because the whole
thing is JSON in/out, a GUI is "just" a front-end over contracts the headless code
already enforces.

---

## 2. The module contract

Every module is a subclass declaring class-level metadata + a `PARAMS`/`INPUTS`/
`OUTPUTS` spec, and implementing one method: `apply()`. The base class handles
parameter coercion, input validation, and the JSON `schema()` a GUI consumes.

```python
# base.py — the load-bearing interface (reproduce this shape verbatim)
from dataclasses import dataclass, asdict
from typing import Any

class StageError(Exception): ...

@dataclass(frozen=True)
class Param:
    name: str
    type: str                       # "float"|"int"|"bool"|"enum"|"str"
    default: Any
    label: str = ""
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple | None = None
    unit: str | None = None

    def coerce(self, value):
        # None -> default; cast to type; enforce min/max/choices. Raises ValueError.
        ...

@dataclass(frozen=True)
class Port:
    name: str
    space: "Space | None" = None    # optional typed constraint on what flows through
    required: bool = True
    help: str = ""

class Stage:
    id: str = ""                    # stable, unique — the "type" a flow node references
    label: str = ""                 # human name (GUI)
    description: str = ""
    INPUTS: list = []               # list[Port]
    OUTPUTS: list = []              # list[Port]
    PARAMS: list = []               # list[Param]

    @classmethod
    def schema(cls) -> dict:
        # -> {"id","label","description","inputs","outputs","params"} — pure JSON.
        # This is the GUI palette entry for this module.
        ...

    @classmethod
    def coerce_params(cls, params) -> dict:
        return {p.name: p.coerce((params or {}).get(p.name)) for p in cls.PARAMS}

    def check(self, inputs, params) -> list:
        # returns list[str] errors: missing required inputs, port-space mismatches.
        ...

    def run(self, inputs, params=None) -> dict:
        p = self.coerce_params(params)
        errs = self.check(inputs, p)
        if errs:
            raise StageError(f"{self.id}: " + "; ".join(errs))
        return self.apply(inputs, p)     # the only thing subclasses override

    def apply(self, inputs, params) -> dict:
        raise NotImplementedError
```

**The contract in one sentence:** `run()` is `coerce params → validate inputs →
apply()`; subclasses only write `apply(inputs, params) -> {port_name: payload}`.

### A concrete module (copy this as your template)

```python
from .base import Stage, Param, Port
from .image import Space
from .registry import register

@register
class BackgroundExtractStage(Stage):
    id = "background_extract"
    label = "Background extraction"
    description = "Per-channel low-order polynomial gradient removal; re-adds a pedestal."
    INPUTS  = [Port("image", space=Space.LINEAR_ADU)]
    OUTPUTS = [Port("image", space=Space.LINEAR_ADU)]
    PARAMS  = [
        Param("degree",   "int",   3,    "Polynomial degree",  min=1, max=6,  step=1),
        Param("sample",   "int",   12,   "Pixel subsampling",  min=1, max=64, step=1),
        Param("pedestal", "float", 0.10, "Pedestal", "Re-added level (fraction of 65535)",
              min=0, max=1, step=0.01),
    ]

    def apply(self, inputs, params):
        img = inputs["image"]
        # ... call the numeric core with params["degree"] etc. ...
        return {"image": img.replace(pixels=out)}   # keyed by OUTPUT port name
```

Notes that make the pattern work:
- **Params are data, not code.** Every knob is a `Param` with type + bounds + help.
  That single list drives coercion, validation, *and* the GUI widget. Never read a
  loose `params["x"]` that isn't declared in `PARAMS`.
- **Ports are named and (optionally) typed.** `apply()` returns a dict keyed by output
  port name; it reads inputs by input port name. The optional `space` on a port lets
  the validator reject nonsensical connections before anything runs.
- **`apply()` is pure-ish:** inputs → outputs, no global state, no I/O side effects
  except for explicit sink modules (see §5).

---

## 3. The registry + auto-discovery

Modules self-register via a decorator; the package auto-imports its submodules so
dropping a file in is the *entire* wiring step.

```python
# registry.py
_REGISTRY: dict = {}

def register(cls):
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate stage id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls

def get(stage_id):      return _REGISTRY[stage_id]
def list_stages():      return [c.schema() for c in _REGISTRY.values()]   # GUI palette
```

```python
# __init__.py — import every submodule so @register fires on package import
def _autoload():
    import importlib, pkgutil
    skip = {"base", "image", "io", "registry"}      # infra, not modules
    for m in pkgutil.iter_modules(__path__):
        if m.name not in skip:
            importlib.import_module(f"{__name__}.{m.name}")
_autoload()
```

`list_stages()` is the JSON node palette — the full set of modules and their params,
ready for a GUI or a CLI `schema` command to emit.

---

## 4. The data payload contract

Modules pass a single well-defined **payload** type between ports. Here it's an
immutable image; in your project it's whatever your modules transform. Keep it:

- **Immutable** with a `.replace(**changes)` helper (frozen dataclass), so `apply()`
  never mutates its inputs and caching stays correct.
- **Self-describing** enough to validate connections. Here each payload carries a
  `space` (an `Enum` like `LINEAR_ADU="linear-adu"`, `NONLINEAR="nonlinear"`), and ports
  can require a specific space. Your equivalent might be units, dtype, or schema
  version — the point is that ports can *type-check the wire*.

```python
class Space(str, Enum):
    LINEAR_ADU = "linear-adu"
    NONLINEAR  = "nonlinear"

@dataclass(frozen=True)
class Image:
    pixels: "np.ndarray"     # the actual data
    space: Space
    header: dict
    def replace(self, **kw): ...   # returns a new Image
```

---

## 5. The flow: a JSON graph of module instances

A flow is a DAG. Nodes are module instances (`type` = a registered module `id` + a
`params` dict); edges connect an output port to an input port. It round-trips to JSON.

```python
# graph.py
@dataclass(frozen=True)
class Endpoint: node: str; port: str
@dataclass(frozen=True)
class Edge:     id: str; src: Endpoint; dst: Endpoint   # src=OUTPUT, dst=INPUT
@dataclass(frozen=True)
class Node:     id: str; type: str; params: dict = {}; ui: dict = {}  # ui = GUI x/y only
@dataclass(frozen=True)
class Graph:    nodes: tuple = (); edges: tuple = (); name: str = ""; version: int = 1
```

### The JSON flow format (this is the save/load format a GUI reads and writes)

```json
{
  "version": 1,
  "name": "linear",
  "nodes": [
    {"id": "src", "type": "load",
     "params": {"path": "{input}", "space": "linear-adu"}},
    {"id": "bg",  "type": "background_extract",
     "params": {"degree": 3, "sample": 12, "pedestal": 0.10},
     "ui": {"x": 220, "y": 80}}
  ],
  "edges": [
    {"id": "e2", "from": {"node": "crop", "port": "image"},
                 "to":   {"node": "bg",   "port": "image"}}
  ]
}
```

- `node.type` must be a registered module `id`.
- `node.params` supplies values for that module's declared `Param`s (anything omitted
  falls back to the `Param.default`).
- `node.ui` is **GUI-only** (node position); the executor round-trips but ignores it.
- **String params support run-scoped tokens** the executor substitutes at run time:
  `{input}` (source file), `{out}` (output basename), `{work}` (scratch dir). This is
  how a static flow stays reusable across different inputs.

### Flows can also be built in code (worked example of wiring)

```python
def linear_flow() -> Graph:
    nodes = (
        Node("src", "load",  {"path": "{input}", "space": "linear-adu"}),
        Node("crop","crop",  {"margin": 40}),
        Node("bg",  "background_extract", {"degree": 3, "sample": 12, "pedestal": 0.10}),
        Node("fin", "finish",{"saturation": 1.20, "luma_denoise": 0.012}),
        Node("exp", "export_image", {"out_base": "{out}"}),
    )
    edges = (
        Edge("e1", Endpoint("src","image"), Endpoint("crop","image")),
        Edge("e2", Endpoint("crop","image"), Endpoint("bg","image")),
        Edge("e3", Endpoint("bg","image"),  Endpoint("fin","image")),
        Edge("e4", Endpoint("fin","image"), Endpoint("exp","image")),
    )
    return Graph(nodes, edges, name="linear")
```

Branch/merge is just multiple edges: a module can emit several output ports (e.g. a
star-removal module producing both `starless` and `stars`) and downstream nodes consume
whichever port they need.

---

## 6. Execution model

The executor is headless and deterministic:

1. **Validate** the graph; abort on any `error`-level issue (collect `warning`s).
2. **Topological sort** (Kahn) to get run order.
3. For each node: gather upstream payloads by port, **resolve `{...}` tokens** in string
   params, compute a **recipe hash** (module type + params + upstream hashes + source
   file signature), reuse the cached output if that hash is on disk, else `run()` the
   module and cache each output port.
4. **Sink modules** (no `OUTPUTS`) just run for their side effect (writing a file) and
   are not cached.

```python
def run(graph, input_path, label, work_dir="work", out_dir="output", cache=True):
    # validate -> abort on errors
    tokens = {"{input}": input_path, "{out}": out_base, "{work}": work_dir}
    for nid in topo(graph):
        node   = graph.node(nid)
        cls    = registry.get(node.type)
        params = resolve(node.params, tokens)
        inputs = {e.dst.port: payloads[(e.src.node, e.src.port)]
                  for e in graph.in_edges(nid)}
        h = recipe_hash(node, upstream_hashes)         # content-addressed cache key
        if cache and cached_on_disk(h): reuse
        else: payloads.update(cls().run(inputs, params))   # coerce+check+apply
```

**Validation is separate and explicit** (`validate(graph) -> list[Issue]`, each
`Issue(level, where, message)`). It checks: duplicate node ids, unknown module type,
param coercion errors, dangling edges, missing/duplicate input connections, **port-type
(space) mismatches**, unconsumed outputs (warning), and cycles. Keep this as a
standalone function: it's the headless gate *and* the GUI's on-canvas error source —
same rules, two front-ends.

---

## 7. Why this is GUI-ready (and what the GUI plugs into)

There is **no GUI in this repo yet** — but the architecture was built so a React-Flow-
style GUI is a thin front-end over contracts the headless code already owns:

| GUI need                         | Already provided by                                  |
|----------------------------------|------------------------------------------------------|
| Node palette (draggable modules) | `list_stages()` → each module's `schema()` JSON       |
| Parameter widgets per node       | `Param` fields: `type`, `min`/`max`/`step`, `choices`, `label`, `help`, `unit` |
| Connect nodes / legal wiring     | named `Port`s + optional `space` typing               |
| Live error badges on canvas      | `validate(graph)` → `Issue(level, where, message)`    |
| Save / load a pipeline           | `Graph.to_json()` / `from_json()` (incl. `ui` x/y)    |
| Run button                       | the headless `run(graph, ...)` executor               |

**Design rule for the GUI:** it should produce and consume *only* the JSON graph and the
schema. It must not contain any pipeline logic — if the GUI needs to know something to
draw or validate a node, expose that as data in `schema()`/`validate()`, don't special-
case it in the front-end.

---

## 8. Recipes

**Add a new module:** create one file in the modules package with a `@register`ed
`Stage` subclass — set `id`, `label`, `description`, `INPUTS`, `OUTPUTS`, `PARAMS`, and
implement `apply()`. Auto-discovery picks it up; it appears in the palette and is usable
in flows with zero other wiring. Write a unit test that calls `run()` with sample inputs.

**Define a new flow:** either author the JSON (`nodes` + `edges`) directly, or build a
`Graph` in code like §5. Reference modules by their `id`, wire output ports to input
ports, use `{input}`/`{out}`/`{work}` tokens for run-scoped paths. Run
`validate(graph)` before executing.

**Emit the palette / a schema for tooling:** `list_stages()` returns the full JSON
palette (every module + params). Wire it behind a CLI subcommand (`... schema`) so the
GUI or other sessions can pull the contract.

---

## 9. Principles to carry over (the short list)

- **One module = one operation = one small class.** If a module is hard to describe in a
  one-line `description`, it's doing too much.
- **Every parameter is a declared, typed `Param`** with bounds and help — never an
  undocumented dict key. The param list is the single source for validation and UI.
- **Modules communicate only through named ports** carrying an immutable, self-
  describing payload. No shared mutable state.
- **The graph is JSON; the JSON is the contract.** Everything (CLI, cache, GUI) is a
  front-end over `Graph.to_json()` + `schema()` + `validate()`.
- **Validation is a pure function over the graph**, reused by headless runs and the GUI.
- **Registration is automatic**; adding capability never means editing a central list.
- **Determinism + content-addressed caching:** identical (module, params, upstream)
  ⇒ identical output, so re-runs are cheap and reproducible.

---

## Canonical source files in this repo

Mental model & contracts:
- `docs/ARCHITECTURE.md` — full narrative (has a "Leveraging this in a new session" section)
- `CLAUDE.md` → "Processing pipeline" — terse restatement

Module layer:
- `scripts/stages/base.py` — `Param` / `Port` / `Stage` (the interface)
- `scripts/stages/registry.py` + `scripts/stages/__init__.py` — registry + auto-discovery
- `scripts/stages/image.py` — the payload contract (`Image`, `Space`)
- `scripts/stages/background.py`, `scripts/stages/source.py` — representative modules

Flow layer:
- `scripts/flow/graph.py` — graph model + JSON (de)serialization
- `scripts/flow/executor.py` — topo-sort, token resolution, caching, run loop
- `scripts/flow/validate.py` — the shared validator (headless gate + GUI errors)
- `scripts/flow/builtins.py` — `linear` / `starless` flows expressed as graphs
- `scripts/flow/__main__.py` — CLI: `python -m flow {run|validate|schema}`

Design rationale:
- `docs/superpowers/plans/2026-07-26-stage-architecture.md`
- `docs/superpowers/specs/2026-07-27-flow-graph-executor-design.md`

**Fastest way to see the live JSON param contract for every module:** run
`python -m flow schema` in this repo — it prints the whole self-describing palette.
