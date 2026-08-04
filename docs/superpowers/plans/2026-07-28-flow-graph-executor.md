# Flow Graph + JSON Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `scripts/flow/` package that connects stages into a DAG, serializes it to JSON, validates it, and executes it headlessly with intermediate caching — unifying the pipeline runner on top of it.

**Architecture:** A `Graph` (Node/Edge) model with JSON round-trip, a `validate()` returning `Issue`s (the GUI's on-canvas errors), a content-addressed `cache`, and an `executor` that resolves `{input}`/`{out}` tokens, topologically sorts, runs each node's `Stage.run(inputs, params)`, and caches producing-node outputs. Built-in `linear_flow()`/`starless_flow()` graphs reproduce today's pipelines; `run_pipeline.sh` becomes a shim over `python -m flow run`.

**Tech Stack:** Python 3.9 (`.venv`), stdlib (dataclasses/json/hashlib/argparse); the existing `scripts/stages/` package + `astrolib`/`pcc`/`starnet`. pytest.

## Global Constraints

- **Python 3.9 compatible:** every new module starts with `from __future__ import annotations`.
- **No new dependencies** (stdlib only; the flow layer reuses `stages`).
- **Reuse, don't reimplement:** nodes execute via `stages.get(type)().run(inputs, params)`; IO via `stages.io.load_fits`/`save_fits`; the numeric core is untouched.
- **Data conventions:** payloads are `stages.image.Image` (pixels (H,W,3) float32; `Space.LINEAR_ADU` ~0..65535, `Space.NONLINEAR` [0,1]); FITS on disk (3,H,W).
- **Token substitution:** the executor replaces `{input}`, `{out}`, `{work}` in string params before running a node. `{out}` = `output/<name>_<label>` (name = input basename sans ext, spaces→`_`).
- **Content-addressed cache** in `work/cache/<hash>__<port>.fits`; a node's hash folds in its upstream nodes' hashes. `--no-cache` forces recompute; `make clean` clears it. Sinks (no outputs) are not cached.
- **Never overwrite outputs:** versioned via `<name>_<label>` (unchanged from today).
- **Tests offline:** mock `pcc`/`starnet`; no Gaia/StarNet2 network/binary in unit tests.
- Run with `.venv/bin/python`; `tests/conftest.py` puts `scripts/` on `sys.path`, so `import flow`, `import stages`, `import astrolib` resolve.
- Registered stage ids after this plan: the existing 12 + `load` (13 total). The existing `tests/test_stages_registry.py` asserts a subset, so adding `load` keeps it green.

---

### Task 1: LoadStage (graph source node)

**Files:**
- Create: `scripts/stages/source.py`
- Test: `tests/test_stage_source.py`

**Interfaces:**
- Produces: `stages.source.LoadStage` (id `load`, 0 inputs → `image`; params `path`(str), `space`(enum "linear-adu"|"nonlinear")). Reads a FITS via `stages.io.load_fits(path, Space(space))`.

- [ ] **Step 1: Write the failing test** — `tests/test_stage_source.py`:
```python
import numpy as np
from astropy.io import fits
import stages
from stages.image import Image, Space
from stages.io import save_fits
from stages.source import LoadStage


def test_load_stage_reads_fits(tmp_path):
    p = str(tmp_path / "in.fit")
    save_fits(p, Image(np.full((8, 8, 3), 1234.0, np.float32), Space.LINEAR_ADU, fits.Header()))
    out = LoadStage().run({}, {"path": p, "space": "linear-adu"})["image"]
    assert out.space is Space.LINEAR_ADU
    assert out.pixels.shape == (8, 8, 3) and np.allclose(out.pixels, 1234.0)


def test_load_stage_registered():
    assert stages.get("load") is LoadStage
    assert any(s["id"] == "load" for s in stages.list_stages())
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_stage_source.py -v` → FAIL (`No module named 'stages.source'`).

- [ ] **Step 3: Implement** — `scripts/stages/source.py`:
```python
"""Source stage: load a FITS file into the pipeline."""
from __future__ import annotations
from .base import Stage, Param, Port
from .image import Space
from .io import load_fits
from .registry import register


@register
class LoadStage(Stage):
    id = "load"
    label = "Load FITS"
    description = "Read a FITS file into the pipeline as the source image."
    INPUTS = []
    OUTPUTS = [Port("image")]
    PARAMS = [
        Param("path", "str", "", "FITS path"),
        Param("space", "enum", "linear-adu", "Space",
              choices=("linear-adu", "nonlinear")),
    ]

    def apply(self, inputs, params):
        return {"image": load_fits(params["path"], Space(params["space"]))}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_stage_source.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/stages/source.py tests/test_stage_source.py
git commit -m "Add LoadStage source node to stages"
```

---

### Task 2: Graph data model + JSON round-trip

**Files:**
- Create: `scripts/flow/__init__.py`, `scripts/flow/graph.py`
- Test: `tests/test_flow_graph.py`

**Interfaces:**
- Produces: `flow.graph.Endpoint(node, port)`, `flow.graph.Edge(id, src, dst)`, `flow.graph.Node(id, type, params={}, ui={})`, `flow.graph.Graph(nodes=(), edges=(), name="", version=1)` with `to_json()`, `from_json(data)`, `node(id)`, `in_edges(id)`, `out_edges(id)`. `flow` package re-exports `Graph`, `Node`, `Edge`, `Endpoint`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_graph.py`:
```python
from flow.graph import Graph, Node, Edge, Endpoint


def _g():
    return Graph(
        nodes=(Node("a", "load", {"path": "{input}"}, {"x": 1, "y": 2}),
               Node("b", "crop", {"margin": 40})),
        edges=(Edge("e1", Endpoint("a", "image"), Endpoint("b", "image")),),
        name="demo", version=1)


def test_roundtrip_preserves_nodes_edges_ui():
    d = _g().to_json()
    g2 = Graph.from_json(d)
    assert g2.name == "demo" and g2.version == 1
    assert g2.node("a").params["path"] == "{input}"
    assert g2.node("a").ui == {"x": 1, "y": 2}      # GUI positions survive
    assert g2.edges[0].src == Endpoint("a", "image")
    assert g2.edges[0].dst == Endpoint("b", "image")


def test_from_json_tolerates_unknown_keys():
    d = _g().to_json()
    d["metadata"] = {"author": "x"}          # unknown top-level key
    d["nodes"][0]["color"] = "red"           # unknown node key
    g2 = Graph.from_json(d)                   # must not raise
    assert len(g2.nodes) == 2


def test_in_out_edges():
    g = _g()
    assert [e.id for e in g.in_edges("b")] == ["e1"]
    assert [e.id for e in g.out_edges("a")] == ["e1"]
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_graph.py -v` → FAIL (`No module named 'flow'`).

- [ ] **Step 3: Implement** — `scripts/flow/graph.py`:
```python
"""Flow graph data model + JSON (de)serialization."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    node: str
    port: str


@dataclass(frozen=True)
class Edge:
    id: str
    src: Endpoint       # an OUTPUT port
    dst: Endpoint       # an INPUT port


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    ui: dict = field(default_factory=dict)      # GUI-only (x/y); runner ignores


@dataclass(frozen=True)
class Graph:
    nodes: tuple = ()
    edges: tuple = ()
    name: str = ""
    version: int = 1

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "nodes": [
                {"id": n.id, "type": n.type, "params": dict(n.params),
                 **({"ui": dict(n.ui)} if n.ui else {})}
                for n in self.nodes],
            "edges": [
                {"id": e.id,
                 "from": {"node": e.src.node, "port": e.src.port},
                 "to": {"node": e.dst.node, "port": e.dst.port}}
                for e in self.edges],
        }

    @classmethod
    def from_json(cls, data) -> "Graph":
        nodes = tuple(
            Node(n["id"], n["type"], dict(n.get("params", {})), dict(n.get("ui", {})))
            for n in data.get("nodes", []))
        edges = tuple(
            Edge(e["id"], Endpoint(e["from"]["node"], e["from"]["port"]),
                 Endpoint(e["to"]["node"], e["to"]["port"]))
            for e in data.get("edges", []))
        return cls(nodes, edges, data.get("name", ""), data.get("version", 1))

    def node(self, node_id):
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def in_edges(self, node_id):
        return [e for e in self.edges if e.dst.node == node_id]

    def out_edges(self, node_id):
        return [e for e in self.edges if e.src.node == node_id]
```

`scripts/flow/__init__.py`:
```python
"""Flow: connect stages into a DAG, validate, and execute."""
from __future__ import annotations
from .graph import Graph, Node, Edge, Endpoint      # noqa: F401
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_graph.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/__init__.py scripts/flow/graph.py tests/test_flow_graph.py
git commit -m "Add flow Graph model with JSON round-trip"
```

---

### Task 3: Graph validation

**Files:**
- Create: `scripts/flow/validate.py`
- Modify: `scripts/flow/__init__.py` (re-export `validate`, `Issue`)
- Test: `tests/test_flow_validate.py`

**Interfaces:**
- Consumes: `flow.graph`, `stages` registry.
- Produces: `flow.validate.Issue(level, where, message)`; `flow.validate.validate(graph) -> list[Issue]`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_validate.py`:
```python
from flow.graph import Graph, Node, Edge, Endpoint
from flow.validate import validate


def _errs(g):
    return [i for i in validate(g) if i.level == "error"]


def test_clean_graph_has_no_errors():
    g = Graph(
        nodes=(Node("s", "load", {"path": "x", "space": "linear-adu"}),
               Node("c", "crop", {"margin": 40})),
        edges=(Edge("e1", Endpoint("s", "image"), Endpoint("c", "image")),))
    assert _errs(g) == []


def test_unknown_type():
    g = Graph(nodes=(Node("n", "nonesuch", {}),))
    assert any("unknown stage type" in i.message for i in _errs(g))


def test_bad_param():
    g = Graph(nodes=(Node("c", "crop", {"margin": -5}),),  # below min 0
              edges=())
    # crop also needs its input connected; filter to the param error
    assert any("margin" in i.message for i in _errs(g))


def test_dangling_edge_endpoint():
    g = Graph(nodes=(Node("c", "crop", {}),),
              edges=(Edge("e", Endpoint("ghost", "image"), Endpoint("c", "image")),))
    assert any("missing" in i.message for i in _errs(g))


def test_wrong_direction_port():
    g = Graph(nodes=(Node("s", "load", {}), Node("c", "crop", {})),
              edges=(Edge("e", Endpoint("s", "nope"), Endpoint("c", "image")),))
    assert any("no output port" in i.message for i in _errs(g))


def test_space_mismatch():
    # background_extract outputs LINEAR_ADU; finish requires NONLINEAR
    g = Graph(nodes=(Node("bg", "background_extract", {}), Node("f", "finish", {})),
              edges=(Edge("e", Endpoint("bg", "image"), Endpoint("f", "image")),))
    assert any("space mismatch" in i.message for i in _errs(g))


def test_missing_required_input():
    g = Graph(nodes=(Node("c", "crop", {}),))     # crop.image not connected
    assert any("required input" in i.message for i in _errs(g))


def test_double_edge_into_input():
    g = Graph(nodes=(Node("s", "load", {}), Node("s2", "load", {}), Node("c", "crop", {})),
              edges=(Edge("e1", Endpoint("s", "image"), Endpoint("c", "image")),
                     Edge("e2", Endpoint("s2", "image"), Endpoint("c", "image"))))
    assert any("inbound edges" in i.message for i in _errs(g))


def test_cycle():
    g = Graph(nodes=(Node("a", "crop", {}), Node("b", "crop", {})),
              edges=(Edge("e1", Endpoint("a", "image"), Endpoint("b", "image")),
                     Edge("e2", Endpoint("b", "image"), Endpoint("a", "image"))))
    assert any("cycle" in i.message for i in _errs(g))


def test_dead_output_is_warning():
    g = Graph(nodes=(Node("s", "load", {}), Node("c", "crop", {})),
              edges=(Edge("e1", Endpoint("s", "image"), Endpoint("c", "image")),))
    # crop.image output has no consumer -> warning, not error
    assert any(i.level == "warning" and "no consumer" in i.message for i in validate(g))
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_validate.py -v` → FAIL (`No module named 'flow.validate'`).

- [ ] **Step 3: Implement** — `scripts/flow/validate.py`:
```python
"""Static validation of a flow graph (the GUI's on-canvas error source)."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import stages


@dataclass(frozen=True)
class Issue:
    level: str      # "error" | "warning"
    where: str      # node id or edge id ("" = whole graph)
    message: str


def validate(graph) -> list:
    issues = []
    ids = {n.id for n in graph.nodes}
    stage_of = {}
    for n in graph.nodes:
        try:
            cls = stages.get(n.type)
        except KeyError:
            issues.append(Issue("error", n.id, f"unknown stage type '{n.type}'"))
            continue
        stage_of[n.id] = cls
        for p in cls.PARAMS:
            if p.name in n.params:
                val = n.params[p.name]
                if isinstance(val, str) and "{" in val:
                    continue                      # unresolved run token; skip
                try:
                    p.coerce(val)
                except Exception as e:
                    issues.append(Issue("error", n.id, f"param {p.name}: {e}"))

    input_conns = defaultdict(int)
    for e in graph.edges:
        if e.src.node not in ids:
            issues.append(Issue("error", e.id, f"edge source node '{e.src.node}' missing"))
            continue
        if e.dst.node not in ids:
            issues.append(Issue("error", e.id, f"edge target node '{e.dst.node}' missing"))
            continue
        scls, dcls = stage_of.get(e.src.node), stage_of.get(e.dst.node)
        if scls is None or dcls is None:
            continue
        outs = {p.name: p for p in scls.OUTPUTS}
        ins = {p.name: p for p in dcls.INPUTS}
        if e.src.port not in outs:
            issues.append(Issue("error", e.id, f"'{e.src.node}' has no output port '{e.src.port}'"))
        if e.dst.port not in ins:
            issues.append(Issue("error", e.id, f"'{e.dst.node}' has no input port '{e.dst.port}'"))
        if e.src.port in outs and e.dst.port in ins:
            so, si = outs[e.src.port].space, ins[e.dst.port].space
            if so is not None and si is not None and so is not si:
                issues.append(Issue("error", e.id, f"space mismatch: {so} -> {si}"))
            input_conns[(e.dst.node, e.dst.port)] += 1

    for (node, port), c in input_conns.items():
        if c > 1:
            issues.append(Issue("error", node, f"input '{port}' has {c} inbound edges"))

    for n in graph.nodes:
        cls = stage_of.get(n.id)
        if cls is None:
            continue
        for p in cls.INPUTS:
            if p.required and input_conns.get((n.id, p.name), 0) == 0:
                issues.append(Issue("error", n.id, f"required input '{p.name}' not connected"))
        for p in cls.OUTPUTS:
            if not any(e.src.node == n.id and e.src.port == p.name for e in graph.edges):
                issues.append(Issue("warning", n.id, f"output '{p.name}' has no consumer"))

    if _has_cycle(graph):
        issues.append(Issue("error", "", "graph has a cycle"))
    return issues


def _has_cycle(graph) -> bool:
    indeg = {n.id: 0 for n in graph.nodes}
    adj = defaultdict(list)
    for e in graph.edges:
        if e.src.node in indeg and e.dst.node in indeg:
            adj[e.src.node].append(e.dst.node)
            indeg[e.dst.node] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen != len(graph.nodes)
```

Add to `scripts/flow/__init__.py`:
```python
from .validate import validate, Issue      # noqa: F401,E402
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_validate.py -v` → PASS (10 tests).

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/validate.py scripts/flow/__init__.py tests/test_flow_validate.py
git commit -m "Add flow graph validation rules"
```

---

### Task 4: Content-addressed cache

**Files:**
- Create: `scripts/flow/cache.py`
- Test: `tests/test_flow_cache.py`

**Interfaces:**
- Consumes: `stages.io.load_fits/save_fits`, `flow.graph.Node`.
- Produces: `flow.cache.recipe_hash(node, input_hashes) -> str`; `flow.cache.cache_path(work_dir, node_hash, port) -> str`; `flow.cache.file_sig(path) -> str`; `flow.cache.load_cached(path) -> Image`; `flow.cache.store_cached(path, img)`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_cache.py`:
```python
import numpy as np
from astropy.io import fits
from flow.graph import Node
from flow import cache as c
from stages.image import Image, Space


def test_recipe_hash_stable_and_sensitive():
    n = Node("x", "crop", {"margin": 40})
    h1 = c.recipe_hash(n, {"image": "up1"})
    assert h1 == c.recipe_hash(Node("x", "crop", {"margin": 40}), {"image": "up1"})  # stable
    assert h1 != c.recipe_hash(Node("x", "crop", {"margin": 41}), {"image": "up1"})  # param change
    assert h1 != c.recipe_hash(n, {"image": "up2"})                                  # upstream change


def test_store_load_roundtrip(tmp_path):
    img = Image(np.random.rand(6, 6, 3).astype(np.float32), Space.NONLINEAR, fits.Header())
    p = c.cache_path(str(tmp_path), "abc123", "image")
    c.store_cached(p, img)
    back = c.load_cached(p)
    assert back.space is Space.NONLINEAR and back.pixels.shape == (6, 6, 3)
    assert np.allclose(back.pixels, img.pixels)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_cache.py -v` → FAIL (`No module named 'flow.cache'`).

- [ ] **Step 3: Implement** — `scripts/flow/cache.py`:
```python
"""Content-addressed cache of stage outputs, keyed on the recipe (not pixels)."""
from __future__ import annotations
import hashlib
import json
import os
from stages.io import load_fits, save_fits


def recipe_hash(node, input_hashes) -> str:
    payload = {
        "type": node.type,
        "params": {k: node.params[k] for k in sorted(node.params)},
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def file_sig(path) -> str:
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def cache_path(work_dir, node_hash, port) -> str:
    return os.path.join(work_dir, "cache", f"{node_hash}__{port}.fits")


def load_cached(path):
    return load_fits(path)


def store_cached(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_fits(path, img)
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_cache.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/cache.py tests/test_flow_cache.py
git commit -m "Add flow content-addressed cache"
```

---

### Task 5: Executor

**Files:**
- Create: `scripts/flow/executor.py`
- Modify: `scripts/flow/__init__.py` (re-export `run`, `RunReport`, `FlowError`)
- Test: `tests/test_flow_executor.py`

**Interfaces:**
- Consumes: `flow.validate.validate`, `flow.cache`, `stages` registry + `stages.io`.
- Produces: `flow.executor.FlowError`; `flow.executor.RunReport(outputs, warnings, cached, ran)`; `flow.executor.run(graph, input_path, label, work_dir="work", out_dir="output", cache=True) -> RunReport`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_executor.py`:
```python
import os
import numpy as np
from astropy.io import fits
import starnet
from flow.graph import Graph, Node, Edge, Endpoint
from flow.executor import run, FlowError
from stages.image import Image, Space
from stages.io import save_fits
import stages.geometry as geo


def _src(tmp_path, shape=(64, 64, 3), space=Space.NONLINEAR, val=0.4):
    p = str(tmp_path / "in.fit")
    save_fits(p, Image(np.full(shape, val, np.float32), space, fits.Header()))
    return p


def _linear_chain():
    # load -> crop -> export  (nonlinear so export's space check passes)
    return Graph(
        nodes=(Node("s", "load", {"path": "{input}", "space": "nonlinear"}),
               Node("c", "crop", {"margin": 4}),
               Node("e", "export_image", {"out_base": "{out}"})),
        edges=(Edge("e1", Endpoint("s", "image"), Endpoint("c", "image")),
               Edge("e2", Endpoint("c", "image"), Endpoint("e", "image"))))


def test_runs_and_writes_outputs(tmp_path):
    inp = _src(tmp_path)
    rep = run(_linear_chain(), inp, "v1",
              work_dir=str(tmp_path / "work"), out_dir=str(tmp_path / "out"))
    base = str(tmp_path / "out" / "in_v1")
    assert os.path.exists(base + ".tif") and os.path.exists(base + ".png")
    assert "c" in rep.ran and "s" in rep.ran


def test_missing_required_input_raises(tmp_path):
    g = Graph(nodes=(Node("c", "crop", {"margin": 2}),))   # crop.image unconnected
    with pytest.raises(FlowError):
        run(g, _src(tmp_path), "v1",
            work_dir=str(tmp_path / "w"), out_dir=str(tmp_path / "o"))


def test_cache_reuses_unchanged_upstream(tmp_path, monkeypatch):
    inp = _src(tmp_path)
    calls = {"n": 0}
    orig = geo.CropStage.apply
    def counting(self, inputs, params):
        calls["n"] += 1
        return orig(self, inputs, params)
    monkeypatch.setattr(geo.CropStage, "apply", counting)
    wd, od = str(tmp_path / "work"), str(tmp_path / "out")
    run(_linear_chain(), inp, "v1", work_dir=wd, out_dir=od)
    run(_linear_chain(), inp, "v1", work_dir=wd, out_dir=od)   # 2nd run: crop from cache
    assert calls["n"] == 1
    # a param change forces recompute
    g2 = _linear_chain()
    g2 = Graph(nodes=(g2.nodes[0], Node("c", "crop", {"margin": 6}), g2.nodes[2]),
               edges=g2.edges)
    run(g2, inp, "v1", work_dir=wd, out_dir=od)
    assert calls["n"] == 2


def test_dag_branch_merge_with_mocked_starnet(tmp_path, monkeypatch):
    inp = _src(tmp_path, shape=(512, 512, 3))
    def fake_remove(px, **k):
        return px * 0.5, np.zeros_like(px)
    monkeypatch.setattr(starnet, "remove_stars", fake_remove)
    g = Graph(
        nodes=(Node("s", "load", {"path": "{input}", "space": "nonlinear"}),
               Node("rs", "remove_stars", {"stride": 256}),
               Node("rec", "screen_recombine", {}),
               Node("e", "export_image", {"out_base": "{out}"})),
        edges=(Edge("e1", Endpoint("s", "image"), Endpoint("rs", "image")),
               Edge("e2", Endpoint("rs", "starless"), Endpoint("rec", "base")),
               Edge("e3", Endpoint("rs", "stars"), Endpoint("rec", "overlay")),
               Edge("e4", Endpoint("rec", "image"), Endpoint("e", "image"))))
    rep = run(g, inp, "v1", work_dir=str(tmp_path / "w"), out_dir=str(tmp_path / "o"))
    assert "rs" in rep.ran and "rec" in rep.ran
    assert os.path.exists(str(tmp_path / "o" / "in_v1.tif"))
```
(add `import pytest` at the top of the test file)

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_executor.py -v` → FAIL (`No module named 'flow.executor'`).

- [ ] **Step 3: Implement** — `scripts/flow/executor.py`:
```python
"""Headless DAG executor: validate -> topo-sort -> run nodes, with caching."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import os
import stages
from .validate import validate
from . import cache as _cache


class FlowError(Exception):
    pass


@dataclass
class RunReport:
    outputs: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    cached: list = field(default_factory=list)
    ran: list = field(default_factory=list)


def _topo(graph):
    indeg = {n.id: 0 for n in graph.nodes}
    adj = defaultdict(list)
    for e in graph.edges:
        adj[e.src.node].append(e.dst.node)
        indeg[e.dst.node] += 1
    queue = [n.id for n in graph.nodes if indeg[n.id] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return order


def _resolve(params, tokens):
    out = {}
    for k, v in params.items():
        if isinstance(v, str):
            for tk, tv in tokens.items():
                v = v.replace(tk, tv)
        out[k] = v
    return out


def run(graph, input_path, label, work_dir="work", out_dir="output", cache=True):
    errs = [i for i in validate(graph) if i.level == "error"]
    if errs:
        raise FlowError("; ".join(f"[{i.where}] {i.message}" for i in errs))

    name = os.path.splitext(os.path.basename(input_path))[0].replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{name}_{label}")
    tokens = {"{input}": input_path, "{out}": out_base, "{work}": work_dir}

    report = RunReport()
    payloads = {}      # (node, port) -> Image
    node_hash = {}     # node id -> recipe hash

    for nid in _topo(graph):
        node = graph.node(nid)
        cls = stages.get(node.type)
        params = _resolve(node.params, tokens)

        inputs, in_hashes = {}, {}
        for e in graph.in_edges(nid):
            inputs[e.dst.port] = payloads[(e.src.node, e.src.port)]
            in_hashes[e.dst.port] = node_hash[e.src.node]
        if node.type == "load":
            in_hashes["__file__"] = _cache.file_sig(params["path"])

        h = _cache.recipe_hash(node, in_hashes)
        node_hash[nid] = h

        out_ports = [p.name for p in cls.OUTPUTS]
        if out_ports:
            files = {p: _cache.cache_path(work_dir, h, p) for p in out_ports}
            if cache and all(os.path.exists(f) for f in files.values()):
                for p, f in files.items():
                    payloads[(nid, p)] = _cache.load_cached(f)
                report.cached.append(nid)
                report.outputs[nid] = files
                continue
            result = cls().run(inputs, params)
            for p in out_ports:
                payloads[(nid, p)] = result[p]
                _cache.store_cached(files[p], result[p])
            report.outputs[nid] = files
            report.ran.append(nid)
        else:
            cls().run(inputs, params)      # sink: writes files, not cached
            report.ran.append(nid)

    return report
```

Add to `scripts/flow/__init__.py`:
```python
from .executor import run, RunReport, FlowError      # noqa: F401,E402
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_executor.py -v` → PASS (4 tests). Then full suite green.

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/executor.py scripts/flow/__init__.py tests/test_flow_executor.py
git commit -m "Add flow executor with token substitution + caching"
```

---

### Task 6: Built-in pipeline flows

**Files:**
- Create: `scripts/flow/builtins.py`
- Modify: `scripts/flow/__init__.py` (re-export `linear_flow`, `starless_flow`)
- Test: `tests/test_flow_builtins.py`

**Interfaces:**
- Consumes: `flow.graph`, `flow.validate`, `flow.executor`.
- Produces: `flow.builtins.linear_flow() -> Graph`, `flow.builtins.starless_flow() -> Graph`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_builtins.py`:
```python
import os
import numpy as np
from astropy.io import fits
import pcc
from flow.builtins import linear_flow, starless_flow
from flow.validate import validate
from flow.executor import run
from stages.image import Image, Space
from stages.io import save_fits


def test_builtins_validate_clean():
    assert [i for i in validate(linear_flow()) if i.level == "error"] == []
    assert [i for i in validate(starless_flow()) if i.level == "error"] == []


def test_linear_flow_runs_end_to_end(tmp_path, monkeypatch):
    # linear ADU source with a header; mock PCC so no network
    p = str(tmp_path / "M 101.fit")
    save_fits(p, Image((5000 + 300 * np.random.rand(64, 64, 3)).astype(np.float32),
                       Space.LINEAR_ADU, fits.Header()))
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.0, 1.0, 1.0), {"n_matched": 10}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, path: None)
    rep = run(linear_flow(), p, "v1",
              work_dir=str(tmp_path / "work"), out_dir=str(tmp_path / "out"))
    base = str(tmp_path / "out" / "M_101_v1")
    assert os.path.exists(base + ".tif") and os.path.exists(base + ".png")
    assert "fin" in rep.ran and "col" in rep.ran
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_builtins.py -v` → FAIL (`No module named 'flow.builtins'`).

- [ ] **Step 3: Implement** — `scripts/flow/builtins.py`:
```python
"""Built-in flow graphs reproducing the standard + starless pipelines."""
from __future__ import annotations
from .graph import Graph, Node, Edge, Endpoint


def _e(i, s, sp, d, dp):
    return Edge(f"e{i}", Endpoint(s, sp), Endpoint(d, dp))


_HEAD_NODES = (
    Node("src", "load", {"path": "{input}", "space": "linear-adu"}),
    Node("crop", "crop", {"margin": 40}),
    Node("bg", "background_extract", {"degree": 3, "sample": 12, "pedestal": 0.10}),
    Node("col", "color_calibrate", {"ref_bp_rp": 0.82, "min_stars": 30,
                                    "diagnostic_path": "{out}_pcc_diagnostic.png"}),
    Node("str", "stretch", {"target_bg": 0.18, "shadows_clip": -1.8}),
)
_HEAD_EDGES = (
    _e(1, "src", "image", "crop", "image"),
    _e(2, "crop", "image", "bg", "image"),
    _e(3, "bg", "image", "col", "image"),
    _e(4, "col", "image", "str", "image"),
)


def linear_flow() -> Graph:
    nodes = _HEAD_NODES + (
        Node("fin", "finish", {"saturation": 1.20, "luma_denoise": 0.012,
                               "chroma_denoise": 4.0, "scnr": True}),
        Node("exp", "export_image", {"out_base": "{out}"}),
    )
    edges = _HEAD_EDGES + (
        _e(5, "str", "image", "fin", "image"),
        _e(6, "fin", "image", "exp", "image"),
    )
    return Graph(nodes, edges, name="linear")


def starless_flow() -> Graph:
    nodes = _HEAD_NODES + (
        Node("rs", "remove_stars", {"stride": 256}),
        Node("dn", "masked_denoise", {"bg_luma": 0.06, "bg_chroma": 10.0,
                                      "gal_luma": 0.010, "gal_chroma": 3.0, "feather": 25.0}),
        Node("sat", "saturate", {"saturation": 1.20}),
        Node("rec", "screen_recombine", {}),
        Node("exp", "export_image", {"out_base": "{out}"}),
        Node("pv_sl", "preview_sink", {"out_path": "{out}_starless.png", "stretch": False}),
        Node("pv_st", "preview_sink", {"out_path": "{out}_starlayer.png", "stretch": False}),
    )
    edges = _HEAD_EDGES + (
        _e(5, "str", "image", "rs", "image"),
        _e(6, "rs", "starless", "dn", "image"),
        _e(7, "dn", "image", "sat", "image"),
        _e(8, "sat", "image", "rec", "base"),
        _e(9, "rs", "stars", "rec", "overlay"),
        _e(10, "rec", "image", "exp", "image"),
        _e(11, "sat", "image", "pv_sl", "image"),
        _e(12, "rs", "stars", "pv_st", "image"),
    )
    return Graph(nodes, edges, name="starless")
```

Add to `scripts/flow/__init__.py`:
```python
from .builtins import linear_flow, starless_flow      # noqa: F401,E402
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_builtins.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/builtins.py scripts/flow/__init__.py tests/test_flow_builtins.py
git commit -m "Add built-in linear + starless flow graphs"
```

---

### Task 7: CLI (`python -m flow`)

**Files:**
- Create: `scripts/flow/__main__.py`
- Test: `tests/test_flow_cli.py`

**Interfaces:**
- Consumes: `flow.builtins`, `flow.graph`, `flow.validate`, `flow.executor`, `stages`.
- Produces: `flow.__main__.main(argv=None) -> int` with subcommands `run` / `validate` / `schema`.

- [ ] **Step 1: Write the failing test** — `tests/test_flow_cli.py`:
```python
import json
import os
import numpy as np
from astropy.io import fits
import pcc
from flow.__main__ import main
from stages.image import Image, Space
from stages.io import save_fits


def test_schema_lists_all_stages(capsys):
    rc = main(["schema"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    ids = {s["id"] for s in data}
    assert "load" in ids and "remove_stars" in ids and len(ids) >= 13


def test_validate_builtin_ok():
    assert main(["validate", "--builtin", "starless"]) == 0


def test_validate_broken_graph_file(tmp_path):
    bad = {"nodes": [{"id": "n", "type": "nonesuch", "params": {}}], "edges": []}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert main(["validate", str(p)]) == 1


def test_run_builtin_linear(tmp_path, monkeypatch):
    p = str(tmp_path / "M 101.fit")
    save_fits(p, Image((5000 + 300 * np.random.rand(64, 64, 3)).astype(np.float32),
                       Space.LINEAR_ADU, fits.Header()))
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.0, 1.0, 1.0), {"n_matched": 10}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, path: None)
    monkeypatch.chdir(tmp_path)     # output/ + work/ land under tmp
    rc = main(["run", "--builtin", "linear", "--input", p, "--label", "v1"])
    assert rc == 0
    assert os.path.exists(str(tmp_path / "output" / "M_101_v1.tif"))
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_flow_cli.py -v` → FAIL (`No module named 'flow.__main__'`).

- [ ] **Step 3: Implement** — `scripts/flow/__main__.py`:
```python
"""Flow CLI: run | validate | schema."""
from __future__ import annotations
import argparse
import json
import sys
import stages
from .graph import Graph
from .validate import validate
from .executor import run, FlowError
from . import builtins as _builtins


def _load_graph(args):
    if args.builtin:
        return getattr(_builtins, f"{args.builtin}_flow")()
    if not args.flow:
        raise SystemExit("provide a FLOW.json or --builtin")
    with open(args.flow) as f:
        return Graph.from_json(json.load(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("flow", nargs="?")
    r.add_argument("--builtin", choices=["linear", "starless"])
    r.add_argument("--input", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--no-cache", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("flow", nargs="?")
    v.add_argument("--builtin", choices=["linear", "starless"])

    sub.add_parser("schema")

    a = ap.parse_args(argv)

    if a.cmd == "schema":
        print(json.dumps(stages.list_stages(), indent=2))
        return 0
    if a.cmd == "validate":
        issues = validate(_load_graph(a))
        for i in issues:
            print(f"{i.level.upper()} [{i.where}] {i.message}")
        return 1 if any(i.level == "error" for i in issues) else 0
    if a.cmd == "run":
        try:
            rep = run(_load_graph(a), a.input, a.label, cache=not a.no_cache)
        except FlowError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"ran {len(rep.ran)}, cached {len(rep.cached)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_flow_cli.py -v` → PASS; full suite green.

- [ ] **Step 5: Commit**
```bash
git add scripts/flow/__main__.py tests/test_flow_cli.py
git commit -m "Add flow CLI (run/validate/schema)"
```

---

### Task 8: Unify run_pipeline + README + end-to-end verification

**Files:**
- Modify: `scripts/run_pipeline.sh`, `README.md`

**Interfaces:**
- Consumes: `python -m flow run --builtin`.
- Produces: `run_pipeline.sh` as a shim over the executor; `make run` / `make run-starless` unchanged.

- [ ] **Step 1: Rewrite `run_pipeline.sh` to shim over the executor.** Keep the venv check and the `--starless`/label parsing; replace the numbered-step chain with a single executor call. The final file:
```bash
#!/usr/bin/env bash
# Run the pipeline via the flow executor (built-in linear/starless graphs).
#
# Usage: scripts/run_pipeline.sh <input.fit> [version-label] [--starless]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Run 'make setup' (or see README.md) first." >&2
  exit 1
fi

STARLESS=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --starless) STARLESS=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

IN="${1:?usage: run_pipeline.sh <input.fit> [version-label] [--starless]}"
case "$IN" in /*) ;; *) IN="$PWD/$IN" ;; esac
LABEL="${2:-$(date +%Y%m%d-%H%M%S)}"

FLOW=linear
[[ "$STARLESS" == "1" ]] && FLOW=starless

cd "$ROOT"                      # so output/ and work/ resolve at repo root
echo ">> flow run --builtin $FLOW"
PYTHONPATH="$HERE" exec "$PY" -m flow run --builtin "$FLOW" --input "$IN" --label "$LABEL"
```

- [ ] **Step 2: Syntax-check** — `bash -n scripts/run_pipeline.sh` → exit 0. `make -n run FITS=x` still prints the `run_pipeline.sh "x"` invocation.

- [ ] **Step 3: Full suite** — `.venv/bin/python -m pytest -q` → all pass.

- [ ] **Step 4: Flow CLI smoke** —
```bash
PYTHONPATH=scripts .venv/bin/python -m flow schema | .venv/bin/python -c "import sys,json; print(len(json.load(sys.stdin)),'stages')"
PYTHONPATH=scripts .venv/bin/python -m flow validate --builtin linear && echo "linear OK"
PYTHONPATH=scripts .venv/bin/python -m flow validate --builtin starless && echo "starless OK"
```
Expected: 13 stages; both builtins print OK (exit 0).

- [ ] **Step 5: Document in README.md.** Add a section for the `scripts/flow/` package: the graph/JSON model, `python -m flow run|validate|schema`, that `run_pipeline.sh` is now a shim over it, the `{input}`/`{out}` tokens, and the `work/cache/` caching (`--no-cache`, `make clean`). Note `stages.list_stages()` / `flow schema` is the GUI node-palette contract.

- [ ] **Step 6: Commit the shim + docs**
```bash
git add scripts/run_pipeline.sh README.md
git commit -m "Unify run_pipeline over the flow executor; document flow package"
```

- [ ] **Step 7: End-to-end parity (real data).** Run the standard finish through the unified runner:
```bash
make run FITS="data/M101_restack_solved.fit" V=flow-check
```
Expected: `>> flow run --builtin linear`, PCC runs (Gaia), `output/M101_restack_solved_flow-check.{tif,png}` produced. Compare visually against `output/M101_restack_solved_stages-check.png` (the pre-flow stages run) — equivalent.

- [ ] **Step 8: Cache reuse + starless.** Re-run the same standard finish and confirm the cache short-circuits, then the starless finish:
```bash
make run FITS="data/M101_restack_solved.fit" V=flow-check2      # 2nd run: mostly cached, fast
make run-starless FITS="data/M101_restack_solved.fit" V=flow-check
```
Expected: the 2nd standard run finishes quickly and its `ran/cached` summary shows upstream nodes served from cache; the starless run produces `output/M101_restack_solved_flow-check.{tif,png}` + `_starless.png` + `_starlayer.png`. (No commit — outputs are git-ignored.)

---

## Self-Review

**Spec coverage:**
- LoadStage source node → Task 1. ✓
- Graph model + JSON round-trip (ui preserved, unknown keys tolerated) → Task 2. ✓
- Validation rules (all 8 error rules + dead-branch warning) → Task 3. ✓
- Content-addressed cache (recipe hash folds upstream; file_sig for source) → Task 4. ✓
- Executor: validate→token-resolve→topo→run→cache, DAG branch/merge, sinks uncached, FlowError-before-sinks → Task 5. ✓
- Built-in linear + starless flows → Task 6. ✓
- CLI run/validate/schema → Task 7. ✓
- run_pipeline unified + tokens/cache documented + real end-to-end parity → Task 8. ✓
- Tokens `{input}`/`{out}`/`{work}` → Task 5 `_resolve`. ✓
- Keep numbered shims (untouched) → confirmed (no task modifies 01–05b). ✓
- Out of scope (GUI, parallelism, eviction) → not implemented. ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `Graph/Node/Edge/Endpoint`, `validate()->[Issue]`, `recipe_hash(node, input_hashes)`, `cache_path(work_dir, hash, port)`, `run(graph, input_path, label, work_dir, out_dir, cache)->RunReport`, `linear_flow()/starless_flow()`, `main(argv)->int`, and stage ids (`load`, plus the existing 12) are used consistently across defining and consuming tasks. The starless builtin omits an unsharp node (matches the shim default SHARPEN_AMOUNT=0.0). ✓
```
