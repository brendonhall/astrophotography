"""Headless DAG executor: validate -> topo-sort -> run nodes, with caching."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import os
import stages
from stages.base import StageError
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
    issues = validate(graph)
    errs = [i for i in issues if i.level == "error"]
    if errs:
        raise FlowError("; ".join(f"[{i.where}] {i.message}" for i in errs))

    name = os.path.splitext(os.path.basename(input_path))[0].replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{name}_{label}")
    tokens = {"{input}": input_path, "{out}": out_base, "{work}": work_dir}

    report = RunReport()
    report.warnings = [f"[{i.where}] {i.message}" for i in issues if i.level == "warning"]
    payloads = {}      # (node, port) -> Image
    node_hash = {}     # node id -> recipe hash

    for nid in _topo(graph):
        node = graph.node(nid)
        cls = stages.get(node.type)
        params = _resolve(node.params, tokens)

        inputs, in_hashes = {}, {}
        for e in graph.in_edges(nid):
            inputs[e.dst.port] = payloads[(e.src.node, e.src.port)]
            in_hashes[e.dst.port] = f"{node_hash[e.src.node]}:{e.src.port}"
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
            try:
                result = cls().run(inputs, params)
            except StageError as e:
                raise FlowError(f"[{nid}] {e}")
            for p in out_ports:
                payloads[(nid, p)] = result[p]
                _cache.store_cached(files[p], result[p])
            report.outputs[nid] = files
            report.ran.append(nid)
        else:
            try:
                cls().run(inputs, params)      # sink: writes files, not cached
            except StageError as e:
                raise FlowError(f"[{nid}] {e}")
            report.ran.append(nid)

    return report
