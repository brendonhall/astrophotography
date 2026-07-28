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
