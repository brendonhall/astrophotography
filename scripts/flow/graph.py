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
