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
