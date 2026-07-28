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
