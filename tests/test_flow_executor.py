import os
import pytest
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


def test_cache_key_folds_source_port(tmp_path, monkeypatch):
    # A multi-output node (remove_stars) has two distinct output ports. A
    # downstream cacheable node (crop) wired from one port vs the other must
    # get a different cache key -- otherwise re-wiring which port feeds it
    # (same node ids, same work_dir) would incorrectly serve the other
    # port's cached result.
    inp = _src(tmp_path, shape=(512, 512, 3))

    def fake_remove(px, **k):
        return px * 0.25, px * 0.75   # starless, stars: distinct arrays

    monkeypatch.setattr(starnet, "remove_stars", fake_remove)

    def _graph(src_port):
        return Graph(
            nodes=(Node("s", "load", {"path": "{input}", "space": "nonlinear"}),
                   Node("rs", "remove_stars", {"stride": 256}),
                   Node("c", "crop", {"margin": 4}),
                   Node("e", "export_image", {"out_base": "{out}"})),
            edges=(Edge("e1", Endpoint("s", "image"), Endpoint("rs", "image")),
                   Edge("e2", Endpoint("rs", src_port), Endpoint("c", "image")),
                   Edge("e3", Endpoint("c", "image"), Endpoint("e", "image"))))

    wd, od = str(tmp_path / "work"), str(tmp_path / "out")
    out_path = str(tmp_path / "out" / "in_v1.tif")

    from skimage.io import imread

    run(_graph("starless"), inp, "v1", work_dir=wd, out_dir=od)
    arr1 = imread(out_path)

    # Second graph: same node ids, same work_dir, but "c" is now wired from
    # rs's OTHER output port into the SAME dst port ("c.image").
    run(_graph("stars"), inp, "v1", work_dir=wd, out_dir=od)
    arr2 = imread(out_path)

    assert not np.array_equal(arr1, arr2)


def test_report_surfaces_validation_warnings(tmp_path):
    # crop's output is never wired to anything -> validate() emits a
    # "no consumer" warning (not an error), which the run() report must
    # surface for the CLI to print.
    inp = _src(tmp_path)
    g = Graph(nodes=(Node("s", "load", {"path": "{input}", "space": "nonlinear"}),
                     Node("c", "crop", {"margin": 4})),
              edges=(Edge("e1", Endpoint("s", "image"), Endpoint("c", "image")),))
    rep = run(g, inp, "v1", work_dir=str(tmp_path / "w"), out_dir=str(tmp_path / "o"))
    assert any("no consumer" in w for w in rep.warnings)


def test_runtime_stage_error_wrapped_as_flow_error(tmp_path):
    # remove_stars.check() raises StageError at runtime when the image is
    # below StarNet2's minimum size; the executor must wrap it as a
    # FlowError with node context rather than let it escape raw.
    inp = _src(tmp_path, shape=(64, 64, 3))
    g = Graph(nodes=(Node("s", "load", {"path": "{input}", "space": "nonlinear"}),
                     Node("rs", "remove_stars", {"stride": 256})),
              edges=(Edge("e1", Endpoint("s", "image"), Endpoint("rs", "image")),))
    with pytest.raises(FlowError, match=r"\[rs\]"):
        run(g, inp, "v1", work_dir=str(tmp_path / "w"), out_dir=str(tmp_path / "o"))
