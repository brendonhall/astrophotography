import os
import numpy as np
from stages.image import Image, Space
from stages.export import ExportImageStage, PreviewSink


def test_export_writes_three_files(tmp_path):
    img = Image(np.clip(np.random.rand(20, 20, 3), 0, 1).astype(np.float64), Space.NONLINEAR)
    base = str(tmp_path / "out")
    ExportImageStage().run({"image": img}, {"out_base": base})
    for suffix in (".tif", ".png", "_preview.png"):
        assert os.path.exists(base + suffix), suffix


def test_preview_sink_writes_png(tmp_path):
    img = Image(np.clip(np.random.rand(20, 20, 3), 0, 1).astype(np.float64), Space.NONLINEAR)
    p = str(tmp_path / "prev.png")
    PreviewSink().run({"image": img}, {"out_path": p, "stretch": False})
    assert os.path.exists(p)
