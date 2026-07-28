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
