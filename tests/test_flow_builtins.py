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
    save_fits(p, Image((5000 + 300 * np.random.rand(200, 200, 3)).astype(np.float32),
                       Space.LINEAR_ADU, fits.Header()))
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.0, 1.0, 1.0), {"n_matched": 10}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, path: None)
    rep = run(linear_flow(), p, "v1",
              work_dir=str(tmp_path / "work"), out_dir=str(tmp_path / "out"))
    base = str(tmp_path / "out" / "M_101_v1")
    assert os.path.exists(base + ".tif") and os.path.exists(base + ".png")
    assert "fin" in rep.ran and "col" in rep.ran
