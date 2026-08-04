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
    save_fits(p, Image((5000 + 300 * np.random.rand(200, 200, 3)).astype(np.float32),
                       Space.LINEAR_ADU, fits.Header()))
    monkeypatch.setattr(pcc, "photometric_calibration",
                        lambda px, hdr, **k: ((1.0, 1.0, 1.0), {"n_matched": 10}))
    monkeypatch.setattr(pcc, "save_diagnostic", lambda r, path: None)
    monkeypatch.chdir(tmp_path)     # output/ + work/ land under tmp
    rc = main(["run", "--builtin", "linear", "--input", p, "--label", "v1"])
    assert rc == 0
    assert os.path.exists(str(tmp_path / "output" / "M_101_v1.tif"))
