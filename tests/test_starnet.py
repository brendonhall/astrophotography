import os
import numpy as np
import pytest
import starnet
from astropy.io import fits


def _fake_exe(tmp_path, name="starnet2"):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return str(p)


def test_find_binary_prefers_explicit_arg(tmp_path):
    exe = _fake_exe(tmp_path)
    assert starnet.find_binary(exe) == exe


def test_find_binary_uses_env(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    monkeypatch.setenv("STARNET2_CLI", exe)
    assert starnet.find_binary() == exe


def test_find_binary_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("STARNET2_CLI", raising=False)
    missing = str(tmp_path / "nope")
    with pytest.raises(starnet.StarNetError):
        starnet.find_binary(missing)


def _write_cube(path, img):  # (H,W,3) -> (3,H,W) float32 FITS
    fits.PrimaryHDU(data=np.moveaxis(img.astype(np.float32), -1, 0)).writeto(
        path, overwrite=True)


def test_remove_stars_rejects_small_image(tmp_path):
    small = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(starnet.StarNetError):
        starnet.remove_stars(small, binary=_fake_exe(tmp_path))


def test_remove_stars_roundtrip(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    img = np.random.RandomState(0).uniform(0, 1, (512, 512, 3)).astype(np.float32)
    calls = {}

    def fake_run(argv, capture_output, text):
        calls["argv"] = argv
        out = argv[argv.index("-o") + 1]
        stars = argv[argv.index("-n") + 1]
        _write_cube(out, img * 0.5)
        _write_cube(stars, np.zeros_like(img))

        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        return R()

    monkeypatch.setattr(starnet.subprocess, "run", fake_run)
    starless, stars = starnet.remove_stars(img, binary=exe, stride=128)
    assert starless.shape == (512, 512, 3) and stars.shape == (512, 512, 3)
    assert starless.min() >= 0 and starless.max() <= 1
    argv = calls["argv"]
    assert argv[0] == exe
    for flag in ("-i", "-o", "-n"):
        assert flag in argv
    assert "128" in argv  # stride forwarded


def test_remove_stars_nonzero_exit_raises(tmp_path, monkeypatch):
    exe = _fake_exe(tmp_path)
    img = np.zeros((512, 512, 3), dtype=np.float32)

    def fake_run(argv, capture_output, text):
        class R:
            returncode = 3
            stderr = "boom"
            stdout = ""
        return R()

    monkeypatch.setattr(starnet.subprocess, "run", fake_run)
    with pytest.raises(starnet.StarNetError) as e:
        starnet.remove_stars(img, binary=exe)
    assert "boom" in str(e.value)


@pytest.mark.skipif(
    not os.access(os.environ.get("STARNET2_CLI", starnet.DEFAULT_BINARY), os.X_OK),
    reason="StarNet2 binary not installed")
def test_remove_stars_real_binary_removes_stars():
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:512, 0:512]
    field = 0.15 + 0.05 * np.exp(-(((xx - 256) ** 2 + (yy - 256) ** 2) / (2 * 120.0 ** 2)))
    for _ in range(30):
        cy, cx = rng.randint(20, 492), rng.randint(20, 492)
        field += 0.8 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2)))
    img = np.clip(np.stack([field] * 3, -1), 0, 1).astype(np.float32)
    starless, stars = starnet.remove_stars(img)
    assert starless.max() < img.max() - 0.2          # stars removed
    import astrolib as al
    assert np.allclose(al.screen(starless, stars), img, atol=0.05)  # split recombines
