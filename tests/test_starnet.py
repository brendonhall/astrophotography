import os
import numpy as np
import pytest
import starnet


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
