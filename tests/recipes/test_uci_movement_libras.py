from pathlib import Path

import pytest

from trajectory_datasets.recipes import uci_movement_libras as libras


def _write(raw_dir: Path) -> Path:
    path = raw_dir / "movement_libras.data"
    path.write_text("0.1,0.2,0.3,0.4,1\n0.5,0.6,0.7,0.8,15\n", encoding="utf-8")
    return path


def test_rows_become_labelled_trajectories(raw_dir: Path) -> None:
    _write(raw_dir)
    trajs, labels = libras._yupify(raw_dir)
    assert labels == ["curved swing", "tremble"]
    assert trajs[0].r.tolist() == [[0.1, 0.2], [0.3, 0.4]]


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(raw_dir)
    monkeypatch.setattr(libras, "download_dataset", lambda *a, **k: path)
    assert libras.build()[1] == ["curved swing", "tremble"]
