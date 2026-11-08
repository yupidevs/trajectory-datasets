from pathlib import Path

import pytest

from trajectory_datasets.recipes import cma_bst


def _write(raw_dir: Path) -> Path:
    text = """66666 0000 storm
2020010100 1 150 1200 0
2020010106 4 155 1210 0
2020010112 2 160 1220 0
66666 0001 short
2020020100 1 150 1200 0
66666 0002 end
"""
    path = raw_dir / "CH2020BST.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_storms_are_read_with_their_strongest_category(raw_dir: Path) -> None:
    _write(raw_dir)
    trajs, labels = cma_bst._yupify(raw_dir)
    assert labels == ["Typhoon"]
    assert trajs[0].t.tolist() == [0.0, 21600.0, 43200.0]
    assert trajs[0].r.y.tolist() == [150.0, 155.0, 160.0]


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(raw_dir)
    monkeypatch.setattr(cma_bst, "download_dataset", lambda *a, **k: path)
    assert cma_bst.build()[1] == ["Typhoon"]
