from pathlib import Path

import pytest

from trajectory_datasets.recipes import uci_pen_digits as pen


def _write(raw_dir: Path) -> Path:
    (raw_dir / "pendigits.tes").write_text("1,2,3,4,8\n", encoding="utf-8")
    train = raw_dir / "pendigits.tra"
    train.write_text("5,6,7,8,3\n", encoding="utf-8")
    return train


def test_train_comes_before_test(raw_dir: Path) -> None:
    _write(raw_dir)
    trajs, labels = pen._yupify(raw_dir)
    assert labels == [3, 8]
    assert trajs[0].r.tolist() == [[5.0, 6.0], [7.0, 8.0]]


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = _write(raw_dir)
    monkeypatch.setattr(pen, "download_dataset", lambda *a, **k: train)
    assert pen.build()[1] == [3, 8]
