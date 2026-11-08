from pathlib import Path

import pytest

from trajectory_datasets.recipes import traffic

FILE = "TRAJECTORY_PUB_SMP.csv"


def _car(car: str, kind: int) -> list[str]:
    return [f"{car},08000{s}000,{kind},0,0,{s}.5,{s + 1}.5" for s in range(3)]


def _write(raw_dir: Path) -> Path:
    rows = [*_car("1", 1), *_car("2", 2), "3,080000000,1,0,0,1.0,1.0", "4,080000000,1,0,0,1.0,1.0"]
    path = raw_dir / FILE
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_cars_become_labelled_trajectories(raw_dir: Path) -> None:
    _write(raw_dir)
    trajs, labels = traffic._yupify(raw_dir)
    assert labels == ["normal", "large"]
    assert trajs[0].r.x.tolist() == [0.5, 1.5, 2.5]
    assert trajs[0].t.tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(raw_dir)
    monkeypatch.setattr(traffic, "download_dataset", lambda *a, **k: path)
    assert traffic.build()[1] == ["normal", "large"]
