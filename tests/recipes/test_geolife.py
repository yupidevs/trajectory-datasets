from pathlib import Path

import pytest

from trajectory_datasets.recipes import geolife

HEADER = "".join(f"line {i}\n" for i in range(6))


def _point(second: int, lat: float) -> str:
    return f"{lat},116.0,0,100,39000.0,2008-04-02,11:00:{second:02d}\n"


def _user(raw_dir: Path, name: str, labels: str | None) -> None:
    folder = raw_dir / "Geolife Trajectories 1.3" / "Data" / name
    (folder / "Trajectory").mkdir(parents=True)
    points = "".join(_point(s, 39.0 + s / 100) for s in range(10))
    points += _point(9, 39.5) + _point(50, 40.0)
    (folder / "Trajectory" / "a.plt").write_text(HEADER + points, encoding="utf-8")
    if labels is not None:
        (folder / "labels.txt").write_text(labels, encoding="utf-8")


LABELS = "Start End Mode\n2008/04/02 11:00:00 2008/04/02 11:00:09 bus\n"


def test_labelled_points_become_trajectories(raw_dir: Path) -> None:
    _user(raw_dir, "000", LABELS)
    trajs, labels = geolife.yupify(raw_dir)
    assert labels == ["bus"]
    assert trajs[0].t.tolist() == [float(s) for s in range(10)]
    assert trajs[0].r.y[1] == pytest.approx(39.01)


def test_users_without_labels_are_skipped(raw_dir: Path) -> None:
    _user(raw_dir, "000", None)
    assert geolife.yupify(raw_dir) == ([], [])


def test_short_segments_are_dropped(raw_dir: Path) -> None:
    _user(raw_dir, "000", "Start End Mode\n2008/04/02 11:00:00 2008/04/02 11:00:02 car\n")
    assert geolife.yupify(raw_dir) == ([], [])


def test_several_labels(raw_dir: Path) -> None:
    labels = (
        "Start End Mode\n2008/04/02 11:00:00 2008/04/02 11:00:02 car\n"
        "2008/04/02 11:00:03 2008/04/02 11:00:09 walk\n"
    )
    _user(raw_dir, "000", labels)
    assert geolife.yupify(raw_dir)[1] == ["walk"]


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _user(raw_dir, "000", LABELS)
    monkeypatch.setattr(geolife, "download_dataset", lambda *a, **k: raw_dir / "archive.zip")
    assert geolife.build()[1] == ["bus"]
