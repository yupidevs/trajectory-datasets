from pathlib import Path

import pytest

from trajectory_datasets.recipes import uci_gotrack


def _point(track: int, second: int) -> str:
    return f"{track},-10.{second},-37.{second},2015-01-01 10:00:{second:02d}\n"


def _write(raw_dir: Path) -> None:
    folder = raw_dir / "GPS Trajectory"
    folder.mkdir()
    (folder / "go_track_tracks.csv").write_text(
        "id,car_or_bus\n1,1\n2,2\n3,1\n4,1\n", encoding="utf-8"
    )
    points = "track_id,latitude,longitude,time\n"
    points += "".join(_point(1, s) for s in (1, 2, 3, 3, 4))
    points += "".join(_point(2, s) for s in range(1, 6))
    points += _point(99, 1)
    points += _point(3, 1)
    points += "".join(_point(4, 1) for _ in range(3))
    points += _point(1, 9)
    (folder / "go_track_trackspoints.csv").write_text(points, encoding="utf-8")


def test_tracks_become_labelled_trajectories(raw_dir: Path) -> None:
    _write(raw_dir)
    trajs, labels = uci_gotrack._yupify(raw_dir)
    assert labels == ["car", "bus"]
    assert trajs[0].t.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert trajs[1].r.x[0] == pytest.approx(-37.1)


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(raw_dir)
    monkeypatch.setattr(uci_gotrack, "download_dataset", lambda *a, **k: raw_dir / "x.rar")
    assert uci_gotrack.build()[1] == ["car", "bus"]
