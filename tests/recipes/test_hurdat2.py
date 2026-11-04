from pathlib import Path

import pytest

from trajectory_datasets.recipes import hurdat2

FILE = "hurdat2-1851-2021-100522.txt"


def _rows(wind: int, lon: str = "94.8W") -> str:
    return "".join(f"1851062{d}, 0000, , HU, 28.{d}N, {lon}, {wind}, 0\n" for d in range(3))


def _write(raw_dir: Path, body: str) -> Path:
    path = raw_dir / FILE
    path.write_text(body + "AL999999, END, 0,\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("wind", "category"), [(137, 5), (136, 4), (113, 4), (96, 3), (83, 2), (64, 1), (63, 0)]
)
def test_saffir_simpson_scale(wind: float, category: int) -> None:
    assert hurdat2._get_saffir_simpson_scale(wind) == category


def test_storms_are_read(raw_dir: Path) -> None:
    _write(raw_dir, "AL011851, UNNAMED, 3,\n" + _rows(90))
    trajs, labels = hurdat2._yupify(raw_dir)
    assert labels == [2]
    assert trajs[0].r.x.tolist() == [-94.8, -94.8, -94.8]
    assert trajs[0].t.tolist() == [0.0, 86400.0, 172800.0]


def test_east_longitudes_stay_positive(raw_dir: Path) -> None:
    _write(raw_dir, "AL011851, UNNAMED, 3,\n" + _rows(30, "10.0E"))
    assert hurdat2._yupify(raw_dir)[0][0].r.x.tolist() == [10.0, 10.0, 10.0]


def test_unknown_positions_are_rejected(raw_dir: Path) -> None:
    _write(raw_dir, "AL011851, UNNAMED, 3,\n" + _rows(30, "10.0X"))
    with pytest.raises(ValueError, match="Unknown position"):
        hurdat2._yupify(raw_dir)


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(raw_dir, "AL011851, UNNAMED, 3,\n" + _rows(30))
    monkeypatch.setattr(hurdat2, "download_dataset", lambda *a, **k: path)
    assert hurdat2.build()[1] == [0]
