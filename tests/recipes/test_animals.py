from pathlib import Path

import numpy as np
import pytest

from trajectory_datasets.recipes import animals

HEADER = "Row, Id, LocDate, LocTime, Species, UTME, UTMN\n"
FILE = "Starkey_OR_Main_Telemetry_1993-1996_Data.txt"


def _write(raw_dir: Path, rows: list[str]) -> None:
    (raw_dir / FILE).write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def test_rows_become_sorted_trajectories(raw_dir: Path) -> None:
    _write(
        raw_dir,
        [
            "1, a, 19930101, 00:01:00,E, 380000, 5000000",
            "2, a, 19930101, 00:00:00,E, 381000, 5001000",
            "3, a, 19930101, 00:00:00,E, 382000, 5002000",
            "4, b, 19930101, 00:00:00,D, 380000, 5000000",
            "5, b, 19930101, 00:02:00,D, 380500, 5000500",
        ],
    )
    trajs, labels = animals._yupify(raw_dir)
    assert labels == ["Elk", "Deer"]
    assert trajs[0].t.tolist() == [0.0, 60.0]
    assert trajs[1].t.tolist() == [0.0, 120.0]
    assert np.all(trajs[0].r.y > 40)


def test_mixed_species_are_rejected(raw_dir: Path) -> None:
    _write(
        raw_dir,
        [
            "1, a, 19930101, 00:00:00,E, 380000, 5000000",
            "2, a, 19930101, 00:01:00,D, 380000, 5000000",
        ],
    )
    with pytest.raises(ValueError, match="Mixed species"):
        animals._yupify(raw_dir)


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(
        raw_dir,
        [
            "1, a, 19930101, 00:00:00,C, 380000, 5000000",
            "2, a, 19930101, 00:01:00,C, 380500, 5000500",
        ],
    )
    monkeypatch.setattr(animals, "download_dataset", lambda *a, **k: raw_dir / FILE)
    assert animals.build()[1] == ["Cattle"]
