from pathlib import Path

import numpy as np
import pytest

from trajectory_datasets.recipes import uci_characters


def _mat(path: str) -> dict:
    consts = [[[None, None, None, None, [[1, 20]], [[0.5]]]]]
    stroke = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    return {"consts": consts, "mixout": [[stroke, stroke]]}


def test_strokes_become_trajectories(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uci_characters, "loadmat", _mat)
    trajs, labels = uci_characters._yupify(raw_dir)
    assert labels == ["a", "z"]
    assert trajs[0].r.x.tolist() == [0.0, 1.0, 2.0]
    assert trajs[0].t.tolist() == [0.0, 0.5, 1.0]


def test_build_uses_the_download(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uci_characters, "loadmat", _mat)
    monkeypatch.setattr(uci_characters, "download_dataset", lambda *a, **k: raw_dir / "x.mat")
    assert uci_characters.build()[1] == ["a", "z"]
