from pathlib import Path
from string import Template

import pytest
from yupi import Trajectory

from trajectory_datasets.recipes import mnist_stroke


def _write(folder: Path, name: str) -> None:
    (folder / name).write_text("x,y\n1,2\n3,4\n-1,-1\n5,6\n", encoding="utf-8")


def test_labels_skip_the_header(raw_dir: Path) -> None:
    path = raw_dir / "labels"
    path.write_bytes(b"12345678" + bytes([7, 2, 1]))
    assert mnist_stroke._read_labels(path, 2) == ["7", "2"]


def test_pen_lift_markers_are_dropped(raw_dir: Path) -> None:
    _write(raw_dir, "a.txt")
    assert mnist_stroke._read_traj(raw_dir / "a.txt").r.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_a_split_pairs_labels_and_trajectories(raw_dir: Path) -> None:
    _write(raw_dir, "img-0-points.txt")
    labels = raw_dir / "labels"
    labels.write_bytes(b"12345678" + bytes([9]))
    trajs, read = mnist_stroke._yupify_mnist(raw_dir, labels, Template("img-$id-points.txt"), 1)
    assert read == ["9"]
    assert len(trajs) == 1


def test_train_and_test_are_joined(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake(folder: Path, labels: Path, template: Template, count: int) -> tuple[list, list]:
        calls.append((labels.name, count))
        return [Trajectory(x=[0.0, float(count)])], [str(count)]

    monkeypatch.setattr(mnist_stroke, "_yupify_mnist", fake)
    assert mnist_stroke._yupify(raw_dir)[1] == ["60000", "10000"]
    assert calls == [("train-labels.idx1-ubyte", 60000), ("t10k-labels.idx1-ubyte", 10000)]


def test_build_fetches_every_file(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_download(url: str, name: str) -> Path:
        urls.append(url)
        return raw_dir / "sequences.tar.gz"

    monkeypatch.setattr(mnist_stroke, "download_dataset", fake_download)
    monkeypatch.setattr(mnist_stroke, "_yupify", lambda folder: ([], [str(folder)]))
    assert mnist_stroke.build()[1] == [str(raw_dir)]
    assert len(urls) == 3
