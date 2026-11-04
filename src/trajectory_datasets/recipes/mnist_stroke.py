import logging
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
from yupi import Trajectory

from trajectory_datasets.downloads import download_dataset

logger = logging.getLogger(__name__)

VERSION = 0
NAME = "mnist_stroke"

_SEQUENCES_URL = (
    "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
)
_TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
_TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


def build() -> tuple[list[Trajectory], list[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_SEQUENCES_URL, NAME)
    download_dataset(_TRAIN_LABELS_URL, NAME)
    download_dataset(_TEST_LABELS_URL, NAME)
    return raw_trajs_filepath.parent


def _read_labels(label_file: Path, count: int) -> list[str]:
    with label_file.open("rb") as f:
        f.read(8)  # discard header info
        return [str(label) for label in f.read(count)]


def _read_traj(trajectory_file: Path) -> Trajectory:
    # Load raw data from the stroke
    traj = np.loadtxt(trajectory_file, delimiter=",", skiprows=1)

    # Filter tokens from changes
    points = [(i, j) for i, j in traj if i >= 0 and j >= 0]

    return Trajectory(points=points)


def _read_trajs(sequence_folder: Path, template: Template, count: int) -> list[Trajectory]:
    trajs = []
    for i in range(count):
        traj_path = sequence_folder / template.substitute(id=i)
        trajs.append(_read_traj(traj_path))

    return trajs


def _yupify_mnist(
    sequence_folder: Path, label_file: Path, template: Template, count: int
) -> tuple[list[Trajectory], list[str]]:
    """Yupifies a part of the dataset"""

    labels = _read_labels(label_file, count)
    trajs = _read_trajs(sequence_folder, template, count)

    return trajs, labels


def _yupify(raw_dir: Path) -> tuple[list[Trajectory], list[str]]:
    # Loads the raw data and preprocess it
    logger.info("Preprocessing MNIST stroke raw data")
    sequence_path = raw_dir / "sequences"
    train_labels_path = raw_dir / "train-labels.idx1-ubyte"
    test_labels_path = raw_dir / "t10k-labels.idx1-ubyte"

    logger.info("Yupifying train dataset...")
    train_template = Template("trainimg-$id-points.txt")
    trajs_train, labels_train = _yupify_mnist(
        sequence_path, train_labels_path, template=train_template, count=60000
    )

    logger.info("Yupifying test dataset...")
    test_template = Template("testimg-$id-points.txt")
    trajs_test, labels_test = _yupify_mnist(
        sequence_path, test_labels_path, template=test_template, count=10000
    )

    return trajs_train + trajs_test, labels_train + labels_test
