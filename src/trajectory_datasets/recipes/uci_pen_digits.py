import csv
import logging
from pathlib import Path
from typing import Any

from yupi import Trajectory

from trajectory_datasets.downloads import download_dataset

logger = logging.getLogger(__name__)

VERSION = 0
NAME = "uci_pen_digits"

_UCI_PEN_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes"

_UCI_PEN_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"


def build() -> tuple[list[Trajectory], list[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    download_dataset(_UCI_PEN_TEST, NAME, uncompress=False)
    raw_trajs_filepath = download_dataset(_UCI_PEN_TRAIN, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def load_tracks(file: Path) -> tuple[list[Trajectory], list[int]]:
    trajs, labels = [], []
    with file.open(encoding="utf-8") as _fd:
        reader = csv.reader(_fd, delimiter=",")
        for row in reader:
            labels.append(int(row[-1]))
            _x = [float(x) for x in row[:-1:2]]
            _y = [float(y) for y in row[1:-1:2]]
            trajs.append(Trajectory(x=_x, y=_y))
    return trajs, labels


def _yupify(raw_dir: Path) -> tuple[list[Trajectory], list[int]]:
    # Loads the raw data and preprocess it
    logger.info("Preprocessing UCI pen digits raw data")
    test_traks = raw_dir / "pendigits.tes"
    train_tracks = raw_dir / "pendigits.tra"
    train_trajs, train_labels = load_tracks(train_tracks)
    test_trajs, test_labels = load_tracks(test_traks)
    return train_trajs + test_trajs, train_labels + test_labels
