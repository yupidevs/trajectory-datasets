import csv
import logging
from pathlib import Path
from typing import Any, List, Tuple

from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "uci_pen_digits"

_UCI_PEN_TEST = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes"
)

_UCI_PEN_TRAIN = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
)


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    download_dataset(_UCI_PEN_TEST, NAME, uncompress=False)
    raw_trajs_filepath = download_dataset(_UCI_PEN_TRAIN, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def load_tracks(file: Path) -> Tuple[List[Trajectory], List[int]]:
    trajs, labels = [], []
    with open(file, "r", encoding="utf-8") as _fd:
        reader = csv.reader(_fd, delimiter=",")
        for row in reader:
            labels.append(int(row[-1]))
            _x = [float(x) for x in row[:-1:2]]
            _y = [float(y) for y in row[1:-1:2]]
            trajs.append(Trajectory(x=_x, y=_y))
    return trajs, labels


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[int]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing UCI pen digits raw data")
    test_traks = raw_dir / "pendigits.tes"
    train_tracks = raw_dir / "pendigits.tra"
    train_trajs, train_labels = load_tracks(train_tracks)
    test_trajs, test_labels = load_tracks(test_traks)
    return train_trajs + test_trajs, train_labels + test_labels
