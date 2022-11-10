import csv
import logging
from pathlib import Path
from typing import Any, List, Tuple

from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "uci_movement_libras"

_UCI_LIBRAS_TRACKS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "libras/movement_libras.data"
)

LABELS = [
    "curved swing",
    "horizontal swing",
    "vertical swing",
    "anti-clockwise arc",
    "clockwise arc",
    "circle",
    "horizontal straight-line",
    "vertical straight-line",
    "horizontal zigzag",
    "vertical zigzag",
    "horizontal wavy",
    "vertical wavy",
    "face-up curve",
    "face-down curve ",
    "tremble",
]


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_UCI_LIBRAS_TRACKS, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[str]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing UCI movement libras raw data")
    data_file = raw_dir / "movement_libras.data"
    trajs, labels = [], []
    with open(data_file, "r", encoding="utf-8") as _fd:
        reader = csv.reader(_fd, delimiter=",")
        for row in reader:
            labels.append(LABELS[int(row[-1]) - 1])
            _x = [float(x) for x in row[:-1:2]]
            _y = [float(y) for y in row[1:-1:2]]
            trajs.append(Trajectory(x=_x, y=_y))
    return trajs, labels
