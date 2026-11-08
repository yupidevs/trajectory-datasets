import logging
from pathlib import Path
from typing import Any

from scipy.io.matlab import loadmat
from yupi import Trajectory

from trajectory_datasets.downloads import download_dataset

logger = logging.getLogger(__name__)

VERSION = 0
NAME = "uci_characters"

_UCI_MAT_FILE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "character-trajectories/mixoutALL_shifted.mat"
)

LABELS = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "g",
    "h",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "u",
    "v",
    "w",
    "y",
    "z",
]


def build() -> tuple[list[Trajectory], list[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_UCI_MAT_FILE, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def _yupify(raw_dir: Path) -> tuple[list[Trajectory], list[str]]:
    # Loads the raw data and preprocess it
    logger.info("Preprocessing UCI characters raw data")
    mat_file = raw_dir / "mixoutALL_shifted.mat"
    mat = loadmat(str(mat_file))
    _dt = mat["consts"][0][0][5][0][0]
    labels = [LABELS[i - 1] for i in mat["consts"][0][0][4][0]]
    trajs = []
    for arr in mat["mixout"][0]:
        _x = arr[0]
        _y = arr[1]
        trajs.append(Trajectory(x=_x, y=_y, dt=_dt))
    return trajs, labels
