import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from scipy.io.matlab import loadmat
from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "cma_bst"

_HUR_TRACKS = "https://tcdata.typhoon.org.cn/data/CMABSTdata/CMABSTdata.rar"

LABELS = [
    "Weaker or unknown",
    "Tropical Depression",
    "Tropical Storm",
    "Severe Tropical Storm",
    "Typhoon",
    "Severe Typhoon",
    "Super Typhoon",
    "",
    "",
    "Extratropical Cyclone",
]


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_HUR_TRACKS, NAME)
    return raw_trajs_filepath.parent


def _get_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str.strip(), "%Y%m%d%H")


def _process_huracane(hur_rows: List[str]) -> Tuple[Trajectory, str]:
    lat, long, time, max_cat = [], [], [], 0
    start_time = None
    for row in hur_rows:
        if start_time is None:
            start_time = _get_datetime(row[0])
            time.append(0)
        else:
            _time = _get_datetime(row[0])
            time.append((_time - start_time).total_seconds())

        max_cat = max(max_cat, int(row[1]))
        lat.append(float(row[2]))
        long.append(float(row[3]))

    label = LABELS[max_cat]
    traj = Trajectory(x=long, y=lat, t=time)
    return traj, label


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[str]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing Typhoon raw data")
    trajs, labels = [], []

    for year_file in raw_dir.glob("*.txt"):
        with open(year_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=" ")
            hur_rows = []
            for row in reader:
                row = [item for item in row if item]
                if row[0].startswith("66666"):
                    if len(hur_rows) > 2:
                        traj, label = _process_huracane(hur_rows)
                        trajs.append(traj)
                        labels.append(label)
                    hur_rows = []
                    continue
                hur_rows.append(row)
    return trajs, labels
