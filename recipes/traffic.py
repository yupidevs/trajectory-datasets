import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "traffic"

_TRAFFIC_URL = (
    "https://zen-traffic-data.net/english/file/archive/TRAJECTORY_PUB_SMP.zip"
)


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_TRAFFIC_URL, NAME)
    return raw_trajs_filepath.parent


def _get_time(date_str: str) -> datetime:
    return datetime.strptime(date_str.strip(), r"%H%M%S%f")


def _process_car(car_rows: List[List[Any]]) -> Tuple[Trajectory, str]:
    x = [float(row[5]) for row in car_rows]
    y = [float(row[6]) for row in car_rows]
    dt0 = _get_time(car_rows[0][1])
    t = [(_get_time(row[1]) - dt0).total_seconds() for row in car_rows]
    label = "normal" if int(car_rows[0][2]) == 1 else "large"
    traj = Trajectory(x=x, y=y, t=t)
    return traj, label


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[str]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing Traffic raw data")
    trajs, labels = [], []

    with open(raw_dir / "TRAJECTORY_PUB_SMP.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        current_car = [next(csv_reader)]
        current_id = current_car[0][0]
        for row in csv_reader:
            if row[0] != current_id:
                if len(current_car) > 2:
                    traj, label = _process_car(current_car)
                    trajs.append(traj)
                    labels.append(label)
                current_car = []
                current_id = row[0]
            current_car.append(row)
    return trajs, labels
