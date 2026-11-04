import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from yupi import Trajectory

from trajectory_datasets.downloads import download_dataset

logger = logging.getLogger(__name__)

VERSION = 0
NAME = "traffic"
MIN_ROWS = 2

_TRAFFIC_URL = "https://zen-traffic-data.net/english/file/archive/TRAJECTORY_PUB_SMP.zip"


def build() -> tuple[list[Trajectory], list[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_TRAFFIC_URL, NAME)
    return raw_trajs_filepath.parent


def _get_time(date_str: str) -> datetime:
    return datetime.strptime(date_str.strip(), r"%H%M%S%f")


def _process_car(car_rows: list[list[Any]]) -> tuple[Trajectory, str]:
    x = [float(row[5]) for row in car_rows]
    y = [float(row[6]) for row in car_rows]
    dt0 = _get_time(car_rows[0][1])
    t = [(_get_time(row[1]) - dt0).total_seconds() for row in car_rows]
    label = "normal" if int(car_rows[0][2]) == 1 else "large"
    traj = Trajectory(x=x, y=y, t=t)
    return traj, label


def _yupify(raw_dir: Path) -> tuple[list[Trajectory], list[str]]:
    # Loads the raw data and preprocess it
    logger.info("Preprocessing Traffic raw data")
    trajs, labels = [], []

    with (raw_dir / "TRAJECTORY_PUB_SMP.csv").open() as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        current_car = [next(csv_reader)]
        current_id = current_car[0][0]
        for row in csv_reader:
            if row[0] != current_id:
                if len(current_car) > MIN_ROWS:
                    traj, label = _process_car(current_car)
                    trajs.append(traj)
                    labels.append(label)
                current_car = []
                current_id = row[0]
            current_car.append(row)
    return trajs, labels
