import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from yupi import Trajectory

from trajectory_datasets.downloads import download_dataset

logger = logging.getLogger(__name__)

VERSION = 0
NAME = "hurdat2"
MIN_ROWS = 2

# Minimum wind speed (kt) of each Saffir-Simpson category
SAFFIR_SIMPSON_SCALE = ((137, 5), (113, 4), (96, 3), (83, 2), (64, 1))

_HUR_TRACKS = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2021-100522.txt"


def build() -> tuple[list[Trajectory], list[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_HUR_TRACKS, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def _get_saffir_simpson_scale(max_wind_speed: float) -> int:
    for threshold, category in SAFFIR_SIMPSON_SCALE:
        if max_wind_speed >= threshold:
            return category
    return 0


def _get_datetime(date_str: str, time_str: str) -> datetime:
    _date = datetime.strptime(date_str.strip(), "%Y%m%d")
    _time = datetime.strptime(time_str.strip(), "%H%M")
    _datetime = datetime.combine(_date, _time.time())
    return _datetime


def _process_huracane(hur_rows: list[list[str]]) -> tuple[Trajectory, int]:
    lat, long, time, max_wind_speed = [], [], [], -1.0
    start_time = None
    for row in hur_rows:
        if start_time is None:
            start_time = _get_datetime(row[0], row[1])
            time.append(0.0)
        else:
            _time = _get_datetime(row[0], row[1])
            time.append((_time - start_time).total_seconds())

        if row[4][-1] not in "NS" or row[5][-1] not in "WE":
            raise ValueError(f"Unknown position: {row[4]} {row[5]}")
        lat.append(float(row[4][:-1]))
        long.append(float(row[5][:-1]))

        if row[5][-1] == "W":
            long[-1] *= -1

        max_wind_speed = max(max_wind_speed, float(row[6]))

    label = _get_saffir_simpson_scale(max_wind_speed)
    traj = Trajectory(x=long, y=lat, t=time)
    return traj, label


def _yupify(raw_dir: Path) -> tuple[list[Trajectory], list[int]]:
    # Loads the raw data and preprocess it
    logger.info("Preprocessing Huracane raw data")
    mat_file = raw_dir / "hurdat2-1851-2021-100522.txt"
    trajs, labels = [], []
    with mat_file.open(encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        hur_rows: list[list[str]] = []
        for row in reader:
            if row[0].startswith("AL"):
                if len(hur_rows) > MIN_ROWS:
                    traj, label = _process_huracane(hur_rows)
                    trajs.append(traj)
                    labels.append(label)
                hur_rows = []
                continue
            hur_rows.append(row)
    return trajs, labels
