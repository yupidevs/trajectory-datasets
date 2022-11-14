import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from scipy.io.matlab import loadmat
from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "hurdat2"

_HUR_TRACKS = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2021-100522.txt"


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_HUR_TRACKS, NAME, uncompress=False)
    return raw_trajs_filepath.parent


def _get_saffir_simpson_scale(max_wind_speed: float) -> int:
    # Wind speed (kt)	Saffir-Simpson Hurricane Scale
    if max_wind_speed >= 137:
        return 5
    if max_wind_speed >= 113:
        return 4
    if max_wind_speed >= 96:
        return 3
    if max_wind_speed >= 83:
        return 2
    if max_wind_speed >= 64:
        return 1
    return 0


def _get_datetime(date_str: str, time_str: str) -> datetime:
    _date = datetime.strptime(date_str.strip(), "%Y%m%d")
    _time = datetime.strptime(time_str.strip(), "%H%M")
    _datetime = datetime.combine(_date, _time.time())
    return _datetime


def _process_huracane(hur_rows: List[List[str]]) -> Tuple[Trajectory, int]:
    lat, long, time, max_wind_speed = [], [], [], -1
    start_time = None
    for row in hur_rows:
        if start_time is None:
            start_time = _get_datetime(row[0], row[1])
            time.append(0)
        else:
            _time = _get_datetime(row[0], row[1])
            time.append((_time - start_time).total_seconds())

        assert row[4][-1] == "N" or row[4][-1] == "S"
        assert row[5][-1] == "W" or row[5][-1] == "E"
        lat.append(float(row[4][:-1]))
        long.append(float(row[5][:-1]))

        max_wind_speed = max(max_wind_speed, float(row[6]))

    label = _get_saffir_simpson_scale(max_wind_speed)
    traj = Trajectory(x=long, y=lat, t=time)
    return traj, label


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[int]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing Huracane raw data")
    mat_file = raw_dir / "hurdat2-1851-2021-100522.txt"
    trajs, labels = [], []
    with open(mat_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        hur_rows = []
        for row in reader:
            if row[0].startswith("AL"):
                if len(hur_rows) > 2:
                    traj, label = _process_huracane(hur_rows)
                    trajs.append(traj)
                    labels.append(label)
                hur_rows = []
                continue
            hur_rows.append(row)
    return trajs, labels
