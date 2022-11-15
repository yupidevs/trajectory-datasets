import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import utm
from yupi import Trajectory

from utils.utils import download_dataset

VERSION = 0
NAME = "animals"

_ANIMALS_TRACKS = (
    "https://raw.githubusercontent.com/ardywibowo/RanchHand/"
    "master/Starkey_OR_Main_Telemetry_1993-1996_Data.txt"
)


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(
        _ANIMALS_TRACKS, NAME, uncompress=False, check_size=False
    )
    return raw_trajs_filepath.parent


def _get_datetime(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(
        date_str.strip() + " " + time_str.strip(), "%Y%m%d %H:%M:%S"
    )


def _process_animal(animal_rows) -> Tuple[Trajectory, str]:
    lat, long, time = [], [], []
    start_time = None
    label = animal_rows[0][" Species"]
    for row in animal_rows:
        if start_time is None:
            start_time = _get_datetime(row[" LocDate"], row[" LocTime"])
            time.append(0)
        else:
            _time = _get_datetime(row[" LocDate"], row[" LocTime"])
            time.append((_time - start_time).total_seconds())

        assert row[" Species"] == label

        utm_e = int(row[" UTME"])
        utm_n = int(row[" UTMN"])
        _lat, _long = utm.to_latlon(utm_e, utm_n, 11, northern=True)
        lat.append(_lat)
        long.append(_long)

    return Trajectory(x=long, y=lat, t=time), label


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[str]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing Animals raw data")
    rows_dict: Dict[str, Any] = {}
    with open(
        raw_dir / "Starkey_OR_Main_Telemetry_1993-1996_Data.txt", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            _id: str = row[" Id"]
            if _id not in rows_dict:
                rows_dict[_id] = []
            rows_dict[_id].append(row)

    trajs, labels = [], []
    for rows in rows_dict.values():
        traj, label = _process_animal(rows)
        trajs.append(traj)
        labels.append(label)
    return trajs, labels
