import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from yupi import Trajectory

import config as cfg
from utils.utils import _get_path, download_dataset

VERSION = 0
NAME = "uci_gotrack"

_UCI_TRACKS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00354/GPS%20Trajectory.rar"
)


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_dir = _fetch_raw_data()
    return _yupify(raw_dir)


def _fetch_raw_data() -> Path:
    raw_trajs_filepath = download_dataset(_UCI_TRACKS_URL, NAME)
    return raw_trajs_filepath.parent


def _get_traj(rows: List[dict]) -> Trajectory:
    lat = [float(row["latitude"]) for row in rows]
    lon = [float(row["longitude"]) for row in rows]
    _t0 = datetime.strptime(rows[0]["time"], "%Y-%m-%d %H:%M:%S")
    time = [
        (datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S") - _t0).total_seconds()
        for row in rows
    ]
    return Trajectory(x=lon, y=lat, t=time)


def _yupify(raw_dir) -> Tuple[List[Trajectory], List[str]]:
    # Loads the raw data and preprocess it
    logging.info("Preprocessing UCI GPS trajectories raw data")
    metadata_path = raw_dir / "GPS Trajectory/go_track_tracks.csv"
    tracks_path = raw_dir / "GPS Trajectory/go_track_trackspoints.csv"

    labels_dict = {}

    with open(metadata_path, "r", encoding="utf-8") as meta_fd:
        reader = csv.DictReader(meta_fd)
        for row in reader:
            label = int(row["car_or_bus"])
            labels_dict[int(row["id"])] = "car" if label == 1 else "bus"

    trajs_rows: Dict[int, List[dict]] = {_id: [] for _id in labels_dict}

    with open(tracks_path, "r", encoding="utf-8") as tracks_fd:
        reader = csv.DictReader(tracks_fd)

        first_row = next(reader)
        _track_id = int(first_row["track_id"])
        _rows = [first_row]
        for row in reader:
            track_id = int(row["track_id"])
            if track_id not in labels_dict:
                continue
            if track_id != _track_id:
                if _track_id in labels_dict:
                    trajs_rows[_track_id].extend(_rows)
                else:
                    logging.warning("Track id %d not in metadata", _track_id)
                _rows = []
                _track_id = track_id
            _rows.append(row)

    # filter out tracks with less than 3 points
    trajs_rows = {k: v for k, v in trajs_rows.items() if len(v) > 2}

    labels = [labels_dict[_id] for _id in trajs_rows]
    trajs = [_get_traj(trajs_rows[_id]) for _id in trajs_rows]
    return trajs, labels
