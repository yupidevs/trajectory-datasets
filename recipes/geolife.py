import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, NamedTuple, Tuple

import numpy as np
from yupi import Trajectory

from utils.utils import download_dataset, _get_progress_log

NAME = "geolife"
VERSION = 0

_GEOLIFE_URL = (
    "https://download.microsoft.com/download/F/4/8/"
    "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
)


LabelData = NamedTuple(
    "LabelData", [("start_dt", datetime), ("end_dt", datetime), ("name", str)]
)


Register = Tuple[float, float, datetime]
"""Register data (GPS point and time) in the form: lat, lon, time"""


def build() -> Tuple[List[Trajectory], List[Any]]:
    raw_data_file = download_dataset(_GEOLIFE_URL, NAME)

    raw_dir = raw_data_file.parent
    return yupify(raw_dir)


def yupify(raw_dir: Path):
    # Loads the raw data and preprocess it
    raw_metadata = []
    logging.info("Preprocessing GeoLife raw data")
    usr_folders_path = raw_dir / "Geolife Trajectories 1.3/Data"
    usr_folders = list(sorted(usr_folders_path.iterdir()))
    for i, usr in enumerate(usr_folders):
        print(_get_progress_log(i + 1, len(usr_folders)), end="\r", flush=True)
        _process_usr_trajs(usr, raw_metadata)

    # Load the preprocessed data and create the yupi trajectories
    logging.info("Creating yupi trajectories")
    trajs, lables = [], []
    for i, traj_data in enumerate(raw_metadata):
        print(_get_progress_log(i + 1, len(raw_metadata)), end="\r", flush=True)
        x_data = traj_data["traj_data"][:, 1]
        y_data = traj_data["traj_data"][:, 0]
        t_data = traj_data["traj_data"][:, 2]
        trajs.append(Trajectory(x=x_data, y=y_data, t=t_data))
        lables.append(traj_data["label"])

    return trajs, lables


def _load_labels(labels_file: Path) -> List[LabelData]:
    """Loads the labels of a user."""
    with open(labels_file, "r", encoding="utf-8") as l_file:
        labels = [_parse_label(line) for line in l_file.readlines()[1:]]
    return labels


def _process_usr_trajs(usr_folder: Path, raw_metadata: List[dict]) -> None:
    """Processes the trajectories of a user."""
    labels_file = usr_folder / "labels.txt"
    if not labels_file.exists():
        return

    labels = _load_labels(labels_file)
    all_regs = _load_registers(usr_folder)

    label_idx = 0
    regs_idx = 0
    label = labels[label_idx]
    traj: List[List[float]] = []
    while regs_idx < len(all_regs):
        lat, long, reg_dt = all_regs[regs_idx]

        # Register inside label bounds
        if label.start_dt <= reg_dt <= label.end_dt:
            time = (reg_dt - label.start_dt).seconds
            if not traj or time != traj[-1][-1]:
                traj.append([lat, long, time])

        # Register outside label bounds
        elif label.start_dt < reg_dt:
            # Save the trajectory if it has at least 5 points
            if len(traj) > 5:
                raw_metadata.append(
                    {
                        "id": f"{usr_folder.name}_{label_idx}",
                        "traj_data": np.array(traj),
                        "label": label.name,
                    }
                )
                traj.clear()

            # Move to the next label
            label_idx += 1
            if label_idx >= len(labels):
                break
            label = labels[label_idx]
            continue  # Don't increment regs_idx yet
        regs_idx += 1


def _load_registers(usr_folder: Path) -> List[Register]:
    """Loads all the registers of a user."""
    all_regs = []
    trajs_folder = usr_folder / "Trajectory"
    sorted_paths = sorted(trajs_folder.iterdir())
    for plt in sorted_paths:
        with open(plt, "r", encoding="utf-8") as reg:
            registers = [_parse_register(line) for line in reg.readlines()[6:]]
            all_regs += registers
    return all_regs


def _parse_register(line: str) -> Register:
    """Parses a register line."""
    reg = line.strip().split(",")
    lat, lon = float(reg[0]), float(reg[1])
    _dt = datetime.strptime(f"{reg[5]} {reg[6]}", "%Y-%m-%d %H:%M:%S")
    return lat, lon, _dt


def _parse_label(line: str) -> LabelData:
    """Parses a label line."""
    items = line.split()
    start_dt = datetime.strptime(f"{items[0]} {items[1]}", "%Y/%m/%d %H:%M:%S")
    end_dt = datetime.strptime(f"{items[2]} {items[3]}", "%Y/%m/%d %H:%M:%S")
    clsf = items[4]
    return LabelData(start_dt, end_dt, clsf)
