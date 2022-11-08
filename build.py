import json
import logging
import zipfile
from pathlib import Path
from typing import Callable, List

from yupi import Trajectory
from yupi.core import JSONSerializer

import config
from utils.utils import _get_path

RECIPIES_DIR = Path("./recipes")


def _read_cache_version(name: str) -> int:
    ds_dir = _get_path(config.DS_DIR, name)
    yupi_data_json = ds_dir / "yupi_data.json"

    if not yupi_data_json.exists():
        return -1

    yupi_data = _load_yupi_data(yupi_data_json)
    return yupi_data.get("version", -1)


def _cache_up_to_date(name: str, version: int) -> bool:
    # Check if rebuild is required
    cache_version = _read_cache_version(name)
    rebuild = version > cache_version
    return not rebuild


def _load_yupi_data(yupi_data_json: Path):
    with open(yupi_data_json, "r", encoding="utf-8") as md_file:
        return json.load(md_file)


def _save_yupi_data(yupi_data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as md_file:
        json.dump(yupi_data, md_file, ensure_ascii=False, indent=4)


def _update_labels(trajs: List[Trajectory]):
    for i, traj in enumerate(trajs):
        traj.traj_id = str(i)
    return trajs


def _build_recipe(output_dir: Path, name: str, version: int, build_func: Callable):
    trajs, labels = build_func()
    trajs = _update_labels(trajs)

    ds_dir = _get_path(config.DS_DIR, name)
    ds_dir.mkdir(parents=True, exist_ok=True)
    json_trajs = [JSONSerializer.to_json(traj) for traj in trajs]
    yupi_data = {"version": version, "trajs": json_trajs, "labels": labels}

    logging.info("Saving yupify trajectories for %s dataset", name)
    data_path = ds_dir / "yupi_data.json"
    _save_yupi_data(yupi_data, data_path)

    # Compress to output dir
    output_zip = output_dir / f"{name}.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.write(filename=data_path, arcname=f"{name}.json")


def build_recipe(output_dir: Path, name: str, version: int, build_func: Callable):
    if _cache_up_to_date(name, version):
        logging.info("Dataset '%s' is up to date (v%s)", name, version)
        return

    _build_recipe(output_dir, name, version, build_func)


def process_recipe(output_dir: Path, recipe_py_path: Path):
    # import NAME, VERSION and build from .py
    module_name = recipe_py_path.name.replace(".py", "")

    recipe = __import__(
        f"recipes.{module_name}", globals(), locals(), ["NAME", "VERSION", "build"], 0
    )

    name = recipe.NAME
    version = recipe.VERSION
    build_func = recipe.build

    build_recipe(output_dir, name, version, build_func)


def main():
    output_dir = Path("./builds")
    for dataset_recipe in RECIPIES_DIR.glob("[!_]*.py"):
        try:
            process_recipe(output_dir, dataset_recipe)
        except AttributeError:
            logging.error("Recipe '%s' has missing fields", str(dataset_recipe))


if __name__ == "__main__":
    main()
