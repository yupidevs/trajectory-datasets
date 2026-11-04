import importlib
import json
import logging
import pkgutil
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from yupi.core import JSONSerializer

from trajectory_datasets import config, recipes
from trajectory_datasets.downloads import _get_path

logger = logging.getLogger(__name__)


def available_recipes() -> dict[str, ModuleType]:
    modules = (
        importlib.import_module(f"{recipes.__name__}.{info.name}")
        for info in pkgutil.iter_modules(recipes.__path__)
        if not info.name.startswith("_")
    )
    return {module.NAME: module for module in modules}


def build_datasets(output_dir: Path, names: Sequence[str]) -> list[str]:
    """Build the datasets, or all of them without names. Returns the ones that failed."""
    known = available_recipes()
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")

    failed = []
    for name in names or sorted(known):
        recipe = known[name]
        try:
            build_recipe(output_dir, name, recipe.VERSION, recipe.build)
        except Exception:
            logger.exception("Could not build dataset '%s'", name)
            failed.append(name)
    return failed


def build_recipe(output_dir: Path, name: str, version: int, build_func: Callable) -> None:
    data_path = _get_path(config.DS_DIR, name) / "yupi_data.json"
    if _read_cache_version(data_path) < version or not data_path.exists():
        _save_yupi_data(_yupify(version, build_func), data_path)
    else:
        logger.info("Dataset '%s' is up to date (v%s)", name, version)
    _compress(data_path, output_dir, name)


def _yupify(version: int, build_func: Callable) -> dict[str, Any]:
    trajs, labels = build_func()
    if len(trajs) != len(labels):
        raise ValueError(
            f"Number of trajectories and labels must be equal. "
            f"Got {len(trajs)} trajectories and {len(labels)} labels."
        )
    if not all(len(traj) > 1 for traj in trajs):
        raise ValueError("All trajectories must have at least 2 points.")
    for index, traj in enumerate(trajs):
        traj.traj_id = str(index)
    return {
        "version": version,
        "trajs": [JSONSerializer.to_json(traj) for traj in trajs],
        "labels": labels,
    }


def _read_cache_version(data_path: Path) -> int:
    if not data_path.exists():
        return -1
    return int(json.loads(data_path.read_text(encoding="utf-8")).get("version", -1))


def _save_yupi_data(yupi_data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(yupi_data, ensure_ascii=False, indent=4), encoding="utf-8")


def _compress(data_path: Path, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_dir / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.write(filename=data_path, arcname=f"{name}.json")
